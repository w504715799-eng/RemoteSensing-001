import argparse
import json
from pathlib import Path

import pytest
import torch

import trustsr.cli.ldsr_gpu as gpu
from trustsr.contracts import SRPair
from trustsr.evaluation.repeatability import RepeatabilityError


def _pairs() -> list[SRPair]:
    return [
        SRPair(
            source="opensr-test/spot/v3",
            sample_id=f"spot-{index:04d}",
            lr=torch.full((4, 2, 3), index / 10, dtype=torch.float32),
            hr=torch.full((4, 8, 12), index / 10, dtype=torch.float32),
            scale=4,
        )
        for index in range(9)
    ]


class FakeModel:
    scale = 4

    def __init__(self, name="ldsr-s2-x4"):
        self.name = name
        self.calls = 0

    def provenance(self):
        return {"name": self.name, "scale": 4}

    def predict(self, lr):
        self.calls += 1
        return torch.zeros((4, lr.shape[1] * 4, lr.shape[2] * 4), dtype=torch.float32)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_cache_dir=tmp_path / "data",
        ldsr_model_dir=tmp_path / "models" / "ldsr",
        sen2srlite_model_dir=tmp_path / "models" / "sen2sr",
        artifacts_dir=tmp_path / "artifacts",
        prediction_cache_dir=tmp_path / "artifacts" / "cache" / "predictions",
    )


@pytest.fixture(autouse=True)
def safe_gpu(monkeypatch):
    monkeypatch.setattr(gpu.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu, "_active_compute_pids", lambda: set())
    monkeypatch.setattr(gpu, "collect_gpu_environment", lambda: {"schema_version": 1})


def test_preflight_constructs_one_model_without_loading_dataset(tmp_path, monkeypatch):
    model = FakeModel()
    calls = 0

    def factory(*args, **kwargs):
        nonlocal calls
        calls += 1
        return model

    monkeypatch.setattr(gpu.LDSRS2X4, "from_pretrained", factory)
    monkeypatch.setattr(gpu, "load_opensr_pairs", lambda *args, **kwargs: pytest.fail("no data"))

    gpu.run_preflight(_args(tmp_path))

    assert calls == 1
    environment = json.loads((tmp_path / "artifacts/phase1b/environment.json").read_text())
    assert environment["model_provenance"] == model.provenance()


def test_single_runs_two_predictions_writes_separate_runtime_and_verified_cache(
    tmp_path, monkeypatch
):
    model = FakeModel()
    monkeypatch.setattr(gpu.LDSRS2X4, "from_pretrained", lambda *args, **kwargs: model)
    monkeypatch.setattr(gpu, "load_opensr_pairs", lambda *args, **kwargs: _pairs())
    monkeypatch.setattr(
        gpu, "compute_opensr_metrics", lambda *args: {key: 0.0 for key in gpu.METRIC_KEYS}
    )
    monkeypatch.setattr(gpu.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(gpu.torch.cuda, "max_memory_allocated", lambda: 12)

    result = gpu.run_single(_args(tmp_path))

    assert model.calls == 2
    assert result["sample_id"] == "spot-0000"
    assert set(result) == {
        "schema_version",
        "source",
        "sample_id",
        "lr_sha256",
        "model_provenance",
        "repeatability",
        "cache_key",
        "metrics",
    }
    deterministic = tmp_path / "artifacts/phase1b/single.json"
    runtime = tmp_path / "artifacts/phase1b/single-runtime.json"
    assert deterministic.exists() and runtime.exists()
    assert "duration" not in deterministic.read_text()
    assert json.loads(runtime.read_text())["peak_memory_bytes"] == 12
    assert (tmp_path / "artifacts/cache/predictions" / f"{result['cache_key']}.json").exists()
    assert (
        tmp_path / "artifacts/cache/predictions" / f"{result['cache_key']}.safetensors"
    ).exists()


def test_single_refuses_to_write_when_repeatability_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu.LDSRS2X4, "from_pretrained", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(gpu, "load_opensr_pairs", lambda *args, **kwargs: _pairs())
    monkeypatch.setattr(
        gpu,
        "run_repeatability",
        lambda *args, **kwargs: (_ for _ in ()).throw(RepeatabilityError("no")),
    )
    monkeypatch.setattr(gpu.torch.cuda, "reset_peak_memory_stats", lambda: None)

    with pytest.raises(RepeatabilityError):
        gpu.run_single(_args(tmp_path))
    assert not (tmp_path / "artifacts/phase1b/single.json").exists()


def test_benchmark_uses_exact_order_three_models_and_cuda_environment(tmp_path, monkeypatch):
    ldsr = FakeModel()
    monkeypatch.setattr(gpu.LDSRS2X4, "from_pretrained", lambda *args, **kwargs: ldsr)
    monkeypatch.setattr(
        gpu.SEN2SRLiteX4, "from_pretrained", lambda *args, **kwargs: FakeModel("sen2srlite-x4")
    )
    called = {}

    def fake_benchmark(**kwargs):
        called.update(kwargs)
        Path(kwargs["result_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["result_path"]).write_text("{}")
        return {"models": {}}

    monkeypatch.setattr(gpu, "load_opensr_pairs", lambda *args, **kwargs: _pairs())
    monkeypatch.setattr(gpu, "run_benchmark", fake_benchmark)
    monkeypatch.setattr(
        gpu,
        "_production_environment",
        lambda: {
            "dataset": "spot",
            "dataset_version": "v3",
            "git_commit": "test",
            "python": "test",
            "torch": "test",
            "opensr_test": "test",
            "device": "cpu",
        },
    )

    gpu.run_three_model_benchmark(_args(tmp_path))

    assert [model.name for model in called["models"]] == [
        "bicubic-x4",
        "sen2srlite-x4",
        "ldsr-s2-x4",
    ]
    assert called["expected_model_count"] == 3
    assert called["environment"]["device"] == "cuda:0"
    assert called["result_path"] == tmp_path / "artifacts/phase1b/spot-v3-three-models.json"


def test_parser_has_only_staged_commands_and_no_scientific_sampling_flags():
    parser = gpu.build_parser()
    actions = [
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ]
    assert set(actions[0].choices) == {"preflight", "single", "benchmark", "manifest"}
    with pytest.raises(SystemExit):
        parser.parse_args(["single", "--seed", "1"])
