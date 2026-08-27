import argparse
import json
import sys
from pathlib import Path

import pytest
import torch

import trustsr.cli.ldsr_gpu as gpu
from trustsr.artifacts import (
    PredictionCache,
    build_identity,
    canonical_json,
    tensor_sha256,
    verify_artifact_manifest,
)
from trustsr.artifacts.gpu_run import GPUHardwareSnapshot
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
        project_root=Path(__file__).resolve().parents[2],
    )


@pytest.fixture(autouse=True)
def safe_gpu(monkeypatch):
    snapshot = GPUHardwareSnapshot(
        name="NVIDIA A100",
        uuid="GPU-test",
        driver_version="555.1",
        memory_total_mib=24576,
        memory_free_mib=20000,
        compute_capability="8.6",
        compute_pids=(),
    )
    monkeypatch.setattr(gpu.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu, "capture_gpu_hardware", lambda: snapshot, raising=False)
    monkeypatch.setattr(
        gpu, "collect_gpu_environment", lambda **_kwargs: {"schema_version": 1}
    )


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
    pair = _pairs()[0]
    expected_prediction = torch.zeros((4, 8, 12), dtype=torch.float32)
    expected_identity = build_identity(model.provenance(), pair.source, pair.sample_id, pair.lr)
    assert result["lr_sha256"] == tensor_sha256(pair.lr)
    assert result["repeatability"]["first_sha256"] == tensor_sha256(expected_prediction)
    assert result["repeatability"]["second_sha256"] == tensor_sha256(expected_prediction)
    assert result["cache_key"] == expected_identity.key
    assert (tmp_path / "artifacts/cache/predictions" / f"{result['cache_key']}.json").exists()
    assert (
        tmp_path / "artifacts/cache/predictions" / f"{result['cache_key']}.safetensors"
    ).exists()


def test_single_computes_metrics_from_the_verified_cached_tensor(tmp_path, monkeypatch):
    model = FakeModel()
    cached = torch.zeros((4, 8, 12), dtype=torch.float32)

    class RecordingCache:
        def __init__(self, _root):
            self.stored = None

        def put(self, _identity, prediction):
            self.stored = prediction

        def get(self, _identity):
            assert self.stored is not None
            assert torch.equal(self.stored, cached)
            return cached

    def metrics(_pair, prediction):
        assert prediction is cached
        return {key: 0.0 for key in gpu.METRIC_KEYS}

    monkeypatch.setattr(gpu.LDSRS2X4, "from_pretrained", lambda *args, **kwargs: model)
    monkeypatch.setattr(gpu, "load_opensr_pairs", lambda *args, **kwargs: _pairs())
    monkeypatch.setattr(gpu, "PredictionCache", RecordingCache)
    monkeypatch.setattr(gpu, "compute_opensr_metrics", metrics)
    monkeypatch.setattr(gpu.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(gpu.torch.cuda, "max_memory_allocated", lambda: 0)

    gpu.run_single(_args(tmp_path))

    assert model.calls == 2


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


@pytest.mark.parametrize(
    "handler",
    [gpu.run_preflight, gpu.run_single, gpu.run_three_model_benchmark],
)
@pytest.mark.parametrize(
    "hardware_error",
    [
        "requires exactly one GPU",
        "requires compute capability at least 8.0",
        "at least 18 GiB initial free VRAM",
        "a foreign CUDA compute process prevents this staged run",
    ],
)
def test_each_compute_stage_rejects_bad_hardware_before_constructor_or_download(
    tmp_path, monkeypatch, handler, hardware_error
):
    constructed = False

    def fail_capture():
        raise RuntimeError(hardware_error)

    def factory(*_args, **_kwargs):
        nonlocal constructed
        constructed = True
        return FakeModel()

    monkeypatch.setattr(gpu, "capture_gpu_hardware", fail_capture)
    monkeypatch.setattr(gpu.LDSRS2X4, "from_pretrained", factory)
    monkeypatch.setattr(gpu, "load_opensr_pairs", lambda *_args, **_kwargs: pytest.fail("no data"))

    with pytest.raises(RuntimeError, match=hardware_error):
        handler(_args(tmp_path))
    assert not constructed


def test_preflight_environment_preserves_the_single_preconstruction_snapshot(
    tmp_path, monkeypatch
):
    snapshot = GPUHardwareSnapshot(
        name="NVIDIA A100",
        uuid="GPU-preconstruction",
        driver_version="555.1",
        memory_total_mib=24576,
        memory_free_mib=19000,
        compute_capability="8.6",
        compute_pids=(),
    )
    captures = 0
    recorded = []

    def capture():
        nonlocal captures
        captures += 1
        return snapshot

    def collect(**kwargs):
        recorded.append(kwargs)
        return {"schema_version": 1, "gpu": snapshot.gpu_record()}

    monkeypatch.setattr(gpu, "capture_gpu_hardware", capture)
    monkeypatch.setattr(gpu, "collect_gpu_environment", collect)
    monkeypatch.setattr(gpu.LDSRS2X4, "from_pretrained", lambda *_args, **_kwargs: FakeModel())

    result = gpu.run_preflight(_args(tmp_path))

    assert captures == 1
    assert recorded == [
        {
            "hardware_snapshot": snapshot,
            "project_root": Path(__file__).resolve().parents[2],
        }
    ]
    assert result["gpu"]["uuid"] == "GPU-preconstruction"


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
        lambda **_kwargs: {
            "dataset": "spot",
            "dataset_version": "v3",
            "git_commit": "test",
            "python": "test",
            "torch": "test",
            "opensr_test": "test",
            "device": "cpu",
        },
    )

    result = gpu.run_three_model_benchmark(_args(tmp_path))

    assert [model.name for model in called["models"]] == [
        "bicubic-x4",
        "sen2srlite-x4",
        "ldsr-s2-x4",
    ]
    assert called["expected_model_count"] == 3
    assert called["environment"]["device"] == "cuda:0"
    assert called["result_path"] == tmp_path / "artifacts/phase1b/spot-v3-three-models.json"
    index_path = tmp_path / "artifacts/phase1b/ldsr-cache-index.json"
    index = json.loads(index_path.read_text())
    assert index["schema_version"] == 1
    assert index["model_provenance"] == ldsr.provenance()
    assert [(entry["source"], entry["sample_id"]) for entry in index["identities"]] == [
        ("opensr-test/spot/v3", f"spot-{item:04d}") for item in range(9)
    ]
    assert len({entry["cache_key"] for entry in index["identities"]}) == 9
    for pair, entry in zip(_pairs(), index["identities"], strict=True):
        identity = build_identity(ldsr.provenance(), pair.source, pair.sample_id, pair.lr)
        assert entry == {
            "source": pair.source,
            "sample_id": pair.sample_id,
            "lr": identity.as_dict()["lr"],
            "cache_key": identity.key,
        }
    assert index_path.read_bytes() == canonical_json(index) + b"\n"
    assert result == {"models": {}}


def test_parser_has_only_staged_commands_and_no_scientific_sampling_flags():
    parser = gpu.build_parser()
    actions = [
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ]
    assert set(actions[0].choices) == {"preflight", "single", "benchmark", "manifest"}
    with pytest.raises(SystemExit):
        parser.parse_args(["single", "--seed", "1"])


def test_main_applies_cli_path_overrides(tmp_path, monkeypatch):
    received = {}

    def factory(model_dir, *, device):
        received["model_dir"] = model_dir
        received["device"] = device
        return FakeModel()

    data = tmp_path / "external-data"
    models = tmp_path / "external-models"
    artifacts = tmp_path / "external-artifacts"
    cache = tmp_path / "external-cache"
    monkeypatch.setattr(gpu.LDSRS2X4, "from_pretrained", factory)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trustsr-ldsr-gpu",
            "preflight",
            "--dataset-cache-dir",
            str(data),
            "--ldsr-model-dir",
            str(models),
            "--artifacts-dir",
            str(artifacts),
            "--prediction-cache-dir",
            str(cache),
        ],
    )

    gpu.main()

    assert received == {"model_dir": models, "device": "cuda:0"}
    assert (artifacts / "phase1b/environment.json").is_file()


def test_main_single_routes_dataset_model_artifact_and_independent_cache_overrides(
    tmp_path, monkeypatch
):
    received = {}
    data = tmp_path / "external-data"
    models = tmp_path / "external-models"
    artifacts = tmp_path / "external-artifacts"
    cache = tmp_path / "independent-cache"
    model = FakeModel()

    def factory(model_dir, *, device):
        received["model_dir"] = model_dir
        received["device"] = device
        return model

    def load_pairs(dataset_name, cache_dir, version, *, limit, expected_count):
        received["dataset"] = (dataset_name, cache_dir, version, limit, expected_count)
        return _pairs()

    monkeypatch.setattr(gpu.LDSRS2X4, "from_pretrained", factory)
    monkeypatch.setattr(gpu, "load_opensr_pairs", load_pairs)
    monkeypatch.setattr(
        gpu, "compute_opensr_metrics", lambda *args: {key: 0.0 for key in gpu.METRIC_KEYS}
    )
    monkeypatch.setattr(gpu.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(gpu.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trustsr-ldsr-gpu",
            "single",
            "--dataset-cache-dir",
            str(data),
            "--ldsr-model-dir",
            str(models),
            "--artifacts-dir",
            str(artifacts),
            "--prediction-cache-dir",
            str(cache),
        ],
    )

    gpu.main()

    assert received["model_dir"] == models
    assert received["device"] == "cuda:0"
    assert received["dataset"] == ("spot", data, "v3", 9, 9)
    result = json.loads((artifacts / "phase1b/single.json").read_text())
    assert (cache / f"{result['cache_key']}.json").is_file()
    assert (cache / f"{result['cache_key']}.safetensors").is_file()


def _write_required_manifest_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[str]]:
    pairs = _pairs()
    provenance = FakeModel().provenance()
    identities = [
        build_identity(provenance, pair.source, pair.sample_id, pair.lr) for pair in pairs
    ]
    index: dict[str, object] = {
        "schema_version": 1,
        "model_provenance": provenance,
        "identities": [
            {
                "source": identity.source,
                "sample_id": identity.sample_id,
                "lr": identity.as_dict()["lr"],
                "cache_key": identity.key,
            }
            for identity in identities
        ],
    }
    phase = args.artifacts_dir / "phase1b"
    phase.mkdir(parents=True)
    (phase / "environment.json").write_text("{}")
    (phase / "single.json").write_text(
        json.dumps(
            {
                "source": pairs[0].source,
                "sample_id": pairs[0].sample_id,
                "lr_sha256": identities[0].lr_sha256,
                "model_provenance": provenance,
                "cache_key": identities[0].key,
            }
        )
    )
    (phase / "single-runtime.json").write_text("{}")
    (phase / "spot-v3-three-models.json").write_text(
        json.dumps(
            {
                "run": {
                    "sample_count": 9,
                    "samples": [
                        {
                            "source": identity.source,
                            "sample_id": identity.sample_id,
                            "lr": identity.as_dict()["lr"],
                        }
                        for identity in identities
                    ],
                },
                "models": {
                    "ldsr-s2-x4": {
                        "provenance": provenance,
                        "samples": [
                            {"source": pair.source, "sample_id": pair.sample_id}
                            for pair in pairs
                        ],
                    }
                },
            }
        )
    )
    (phase / "ldsr-cache-index.json").write_bytes(canonical_json(index) + b"\n")
    return index, [identity.key for identity in identities]


def _write_prediction_cache(args: argparse.Namespace) -> None:
    cache = PredictionCache(args.prediction_cache_dir)
    provenance = FakeModel().provenance()
    for pair in _pairs():
        identity = build_identity(provenance, pair.source, pair.sample_id, pair.lr)
        cache.put(identity, torch.zeros_like(pair.hr))


def test_main_manifest_stages_only_named_cache_entries_from_independent_cache_root(
    tmp_path, monkeypatch
):
    args = _args(tmp_path)
    args.prediction_cache_dir = tmp_path / "independent-cache"
    old_key = "b" * 64
    _, keys = _write_required_manifest_inputs(args)
    _write_prediction_cache(args)
    (args.prediction_cache_dir / f"{old_key}.json").write_text("old-metadata")
    (args.prediction_cache_dir / f"{old_key}.safetensors").write_bytes(b"old")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trustsr-ldsr-gpu",
            "manifest",
            "--project-root",
            str(args.project_root),
            "--dataset-cache-dir",
            str(args.dataset_cache_dir),
            "--ldsr-model-dir",
            str(args.ldsr_model_dir),
            "--sen2srlite-model-dir",
            str(args.sen2srlite_model_dir),
            "--artifacts-dir",
            str(args.artifacts_dir),
            "--prediction-cache-dir",
            str(args.prediction_cache_dir),
        ],
    )

    gpu.main()

    manifest_path = args.artifacts_dir / "phase1b/artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    paths = [entry["path"] for entry in manifest["files"]]
    assert "phase1b/ldsr-cache-index.json" in paths
    assert {f"phase1b/cache/{key}.json" for key in keys}.issubset(paths)
    assert {f"phase1b/cache/{key}.safetensors" for key in keys}.issubset(paths)
    assert len([path for path in paths if path.startswith("phase1b/cache/")]) == 18
    assert old_key not in "".join(paths)
    assert all(not Path(path).is_absolute() for path in paths)
    verify_artifact_manifest(
        args.artifacts_dir, manifest_path
    )


@pytest.mark.parametrize(
    "missing_name",
    [
        "environment.json",
        "single.json",
        "single-runtime.json",
        "spot-v3-three-models.json",
        "ldsr-cache-index.json",
    ],
)
def test_manifest_rejects_each_missing_required_output(tmp_path, missing_name):
    args = _args(tmp_path)
    _write_required_manifest_inputs(args)
    _write_prediction_cache(args)
    (args.artifacts_dir / "phase1b" / missing_name).unlink()

    with pytest.raises(FileNotFoundError, match="allowlisted"):
        gpu.run_manifest(args)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "extra", "invalid_key", "invalid_lr", "wrong_provenance"],
)
def test_manifest_rejects_unbound_or_nonexact_ldsr_cache_identity_data_before_staging(
    tmp_path, mutation
):
    args = _args(tmp_path)
    index, _ = _write_required_manifest_inputs(args)
    _write_prediction_cache(args)
    identities = index["identities"]
    assert isinstance(identities, list)
    if mutation == "missing":
        identities.pop()
    elif mutation == "duplicate":
        identities[-1] = dict(identities[0])
    elif mutation == "extra":
        identities.append(dict(identities[-1]))
    elif mutation == "invalid_key":
        identities[0]["cache_key"] = "0" * 64
    elif mutation == "invalid_lr":
        identities[0]["lr"] = {**identities[0]["lr"], "sha256": "0" * 64}
    else:
        index["model_provenance"] = {"name": "ldsr-s2-x4", "scale": 4, "changed": True}
    index_path = args.artifacts_dir / "phase1b/ldsr-cache-index.json"
    index_path.write_bytes(canonical_json(index) + b"\n")

    with pytest.raises(ValueError, match="cache index"):
        gpu.run_manifest(args)

    assert not (args.artifacts_dir / "phase1b/cache").exists()
