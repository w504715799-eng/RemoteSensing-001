import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import trustsr.cli.benchmark_baselines as benchmark
from trustsr.contracts import SRPair

METRICS = {
    "reflectance": 0.1,
    "spectral": 0.2,
    "spatial": 0.3,
    "synthesis": 0.4,
    "ha_metric": 0.5,
    "om_metric": 0.6,
    "im_metric": 0.7,
}


class FakeModel:
    scale = 4

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[SRPair] = []

    def provenance(self) -> dict[str, str | int | bool]:
        return {"name": self.name, "scale": self.scale, "fake": True}

    def predict(self, lr: torch.Tensor) -> torch.Tensor:
        self.calls.append(next(pair for pair in PAIRS if pair.lr is lr))
        return torch.zeros((4, lr.shape[1] * 4, lr.shape[2] * 4), dtype=torch.float32)


class BadModel(FakeModel):
    def predict(self, lr: torch.Tensor) -> torch.Tensor:
        self.calls.append(next(pair for pair in PAIRS if pair.lr is lr))
        return torch.zeros((4, lr.shape[1] * 4 - 1, lr.shape[2] * 4), dtype=torch.float32)


def _pairs(count: int = 9) -> list[SRPair]:
    return [
        SRPair(
            sample_id=f"spot-{index:04d}",
            source="opensr-test/spot/v3",
            lr=torch.full((4, 2, 3), index / 10, dtype=torch.float32),
            hr=torch.full((4, 8, 12), index / 10, dtype=torch.float32),
            scale=4,
        )
        for index in range(count)
    ]


PAIRS: list[SRPair] = []
REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def fake_metrics(monkeypatch):
    monkeypatch.setattr(
        "trustsr.cli.benchmark_baselines.compute_opensr_metrics",
        lambda pair, sr: METRICS,
    )


def _run(
    tmp_path: Path,
    pairs: list[SRPair],
    models: list[FakeModel],
    *,
    expected_model_count: int = 2,
):
    from trustsr.cli.benchmark_baselines import run_benchmark

    return run_benchmark(
        pairs=pairs,
        models=models,
        cache_root=tmp_path / "cache",
        result_path=tmp_path / "result.json",
        environment={
            "dataset": "spot",
            "dataset_version": "v3",
            "git_commit": "test-commit",
            "python": "test-python",
            "torch": "test-torch",
            "opensr_test": "test-opensr",
            "device": "cpu",
        },
        expected_model_count=expected_model_count,
    )


def test_three_models_preserve_input_order(tmp_path: Path):
    global PAIRS
    PAIRS = _pairs()
    result = _run(
        tmp_path,
        PAIRS,
        [FakeModel("a"), FakeModel("b"), FakeModel("c")],
        expected_model_count=3,
    )
    assert list(result["models"]) == ["a", "b", "c"]


def test_production_environment_reads_git_from_the_reviewed_root_not_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "reviewed-commit\n", "")

    monkeypatch.setattr(benchmark.subprocess, "run", run)

    environment = benchmark._production_environment(project_root=REPOSITORY)

    assert environment["git_commit"] == "reviewed-commit"
    assert calls == [
        (
            ["git", "-C", str(REPOSITORY.resolve()), "rev-parse", "HEAD"],
            {"check": False, "capture_output": True, "text": True},
        )
    ]


@pytest.mark.parametrize("count", [1, 3])
def test_default_expected_model_count_rejects_wrong_count(tmp_path: Path, count: int):
    global PAIRS
    PAIRS = _pairs()
    with pytest.raises(ValueError, match="exactly two"):
        _run(tmp_path, PAIRS, [FakeModel(str(i)) for i in range(count)])


@pytest.mark.parametrize("count", [0, -1, True, 2.0])
def test_expected_model_count_requires_exact_positive_int(tmp_path: Path, count):
    global PAIRS
    PAIRS = _pairs()
    with pytest.raises(ValueError, match="positive int"):
        _run(tmp_path, PAIRS, [FakeModel("a"), FakeModel("b")], expected_model_count=count)


def test_requires_exactly_nine_unique_spot_samples(tmp_path: Path):
    global PAIRS
    models = [FakeModel("first"), FakeModel("second")]
    for count in (8, 10):
        PAIRS = _pairs(count)
        with pytest.raises(ValueError, match="exactly nine"):
            _run(tmp_path, PAIRS, models)

    PAIRS = _pairs()
    duplicate = PAIRS.copy()
    duplicate[-1] = SRPair("spot-0000", PAIRS[-1].source, PAIRS[-1].lr, PAIRS[-1].hr, 4)
    with pytest.raises(ValueError, match="unique"):
        _run(tmp_path, duplicate, models)


def test_two_models_share_manifest_and_cache_replay_is_prediction_free(tmp_path: Path):
    global PAIRS
    PAIRS = _pairs()
    first, second = FakeModel("first"), FakeModel("second")
    result = _run(tmp_path, PAIRS, [first, second])

    assert first.calls == PAIRS
    assert second.calls == PAIRS
    assert len(first.calls) == len(second.calls) == 9
    assert result["models"]["first"]["sample_manifest_sha256"] == result["models"]["second"][
        "sample_manifest_sha256"
    ] == result["run"]["sample_manifest_sha256"]

    replay_first, replay_second = FakeModel("first"), FakeModel("second")
    replay = _run(tmp_path, PAIRS, [replay_first, replay_second])
    assert replay_first.calls == replay_second.calls == []
    assert replay == result


def test_result_is_deterministic_complete_and_has_only_finite_metrics(tmp_path: Path):
    global PAIRS
    PAIRS = _pairs()
    result = _run(tmp_path, PAIRS, [FakeModel("first"), FakeModel("second")])
    first_bytes = (tmp_path / "result.json").read_bytes()
    result_again = _run(tmp_path, PAIRS, [FakeModel("first"), FakeModel("second")])
    second_bytes = (tmp_path / "result.json").read_bytes()

    assert first_bytes == second_bytes
    assert result_again == result
    assert set(result) == {"run", "models"}
    assert result["run"]["dataset"] == "spot"
    assert result["run"]["dataset_version"] == "v3"
    assert result["run"]["sample_count"] == 9
    assert len(result["run"]["samples"]) == 9
    assert result["models"]["first"]["provenance"] == FakeModel("first").provenance()
    assert {item["sample_id"] for item in result["run"]["samples"]} == {
        f"spot-{index:04d}" for index in range(9)
    }
    for model_result in result["models"].values():
        assert len(model_result["samples"]) == 9
        for metric_group in [model_result["mean_metrics"]] + [
            item["metrics"] for item in model_result["samples"]
        ]:
            assert all(math.isfinite(value) for value in metric_group.values())

    text = first_bytes.decode("utf-8")
    assert str(tmp_path) not in text
    assert "timestamp" not in text.lower()
    assert "duration" not in text.lower()
    assert "cache_hit" not in text.lower()
    assert json.loads(text) == result


def test_invalid_model_output_fails_before_metric_computation(tmp_path: Path, monkeypatch):
    global PAIRS
    PAIRS = _pairs()
    called = False

    def metrics_should_not_run(pair, sr):
        nonlocal called
        called = True
        return METRICS

    monkeypatch.setattr(
        "trustsr.cli.benchmark_baselines.compute_opensr_metrics", metrics_should_not_run
    )
    with pytest.raises(ValueError, match="prediction"):
        _run(tmp_path, PAIRS, [BadModel("bad"), FakeModel("second")])
    assert not called


@pytest.mark.parametrize("upstream_count", [8, 9, 10])
def test_production_main_requires_exact_raw_upstream_spot_count(
    tmp_path: Path, monkeypatch, upstream_count: int
):
    raw = {
        "L2A": np.zeros((upstream_count, 8, 2, 3), dtype=np.uint16),
        "HRharm": np.zeros((upstream_count, 4, 8, 12), dtype=np.uint16),
    }
    monkeypatch.setattr("trustsr.data.opensr.opensr_test.load", lambda *_args, **_kwargs: raw)
    monkeypatch.setattr(benchmark, "run_benchmark", lambda **_kwargs: {"models": {}})
    monkeypatch.setattr(
        benchmark.SEN2SRLiteX4,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeModel("sen2srlite-x4"),
    )
    monkeypatch.setattr(sys, "argv", ["trustsr-benchmark", "--dataset-cache-dir", str(tmp_path)])

    if upstream_count == 9:
        benchmark.main()
    else:
        with pytest.raises(ValueError, match="exactly 9 raw SPOT v3 samples"):
            benchmark.main()
