import json
from pathlib import Path

import pytest
import torch

from trustsr.contracts import SRPair


def _metrics() -> dict[str, float]:
    return {
        "reflectance": 0.1,
        "spectral": 0.2,
        "spatial": 0.3,
        "synthesis": 0.4,
        "ha_metric": 0.5,
        "om_metric": 0.6,
        "im_metric": 0.7,
    }


def test_run_writes_reproducible_json(monkeypatch, tmp_path: Path) -> None:
    from trustsr.cli.smoke_baseline import run

    pair = SRPair("spot-0000", "fixture", torch.rand(4, 4, 4), torch.rand(4, 16, 16), 4)
    monkeypatch.setattr(
        "trustsr.cli.smoke_baseline.load_opensr_pairs",
        lambda dataset_name, cache_dir, version, limit: [pair],
    )
    monkeypatch.setattr(
        "trustsr.cli.smoke_baseline.compute_opensr_metrics",
        lambda pair, sr: _metrics(),
    )
    output = tmp_path / "result.json"

    result = run("spot", "v3", 1, tmp_path / "cache", output)

    assert json.loads(output.read_text()) == result
    assert result["run"]["dataset_role"] == "development_smoke_only"
    assert result["run"]["bands"] == ["B04", "B03", "B02", "B08"]
    assert len(result["run"]["sample_manifest_sha256"]) == 64
    assert result["samples"][0]["sample_id"] == "spot-0000"
    assert result["mean_metrics"]["ha_metric"] == 0.5


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_run_rejects_non_finite_metrics(monkeypatch, tmp_path: Path, value: float) -> None:
    from trustsr.cli.smoke_baseline import run

    pair = SRPair("spot-0000", "fixture", torch.rand(4, 4, 4), torch.rand(4, 16, 16), 4)
    monkeypatch.setattr(
        "trustsr.cli.smoke_baseline.load_opensr_pairs",
        lambda dataset_name, cache_dir, version, limit: [pair],
    )
    monkeypatch.setattr(
        "trustsr.cli.smoke_baseline.compute_opensr_metrics",
        lambda pair, sr: {**_metrics(), "ha_metric": value},
    )

    with pytest.raises(ValueError, match="non-finite metric"):
        run("spot", "v3", 1, tmp_path / "cache", tmp_path / "result.json")
