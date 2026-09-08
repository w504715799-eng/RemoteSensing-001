"""CPU tripwires for a bounded, development-only hardware measurement."""

import importlib
import json

import pytest
import torch

from trustsr.artifacts.predictions import tensor_sha256
from trustsr.models.bicubic import BicubicX4


def api():
    return importlib.import_module("scripts.paper.benchmark_inference")


def membership():
    rows = [{"sample_id": f"r-{i:03d}", "split": "development"} for i in range(120)]
    published = {"samples": [{"sample_id": r["sample_id"]} for r in rows]}
    return rows, published


def test_selection_is_first_three_development_ids_not_manifest_order():
    rows, published = membership()
    rows = [{"sample_id": "a-test", "split": "internal_test"}, *reversed(rows)]
    assert [r["sample_id"] for r in api().select_measurement_records(rows, published)] == [
        "r-000", "r-001", "r-002",
    ]


@pytest.mark.parametrize("change", ["missing", "duplicate", "changed_split", "different_id"])
def test_inconsistent_membership_fails_before_any_pixel_read(change):
    rows, published = membership()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0].copy()
    elif change == "changed_split":
        rows[-1]["split"] = "calibration"
    else:
        rows[-1]["sample_id"] = "not-in-audit"
    with pytest.raises(ValueError):
        api().select_measurement_records(rows, published)


def test_real_cpu_prediction_reports_bound_input_output_and_elapsed_time():
    # Zero is a bit-exact interpolation oracle; nonzero constants can round.
    lr = torch.zeros(4, 128, 128)
    result = api().measure_prediction(BicubicX4(), lr)
    expected = torch.zeros(4, 512, 512)
    assert result["input_sha256"] == tensor_sha256(lr)
    assert result["output_sha256"] == tensor_sha256(expected)
    assert result["seconds"] > 0
    assert result["cuda_peak_allocated_bytes"] is None
    assert result["cuda_peak_reserved_bytes"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("shape,value", [((4, 128, 128), float("nan")),
                                         ((4, 512, 512), 1.1),
                                         ((4, 511, 512), 0.2)])
def test_invalid_prediction_is_not_published_as_valid_timing(shape, value):
    class InvalidModel:
        def predict(self, lr):
            return torch.full(shape, value)

    with pytest.raises(ValueError):
        api().measure_prediction(InvalidModel(), torch.zeros(4, 128, 128))


def test_missing_cuda_stops_before_creating_outputs_or_reading_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "new-output"
    with pytest.raises(RuntimeError, match="CUDA"):
        api().run(tmp_path / "missing-data", tmp_path / "missing-models",
                  tmp_path / "missing-sen2sr", output)
    assert list(tmp_path.iterdir()) == []


def test_output_cannot_alias_historical_or_model_trees(tmp_path):
    storage = tmp_path / "storage"
    model = tmp_path / "model"
    storage.mkdir()
    model.mkdir()
    for output in [storage / "trustsr" / "new", model / "new", storage]:
        with pytest.raises(ValueError):
            api().validate_output(output, storage, model, model)
    api().validate_output(tmp_path / "separate-new-run", storage, model, model)
