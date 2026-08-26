import json

import pytest
import torch

from trustsr.models.bicubic import BicubicX4
from trustsr.models.protocols import SRModel


def test_bicubic_implements_sr_model_protocol() -> None:
    assert isinstance(BicubicX4(), SRModel)


def test_bicubic_provenance_is_stable_json_scalars() -> None:
    provenance = BicubicX4().provenance()
    assert provenance == {
        "name": "bicubic-x4",
        "scale": 4,
        "implementation": "torch.nn.functional.interpolate",
        "mode": "bicubic",
        "align_corners": False,
        "antialias": True,
        "output_policy": "clip_to_[0,1]",
        "torch_version": torch.__version__,
        "implementation_schema_version": 1,
    }
    assert json.loads(json.dumps(provenance)) == provenance
    assert all(
        value is None or isinstance(value, (str, int, float, bool))
        for value in provenance.values()
    )


def test_bicubic_provenance_runtime_and_adapter_versions_affect_cache_identity() -> None:
    from trustsr.artifacts.predictions import build_identity

    lr = torch.zeros((4, 2, 3), dtype=torch.float32)
    provenance = BicubicX4().provenance()
    changed_torch = dict(provenance, torch_version="different")
    changed_adapter = dict(provenance, implementation_schema_version=2)

    assert build_identity(provenance, "source", "id", lr).key != build_identity(
        changed_torch, "source", "id", lr
    ).key
    assert build_identity(provenance, "source", "id", lr).key != build_identity(
        changed_adapter, "source", "id", lr
    ).key


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.01, 1.01])
def test_bicubic_rejects_nonfinite_and_out_of_range_input(value: float) -> None:
    lr = torch.full((4, 2, 2), value, dtype=torch.float32)
    with pytest.raises(ValueError):
        BicubicX4().predict(lr)
