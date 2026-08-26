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
        "name": "bicubic_x4",
        "scale": 4,
        "implementation": "torch.nn.functional.interpolate",
        "mode": "bicubic",
        "align_corners": False,
        "antialias": True,
        "output_policy": "clip_to_[0,1]",
    }
    assert json.loads(json.dumps(provenance)) == provenance
    assert all(
        value is None or isinstance(value, (str, int, float, bool))
        for value in provenance.values()
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.01, 1.01])
def test_bicubic_rejects_nonfinite_and_out_of_range_input(value: float) -> None:
    lr = torch.full((4, 2, 2), value, dtype=torch.float32)
    with pytest.raises(ValueError):
        BicubicX4().predict(lr)
