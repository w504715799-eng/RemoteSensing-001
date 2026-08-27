import re
from dataclasses import FrozenInstanceError

import pytest
import torch

from trustsr.calibration.conformal import (
    ConformalCalibration,
    calibrate_fidelity_mask,
    trusted_mask,
)


def test_calibration_uses_roi_maxima_and_finite_sample_correction() -> None:
    scores = (
        torch.tensor([[0.1, 0.2]], dtype=torch.float64),
        torch.tensor([[0.1, 0.3]], dtype=torch.float64),
    )
    risks = (
        torch.tensor([[0.1, 0.4]], dtype=torch.float64),
        torch.tensor([[0.2, 0.9]], dtype=torch.float64),
    )

    result = calibrate_fidelity_mask(scores, risks, alpha=0.55)

    assert result.threshold == pytest.approx(0.2)
    assert result.risk_bound == pytest.approx((0.4 + 0.2 + 1.0) / 3.0)
    assert result.calibration_size == 2
    assert result.trusted_pixels == 3
    assert result.total_pixels == 4


def test_alpha_below_finite_sample_correction_abstains_everywhere() -> None:
    result = calibrate_fidelity_mask(
        (torch.tensor([[0.1]]),),
        (torch.tensor([[0.0]]),),
        alpha=0.49,
    )

    assert result.threshold == float("-inf")
    assert result.trusted_pixels == 0
    assert result.risk_bound == pytest.approx(0.5)


def test_pixels_are_not_treated_as_independent_calibration_items() -> None:
    result = calibrate_fidelity_mask(
        (torch.tensor([[0.1, 0.2]]),),
        (torch.tensor([[0.1, 0.9]]),),
        alpha=0.6,
    )

    assert result.calibration_size == 1
    assert result.threshold == pytest.approx(0.1)


def test_trusted_mask_is_boolean_and_includes_scores_at_threshold() -> None:
    calibration = ConformalCalibration(
        alpha=0.5,
        threshold=0.2,
        risk_bound=0.5,
        calibration_size=1,
        trusted_pixels=1,
        total_pixels=2,
    )

    result = trusted_mask(torch.tensor([[0.1, 0.2, 0.3]]), calibration)

    assert result.dtype is torch.bool
    assert torch.equal(result, torch.tensor([[True, True, False]]))


def test_trusted_mask_abstains_everywhere_for_negative_infinity() -> None:
    calibration = ConformalCalibration(
        alpha=0.5,
        threshold=float("-inf"),
        risk_bound=0.5,
        calibration_size=1,
        trusted_pixels=0,
        total_pixels=1,
    )

    result = trusted_mask(torch.tensor([[0.0, 0.2]]), calibration)

    assert result.dtype is torch.bool
    assert not result.any()


def test_trusted_mask_rejects_nonfinite_score_maps() -> None:
    calibration = ConformalCalibration(0.5, 0.2, 0.5, 1, 1, 1)

    with pytest.raises(ValueError, match="score maps must contain only finite values"):
        trusted_mask(torch.tensor([[float("nan")]]), calibration)


def test_calibration_result_is_immutable() -> None:
    result = calibrate_fidelity_mask(
        (torch.tensor([[0.1]]),),
        (torch.tensor([[0.0]]),),
        alpha=0.5,
    )

    with pytest.raises(FrozenInstanceError):
        result.threshold = 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("alpha", float("nan"), "alpha must be a positive finite number"),
        ("threshold", float("inf"), "threshold must be finite or -inf"),
        ("risk_bound", float("inf"), "risk_bound must be a positive finite number"),
        ("calibration_size", -1, "calibration_size must be a positive integer"),
        ("trusted_pixels", -1, "trusted_pixels must be a non-negative integer"),
        ("total_pixels", -1, "total_pixels must be a non-negative integer"),
        ("trusted_pixels", 2, "trusted_pixels must not exceed total_pixels"),
    ],
)
def test_conformal_calibration_rejects_invalid_invariants(
    field: str, value: float | int, message: str
) -> None:
    values = {
        "alpha": 0.5,
        "threshold": 0.2,
        "risk_bound": 0.5,
        "calibration_size": 1,
        "trusted_pixels": 1,
        "total_pixels": 1,
    }
    values[field] = value

    with pytest.raises(ValueError, match=re.escape(message)):
        ConformalCalibration(**values)


@pytest.mark.parametrize(
    ("scores", "risks", "message"),
    [
        (
            (torch.ones((1, 1), dtype=torch.complex64),),
            (torch.zeros((1, 1)),),
            "score maps must be real-valued",
        ),
        (
            (torch.zeros((1, 1)),),
            (torch.ones((1, 1), dtype=torch.complex64),),
            "risk maps must be real-valued",
        ),
    ],
)
def test_calibration_rejects_complex_maps(
    scores: tuple[torch.Tensor, ...], risks: tuple[torch.Tensor, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        calibrate_fidelity_mask(scores, risks, alpha=0.5)


@pytest.mark.parametrize(
    ("scores", "risks", "alpha", "risk_upper_bound", "message"),
    [
        ((), (), 0.5, 1.0, "scores and risks must be non-empty"),
        (
            (torch.zeros((1, 1)),),
            (torch.zeros((1, 1)), torch.zeros((1, 1))),
            0.5,
            1.0,
            "scores and risks must have matching lengths",
        ),
        (
            (torch.zeros((1, 1, 1)),),
            (torch.zeros((1, 1, 1)),),
            0.5,
            1.0,
            "score maps must be two-dimensional",
        ),
        (
            (torch.zeros((1, 1)),),
            (torch.zeros((1, 1, 1)),),
            0.5,
            1.0,
            "risk maps must be two-dimensional",
        ),
        (
            (torch.zeros((1, 2)),),
            (torch.zeros((2, 1)),),
            0.5,
            1.0,
            "ROI shapes must match",
        ),
        (
            (torch.tensor([[float("nan")]]),),
            (torch.zeros((1, 1)),),
            0.5,
            1.0,
            "score maps must contain only finite values",
        ),
        (
            (torch.zeros((1, 1)),),
            (torch.tensor([[float("inf")]]),),
            0.5,
            1.0,
            "risk maps must contain only finite values",
        ),
        (
            (torch.tensor([[-0.1]]),),
            (torch.zeros((1, 1)),),
            0.5,
            1.0,
            "scores must be non-negative",
        ),
        (
            (torch.zeros((1, 1)),),
            (torch.tensor([[-0.1]]),),
            0.5,
            1.0,
            "risks must be non-negative",
        ),
        (
            (torch.zeros((1, 1)),),
            (torch.tensor([[1.1]]),),
            0.5,
            1.0,
            "risk exceeds risk_upper_bound",
        ),
        (
            (torch.zeros((1, 1)),),
            (torch.zeros((1, 1)),),
            0.0,
            1.0,
            "alpha must be in (0, risk_upper_bound]",
        ),
        (
            (torch.zeros((1, 1)),),
            (torch.zeros((1, 1)),),
            1.1,
            1.0,
            "alpha must be in (0, risk_upper_bound]",
        ),
        (
            (torch.zeros((1, 1)),),
            (torch.zeros((1, 1)),),
            0.5,
            0.0,
            "risk_upper_bound must be a positive finite number",
        ),
        (
            (torch.zeros((1, 1)),),
            (torch.zeros((1, 1)),),
            0.5,
            float("inf"),
            "risk_upper_bound must be a positive finite number",
        ),
    ],
)
def test_calibration_rejects_invalid_inputs(
    scores: tuple[torch.Tensor, ...],
    risks: tuple[torch.Tensor, ...],
    alpha: float,
    risk_upper_bound: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        calibrate_fidelity_mask(
            scores,
            risks,
            alpha=alpha,
            risk_upper_bound=risk_upper_bound,
        )
