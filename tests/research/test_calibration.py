import numpy as np
import pytest

from research.trustmask.calibration import calibrate, evaluate_comparison


def test_crc_correction_and_largest_feasible_threshold():
    losses = np.tile([0.0, 0.02, 0.8], (99, 1))
    result = calibrate(losses, np.array([0.0, 0.5, 1.0]), alpha=0.05)
    assert result["threshold"] == 0.5
    assert result["corrected_calibration_risk"] == pytest.approx(0.0298)


def test_small_sample_falls_back_to_structural_rejection():
    result = calibrate(np.zeros((1, 2)), np.array([0.0, 1.0]), alpha=0.05)
    assert result["threshold"] == -1
    assert result["reject_all"]


def test_nonmonotone_curve_rejected():
    with pytest.raises(ValueError, match="monotone"):
        calibrate(np.array([[0.4, 0.2]]), np.array([0.0, 1.0]))


def test_paired_bound_uses_range_two_and_family_size():
    n = 4000
    result = evaluate_comparison(
        {"full": np.zeros(n), "w3": np.zeros(n)},
        {"full": np.full(n, 0.8), "w3": np.full(n, 0.5)},
    )
    assert result["primary"]["lower_bound"] == pytest.approx(
        0.3 - np.sqrt(2 * np.log(1 / 0.025) / n)
    )
    assert result["methods"]["full"]["risk_upper"] == pytest.approx(
        np.sqrt(np.log(2 / 0.025) / (2 * n))
    )
    assert result["primary"]["qualified_positive_gain"]
