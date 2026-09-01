import pytest
import torch

from trustsr.evaluation.score_diagnostics import (
    RoiScoreDiagnostics,
    evaluate_roi_score,
    score_map_spearman,
    top_fraction_jaccard,
)


def test_evaluate_roi_score_uses_stable_low_score_coverage() -> None:
    score = torch.tensor([[0.0, 0.0], [2.0, 3.0]], dtype=torch.float64)
    risk = torch.tensor([[0.1, 0.2], [0.8, 0.9]], dtype=torch.float64)

    result = evaluate_roi_score(score, risk, coverages=(0.5, 1.0))

    assert result.constant_score is False
    assert result.coverages == (0.5, 1.0)
    assert result.selective_mean_risks == pytest.approx((0.15, 0.5))
    assert result.aurc == pytest.approx(0.325)
    assert result.random_aurc == pytest.approx(0.5)
    assert result.aurc_gain == pytest.approx(0.175)


def test_constant_score_has_zero_spearman_and_row_major_tie_break() -> None:
    score = torch.zeros((2, 2), dtype=torch.float64)
    risk = torch.tensor([[0.4, 0.3], [0.2, 0.1]], dtype=torch.float64)

    result = evaluate_roi_score(score, risk, coverages=(0.5, 1.0))

    assert result.rho == 0.0
    assert result.constant_score is True
    assert result.selective_mean_risks[0] == pytest.approx(0.35)


def test_top_fraction_jaccard_uses_exact_count_and_row_major_ties() -> None:
    first = torch.tensor([[4.0, 3.0], [2.0, 1.0]], dtype=torch.float64)
    second = torch.tensor([[4.0, 1.0], [3.0, 2.0]], dtype=torch.float64)

    assert top_fraction_jaccard(first, second, fraction=0.5) == pytest.approx(1.0 / 3.0)


def test_score_map_spearman_uses_average_ranks_for_ties() -> None:
    score = torch.tensor([[1.0, 1.0, 2.0, 3.0]], dtype=torch.float64)
    risk = torch.tensor([[1.0, 2.0, 2.0, 3.0]], dtype=torch.float64)

    assert score_map_spearman(score, risk) == pytest.approx(0.8333333333333334)


def test_high_risk_miss_rate_at_80_uses_highest_tenth_and_lowest_scores() -> None:
    score = torch.arange(10, dtype=torch.float64).reshape(2, 5)
    risk = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])

    result = evaluate_roi_score(score, risk)

    assert result.high_risk_miss_rate_at_80 == pytest.approx(1.0)


def test_diagnostics_record_is_frozen() -> None:
    result = RoiScoreDiagnostics(0.0, True, (1.0,), (0.5,), 0.5, 0.5, 0.0, 0.0)

    with pytest.raises(AttributeError):
        result.rho = 1.0


@pytest.mark.parametrize("coverages", [(), (0.0,), (1.1,), (float("nan"),), (0.5, 0.5)])
def test_invalid_coverages_are_rejected(coverages: tuple[float, ...]) -> None:
    score = torch.zeros((1, 1), dtype=torch.float64)
    risk = torch.zeros((1, 1), dtype=torch.float64)

    with pytest.raises(ValueError, match="coverages"):
        evaluate_roi_score(score, risk, coverages=coverages)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_are_rejected(value: float) -> None:
    valid = torch.zeros((1, 1), dtype=torch.float64)

    with pytest.raises(ValueError, match="finite"):
        evaluate_roi_score(torch.tensor([[value]]), valid)
    with pytest.raises(ValueError, match="finite"):
        evaluate_roi_score(valid, torch.tensor([[value]]))


def test_negative_score_and_out_of_range_risk_are_rejected() -> None:
    valid = torch.zeros((1, 1), dtype=torch.float64)

    with pytest.raises(ValueError, match="non-negative"):
        evaluate_roi_score(torch.tensor([[-0.1]]), valid)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluate_roi_score(valid, torch.tensor([[1.1]]))


def test_unequal_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="shapes"):
        evaluate_roi_score(torch.zeros((1, 1)), torch.zeros((1, 2)))


def test_evaluation_does_not_mutate_inputs() -> None:
    score = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    risk = torch.tensor([[0.8, 0.2]], dtype=torch.float64)
    original_score = score.clone()
    original_risk = risk.clone()

    evaluate_roi_score(score, risk)

    assert torch.equal(score, original_score)
    assert torch.equal(risk, original_risk)
