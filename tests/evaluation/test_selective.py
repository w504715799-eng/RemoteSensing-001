import pytest
import torch

from trustsr.evaluation.selective import SelectivePoint, evaluate_selective_point


def test_selective_point_reports_pixel_coverage_and_mean_roi_maximum() -> None:
    scores = (
        torch.tensor([[0.1, 0.4]]),
        torch.tensor([[0.2, 0.3]]),
    )
    risks = (
        torch.tensor([[0.2, 0.8]]),
        torch.tensor([[0.3, 0.7]]),
    )

    result = evaluate_selective_point(scores, risks, threshold=0.2)

    assert result.threshold == 0.2
    assert result.coverage == pytest.approx(0.5)
    assert result.roi_max_risk == pytest.approx(0.25)


def test_all_abstain_reports_zero_coverage_and_risk() -> None:
    scores = (torch.tensor([[0.1, 0.4]]), torch.tensor([[0.2, 0.3]]))
    risks = (torch.tensor([[0.2, 0.8]]), torch.tensor([[0.3, 0.7]]))

    result = evaluate_selective_point(scores, risks, threshold=float("-inf"))

    assert result == SelectivePoint(threshold=float("-inf"), coverage=0.0, roi_max_risk=0.0)


def test_lower_threshold_has_no_larger_mean_roi_maximum_risk() -> None:
    scores = (torch.tensor([[0.1, 0.2, 0.3]]),)
    risks = (torch.tensor([[0.1, 0.4, 0.9]]),)

    lower = evaluate_selective_point(scores, risks, threshold=0.1)
    higher = evaluate_selective_point(scores, risks, threshold=0.3)

    assert lower.roi_max_risk <= higher.roi_max_risk


def test_selective_point_is_frozen() -> None:
    point = SelectivePoint(threshold=0.2, coverage=0.5, roi_max_risk=0.2)

    with pytest.raises(AttributeError):
        point.coverage = 0.6


@pytest.mark.parametrize(
    ("scores", "risks", "message"),
    [
        ((), (), "non-empty"),
        (
            (torch.zeros((1, 1)), torch.zeros((1, 1))),
            (torch.zeros((1, 1)),),
            "matching lengths",
        ),
        ((torch.zeros((1, 1)),), (torch.zeros((1, 2)),), "shapes"),
        ((torch.zeros(1),), (torch.zeros(1),), "two-dimensional"),
        ((torch.tensor([[float("nan")]]),), (torch.zeros((1, 1)),), "finite"),
        ((torch.tensor([[-0.1]]),), (torch.zeros((1, 1)),), "non-negative"),
        ((torch.zeros((1, 1)),), (torch.tensor([[-0.1]]),), r"\[0, 1\]"),
        ((torch.zeros((1, 1)),), (torch.tensor([[1.1]]),), r"\[0, 1\]"),
    ],
)
def test_invalid_score_and_risk_inputs_are_rejected(scores, risks, message) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_selective_point(scores, risks, threshold=0.5)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf")])
def test_threshold_must_be_finite_or_negative_infinity(threshold: float) -> None:
    with pytest.raises(ValueError, match="finite or -inf"):
        evaluate_selective_point(
            (torch.zeros((1, 1)),),
            (torch.zeros((1, 1)),),
            threshold=threshold,
        )
