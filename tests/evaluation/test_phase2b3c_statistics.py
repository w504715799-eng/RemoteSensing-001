"""Independent synthetic checks for Phase 2B3-C evaluation statistics."""

from __future__ import annotations

import importlib
import math
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext

import pytest
import torch


def _statistics():
    return importlib.import_module("trustsr.evaluation.phase2b3c_statistics")


def _roi(
    index: int,
    *,
    loss: float = 0.0,
    trusted_pixels: int = 10,
    total_pixels: int = 100,
):
    statistics = _statistics()
    return statistics.ROIEvaluation(
        sample_id=f"test-{index:03d}",
        days_between=(-1, 0, 1)[(index % 12) // 4],
        correlation_bin=index % 4,
        selection_round=index // 12 + 1,
        trusted_pixels=trusted_pixels,
        total_pixels=total_pixels,
        coverage=trusted_pixels / total_pixels,
        loss=loss,
    )


def _rois(
    *,
    loss: float = 0.0,
    trusted_pixels: int = 10,
    total_pixels: int = 100,
):
    return tuple(
        _roi(
            index,
            loss=loss,
            trusted_pixels=trusted_pixels,
            total_pixels=total_pixels,
        )
        for index in range(120)
    )


def _decimal_log_evalue(losses: tuple[float, ...], candidate_mean: float) -> float:
    """Test-only direct-product oracle, intentionally unlike production log arithmetic."""

    with localcontext() as context:
        context.prec = 100
        mean = Decimal.from_float(candidate_mean)
        one = Decimal(1)
        wealth = Decimal(0)
        for j in range(1, 21):
            rho = Decimal(j) / Decimal(21)
            component = Decimal(1)
            for loss in losses:
                component *= one - rho * (Decimal.from_float(loss) - mean) / (one - mean)
            wealth += component
        return math.log(float(wealth / Decimal(20)))


def test_roi_loss_uses_inclusive_threshold_and_maximum_trusted_risk() -> None:
    statistics = _statistics()
    threshold = 7.970395366024563e-06
    score = torch.tensor(
        [[threshold, threshold + 1e-12], [threshold / 2, threshold * 2]],
        dtype=torch.float64,
    )
    risk = torch.tensor([[0.02, 0.9], [0.04, 1.0]], dtype=torch.float64)

    result = statistics.evaluate_roi_loss(
        sample_id="test-000",
        days_between=-1,
        correlation_bin=0,
        selection_round=1,
        score=score,
        risk=risk,
        threshold=threshold,
    )

    assert result.trusted_pixels == 2
    assert result.total_pixels == 4
    assert result.coverage == 0.5
    assert result.loss == 0.04


def test_roi_loss_assigns_zero_to_an_empty_trusted_mask() -> None:
    statistics = _statistics()
    threshold = 7.970395366024563e-06

    result = statistics.evaluate_roi_loss(
        sample_id="test-000",
        days_between=-1,
        correlation_bin=0,
        selection_round=1,
        score=torch.full((2, 2), threshold * 2, dtype=torch.float64),
        risk=torch.ones((2, 2), dtype=torch.float64),
        threshold=threshold,
    )

    assert result.trusted_pixels == 0
    assert result.coverage == 0.0
    assert result.loss == 0.0


@pytest.mark.parametrize(
    ("score", "risk", "message"),
    (
        (
            torch.zeros((2, 2), dtype=torch.float32),
            torch.zeros((2, 2), dtype=torch.float64),
            "float64",
        ),
        (
            torch.zeros((2, 1), dtype=torch.float64),
            torch.zeros((1, 2), dtype=torch.float64),
            "shape",
        ),
        (
            torch.tensor([[float("nan")]], dtype=torch.float64),
            torch.zeros((1, 1), dtype=torch.float64),
            "finite",
        ),
        (
            torch.tensor([[0.251]], dtype=torch.float64),
            torch.zeros((1, 1), dtype=torch.float64),
            r"\[0, 0.25\]",
        ),
        (
            torch.zeros((1, 1), dtype=torch.float64),
            torch.tensor([[1.01]], dtype=torch.float64),
            r"\[0, 1\]",
        ),
    ),
)
def test_roi_loss_rejects_malformed_maps(
    score: torch.Tensor, risk: torch.Tensor, message: str
) -> None:
    statistics = _statistics()

    with pytest.raises(ValueError, match=message):
        statistics.evaluate_roi_loss(
            sample_id="test-000",
            days_between=-1,
            correlation_bin=0,
            selection_round=1,
            score=score,
            risk=risk,
            threshold=7.970395366024563e-06,
        )


def test_grid_kelly_log_evalue_matches_independent_direct_product_oracle() -> None:
    statistics = _statistics()
    losses = tuple([0.0] * 40 + [0.01] * 40 + [0.04] * 40)

    observed = statistics.grid_kelly_log_evalue(losses, 0.05)

    assert observed == pytest.approx(_decimal_log_evalue(losses, 0.05), abs=2e-14)


def test_grid_kelly_is_permutation_invariant_and_nondecreasing_in_candidate_mean() -> None:
    statistics = _statistics()
    losses = tuple(index / 1000 for index in range(120))
    reversed_losses = tuple(reversed(losses))

    values = [statistics.grid_kelly_log_evalue(losses, mean) for mean in (0.01, 0.05, 0.1)]

    assert values[0] <= values[1] <= values[2]
    assert statistics.grid_kelly_log_evalue(losses, 0.05) == pytest.approx(
        statistics.grid_kelly_log_evalue(reversed_losses, 0.05), abs=1e-14
    )


def test_grid_kelly_upper_bound_has_hand_checked_endpoint_behavior() -> None:
    statistics = _statistics()

    assert statistics.grid_kelly_upper_bound((0.0,) * 120) == pytest.approx(
        0.037449055951085546, abs=1e-15
    )
    assert statistics.grid_kelly_upper_bound((1.0,) * 120) == 1.0


def test_statistics_confirm_only_when_primary_ucb_and_coverage_pass() -> None:
    statistics = _statistics()

    result = statistics.build_phase2b3c_statistics(_rois(loss=0.0))

    assert result.mean_loss == 0.0
    assert result.empirical_risk_delta == -0.05
    assert result.empirical_risk_violation == 0.0
    assert result.risk_ucb == pytest.approx(0.037449055951085546, abs=1e-15)
    assert result.finite_sample_margin == result.risk_ucb
    assert result.finite_sample_delta == pytest.approx(result.risk_ucb - 0.05)
    assert result.hoeffding_ucb == pytest.approx(math.sqrt(math.log(20) / 240))
    assert result.coverage == 0.1
    assert result.phase_decision == "confirmed"
    assert result.reasons == ()


def test_statistics_report_empirical_success_as_inconclusive_when_ucb_misses() -> None:
    statistics = _statistics()

    result = statistics.build_phase2b3c_statistics(_rois(loss=0.02))

    assert result.mean_loss == pytest.approx(0.02)
    assert result.risk_ucb > 0.05
    assert result.phase_decision == "empirically_met_but_inconclusive"
    assert result.reasons == ("finite_sample_upper_bound_exceeds_target",)


def test_statistics_fail_and_report_every_applicable_reason() -> None:
    statistics = _statistics()

    result = statistics.build_phase2b3c_statistics(
        _rois(loss=0.06, trusted_pixels=9, total_pixels=100)
    )

    assert result.coverage == 0.09
    assert result.mean_loss == pytest.approx(0.06)
    assert result.empirical_risk_violation == pytest.approx(0.01)
    assert result.phase_decision == "failed"
    assert result.reasons == (
        "insufficient_coverage",
        "empirical_risk_exceeds_target",
    )


def test_statistics_emit_exact_ordered_stratum_diagnostics() -> None:
    statistics = _statistics()
    rois = list(_rois())
    rois[0] = replace(rois[0], loss=0.04, trusted_pixels=20, coverage=0.2)

    result = statistics.build_phase2b3c_statistics(rois)

    assert len(result.strata) == 12
    assert result.strata[0].days_between == -1
    assert result.strata[0].correlation_bin == 0
    assert result.strata[0].roi_count == 10
    assert result.strata[0].trusted_pixels == 110
    assert result.strata[0].total_pixels == 1000
    assert result.strata[0].coverage == 0.11
    assert result.strata[0].mean_loss == 0.004
    assert result.strata[0].maximum_loss == 0.04


@pytest.mark.parametrize("count", (119, 121))
def test_statistics_reject_any_roi_count_other_than_120(count: int) -> None:
    statistics = _statistics()
    values = list(_rois())
    if count == 119:
        values.pop()
    else:
        values.append(replace(values[-1], sample_id="test-extra"))

    with pytest.raises(ValueError, match="120"):
        statistics.build_phase2b3c_statistics(values)


def test_statistics_reject_noncanonical_or_duplicate_membership() -> None:
    statistics = _statistics()
    reordered = list(_rois())
    reordered[0], reordered[1] = reordered[1], reordered[0]
    duplicated = list(_rois())
    duplicated[-1] = duplicated[0]

    with pytest.raises(ValueError, match="canonical"):
        statistics.build_phase2b3c_statistics(reordered)
    with pytest.raises(ValueError, match="unique"):
        statistics.build_phase2b3c_statistics(duplicated)


def test_public_statistic_records_are_frozen_and_json_safe() -> None:
    statistics = _statistics()
    result = statistics.build_phase2b3c_statistics(_rois())

    payload = result.as_dict()
    assert payload["method"] == {
        "name": "one_sided_diversified_grid_kelly",
        "confidence_error": 0.05,
        "grid_size": 20,
        "bet_fractions": [index / 21 for index in range(1, 21)],
        "bisection_iterations": 128,
    }
    assert len(payload["strata"]) == 12
    payload["strata"].append({"mutated": True})
    assert len(result.as_dict()["strata"]) == 12
    with pytest.raises(FrozenInstanceError):
        result.mean_loss = 0.5  # type: ignore[misc]
