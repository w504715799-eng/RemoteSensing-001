from dataclasses import replace

import numpy as np
import pytest

from trustsr.evaluation.score_selection import (
    DevelopmentRoiResult,
    build_bootstrap_indices,
    freeze_score,
    summarize_candidate,
)


def _candidate(rho: float, gain: float) -> list[DevelopmentRoiResult]:
    return [
        DevelopmentRoiResult(
            sample_id=f"sample-{day}-{bin_index}-{round_index}",
            spatial_group_id=f"group-{day}-{bin_index}-{round_index}",
            days_between=day,
            correlation_bin=bin_index,
            selection_round=round_index,
            rho=rho,
            constant_score=False,
            aurc_gain=gain,
            high_risk_miss_rate_at_80=0.5,
        )
        for day in (-1, 0, 1)
        for bin_index in range(4)
        for round_index in range(1, 11)
    ]


def test_bootstrap_indices_are_fixed_roi_resamples() -> None:
    first = build_bootstrap_indices()
    second = build_bootstrap_indices()

    assert first.shape == (10_000, 120)
    assert first.dtype == np.int64
    assert np.array_equal(first, second)
    assert int(first.min()) >= 0 and int(first.max()) < 120


def test_candidate_must_pass_all_five_eligibility_rules() -> None:
    summary = summarize_candidate("lr_reprojection_l1", _candidate(0.2, 0.03))

    assert summary.eligible is True
    assert summary.failure_reasons == ()


def test_candidate_rejects_fewer_than_114_nonconstant_rois() -> None:
    results = _candidate(0.2, 0.03)
    for index in range(7):
        results[index] = replace(results[index], constant_score=True)

    summary = summarize_candidate("lr_reprojection_l1", results)

    assert summary.eligible is False
    assert summary.failure_reasons == ("fewer than 114 nonconstant ROI scores",)


def test_candidate_rejects_rho_ci_with_lower_bound_equal_to_zero() -> None:
    results = _candidate(0.0, 0.03)
    for index in range(0, 90, 10):
        results[index] = replace(results[index], rho=np.nextafter(0.0, 1.0))

    summary = summarize_candidate("lr_reprojection_l1", results)

    assert summary.mean_rho_ci95[0] == 0.0
    assert summary.eligible is False
    assert "mean rho bootstrap lower bound is not greater than zero" in summary.failure_reasons


def test_candidate_rejects_aurc_gain_ci_with_lower_bound_equal_to_zero() -> None:
    results = _candidate(0.2, 0.0)
    results[0] = replace(results[0], aurc_gain=np.nextafter(0.0, 1.0))

    summary = summarize_candidate("lr_reprojection_l1", results)

    assert summary.mean_aurc_gain_ci95[0] == 0.0
    assert summary.eligible is False
    assert summary.failure_reasons == (
        "mean AURC gain bootstrap lower bound is not greater than zero",
    )


def test_candidate_rejects_only_eight_positive_strata() -> None:
    results = _candidate(0.0, 0.03)
    for index in range(80):
        results[index] = replace(results[index], rho=0.2)

    summary = summarize_candidate("lr_reprojection_l1", results)

    assert summary.positive_strata == 8
    assert summary.eligible is False
    assert summary.failure_reasons == ("fewer than 9 strata have positive mean rho",)


def test_candidate_rejects_a_stratum_below_negative_point_ten() -> None:
    results = _candidate(0.2, 0.03)
    for index in range(10):
        results[index] = replace(results[index], rho=-0.11)

    summary = summarize_candidate("lr_reprojection_l1", results)

    assert summary.minimum_stratum_mean_rho == pytest.approx(-0.11)
    assert summary.eligible is False
    assert "a stratum mean rho is below -0.10" in summary.failure_reasons


def test_candidate_rejects_a_missing_roi() -> None:
    with pytest.raises(ValueError, match="exactly 120"):
        summarize_candidate("lr_reprojection_l1", _candidate(0.2, 0.03)[:-1])


@pytest.mark.parametrize("field", ("sample_id", "spatial_group_id"))
def test_candidate_rejects_duplicate_sample_or_spatial_group(field: str) -> None:
    results = _candidate(0.2, 0.03)
    results[-1] = replace(results[-1], **{field: getattr(results[0], field)})

    with pytest.raises(ValueError, match=f"unique {field}"):
        summarize_candidate("lr_reprojection_l1", results)


def test_freeze_rejects_mismatched_candidate_membership() -> None:
    first = _candidate(0.2, 0.03)
    second = _candidate(0.2, 0.03)
    second[-1] = replace(second[-1], sample_id="other-sample")

    with pytest.raises(ValueError, match="membership"):
        freeze_score({"lr_reprojection_l1": first, "three_model_disagreement": second})


def test_freeze_prefers_cheapest_statistically_indistinguishable_score() -> None:
    candidates = {
        "lr_reprojection_l1": _candidate(0.20, 0.03),
        "three_model_disagreement": _candidate(0.20, 0.03),
        "ldsr_variance_k5": _candidate(0.20, 0.03),
    }

    frozen = freeze_score(candidates)

    assert frozen.name == "lr_reprojection_l1"
    assert frozen.cost_rank == 0
    assert frozen.indistinguishable_candidates == (
        "lr_reprojection_l1",
        "three_model_disagreement",
        "ldsr_variance_k5",
    )


def test_freeze_selects_a_clearly_superior_expensive_candidate() -> None:
    cheaper = _candidate(0.1, 0.03)
    expensive = _candidate(0.9, 0.03)

    frozen = freeze_score(
        {"lr_reprojection_l1": cheaper, "ldsr_variance_k5": expensive}
    )

    assert frozen.statistical_leader == "ldsr_variance_k5"
    assert frozen.indistinguishable_candidates == ("ldsr_variance_k5",)


def test_freeze_raises_when_no_candidate_is_eligible() -> None:
    with pytest.raises(ValueError, match="no development score candidate is eligible"):
        freeze_score({"lr_reprojection_l1": _candidate(0.0, 0.0)})
