"""Deterministic development-ROI score qualification and freezing."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

BOOTSTRAP_SEED = 23031
BOOTSTRAP_RESAMPLES = 10_000
ROI_COUNT = 120
COST_ORDER = (
    "lr_reprojection_l1",
    "three_model_disagreement",
    "ldsr_variance_k5",
)


@dataclass(frozen=True)
class DevelopmentRoiResult:
    sample_id: str
    spatial_group_id: str
    days_between: int
    correlation_bin: int
    selection_round: int
    rho: float
    constant_score: bool
    aurc_gain: float
    high_risk_miss_rate_at_80: float


@dataclass(frozen=True)
class CandidateSummary:
    name: str
    eligible: bool
    failure_reasons: tuple[str, ...]
    nonconstant_count: int
    mean_rho: float
    mean_rho_ci95: tuple[float, float]
    mean_aurc_gain: float
    mean_aurc_gain_ci95: tuple[float, float]
    positive_strata: int
    minimum_stratum_mean_rho: float


@dataclass(frozen=True)
class FrozenScore:
    name: str
    cost_rank: int
    statistical_leader: str
    indistinguishable_candidates: tuple[str, ...]
    candidate_summaries: tuple[CandidateSummary, ...]


def build_bootstrap_indices() -> np.ndarray:
    """Return the preregistered ROI-level bootstrap resamples."""
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    return rng.integers(
        0, ROI_COUNT, size=(BOOTSTRAP_RESAMPLES, ROI_COUNT), dtype=np.int64
    )


def summarize_candidate(
    name: str,
    results: Sequence[DevelopmentRoiResult],
    *,
    bootstrap_indices: np.ndarray | None = None,
) -> CandidateSummary:
    """Validate and summarize a candidate from all fixed development ROIs."""
    indices = build_bootstrap_indices() if bootstrap_indices is None else bootstrap_indices
    validated = _validate_candidate(results)
    _validate_bootstrap_indices(indices)
    return _summarize_validated_candidate(name, validated, indices)


def freeze_score(candidates: Mapping[str, Sequence[DevelopmentRoiResult]]) -> FrozenScore:
    """Freeze the cheapest eligible candidate indistinguishable from the leader."""
    indices = build_bootstrap_indices()
    validated = _validate_matching_candidate_membership(candidates)
    summaries = tuple(
        _summarize_validated_candidate(name, validated[name], indices)
        for name in COST_ORDER
        if name in validated
    )
    eligible = tuple(summary for summary in summaries if summary.eligible)
    if not eligible:
        raise ValueError("no development score candidate is eligible")
    leader = max(eligible, key=lambda item: item.mean_rho)
    indistinguishable = _paired_indistinguishable(leader, eligible, validated, indices)
    chosen = min(indistinguishable, key=COST_ORDER.index)
    return FrozenScore(
        name=chosen,
        cost_rank=COST_ORDER.index(chosen),
        statistical_leader=leader.name,
        indistinguishable_candidates=tuple(
            name for name in COST_ORDER if name in indistinguishable
        ),
        candidate_summaries=summaries,
    )


def _validate_candidate(
    results: Sequence[DevelopmentRoiResult],
) -> tuple[DevelopmentRoiResult, ...]:
    if len(results) != ROI_COUNT:
        raise ValueError("candidate must contain exactly 120 ROI results")
    if len({item.sample_id for item in results}) != ROI_COUNT:
        raise ValueError("candidate must contain unique sample_id values")
    if len({item.spatial_group_id for item in results}) != ROI_COUNT:
        raise ValueError("candidate must contain unique spatial_group_id values")
    cells: dict[tuple[int, int], list[int]] = {}
    for item in results:
        if item.days_between not in (-1, 0, 1):
            raise ValueError("days_between must be one of -1, 0, 1")
        if item.correlation_bin not in range(4):
            raise ValueError("correlation_bin must be between 0 and 3")
        if item.selection_round not in range(1, 11):
            raise ValueError("selection_round must be between 1 and 10")
        if not np.isfinite(
            (item.rho, item.aurc_gain, item.high_risk_miss_rate_at_80)
        ).all():
            raise ValueError("ROI statistics must be finite")
        cells.setdefault((item.days_between, item.correlation_bin), []).append(
            item.selection_round
        )
    expected_cells = {(day, bin_index) for day in (-1, 0, 1) for bin_index in range(4)}
    if set(cells) != expected_cells or any(
        sorted(rounds) != list(range(1, 11)) for rounds in cells.values()
    ):
        raise ValueError("candidate must contain exactly 10 selection rounds in every stratum")
    return tuple(sorted(results, key=_roi_sort_key))


def _validate_matching_candidate_membership(
    candidates: Mapping[str, Sequence[DevelopmentRoiResult]],
) -> dict[str, tuple[DevelopmentRoiResult, ...]]:
    unknown = set(candidates).difference(COST_ORDER)
    if unknown:
        raise ValueError("unknown development score candidate")
    validated = {name: _validate_candidate(results) for name, results in candidates.items()}
    if not validated:
        return validated
    memberships = {
        name: tuple(_membership(item) for item in values)
        for name, values in validated.items()
    }
    first = next(iter(memberships.values()))
    if any(membership != first for membership in memberships.values()):
        raise ValueError("candidate membership must match exactly")
    return validated


def _summarize_validated_candidate(
    name: str,
    results: tuple[DevelopmentRoiResult, ...],
    indices: np.ndarray,
) -> CandidateSummary:
    rho = np.asarray([item.rho for item in results], dtype=np.float64)
    gains = np.asarray([item.aurc_gain for item in results], dtype=np.float64)
    rho_means = rho[indices].mean(axis=1)
    gain_means = gains[indices].mean(axis=1)
    rho_ci = _percentile_interval(rho_means)
    gain_ci = _percentile_interval(gain_means)
    stratum_means = np.asarray(
        [
            np.mean(
                [
                    item.rho
                    for item in results
                    if (item.days_between, item.correlation_bin) == (day, bin_index)
                ]
            )
            for day in (-1, 0, 1)
            for bin_index in range(4)
        ],
        dtype=np.float64,
    )
    nonconstant_count = sum(not item.constant_score for item in results)
    positive_strata = int(np.count_nonzero(stratum_means > 0))
    minimum_stratum_mean_rho = float(stratum_means.min())
    failure_reasons = tuple(
        reason
        for passed, reason in (
            (nonconstant_count >= 114, "fewer than 114 nonconstant ROI scores"),
            (rho_ci[0] > 0, "mean rho bootstrap lower bound is not greater than zero"),
            (gain_ci[0] > 0, "mean AURC gain bootstrap lower bound is not greater than zero"),
            (positive_strata >= 9, "fewer than 9 strata have positive mean rho"),
            (minimum_stratum_mean_rho >= -0.10, "a stratum mean rho is below -0.10"),
        )
        if not passed
    )
    return CandidateSummary(
        name=name,
        eligible=not failure_reasons,
        failure_reasons=failure_reasons,
        nonconstant_count=nonconstant_count,
        mean_rho=float(rho.mean()),
        mean_rho_ci95=rho_ci,
        mean_aurc_gain=float(gains.mean()),
        mean_aurc_gain_ci95=gain_ci,
        positive_strata=positive_strata,
        minimum_stratum_mean_rho=minimum_stratum_mean_rho,
    )


def _paired_indistinguishable(
    leader: CandidateSummary,
    eligible: tuple[CandidateSummary, ...],
    candidates: Mapping[str, tuple[DevelopmentRoiResult, ...]],
    indices: np.ndarray,
) -> set[str]:
    leader_rho = np.asarray([item.rho for item in candidates[leader.name]], dtype=np.float64)
    indistinguishable = {leader.name}
    for candidate in eligible:
        if candidate.name == leader.name:
            continue
        candidate_rho = np.asarray(
            [item.rho for item in candidates[candidate.name]], dtype=np.float64
        )
        difference_ci = _percentile_interval((leader_rho - candidate_rho)[indices].mean(axis=1))
        if difference_ci[0] <= 0:
            indistinguishable.add(candidate.name)
    return indistinguishable


def _validate_bootstrap_indices(indices: np.ndarray) -> None:
    if indices.shape != (BOOTSTRAP_RESAMPLES, ROI_COUNT) or indices.dtype != np.int64:
        raise ValueError("bootstrap indices must be an int64 array with shape (10000, 120)")
    if int(indices.min()) < 0 or int(indices.max()) >= ROI_COUNT:
        raise ValueError("bootstrap indices must refer to the 120 ROI results")


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.percentile(values, (2.5, 97.5))
    return float(lower), float(upper)


def _roi_sort_key(item: DevelopmentRoiResult) -> tuple[int, int, int]:
    return item.days_between, item.correlation_bin, item.selection_round


def _membership(item: DevelopmentRoiResult) -> tuple[str, str, int, int, int]:
    return (
        item.sample_id,
        item.spatial_group_id,
        item.days_between,
        item.correlation_bin,
        item.selection_round,
    )
