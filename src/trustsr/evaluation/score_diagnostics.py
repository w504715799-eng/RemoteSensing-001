"""Deterministic diagnostics for one ROI score map and its empirical risk map."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import rankdata

DEFAULT_COVERAGES = tuple(index / 10 for index in range(1, 11))


@dataclass(frozen=True)
class RoiScoreDiagnostics:
    """Score-quality diagnostics computed for a single ROI."""

    rho: float
    constant_score: bool
    coverages: tuple[float, ...]
    selective_mean_risks: tuple[float, ...]
    aurc: float
    random_aurc: float
    aurc_gain: float
    high_risk_miss_rate_at_80: float


def _validate_map(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if value.is_complex():
        raise ValueError(f"{name} must be real-valued")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def _validated_flat_arrays(
    score: torch.Tensor, risk: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    _validate_map(score, name="score")
    _validate_map(risk, name="risk")
    if score.shape != risk.shape:
        raise ValueError("score and risk shapes must match")
    if (score < 0).any():
        raise ValueError("scores must be non-negative")
    if (risk < 0).any() or (risk > 1).any():
        raise ValueError("risks must be in [0, 1]")
    return (
        score.detach().to(device="cpu", dtype=torch.float64).numpy().reshape(-1),
        risk.detach().to(device="cpu", dtype=torch.float64).numpy().reshape(-1),
    )


def _validated_matching_arrays(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    _validate_map(first, name="first")
    _validate_map(second, name="second")
    if first.shape != second.shape:
        raise ValueError("map shapes must match")
    return (
        first.detach().to(device="cpu", dtype=torch.float64).numpy().reshape(-1),
        second.detach().to(device="cpu", dtype=torch.float64).numpy().reshape(-1),
    )


def _validated_fraction(value: float, *, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be finite and in (0, 1]")
    fraction = float(value)
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError(f"{name} must be finite and in (0, 1]")
    return fraction


def _validated_coverages(coverages: tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(coverages, tuple) or not coverages:
        raise ValueError("coverages must be a non-empty tuple")
    values = tuple(_validated_fraction(coverage, name="coverages") for coverage in coverages)
    if any(left >= right for left, right in zip(values, values[1:], strict=False)):
        raise ValueError("coverages must be strictly increasing")
    return values


def _finite(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("diagnostic results must be finite")
    return result


def _spearman(score_values: np.ndarray, risk_values: np.ndarray) -> float:
    if np.ptp(score_values) == 0.0 or np.ptp(risk_values) == 0.0:
        return 0.0
    score_ranks = rankdata(score_values, method="average")
    risk_ranks = rankdata(risk_values, method="average")
    return _finite(np.corrcoef(score_ranks, risk_ranks)[0, 1])


def score_map_spearman(score: torch.Tensor, risk: torch.Tensor) -> float:
    """Return average-rank Spearman correlation for matching score and risk maps."""
    score_values, risk_values = _validated_matching_arrays(score, risk)
    return _spearman(score_values, risk_values)


def _top_indices(values: np.ndarray, fraction: float) -> np.ndarray:
    count = math.ceil(fraction * values.size)
    return np.lexsort((np.arange(values.size), -values))[:count]


def top_fraction_jaccard(
    first: torch.Tensor, second: torch.Tensor, *, fraction: float
) -> float:
    """Return the Jaccard overlap of the deterministically highest-valued pixels."""
    first_values, second_values = _validated_matching_arrays(first, second)
    fraction_value = _validated_fraction(fraction, name="fraction")
    first_indices = _top_indices(first_values, fraction_value)
    second_indices = _top_indices(second_values, fraction_value)
    intersection = np.intersect1d(first_indices, second_indices, assume_unique=True).size
    union = np.union1d(first_indices, second_indices).size
    return _finite(intersection / union)


def _high_risk_miss(score_values: np.ndarray, risk_values: np.ndarray) -> float:
    high_risk_indices = _top_indices(risk_values, 0.1)
    trusted_count = math.ceil(0.8 * score_values.size)
    trusted_indices = np.lexsort((np.arange(score_values.size), score_values))[:trusted_count]
    selected_high_risk = np.intersect1d(
        high_risk_indices, trusted_indices, assume_unique=True
    ).size
    return _finite(1.0 - selected_high_risk / high_risk_indices.size)


def evaluate_roi_score(
    score: torch.Tensor,
    risk: torch.Tensor,
    *,
    coverages: tuple[float, ...] = DEFAULT_COVERAGES,
) -> RoiScoreDiagnostics:
    """Compute stable one-ROI diagnostics, selecting lower score values first."""
    coverage_values = _validated_coverages(coverages)
    score_values, risk_values = _validated_flat_arrays(score, risk)
    order = np.lexsort((np.arange(score_values.size), score_values))
    selected_risks = tuple(
        _finite(risk_values[order[: math.ceil(coverage * order.size)]].mean())
        for coverage in coverage_values
    )
    random_aurc = _finite(risk_values.mean())
    aurc = _finite(np.mean(selected_risks))
    return RoiScoreDiagnostics(
        rho=_spearman(score_values, risk_values),
        constant_score=bool(np.ptp(score_values) == 0.0),
        coverages=coverage_values,
        selective_mean_risks=selected_risks,
        aurc=aurc,
        random_aurc=random_aurc,
        aurc_gain=_finite(random_aurc - aurc),
        high_risk_miss_rate_at_80=_high_risk_miss(score_values, risk_values),
    )
