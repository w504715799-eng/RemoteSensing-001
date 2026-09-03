"""ROI-level conformal calibration for fidelity-based trusted masks."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ConformalCalibration:
    """A calibrated score threshold and its ROI-level risk bound."""

    alpha: float
    threshold: float
    risk_bound: float
    calibration_size: int
    trusted_pixels: int
    total_pixels: int

    def __post_init__(self) -> None:
        _validate_positive_finite(self.alpha, name="alpha")
        _validate_threshold(self.threshold)
        _validate_positive_finite(self.risk_bound, name="risk_bound")
        _validate_count(self.calibration_size, name="calibration_size", allow_zero=False)
        _validate_count(self.trusted_pixels, name="trusted_pixels", allow_zero=True)
        _validate_count(self.total_pixels, name="total_pixels", allow_zero=False)
        if self.trusted_pixels > self.total_pixels:
            raise ValueError("trusted_pixels must not exceed total_pixels")
        if self.threshold == float("-inf") and self.trusted_pixels != 0:
            raise ValueError("-inf threshold must have zero trusted_pixels")


@dataclass(frozen=True)
class _ThresholdSweep:
    """Grouped score events used by the private exact threshold sweep."""

    thresholds: torch.Tensor
    worst_risk_sums: torch.Tensor
    event_count: int


def _validate_positive_finite(value: float, *, name: str) -> float:
    if not _is_runtime_real(value):
        raise ValueError(f"{name} must be a positive finite number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return numeric_value


def _validate_alpha(alpha: float, *, risk_upper_bound: float) -> float:
    if not _is_runtime_real(alpha):
        raise ValueError("alpha must be in (0, risk_upper_bound]")
    numeric_alpha = float(alpha)
    if not math.isfinite(numeric_alpha) or not 0 < numeric_alpha <= risk_upper_bound:
        raise ValueError("alpha must be in (0, risk_upper_bound]")
    return numeric_alpha


def _validate_threshold(threshold: float) -> float:
    if not _is_runtime_real(threshold):
        raise ValueError("threshold must be finite or -inf")
    numeric_threshold = float(threshold)
    if numeric_threshold == float("-inf"):
        return numeric_threshold
    if not math.isfinite(numeric_threshold) or numeric_threshold < 0:
        raise ValueError("threshold must be finite or -inf")
    return numeric_threshold


def _is_runtime_real(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _validate_count(value: int, *, name: str, allow_zero: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")


def _validate_score_map(score: torch.Tensor) -> torch.Tensor:
    if not isinstance(score, torch.Tensor):
        raise ValueError("score maps must be torch.Tensors")
    if score.ndim != 2:
        raise ValueError("score maps must be two-dimensional")
    if 0 in score.shape:
        raise ValueError("score maps must be non-empty two-dimensional tensors")
    if score.is_complex():
        raise ValueError("score maps must be real-valued")
    normalized_score = score.to(dtype=torch.float64)
    if not torch.isfinite(normalized_score).all():
        raise ValueError("score maps must contain only finite values")
    if (normalized_score < 0).any():
        raise ValueError("scores must be non-negative")
    return normalized_score


def _validate_risk_map(
    risk: torch.Tensor, *, risk_upper_bound: float
) -> torch.Tensor:
    if not isinstance(risk, torch.Tensor):
        raise ValueError("risk maps must be torch.Tensors")
    if risk.ndim != 2:
        raise ValueError("risk maps must be two-dimensional")
    if 0 in risk.shape:
        raise ValueError("risk maps must be non-empty two-dimensional tensors")
    if risk.is_complex():
        raise ValueError("risk maps must be real-valued")
    normalized_risk = risk.to(dtype=torch.float64)
    if not torch.isfinite(normalized_risk).all():
        raise ValueError("risk maps must contain only finite values")
    if (normalized_risk < 0).any():
        raise ValueError("risks must be non-negative")
    if (normalized_risk > risk_upper_bound).any():
        raise ValueError("risk exceeds risk_upper_bound")
    return normalized_risk


def _validated_maps(
    scores: Sequence[torch.Tensor],
    risks: Sequence[torch.Tensor],
    *,
    risk_upper_bound: float,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    try:
        score_count = len(scores)
        risk_count = len(risks)
    except TypeError as error:
        raise ValueError("scores and risks must be sequences") from error
    if score_count == 0 or risk_count == 0:
        raise ValueError("scores and risks must be non-empty")
    if score_count != risk_count:
        raise ValueError("scores and risks must have matching lengths")

    validated_scores: list[torch.Tensor] = []
    validated_risks: list[torch.Tensor] = []
    for score, risk in zip(scores, risks, strict=True):
        normalized_score = _validate_score_map(score)
        normalized_risk = _validate_risk_map(
            risk, risk_upper_bound=risk_upper_bound
        )
        if score.shape != risk.shape:
            raise ValueError("ROI shapes must match")
        validated_scores.append(normalized_score.to(device="cpu"))
        validated_risks.append(normalized_risk.to(device="cpu"))
    return tuple(validated_scores), tuple(validated_risks)


def _risk_bound(
    scores: Sequence[torch.Tensor],
    risks: Sequence[torch.Tensor],
    *,
    threshold: float,
    risk_upper_bound: float,
) -> float:
    worst_risks = []
    for score, risk in zip(scores, risks, strict=True):
        selected_risks = risk[score <= threshold]
        worst_risks.append(0.0 if selected_risks.numel() == 0 else selected_risks.max().item())
    return (sum(worst_risks) + risk_upper_bound) / (len(scores) + 1)


def _roi_score_events(
    score: torch.Tensor, risk: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return score-group events for one ROI's cumulative maximum risk."""
    sorted_score, order = torch.sort(score.reshape(-1))
    sorted_risk = risk.reshape(-1)[order]
    group_starts = torch.ones_like(sorted_score, dtype=torch.bool)
    group_starts[1:] = sorted_score[1:] != sorted_score[:-1]
    group_ids = group_starts.cumsum(dim=0, dtype=torch.int64) - 1
    group_count = int(group_starts.sum().item())

    group_maxima = torch.zeros(group_count, dtype=torch.float64)
    group_maxima.scatter_reduce_(
        0, group_ids, sorted_risk, reduce="amax", include_self=False
    )
    running_maxima = torch.cummax(group_maxima, dim=0).values
    increments = running_maxima.clone()
    increments[1:] -= running_maxima[:-1]
    return sorted_score[group_starts], increments


def _sweep_thresholds(
    scores: Sequence[torch.Tensor], risks: Sequence[torch.Tensor]
) -> _ThresholdSweep:
    """Sweep all observed thresholds with one grouped event per ROI score value."""
    total_pixels = sum(score.numel() for score in scores)
    event_scores = torch.empty(total_pixels, dtype=torch.float64)
    event_increments = torch.empty(total_pixels, dtype=torch.float64)
    event_offset = 0
    for score, risk in zip(scores, risks, strict=True):
        roi_scores, roi_increments = _roi_score_events(score, risk)
        next_offset = event_offset + roi_scores.numel()
        event_scores[event_offset:next_offset] = roi_scores
        event_increments[event_offset:next_offset] = roi_increments
        event_offset = next_offset

    event_scores = event_scores[:event_offset]
    event_increments = event_increments[:event_offset]
    sorted_scores, order = torch.sort(event_scores)
    sorted_increments = event_increments[order]
    del event_scores, event_increments, order
    thresholds, threshold_pixel_counts = torch.unique_consecutive(
        sorted_scores, return_counts=True
    )
    risk_increments = torch.segment_reduce(
        sorted_increments, reduce="sum", lengths=threshold_pixel_counts
    )
    return _ThresholdSweep(
        thresholds=thresholds,
        worst_risk_sums=torch.cumsum(risk_increments, dim=0),
        event_count=event_offset,
    )


def calibrate_fidelity_mask(
    scores: Sequence[torch.Tensor],
    risks: Sequence[torch.Tensor],
    *,
    alpha: float,
    risk_upper_bound: float = 1.0,
) -> ConformalCalibration:
    """Calibrate the largest observed score threshold satisfying an ROI-level bound."""
    validated_upper_bound = _validate_positive_finite(
        risk_upper_bound, name="risk_upper_bound"
    )
    validated_alpha = _validate_alpha(alpha, risk_upper_bound=validated_upper_bound)
    validated_scores, validated_risks = _validated_maps(
        scores, risks, risk_upper_bound=validated_upper_bound
    )

    sweep = _sweep_thresholds(validated_scores, validated_risks)
    bounds = (sweep.worst_risk_sums + validated_upper_bound) / (
        len(validated_scores) + 1
    )
    passing_indices = torch.nonzero(bounds <= validated_alpha, as_tuple=False)
    threshold = float("-inf")
    trusted_pixels = 0
    if passing_indices.numel() != 0:
        selected_index = int(passing_indices[-1].item())
        threshold = sweep.thresholds[selected_index].item()
        trusted_pixels = sum(
            int((score <= threshold).sum().item()) for score in validated_scores
        )
    total_pixels = sum(score.numel() for score in validated_scores)
    selected_bound = _risk_bound(
        validated_scores,
        validated_risks,
        threshold=threshold,
        risk_upper_bound=validated_upper_bound,
    )
    return ConformalCalibration(
        alpha=validated_alpha,
        threshold=threshold,
        risk_bound=selected_bound,
        calibration_size=len(validated_scores),
        trusted_pixels=trusted_pixels,
        total_pixels=total_pixels,
    )


def trusted_mask(score: torch.Tensor, calibration: ConformalCalibration) -> torch.Tensor:
    """Return the boolean mask of score values trusted by a calibration result."""
    normalized_score = _validate_score_map(score)
    if not isinstance(calibration, ConformalCalibration):
        raise ValueError("calibration must be a ConformalCalibration")
    if calibration.threshold == float("-inf"):
        return torch.zeros_like(score, dtype=torch.bool)
    if not math.isfinite(calibration.threshold):
        raise ValueError("calibration threshold must be finite or -inf")
    return normalized_score <= calibration.threshold
