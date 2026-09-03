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
    roi_scores: tuple[torch.Tensor, ...]
    roi_maxima: tuple[torch.Tensor, ...]
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


def _roi_score_maxima(
    score: torch.Tensor, risk: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return score groups and their cumulative maximum risk for one ROI."""
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
    return sorted_score[group_starts], torch.cummax(group_maxima, dim=0).values


def _sweep_thresholds(
    scores: Sequence[torch.Tensor], risks: Sequence[torch.Tensor]
) -> _ThresholdSweep:
    """Prepare sorted threshold candidates and cumulative ROI-risk curves."""
    total_pixels = sum(score.numel() for score in scores)
    event_scores = torch.empty(total_pixels, dtype=torch.float64)
    roi_scores: list[torch.Tensor] = []
    roi_maxima: list[torch.Tensor] = []
    event_offset = 0
    for score, risk in zip(scores, risks, strict=True):
        score_groups, cumulative_maxima = _roi_score_maxima(score, risk)
        roi_scores.append(score_groups)
        roi_maxima.append(cumulative_maxima)
        next_offset = event_offset + score_groups.numel()
        event_scores[event_offset:next_offset] = score_groups
        event_offset = next_offset

    event_scores = event_scores[:event_offset]
    sorted_scores, order = torch.sort(event_scores)
    del event_scores, order
    return _ThresholdSweep(
        thresholds=torch.unique_consecutive(sorted_scores),
        roi_scores=tuple(roi_scores),
        roi_maxima=tuple(roi_maxima),
        event_count=event_offset,
    )


def _sweep_risk_bound(
    sweep: _ThresholdSweep, *, threshold: float, risk_upper_bound: float
) -> float:
    """Compute the original ROI-ordered finite-sample bound at one threshold."""
    threshold_tensor = torch.tensor(threshold, dtype=torch.float64)
    worst_risks = []
    for roi_scores, roi_maxima in zip(
        sweep.roi_scores, sweep.roi_maxima, strict=True
    ):
        score_index = int(
            torch.searchsorted(roi_scores, threshold_tensor, right=True).item()
        ) - 1
        worst_risks.append(0.0 if score_index < 0 else roi_maxima[score_index].item())
    return (sum(worst_risks) + risk_upper_bound) / (len(sweep.roi_scores) + 1)


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
    threshold = float("-inf")
    trusted_pixels = 0
    lower_index = 0
    upper_index = sweep.thresholds.numel() - 1
    selected_index = -1
    while lower_index <= upper_index:
        candidate_index = (lower_index + upper_index) // 2
        candidate_threshold = sweep.thresholds[candidate_index].item()
        candidate_bound = _sweep_risk_bound(
            sweep,
            threshold=candidate_threshold,
            risk_upper_bound=validated_upper_bound,
        )
        if candidate_bound <= validated_alpha:
            selected_index = candidate_index
            lower_index = candidate_index + 1
        else:
            upper_index = candidate_index - 1
    if selected_index >= 0:
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
