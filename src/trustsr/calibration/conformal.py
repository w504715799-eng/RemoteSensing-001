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
        _validate_count(self.total_pixels, name="total_pixels", allow_zero=True)
        if self.trusted_pixels > self.total_pixels:
            raise ValueError("trusted_pixels must not exceed total_pixels")
        if self.threshold == float("-inf") and self.trusted_pixels != 0:
            raise ValueError("-inf threshold must have zero trusted_pixels")


def _validate_positive_finite(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite number") from error
    if not math.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return numeric_value


def _validate_alpha(alpha: float, *, risk_upper_bound: float) -> float:
    if isinstance(alpha, bool):
        raise ValueError("alpha must be in (0, risk_upper_bound]")
    try:
        numeric_alpha = float(alpha)
    except (TypeError, ValueError) as error:
        raise ValueError("alpha must be in (0, risk_upper_bound]") from error
    if not math.isfinite(numeric_alpha) or not 0 < numeric_alpha <= risk_upper_bound:
        raise ValueError("alpha must be in (0, risk_upper_bound]")
    return numeric_alpha


def _validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool):
        raise ValueError("threshold must be finite or -inf")
    try:
        numeric_threshold = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("threshold must be finite or -inf") from error
    if numeric_threshold == float("-inf"):
        return numeric_threshold
    if not math.isfinite(numeric_threshold) or numeric_threshold < 0:
        raise ValueError("threshold must be finite or -inf")
    return numeric_threshold


def _validate_count(value: int, *, name: str, allow_zero: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")


def _validate_score_map(score: torch.Tensor) -> None:
    if not isinstance(score, torch.Tensor):
        raise ValueError("score maps must be torch.Tensors")
    if score.ndim != 2:
        raise ValueError("score maps must be two-dimensional")
    if score.is_complex():
        raise ValueError("score maps must be real-valued")
    if not torch.isfinite(score).all():
        raise ValueError("score maps must contain only finite values")
    if (score < 0).any():
        raise ValueError("scores must be non-negative")


def _validate_risk_map(risk: torch.Tensor, *, risk_upper_bound: float) -> None:
    if not isinstance(risk, torch.Tensor):
        raise ValueError("risk maps must be torch.Tensors")
    if risk.ndim != 2:
        raise ValueError("risk maps must be two-dimensional")
    if risk.is_complex():
        raise ValueError("risk maps must be real-valued")
    if not torch.isfinite(risk).all():
        raise ValueError("risk maps must contain only finite values")
    if (risk < 0).any():
        raise ValueError("risks must be non-negative")
    if (risk > risk_upper_bound).any():
        raise ValueError("risk exceeds risk_upper_bound")


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
        _validate_score_map(score)
        _validate_risk_map(risk, risk_upper_bound=risk_upper_bound)
        if score.shape != risk.shape:
            raise ValueError("ROI shapes must match")
        validated_scores.append(score.to(device="cpu", dtype=torch.float64))
        validated_risks.append(risk.to(device="cpu", dtype=torch.float64))
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

    candidates = torch.unique(torch.cat([score.reshape(-1) for score in validated_scores]))
    candidates = torch.sort(candidates).values
    threshold = float("-inf")
    for candidate in candidates:
        candidate_threshold = candidate.item()
        candidate_bound = _risk_bound(
            validated_scores,
            validated_risks,
            threshold=candidate_threshold,
            risk_upper_bound=validated_upper_bound,
        )
        if candidate_bound <= validated_alpha:
            threshold = candidate_threshold

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
    _validate_score_map(score)
    if not isinstance(calibration, ConformalCalibration):
        raise ValueError("calibration must be a ConformalCalibration")
    if calibration.threshold == float("-inf"):
        return torch.zeros_like(score, dtype=torch.bool)
    if not math.isfinite(calibration.threshold):
        raise ValueError("calibration threshold must be finite or -inf")
    return score <= calibration.threshold
