"""Empirical selective-risk evaluation for calibrated score thresholds."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SelectivePoint:
    """Pixel coverage and mean ROI-maximum risk at one score threshold."""

    threshold: float
    coverage: float
    roi_max_risk: float


def _is_runtime_real(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _validate_threshold(threshold: float) -> float:
    if not _is_runtime_real(threshold):
        raise ValueError("threshold must be finite or -inf")
    numeric_threshold = float(threshold)
    if numeric_threshold != float("-inf") and not math.isfinite(numeric_threshold):
        raise ValueError("threshold must be finite or -inf")
    return numeric_threshold


def _validate_score_map(score: torch.Tensor) -> None:
    if not isinstance(score, torch.Tensor):
        raise ValueError("score maps must be torch.Tensors")
    if score.ndim != 2 or 0 in score.shape:
        raise ValueError("score maps must be non-empty two-dimensional tensors")
    if score.is_complex():
        raise ValueError("score maps must be real-valued")
    if not torch.isfinite(score).all():
        raise ValueError("score maps must contain only finite values")
    if (score < 0).any():
        raise ValueError("scores must be non-negative")


def _validate_risk_map(risk: torch.Tensor) -> None:
    if not isinstance(risk, torch.Tensor):
        raise ValueError("risk maps must be torch.Tensors")
    if risk.ndim != 2 or 0 in risk.shape:
        raise ValueError("risk maps must be non-empty two-dimensional tensors")
    if risk.is_complex():
        raise ValueError("risk maps must be real-valued")
    if not torch.isfinite(risk).all():
        raise ValueError("risk maps must contain only finite values")
    if (risk < 0).any() or (risk > 1).any():
        raise ValueError("risks must be in [0, 1]")


def _validated_maps(
    scores: Sequence[torch.Tensor], risks: Sequence[torch.Tensor]
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
        _validate_risk_map(risk)
        if score.shape != risk.shape:
            raise ValueError("ROI shapes must match")
        validated_scores.append(score.to(device="cpu", dtype=torch.float64))
        validated_risks.append(risk.to(device="cpu", dtype=torch.float64))
    return tuple(validated_scores), tuple(validated_risks)


def evaluate_selective_point(
    scores: Sequence[torch.Tensor],
    risks: Sequence[torch.Tensor],
    *,
    threshold: float,
) -> SelectivePoint:
    """Evaluate trusted pixel coverage and empirical ROI-level selective risk."""
    validated_threshold = _validate_threshold(threshold)
    validated_scores, validated_risks = _validated_maps(scores, risks)

    trusted_pixels = 0
    total_pixels = 0
    roi_max_risks: list[float] = []
    for score, risk in zip(validated_scores, validated_risks, strict=True):
        mask = score <= validated_threshold
        trusted_pixels += int(mask.sum().item())
        total_pixels += score.numel()
        selected_risks = risk[mask]
        roi_max_risks.append(
            0.0 if selected_risks.numel() == 0 else float(selected_risks.max().item())
        )

    return SelectivePoint(
        threshold=validated_threshold,
        coverage=float(trusted_pixels / total_pixels),
        roi_max_risk=float(sum(roi_max_risks) / len(roi_max_risks)),
    )
