"""Deterministic CPU score proxies for development-only audits."""

from collections.abc import Sequence

import torch
from torch.nn import functional as F

from trustsr.risk.local import ensemble_variance_score


def _require_rgbn(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        raise ValueError(f"{name} must be a torch.float32 tensor")
    if value.ndim != 3 or value.shape[0] != 4 or min(value.shape[1:]) <= 0:
        raise ValueError(f"{name} must have shape (4, H, W)")
    if not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError(f"{name} must contain finite reflectance in [0, 1]")


def lr_reprojection_l1_score(
    prediction: torch.Tensor,
    lr: torch.Tensor,
    *,
    scale: int = 4,
) -> torch.Tensor:
    """Return a high-resolution map of area-projected LR L1 residuals."""
    _require_rgbn(prediction, name="prediction")
    _require_rgbn(lr, name="lr")
    if type(scale) is not int or scale != 4:
        raise ValueError("scale must equal 4")
    if prediction.shape[1:] != (lr.shape[1] * scale, lr.shape[2] * scale):
        raise ValueError("prediction spatial shape must be four times lr")
    prediction64 = prediction.detach().to(device="cpu", dtype=torch.float64)
    lr64 = lr.detach().to(device="cpu", dtype=torch.float64)
    projected = F.interpolate(
        prediction64.unsqueeze(0), size=lr.shape[1:], mode="area"
    ).squeeze(0)
    residual = (projected - lr64).abs().mean(dim=0)
    return residual.repeat_interleave(scale, 0).repeat_interleave(scale, 1).contiguous()


def three_model_disagreement_score(
    predictions: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Return per-pixel population variance averaged across RGBN bands."""
    if len(predictions) != 3:
        raise ValueError("three_model_disagreement requires exactly three predictions")
    for index, prediction in enumerate(predictions):
        _require_rgbn(prediction, name=f"predictions[{index}]")
    if len({tuple(prediction.shape) for prediction in predictions}) != 1:
        raise ValueError("all model predictions must have matching shapes")
    stacked = torch.stack(
        [item.detach().to(device="cpu") for item in predictions], dim=0
    )
    return ensemble_variance_score(stacked).contiguous()

