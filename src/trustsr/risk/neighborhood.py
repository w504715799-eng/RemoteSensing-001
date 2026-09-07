"""Paper-formula RGBN neighborhood variance adaptation."""

import torch
from torch.nn import functional as F


def neighborhood_variance_score(samples: torch.Tensor, *, window: int) -> torch.Tensor:
    """Joint seed/neighborhood population variance, then fixed Gaussian smoothing.

    Separate from the frozen K5 score. No reference image enters this calculation.
    """
    if type(window) is not int or window not in (3, 9):
        raise ValueError("window must be a registered candidate: 3 or 9")
    if not isinstance(samples, torch.Tensor) or not samples.is_floating_point():
        raise ValueError("samples must be a floating tensor")
    if samples.ndim != 4 or samples.shape[:2] != (5, 4):
        raise ValueError("samples must have shape (5, 4, H, W)")
    if min(samples.shape[2:]) < max(window, 4):
        raise ValueError("spatial dimensions too small for reflected kernels")
    values = samples.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("samples must contain finite reflectance in [0, 1]")
    radius = window // 2
    # Linearity permits averaging over seeds before the spatial mean.
    moments = torch.stack((values.mean(0), values.square().mean(0)))
    pooled = F.avg_pool2d(F.pad(moments, (radius,) * 4, mode="reflect"), window, stride=1)
    variance = pooled[1] - pooled[0].square()
    if (variance < -1e-12).any():
        raise ValueError("negative variance exceeds rounding tolerance")
    score = variance.clamp_min(0).mean(0)[None, None]
    positions = torch.arange(-3, 4, dtype=torch.float64)
    weights = torch.exp(-positions.square() / 2)
    weights /= weights.sum()
    kernel = torch.outer(weights, weights)[None, None]
    return F.conv2d(F.pad(score, (3,) * 4, mode="reflect"), kernel)[0, 0].contiguous()
