"""Reference-free CPU scoring primitives; fitting and calibration are separate steps."""

import math
from numbers import Real

import torch
from torch.nn import functional as F


def _map(value):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.ndim != 2 or not value.numel() or not torch.isfinite(value).all()
            or (value < 0).any()):
        raise ValueError('score map must be finite nonnegative floating 2D tensor')
    return value.detach().to(device='cpu', dtype=torch.float64)


def _scalar(value, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            or (value <= 0 if positive else not 0 <= value <= 1)):
        raise ValueError('positive finite scale or weight in [0,1] required')
    return float(value)


def _pool(value):
    return F.avg_pool2d(F.pad(value, (1,) * 4, mode='reflect'), 3, stride=1)


def _gaussian(value):
    positions = torch.arange(-3, 4, dtype=torch.float64)
    weights = torch.exp(-positions.square() / 2)
    weights /= weights.sum()
    kernel = torch.outer(weights, weights)[None, None]
    return F.conv2d(F.pad(value[None, None], (3,) * 4, mode='reflect'), kernel)[0, 0]


def spatial_components(samples):
    """Return separately smoothed seed-variance and ensemble-mean texture terms.

    Their sum equals the joint population neighborhood variance (window 3).
    This decomposition is an algebraic identity, not a new uncertainty theorem.
    """
    if (not isinstance(samples, torch.Tensor) or not samples.is_floating_point()
            or samples.ndim != 4 or samples.shape[:2] != (5, 4)
            or min(samples.shape[2:]) < 4 or not torch.isfinite(samples).all()
            or (samples < 0).any() or (samples > 1).any()):
        raise ValueError('samples must be finite floating (5,4,H,W) in [0,1], H/W >=4')
    values = samples.detach().to(device='cpu', dtype=torch.float64)
    mean = values.mean(0)
    seed_variance = values.var(0, correction=0).mean(0)
    uncertainty = _pool(seed_variance[None, None])[0, 0]
    moments = _pool(torch.stack((mean, mean.square())))
    texture = moments[1] - moments[0].square()
    if (texture < -1e-12).any():
        raise ValueError('negative spatial variance exceeds roundoff tolerance')
    return _gaussian(uncertainty).contiguous(), _gaussian(texture.clamp_min(0).mean(0)).contiguous()


def fuse_scores(lr, uncertainty, *, lr_scale, uncertainty_scale, lr_weight):
    """Apply fixed positive scales then convex fusion; no per-image fitting or HR."""
    a, b = _scalar(lr_scale, positive=True), _scalar(uncertainty_scale, positive=True)
    weight = _scalar(lr_weight)
    left, right = _map(lr), _map(uncertainty)
    if left.shape != right.shape:
        raise ValueError('fusion requires identical score grids')
    # Divide both x and a by their maximum before addition, including subnormals.
    left_scale, right_scale = torch.full_like(left, a), torch.full_like(right, b)
    left_max = torch.maximum(left, left_scale)
    right_max = torch.maximum(right, right_scale)
    left_scaled = (left / left_max) / (left / left_max + left_scale / left_max)
    right_scaled = (right / right_max) / (right / right_max + right_scale / right_max)
    return (weight * left_scaled + (1 - weight) * right_scaled).contiguous()
