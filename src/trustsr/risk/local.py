"""Local fidelity and ensemble uncertainty scores for four-band reflectance."""

import torch
from torch.nn import functional as F


def _validate_reflectance(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.ndim != 3 or value.shape[0] != 4:
        raise ValueError(f"{name} must have shape (4, H, W)")
    if value.shape[1] == 0 or value.shape[2] == 0:
        raise ValueError(f"{name} must have non-empty spatial dimensions")
    if value.is_complex():
        raise ValueError(f"{name} must be real-valued")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if (value < 0).any() or (value > 1).any():
        raise ValueError(f"{name} values must be in [0, 1]")


def _validate_window(window: int, *, height: int, width: int) -> None:
    if isinstance(window, bool) or not isinstance(window, int):
        raise ValueError("window must be a positive odd integer")
    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if window > min(height, width):
        raise ValueError("window must not exceed the spatial dimensions")


def local_l1_risk(sr: torch.Tensor, hr: torch.Tensor, *, window: int) -> torch.Tensor:
    """Compute a reflected-window local mean absolute error."""
    _validate_reflectance(sr, name="sr")
    _validate_reflectance(hr, name="hr")
    if sr.shape != hr.shape:
        raise ValueError("sr and hr must have matching shapes")
    height, width = sr.shape[1:]
    _validate_window(window, height=height, width=width)

    pixel_error = (sr.to(torch.float64) - hr.to(torch.float64)).abs().mean(dim=0)
    if window == 1:
        return pixel_error
    radius = window // 2
    padded = F.pad(pixel_error.unsqueeze(0).unsqueeze(0), (radius,) * 4, mode="reflect")
    return F.avg_pool2d(padded, kernel_size=window, stride=1)[0, 0]


def ensemble_variance_score(samples: torch.Tensor) -> torch.Tensor:
    """Compute mean per-band population variance over ensemble members."""
    if not isinstance(samples, torch.Tensor):
        raise ValueError("samples must be a torch.Tensor")
    if samples.ndim != 4 or samples.shape[1] != 4:
        raise ValueError("samples must have shape (K, 4, H, W)")
    if samples.shape[0] < 2 or samples.shape[2] == 0 or samples.shape[3] == 0:
        raise ValueError("samples must contain at least two non-empty samples")
    if samples.is_complex():
        raise ValueError("samples must be real-valued")
    if not torch.isfinite(samples).all():
        raise ValueError("samples must contain only finite values")
    if (samples < 0).any() or (samples > 1).any():
        raise ValueError("samples values must be in [0, 1]")
    return samples.to(torch.float64).var(dim=0, correction=0).mean(dim=0)
