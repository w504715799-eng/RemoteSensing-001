"""Deterministic repeatability gate for super-resolution model predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from trustsr.artifacts.predictions import tensor_sha256
from trustsr.models.protocols import SRModel


@dataclass(frozen=True)
class RepeatabilitySummary:
    first_sha256: str
    second_sha256: str
    bitwise_equal: bool
    max_abs_diff: float
    tolerance: float

    def as_dict(self) -> dict[str, str | bool | float]:
        return {
            "first_sha256": self.first_sha256,
            "second_sha256": self.second_sha256,
            "bitwise_equal": self.bitwise_equal,
            "max_abs_diff": self.max_abs_diff,
            "tolerance": self.tolerance,
        }


class RepeatabilityError(RuntimeError):
    """Raised when repeated model predictions violate the output contract."""


def _validate_tolerance(tolerance: float) -> float:
    try:
        valid = math.isfinite(tolerance) and tolerance >= 0
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise RepeatabilityError("tolerance must be finite and nonnegative")
    return float(tolerance)


def _validate_output(model: SRModel, lr: torch.Tensor, output: object) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise RepeatabilityError("model output must be a torch.Tensor")
    if output.dtype != torch.float32:
        raise RepeatabilityError("model output must be torch.float32")
    if output.device.type != "cpu" or not output.is_contiguous() or output.requires_grad:
        raise RepeatabilityError("model output must be detached, contiguous, and on CPU")
    if not isinstance(lr, torch.Tensor) or lr.ndim != 3:
        raise RepeatabilityError("lr must be a three-dimensional tensor")
    expected_shape = (4, lr.shape[1] * model.scale, lr.shape[2] * model.scale)
    if tuple(output.shape) != expected_shape:
        raise RepeatabilityError(f"model output must have shape {expected_shape}")
    if not torch.isfinite(output).all() or (output < 0).any() or (output > 1).any():
        raise RepeatabilityError("model output must be finite and in [0, 1]")
    return output


def run_repeatability(
    model: SRModel,
    lr: torch.Tensor,
    *,
    tolerance: float = 1e-6,
) -> tuple[torch.Tensor, RepeatabilitySummary]:
    """Run exactly two uncached predictions and compare them."""
    tolerance_value = _validate_tolerance(tolerance)
    first_raw = model.predict(lr)
    second_raw = model.predict(lr)
    first = _validate_output(model, lr, first_raw)
    second = _validate_output(model, lr, second_raw)
    if first.shape != second.shape or first.dtype != second.dtype:
        raise RepeatabilityError("model outputs must share shape and dtype")
    bitwise_equal = torch.equal(first, second)
    max_abs_diff = float((first - second).abs().max().item())
    summary = RepeatabilitySummary(
        first_sha256=tensor_sha256(first),
        second_sha256=tensor_sha256(second),
        bitwise_equal=bitwise_equal,
        max_abs_diff=max_abs_diff,
        tolerance=tolerance_value,
    )
    if not bitwise_equal and max_abs_diff > tolerance_value:
        raise RepeatabilityError("model outputs differ above tolerance")
    return first, summary
