"""Deterministic CPU-only synthetic conformal smoke workflow."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from typing import Any

import torch

from trustsr.calibration.conformal import calibrate_fidelity_mask
from trustsr.evaluation.selective import evaluate_selective_point
from trustsr.risk.local import ensemble_variance_score, local_l1_risk


def _positive_finite_alpha(value: str) -> float:
    alpha = float(value)
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise argparse.ArgumentTypeError("alpha must be a finite number in (0, 1]")
    return alpha


def _positive_odd_window(value: str) -> int:
    window = int(value)
    if window <= 0 or window % 2 == 0:
        raise argparse.ArgumentTypeError("window must be a positive odd integer")
    return window


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the synthetic smoke workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=_positive_finite_alpha, default=0.27)
    parser.add_argument("--window", type=_positive_odd_window, default=1)
    return parser


def _synthetic_rois() -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    base = torch.linspace(0.1, 0.7, 64, dtype=torch.float64).reshape(4, 4, 4)
    pattern = torch.linspace(0.0, 1.0, 16, dtype=torch.float64).reshape(1, 4, 4)
    pattern = pattern.expand(4, -1, -1)

    hrs: list[torch.Tensor] = []
    sample_sets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    for index in range(5):
        hr = torch.roll(base, shifts=(index % 3, (2 * index) % 3), dims=(-2, -1))
        hr = hr + 0.01 * index
        spread = 0.005 * (index + 1) * (1.0 + pattern)
        center = hr + 2.0 * spread
        samples = torch.stack((center - spread, center + spread))
        hrs.append(hr)
        sample_sets.append(samples)
        predictions.append(samples.mean(dim=0))
    return hrs, sample_sets, predictions


def run(*, alpha: float = 0.27, window: int = 1) -> dict[str, Any]:
    """Run the fixed synthetic ROI experiment and return JSON-native evidence."""
    hrs, sample_sets, predictions = _synthetic_rois()
    risks = [
        local_l1_risk(prediction, hr, window=window)
        for prediction, hr in zip(predictions, hrs, strict=True)
    ]
    scores = [ensemble_variance_score(samples) for samples in sample_sets]

    calibration = calibrate_fidelity_mask(scores[:3], risks[:3], alpha=alpha)
    calibration_point = evaluate_selective_point(
        scores[:3], risks[:3], threshold=calibration.threshold
    )
    test_point = evaluate_selective_point(scores[3:], risks[3:], threshold=calibration.threshold)

    return {
        "schema": "trustsr.conformal-smoke.v1",
        "synthetic_smoke": True,
        "config": {"alpha": alpha, "channels": 4, "scale": 4, "window": window},
        "calibration": {
            "calibration_size": calibration.calibration_size,
            "coverage": calibration_point.coverage,
            "risk_bound": calibration.risk_bound,
            "roi_max_risk": calibration_point.roi_max_risk,
            "threshold": calibration.threshold,
        },
        "test": {
            "coverage": test_point.coverage,
            "roi_count": 2,
            "roi_max_risk": test_point.roi_max_risk,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print the canonical synthetic smoke payload."""
    args = build_parser().parse_args(argv)
    payload = run(alpha=args.alpha, window=args.window)
    output = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
