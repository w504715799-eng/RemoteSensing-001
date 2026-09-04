"""Frozen scientific policy for the one-time Phase 2B3-C evaluation."""

from __future__ import annotations

from typing import Final

PHASE2B3C_ALPHA: Final[float] = 0.05
PHASE2B3C_MINIMUM_COVERAGE: Final[float] = 0.10
PHASE2B3C_DELTA: Final[float] = 0.05
PHASE2B3C_GRID_SIZE: Final[int] = 20
PHASE2B3C_THRESHOLD: Final[float] = 7.970395366024563e-06
PHASE2B3C_EVALUATION_SIZE: Final[int] = 120
PHASE2B3C_BISECTION_ITERATIONS: Final[int] = 128
PHASE2B3C_RISK_UPPER_BOUND: Final[float] = 1.0


def require_phase2b3c_policy(
    *,
    alpha: object,
    minimum_coverage: object,
    delta: object,
    grid_size: object,
    threshold: object,
) -> tuple[float, float, float, int, float]:
    """Return the sole approved Phase 2B3-C policy or fail closed."""

    values = (
        ("alpha", alpha, float, PHASE2B3C_ALPHA),
        ("minimum coverage", minimum_coverage, float, PHASE2B3C_MINIMUM_COVERAGE),
        ("confidence error", delta, float, PHASE2B3C_DELTA),
        ("grid size", grid_size, int, PHASE2B3C_GRID_SIZE),
        ("threshold", threshold, float, PHASE2B3C_THRESHOLD),
    )
    for name, value, expected_type, expected in values:
        if type(value) is not expected_type or value != expected:
            raise ValueError(
                f"{name} must equal the approved Phase 2B3-C value {expected!r}"
            )
    return (
        PHASE2B3C_ALPHA,
        PHASE2B3C_MINIMUM_COVERAGE,
        PHASE2B3C_DELTA,
        PHASE2B3C_GRID_SIZE,
        PHASE2B3C_THRESHOLD,
    )
