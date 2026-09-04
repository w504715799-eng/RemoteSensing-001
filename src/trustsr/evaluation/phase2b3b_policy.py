"""Frozen scientific operating point for formal Phase 2B3-B evidence."""

from __future__ import annotations

from typing import Final

APPROVED_ALPHA: Final[float] = 0.05
APPROVED_MINIMUM_COVERAGE: Final[float] = 0.10


def require_approved_operating_point(
    alpha: object, minimum_coverage: object
) -> tuple[float, float]:
    """Return the sole preregistered target or reject the evidence."""

    if type(alpha) is not float or alpha != APPROVED_ALPHA:
        raise ValueError("alpha must equal the approved Phase 2B3-B value 0.05")
    if (
        type(minimum_coverage) is not float
        or minimum_coverage != APPROVED_MINIMUM_COVERAGE
    ):
        raise ValueError(
            "minimum coverage must equal the approved Phase 2B3-B value 0.10"
        )
    return alpha, minimum_coverage
