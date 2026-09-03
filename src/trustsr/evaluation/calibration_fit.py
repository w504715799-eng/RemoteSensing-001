"""Pure calibration-only conformal fitting over the fixed 120 ROI map set."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from trustsr.calibration.conformal import calibrate_fidelity_mask
from trustsr.evaluation.calibration_maps import CalibrationMaps

CALIBRATION_SIZE = 120
RISK_UPPER_BOUND = 1.0
FREEZE_CALIBRATION = "freeze_calibration"
STOP_INSUFFICIENT_COVERAGE = "stop_insufficient_coverage"


def _validate_real(
    value: object, *, name: str, lower: float, include_lower: bool
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    numeric = float(value)
    if (
        not math.isfinite(numeric)
        or numeric > RISK_UPPER_BOUND
        or numeric < lower
        or (not include_lower and numeric == lower)
    ):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return numeric


def _validate_alpha(alpha: object) -> float:
    return _validate_real(alpha, name="alpha", lower=0.0, include_lower=False)


def _validate_minimum_coverage(minimum_coverage: object) -> float:
    return _validate_real(
        minimum_coverage, name="minimum_coverage", lower=0.0, include_lower=True
    )


@dataclass(frozen=True)
class CalibrationFit:
    """JSON-safe summary of one calibration-only conformal fit."""

    alpha: float
    minimum_coverage: float
    threshold: float | None
    all_abstain: bool
    risk_bound: float
    risk_upper_bound: float
    calibration_size: int
    trusted_pixels: int
    total_pixels: int
    coverage: float
    phase_decision: str
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_alpha(self.alpha)
        _validate_minimum_coverage(self.minimum_coverage)
        if type(self.risk_upper_bound) is not float or self.risk_upper_bound != RISK_UPPER_BOUND:
            raise ValueError("calibration fit risk_upper_bound must be exactly 1")
        if not isinstance(self.risk_bound, int | float) or isinstance(self.risk_bound, bool):
            raise ValueError("calibration fit risk_bound must be finite and in [0, 1]")
        if not math.isfinite(float(self.risk_bound)) or not 0 < self.risk_bound <= RISK_UPPER_BOUND:
            raise ValueError("calibration fit risk_bound must be finite and in [0, 1]")
        if self.calibration_size != CALIBRATION_SIZE:
            raise ValueError("calibration fit requires exactly 120 ROI")
        if type(self.sample_ids) is not tuple or len(self.sample_ids) != CALIBRATION_SIZE:
            raise ValueError("calibration fit requires exactly 120 ordered sample IDs")
        if (
            any(type(sample_id) is not str or not sample_id for sample_id in self.sample_ids)
            or len(set(self.sample_ids)) != CALIBRATION_SIZE
        ):
            raise ValueError("calibration fit sample IDs must be unique non-empty strings")
        for name, value, allow_zero in (
            ("trusted_pixels", self.trusted_pixels, True),
            ("total_pixels", self.total_pixels, False),
        ):
            if type(value) is not int or value < 0 or (not allow_zero and value == 0):
                raise ValueError(f"calibration fit {name} is invalid")
        if self.trusted_pixels > self.total_pixels:
            raise ValueError("calibration fit trusted_pixels exceeds total_pixels")
        if not isinstance(self.coverage, int | float) or isinstance(self.coverage, bool):
            raise ValueError("calibration fit coverage must be finite and in [0, 1]")
        if (
            not math.isfinite(float(self.coverage))
            or not 0 <= self.coverage <= RISK_UPPER_BOUND
            or self.coverage != self.trusted_pixels / self.total_pixels
        ):
            raise ValueError("calibration fit coverage is invalid")
        if type(self.all_abstain) is not bool:
            raise TypeError("calibration fit all_abstain must be bool")
        if self.threshold is None:
            if not self.all_abstain or self.trusted_pixels != 0:
                raise ValueError("null calibration threshold requires all abstain")
        elif (
            not isinstance(self.threshold, int | float)
            or isinstance(self.threshold, bool)
            or not math.isfinite(float(self.threshold))
            or not 0 <= self.threshold <= 0.25
            or self.all_abstain
        ):
            raise ValueError("calibration threshold must be finite in [0, 0.25]")
        expected_decision = (
            FREEZE_CALIBRATION
            if self.threshold is not None and self.coverage >= self.minimum_coverage
            else STOP_INSUFFICIENT_COVERAGE
        )
        if self.phase_decision != expected_decision:
            raise ValueError("calibration fit phase decision is inconsistent")

    def as_dict(self) -> dict[str, object]:
        """Return a fresh, JSON-native copy without tensors or runtime details."""

        return {
            "alpha": self.alpha,
            "minimum_coverage": self.minimum_coverage,
            "threshold": self.threshold,
            "all_abstain": self.all_abstain,
            "risk_bound": self.risk_bound,
            "risk_upper_bound": self.risk_upper_bound,
            "calibration_size": self.calibration_size,
            "trusted_pixels": self.trusted_pixels,
            "total_pixels": self.total_pixels,
            "coverage": self.coverage,
            "phase_decision": self.phase_decision,
            "sample_ids": list(self.sample_ids),
        }


def _validated_maps(maps: Sequence[CalibrationMaps]) -> tuple[CalibrationMaps, ...]:
    if isinstance(maps, str | bytes) or not isinstance(maps, Sequence):
        raise TypeError("calibration maps must be a stable sequence")
    values = tuple(maps)
    if len(values) != CALIBRATION_SIZE:
        raise ValueError("calibration fit requires exactly 120 CalibrationMaps")
    for value in values:
        if not isinstance(value, CalibrationMaps):
            raise TypeError("calibration fit requires CalibrationMaps")
        try:
            CalibrationMaps.__post_init__(value)
        except AttributeError as exc:
            raise ValueError("calibration map public contract is invalid") from exc
    sample_ids = tuple(value.sample_id for value in values)
    if any(type(sample_id) is not str or not sample_id for sample_id in sample_ids):
        raise ValueError("calibration map sample IDs must be non-empty strings")
    if len(set(sample_ids)) != CALIBRATION_SIZE:
        raise ValueError("calibration map sample IDs must be unique")
    return values


def fit_calibration_maps(
    maps: Sequence[CalibrationMaps], *, alpha: float, minimum_coverage: float
) -> CalibrationFit:
    """Fit one synthetic-verification conformal threshold from 120 calibration ROI.

    Callers must explicitly supply both scientific parameters; this function has
    no defaults and makes no recommendation about either value.
    """

    validated_alpha = _validate_alpha(alpha)
    validated_minimum_coverage = _validate_minimum_coverage(minimum_coverage)
    values = _validated_maps(maps)
    calibration = calibrate_fidelity_mask(
        tuple(value.score.tensor for value in values),
        tuple(value.risk for value in values),
        alpha=validated_alpha,
        risk_upper_bound=RISK_UPPER_BOUND,
    )
    threshold = None if calibration.threshold == float("-inf") else calibration.threshold
    all_abstain = threshold is None
    coverage = calibration.trusted_pixels / calibration.total_pixels
    decision = (
        FREEZE_CALIBRATION
        if threshold is not None and coverage >= validated_minimum_coverage
        else STOP_INSUFFICIENT_COVERAGE
    )
    return CalibrationFit(
        alpha=validated_alpha,
        minimum_coverage=validated_minimum_coverage,
        threshold=threshold,
        all_abstain=all_abstain,
        risk_bound=calibration.risk_bound,
        risk_upper_bound=RISK_UPPER_BOUND,
        calibration_size=calibration.calibration_size,
        trusted_pixels=calibration.trusted_pixels,
        total_pixels=calibration.total_pixels,
        coverage=coverage,
        phase_decision=decision,
        sample_ids=tuple(value.sample_id for value in values),
    )
