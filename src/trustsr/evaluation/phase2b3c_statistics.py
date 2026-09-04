"""Pure preregistered statistics for the one-time Phase 2B3-C evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from trustsr.evaluation.phase2b3c_policy import (
    PHASE2B3C_ALPHA,
    PHASE2B3C_BISECTION_ITERATIONS,
    PHASE2B3C_DELTA,
    PHASE2B3C_EVALUATION_SIZE,
    PHASE2B3C_GRID_SIZE,
    PHASE2B3C_MINIMUM_COVERAGE,
    PHASE2B3C_RISK_UPPER_BOUND,
    PHASE2B3C_THRESHOLD,
    require_phase2b3c_policy,
)

CONFIRMED = "confirmed"
EMPIRICALLY_MET_BUT_INCONCLUSIVE = "empirically_met_but_inconclusive"
FAILED = "failed"
INSUFFICIENT_COVERAGE = "insufficient_coverage"
EMPIRICAL_RISK_EXCEEDS_TARGET = "empirical_risk_exceeds_target"
FINITE_SAMPLE_UPPER_BOUND_EXCEEDS_TARGET = "finite_sample_upper_bound_exceeds_target"
METHOD_NAME = "one_sided_diversified_grid_kelly"
_DAYS_BETWEEN = (-1, 0, 1)
_CORRELATION_BINS = (0, 1, 2, 3)
_ROIS_PER_STRATUM = 10


def _finite_unit_interval(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite built-in float in [0, 1]")
    return value


@dataclass(frozen=True)
class ROIEvaluation:
    """One ROI-level loss and coverage observation."""

    sample_id: str
    days_between: int
    correlation_bin: int
    selection_round: int
    trusted_pixels: int
    total_pixels: int
    coverage: float
    loss: float

    def __post_init__(self) -> None:
        if type(self.sample_id) is not str or not self.sample_id:
            raise ValueError("ROI sample_id must be a non-empty string")
        if type(self.days_between) is not int or self.days_between not in _DAYS_BETWEEN:
            raise ValueError("ROI days_between is invalid")
        if type(self.correlation_bin) is not int or self.correlation_bin not in _CORRELATION_BINS:
            raise ValueError("ROI correlation_bin is invalid")
        if (
            type(self.selection_round) is not int
            or not 1 <= self.selection_round <= _ROIS_PER_STRATUM
        ):
            raise ValueError("ROI selection_round is invalid")
        if type(self.trusted_pixels) is not int or self.trusted_pixels < 0:
            raise ValueError("ROI trusted_pixels is invalid")
        if type(self.total_pixels) is not int or self.total_pixels <= 0:
            raise ValueError("ROI total_pixels is invalid")
        if self.trusted_pixels > self.total_pixels:
            raise ValueError("ROI trusted_pixels exceeds total_pixels")
        _finite_unit_interval(self.coverage, "ROI coverage")
        _finite_unit_interval(self.loss, "ROI loss")
        if self.coverage != self.trusted_pixels / self.total_pixels:
            raise ValueError("ROI coverage does not match pixel counts")

    def as_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "days_between": self.days_between,
            "correlation_bin": self.correlation_bin,
            "selection_round": self.selection_round,
            "trusted_pixels": self.trusted_pixels,
            "total_pixels": self.total_pixels,
            "coverage": self.coverage,
            "loss": self.loss,
        }


@dataclass(frozen=True)
class StratumDiagnostic:
    """Fixed descriptive aggregate for one evaluation stratum."""

    days_between: int
    correlation_bin: int
    roi_count: int
    trusted_pixels: int
    total_pixels: int
    coverage: float
    mean_loss: float
    maximum_loss: float

    def as_dict(self) -> dict[str, object]:
        return {
            "days_between": self.days_between,
            "correlation_bin": self.correlation_bin,
            "roi_count": self.roi_count,
            "trusted_pixels": self.trusted_pixels,
            "total_pixels": self.total_pixels,
            "coverage": self.coverage,
            "mean_loss": self.mean_loss,
            "maximum_loss": self.maximum_loss,
        }


@dataclass(frozen=True)
class Phase2B3CStatistics:
    """JSON-safe aggregate of the sole preregistered evaluation."""

    evaluation_size: int
    trusted_pixels: int
    total_pixels: int
    coverage: float
    mean_loss: float
    empirical_risk_delta: float
    empirical_risk_violation: float
    risk_ucb: float
    finite_sample_margin: float
    finite_sample_delta: float
    hoeffding_ucb: float
    phase_decision: str
    reasons: tuple[str, ...]
    strata: tuple[StratumDiagnostic, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "target": {
                "alpha": PHASE2B3C_ALPHA,
                "minimum_coverage": PHASE2B3C_MINIMUM_COVERAGE,
                "risk_upper_bound": PHASE2B3C_RISK_UPPER_BOUND,
            },
            "method": {
                "name": METHOD_NAME,
                "confidence_error": PHASE2B3C_DELTA,
                "grid_size": PHASE2B3C_GRID_SIZE,
                "bet_fractions": [
                    index / (PHASE2B3C_GRID_SIZE + 1)
                    for index in range(1, PHASE2B3C_GRID_SIZE + 1)
                ],
                "bisection_iterations": PHASE2B3C_BISECTION_ITERATIONS,
            },
            "evaluation_size": self.evaluation_size,
            "trusted_pixels": self.trusted_pixels,
            "total_pixels": self.total_pixels,
            "coverage": self.coverage,
            "mean_loss": self.mean_loss,
            "empirical_risk_delta": self.empirical_risk_delta,
            "empirical_risk_violation": self.empirical_risk_violation,
            "risk_ucb": self.risk_ucb,
            "finite_sample_margin": self.finite_sample_margin,
            "finite_sample_delta": self.finite_sample_delta,
            "hoeffding_ucb": self.hoeffding_ucb,
            "phase_decision": self.phase_decision,
            "reasons": list(self.reasons),
            "strata": [value.as_dict() for value in self.strata],
        }


def _validate_map(tensor: object, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch tensor")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if tensor.dtype != torch.float64:
        raise ValueError(f"{name} must use float64")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


def evaluate_roi_loss(
    *,
    sample_id: str,
    days_between: int,
    correlation_bin: int,
    selection_round: int,
    score: object,
    risk: object,
    threshold: object,
) -> ROIEvaluation:
    """Evaluate the frozen trusted mask into one bounded ROI observation."""

    require_phase2b3c_policy(
        alpha=PHASE2B3C_ALPHA,
        minimum_coverage=PHASE2B3C_MINIMUM_COVERAGE,
        delta=PHASE2B3C_DELTA,
        grid_size=PHASE2B3C_GRID_SIZE,
        threshold=threshold,
    )
    score_tensor = _validate_map(score, "score")
    risk_tensor = _validate_map(risk, "risk")
    if score_tensor.shape != risk_tensor.shape:
        raise ValueError("score and risk shape must match")
    if bool(((score_tensor < 0.0) | (score_tensor > 0.25)).any()):
        raise ValueError("score must be in [0, 0.25]")
    if bool(((risk_tensor < 0.0) | (risk_tensor > 1.0)).any()):
        raise ValueError("risk must be in [0, 1]")
    trusted = score_tensor <= PHASE2B3C_THRESHOLD
    trusted_pixels = int(torch.count_nonzero(trusted).item())
    total_pixels = score_tensor.numel()
    loss = float(torch.max(risk_tensor[trusted]).item()) if trusted_pixels else 0.0
    return ROIEvaluation(
        sample_id=sample_id,
        days_between=days_between,
        correlation_bin=correlation_bin,
        selection_round=selection_round,
        trusted_pixels=trusted_pixels,
        total_pixels=total_pixels,
        coverage=trusted_pixels / total_pixels,
        loss=loss,
    )


def _validated_losses(losses: object) -> tuple[float, ...]:
    if isinstance(losses, str | bytes) or not isinstance(losses, Sequence):
        raise TypeError("losses must be a stable sequence")
    values = tuple(losses)
    if len(values) != PHASE2B3C_EVALUATION_SIZE:
        raise ValueError("grid-Kelly evaluation requires exactly 120 losses")
    return tuple(_finite_unit_interval(value, "loss") for value in values)


def grid_kelly_log_evalue(losses: object, candidate_mean: object) -> float:
    """Return the preregistered one-sided diversified grid-Kelly log e-value."""

    values = _validated_losses(losses)
    mean = _finite_unit_interval(candidate_mean, "candidate_mean")
    if mean == 1.0:
        if any(value < 1.0 for value in values):
            return math.inf
        component_logs = [
            PHASE2B3C_EVALUATION_SIZE
            * math.log1p(-index / (PHASE2B3C_GRID_SIZE + 1))
            for index in range(1, PHASE2B3C_GRID_SIZE + 1)
        ]
    else:
        component_logs = []
        denominator = 1.0 - mean
        for index in range(1, PHASE2B3C_GRID_SIZE + 1):
            rho = index / (PHASE2B3C_GRID_SIZE + 1)
            terms = (
                math.log1p(-rho * (value - mean) / denominator) for value in values
            )
            component_logs.append(math.fsum(terms))
    maximum = max(component_logs)
    mixture = math.fsum(math.exp(value - maximum) for value in component_logs)
    result = maximum + math.log(mixture / PHASE2B3C_GRID_SIZE)
    if not math.isfinite(result):
        raise ValueError("grid-Kelly log e-value must be finite")
    return result


def grid_kelly_upper_bound(losses: object) -> float:
    """Invert the fixed grid-Kelly e-value into a conservative upper bound."""

    values = _validated_losses(losses)
    rejection_log = math.log(1.0 / PHASE2B3C_DELTA)
    if grid_kelly_log_evalue(values, 1.0) < rejection_log:
        return 1.0
    accepted = 0.0
    rejected = 1.0
    for _ in range(PHASE2B3C_BISECTION_ITERATIONS):
        midpoint = (accepted + rejected) / 2.0
        if grid_kelly_log_evalue(values, midpoint) < rejection_log:
            accepted = midpoint
        else:
            rejected = midpoint
    return rejected


def _validated_rois(rois: object) -> tuple[ROIEvaluation, ...]:
    if isinstance(rois, str | bytes) or not isinstance(rois, Sequence):
        raise TypeError("ROI evaluations must be a stable sequence")
    values = tuple(rois)
    if len(values) != PHASE2B3C_EVALUATION_SIZE:
        raise ValueError("Phase 2B3-C statistics require exactly 120 ROI evaluations")
    if any(not isinstance(value, ROIEvaluation) for value in values):
        raise TypeError("Phase 2B3-C statistics require ROIEvaluation values")
    for value in values:
        ROIEvaluation.__post_init__(value)
    sample_ids = tuple(value.sample_id for value in values)
    if len(set(sample_ids)) != PHASE2B3C_EVALUATION_SIZE:
        raise ValueError("Phase 2B3-C ROI sample IDs must be unique")
    expected_positions = {
        (days_between, correlation_bin, selection_round)
        for days_between in _DAYS_BETWEEN
        for correlation_bin in _CORRELATION_BINS
        for selection_round in range(1, _ROIS_PER_STRATUM + 1)
    }
    actual_positions = tuple(
        (value.days_between, value.correlation_bin, value.selection_round)
        for value in values
    )
    if len(set(actual_positions)) != len(actual_positions) or set(
        actual_positions
    ) != expected_positions:
        raise ValueError("Phase 2B3-C ROI evaluations must cover all balanced design positions")
    return values


def _stratum_diagnostics(values: tuple[ROIEvaluation, ...]) -> tuple[StratumDiagnostic, ...]:
    diagnostics: list[StratumDiagnostic] = []
    for days_between in _DAYS_BETWEEN:
        for correlation_bin in _CORRELATION_BINS:
            members = tuple(
                value
                for value in values
                if value.days_between == days_between
                and value.correlation_bin == correlation_bin
            )
            if len(members) != _ROIS_PER_STRATUM:
                raise ValueError("Phase 2B3-C requires exactly 10 ROI per stratum")
            trusted_pixels = sum(value.trusted_pixels for value in members)
            total_pixels = sum(value.total_pixels for value in members)
            losses = tuple(value.loss for value in members)
            diagnostics.append(
                StratumDiagnostic(
                    days_between=days_between,
                    correlation_bin=correlation_bin,
                    roi_count=len(members),
                    trusted_pixels=trusted_pixels,
                    total_pixels=total_pixels,
                    coverage=trusted_pixels / total_pixels,
                    mean_loss=math.fsum(losses) / len(losses),
                    maximum_loss=max(losses),
                )
            )
    return tuple(diagnostics)


def build_phase2b3c_statistics(rois: object) -> Phase2B3CStatistics:
    """Aggregate exactly 120 ROI observations under the frozen decision rule."""

    values = _validated_rois(rois)
    trusted_pixels = sum(value.trusted_pixels for value in values)
    total_pixels = sum(value.total_pixels for value in values)
    coverage = trusted_pixels / total_pixels
    losses = tuple(value.loss for value in values)
    mean_loss = math.fsum(losses) / PHASE2B3C_EVALUATION_SIZE
    risk_ucb = grid_kelly_upper_bound(losses)
    empirical_delta = mean_loss - PHASE2B3C_ALPHA
    empirical_violation = max(0.0, empirical_delta)
    finite_sample_margin = risk_ucb - mean_loss
    finite_sample_delta = risk_ucb - PHASE2B3C_ALPHA
    hoeffding_ucb = min(
        1.0,
        mean_loss
        + math.sqrt(
            math.log(1.0 / PHASE2B3C_DELTA)
            / (2 * PHASE2B3C_EVALUATION_SIZE)
        ),
    )
    if coverage < PHASE2B3C_MINIMUM_COVERAGE or mean_loss > PHASE2B3C_ALPHA:
        decision = FAILED
        reasons = tuple(
            reason
            for condition, reason in (
                (coverage < PHASE2B3C_MINIMUM_COVERAGE, INSUFFICIENT_COVERAGE),
                (mean_loss > PHASE2B3C_ALPHA, EMPIRICAL_RISK_EXCEEDS_TARGET),
            )
            if condition
        )
    elif risk_ucb <= PHASE2B3C_ALPHA:
        decision = CONFIRMED
        reasons = ()
    else:
        decision = EMPIRICALLY_MET_BUT_INCONCLUSIVE
        reasons = (FINITE_SAMPLE_UPPER_BOUND_EXCEEDS_TARGET,)
    return Phase2B3CStatistics(
        evaluation_size=PHASE2B3C_EVALUATION_SIZE,
        trusted_pixels=trusted_pixels,
        total_pixels=total_pixels,
        coverage=coverage,
        mean_loss=mean_loss,
        empirical_risk_delta=empirical_delta,
        empirical_risk_violation=empirical_violation,
        risk_ucb=risk_ucb,
        finite_sample_margin=finite_sample_margin,
        finite_sample_delta=finite_sample_delta,
        hoeffding_ucb=hoeffding_ucb,
        phase_decision=decision,
        reasons=reasons,
        strata=_stratum_diagnostics(values),
    )
