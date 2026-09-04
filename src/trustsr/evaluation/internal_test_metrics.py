"""Frozen ROI loss/coverage projection for Phase 2B3-C."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from trustsr.data.crosssensor_pairs import LoadedCrosssensorPair
from trustsr.evaluation.internal_test_maps import InternalTestMaps
from trustsr.evaluation.phase2b3c_policy import (
    PHASE2B3C_EVALUATION_SIZE,
    require_phase2b3c_policy,
)
from trustsr.evaluation.phase2b3c_statistics import ROIEvaluation, evaluate_roi_loss
from trustsr.jsonio import canonical_json

_DIGEST = re.compile(r"[0-9a-f]{64}")
_DAYS = (-1, 0, 1)
_BINS = (0, 1, 2, 3)
_ROUNDS = tuple(range(1, 11))


@dataclass(frozen=True)
class InternalTestMetricSet:
    """Ordered ROI observations plus the exact source-map identity."""

    rois: tuple[ROIEvaluation, ...]
    map_evidence_sha256: str

    def __post_init__(self) -> None:
        if type(self.rois) is not tuple or len(self.rois) != PHASE2B3C_EVALUATION_SIZE:
            raise ValueError("internal_test metrics require exactly 120 ROI evaluations")
        if any(type(value) is not ROIEvaluation for value in self.rois):
            raise TypeError("internal_test metrics require exact ROIEvaluation values")
        for value in self.rois:
            value.__post_init__()
        sample_ids = tuple(value.sample_id for value in self.rois)
        if len(set(sample_ids)) != PHASE2B3C_EVALUATION_SIZE:
            raise ValueError("internal_test metric sample IDs must be unique")
        positions = tuple(
            (value.days_between, value.correlation_bin, value.selection_round)
            for value in self.rois
        )
        expected = {
            (day, bin_index, round_index)
            for day in _DAYS
            for bin_index in _BINS
            for round_index in _ROUNDS
        }
        if len(set(positions)) != len(positions) or set(positions) != expected:
            raise ValueError("internal_test metrics must cover every balanced design position")
        if (
            type(self.map_evidence_sha256) is not str
            or _DIGEST.fullmatch(self.map_evidence_sha256) is None
        ):
            raise ValueError("map evidence digest must be lowercase SHA-256")


def _validated_pair_and_maps(
    pair: object, maps: object
) -> tuple[LoadedCrosssensorPair, InternalTestMaps]:
    if type(pair) is not LoadedCrosssensorPair:
        raise TypeError("internal_test metric requires a LoadedCrosssensorPair")
    if type(maps) is not InternalTestMaps:
        raise TypeError("internal_test metric requires InternalTestMaps")
    pair.pair.validate()
    maps.__post_init__()
    if pair.metadata.split != "internal_test":
        raise ValueError("internal_test metric accepts only the internal_test split")
    if (
        pair.pair.sample_id != pair.metadata.sample_id
        or maps.sample_id != pair.pair.sample_id
    ):
        raise ValueError("internal_test metric pair and map sample identities differ")
    return pair, maps


def evaluate_internal_test_metric(
    pair: object, maps: object, *, threshold: float
) -> ROIEvaluation:
    """Compute inclusive-threshold coverage and maximum trusted R9 loss."""

    loaded, validated_maps = _validated_pair_and_maps(pair, maps)
    return evaluate_roi_loss(
        sample_id=loaded.pair.sample_id,
        days_between=loaded.metadata.days_between,
        correlation_bin=loaded.metadata.correlation_bin,
        selection_round=loaded.metadata.selection_round,
        score=validated_maps.score.tensor,
        risk=validated_maps.risk,
        threshold=threshold,
    )


def build_internal_test_metrics(
    pairs: Sequence[LoadedCrosssensorPair],
    maps: Sequence[InternalTestMaps],
    *,
    threshold: float,
) -> InternalTestMetricSet:
    """Build the fixed 120 ROI observations without aggregation or disclosure."""

    if isinstance(pairs, str | bytes) or not isinstance(pairs, Sequence):
        raise TypeError("internal_test metric pairs must be a stable sequence")
    if isinstance(maps, str | bytes) or not isinstance(maps, Sequence):
        raise TypeError("internal_test maps must be a stable sequence")
    pair_values = tuple(pairs)
    map_values = tuple(maps)
    if len(pair_values) != PHASE2B3C_EVALUATION_SIZE or len(map_values) != len(
        pair_values
    ):
        raise ValueError("internal_test metrics require exactly 120 aligned pairs and maps")
    # Reject a policy mutation before iterating over any scientific values.
    require_phase2b3c_policy(
        alpha=0.05,
        minimum_coverage=0.10,
        delta=0.05,
        grid_size=20,
        threshold=threshold,
    )
    rois = tuple(
        evaluate_internal_test_metric(pair, sample_maps, threshold=threshold)
        for pair, sample_maps in zip(pair_values, map_values, strict=True)
    )
    evidence = [
        {
            "sample_id": sample_maps.sample_id,
            "score_sha256": sample_maps.score.score_sha256,
            "risk_sha256": sample_maps.risk_sha256,
        }
        for sample_maps in map_values
    ]
    return InternalTestMetricSet(
        rois=rois,
        map_evidence_sha256=hashlib.sha256(canonical_json(evidence)).hexdigest(),
    )
