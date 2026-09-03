"""Deterministic metadata-only preflight for Phase 2B3-B calibration."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from trustsr.data.calibration_subset import load_calibration_records
from trustsr.evaluation.phase2b3b_evidence import (
    CROP_POLICY,
    INPUT_AUDIT_SHA256,
    NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    PRODUCER_REVISION,
    PUBLICATION_COMMIT,
    PUBLISHED_EVIDENCE_SHA256S,
    FrozenPhase2B3AEvidence,
    load_frozen_phase2b3a_evidence,
)

_DAYS = (-1, 0, 1)
_BINS = (0, 1, 2, 3)
_ROUNDS = tuple(range(1, 11))
_SEEDS = (3407, 3408, 3409, 3410, 3411)
_OPERATOR_PARAMETERS = {
    "algorithm": "ensemble_variance_score",
    "band_reduction": "mean",
    "correction": 0,
    "seed_count": 5,
    "seed_first": 3407,
    "seed_last": 3411,
}


def _freeze(value: Any) -> object:
    if type(value) is dict:
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if type(value) in (list, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _validated_evidence(value: object) -> FrozenPhase2B3AEvidence:
    if not isinstance(value, FrozenPhase2B3AEvidence):
        raise ValueError("preflight requires frozen Phase 2B3-A evidence")
    if (
        not isinstance(value.source_digests, Mapping)
        or dict(value.source_digests) != dict(PUBLISHED_EVIDENCE_SHA256S)
        or value.score_name != "ldsr_variance_k5"
        or not isinstance(value.operator_parameters, Mapping)
        or dict(value.operator_parameters) != _OPERATOR_PARAMETERS
        or value.seeds != _SEEDS
        or value.risk_name != "local_l1_risk"
        or type(value.risk_window) is not int
        or value.risk_window != 9
        or type(value.risk_upper_bound) is not float
        or value.risk_upper_bound != 1.0
        or value.normalization_policy != NORMALIZATION_POLICY
        or value.crop_policy != CROP_POLICY
        or value.bands != ("B04", "B03", "B02", "B08")
        or type(value.scale) is not int
        or value.scale != 4
        or value.post_manifest_sha256 != POST_MANIFEST_SHA256
        or value.input_audit_sha256 != INPUT_AUDIT_SHA256
        or value.producer_revision != PRODUCER_REVISION
        or value.publication_commit != PUBLICATION_COMMIT
    ):
        raise ValueError("Phase 2B3-A evidence does not match the frozen B3-B identity")
    candidates = value.candidate_eligibility_evidence
    expected_names = ("lr_reprojection_l1", "three_model_disagreement", "ldsr_variance_k5")
    if (
        type(candidates) is not tuple
        or len(candidates) != len(expected_names)
        or any(
            not isinstance(candidate, Mapping)
            or candidate.get("name") != name
            or candidate.get("eligible") is not True
            or candidate.get("failure_reasons") != ()
            for candidate, name in zip(candidates, expected_names, strict=True)
        )
    ):
        raise ValueError("Phase 2B3-A candidate eligibility evidence is invalid")
    return value


def _calibration_strata(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, int], ...]:
    if not isinstance(records, Sequence) or len(records) != 120:
        raise ValueError("Phase 2B3-B preflight requires exactly 120 calibration records")
    rounds: dict[tuple[int, int], list[int]] = {
        (day, bin_index): [] for day in _DAYS for bin_index in _BINS
    }
    for record in records:
        if not isinstance(record, Mapping) or record.get("split") != "calibration":
            raise ValueError("Phase 2B3-B preflight accepts calibration metadata only")
        day = record.get("days_between")
        bin_index = record.get("correlation_bin")
        selection_round = record.get("selection_round")
        if (
            type(day) is not int
            or type(bin_index) is not int
            or type(selection_round) is not int
            or (day, bin_index) not in rounds
        ):
            raise ValueError("Phase 2B3-B calibration stratum metadata is invalid")
        rounds[(day, bin_index)].append(selection_round)
    if any(tuple(sorted(cell_rounds)) != _ROUNDS for cell_rounds in rounds.values()):
        raise ValueError("each Phase 2B3-B calibration stratum requires rounds 1 through 10")
    counts = Counter(
        (record["days_between"], record["correlation_bin"]) for record in records
    )
    return tuple(
        {
            "days_between": day,
            "correlation_bin": bin_index,
            "sample_count": counts[(day, bin_index)],
        }
        for day in _DAYS
        for bin_index in _BINS
    )


def build_phase2b3b_preflight(
    evidence: FrozenPhase2B3AEvidence,
    calibration_records: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Build an immutable host-free summary without loading sample pixels."""

    frozen = _validated_evidence(evidence)
    strata = _calibration_strata(calibration_records)
    result = {
        "schema": "trustsr.phase2b3b-preflight.v1",
        "upstream": {
            "publication_commit": frozen.publication_commit,
            "producer_revision": frozen.producer_revision,
            "post_manifest_sha256": frozen.post_manifest_sha256,
            "input_audit_sha256": frozen.input_audit_sha256,
            "evidence_sha256s": dict(frozen.source_digests),
        },
        "calibration": {"sample_count": 120, "strata": strata},
        "score": {
            "name": frozen.score_name,
            "operator_parameters": dict(frozen.operator_parameters),
            "seeds": frozen.seeds,
        },
        "risk": {
            "name": frozen.risk_name,
            "window": frozen.risk_window,
            "upper_bound": frozen.risk_upper_bound,
        },
        "input": {
            "normalization_policy": frozen.normalization_policy,
            "crop_policy": frozen.crop_policy,
            "bands": frozen.bands,
            "scale": frozen.scale,
        },
    }
    immutable = _freeze(result)
    if not isinstance(immutable, Mapping):
        raise AssertionError("preflight result must be an immutable mapping")
    return immutable


def load_phase2b3b_preflight(
    evidence_dir: Path,
    storage_root: Path,
    manifest_path: Path,
) -> Mapping[str, object]:
    """Load verified evidence and calibration metadata, then build the preflight summary."""

    evidence = load_frozen_phase2b3a_evidence(evidence_dir)
    calibration_records = load_calibration_records(storage_root, manifest_path)
    return build_phase2b3b_preflight(evidence, calibration_records)
