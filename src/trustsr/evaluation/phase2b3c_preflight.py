"""Deterministic metadata-only preflight for Phase 2B3-C."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
)
from trustsr.data.internal_test_pairs import validate_internal_test_records
from trustsr.data.internal_test_subset import load_internal_test_records
from trustsr.evaluation.phase2b3c_access import VerifiedAccessPermit
from trustsr.evaluation.phase2b3c_evidence import (
    INPUT_AUDIT_SHA256,
    PHASE2B3B_FILE_SHA256S,
    PHASE2B3B_PUBLICATION_COMMIT,
    PRODUCER_REVISION,
    FrozenPhase2B3BEvidence,
    load_frozen_phase2b3b_evidence,
)
from trustsr.evaluation.phase2b3c_policy import (
    PHASE2B3C_ALPHA,
    PHASE2B3C_BISECTION_ITERATIONS,
    PHASE2B3C_DELTA,
    PHASE2B3C_EVALUATION_SIZE,
    PHASE2B3C_GRID_SIZE,
    PHASE2B3C_MINIMUM_COVERAGE,
    PHASE2B3C_RISK_UPPER_BOUND,
    PHASE2B3C_THRESHOLD,
)
from trustsr.evaluation.phase2b3c_revision import (
    VerifiedRevision,
    verify_phase2b3c_implementation_revision,
)
from trustsr.jsonio import canonical_json

SCHEMA = "trustsr.phase2b3c-preflight.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_DAYS = (-1, 0, 1)
_BINS = (0, 1, 2, 3)


def _freeze(value: Any) -> object:
    if type(value) is dict:
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if type(value) in (list, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validated_evidence(value: object) -> FrozenPhase2B3BEvidence:
    if type(value) is not FrozenPhase2B3BEvidence:
        raise ValueError("preflight requires exact frozen Phase 2B3-B evidence")
    if (
        value.result_sha256
        != PHASE2B3B_FILE_SHA256S["sen2naipv2-calibration-conformal-v1.json"]
        or value.cache_audit_sha256
        != PHASE2B3B_FILE_SHA256S[
            "sen2naipv2-calibration-conformal-cache-audit-v1.json"
        ]
        or value.acceptance_sha256
        != PHASE2B3B_FILE_SHA256S[
            "sen2naipv2-calibration-conformal-acceptance-v1.json"
        ]
        or value.threshold != PHASE2B3C_THRESHOLD
        or value.alpha != PHASE2B3C_ALPHA
        or value.minimum_coverage != PHASE2B3C_MINIMUM_COVERAGE
        or value.seeds != (3407, 3408, 3409, 3410, 3411)
        or value.post_manifest_sha256 != POST_MANIFEST_SHA256
        or value.input_audit_sha256 != INPUT_AUDIT_SHA256
        or value.producer_revision != PRODUCER_REVISION
        or value.publication_commit != PHASE2B3B_PUBLICATION_COMMIT
    ):
        raise ValueError("Phase 2B3-B evidence differs from the frozen C input")
    frozen = value.as_dict()
    if (
        frozen.get("threshold") != PHASE2B3C_THRESHOLD
        or frozen.get("target")
        != {"alpha": PHASE2B3C_ALPHA, "minimum_coverage": PHASE2B3C_MINIMUM_COVERAGE}
    ):
        raise ValueError("Phase 2B3-B frozen payload is invalid")
    return value


def _validated_revision(value: object) -> VerifiedRevision:
    if (
        type(value) is not VerifiedRevision
        or value.branch != "main"
        or type(value.head_revision) is not str
        or _REVISION.fullmatch(value.head_revision) is None
        or type(value.implementation_revision) is not str
        or _REVISION.fullmatch(value.implementation_revision) is None
        or type(value.computation_tree_sha256) is not str
        or _DIGEST.fullmatch(value.computation_tree_sha256) is None
    ):
        raise ValueError("preflight requires the verified main computation revision")
    return value


def _membership(record: Mapping[str, object]) -> dict[str, object]:
    lr_asset = record["lr_asset"]
    hr_asset = record["hr_asset"]
    if not isinstance(lr_asset, Mapping) or not isinstance(hr_asset, Mapping):
        raise ValueError("internal_test preflight requires LR and HR asset metadata")
    return {
        "sample_id": record["sample_id"],
        "selection_sha256": record["selection_sha256"],
        "spatial_group_id": record["spatial_group_id"],
        "lr_asset_sha256": _digest(lr_asset.get("sha256"), "LR asset digest"),
        "hr_asset_sha256": _digest(hr_asset.get("sha256"), "HR asset digest"),
        "days_between": record["days_between"],
        "correlation_bin": record["correlation_bin"],
        "selection_round": record["selection_round"],
    }


def _policy() -> dict[str, object]:
    return {
        "alpha": PHASE2B3C_ALPHA,
        "minimum_coverage": PHASE2B3C_MINIMUM_COVERAGE,
        "confidence_error": PHASE2B3C_DELTA,
        "bet_grid_size": PHASE2B3C_GRID_SIZE,
        "bisection_iterations": PHASE2B3C_BISECTION_ITERATIONS,
        "threshold": PHASE2B3C_THRESHOLD,
        "score": {
            "name": "ldsr_variance_k5",
            "operator_parameters": {
                "algorithm": "ensemble_variance_score",
                "band_reduction": "mean",
                "correction": 0,
                "seed_count": 5,
                "seed_first": 3407,
                "seed_last": 3411,
            },
            "seeds": [3407, 3408, 3409, 3410, 3411],
        },
        "risk": {
            "name": "local_l1_risk",
            "window": 9,
            "upper_bound": PHASE2B3C_RISK_UPPER_BOUND,
        },
    }


def build_phase2b3c_preflight(
    evidence: FrozenPhase2B3BEvidence,
    internal_test_records: Sequence[Mapping[str, object]],
    revision: VerifiedRevision,
    *,
    permit: VerifiedAccessPermit | None = None,
) -> Mapping[str, object]:
    """Build an immutable host-free summary without opening pixels or caches."""

    frozen = _validated_evidence(evidence)
    verified_revision = _validated_revision(revision)
    records = validate_internal_test_records(internal_test_records)
    memberships = tuple(_membership(record) for record in records)
    sample_ids = [membership["sample_id"] for membership in memberships]
    counts = Counter(
        (membership["days_between"], membership["correlation_bin"])
        for membership in memberships
    )
    ordered_membership_sha256 = hashlib.sha256(
        canonical_json(memberships)
    ).hexdigest()
    if permit is not None:
        if (
            type(permit) is not VerifiedAccessPermit
            or permit.implementation_revision
            != verified_revision.implementation_revision
            or permit.computation_tree_sha256
            != verified_revision.computation_tree_sha256
            or permit.ordered_membership_sha256 != ordered_membership_sha256
            or permit.phase2b3b_acceptance_sha256 != frozen.acceptance_sha256
        ):
            raise ValueError("optional access permit differs from preflight identity")
        permit_projection: dict[str, object] | None = {
            "evaluation_id": _digest(permit.evaluation_id, "permit evaluation ID"),
            "permit_sha256": _digest(permit.permit_sha256, "permit digest"),
            "readiness_sha256": _digest(
                permit.readiness_sha256, "permit readiness digest"
            ),
            "environment_sha256": _digest(
                permit.environment_sha256, "permit environment digest"
            ),
        }
    else:
        permit_projection = None
    result = {
        "schema": SCHEMA,
        "authorization_required": True,
        "upstream": {
            "publication_commit": frozen.publication_commit,
            "producer_revision": frozen.producer_revision,
            "post_manifest_sha256": frozen.post_manifest_sha256,
            "input_audit_sha256": frozen.input_audit_sha256,
            "evidence_sha256s": dict(PHASE2B3B_FILE_SHA256S),
        },
        "implementation": {
            "revision": verified_revision.implementation_revision,
            "computation_tree_sha256": verified_revision.computation_tree_sha256,
        },
        "evaluation": {
            "split": "internal_test",
            "sample_count": PHASE2B3C_EVALUATION_SIZE,
            "ordered_sample_ids_sha256": hashlib.sha256(
                canonical_json(sample_ids)
            ).hexdigest(),
            "ordered_membership_sha256": ordered_membership_sha256,
            "input_receipt_sha256s": [
                hashlib.sha256(canonical_json(item)).hexdigest()
                for item in memberships
            ],
            "strata": [
                {
                    "days_between": day,
                    "correlation_bin": bin_index,
                    "sample_count": counts[(day, bin_index)],
                }
                for day in _DAYS
                for bin_index in _BINS
            ],
        },
        "policy": _policy(),
        "input": {
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "crop_policy": CROP_POLICY,
            "bands": ["B04", "B03", "B02", "B08"],
            "scale": 4,
        },
        "permit": permit_projection,
    }
    immutable = _freeze(result)
    if not isinstance(immutable, Mapping):  # pragma: no cover - invariant
        raise RuntimeError("preflight immutability invariant was violated")
    return immutable


def load_phase2b3c_preflight(
    *,
    project_root: Path,
    evidence_dir: Path,
    storage_root: Path,
    manifest_path: Path,
    implementation_revision: str,
    computation_tree_sha256: str,
) -> Mapping[str, object]:
    """Compose only Git, evidence, and manifest-metadata gates."""

    revision = verify_phase2b3c_implementation_revision(
        project_root, implementation_revision, computation_tree_sha256
    )
    evidence = load_frozen_phase2b3b_evidence(evidence_dir, project_root)
    records = load_internal_test_records(storage_root, manifest_path)
    return build_phase2b3c_preflight(evidence, records, revision)
