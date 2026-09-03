"""Semantic verification entry point for Phase 2B3-B candidate bundles."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from trustsr.evaluation.phase2b3b_bundle import (
    BUNDLE_DOCUMENT_SCHEMAS,
    read_phase2b3b_bundle,
)
from trustsr.evaluation.phase2b3b_result_verify import (
    VerifiedPhase2B3BResult,
    verify_phase2b3b_result,
)
from trustsr.evaluation.phase2b3b_runtime import (
    VerifiedPhase2B3BRuntime,
    verify_phase2b3b_runtime_manifest,
)

SCHEMA = "trustsr.phase2b3b-candidate-bundle-metadata-verification.v1"
_REPLAY_SCHEMA = "trustsr.phase2b3b-calibration-replay.v1"
_RESULT_NAME = "phase2b3b-calibration-result.json"
_AUDIT_NAME = "phase2b3b-calibration-cache-audit.json"
_RUNTIME_NAME = "phase2b3b-calibration-runtime.json"
_REPLAY_NAME = "phase2b3b-calibration-replay.json"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_REPLAY_KEYS = {
    "schema",
    "byte_identical",
    "result_sha256",
    "cache_audit_sha256",
    "runtime_manifest_sha256",
}


@dataclass(frozen=True)
class VerifiedPhase2B3BBundle:
    """Candidate metadata-consistency receipt for transport and cross-document checks.

    It cannot prove cache pixels, prediction, or score/risk computations and cannot
    authorize acceptance on its own.
    """

    schema: str
    verification_scope: str
    cache_computation_verified: bool
    manifest_sha256: str
    result_sha256: str
    cache_audit_sha256: str
    runtime_manifest_sha256: str
    replay_sha256: str
    producer_revision: str
    ordered_sample_ids_sha256: str
    ordered_membership_sha256: str
    input_receipt_sha256: str
    ordered_inputs_sha256: str
    map_evidence_sha256: str
    radiometry_aggregate_sha256: str
    phase_decision: str


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise TypeError("bundle snapshots must retain immutable raw bytes")
    return hashlib.sha256(payload).hexdigest()


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _replay(
    value: object,
    *,
    result_sha256: str,
    audit_sha256: str,
    runtime_sha256: str,
) -> None:
    if type(value) is not dict or set(value) != _REPLAY_KEYS:
        raise ValueError("replay receipt must use the exact frozen schema")
    if value["schema"] != _REPLAY_SCHEMA or value["byte_identical"] is not True:
        raise ValueError("replay receipt does not attest byte-identical reconstruction")
    expected = {
        "result_sha256": result_sha256,
        "cache_audit_sha256": audit_sha256,
        "runtime_manifest_sha256": runtime_sha256,
    }
    for key, digest in expected.items():
        if _digest(value[key], f"replay {key}") != digest:
            raise ValueError("replay receipt digest differs from actual bundle bytes")


def verify_phase2b3b_bundle(
    bundle_dir: Path,
    *,
    project_root: Path,
    evidence_dir: Path,
    storage_root: Path,
    manifest_path: Path,
) -> VerifiedPhase2B3BBundle:
    """Verify candidate metadata consistency without publishing an acceptance decision.

    The atomic writer proves publication integrity only. This function is the
    semantic entry point: it retains the reader's raw byte snapshot, validates
    replay identities against those bytes, and delegates result authority to the
    independent final-result verifier. It does not prove cache pixels, prediction,
    or score/risk computations and cannot authorize acceptance on its own.
    """

    loaded = read_phase2b3b_bundle(bundle_dir)
    raw = dict(loaded.payloads)
    if set(raw) != set(BUNDLE_DOCUMENT_SCHEMAS):
        raise ValueError("loaded bundle does not contain the exact document set")
    documents = loaded.documents()
    result_payload = raw[_RESULT_NAME]
    audit_payload = raw[_AUDIT_NAME]
    runtime_payload = raw[_RUNTIME_NAME]
    replay_payload = raw[_REPLAY_NAME]
    result_sha256 = _sha256(result_payload)
    audit_sha256 = _sha256(audit_payload)
    runtime_sha256 = _sha256(runtime_payload)
    replay_sha256 = _sha256(replay_payload)

    _replay(
        documents[_REPLAY_NAME],
        result_sha256=result_sha256,
        audit_sha256=audit_sha256,
        runtime_sha256=runtime_sha256,
    )
    verified_result = verify_phase2b3b_result(
        documents[_RESULT_NAME],
        documents[_AUDIT_NAME],
        project_root=project_root,
        evidence_dir=evidence_dir,
        storage_root=storage_root,
        manifest_path=manifest_path,
    )
    if type(verified_result) is not VerifiedPhase2B3BResult:
        raise TypeError("result verifier returned an invalid receipt type")
    if (
        verified_result.result_sha256 != result_sha256
        or verified_result.cache_audit_sha256 != audit_sha256
    ):
        raise ValueError("result verification receipt differs from actual bundle bytes")
    verified_runtime = verify_phase2b3b_runtime_manifest(
        runtime_payload,
        documents[_RESULT_NAME],
        documents[_AUDIT_NAME],
        project_root=project_root,
        evidence_dir=evidence_dir,
        storage_root=storage_root,
        manifest_path=manifest_path,
    )
    if type(verified_runtime) is not VerifiedPhase2B3BRuntime:
        raise TypeError("runtime verifier returned an invalid receipt type")
    if (
        verified_runtime.runtime_sha256 != runtime_sha256
        or verified_runtime.result_sha256 != result_sha256
        or verified_runtime.cache_audit_sha256 != audit_sha256
        or verified_runtime.input_receipt_sha256
        != verified_result.input_receipt_sha256
        or verified_runtime.ordered_inputs_sha256
        != verified_result.ordered_inputs_sha256
        or verified_runtime.map_evidence_sha256 != verified_result.map_evidence_sha256
        or verified_runtime.producer_revision != verified_result.producer_revision
    ):
        raise ValueError("runtime verification receipt differs from actual bundle bytes")

    return VerifiedPhase2B3BBundle(
        schema=SCHEMA,
        verification_scope="metadata_consistency_only",
        cache_computation_verified=False,
        manifest_sha256=loaded.manifest_sha256,
        result_sha256=result_sha256,
        cache_audit_sha256=audit_sha256,
        runtime_manifest_sha256=runtime_sha256,
        replay_sha256=replay_sha256,
        producer_revision=verified_result.producer_revision,
        ordered_sample_ids_sha256=verified_result.ordered_sample_ids_sha256,
        ordered_membership_sha256=verified_result.ordered_membership_sha256,
        input_receipt_sha256=verified_result.input_receipt_sha256,
        ordered_inputs_sha256=verified_result.ordered_inputs_sha256,
        map_evidence_sha256=verified_result.map_evidence_sha256,
        radiometry_aggregate_sha256=verified_result.radiometry_aggregate_sha256,
        phase_decision=verified_result.phase_decision,
    )
