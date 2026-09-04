"""Byte-identical, inference-free replay receipts for Phase 2B3-C."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from trustsr.jsonio import canonical_json

MAX_DOCUMENT_BYTES = 5 * 1024 * 1024
RESULT_SCHEMA = "trustsr.phase2b3c-evaluation.v1"
CACHE_AUDIT_SCHEMA = "trustsr.phase2b3c-evaluation-cache-audit.v1"
RUNTIME_SCHEMA = "trustsr.phase2b3c-evaluation-runtime.v1"
REPLAY_SCHEMA = "trustsr.phase2b3c-evaluation-replay.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class VerifiedPhase2B3CReplay:
    """Metadata-consistent receipt identity without computation authority."""

    replay_sha256: str
    result_sha256: str
    cache_audit_sha256: str
    runtime_sha256: str
    map_evidence_sha256: str
    cache_computation_verified: bool
    prediction_inference_verified: bool
    acceptance_authorized: bool

    def __post_init__(self) -> None:
        for value in (
            self.replay_sha256,
            self.result_sha256,
            self.cache_audit_sha256,
            self.runtime_sha256,
            self.map_evidence_sha256,
        ):
            _digest(value, "verified replay digest")
        if (
            self.cache_computation_verified is not False
            or self.prediction_inference_verified is not False
            or self.acceptance_authorized is not False
        ):
            raise ValueError("verified replay scope is invalid")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_document(payload: object, *, schema: str, label: str) -> bytes:
    if type(payload) is not bytes:
        raise TypeError(f"committed {label} must be immutable bytes")
    if len(payload) > MAX_DOCUMENT_BYTES:
        raise ValueError(f"committed {label} exceeds the 5 MiB limit")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"committed {label} is not canonical UTF-8 JSON") from exc
    if type(value) is not dict or value.get("schema") != schema:
        raise ValueError(f"committed {label} schema is invalid")
    try:
        canonical = canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"committed {label} is not canonical JSON") from exc
    if canonical != payload:
        raise ValueError(f"committed {label} is not canonical JSON")
    return payload


def _canonical_rebuild(value: object, *, schema: str, label: str) -> bytes:
    if type(value) is bytes:
        return _canonical_document(value, schema=schema, label=f"rebuilt {label}")
    if type(value) is not dict:
        raise TypeError(f"rebuilt {label} must be immutable bytes or an exact JSON object")
    if value.get("schema") != schema:
        raise ValueError(f"rebuilt {label} schema is invalid")
    try:
        payload = canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"rebuilt {label} is not canonical JSON data") from exc
    if len(payload) > MAX_DOCUMENT_BYTES:
        raise ValueError(f"rebuilt {label} exceeds the 5 MiB limit")
    return payload


def _result_bindings(result: bytes) -> tuple[dict[str, str], str]:
    value = json.loads(result)
    evaluation = value.get("evaluation")
    digests = value.get("digests")
    if type(evaluation) is not dict or type(digests) is not dict:
        raise ValueError("committed result replay bindings are invalid")
    evaluation_id = _digest(evaluation.get("evaluation_id"), "evaluation ID")
    permit_sha256 = _digest(evaluation.get("permit_sha256"), "permit digest")
    map_evidence_sha256 = _digest(
        digests.get("map_evidence_sha256"), "map evidence digest"
    )
    return {
        "evaluation_id": evaluation_id,
        "permit_sha256": permit_sha256,
    }, map_evidence_sha256


def build_phase2b3c_replay_receipt(
    committed_result: bytes,
    committed_cache_audit: bytes,
    committed_runtime: bytes,
    *,
    rebuilt_result: object,
    rebuilt_cache_audit: object,
) -> dict[str, object]:
    """Bind byte-identical downstream reconstruction without authorizing acceptance."""

    result = _canonical_document(
        committed_result, schema=RESULT_SCHEMA, label="result"
    )
    audit = _canonical_document(
        committed_cache_audit, schema=CACHE_AUDIT_SCHEMA, label="cache audit"
    )
    runtime = _canonical_document(
        committed_runtime, schema=RUNTIME_SCHEMA, label="runtime"
    )
    if _canonical_rebuild(
        rebuilt_result, schema=RESULT_SCHEMA, label="result"
    ) != result:
        raise ValueError("rebuilt result is not byte-identical to the committed result")
    if _canonical_rebuild(
        rebuilt_cache_audit,
        schema=CACHE_AUDIT_SCHEMA,
        label="cache audit",
    ) != audit:
        raise ValueError("rebuilt cache audit is not byte-identical to the committed cache audit")
    evaluation, map_evidence_sha256 = _result_bindings(result)
    return {
        "schema": REPLAY_SCHEMA,
        "phase": "internal_test_evaluation",
        "verification_scope": "cache_computation_replay",
        "cache_computation_verified": True,
        "prediction_inference_verified": False,
        "membership_authority_verified": False,
        "acceptance_authorized": False,
        "byte_identical": True,
        "evaluation": evaluation,
        "artifacts": {
            "result_sha256": _sha256(result),
            "cache_audit_sha256": _sha256(audit),
            "runtime_sha256": _sha256(runtime),
            "map_evidence_sha256": map_evidence_sha256,
        },
    }


def verify_phase2b3c_replay_receipt(
    receipt: object,
    *,
    committed_result: bytes,
    committed_cache_audit: bytes,
    committed_runtime: bytes,
) -> VerifiedPhase2B3CReplay:
    """Verify an exact replay receipt against its three predecessor documents."""

    if type(receipt) is bytes:
        payload = _canonical_document(receipt, schema=REPLAY_SCHEMA, label="replay")
        value = json.loads(payload)
    elif type(receipt) is dict:
        try:
            payload = canonical_json(receipt)
            value = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("replay receipt is not canonical JSON data") from exc
    else:
        raise TypeError("replay receipt must be canonical bytes or an exact JSON object")
    expected = build_phase2b3c_replay_receipt(
        committed_result,
        committed_cache_audit,
        committed_runtime,
        rebuilt_result=committed_result,
        rebuilt_cache_audit=committed_cache_audit,
    )
    if value != expected:
        raise ValueError("replay receipt differs from its predecessor documents")
    artifacts = value["artifacts"]
    return VerifiedPhase2B3CReplay(
        replay_sha256=_sha256(payload),
        result_sha256=artifacts["result_sha256"],
        cache_audit_sha256=artifacts["cache_audit_sha256"],
        runtime_sha256=artifacts["runtime_sha256"],
        map_evidence_sha256=artifacts["map_evidence_sha256"],
        cache_computation_verified=False,
        prediction_inference_verified=False,
        acceptance_authorized=False,
    )
