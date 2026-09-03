"""Byte-identical, inference-free receipts for Phase 2B3-B calibration replay."""

from __future__ import annotations

import hashlib
import json

from trustsr.jsonio import canonical_json

MAX_DOCUMENT_BYTES = 5 * 1024 * 1024
RESULT_SCHEMA = "trustsr.phase2b3b-calibration.v1"
CACHE_AUDIT_SCHEMA = "trustsr.phase2b3b-calibration-cache-audit.v1"
RUNTIME_MANIFEST_SCHEMA = "trustsr.phase2b3b-calibration-runtime.v1"
REPLAY_RECEIPT_SCHEMA = "trustsr.phase2b3b-calibration-replay.v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_document(payload: object, *, schema: str, label: str) -> bytes:
    if type(payload) is not bytes:
        raise TypeError(f"committed {label} must be immutable bytes")
    if len(payload) > MAX_DOCUMENT_BYTES:
        raise ValueError(f"committed {label} exceeds the 5 MiB limit")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"committed {label} is not valid UTF-8 JSON") from exc
    if type(value) is not dict or value.get("schema") != schema:
        raise ValueError(f"committed {label} schema is invalid")
    try:
        canonical = canonical_json(value)
    except ValueError as exc:
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
    except ValueError as exc:
        raise ValueError(f"rebuilt {label} is not canonical JSON data") from exc
    if len(payload) > MAX_DOCUMENT_BYTES:
        raise ValueError(f"rebuilt {label} exceeds the 5 MiB limit")
    return payload


def build_calibration_replay_receipt(
    committed_result: bytes,
    committed_cache_audit: bytes,
    committed_runtime_manifest: bytes,
    rebuilt_result: object,
    rebuilt_cache_audit: object,
) -> dict[str, object]:
    """Verify byte-for-byte replay of result and cache audit without inference."""

    result = _canonical_document(committed_result, schema=RESULT_SCHEMA, label="result")
    cache_audit = _canonical_document(
        committed_cache_audit, schema=CACHE_AUDIT_SCHEMA, label="cache audit"
    )
    runtime_manifest = _canonical_document(
        committed_runtime_manifest,
        schema=RUNTIME_MANIFEST_SCHEMA,
        label="runtime manifest",
    )
    if _canonical_rebuild(rebuilt_result, schema=RESULT_SCHEMA, label="result") != result:
        raise ValueError("rebuilt result is not byte-identical to the committed result")
    if (
        _canonical_rebuild(
            rebuilt_cache_audit,
            schema=CACHE_AUDIT_SCHEMA,
            label="cache audit",
        )
        != cache_audit
    ):
        raise ValueError("rebuilt cache audit is not byte-identical to the committed cache audit")
    return {
        "schema": REPLAY_RECEIPT_SCHEMA,
        "byte_identical": True,
        "result_sha256": _sha256(result),
        "cache_audit_sha256": _sha256(cache_audit),
        "runtime_manifest_sha256": _sha256(runtime_manifest),
    }
