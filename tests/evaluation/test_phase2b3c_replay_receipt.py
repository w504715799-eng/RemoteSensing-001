"""Inference-free Phase 2B3-C replay receipt contracts."""

from __future__ import annotations

import hashlib

import pytest

from trustsr.evaluation.phase2b3c_replay_receipt import (
    MAX_DOCUMENT_BYTES,
    build_phase2b3c_replay_receipt,
    verify_phase2b3c_replay_receipt,
)
from trustsr.jsonio import canonical_json

_RESULT_SCHEMA = "trustsr.phase2b3c-evaluation.v1"
_AUDIT_SCHEMA = "trustsr.phase2b3c-evaluation-cache-audit.v1"
_RUNTIME_SCHEMA = "trustsr.phase2b3c-evaluation-runtime.v1"


def _payload(schema: str, **fields: object) -> bytes:
    return canonical_json({"schema": schema, **fields})


def _documents():
    result = _payload(
        _RESULT_SCHEMA,
        evaluation={"evaluation_id": "a" * 64, "permit_sha256": "b" * 64},
        digests={"map_evidence_sha256": "c" * 64},
    )
    audit = _payload(_AUDIT_SCHEMA, sample_count=120)
    runtime = _payload(_RUNTIME_SCHEMA, inventory="host-free")
    return result, audit, runtime


def test_builds_non_authorizing_receipt_from_byte_identical_rebuilds() -> None:
    result, audit, runtime = _documents()

    receipt = build_phase2b3c_replay_receipt(
        result,
        audit,
        runtime,
        rebuilt_result=result,
        rebuilt_cache_audit=audit,
    )

    assert receipt == {
        "schema": "trustsr.phase2b3c-evaluation-replay.v1",
        "phase": "internal_test_evaluation",
        "verification_scope": "cache_computation_replay",
        "cache_computation_verified": True,
        "prediction_inference_verified": False,
        "membership_authority_verified": False,
        "acceptance_authorized": False,
        "byte_identical": True,
        "evaluation": {
            "evaluation_id": "a" * 64,
            "permit_sha256": "b" * 64,
        },
        "artifacts": {
            "result_sha256": hashlib.sha256(result).hexdigest(),
            "cache_audit_sha256": hashlib.sha256(audit).hexdigest(),
            "runtime_sha256": hashlib.sha256(runtime).hexdigest(),
            "map_evidence_sha256": "c" * 64,
        },
    }
    verified = verify_phase2b3c_replay_receipt(
        receipt,
        committed_result=result,
        committed_cache_audit=audit,
        committed_runtime=runtime,
    )
    assert verified.cache_computation_verified is False
    assert verified.prediction_inference_verified is False
    assert verified.acceptance_authorized is False


def test_rejects_nonidentical_rebuilds_and_noncanonical_committed_bytes() -> None:
    result, audit, runtime = _documents()
    with pytest.raises(ValueError, match="byte-identical"):
        build_phase2b3c_replay_receipt(
            result,
            audit,
            runtime,
            rebuilt_result=_payload(_RESULT_SCHEMA, changed=True),
            rebuilt_cache_audit=audit,
        )
    with pytest.raises(ValueError, match="byte-identical"):
        build_phase2b3c_replay_receipt(
            result,
            audit,
            runtime,
            rebuilt_result=result,
            rebuilt_cache_audit=_payload(_AUDIT_SCHEMA, sample_count=119),
        )
    with pytest.raises(ValueError, match="canonical"):
        build_phase2b3c_replay_receipt(
            result + b"\n",
            audit,
            runtime,
            rebuilt_result=result,
            rebuilt_cache_audit=audit,
        )


@pytest.mark.parametrize("bad", (bytearray(b"{}"), memoryview(b"{}"), "{}"))
def test_committed_documents_must_be_immutable_bytes(bad: object) -> None:
    result, audit, runtime = _documents()
    with pytest.raises(TypeError, match="bytes"):
        build_phase2b3c_replay_receipt(
            bad,
            audit,
            runtime,
            rebuilt_result=result,
            rebuilt_cache_audit=audit,
        )


def test_rejects_oversized_document() -> None:
    _result, audit, runtime = _documents()
    oversized = b"{" + b" " * MAX_DOCUMENT_BYTES + b"}"
    with pytest.raises(ValueError, match="5 MiB"):
        build_phase2b3c_replay_receipt(
            oversized,
            audit,
            runtime,
            rebuilt_result=oversized,
            rebuilt_cache_audit=audit,
        )


def test_replay_verifier_rejects_authority_or_digest_mutation() -> None:
    result, audit, runtime = _documents()
    receipt = build_phase2b3c_replay_receipt(
        result,
        audit,
        runtime,
        rebuilt_result=result,
        rebuilt_cache_audit=audit,
    )
    receipt["acceptance_authorized"] = True
    with pytest.raises(ValueError, match="replay"):
        verify_phase2b3c_replay_receipt(
            receipt,
            committed_result=result,
            committed_cache_audit=audit,
            committed_runtime=runtime,
        )

    receipt = build_phase2b3c_replay_receipt(
        result,
        audit,
        runtime,
        rebuilt_result=result,
        rebuilt_cache_audit=audit,
    )
    receipt["artifacts"]["runtime_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="replay"):
        verify_phase2b3c_replay_receipt(
            receipt,
            committed_result=result,
            committed_cache_audit=audit,
            committed_runtime=runtime,
        )
