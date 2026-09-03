"""Byte-identical Phase 2B3-B calibration replay receipt contracts."""

from __future__ import annotations

import hashlib

import pytest

from trustsr.evaluation.calibration_replay_receipt import (
    MAX_DOCUMENT_BYTES,
    build_calibration_replay_receipt,
)
from trustsr.jsonio import canonical_json

_RESULT_SCHEMA = "trustsr.phase2b3b-calibration.v1"
_AUDIT_SCHEMA = "trustsr.phase2b3b-calibration-cache-audit.v1"
_RUNTIME_SCHEMA = "trustsr.phase2b3b-calibration-runtime.v1"


def _payload(schema: str, **fields: object) -> bytes:
    return canonical_json({"schema": schema, **fields})


def _documents() -> tuple[bytes, bytes, bytes, dict[str, object], dict[str, object]]:
    result = _payload(_RESULT_SCHEMA, calibration=120)
    audit = _payload(_AUDIT_SCHEMA, scores=120)
    runtime = _payload(_RUNTIME_SCHEMA, runtime="recorded-only")
    return (
        result,
        audit,
        runtime,
        {"schema": _RESULT_SCHEMA, "calibration": 120},
        {"schema": _AUDIT_SCHEMA, "scores": 120},
    )


def test_builds_fresh_host_free_receipt_from_byte_identical_rebuilds() -> None:
    result, audit, runtime, rebuilt_result, rebuilt_audit = _documents()

    first = build_calibration_replay_receipt(result, audit, runtime, rebuilt_result, rebuilt_audit)
    second = build_calibration_replay_receipt(result, audit, runtime, result, audit)

    expected = {
        "schema": "trustsr.phase2b3b-calibration-replay.v1",
        "byte_identical": True,
        "result_sha256": hashlib.sha256(result).hexdigest(),
        "cache_audit_sha256": hashlib.sha256(audit).hexdigest(),
        "runtime_manifest_sha256": hashlib.sha256(runtime).hexdigest(),
    }
    assert first == second == expected
    assert first is not second
    assert set(first) == set(expected)
    first["byte_identical"] = False
    assert second["byte_identical"] is True
    assert canonical_json(second)


@pytest.mark.parametrize("bad", (bytearray(b"{}"), memoryview(b"{}"), "{}", object()))
def test_rejects_non_bytes_committed_documents(bad: object) -> None:
    result, audit, runtime, rebuilt_result, rebuilt_audit = _documents()

    with pytest.raises(TypeError, match="bytes"):
        build_calibration_replay_receipt(bad, audit, runtime, rebuilt_result, rebuilt_audit)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("document", "payload"),
    (
        ("result", b'{"calibration":120,"schema":"trustsr.phase2b3b-calibration.v1"} '),
        ("audit", _payload("wrong", scores=120)),
        ("runtime", b'{"schema":"trustsr.phase2b3b-calibration-runtime.v1","x":NaN}'),
    ),
)
def test_rejects_noncanonical_or_wrong_schema_committed_documents(
    document: str, payload: bytes
) -> None:
    result, audit, runtime, rebuilt_result, rebuilt_audit = _documents()
    values = {"result": result, "audit": audit, "runtime": runtime}
    values[document] = payload

    with pytest.raises(ValueError):
        build_calibration_replay_receipt(
            values["result"],
            values["audit"],
            values["runtime"],
            rebuilt_result,
            rebuilt_audit,
        )


def test_rejects_rebuilt_result_or_audit_that_is_not_byte_identical() -> None:
    result, audit, runtime, rebuilt_result, rebuilt_audit = _documents()
    rebuilt_result["calibration"] = 119

    with pytest.raises(ValueError, match="byte-identical"):
        build_calibration_replay_receipt(result, audit, runtime, rebuilt_result, rebuilt_audit)

    with pytest.raises(ValueError, match="byte-identical"):
        build_calibration_replay_receipt(
            result,
            audit,
            runtime,
            result,
            _payload(_AUDIT_SCHEMA, scores=119),
        )


@pytest.mark.parametrize("bad", ([], "{}", bytearray(b"{}"), float("nan")))
def test_rejects_rebuilt_values_that_are_not_canonical_bytes_or_objects(bad: object) -> None:
    result, audit, runtime, _rebuilt_result, rebuilt_audit = _documents()

    with pytest.raises((TypeError, ValueError)):
        build_calibration_replay_receipt(result, audit, runtime, bad, rebuilt_audit)

    with pytest.raises(ValueError, match="schema"):
        build_calibration_replay_receipt(
            result,
            audit,
            runtime,
            {"schema": "wrong"},
            rebuilt_audit,
        )


def test_rejects_document_larger_than_five_mebibytes() -> None:
    result, audit, runtime, rebuilt_result, rebuilt_audit = _documents()
    oversized = b"{" + b" " * MAX_DOCUMENT_BYTES + b"}"

    with pytest.raises(ValueError, match="5 MiB"):
        build_calibration_replay_receipt(oversized, audit, runtime, rebuilt_result, rebuilt_audit)
