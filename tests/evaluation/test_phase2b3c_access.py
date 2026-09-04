"""Readiness, permit, and immutable one-time access ledger contracts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from trustsr.evaluation.phase2b3c_access import (
    ACCESS_PERMIT_FILENAME,
    AUTHORIZATION_STATEMENT,
    advance_access_ledger,
    build_phase2b3c_readiness,
    load_access_ledger,
    phase2b3c_access_lock,
    verify_phase2b3c_access_permit,
)
from trustsr.jsonio import canonical_json


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _readiness() -> dict[str, object]:
    return build_phase2b3c_readiness(
        implementation_revision="1" * 40,
        computation_tree_sha256="2" * 64,
        phase2b3b_result_sha256=(
            "5fe01daaaa27443d5289cc366eacbe1fb628492c25bab16fadd164d3a8eb3174"
        ),
        phase2b3b_cache_audit_sha256=(
            "40aff8aaccb27719d958d42026d68d8efc0c7f79479e3bd99777053ee1d6399f"
        ),
        phase2b3b_acceptance_sha256=(
            "ce7f1a91e9954235e289dfc6eececb67261ff57110caaf15fe6640a0fb3b69ab"
        ),
        ordered_membership_sha256="3" * 64,
        environment_sha256="4" * 64,
    )


def _permit_document(readiness: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "trustsr.phase2b3c-internal-test-access-authorization.v1",
        "evaluation_id": readiness["evaluation_id"],
        "readiness_sha256": _sha256(readiness),
        "implementation": readiness["implementation"],
        "phase2b3b": readiness["phase2b3b"],
        "input": readiness["input"],
        "policy": readiness["policy"],
        "environment_sha256": readiness["environment_sha256"],
        "one_time_evaluation": True,
        "authorization_statement": AUTHORIZATION_STATEMENT,
    }


def _verified_permit(tmp_path: Path):
    project = tmp_path / "project"
    permit_path = project / "artifacts" / "phase2b3c" / ACCESS_PERMIT_FILENAME
    permit_path.parent.mkdir(parents=True)
    readiness = _readiness()
    permit_path.write_bytes(canonical_json(_permit_document(readiness)))
    return verify_phase2b3c_access_permit(permit_path, project, readiness)


def test_readiness_is_deterministic_complete_and_host_free() -> None:
    first = _readiness()
    second = _readiness()

    assert canonical_json(first) == canonical_json(second)
    assert first["evaluation_id"] == second["evaluation_id"]
    assert first["one_time_evaluation"] is True
    assert first["input"]["sample_count"] == 120  # type: ignore[index]
    policy = first["policy"]
    assert policy["alpha"] == 0.05  # type: ignore[index]
    assert policy["minimum_coverage"] == 0.10  # type: ignore[index]
    assert policy["delta"] == 0.05  # type: ignore[index]
    assert policy["bet_grid_size"] == 20  # type: ignore[index]
    payload = canonical_json(first).decode("utf-8")
    for forbidden in ("/home/", "hostname", "timestamp", "secret", "token"):
        assert forbidden not in payload


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("implementation_revision", "HEAD", "implementation revision"),
        ("computation_tree_sha256", "A" * 64, "computation tree"),
        ("ordered_membership_sha256", "3" * 63, "membership"),
        ("environment_sha256", "", "environment"),
    ],
)
def test_readiness_rejects_malformed_identity(
    field: str, value: str, message: str
) -> None:
    arguments = {
        "implementation_revision": "1" * 40,
        "computation_tree_sha256": "2" * 64,
        "phase2b3b_result_sha256": "5" * 64,
        "phase2b3b_cache_audit_sha256": "6" * 64,
        "phase2b3b_acceptance_sha256": "7" * 64,
        "ordered_membership_sha256": "3" * 64,
        "environment_sha256": "4" * 64,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        build_phase2b3c_readiness(**arguments)


def test_verifies_only_exact_canonical_repository_permit(tmp_path: Path) -> None:
    verified = _verified_permit(tmp_path)

    assert verified.evaluation_id == _readiness()["evaluation_id"]
    assert len(verified.permit_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        verified.evaluation_id = "0" * 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("bytes", "canonical"),
        ("statement", "authorization"),
        ("one_time", "one-time"),
        ("policy", "readiness"),
        ("location", "location"),
        ("symlink", "symlink"),
    ],
)
def test_rejects_forged_noncanonical_or_misplaced_permit(
    tmp_path: Path, mutation: str, message: str
) -> None:
    project = tmp_path / "project"
    permit_path = project / "artifacts" / "phase2b3c" / ACCESS_PERMIT_FILENAME
    permit_path.parent.mkdir(parents=True)
    readiness = _readiness()
    document = _permit_document(readiness)
    if mutation == "statement":
        document["authorization_statement"] = "approved"
    elif mutation == "one_time":
        document["one_time_evaluation"] = False
    elif mutation == "policy":
        document["policy"] = {**document["policy"], "alpha": 0.1}  # type: ignore[arg-type]
    payload = canonical_json(document)
    if mutation == "bytes":
        payload += b"\n"
    permit_path.write_bytes(payload)
    candidate = permit_path
    if mutation == "location":
        candidate = project / "permit.json"
        candidate.write_bytes(payload)
    elif mutation == "symlink":
        target = project / "permit-target.json"
        permit_path.replace(target)
        permit_path.symlink_to(target)

    with pytest.raises(ValueError, match=message):
        verify_phase2b3c_access_permit(candidate, project, readiness)


def test_ledger_advances_monotonically_and_issues_guard(tmp_path: Path) -> None:
    permit = _verified_permit(tmp_path)
    storage = tmp_path / "storage"
    storage.mkdir()

    reserved = advance_access_ledger(storage, permit, "reserved")
    opened = advance_access_ledger(storage, permit, "pixels_opened")
    assert reserved.state == "reserved"
    assert opened.state == "pixels_opened"
    assert opened.previous_event_sha256 == reserved.event_sha256
    guard = opened.access_guard()
    assert guard.evaluation_id == permit.evaluation_id
    assert guard.ledger_event_sha256 == opened.event_sha256

    for state in ("caches_complete", "bundle_complete", "accepted"):
        snapshot = advance_access_ledger(storage, permit, state)
        assert snapshot.state == state
    assert load_access_ledger(storage, permit).state == "accepted"


def test_exact_resume_is_idempotent_but_skip_duplicate_and_rollback_fail(
    tmp_path: Path,
) -> None:
    permit = _verified_permit(tmp_path)
    storage = tmp_path / "storage"
    storage.mkdir()
    first = advance_access_ledger(storage, permit, "reserved")
    resumed = advance_access_ledger(storage, permit, "reserved")
    assert resumed == first
    with pytest.raises(ValueError, match="transition"):
        advance_access_ledger(storage, permit, "caches_complete")
    opened = advance_access_ledger(storage, permit, "pixels_opened")
    with pytest.raises(ValueError, match="transition"):
        advance_access_ledger(storage, permit, "reserved")
    assert load_access_ledger(storage, permit) == opened


def test_detects_deleted_intermediate_event_and_root_drift(tmp_path: Path) -> None:
    permit = _verified_permit(tmp_path)
    storage = tmp_path / "storage"
    storage.mkdir()
    advance_access_ledger(storage, permit, "reserved")
    advance_access_ledger(storage, permit, "pixels_opened")
    advance_access_ledger(storage, permit, "caches_complete")
    ledger = (
        storage / "trustsr" / "phase2b3c" / "access-ledger" / permit.evaluation_id
    )
    (ledger / "001-pixels_opened.json").unlink()
    with pytest.raises(ValueError, match="contiguous"):
        load_access_ledger(storage, permit)

    drift_source = tmp_path / "drift-source"
    drift_source.mkdir()
    advance_access_ledger(drift_source, permit, "reserved")
    other = tmp_path / "other-storage"
    other.mkdir()
    shutil.copytree(drift_source / "trustsr", other / "trustsr")
    with pytest.raises(ValueError, match="storage root"):
        load_access_ledger(other, permit)


def test_invalidation_is_terminal(tmp_path: Path) -> None:
    permit = _verified_permit(tmp_path)
    storage = tmp_path / "storage"
    storage.mkdir()
    advance_access_ledger(storage, permit, "reserved")
    advance_access_ledger(storage, permit, "pixels_opened")
    invalidated = advance_access_ledger(storage, permit, "invalidated")
    assert invalidated.state == "invalidated"
    with pytest.raises(ValueError, match="terminal"):
        advance_access_ledger(storage, permit, "caches_complete")
    with pytest.raises(ValueError, match="invalidated"):
        invalidated.access_guard()


def test_lock_contention_fails_without_waiting(tmp_path: Path) -> None:
    permit = _verified_permit(tmp_path)
    storage = tmp_path / "storage"
    storage.mkdir()
    with phase2b3c_access_lock(storage, permit.evaluation_id):
        with pytest.raises(RuntimeError, match="lock"):
            with phase2b3c_access_lock(storage, permit.evaluation_id):
                pytest.fail("contended lock was acquired")


def test_partial_write_failure_leaves_no_event_or_staging_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    permit = _verified_permit(tmp_path)
    storage = tmp_path / "storage"
    storage.mkdir()
    real_link = os.link

    def fail_link(*args: object, **kwargs: object) -> None:
        raise OSError("injected link failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(ValueError, match="publish ledger event"):
        advance_access_ledger(storage, permit, "reserved")
    monkeypatch.setattr(os, "link", real_link)
    ledger = (
        storage / "trustsr" / "phase2b3c" / "access-ledger" / permit.evaluation_id
    )
    if ledger.exists():
        assert not list(ledger.glob("*.json"))
        assert not list(ledger.glob("*.partial"))


def test_rejects_event_content_tampering(tmp_path: Path) -> None:
    permit = _verified_permit(tmp_path)
    storage = tmp_path / "storage"
    storage.mkdir()
    advance_access_ledger(storage, permit, "reserved")
    ledger = (
        storage / "trustsr" / "phase2b3c" / "access-ledger" / permit.evaluation_id
    )
    event = ledger / "000-reserved.json"
    document = json.loads(event.read_bytes())
    document["binding"]["environment_sha256"] = "0" * 64
    event.write_bytes(canonical_json(document))

    with pytest.raises(ValueError, match="binding"):
        load_access_ledger(storage, permit)
