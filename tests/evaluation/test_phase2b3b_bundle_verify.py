"""Semantic verification for complete Phase 2B3-B candidate bundles."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import test_phase2b3b_result as result_fixtures

from trustsr.evaluation import (
    phase2b3b_bundle_verify,
    phase2b3b_result,
    phase2b3b_result_verify,
)
from trustsr.evaluation.phase2b3b_bundle import write_phase2b3b_bundle
from trustsr.evaluation.phase2b3b_result_verify import VerifiedPhase2B3BResult
from trustsr.jsonio import canonical_json

_RESULT_SCHEMA = "trustsr.phase2b3b-calibration.v1"
_AUDIT_SCHEMA = "trustsr.phase2b3b-calibration-cache-audit.v1"
_RUNTIME_SCHEMA = "trustsr.phase2b3b-calibration-runtime.v1"
_REPLAY_SCHEMA = "trustsr.phase2b3b-calibration-replay.v1"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _documents() -> dict[str, dict[str, object]]:
    result = {"schema": _RESULT_SCHEMA, "candidate": "result"}
    audit = {"schema": _AUDIT_SCHEMA, "candidate": "audit"}
    runtime = {"schema": _RUNTIME_SCHEMA, "runtime": "recorded-only"}
    replay = {
        "schema": _REPLAY_SCHEMA,
        "byte_identical": True,
        "result_sha256": _sha(canonical_json(result)),
        "cache_audit_sha256": _sha(canonical_json(audit)),
        "runtime_manifest_sha256": _sha(canonical_json(runtime)),
    }
    return {
        "result": result,
        "audit": audit,
        "runtime": runtime,
        "replay": replay,
    }


def _write(directory: Path, documents: dict[str, dict[str, object]]) -> None:
    write_phase2b3b_bundle(
        directory,
        result=documents["result"],
        cache_audit=documents["audit"],
        runtime=documents["runtime"],
        replay=documents["replay"],
    )


def _result_receipt(documents: dict[str, dict[str, object]]) -> VerifiedPhase2B3BResult:
    return VerifiedPhase2B3BResult(
        schema="trustsr.phase2b3b-calibration-result-verification.v1",
        result_sha256=_sha(canonical_json(documents["result"])),
        cache_audit_sha256=_sha(canonical_json(documents["audit"])),
        producer_revision="c" * 40,
        ordered_sample_ids_sha256="1" * 64,
        ordered_membership_sha256="2" * 64,
        input_receipt_sha256="3" * 64,
        ordered_inputs_sha256="4" * 64,
        map_evidence_sha256="5" * 64,
        radiometry_aggregate_sha256="6" * 64,
        phase_decision="freeze_calibration",
    )


def _records(sample_ids: tuple[str, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "sample_id": membership["sample_id"],
            "selection_sha256": membership["selection_sha256"],
            "spatial_group_id": membership["spatial_group_id"],
            "split": "calibration",
            "days_between": membership["days_between"],
            "correlation_bin": membership["correlation_bin"],
            "selection_round": membership["selection_round"],
            "lr_asset": {"sha256": membership["lr_asset_sha256"]},
            "hr_asset": {"sha256": membership["hr_asset_sha256"]},
        }
        for membership in result_fixtures._membership(sample_ids)
    )


def test_verifies_semantics_from_reader_raw_bytes_and_forwards_trusted_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "bundle"
    documents = _documents()
    expected = _result_receipt(documents)
    _write(directory, documents)
    calls: list[tuple[object, ...]] = []

    def verify(result: object, audit: object, **paths: object) -> object:
        calls.append((result, audit, paths))
        return expected

    monkeypatch.setattr(phase2b3b_bundle_verify, "verify_phase2b3b_result", verify)

    receipt = phase2b3b_bundle_verify.verify_phase2b3b_bundle(
        directory,
        project_root=tmp_path,
        evidence_dir=tmp_path / "evidence",
        storage_root=tmp_path / "storage",
        manifest_path=tmp_path / "post.json",
    )

    assert len(calls) == 1
    assert calls[0][0:2] == (documents["result"], documents["audit"])
    assert calls[0][2] == {
        "project_root": tmp_path,
        "evidence_dir": tmp_path / "evidence",
        "storage_root": tmp_path / "storage",
        "manifest_path": tmp_path / "post.json",
    }
    assert receipt.schema == "trustsr.phase2b3b-bundle-verification.v1"
    assert receipt.result_sha256 == expected.result_sha256
    assert receipt.cache_audit_sha256 == expected.cache_audit_sha256
    assert receipt.runtime_manifest_sha256 == documents["replay"][
        "runtime_manifest_sha256"
    ]
    assert receipt.producer_revision == "c" * 40
    assert receipt.phase_decision == "freeze_calibration"
    assert str(directory) not in repr(receipt)
    with pytest.raises(FrozenInstanceError):
        receipt.phase_decision = "accept"  # type: ignore[misc]


def test_calls_real_result_verifier_with_forwarded_authority_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_ids = result_fixtures._sample_ids()
    preflight = result_fixtures._preflight(sample_ids)
    audit = result_fixtures._audit(sample_ids)
    result = phase2b3b_result.build_phase2b3b_result(
        preflight,
        result_fixtures._input_receipt(sample_ids),
        result_fixtures._fit(sample_ids),
        audit,
        result_fixtures._radiometry(sample_ids),
        result_fixtures._revision(),
    )
    runtime = {"schema": _RUNTIME_SCHEMA}
    documents = {
        "result": result,
        "audit": audit,
        "runtime": runtime,
        "replay": {
            "schema": _REPLAY_SCHEMA,
            "byte_identical": True,
            "result_sha256": _sha(canonical_json(result)),
            "cache_audit_sha256": _sha(canonical_json(audit)),
            "runtime_manifest_sha256": _sha(canonical_json(runtime)),
        },
    }
    directory = tmp_path / "bundle"
    _write(directory, documents)
    calls: list[tuple[object, ...]] = []

    def load_preflight(*args: object) -> object:
        calls.append(("preflight", *args))
        return deepcopy(preflight)

    def load_records(*args: object) -> object:
        calls.append(("records", *args))
        return _records(sample_ids)

    def verify_revision(project_root: Path, revision: str) -> str:
        calls.append(("revision", project_root, revision))
        return revision

    monkeypatch.setattr(phase2b3b_result_verify, "load_phase2b3b_preflight", load_preflight)
    monkeypatch.setattr(phase2b3b_result_verify, "load_calibration_records", load_records)
    monkeypatch.setattr(
        phase2b3b_result_verify,
        "verify_recorded_phase2b3b_revision",
        verify_revision,
    )

    receipt = phase2b3b_bundle_verify.verify_phase2b3b_bundle(
        directory,
        project_root=tmp_path,
        evidence_dir=tmp_path / "evidence",
        storage_root=tmp_path / "storage",
        manifest_path=tmp_path / "post.json",
    )

    assert receipt.result_sha256 == _sha(canonical_json(result))
    assert calls == [
        (
            "preflight",
            tmp_path / "evidence",
            tmp_path / "storage",
            tmp_path / "post.json",
        ),
        ("records", tmp_path / "storage", tmp_path / "post.json"),
        ("revision", tmp_path, "c" * 40),
    ]


@pytest.mark.parametrize("fault", ("false", "result", "audit", "runtime"))
def test_rejects_false_or_non_byte_identical_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    directory = tmp_path / "bundle"
    documents = _documents()
    if fault == "false":
        documents["replay"]["byte_identical"] = False
    else:
        key = {
            "result": "result_sha256",
            "audit": "cache_audit_sha256",
            "runtime": "runtime_manifest_sha256",
        }[fault]
        documents["replay"][key] = "f" * 64
    _write(directory, documents)
    monkeypatch.setattr(
        phase2b3b_bundle_verify,
        "verify_phase2b3b_result",
        lambda *args, **kwargs: _result_receipt(documents),
    )

    with pytest.raises(ValueError, match="replay|byte"):
        phase2b3b_bundle_verify.verify_phase2b3b_bundle(
            directory,
            project_root=tmp_path,
            evidence_dir=tmp_path,
            storage_root=tmp_path,
            manifest_path=tmp_path / "post.json",
        )


def test_rejects_extra_replay_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "bundle"
    documents = _documents()
    documents["replay"]["accepted"] = True
    _write(directory, documents)
    monkeypatch.setattr(
        phase2b3b_bundle_verify,
        "verify_phase2b3b_result",
        lambda *args, **kwargs: _result_receipt(documents),
    )

    with pytest.raises(ValueError, match="replay"):
        phase2b3b_bundle_verify.verify_phase2b3b_bundle(
            directory,
            project_root=tmp_path,
            evidence_dir=tmp_path,
            storage_root=tmp_path,
            manifest_path=tmp_path / "post.json",
        )


def test_minimal_result_is_rejected_without_mocking_semantic_verifier(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "bundle"
    _write(directory, _documents())

    with pytest.raises(ValueError, match="result|JSON object"):
        phase2b3b_bundle_verify.verify_phase2b3b_bundle(
            directory,
            project_root=tmp_path,
            evidence_dir=tmp_path,
            storage_root=tmp_path,
            manifest_path=tmp_path / "post.json",
        )


@pytest.mark.parametrize("fault", ("extra", "symlink"))
def test_reader_rejects_extra_or_symlinked_bundle_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    directory = tmp_path / "bundle"
    documents = _documents()
    _write(directory, documents)
    if fault == "extra":
        (directory / "extra.json").write_text("{}", encoding="utf-8")
    else:
        runtime = directory / "phase2b3b-calibration-runtime.json"
        runtime.unlink()
        runtime.symlink_to(directory / "phase2b3b-bundle-manifest.json")
    monkeypatch.setattr(
        phase2b3b_bundle_verify,
        "verify_phase2b3b_result",
        lambda *args, **kwargs: _result_receipt(documents),
    )

    with pytest.raises(ValueError):
        phase2b3b_bundle_verify.verify_phase2b3b_bundle(
            directory,
            project_root=tmp_path,
            evidence_dir=tmp_path,
            storage_root=tmp_path,
            manifest_path=tmp_path / "post.json",
        )
