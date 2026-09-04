"""Independent metadata-only verification of Phase 2B3-C bundles."""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import replace
from pathlib import Path

import pytest
import test_phase2b3c_computation_verify as computation_fixtures
import test_phase2b3c_workflow as workflow_fixtures

from trustsr.evaluation.phase2b3c_bundle import (
    build_phase2b3c_ledger_snapshot,
    read_phase2b3c_bundle,
    write_phase2b3c_bundle,
)
from trustsr.evaluation.phase2b3c_replay_receipt import (
    build_phase2b3c_replay_receipt,
)
from trustsr.jsonio import canonical_json


def _module():
    return importlib.import_module("trustsr.evaluation.phase2b3c_bundle_verify")


def _bundle(tmp_path: Path):
    candidate = computation_fixtures._candidate(tmp_path / "scores")
    result = __import__("json").loads(candidate["result"])
    audit = __import__("json").loads(candidate["audit"])
    runtime = __import__("json").loads(candidate["runtime"])
    replay = build_phase2b3c_replay_receipt(
        candidate["result"],
        candidate["audit"],
        candidate["runtime"],
        rebuilt_result=candidate["result"],
        rebuilt_cache_audit=candidate["audit"],
    )
    ledger_document = build_phase2b3c_ledger_snapshot(
        evaluation_id=result["evaluation"]["evaluation_id"],
        permit_sha256=result["evaluation"]["permit_sha256"],
        state="caches_complete",
        sequence=2,
        event_sha256=result["evaluation"]["ledger_event_sha256"],
        previous_event_sha256=workflow_fixtures._snapshot(
            "caches_complete"
        ).previous_event_sha256,
    )
    directory = tmp_path / "bundle"
    write_phase2b3c_bundle(
        directory,
        result=result,
        cache_audit=audit,
        runtime=runtime,
        replay=replay,
        ledger_snapshot=ledger_document,
    )
    loaded = read_phase2b3c_bundle(directory)
    permit = workflow_fixtures._permit()
    ledger = replace(
        workflow_fixtures._snapshot("bundle_complete"),
        previous_event_sha256=result["evaluation"]["ledger_event_sha256"],
    )
    authority = {
        "revision": result["producer_revision"],
        "head_revision": result["producer_revision"],
        "dependencies": runtime["dependencies"],
        "records": tuple(computation_fixtures._record(i) for i in range(120)),
        "ordered_sample_ids_sha256": result["digests"][
            "ordered_sample_ids_sha256"
        ],
        "ordered_membership_sha256": result["digests"][
            "ordered_membership_sha256"
        ],
    }
    return candidate, directory, loaded, permit, ledger, authority


def _authority_capability(module, permit, ledger, authority):
    return module.VerifiedPhase2B3CMetadataAuthority._from_verified(
        permit=permit,
        ledger=ledger,
        revision=authority["revision"],
        head_revision=authority["head_revision"],
        dependencies=authority["dependencies"],
        records=authority["records"],
        ordered_sample_ids_sha256=authority["ordered_sample_ids_sha256"],
        ordered_membership_sha256=authority["ordered_membership_sha256"],
    )


def test_cross_binds_all_six_documents_permit_ledger_membership_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    candidate, directory, loaded, permit, ledger, authority = _bundle(tmp_path)
    calls: list[dict[str, object]] = []

    capability = _authority_capability(module, permit, ledger, authority)

    def metadata(**kwargs: object):
        calls.append(kwargs)
        return capability

    monkeypatch.setattr(module, "load_phase2b3c_metadata_authority", metadata)

    receipt = module.verify_phase2b3c_bundle(
        directory,
        project_root=tmp_path / "project",
        evidence_dir=tmp_path / "evidence",
        storage_root=tmp_path / "storage",
        manifest_path=tmp_path / "manifest.jsonl",
        access_permit_path=tmp_path / "permit.json",
    )

    payloads = dict(loaded.payloads)
    assert receipt.manifest_sha256 == loaded.manifest_sha256
    assert receipt.result_sha256 == hashlib.sha256(
        payloads["phase2b3c-evaluation-result.json"]
    ).hexdigest()
    assert receipt.cache_audit_sha256 == hashlib.sha256(
        payloads["phase2b3c-evaluation-cache-audit.json"]
    ).hexdigest()
    assert receipt.runtime_sha256 == hashlib.sha256(
        payloads["phase2b3c-evaluation-runtime.json"]
    ).hexdigest()
    assert receipt.replay_sha256 == hashlib.sha256(
        payloads["phase2b3c-evaluation-replay.json"]
    ).hexdigest()
    assert receipt.ledger_snapshot_sha256 == hashlib.sha256(
        payloads["phase2b3c-access-ledger-snapshot.json"]
    ).hexdigest()
    assert receipt.bundle_complete_event_sha256 == ledger.event_sha256
    assert receipt.model_identity_sha256 == __import__("json").loads(
        candidate["runtime"]
    )["model_inventory"]["scientific_identity_sha256"]
    assert receipt.cache_computation_verified is False
    assert receipt.acceptance_authorized is False
    assert calls == [
        {
            "project_root": tmp_path / "project",
            "evidence_dir": tmp_path / "evidence",
            "storage_root": tmp_path / "storage",
            "manifest_path": tmp_path / "manifest.jsonl",
            "access_permit_path": tmp_path / "permit.json",
        }
    ]


@pytest.mark.parametrize(
    "fault", ("permit", "ledger_state", "ledger_link", "revision", "membership", "environment")
)
def test_rejects_each_external_authority_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    module = _module()
    _candidate, directory, _loaded, permit, ledger, authority = _bundle(tmp_path)
    authority = dict(authority)
    if fault == "permit":
        permit = replace(permit, permit_sha256="f" * 64)
    elif fault == "ledger_state":
        ledger = replace(ledger, state="caches_complete", sequence=2)
    elif fault == "ledger_link":
        ledger = replace(ledger, previous_event_sha256="f" * 64)
    elif fault == "revision":
        authority["revision"] = "f" * 40
    elif fault == "membership":
        authority["ordered_membership_sha256"] = "f" * 64
    else:
        authority["dependencies"] = {
            **authority["dependencies"],
            "python": "3.13",
        }
    capability = _authority_capability(module, permit, ledger, authority)
    monkeypatch.setattr(
        module,
        "load_phase2b3c_metadata_authority",
        lambda **kwargs: capability,
    )

    with pytest.raises(ValueError, match="permit|ledger|membership|dependencies"):
        module.verify_phase2b3c_bundle(
            directory,
            project_root=tmp_path,
            evidence_dir=tmp_path,
            storage_root=tmp_path,
            manifest_path=tmp_path / "manifest.jsonl",
            access_permit_path=tmp_path / "permit.json",
        )


def test_bundle_verifier_surface_is_metadata_only() -> None:
    module = _module()

    for forbidden in (
        "load_internal_test_pairs",
        "probe_cached_internal_test_bundles",
        "verify_phase2b3c_computation",
        "LDSRS2X4",
        "cuda",
    ):
        assert forbidden not in module.__dict__
    assert "cannot prove cache pixels" in module.VerifiedPhase2B3CBundle.__doc__
    assert "without reading pixels" in module.verify_phase2b3c_bundle.__doc__


def test_receipt_cannot_be_directly_constructed() -> None:
    with pytest.raises(TypeError, match="created only by"):
        _module().VerifiedPhase2B3CBundle()


def test_bundle_document_snapshot_is_canonical(tmp_path: Path) -> None:
    _candidate, _directory, loaded, _permit, _ledger, _authority = _bundle(tmp_path)

    for _name, payload in loaded.payloads:
        assert canonical_json(__import__("json").loads(payload)) == payload
