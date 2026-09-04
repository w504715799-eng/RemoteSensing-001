"""Acceptance and publication authority for Phase 2B3-C."""

from __future__ import annotations

import hashlib
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_phase2b3c_bundle_verify as bundle_fixtures
import test_phase2b3c_computation_verify as computation_fixtures
import test_phase2b3c_result as result_fixtures
import test_phase2b3c_workflow as workflow_fixtures

from trustsr.evaluation import phase2b3c_bundle_verify, phase2b3c_workflow
from trustsr.evaluation.internal_test_predictions import (
    InternalTestPredictionCacheProbe,
)
from trustsr.evaluation.phase2b3c_bundle import (
    build_phase2b3c_ledger_snapshot,
    read_phase2b3c_bundle,
    write_phase2b3c_bundle,
)
from trustsr.evaluation.phase2b3c_bundle_verify import VerifiedPhase2B3CBundle
from trustsr.evaluation.phase2b3c_computation_verify import (
    VerifiedPhase2B3CComputation,
    verify_phase2b3c_computation,
)
from trustsr.evaluation.phase2b3c_replay_receipt import (
    build_phase2b3c_replay_receipt,
)
from trustsr.jsonio import canonical_json


def _module():
    return importlib.import_module("trustsr.evaluation.phase2b3c_acceptance")


def _sha(value: bytes | str) -> str:
    payload = value if type(value) is bytes else value.encode()
    return hashlib.sha256(payload).hexdigest()


def _case(tmp_path: Path, decision: str = "confirmed"):
    statistics = {
        "confirmed": result_fixtures._statistics(loss=0.0),
        "empirically_met_but_inconclusive": result_fixtures._statistics(loss=0.04),
        "failed": result_fixtures._statistics(loss=0.06),
    }[decision]
    result = result_fixtures._result(statistics)
    result["evaluation"]["permit_sha256"] = _sha(b"reviewed permit\n")
    audit = {
        "schema": "trustsr.phase2b3c-evaluation-cache-audit.v1",
        "test_fixture": decision,
    }
    runtime = {
        "schema": "trustsr.phase2b3c-evaluation-runtime.v1",
        "test_fixture": decision,
    }
    result_payload = canonical_json(result)
    audit_payload = canonical_json(audit)
    runtime_payload = canonical_json(runtime)
    replay = build_phase2b3c_replay_receipt(
        result_payload,
        audit_payload,
        runtime_payload,
        rebuilt_result=result_payload,
        rebuilt_cache_audit=audit_payload,
    )
    ledger = build_phase2b3c_ledger_snapshot(
        evaluation_id=result["evaluation"]["evaluation_id"],
        permit_sha256=result["evaluation"]["permit_sha256"],
        state="caches_complete",
        sequence=2,
        event_sha256=result["evaluation"]["ledger_event_sha256"],
        previous_event_sha256=_sha("pixels_opened"),
    )
    directory = tmp_path / f"bundle-{decision}"
    write_phase2b3c_bundle(
        directory,
        result=result,
        cache_audit=audit,
        runtime=runtime,
        replay=replay,
        ledger_snapshot=ledger,
    )
    loaded = read_phase2b3c_bundle(directory)
    payloads = dict(loaded.payloads)
    digests = result["digests"]
    metadata = VerifiedPhase2B3CBundle._from_verified(
        schema="trustsr.phase2b3c-candidate-bundle-metadata-verification.v1",
        verification_scope="metadata_consistency_only",
        cache_computation_verified=False,
        acceptance_authorized=False,
        manifest_sha256=loaded.manifest_sha256,
        result_sha256=_sha(payloads["phase2b3c-evaluation-result.json"]),
        cache_audit_sha256=_sha(
            payloads["phase2b3c-evaluation-cache-audit.json"]
        ),
        runtime_sha256=_sha(payloads["phase2b3c-evaluation-runtime.json"]),
        replay_sha256=_sha(payloads["phase2b3c-evaluation-replay.json"]),
        ledger_snapshot_sha256=_sha(
            payloads["phase2b3c-access-ledger-snapshot.json"]
        ),
        evaluation_id=result["evaluation"]["evaluation_id"],
        permit_sha256=result["evaluation"]["permit_sha256"],
        producer_revision=result["producer_revision"],
        ordered_sample_ids_sha256=digests["ordered_sample_ids_sha256"],
        ordered_membership_sha256=digests["ordered_membership_sha256"],
        input_receipt_sha256=digests["input_receipt_sha256"],
        ordered_inputs_sha256=digests["ordered_inputs_sha256"],
        map_evidence_sha256=digests["map_evidence_sha256"],
        model_identity_sha256=_sha("model"),
        bundle_complete_event_sha256=_sha("bundle_complete"),
        phase_decision=decision,
    )
    computation = VerifiedPhase2B3CComputation._from_verified(
        schema="trustsr.phase2b3c-evaluation-computation-verification.v1",
        verification_scope="cache_computation_replay",
        cache_computation_verified=True,
        prediction_inference_verified=False,
        membership_authority_verified=False,
        acceptance_authorized=False,
        result_sha256=metadata.result_sha256,
        cache_audit_sha256=metadata.cache_audit_sha256,
        runtime_sha256=metadata.runtime_sha256,
        map_evidence_sha256=metadata.map_evidence_sha256,
        phase_decision=decision,
    )
    return loaded, metadata, computation


@pytest.mark.parametrize(
    "decision", ("confirmed", "empirically_met_but_inconclusive", "failed")
)
def test_acceptance_requires_both_receipts_and_preserves_all_decisions(
    tmp_path: Path, decision: str
) -> None:
    module = _module()
    loaded, metadata, computation = _case(tmp_path, decision)

    receipt = module.build_phase2b3c_acceptance(
        loaded, metadata, computation, verifier_revision="2" * 40
    )
    value = receipt.as_dict()

    assert value["schema"] == "trustsr.phase2b3c-evaluation-acceptance.v1"
    assert value["verification_scope"] == "independent_internal_test_acceptance"
    assert value["acceptance_authorized"] is True
    assert value["phase_decision"] == decision
    assert value["one_time_access"]["terminal_state"] == "accepted"
    assert value["checks"]["byte_identical_replay_pass"] is True


@pytest.mark.parametrize("layer", ("metadata", "computation"))
def test_rejects_forged_receipt_types(tmp_path: Path, layer: str) -> None:
    module = _module()
    loaded, metadata, computation = _case(tmp_path)

    with pytest.raises(TypeError, match="exact .*verification receipt"):
        module.build_phase2b3c_acceptance(
            loaded,
            object() if layer == "metadata" else metadata,
            object() if layer == "computation" else computation,
            verifier_revision="2" * 40,
        )


def test_rejects_computation_receipt_for_different_bundle(tmp_path: Path) -> None:
    module = _module()
    loaded, metadata, computation = _case(tmp_path)
    object.__setattr__(computation, "result_sha256", "f" * 64)

    with pytest.raises(ValueError, match="computation.*bundle"):
        module.build_phase2b3c_acceptance(
            loaded, metadata, computation, verifier_revision="2" * 40
        )


@pytest.mark.parametrize("layer", ("metadata", "computation"))
def test_rejects_forged_instances_of_receipt_classes(
    tmp_path: Path, layer: str
) -> None:
    module = _module()
    loaded, metadata, computation = _case(tmp_path)
    original = metadata if layer == "metadata" else computation
    forged = object.__new__(type(original))
    for name in original.__dataclass_fields__:
        if name != "_authority":
            object.__setattr__(forged, name, getattr(original, name))

    with pytest.raises((TypeError, ValueError), match="receipt|verifier"):
        module.build_phase2b3c_acceptance(
            loaded,
            forged if layer == "metadata" else metadata,
            forged if layer == "computation" else computation,
            verifier_revision="2" * 40,
        )


def test_acceptance_uses_a_real_independent_synthetic_computation_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    candidate, directory, loaded, permit, ledger, authority = bundle_fixtures._bundle(
        tmp_path
    )
    monkeypatch.setattr(
        phase2b3c_bundle_verify,
        "load_phase2b3c_metadata_authority",
        lambda **kwargs: bundle_fixtures._authority_capability(
            phase2b3c_bundle_verify, permit, ledger, authority
        ),
    )
    metadata = phase2b3c_bundle_verify.verify_phase2b3c_bundle(
        directory,
        project_root=tmp_path,
        evidence_dir=tmp_path,
        storage_root=tmp_path,
        manifest_path=tmp_path / "manifest.jsonl",
        access_permit_path=tmp_path / "permit.json",
    )
    computation = verify_phase2b3c_computation(
        candidate["result"],
        candidate["audit"],
        candidate["runtime"],
        input_receipt=candidate["input_receipt"],
        pairs=candidate["pairs"],
        bundles=candidate["bundles"],
        dependencies=candidate["dependencies"],
    )

    acceptance = module.build_phase2b3c_acceptance(
        loaded,
        metadata,
        computation,
        verifier_revision="2" * 40,
    )

    assert acceptance.as_dict()["checks"]["cache_computation_replay_pass"] is True


def test_acceptance_independently_requires_byte_identical_replay(tmp_path: Path) -> None:
    module = _module()
    loaded, metadata, computation = _case(tmp_path)
    documents = loaded.documents()
    replay = dict(documents["phase2b3c-evaluation-replay.json"])
    replay["byte_identical"] = False
    changed_directory = tmp_path / "changed-replay"
    write_phase2b3c_bundle(
        changed_directory,
        result=documents["phase2b3c-evaluation-result.json"],
        cache_audit=documents["phase2b3c-evaluation-cache-audit.json"],
        runtime=documents["phase2b3c-evaluation-runtime.json"],
        replay=replay,
        ledger_snapshot=documents["phase2b3c-access-ledger-snapshot.json"],
    )
    changed = read_phase2b3c_bundle(changed_directory)
    changed_payloads = dict(changed.payloads)
    object.__setattr__(metadata, "manifest_sha256", changed.manifest_sha256)
    object.__setattr__(
        metadata,
        "replay_sha256",
        _sha(changed_payloads["phase2b3c-evaluation-replay.json"]),
    )

    with pytest.raises(ValueError, match="replay"):
        module.build_phase2b3c_acceptance(
            changed, metadata, computation, verifier_revision="2" * 40
        )


def _publication_case(tmp_path: Path):
    module = _module()
    loaded, metadata, computation = _case(tmp_path)
    acceptance = module.build_phase2b3c_acceptance(
        loaded, metadata, computation, verifier_revision="2" * 40
    )
    project = tmp_path / "project"
    publication = project / "artifacts" / "phase2b3c"
    publication.mkdir(parents=True)
    permit = publication / "sen2naipv2-internal-test-access-authorization-v1.json"
    permit.write_bytes(b"reviewed permit\n")
    return module, project, publication, permit, loaded, acceptance


def test_publication_preserves_permit_and_is_exact_and_idempotent(tmp_path: Path) -> None:
    module, project, publication, permit, loaded, acceptance = _publication_case(
        tmp_path
    )
    permit_bytes = permit.read_bytes()

    first = module.publish_phase2b3c_evidence(project, loaded, acceptance)
    second = module.publish_phase2b3c_evidence(project, loaded, acceptance)

    assert {item.name for item in publication.iterdir()} == {
        "sen2naipv2-internal-test-access-authorization-v1.json",
        *module._PUBLICATION_NAMES,
    }
    assert permit.read_bytes() == permit_bytes
    assert first.publication_sha256 == second.publication_sha256
    assert first.reused is False
    assert second.reused is True


def test_link_failure_rolls_back_three_files_without_altering_permit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, project, publication, permit, loaded, acceptance = _publication_case(
        tmp_path
    )
    permit_bytes = permit.read_bytes()
    real_link = module.os.link
    calls = 0

    def fail_second(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected link failure")
        real_link(*args, **kwargs)

    monkeypatch.setattr(module.os, "link", fail_second)

    with pytest.raises(OSError, match="injected"):
        module.publish_phase2b3c_evidence(project, loaded, acceptance)

    assert {item.name for item in publication.iterdir()} == {permit.name}
    assert permit.read_bytes() == permit_bytes
    assert not tuple((project / "artifacts").glob(".phase2b3c-publication.*"))


def test_rollback_does_not_delete_an_adversarial_same_byte_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, project, publication, permit, loaded, acceptance = _publication_case(
        tmp_path
    )
    real_link = module.os.link
    first_destination: Path | None = None
    attacker_inode: int | None = None
    calls = 0

    def replace_then_fail(source: str, destination: str, **kwargs: object) -> None:
        nonlocal attacker_inode, calls, first_destination
        calls += 1
        if calls == 1:
            real_link(source, destination, **kwargs)
            first_destination = publication / destination
            return
        assert first_destination is not None
        first_payload = first_destination.read_bytes()
        first_destination.unlink()
        first_destination.write_bytes(first_payload)
        attacker_inode = first_destination.stat().st_ino
        raise OSError("injected adversarial replacement")

    monkeypatch.setattr(module.os, "link", replace_then_fail)

    with pytest.raises(OSError, match="adversarial"):
        module.publish_phase2b3c_evidence(project, loaded, acceptance)

    assert first_destination is not None
    assert first_destination.exists()
    assert first_destination.stat().st_ino == attacker_inode
    assert permit.read_bytes() == b"reviewed permit\n"


def test_publication_collision_preserves_the_racing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, project, publication, permit, loaded, acceptance = _publication_case(
        tmp_path
    )
    real_link = module.os.link
    racing = publication / module.RESULT_PUBLICATION_NAME

    def collide(source: str, destination: str, **kwargs: object) -> None:
        racing.write_bytes(b"racing writer")
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(module.os, "link", collide)

    with pytest.raises(FileExistsError):
        module.publish_phase2b3c_evidence(project, loaded, acceptance)

    assert racing.read_bytes() == b"racing writer"
    assert {item.name for item in publication.iterdir()} == {permit.name, racing.name}


def test_full_preexisting_publication_with_different_bytes_fails_closed(
    tmp_path: Path
) -> None:
    module, project, publication, permit, loaded, acceptance = _publication_case(
        tmp_path
    )
    for name in module._PUBLICATION_NAMES:
        (publication / name).write_bytes(b"preexisting collision")

    with pytest.raises(ValueError, match="different bytes"):
        module.publish_phase2b3c_evidence(project, loaded, acceptance)

    assert permit.read_bytes() == b"reviewed permit\n"
    assert all(
        (publication / name).read_bytes() == b"preexisting collision"
        for name in module._PUBLICATION_NAMES
    )


def test_publication_rejects_a_symlinked_result_entry(tmp_path: Path) -> None:
    module, project, publication, permit, loaded, acceptance = _publication_case(
        tmp_path
    )
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    (publication / module.RESULT_PUBLICATION_NAME).symlink_to(outside)

    with pytest.raises(ValueError, match="partial|unsafe|different bytes"):
        module.publish_phase2b3c_evidence(project, loaded, acceptance)

    assert outside.read_bytes() == b"outside"
    assert permit.read_bytes() == b"reviewed permit\n"


def test_partial_publication_fails_closed_without_cleanup(tmp_path: Path) -> None:
    module, project, publication, permit, loaded, acceptance = _publication_case(
        tmp_path
    )
    partial = publication / module.RESULT_PUBLICATION_NAME
    partial.write_bytes(b"preexisting")

    with pytest.raises(ValueError, match="partial"):
        module.publish_phase2b3c_evidence(project, loaded, acceptance)

    assert partial.read_bytes() == b"preexisting"
    assert permit.read_bytes() == b"reviewed permit\n"


def test_concurrent_identical_publication_commits_one_set(tmp_path: Path) -> None:
    module, project, publication, _permit, loaded, acceptance = _publication_case(
        tmp_path
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        receipts = tuple(
            executor.map(
                lambda _: module.publish_phase2b3c_evidence(
                    project, loaded, acceptance
                ),
                range(2),
            )
        )

    assert sorted(receipt.reused for receipt in receipts) == [False, True]
    assert {item.name for item in publication.iterdir()} == {
        "sen2naipv2-internal-test-access-authorization-v1.json",
        *module._PUBLICATION_NAMES,
    }


def test_publication_revalidates_forged_acceptance(tmp_path: Path) -> None:
    module, project, publication, permit, loaded, acceptance = _publication_case(
        tmp_path
    )
    document = acceptance.as_dict()
    document["checks"]["metadata_authority_pass"] = False
    forged = object.__new__(module.VerifiedPhase2B3CAcceptance)
    object.__setattr__(forged, "payload", canonical_json(document))
    object.__setattr__(forged, "phase_decision", document["phase_decision"])

    with pytest.raises(ValueError, match="authority"):
        module.publish_phase2b3c_evidence(project, loaded, forged)

    assert {item.name for item in publication.iterdir()} == {permit.name}


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["target"].update({"alpha": 0.051}),
        lambda value: value["implementation"].update(
            {"producer_revision": "3" * 40}
        ),
        lambda value: value["implementation"].update(
            {"verifier_revision": "4" * 40}
        ),
        lambda value: value["one_time_access"].update(
            {"evaluation_id": "5" * 64}
        ),
        lambda value: value["one_time_access"].update(
            {"bundle_complete_event_sha256": "6" * 64}
        ),
        lambda value: value["digests"].update({"result_sha256": "7" * 64}),
        lambda value: value["digests"].update(
            {"access_permit_sha256": "8" * 64}
        ),
    ),
)
def test_publication_revalidates_every_acceptance_cross_binding(
    tmp_path: Path, mutation
) -> None:
    module, project, publication, permit, loaded, acceptance = _publication_case(
        tmp_path
    )
    document = acceptance.as_dict()
    mutation(document)
    object.__setattr__(acceptance, "payload", canonical_json(document))
    object.__setattr__(acceptance, "phase_decision", document["phase_decision"])

    with pytest.raises(ValueError, match="acceptance|authority|bundle|receipt"):
        module.publish_phase2b3c_evidence(project, loaded, acceptance)

    assert {item.name for item in publication.iterdir()} == {permit.name}


def test_acceptance_json_is_canonical_and_host_free(tmp_path: Path) -> None:
    module = _module()
    loaded, metadata, computation = _case(tmp_path)
    acceptance = module.build_phase2b3c_acceptance(
        loaded, metadata, computation, verifier_revision="2" * 40
    )

    assert canonical_json(json.loads(acceptance.payload)) == acceptance.payload
    for forbidden in (str(tmp_path), "hostname", "endpoint", "credential"):
        assert forbidden not in acceptance.payload.decode().casefold()


def test_independent_verifier_reacquires_the_formal_exclusive_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    storage = tmp_path / "storage"
    storage.mkdir()
    monkeypatch.setattr(phase2b3c_workflow, "_MINIMUM_FREE_BYTES", 0)
    paths = phase2b3c_workflow.validate_phase2b3c_storage(storage, True)

    with phase2b3c_workflow.phase2b3c_formal_lock(paths):
        with pytest.raises(RuntimeError, match="holds the formal lock"):
            module.run_independent_phase2b3c_verification(
                bundle_dir=tmp_path / "copied-bundle",
                project_root=tmp_path,
                evidence_dir=tmp_path / "evidence",
                storage_root=storage,
                manifest_path=tmp_path / "manifest.jsonl",
                access_permit_path=tmp_path / "permit.json",
                confirmed_persistent_storage=True,
            )


def test_independent_verifier_rejects_producer_bundle_before_authority_or_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    storage = tmp_path / "storage"
    storage.mkdir()
    monkeypatch.setattr(phase2b3c_workflow, "_MINIMUM_FREE_BYTES", 0)
    paths = phase2b3c_workflow.validate_phase2b3c_storage(storage, True)
    paths.bundle_dir.mkdir(parents=True)
    monkeypatch.setattr(
        module,
        "_run_locked_independent_verification",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("authority or pixel read crossed the copied-bundle gate")
        ),
    )

    with pytest.raises(ValueError, match="copied bundle"):
        module.run_independent_phase2b3c_verification(
            bundle_dir=paths.bundle_dir,
            project_root=tmp_path,
            evidence_dir=tmp_path / "evidence",
            storage_root=storage,
            manifest_path=tmp_path / "manifest.jsonl",
            access_permit_path=tmp_path / "permit.json",
            confirmed_persistent_storage=True,
        )


def _install_verifier_orchestration(
    module: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    publication_fails: bool = False,
):
    storage = tmp_path / "storage"
    storage.mkdir()
    paths = phase2b3c_workflow.phase2b3c_storage_paths(storage)
    paths.prediction_cache_dir.mkdir(parents=True)
    copied_bundle = tmp_path / "copied-bundle"
    copied_bundle.mkdir()
    permit = workflow_fixtures._permit()
    ledger = workflow_fixtures._snapshot("bundle_complete")
    metadata = SimpleNamespace(bundle_complete_event_sha256=ledger.event_sha256)
    records = tuple(computation_fixtures._record(index) for index in range(120))
    pairs = tuple(object() for _ in range(120))
    bundles = tuple(object() for _ in range(120))
    authority = phase2b3c_bundle_verify.VerifiedPhase2B3CMetadataAuthority._from_verified(
        permit=permit,
        ledger=ledger,
        revision="1" * 40,
        head_revision="2" * 40,
        dependencies=computation_fixtures._dependencies(),
        records=records,
        ordered_sample_ids_sha256=computation_fixtures._sha("sample ids"),
        ordered_membership_sha256=computation_fixtures._sha("membership"),
    )
    payloads = tuple(
        (name, name.encode())
        for name in (
            "phase2b3c-evaluation-result.json",
            "phase2b3c-evaluation-cache-audit.json",
            "phase2b3c-evaluation-runtime.json",
        )
    )
    loaded = SimpleNamespace(payloads=payloads)
    computation = object()
    acceptance = object()
    publication = module.Phase2B3CPublicationReceipt(
        publication_sha256="0" * 64,
        result_sha256="1" * 64,
        cache_audit_sha256="2" * 64,
        acceptance_sha256="3" * 64,
        phase_decision="confirmed",
        reused=False,
    )
    events: list[str] = []

    monkeypatch.setattr(
        module,
        "verify_phase2b3c_bundle",
        lambda *args, **kwargs: events.append("metadata") or metadata,
    )
    monkeypatch.setattr(
        module,
        "load_phase2b3c_metadata_authority",
        lambda **kwargs: events.append("authority") or authority,
    )
    monkeypatch.setattr(
        module,
        "load_internal_test_pairs",
        lambda root, loaded_records, guard: events.append("pairs") or pairs,
    )
    monkeypatch.setattr(
        module,
        "build_internal_test_input_receipt",
        lambda loaded_records, loaded_pairs: events.append("input_receipt") or {},
    )
    monkeypatch.setattr(
        module,
        "probe_cached_internal_test_bundles",
        lambda loaded_pairs, cache: events.append("cache_probe")
        or InternalTestPredictionCacheProbe(bundles, 600, 0),
    )
    monkeypatch.setattr(
        module,
        "read_phase2b3c_bundle",
        lambda path: events.append("bundle_snapshot") or loaded,
    )
    monkeypatch.setattr(
        module,
        "verify_phase2b3c_computation",
        lambda *args, **kwargs: events.append("computation") or computation,
    )
    monkeypatch.setattr(
        module,
        "build_phase2b3c_acceptance",
        lambda *args, **kwargs: events.append("acceptance") or acceptance,
    )

    def publish(*args: object, **kwargs: object):
        events.append("publication")
        if publication_fails:
            raise OSError("injected publication failure")
        return publication

    monkeypatch.setattr(module, "publish_phase2b3c_evidence", publish)

    def advance(root: Path, verified_permit: object, state: str):
        events.append(f"ledger:{state}")
        return replace(
            ledger,
            state="accepted",
            sequence=4,
            event_sha256=computation_fixtures._sha("accepted"),
            previous_event_sha256=ledger.event_sha256,
        )

    monkeypatch.setattr(module, "advance_access_ledger", advance)
    return paths, copied_bundle, events


def test_locked_verifier_requires_complete_cache_and_publishes_before_accepting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    paths, copied_bundle, events = _install_verifier_orchestration(
        module, tmp_path, monkeypatch
    )

    receipt = module._run_locked_independent_verification(
        paths=paths,
        bundle_dir=copied_bundle,
        project_root=tmp_path,
        evidence_dir=tmp_path / "evidence",
        manifest_path=tmp_path / "manifest.jsonl",
        access_permit_path=tmp_path / "permit.json",
    )

    assert receipt.publication.phase_decision == "confirmed"
    assert events == [
        "authority",
        "metadata",
        "pairs",
        "input_receipt",
        "cache_probe",
        "bundle_snapshot",
        "computation",
        "acceptance",
        "publication",
        "ledger:accepted",
    ]
    for forbidden in ("LDSRS2X4", "_load_ldsr", "cuda"):
        assert forbidden not in module.__dict__


def test_publication_failure_does_not_advance_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    paths, copied_bundle, events = _install_verifier_orchestration(
        module, tmp_path, monkeypatch, publication_fails=True
    )

    with pytest.raises(OSError, match="publication failure"):
        module._run_locked_independent_verification(
            paths=paths,
            bundle_dir=copied_bundle,
            project_root=tmp_path,
            evidence_dir=tmp_path / "evidence",
            manifest_path=tmp_path / "manifest.jsonl",
            access_permit_path=tmp_path / "permit.json",
        )

    assert events[-1] == "publication"
    assert "ledger:accepted" not in events


def test_locked_verifier_rejects_a_nonexact_cache_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    paths, copied_bundle, events = _install_verifier_orchestration(
        module, tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        module,
        "probe_cached_internal_test_bundles",
        lambda pairs, cache: events.append("cache_probe")
        or SimpleNamespace(
            bundles=tuple(object() for _ in range(120)),
            present_count=599,
            missing_count=1,
        ),
    )

    with pytest.raises(ValueError, match="complete exact K5 cache"):
        module._run_locked_independent_verification(
            paths=paths,
            bundle_dir=copied_bundle,
            project_root=tmp_path,
            evidence_dir=tmp_path / "evidence",
            manifest_path=tmp_path / "manifest.jsonl",
            access_permit_path=tmp_path / "permit.json",
        )

    assert "bundle_snapshot" not in events
