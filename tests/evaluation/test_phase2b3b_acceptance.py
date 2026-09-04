"""Independent Phase 2B3-B acceptance and publication contracts."""

from __future__ import annotations

import hashlib
import importlib
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest
import test_phase2b3b_result as result_fixtures

from trustsr.evaluation import phase2b3b_workflow
from trustsr.evaluation.phase2b3b_bundle import (
    LoadedPhase2B3BBundle,
    read_phase2b3b_bundle,
    write_phase2b3b_bundle,
)
from trustsr.evaluation.phase2b3b_bundle_verify import VerifiedPhase2B3BBundle
from trustsr.evaluation.phase2b3b_computation_verify import VerifiedPhase2B3BComputation
from trustsr.evaluation.phase2b3b_result import build_phase2b3b_result
from trustsr.jsonio import canonical_json


def _module():
    return importlib.import_module("trustsr.evaluation.phase2b3b_acceptance")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _case(
    tmp_path: Path, *, all_abstain: bool = False, result_mutation=None
) -> tuple[LoadedPhase2B3BBundle, VerifiedPhase2B3BBundle, VerifiedPhase2B3BComputation]:
    sample_ids = result_fixtures._sample_ids()
    audit = result_fixtures._audit(sample_ids)
    result = build_phase2b3b_result(
        result_fixtures._preflight(sample_ids),
        result_fixtures._input_receipt(sample_ids),
        result_fixtures._fit(sample_ids, all_abstain=all_abstain),
        audit,
        result_fixtures._radiometry(sample_ids),
        result_fixtures._revision(),
    )
    if result_mutation is not None:
        result_mutation(result)
    runtime = {
        "schema": "trustsr.phase2b3b-calibration-runtime.v1",
        "model_inventory": {
            "identity": {"name": "ldsr-s2-x4", "scale": 4},
            "seeds": [3407, 3408, 3409, 3410, 3411],
        },
    }
    replay = {
        "schema": "trustsr.phase2b3b-calibration-replay.v1",
        "byte_identical": True,
        "result_sha256": _sha(canonical_json(result)),
        "cache_audit_sha256": _sha(canonical_json(audit)),
        "runtime_manifest_sha256": _sha(canonical_json(runtime)),
    }
    bundle_dir = tmp_path / ("bundle-stop" if all_abstain else "bundle-freeze")
    write_phase2b3b_bundle(
        bundle_dir,
        result=result,
        cache_audit=audit,
        runtime=runtime,
        replay=replay,
    )
    loaded = read_phase2b3b_bundle(bundle_dir)
    payloads = dict(loaded.payloads)
    metadata = VerifiedPhase2B3BBundle._from_verified(
        schema="trustsr.phase2b3b-candidate-bundle-metadata-verification.v1",
        verification_scope="metadata_consistency_only",
        cache_computation_verified=False,
        manifest_sha256=loaded.manifest_sha256,
        result_sha256=_sha(payloads["phase2b3b-calibration-result.json"]),
        cache_audit_sha256=_sha(payloads["phase2b3b-calibration-cache-audit.json"]),
        runtime_manifest_sha256=_sha(payloads["phase2b3b-calibration-runtime.json"]),
        replay_sha256=_sha(payloads["phase2b3b-calibration-replay.json"]),
        producer_revision=result["producer_revision"],
        ordered_sample_ids_sha256=result["upstream"]["ordered_sample_ids_sha256"],
        ordered_membership_sha256=result["upstream"]["ordered_membership_sha256"],
        input_receipt_sha256=result["input_receipt_sha256"],
        ordered_inputs_sha256=result["ordered_inputs_sha256"],
        map_evidence_sha256=result["map_evidence_sha256"],
        radiometry_aggregate_sha256="a" * 64,
        phase_decision=result["phase_decision"],
    )
    computation = VerifiedPhase2B3BComputation._from_verified(
        schema="trustsr.phase2b3b-calibration-computation-verification.v1",
        verification_scope="cache_computation_replay",
        cache_computation_verified=True,
        prediction_inference_verified=False,
        membership_authority_verified=False,
        acceptance_authorized=False,
        result_sha256=metadata.result_sha256,
        cache_audit_sha256=metadata.cache_audit_sha256,
        map_evidence_sha256=metadata.map_evidence_sha256,
    )
    return loaded, metadata, computation


@pytest.mark.parametrize(
    ("all_abstain", "decision", "has_frozen_payload"),
    (
        (False, "freeze_calibration", True),
        (True, "stop_insufficient_coverage", False),
    ),
)
def test_acceptance_requires_both_verifiers_and_preserves_the_observed_decision(
    tmp_path: Path,
    all_abstain: bool,
    decision: str,
    has_frozen_payload: bool,
) -> None:
    module = _module()
    loaded, metadata, computation = _case(tmp_path, all_abstain=all_abstain)

    verified_acceptance = module.build_phase2b3b_acceptance(
        loaded, metadata, computation
    )
    acceptance = verified_acceptance.as_dict()

    assert acceptance["schema"] == "trustsr.phase2b3b-calibration-acceptance.v1"
    assert acceptance["verification_scope"] == "independent_calibration_acceptance"
    assert acceptance["acceptance_authorized"] is True
    assert acceptance["target"] == {"alpha": 0.05, "minimum_coverage": 0.10}
    assert acceptance["phase_decision"] == decision
    assert (acceptance["frozen_calibration"] is not None) is has_frozen_payload
    assert acceptance["checks"] == {
        "bundle_integrity_pass": True,
        "metadata_authority_pass": True,
        "cache_computation_replay_pass": True,
        "byte_identical_replay_pass": True,
        "calibration_only_pass": True,
    }
    assert acceptance["digests"]["bundle_manifest_sha256"] == loaded.manifest_sha256
    assert canonical_json(acceptance) == canonical_json(deepcopy(acceptance))


def test_rejects_nonapproved_target_even_with_cross_bound_receipts(tmp_path: Path) -> None:
    module = _module()

    def mutate(result: dict[str, object]) -> None:
        result["target"]["alpha"] = 0.051

    loaded, metadata, computation = _case(tmp_path, result_mutation=mutate)

    with pytest.raises(ValueError, match="approved Phase 2B3-B"):
        module.build_phase2b3b_acceptance(loaded, metadata, computation)


def test_rejects_computation_receipt_that_differs_from_bundle_bytes(tmp_path: Path) -> None:
    module = _module()
    loaded, metadata, computation = _case(tmp_path)
    object.__setattr__(computation, "result_sha256", "f" * 64)

    with pytest.raises(ValueError, match="computation.*bundle"):
        module.build_phase2b3b_acceptance(loaded, metadata, computation)


@pytest.mark.parametrize("forged_layer", ("metadata", "computation"))
def test_rejects_forged_verification_receipt_types(
    tmp_path: Path, forged_layer: str
) -> None:
    module = _module()
    loaded, metadata, computation = _case(tmp_path)

    with pytest.raises(TypeError, match="exact .*verification receipt"):
        module.build_phase2b3b_acceptance(
            loaded,
            object() if forged_layer == "metadata" else metadata,
            object() if forged_layer == "computation" else computation,
        )


def test_verification_and_acceptance_receipts_cannot_be_directly_constructed() -> None:
    module = _module()

    with pytest.raises(TypeError, match="created only by"):
        VerifiedPhase2B3BBundle()
    with pytest.raises(TypeError, match="created only by"):
        VerifiedPhase2B3BComputation()
    with pytest.raises(TypeError, match="created only by"):
        module.VerifiedPhase2B3BAcceptance()


def test_publication_is_exact_atomic_and_idempotent(tmp_path: Path) -> None:
    module = _module()
    loaded, metadata, computation = _case(tmp_path)
    acceptance = module.build_phase2b3b_acceptance(loaded, metadata, computation)
    project_root = tmp_path / "project"
    (project_root / "artifacts").mkdir(parents=True)

    first = module.publish_phase2b3b_evidence(project_root, loaded, acceptance)
    second = module.publish_phase2b3b_evidence(project_root, loaded, acceptance)

    publication = project_root / "artifacts" / "phase2b3b"
    assert {path.name for path in publication.iterdir()} == {
        "sen2naipv2-calibration-conformal-v1.json",
        "sen2naipv2-calibration-conformal-cache-audit-v1.json",
        "sen2naipv2-calibration-conformal-acceptance-v1.json",
    }
    assert first.publication_sha256 == second.publication_sha256
    assert first.reused is False
    assert second.reused is True
    assert canonical_json(acceptance.as_dict()) == (
        publication / "sen2naipv2-calibration-conformal-acceptance-v1.json"
    ).read_bytes()


def test_publication_failure_leaves_no_partial_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    loaded, metadata, computation = _case(tmp_path)
    acceptance = module.build_phase2b3b_acceptance(loaded, metadata, computation)
    project_root = tmp_path / "project"
    artifacts = project_root / "artifacts"
    artifacts.mkdir(parents=True)
    real_write = module._write_publication_file
    calls = 0

    def fail_second(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        real_write(path, payload)

    monkeypatch.setattr(module, "_write_publication_file", fail_second)

    with pytest.raises(OSError, match="injected"):
        module.publish_phase2b3b_evidence(project_root, loaded, acceptance)

    assert not (artifacts / "phase2b3b").exists()
    assert not tuple(artifacts.glob(".phase2b3b.*"))


def test_existing_publication_with_different_bytes_fails_closed(tmp_path: Path) -> None:
    module = _module()
    loaded, metadata, computation = _case(tmp_path)
    acceptance = module.build_phase2b3b_acceptance(loaded, metadata, computation)
    project_root = tmp_path / "project"
    (project_root / "artifacts").mkdir(parents=True)
    module.publish_phase2b3b_evidence(project_root, loaded, acceptance)
    changed_document = acceptance.as_dict()
    changed_document["unexpected"] = True
    changed = object.__new__(module.VerifiedPhase2B3BAcceptance)
    object.__setattr__(changed, "payload", canonical_json(changed_document))
    object.__setattr__(changed, "phase_decision", changed_document["phase_decision"])

    with pytest.raises(ValueError, match="acceptance.*keys"):
        module.publish_phase2b3b_evidence(project_root, loaded, changed)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.update({"acceptance_authorized": False}),
        lambda value: value["checks"].update({"metadata_authority_pass": False}),
        lambda value: value["target"].update({"alpha": 0.051}),
        lambda value: value["digests"].update({"result_sha256": "f" * 64}),
        lambda value: value.update({"phase_decision": "stop_insufficient_coverage"}),
    ),
)
def test_publication_revalidates_acceptance_authority_and_bundle_binding(
    tmp_path: Path, mutation
) -> None:
    module = _module()
    loaded, metadata, computation = _case(tmp_path)
    document = module.build_phase2b3b_acceptance(
        loaded, metadata, computation
    ).as_dict()
    mutation(document)
    forged = object.__new__(module.VerifiedPhase2B3BAcceptance)
    object.__setattr__(forged, "payload", canonical_json(document))
    object.__setattr__(forged, "phase_decision", document["phase_decision"])
    project_root = tmp_path / "project"
    (project_root / "artifacts").mkdir(parents=True)

    with pytest.raises(ValueError):
        module.publish_phase2b3b_evidence(project_root, loaded, forged)

    assert not (project_root / "artifacts" / "phase2b3b").exists()


def test_publication_rejects_acceptance_built_for_a_different_bundle(
    tmp_path: Path,
) -> None:
    module = _module()
    loaded, metadata, computation = _case(tmp_path)
    acceptance = module.build_phase2b3b_acceptance(loaded, metadata, computation)
    other_loaded, _, _ = _case(tmp_path, all_abstain=True)
    project_root = tmp_path / "project"
    (project_root / "artifacts").mkdir(parents=True)

    with pytest.raises(ValueError, match="acceptance.*bundle"):
        module.publish_phase2b3b_evidence(project_root, other_loaded, acceptance)


def test_concurrent_identical_publication_commits_one_exact_directory(
    tmp_path: Path,
) -> None:
    module = _module()
    loaded, metadata, computation = _case(tmp_path)
    acceptance = module.build_phase2b3b_acceptance(loaded, metadata, computation)
    project_root = tmp_path / "project"
    (project_root / "artifacts").mkdir(parents=True)

    with ThreadPoolExecutor(max_workers=2) as executor:
        receipts = tuple(
            executor.map(
                lambda _: module.publish_phase2b3b_evidence(
                    project_root, loaded, acceptance
                ),
                range(2),
            )
        )

    assert {receipt.publication_sha256 for receipt in receipts} == {
        receipts[0].publication_sha256
    }
    assert sorted(receipt.reused for receipt in receipts) == [False, True]
    assert not tuple((project_root / "artifacts").glob(".phase2b3b.*"))


def test_independent_verifier_shares_the_formal_exclusive_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    storage_root = tmp_path / "storage"
    project_root = tmp_path / "project"
    storage_root.mkdir()
    project_root.mkdir()
    monkeypatch.setattr(phase2b3b_workflow, "_MINIMUM_FREE_BYTES", 0)
    monkeypatch.setattr(
        module,
        "verify_phase2b3b_revision",
        lambda root: result_fixtures._revision(),
    )
    monkeypatch.setattr(
        module,
        "verify_phase2b3b_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("metadata verification crossed a held formal lock")
        ),
    )
    paths = phase2b3b_workflow.validate_phase2b3b_storage(storage_root, True)

    with phase2b3b_workflow.phase2b3b_formal_lock(paths):
        with pytest.raises(RuntimeError, match="holds the lock"):
            module.run_independent_phase2b3b_verification(
                bundle_dir=tmp_path / "bundle",
                project_root=project_root,
                evidence_dir=tmp_path / "evidence",
                storage_root=storage_root,
                manifest_path=tmp_path / "manifest.jsonl",
                confirmed_persistent_storage=True,
            )


def test_independent_verifier_rejects_the_producer_bundle_before_authority_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    storage_root = tmp_path / "storage"
    project_root = tmp_path / "project"
    storage_root.mkdir()
    project_root.mkdir()
    monkeypatch.setattr(phase2b3b_workflow, "_MINIMUM_FREE_BYTES", 0)
    paths = phase2b3b_workflow.validate_phase2b3b_storage(storage_root, True)
    paths.bundle_dir.mkdir(parents=True)
    monkeypatch.setattr(
        module,
        "verify_phase2b3b_revision",
        lambda _: (_ for _ in ()).throw(
            AssertionError("authority read crossed the copied-bundle gate")
        ),
    )

    with pytest.raises(ValueError, match="copied bundle"):
        module.run_independent_phase2b3b_verification(
            bundle_dir=paths.bundle_dir,
            project_root=project_root,
            evidence_dir=tmp_path / "evidence",
            storage_root=storage_root,
            manifest_path=tmp_path / "manifest.jsonl",
            confirmed_persistent_storage=True,
        )
