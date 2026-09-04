"""Complete synthetic CPU-only Phase 2B3-C evaluation and publication."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import test_phase2b3c_computation_verify as fixtures
import test_phase2b3c_workflow as workflow_fixtures
import torch

from trustsr.artifacts.predictions import tensor_sha256
from trustsr.data.internal_test_subset import select_internal_test_records
from trustsr.evaluation import (
    phase2b3c_acceptance,
    phase2b3c_bundle_verify,
    phase2b3c_workflow,
)
from trustsr.evaluation.internal_test_predictions import (
    InternalTestPredictionCacheProbe,
)
from trustsr.evaluation.phase2b3c_access import AccessLedgerSnapshot
from trustsr.evaluation.phase2b3c_bundle import read_phase2b3c_bundle
from trustsr.evaluation.phase2b3c_policy_scan import (
    phase2b3c_publication_policy_violations,
)
from trustsr.jsonio import canonical_json


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _complete_manifest() -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for split_index, split in enumerate(
        ("calibration", "development", "internal_test")
    ):
        for index in range(120):
            record = fixtures._record(index)
            if split != "internal_test":
                identity = (split_index + 1) * 120 + index
                record = {
                    **record,
                    "sample_id": f"{split}-{index:03d}",
                    "selection_sha256": fixtures._sha(f"selection:{identity}"),
                    "spatial_group_id": fixtures._sha(f"group:{identity}"),
                    "split": split,
                }
            records.append(record)
    return tuple(sorted(records, key=lambda item: str(item["sample_id"])))


def _bundles_for_loss(candidate: dict[str, object], loss: float):
    value = 0.5 - loss
    rebuilt = []
    for bundle in candidate["bundles"]:
        items = []
        for item in bundle.items:
            tensor = torch.full((4, 12, 12), value, dtype=torch.float32)
            items.append(
                replace(
                    item,
                    prediction_sha256=tensor_sha256(tensor),
                    tensor=tensor,
                )
            )
        rebuilt.append(replace(bundle, items=tuple(items)))
    return tuple(rebuilt)


def _snapshot(
    state: str,
    *,
    permit_sha256: str,
    previous_event_sha256: str | None,
) -> AccessLedgerSnapshot:
    sequence = {
        "pixels_opened": 1,
        "caches_complete": 2,
        "bundle_complete": 3,
        "accepted": 4,
    }[state]
    return replace(
        workflow_fixtures._snapshot(
            "bundle_complete" if state == "accepted" else state
        ),
        state=state,
        sequence=sequence,
        event_sha256=fixtures._sha(state),
        previous_event_sha256=previous_event_sha256,
        permit_sha256=permit_sha256,
    )


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("loss", "decision"),
    (
        (0.0, "confirmed"),
        (0.04, "empirically_met_but_inconclusive"),
        (0.06, "failed"),
    ),
)
def test_synthetic_evaluate_replay_copied_verify_and_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loss: float,
    decision: str,
) -> None:
    records = select_internal_test_records(_complete_manifest())
    assert len(records) == 120
    candidate = fixtures._candidate(tmp_path / "candidate-scores")
    assert tuple(record["sample_id"] for record in records) == tuple(
        pair.pair.sample_id for pair in candidate["pairs"]
    )
    bundles = _bundles_for_loss(candidate, loss)

    project = tmp_path / "project"
    project.mkdir()
    _git(project, "init")
    publication = project / "artifacts" / "phase2b3c"
    publication.mkdir(parents=True)
    permit_payload = canonical_json(
        {
            "schema": "trustsr.phase2b3c-internal-test-access-authorization.v1",
            "synthetic_fixture": True,
        }
    )
    permit_path = publication / phase2b3c_acceptance._PERMIT_PUBLICATION_NAME
    permit_path.write_bytes(permit_payload)
    permit = replace(
        workflow_fixtures._permit(),
        permit_sha256=_sha_bytes(permit_payload),
    )
    authority = phase2b3c_workflow.Phase2B3CAuthority(
        revision=phase2b3c_workflow.VerifiedRevision(
            "main", "2" * 40, "1" * 40, fixtures._sha("tree")
        ),
        records=records,
        permit=permit,
        dependencies=fixtures._dependencies(),
    )
    pixels = _snapshot(
        "pixels_opened",
        permit_sha256=permit.permit_sha256,
        previous_event_sha256=fixtures._sha("reserved"),
    )
    caches = _snapshot(
        "caches_complete",
        permit_sha256=permit.permit_sha256,
        previous_event_sha256=pixels.event_sha256,
    )
    bundle_complete = _snapshot(
        "bundle_complete",
        permit_sha256=permit.permit_sha256,
        previous_event_sha256=caches.event_sha256,
    )
    accepted = _snapshot(
        "accepted",
        permit_sha256=permit.permit_sha256,
        previous_event_sha256=bundle_complete.event_sha256,
    )
    storage = tmp_path / "storage"
    storage.mkdir()
    monkeypatch.setattr(phase2b3c_workflow, "_MINIMUM_FREE_BYTES", 0)
    monkeypatch.setattr(
        phase2b3c_workflow, "_prepare_authority", lambda **kwargs: authority
    )
    monkeypatch.setattr(
        phase2b3c_workflow,
        "_open_authorized_access",
        lambda paths, verified_permit: pixels,
    )
    monkeypatch.setattr(
        phase2b3c_workflow,
        "load_internal_test_pairs",
        lambda root, selected, guard: candidate["pairs"],
    )
    monkeypatch.setattr(
        phase2b3c_workflow,
        "probe_cached_internal_test_bundles",
        lambda pairs, cache: InternalTestPredictionCacheProbe(bundles, 600, 0),
    )
    monkeypatch.setattr(
        phase2b3c_workflow,
        "advance_access_ledger",
        lambda root, verified_permit, state: {
            "caches_complete": caches,
            "bundle_complete": bundle_complete,
        }[state],
    )
    monkeypatch.setattr(
        phase2b3c_workflow,
        "_load_ldsr",
        lambda path: (_ for _ in ()).throw(
            AssertionError("synthetic complete-cache path constructed LDSR")
        ),
    )

    produced = phase2b3c_workflow.run_formal_evaluation(
        project_root=project,
        evidence_dir=tmp_path / "evidence",
        storage_root=storage,
        manifest_path=tmp_path / "manifest.jsonl",
        access_permit_path=permit_path,
        ldsr_model_dir=None,
        confirmed_persistent_storage=True,
    )
    assert produced.phase_decision == decision

    monkeypatch.setattr(
        phase2b3c_workflow,
        "_open_authorized_access",
        lambda paths, verified_permit: bundle_complete,
    )
    replayed = phase2b3c_workflow.run_formal_evaluation_replay(
        project_root=project,
        evidence_dir=tmp_path / "evidence",
        storage_root=storage,
        manifest_path=tmp_path / "manifest.jsonl",
        access_permit_path=permit_path,
        confirmed_persistent_storage=True,
    )
    assert replayed.manifest_sha256 == produced.manifest_sha256

    paths = phase2b3c_workflow.phase2b3c_storage_paths(storage)
    copied = tmp_path / "copied-bundle"
    shutil.copytree(paths.bundle_dir, copied)
    loaded = read_phase2b3c_bundle(copied)
    result = loaded.documents()["phase2b3c-evaluation-result.json"]
    metadata_authority = (
        phase2b3c_bundle_verify.VerifiedPhase2B3CMetadataAuthority._from_verified(
            permit=permit,
            ledger=bundle_complete,
            revision="1" * 40,
            head_revision="2" * 40,
            dependencies=fixtures._dependencies(),
            records=records,
            ordered_sample_ids_sha256=result["digests"][
                "ordered_sample_ids_sha256"
            ],
            ordered_membership_sha256=result["digests"][
                "ordered_membership_sha256"
            ],
        )
    )
    monkeypatch.setattr(
        phase2b3c_acceptance,
        "load_phase2b3c_metadata_authority",
        lambda **kwargs: metadata_authority,
    )
    monkeypatch.setattr(
        phase2b3c_acceptance,
        "load_internal_test_pairs",
        lambda root, selected, guard: candidate["pairs"],
    )
    monkeypatch.setattr(
        phase2b3c_acceptance,
        "probe_cached_internal_test_bundles",
        lambda pairs, cache: InternalTestPredictionCacheProbe(bundles, 600, 0),
    )
    monkeypatch.setattr(
        phase2b3c_acceptance,
        "advance_access_ledger",
        lambda root, verified_permit, state: accepted,
    )

    verified = phase2b3c_acceptance.run_independent_phase2b3c_verification(
        bundle_dir=copied,
        project_root=project,
        evidence_dir=tmp_path / "evidence",
        storage_root=storage,
        manifest_path=tmp_path / "manifest.jsonl",
        access_permit_path=permit_path,
        confirmed_persistent_storage=True,
    )
    assert verified.publication.phase_decision == decision
    assert verified.accepted_ledger_event_sha256 == accepted.event_sha256

    _git(project, "add", "-f", "artifacts/phase2b3c")
    assert phase2b3c_publication_policy_violations(project) == ()
