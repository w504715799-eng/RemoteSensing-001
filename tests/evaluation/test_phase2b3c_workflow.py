"""Formal Phase 2B3-C workflow ordering with synthetic CPU tensors."""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path

import pytest
import test_phase2b3c_computation_verify as fixtures

from trustsr.evaluation import phase2b3c_access
from trustsr.evaluation.internal_test_predictions import (
    InternalTestPredictionCacheProbe,
)
from trustsr.evaluation.phase2b3c_access import (
    AccessLedgerSnapshot,
    VerifiedAccessPermit,
)
from trustsr.evaluation.phase2b3c_bundle import read_phase2b3c_bundle
from trustsr.evaluation.phase2b3c_revision import VerifiedRevision


def _module():
    return importlib.import_module("trustsr.evaluation.phase2b3c_workflow")


def _permit() -> VerifiedAccessPermit:
    return VerifiedAccessPermit(
        evaluation_id=fixtures._sha("evaluation"),
        permit_sha256=fixtures._sha("permit"),
        readiness_sha256=fixtures._sha("readiness"),
        implementation_revision="1" * 40,
        computation_tree_sha256=fixtures._sha("tree"),
        ordered_membership_sha256=fixtures._sha("membership"),
        environment_sha256=fixtures._sha("environment"),
        phase2b3b_acceptance_sha256=(
            "ce7f1a91e9954235e289dfc6eececb67261ff57110caaf15fe6640a0fb3b69ab"
        ),
        _authority=phase2b3c_access._PERMIT_AUTHORITY,
    )


def _snapshot(state: str) -> AccessLedgerSnapshot:
    sequence = {"pixels_opened": 1, "caches_complete": 2, "bundle_complete": 3}[state]
    return AccessLedgerSnapshot(
        state=state,
        sequence=sequence,
        evaluation_id=fixtures._sha("evaluation"),
        event_sha256=fixtures._sha(state),
        previous_event_sha256=(
            fixtures._sha("previous") if state == "pixels_opened" else fixtures._sha(
                "pixels_opened" if state == "caches_complete" else "caches_complete"
            )
        ),
        permit_sha256=fixtures._sha("permit"),
        _authority=phase2b3c_access._LEDGER_AUTHORITY,
    )


def _install(
    module: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    complete: bool,
):
    candidate = fixtures._candidate(tmp_path / "candidate")
    records = tuple(fixtures._record(index) for index in range(120))
    authority = module.Phase2B3CAuthority(
        revision=VerifiedRevision("main", "1" * 40, "1" * 40, fixtures._sha("tree")),
        records=records,
        permit=_permit(),
        dependencies=fixtures._dependencies(),
    )
    events: list[str] = []
    monkeypatch.setattr(module, "_MINIMUM_FREE_BYTES", 0)
    monkeypatch.setattr(
        module,
        "_prepare_authority",
        lambda **kwargs: events.append("authority") or authority,
    )
    monkeypatch.setattr(
        module,
        "_open_authorized_access",
        lambda paths, permit: events.append("pixels_opened")
        or _snapshot("pixels_opened"),
    )
    monkeypatch.setattr(
        module,
        "load_internal_test_pairs",
        lambda root, loaded_records, guard: events.append("pairs")
        or candidate["pairs"],
    )
    probe = InternalTestPredictionCacheProbe(
        candidate["bundles"] if complete else None,
        600 if complete else 599,
        0 if complete else 1,
    )
    monkeypatch.setattr(
        module,
        "probe_cached_internal_test_bundles",
        lambda pairs, cache: events.append("probe") or probe,
    )

    def advance(root: Path, permit: VerifiedAccessPermit, state: str):
        events.append(state)
        return _snapshot(state)

    monkeypatch.setattr(module, "advance_access_ledger", advance)
    return candidate, authority, events


def test_complete_cache_evaluation_stays_cpu_and_publishes_atomic_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    storage = tmp_path / "storage"
    storage.mkdir()
    candidate, _authority, events = _install(
        module, monkeypatch, tmp_path, complete=True
    )
    monkeypatch.setattr(
        module,
        "_load_ldsr",
        lambda path: (_ for _ in ()).throw(AssertionError("complete cache loaded LDSR")),
    )

    receipt = module.run_formal_evaluation(
        project_root=tmp_path / "project",
        evidence_dir=tmp_path / "evidence",
        storage_root=storage,
        manifest_path=tmp_path / "manifest.jsonl",
        access_permit_path=tmp_path / "permit.json",
        ldsr_model_dir=None,
        confirmed_persistent_storage=True,
    )

    assert receipt.stage == "evaluate"
    assert receipt.phase_decision == "confirmed"
    assert events == [
        "authority",
        "pixels_opened",
        "pairs",
        "probe",
        "caches_complete",
        "bundle_complete",
    ]
    paths = module.phase2b3c_storage_paths(storage)
    bundle = read_phase2b3c_bundle(paths.bundle_dir)
    assert bundle.manifest_sha256 == receipt.manifest_sha256
    assert len(bundle.documents()) == 5
    assert len(candidate["pairs"]) == 120


def test_missing_cache_stops_with_exact_count_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    storage = tmp_path / "storage"
    storage.mkdir()
    _candidate, _authority, events = _install(
        module, monkeypatch, tmp_path, complete=False
    )
    monkeypatch.setattr(
        module,
        "_load_ldsr",
        lambda path: (_ for _ in ()).throw(AssertionError("missing cache loaded LDSR")),
    )
    monkeypatch.setattr(
        module,
        "_documents_from_caches",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("incomplete caches produced partial metrics")
        ),
    )

    with pytest.raises(RuntimeError, match="599.*1 missing.*GPU"):
        module.run_formal_evaluation(
            project_root=tmp_path / "project",
            evidence_dir=tmp_path / "evidence",
            storage_root=storage,
            manifest_path=tmp_path / "manifest.jsonl",
            access_permit_path=tmp_path / "permit.json",
            ldsr_model_dir=None,
            confirmed_persistent_storage=True,
        )

    assert events == ["authority", "pixels_opened", "pairs", "probe"]
    assert capsys.readouterr() == ("", "")
    assert not module.phase2b3c_storage_paths(storage).bundle_dir.exists()


def test_authorized_generation_constructs_ldsr_only_after_the_exact_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    storage = tmp_path / "storage"
    storage.mkdir()
    candidate, _authority, events = _install(
        module, monkeypatch, tmp_path, complete=False
    )
    incomplete = InternalTestPredictionCacheProbe(None, 599, 1)
    complete = InternalTestPredictionCacheProbe(candidate["bundles"], 600, 0)
    probes = iter((incomplete, complete))
    monkeypatch.setattr(
        module,
        "probe_cached_internal_test_bundles",
        lambda pairs, cache: events.append("probe") or next(probes),
    )
    model = object()
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    def load_model(path: Path) -> object:
        assert path == model_dir
        events.append("model")
        return model

    generated: list[str] = []

    def generate(pair: object, *, ldsr: object, cache: object) -> object:
        assert ldsr is model
        generated.append(pair.pair.sample_id)
        return candidate["bundles"][len(generated) - 1]

    monkeypatch.setattr(module, "_load_ldsr", load_model)
    monkeypatch.setattr(module, "load_or_generate_internal_test_bundle", generate)

    receipt = module.run_formal_evaluation(
        project_root=tmp_path / "project",
        evidence_dir=tmp_path / "evidence",
        storage_root=storage,
        manifest_path=tmp_path / "manifest.jsonl",
        access_permit_path=tmp_path / "permit.json",
        ldsr_model_dir=model_dir,
        confirmed_persistent_storage=True,
    )

    assert receipt.stage == "evaluate"
    assert len(generated) == 120
    assert events == [
        "authority",
        "pixels_opened",
        "pairs",
        "probe",
        "model",
        "probe",
        "caches_complete",
        "bundle_complete",
    ]


def test_caches_complete_resume_reuses_verified_caches_without_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    storage = tmp_path / "storage"
    storage.mkdir()
    _candidate, _authority, events = _install(
        module, monkeypatch, tmp_path, complete=True
    )
    monkeypatch.setattr(
        module,
        "_open_authorized_access",
        lambda paths, permit: events.append("resume:caches_complete")
        or _snapshot("caches_complete"),
    )
    monkeypatch.setattr(
        module,
        "_load_ldsr",
        lambda path: (_ for _ in ()).throw(AssertionError("resume loaded LDSR")),
    )

    receipt = module.run_formal_evaluation(
        project_root=tmp_path / "project",
        evidence_dir=tmp_path / "evidence",
        storage_root=storage,
        manifest_path=tmp_path / "manifest.jsonl",
        access_permit_path=tmp_path / "permit.json",
        ldsr_model_dir=None,
        confirmed_persistent_storage=True,
    )

    assert receipt.stage == "evaluate"
    assert events == [
        "authority",
        "resume:caches_complete",
        "pairs",
        "probe",
        "bundle_complete",
    ]


def test_rejects_storage_without_confirmation_before_authority_or_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    storage = tmp_path / "storage"
    storage.mkdir()
    monkeypatch.setattr(module, "_MINIMUM_FREE_BYTES", 0)
    monkeypatch.setattr(
        module,
        "_prepare_authority",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("rejected storage reached authority")
        ),
    )
    monkeypatch.setattr(
        module,
        "load_internal_test_pairs",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("rejected storage reached pixels")
        ),
    )

    with pytest.raises(ValueError, match="confirmation"):
        module.run_formal_evaluation(
            project_root=tmp_path / "project",
            evidence_dir=tmp_path / "evidence",
            storage_root=storage,
            manifest_path=tmp_path / "manifest.jsonl",
            access_permit_path=tmp_path / "permit.json",
            ldsr_model_dir=None,
            confirmed_persistent_storage=False,
        )


def test_metadata_preflight_optionally_verifies_permit_without_ledger_or_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    storage = tmp_path / "storage"
    storage.mkdir()
    records = tuple(fixtures._record(index) for index in range(120))
    revision = VerifiedRevision("main", "1" * 40, "1" * 40, fixtures._sha("tree"))
    load_evidence = module.load_frozen_phase2b3b_evidence
    evidence = load_evidence(
        Path(__file__).parents[2] / "artifacts" / "phase2b3b",
        Path(__file__).parents[2],
    )
    initial = module.build_phase2b3c_preflight(evidence, records, revision)
    permit = replace(
        _permit(),
        ordered_membership_sha256=initial["evaluation"][
            "ordered_membership_sha256"
        ],
    )
    events: list[str] = []
    monkeypatch.setattr(module, "_MINIMUM_FREE_BYTES", 0)
    monkeypatch.setattr(module, "_current_revision", lambda root: revision)
    monkeypatch.setattr(
        module,
        "load_frozen_phase2b3b_evidence",
        lambda evidence_dir, root: evidence,
    )
    monkeypatch.setattr(module, "load_internal_test_records", lambda *args: records)
    monkeypatch.setattr(
        module, "_capture_dependencies", lambda root: fixtures._dependencies()
    )
    monkeypatch.setattr(
        module,
        "verify_phase2b3c_access_permit",
        lambda path, root, readiness: events.append("permit") or permit,
    )
    for forbidden in (
        "_open_authorized_access",
        "load_internal_test_pairs",
        "probe_cached_internal_test_bundles",
        "_load_ldsr",
        "write_phase2b3c_bundle",
    ):
        monkeypatch.setattr(
            module,
            forbidden,
            lambda *args, _name=forbidden, **kwargs: (_ for _ in ()).throw(
                AssertionError(f"metadata preflight called {_name}")
            ),
        )

    result = module.run_metadata_preflight(
        project_root=tmp_path / "project",
        evidence_dir=tmp_path / "evidence",
        storage_root=storage,
        manifest_path=tmp_path / "manifest.jsonl",
        access_permit_path=tmp_path / "permit.json",
        confirmed_persistent_storage=True,
    )

    assert events == ["permit"]
    assert result["preflight"]["permit"]["permit_sha256"] == permit.permit_sha256


@pytest.mark.parametrize("failure_point", ["authority", "pairs"])
def test_interruption_on_either_side_of_pixel_access_leaves_no_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    module = _module()
    storage = tmp_path / "storage"
    storage.mkdir()
    _candidate, _authority, events = _install(
        module, monkeypatch, tmp_path, complete=True
    )
    if failure_point == "authority":
        monkeypatch.setattr(
            module,
            "_prepare_authority",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
        )
    else:
        monkeypatch.setattr(
            module,
            "load_internal_test_pairs",
            lambda *args: events.append("pairs:interrupted")
            or (_ for _ in ()).throw(RuntimeError("interrupted")),
        )
    monkeypatch.setattr(
        module,
        "_documents_from_caches",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("interruption produced partial metrics")
        ),
    )

    with pytest.raises(RuntimeError, match="interrupted"):
        module.run_formal_evaluation(
            project_root=tmp_path / "project",
            evidence_dir=tmp_path / "evidence",
            storage_root=storage,
            manifest_path=tmp_path / "manifest.jsonl",
            access_permit_path=tmp_path / "permit.json",
            ldsr_model_dir=None,
            confirmed_persistent_storage=True,
        )

    assert not module.phase2b3c_storage_paths(storage).bundle_dir.exists()
    if failure_point == "authority":
        assert "pixels_opened" not in events
    else:
        assert events[:3] == ["authority", "pixels_opened", "pairs:interrupted"]


def test_replay_uses_existing_cache_and_never_accepts_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    storage = tmp_path / "storage"
    storage.mkdir()
    _candidate, _authority, _events = _install(
        module, monkeypatch, tmp_path, complete=True
    )
    module.run_formal_evaluation(
        project_root=tmp_path / "project",
        evidence_dir=tmp_path / "evidence",
        storage_root=storage,
        manifest_path=tmp_path / "manifest.jsonl",
        access_permit_path=tmp_path / "permit.json",
        ldsr_model_dir=None,
        confirmed_persistent_storage=True,
    )
    monkeypatch.setattr(
        module,
        "_open_authorized_access",
        lambda paths, permit: _snapshot("bundle_complete"),
    )
    monkeypatch.setattr(
        module,
        "_load_ldsr",
        lambda path: (_ for _ in ()).throw(AssertionError("replay loaded LDSR")),
    )

    replay = module.run_formal_evaluation_replay(
        project_root=tmp_path / "project",
        evidence_dir=tmp_path / "evidence",
        storage_root=storage,
        manifest_path=tmp_path / "manifest.jsonl",
        access_permit_path=tmp_path / "permit.json",
        confirmed_persistent_storage=True,
    )

    assert replay.stage == "evaluation-replay"
    assert replay.phase_decision == "confirmed"
