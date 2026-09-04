"""Formal Phase 2B3-B workflow tests using tiny synthetic CPU tensors."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import test_calibration_cache_replay as replay_fixtures
import test_calibration_predictions as prediction_fixtures
import test_phase2b3b_computation_verify as computation_fixtures
import test_phase2b3b_result as result_fixtures
import torch

from trustsr.evaluation.phase2b3b_bundle import read_phase2b3b_bundle


def _module():
    return importlib.import_module("trustsr.evaluation.phase2b3b_workflow")


def _runtime(*args: object, **kwargs: object) -> dict[str, object]:
    return {"schema": "trustsr.phase2b3b-calibration-runtime.v1"}


def _install_authority(
    module: object,
    monkeypatch: pytest.MonkeyPatch,
    pairs: tuple[object, ...],
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    records = computation_fixtures._records(pairs)
    preflight = computation_fixtures._preflight(records)
    monkeypatch.setattr(
        module, "verify_phase2b3b_revision", lambda root: result_fixtures._revision()
    )
    monkeypatch.setattr(
        module, "verify_recorded_phase2b3b_revision", lambda root, revision: revision
    )
    monkeypatch.setattr(module, "load_phase2b3b_preflight", lambda *args: preflight)
    monkeypatch.setattr(module, "load_calibration_records", lambda *args: records)
    monkeypatch.setattr(module, "load_calibration_pairs", lambda *args: pairs)
    monkeypatch.setattr(module, "build_phase2b3b_runtime_manifest", _runtime)
    monkeypatch.setattr(module, "_MINIMUM_FREE_BYTES", 0)
    return preflight, records


def test_calibration_then_replay_is_byte_identical_and_replay_never_predicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    storage_root = tmp_path / "storage"
    project_root = tmp_path / "project"
    evidence_dir = tmp_path / "evidence"
    model_dir = tmp_path / "model"
    for path in (storage_root, project_root, evidence_dir, model_dir):
        path.mkdir()
    pairs = tuple(replay_fixtures._pair(index) for index in range(120))
    _install_authority(module, monkeypatch, pairs)
    ldsr = prediction_fixtures._FakeLDSR()
    monkeypatch.setattr(module, "_load_ldsr", lambda path: ldsr)

    calibrated = module.run_formal_calibration(
        project_root=project_root,
        evidence_dir=evidence_dir,
        storage_root=storage_root,
        manifest_path=tmp_path / "manifest.jsonl",
        ldsr_model_dir=model_dir,
        confirmed_persistent_storage=True,
    )
    calls_after_calibration = dict(ldsr.calls_by_seed)
    monkeypatch.setattr(
        module,
        "_load_ldsr",
        lambda path: (_ for _ in ()).throw(AssertionError("replay constructed LDSR")),
    )

    replayed = module.run_formal_calibration_replay(
        project_root=project_root,
        evidence_dir=evidence_dir,
        storage_root=storage_root,
        manifest_path=tmp_path / "manifest.jsonl",
        confirmed_persistent_storage=True,
    )

    paths = module.phase2b3b_storage_paths(storage_root)
    bundle = read_phase2b3b_bundle(paths.bundle_dir)
    result = bundle.documents()["phase2b3b-calibration-result.json"]
    assert calibrated.stage == "calibration"
    assert replayed.stage == "calibration-replay"
    assert calibrated.manifest_sha256 == replayed.manifest_sha256 == bundle.manifest_sha256
    assert calibrated.result_sha256 == replayed.result_sha256
    assert calibrated.cache_audit_sha256 == replayed.cache_audit_sha256
    assert calibrated.phase_decision == replayed.phase_decision
    assert result["target"] == {"alpha": 0.05, "minimum_coverage": 0.10}
    assert sum(calls_after_calibration.values()) == 600
    assert ldsr.calls_by_seed == calls_after_calibration


def test_complete_verified_prediction_caches_skip_model_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    storage_root = tmp_path / "storage"
    project_root = tmp_path / "project"
    evidence_dir = tmp_path / "evidence"
    model_dir = tmp_path / "model"
    for path in (storage_root, project_root, evidence_dir, model_dir):
        path.mkdir()
    pairs = tuple(replay_fixtures._pair(index) for index in range(120))
    _install_authority(module, monkeypatch, pairs)
    ldsr = prediction_fixtures._FakeLDSR()
    monkeypatch.setattr(module, "_load_ldsr", lambda path: ldsr)
    first = module.run_formal_calibration(
        project_root=project_root,
        evidence_dir=evidence_dir,
        storage_root=storage_root,
        manifest_path=tmp_path / "manifest.jsonl",
        ldsr_model_dir=model_dir,
        confirmed_persistent_storage=True,
    )
    monkeypatch.setattr(
        module,
        "_load_ldsr",
        lambda path: (_ for _ in ()).throw(
            AssertionError("complete caches constructed LDSR")
        ),
    )

    cached = module.run_formal_calibration(
        project_root=project_root,
        evidence_dir=evidence_dir,
        storage_root=storage_root,
        manifest_path=tmp_path / "manifest.jsonl",
        ldsr_model_dir=None,
        confirmed_persistent_storage=True,
    )

    assert cached.manifest_sha256 == first.manifest_sha256
    assert cached.result_sha256 == first.result_sha256
    assert sum(ldsr.calls_by_seed.values()) == 600


def test_missing_prediction_cache_requires_gpu_authorization_before_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    storage_root = tmp_path / "storage"
    project_root = tmp_path / "project"
    evidence_dir = tmp_path / "evidence"
    for path in (storage_root, project_root, evidence_dir):
        path.mkdir()
    pairs = tuple(replay_fixtures._pair(index) for index in range(120))
    _install_authority(module, monkeypatch, pairs)
    monkeypatch.setattr(
        module,
        "_load_ldsr",
        lambda path: (_ for _ in ()).throw(
            AssertionError("missing model path still constructed LDSR")
        ),
    )

    with pytest.raises(RuntimeError, match="K5 prediction caches are missing.*GPU"):
        module.run_formal_calibration(
            project_root=project_root,
            evidence_dir=evidence_dir,
            storage_root=storage_root,
            manifest_path=tmp_path / "manifest.jsonl",
            ldsr_model_dir=None,
            confirmed_persistent_storage=True,
        )


def test_rejects_missing_storage_confirmation_before_pixel_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    loaded_pixels = False

    def load_pairs(*args: object) -> object:
        nonlocal loaded_pixels
        loaded_pixels = True
        raise AssertionError("pixel loading crossed a rejected storage gate")

    monkeypatch.setattr(module, "load_calibration_pairs", load_pairs)

    with pytest.raises(ValueError, match="confirmation"):
        module.run_formal_calibration_replay(
            project_root=tmp_path / "project",
            evidence_dir=tmp_path / "evidence",
            storage_root=storage_root,
            manifest_path=tmp_path / "manifest.jsonl",
            confirmed_persistent_storage=False,
        )

    assert loaded_pixels is False


def test_calibration_replay_recomputes_the_frozen_score_algorithm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    storage_root = tmp_path / "storage"
    project_root = tmp_path / "project"
    evidence_dir = tmp_path / "evidence"
    model_dir = tmp_path / "model"
    for path in (storage_root, project_root, evidence_dir, model_dir):
        path.mkdir()
    pairs = tuple(replay_fixtures._pair(index) for index in range(120))
    _install_authority(module, monkeypatch, pairs)
    monkeypatch.setattr(module, "_load_ldsr", lambda path: prediction_fixtures._FakeLDSR())
    monkeypatch.setattr(
        module,
        "ensemble_variance_score",
        lambda samples: torch.zeros(tuple(samples.shape[-2:]), dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="recomputed ensemble score"):
        module.run_formal_calibration(
            project_root=project_root,
            evidence_dir=evidence_dir,
            storage_root=storage_root,
            manifest_path=tmp_path / "manifest.jsonl",
            ldsr_model_dir=model_dir,
            confirmed_persistent_storage=True,
        )


def test_storage_paths_are_fixed_below_the_confirmed_root(tmp_path: Path) -> None:
    module = _module()
    storage_root = tmp_path / "storage"
    storage_root.mkdir()

    paths = module.phase2b3b_storage_paths(storage_root)

    suffix = "c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a"
    assert paths.prediction_cache_dir == storage_root / "trustsr/phase2b3b/predictions" / suffix
    assert paths.score_cache_dir == storage_root / "trustsr/phase2b3b/scores" / suffix
    assert paths.bundle_dir == storage_root / "trustsr/phase2b3b/bundles" / suffix
    assert paths.lock_path == storage_root / "trustsr/phase2b3b/.formal.lock"
