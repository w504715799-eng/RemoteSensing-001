"""Hostile-input tests for the Phase 2B3-B runtime metadata manifest."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest
import test_phase2b3b_result as result_fixtures

from trustsr.evaluation import (
    calibration_cache_verify,
    calibration_input_receipt,
    phase2b3b_result,
    phase2b3b_result_verify,
    phase2b3b_runtime,
)
from trustsr.jsonio import canonical_json


@pytest.fixture
def artifacts() -> tuple[dict[str, object], dict[str, object], dict[str, object], object, object]:
    sample_ids = result_fixtures._sample_ids()
    preflight = result_fixtures._preflight(sample_ids)
    input_receipt = result_fixtures._input_receipt(sample_ids)
    audit = result_fixtures._audit(sample_ids)
    result = phase2b3b_result.build_phase2b3b_result(
        preflight,
        input_receipt,
        result_fixtures._fit(sample_ids),
        audit,
        result_fixtures._radiometry(sample_ids),
        result_fixtures._revision(),
    )
    input_verification = calibration_input_receipt.verify_calibration_input_receipt(
        input_receipt
    )
    audit_verification = calibration_cache_verify.verify_calibration_cache_audit(audit)
    result_verification = phase2b3b_result_verify.VerifiedPhase2B3BResult(
        schema=phase2b3b_result_verify.SCHEMA,
        verification_scope="metadata_consistency_only",
        cache_computation_verified=False,
        result_sha256=hashlib.sha256(canonical_json(result)).hexdigest(),
        cache_audit_sha256=audit_verification["digests"]["audit_sha256"],
        producer_revision=result["producer_revision"],
        ordered_sample_ids_sha256=result["upstream"]["ordered_sample_ids_sha256"],
        ordered_membership_sha256=result["upstream"]["ordered_membership_sha256"],
        input_receipt_sha256=input_verification.source_sha256,
        ordered_inputs_sha256=input_verification.ordered_inputs_sha256,
        map_evidence_sha256=result["map_evidence_sha256"],
        radiometry_aggregate_sha256="a" * 64,
        phase_decision=result["phase_decision"],
    )
    return result, audit, input_receipt, result_verification, input_verification


@pytest.fixture
def dependency_snapshot() -> dict[str, object]:
    return {
        "python": {"major_minor": "3.12"},
        "uv_lock_sha256": "b" * 64,
        "packages": {
            "numpy": "2.5.2",
            "opensr-model": "1.1.1",
            "rasterio": "1.5.1",
            "torch": "2.12.1+cu130",
            "trustsr": "0.1.0",
        },
    }


def _build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifacts: tuple[dict[str, object], dict[str, object], dict[str, object], object, object],
    dependency_snapshot: dict[str, object],
) -> dict[str, object]:
    result, audit, input_receipt, result_verification, input_verification = artifacts
    monkeypatch.setattr(
        phase2b3b_runtime,
        "verify_recorded_phase2b3b_revision",
        lambda project_root, revision: revision,
    )
    monkeypatch.setattr(
        phase2b3b_runtime,
        "_capture_dependencies",
        lambda project_root: deepcopy(dependency_snapshot),
    )
    monkeypatch.setattr(
        phase2b3b_runtime,
        "verify_phase2b3b_result",
        lambda result, audit, **kwargs: result_verification,
    )
    monkeypatch.setattr(
        phase2b3b_runtime,
        "verify_authoritative_calibration_input_receipt",
        lambda receipt, **kwargs: input_verification,
    )
    return phase2b3b_runtime.build_phase2b3b_runtime_manifest(
        deepcopy(result),
        deepcopy(audit),
        deepcopy(input_receipt),
        project_root=tmp_path,
        evidence_dir=tmp_path / "evidence",
        storage_root=tmp_path / "storage",
        manifest_path=tmp_path / "manifest.json",
    )


def test_builds_exact_host_free_runtime_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifacts: tuple[dict[str, object], dict[str, object], dict[str, object], object, object],
    dependency_snapshot: dict[str, object],
) -> None:
    runtime = _build(monkeypatch, tmp_path, artifacts, dependency_snapshot)
    result, audit, input_receipt, result_verification, input_verification = artifacts

    assert set(runtime) == {
        "schema",
        "phase",
        "verification_scope",
        "cache_computation_verified",
        "dependencies",
        "model_inventory",
        "inputs",
        "artifacts",
        "revision",
    }
    assert runtime["schema"] == "trustsr.phase2b3b-calibration-runtime.v1"
    assert runtime["verification_scope"] == "metadata_inventory_only"
    assert runtime["cache_computation_verified"] is False
    assert runtime["model_inventory"]["seeds"] == [3407, 3408, 3409, 3410, 3411]
    assert "seed" not in runtime["model_inventory"]["identity"]
    assert runtime["model_inventory"]["identity"]["checkpoint_name"] == (
        "opensr-ldsrs2_v1_0_0.ckpt"
    )
    assert runtime["inputs"]["input_receipt_sha256"] == input_verification.source_sha256
    assert runtime["artifacts"]["result_sha256"] == result_verification.result_sha256

    verified = phase2b3b_runtime.verify_phase2b3b_runtime_manifest(
        runtime,
        deepcopy(result),
        deepcopy(audit),
        project_root=tmp_path,
        evidence_dir=tmp_path / "evidence",
        storage_root=tmp_path / "storage",
        manifest_path=tmp_path / "manifest.json",
    )
    assert verified.schema == runtime["schema"]
    assert verified.verification_scope == "metadata_inventory_only"
    assert verified.cache_computation_verified is False
    assert verified.runtime_sha256 == hashlib.sha256(canonical_json(runtime)).hexdigest()


def test_verify_uses_result_authority_without_raw_input_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifacts: tuple[dict[str, object], dict[str, object], dict[str, object], object, object],
    dependency_snapshot: dict[str, object],
) -> None:
    runtime = _build(monkeypatch, tmp_path, artifacts, dependency_snapshot)
    result, audit, input_receipt, result_verification, _ = artifacts
    calls: list[object] = []
    monkeypatch.setattr(
        phase2b3b_runtime,
        "verify_phase2b3b_result",
        lambda result, audit, **kwargs: calls.append((result, audit, kwargs))
        or result_verification,
    )
    monkeypatch.setattr(
        phase2b3b_runtime,
        "verify_authoritative_calibration_input_receipt",
        lambda receipt, **kwargs: pytest.fail("verify must not request raw input receipt"),
    )

    phase2b3b_runtime.verify_phase2b3b_runtime_manifest(
        runtime,
        result,
        audit,
        project_root=tmp_path,
        evidence_dir=tmp_path / "evidence",
        storage_root=tmp_path / "storage",
        manifest_path=tmp_path / "manifest.json",
    )
    assert len(calls) == 1

    with pytest.raises(TypeError):
        phase2b3b_runtime.verify_phase2b3b_runtime_manifest(
            runtime,
            result,
            audit,
            project_root=tmp_path,
            evidence_dir=tmp_path / "evidence",
            storage_root=tmp_path / "storage",
            manifest_path=tmp_path / "manifest.json",
            input_receipt=input_receipt,
        )


def test_public_api_forwards_raw_inputs_to_authority_verifiers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifacts: tuple[dict[str, object], dict[str, object], dict[str, object], object, object],
    dependency_snapshot: dict[str, object],
) -> None:
    result, audit, input_receipt, result_verification, input_verification = artifacts
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        phase2b3b_runtime,
        "verify_recorded_phase2b3b_revision",
        lambda project_root, revision: revision,
    )
    monkeypatch.setattr(
        phase2b3b_runtime,
        "_capture_dependencies",
        lambda project_root: deepcopy(dependency_snapshot),
    )
    monkeypatch.setattr(
        phase2b3b_runtime,
        "verify_phase2b3b_result",
        lambda received_result, received_audit, **kwargs: observed.update(
            result=(received_result, received_audit, kwargs)
        ) or result_verification,
    )
    monkeypatch.setattr(
        phase2b3b_runtime,
        "verify_authoritative_calibration_input_receipt",
        lambda received_receipt, **kwargs: observed.update(
            input=(received_receipt, kwargs)
        ) or input_verification,
    )
    evidence_dir = tmp_path / "evidence"
    storage_root = tmp_path / "storage"
    manifest_path = tmp_path / "manifest.json"

    phase2b3b_runtime.build_phase2b3b_runtime_manifest(
        result,
        audit,
        input_receipt,
        project_root=tmp_path,
        evidence_dir=evidence_dir,
        storage_root=storage_root,
        manifest_path=manifest_path,
    )

    assert observed == {
        "result": (
            result,
            audit,
            {
                "project_root": tmp_path,
                "evidence_dir": evidence_dir,
                "storage_root": storage_root,
                "manifest_path": manifest_path,
            },
        ),
        "input": (
            input_receipt,
            {
                "evidence_dir": evidence_dir,
                "storage_root": storage_root,
                "manifest_path": manifest_path,
            },
        ),
    }


def test_public_api_has_no_direct_receipt_entrypoint(
    tmp_path: Path,
    artifacts: tuple[dict[str, object], dict[str, object], dict[str, object], object, object],
) -> None:
    result, audit, input_receipt, result_verification, input_verification = artifacts

    with pytest.raises(TypeError):
        phase2b3b_runtime.build_phase2b3b_runtime_manifest(
            result,
            audit,
            input_receipt,
            project_root=tmp_path,
            evidence_dir=tmp_path / "evidence",
            storage_root=tmp_path / "storage",
            manifest_path=tmp_path / "manifest.json",
            result_verification=result_verification,
            input_verification=input_verification,
        )


@pytest.mark.parametrize("authority", ("result", "input"))
def test_authority_rejection_aborts_runtime_composition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifacts: tuple[dict[str, object], dict[str, object], dict[str, object], object, object],
    authority: str,
) -> None:
    result, audit, input_receipt, result_verification, _ = artifacts
    if authority == "result":
        monkeypatch.setattr(
            phase2b3b_runtime,
            "verify_phase2b3b_result",
            lambda result, audit, **kwargs: (_ for _ in ()).throw(ValueError("result authority")),
        )
    else:
        monkeypatch.setattr(
            phase2b3b_runtime,
            "verify_phase2b3b_result",
            lambda result, audit, **kwargs: result_verification,
        )
        monkeypatch.setattr(
            phase2b3b_runtime,
            "verify_authoritative_calibration_input_receipt",
            lambda receipt, **kwargs: (_ for _ in ()).throw(ValueError("input authority")),
        )

    with pytest.raises(ValueError, match="authority"):
        phase2b3b_runtime.build_phase2b3b_runtime_manifest(
            result,
            audit,
            input_receipt,
            project_root=tmp_path,
            evidence_dir=tmp_path / "evidence",
            storage_root=tmp_path / "storage",
            manifest_path=tmp_path / "manifest.json",
        )


@pytest.mark.parametrize(
    "fault",
    (
        "extra",
        "missing",
        "scope",
        "boolean",
        "path",
        "version",
        "model_leak",
        "result_digest",
        "revision",
        "input_digest",
    ),
)
def test_rejects_hostile_runtime_mutations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifacts: tuple[dict[str, object], dict[str, object], dict[str, object], object, object],
    dependency_snapshot: dict[str, object],
    fault: str,
) -> None:
    runtime = _build(monkeypatch, tmp_path, artifacts, dependency_snapshot)
    if fault == "extra":
        runtime["extra"] = True
    elif fault == "missing":
        runtime.pop("phase")
    elif fault == "scope":
        runtime["verification_scope"] = "cache_computation"
    elif fault == "boolean":
        runtime["cache_computation_verified"] = True
    elif fault == "path":
        runtime["dependencies"]["packages"]["torch"] = "/tmp/torch"
    elif fault == "version":
        runtime["dependencies"]["packages"]["torch"] = "internal_test"
    elif fault == "model_leak":
        runtime["model_inventory"]["identity"]["checkpoint_name"] = "token=secret"
    elif fault == "result_digest":
        runtime["artifacts"]["result_sha256"] = "0" * 64
    elif fault == "revision":
        runtime["revision"]["producer_revision"] = "0" * 40
    else:
        runtime["inputs"]["ordered_inputs_sha256"] = "0" * 64

    result, audit, input_receipt, result_verification, input_verification = artifacts
    with pytest.raises((TypeError, ValueError)):
        phase2b3b_runtime.verify_phase2b3b_runtime_manifest(
            runtime,
            deepcopy(result),
            deepcopy(audit),
            project_root=tmp_path,
            evidence_dir=tmp_path / "evidence",
            storage_root=tmp_path / "storage",
            manifest_path=tmp_path / "manifest.json",
        )


def test_rejects_noncanonical_inputs_and_mixed_audit_model_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifacts: tuple[dict[str, object], dict[str, object], dict[str, object], object, object],
    dependency_snapshot: dict[str, object],
) -> None:
    result, audit, input_receipt, result_verification, input_verification = artifacts
    monkeypatch.setattr(
        phase2b3b_runtime,
        "verify_recorded_phase2b3b_revision",
        lambda project_root, revision: revision,
    )
    monkeypatch.setattr(
        phase2b3b_runtime,
        "_capture_dependencies",
        lambda project_root: deepcopy(dependency_snapshot),
    )
    monkeypatch.setattr(
        phase2b3b_runtime,
        "verify_phase2b3b_result",
        lambda result, audit, **kwargs: result_verification,
    )
    monkeypatch.setattr(
        phase2b3b_runtime,
        "verify_authoritative_calibration_input_receipt",
        lambda receipt, **kwargs: input_verification,
    )
    with pytest.raises(ValueError, match="canonical"):
        phase2b3b_runtime.build_phase2b3b_runtime_manifest(
            canonical_json(result) + b" ",
            audit,
            input_receipt,
            project_root=tmp_path,
            evidence_dir=tmp_path / "evidence",
            storage_root=tmp_path / "storage",
            manifest_path=tmp_path / "manifest.json",
        )

    mixed = deepcopy(audit)
    mixed["samples"][0]["predictions"][0]["identity"]["model_provenance"][
        "torch_version"
    ] = "2.13.0+cu130"
    with pytest.raises(ValueError, match="model|provenance|cache key"):
        phase2b3b_runtime.build_phase2b3b_runtime_manifest(
            result,
            mixed,
            input_receipt,
            project_root=tmp_path,
            evidence_dir=tmp_path / "evidence",
            storage_root=tmp_path / "storage",
            manifest_path=tmp_path / "manifest.json",
        )
