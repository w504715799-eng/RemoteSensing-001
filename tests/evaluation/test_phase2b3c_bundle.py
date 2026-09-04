"""Tests for the Phase 2B3-C canonical bundle boundary."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from trustsr.evaluation.internal_test_predictions import build_cache_provenance
from trustsr.evaluation.phase2b3c_bundle import (
    BUNDLE_DOCUMENT_SCHEMAS,
    BUNDLE_MANIFEST_BASENAME,
    BundleWriteReceipt,
    LoadedPhase2B3CBundle,
    build_phase2b3c_bundle_manifest,
    build_phase2b3c_ledger_snapshot,
    read_phase2b3c_bundle,
    verify_phase2b3c_bundle_documents,
    write_phase2b3c_bundle,
)
from trustsr.evaluation.phase2b3c_replay_receipt import (
    build_phase2b3c_replay_receipt,
)
from trustsr.evaluation.phase2b3c_result import build_phase2b3c_result
from trustsr.evaluation.phase2b3c_runtime import build_phase2b3c_runtime_manifest
from trustsr.evaluation.phase2b3c_statistics import (
    ROIEvaluation,
    build_phase2b3c_statistics,
)
from trustsr.jsonio import canonical_json
from trustsr.models.ldsr_assets import (
    CHECKPOINT_NAME,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    CHECKPOINT_URL,
    CONFIG_SHA256,
)
from trustsr.models.versions import OPENSR_MODEL_VERSION


def _documents() -> dict[str, dict[str, object]]:
    return {
        name: {"schema": schema, "payload": {"value": index}}
        for index, (name, schema) in enumerate(BUNDLE_DOCUMENT_SCHEMAS.items())
    }


def _write(directory: Path, documents: dict[str, dict[str, object]]) -> object:
    values = tuple(documents[name] for name in BUNDLE_DOCUMENT_SCHEMAS)
    return write_phase2b3c_bundle(
        directory,
        result=values[0],
        cache_audit=values[1],
        runtime=values[2],
        replay=values[3],
        ledger_snapshot=values[4],
    )


def test_builds_sorted_manifest_over_exact_canonical_documents() -> None:
    documents = _documents()

    manifest = build_phase2b3c_bundle_manifest(documents)

    assert manifest["schema"] == "trustsr.phase2b3c-bundle-manifest.v1"
    assert manifest["phase"] == "internal_test_evaluation"
    assert [entry["basename"] for entry in manifest["files"]] == sorted(documents)
    for entry in manifest["files"]:
        payload = canonical_json(documents[entry["basename"]])
        assert entry["size_bytes"] == len(payload)
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()


def test_uses_frozen_six_file_allowlist() -> None:
    assert tuple(BUNDLE_DOCUMENT_SCHEMAS) == (
        "phase2b3c-evaluation-result.json",
        "phase2b3c-evaluation-cache-audit.json",
        "phase2b3c-evaluation-runtime.json",
        "phase2b3c-evaluation-replay.json",
        "phase2b3c-access-ledger-snapshot.json",
    )
    assert BUNDLE_MANIFEST_BASENAME == "phase2b3c-bundle-manifest.json"


@pytest.mark.parametrize("mutation", ("missing", "extra", "schema", "non_mapping"))
def test_rejects_invalid_document_set(mutation: str) -> None:
    documents: dict[str, object] = _documents()
    name = next(iter(documents))
    if mutation == "missing":
        del documents[name]
    elif mutation == "extra":
        documents["extra.json"] = {"schema": "forged"}
    elif mutation == "schema":
        documents[name]["schema"] = "forged"  # type: ignore[index]
    else:
        documents[name] = []

    with pytest.raises((TypeError, ValueError)):
        build_phase2b3c_bundle_manifest(documents)  # type: ignore[arg-type]


def test_writes_and_verifies_bundle_idempotently(tmp_path: Path) -> None:
    directory = tmp_path / "bundle"
    documents = _documents()

    first = _write(directory, documents)
    second = _write(directory, documents)
    verified = read_phase2b3c_bundle(directory)

    assert first == second
    assert verified.documents() == documents
    assert verified.documents() is not verified.documents()
    assert {entry.name for entry in directory.iterdir()} == {
        *documents,
        BUNDLE_MANIFEST_BASENAME,
    }
    assert all(
        canonical_json(json.loads(path.read_bytes())) == path.read_bytes()
        for path in directory.iterdir()
    )
    with pytest.raises(FrozenInstanceError):
        first.manifest_sha256 = "0" * 64  # type: ignore[misc]


def test_loaded_bundle_constructor_rejects_forged_unverified_bytes() -> None:
    manifest_payload = canonical_json({})
    payloads = tuple((name, canonical_json({})) for name in BUNDLE_DOCUMENT_SCHEMAS)

    with pytest.raises(ValueError, match="manifest|schema|canonical"):
        LoadedPhase2B3CBundle(
            payloads=payloads,
            manifest_payload=manifest_payload,
            manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        )

    documents = _documents()
    valid_manifest_payload = canonical_json(build_phase2b3c_bundle_manifest(documents))
    valid_payloads = tuple(
        (name, canonical_json(documents[name])) for name in BUNDLE_DOCUMENT_SCHEMAS
    )
    changed = list(valid_payloads)
    changed[0] = (
        changed[0][0],
        canonical_json(
            {
                "schema": BUNDLE_DOCUMENT_SCHEMAS[changed[0][0]],
                "payload": {"value": 999},
            }
        ),
    )

    with pytest.raises(ValueError, match="digest"):
        LoadedPhase2B3CBundle(
            payloads=tuple(changed),
            manifest_payload=valid_manifest_payload,
            manifest_sha256=hashlib.sha256(valid_manifest_payload).hexdigest(),
        )


def test_write_receipt_cannot_be_constructed_from_unverified_digests() -> None:
    with pytest.raises(TypeError):
        BundleWriteReceipt()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        BundleWriteReceipt(
            manifest_sha256="0" * 64,
            file_sha256s=tuple((name, "0" * 64) for name in BUNDLE_DOCUMENT_SCHEMAS),
        )


def test_writer_snapshots_nested_documents_once_before_manifesting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trustsr.evaluation.phase2b3c_bundle as module

    documents = _documents()
    original_value = documents[next(iter(documents))]["payload"]["value"]  # type: ignore[index]
    real_canonical_json = module.canonical_json
    calls = 0

    def mutate_after_document_snapshot(value: object) -> bytes:
        nonlocal calls
        payload = real_canonical_json(value)
        calls += 1
        if calls == len(BUNDLE_DOCUMENT_SCHEMAS):
            documents[next(iter(documents))]["payload"]["value"] = 999  # type: ignore[index]
        return payload

    monkeypatch.setattr(module, "canonical_json", mutate_after_document_snapshot)
    directory = tmp_path / "bundle"

    _write(directory, documents)

    loaded = read_phase2b3c_bundle(directory).documents()
    assert loaded[next(iter(documents))]["payload"]["value"] == original_value  # type: ignore[index]


def test_refuses_to_overwrite_existing_different_bundle(tmp_path: Path) -> None:
    directory = tmp_path / "bundle"
    _write(directory, _documents())
    changed = _documents()
    changed[next(iter(changed))]["payload"] = {"value": 999}

    with pytest.raises(ValueError, match="different bytes"):
        _write(directory, changed)


def test_publish_race_does_not_overwrite_an_empty_target_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "bundle"
    real_fsync = os.fsync
    raced = False

    def create_target_after_absence_check(descriptor: int) -> None:
        nonlocal raced
        if not raced:
            raced = True
            directory.mkdir()
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", create_target_after_absence_check)

    with pytest.raises(ValueError):
        _write(directory, _documents())

    assert directory.is_dir()
    assert list(directory.iterdir()) == []
    assert list(tmp_path.glob(".bundle.*.tmp")) == []


def test_concurrent_identical_writers_publish_one_identical_bundle(tmp_path: Path) -> None:
    directory = tmp_path / "bundle"
    documents = _documents()

    with ThreadPoolExecutor(max_workers=2) as executor:
        receipts = tuple(executor.map(lambda _: _write(directory, documents), range(2)))

    assert receipts[0] == receipts[1]
    assert read_phase2b3c_bundle(directory).documents() == documents


@pytest.mark.parametrize("mutation", ("extra", "noncanonical", "tamper", "symlink"))
def test_verifier_rejects_hostile_bundle_files(tmp_path: Path, mutation: str) -> None:
    directory = tmp_path / "bundle"
    documents = _documents()
    _write(directory, documents)
    target = directory / next(iter(documents))
    if mutation == "extra":
        (directory / "extra.json").write_text("{}", encoding="utf-8")
    elif mutation == "noncanonical":
        target.write_text(json.dumps(documents[target.name], indent=2), encoding="utf-8")
    elif mutation == "tamper":
        target.write_bytes(canonical_json({"schema": BUNDLE_DOCUMENT_SCHEMAS[target.name]}))
    else:
        target.unlink()
        target.symlink_to(directory / BUNDLE_MANIFEST_BASENAME)

    with pytest.raises(ValueError):
        read_phase2b3c_bundle(directory)


@pytest.mark.parametrize("mutation", ("reordered", "duplicate", "traversal", "size", "digest"))
def test_verifier_rejects_hostile_manifest_entries(tmp_path: Path, mutation: str) -> None:
    directory = tmp_path / "bundle"
    documents = _documents()
    _write(directory, documents)
    path = directory / BUNDLE_MANIFEST_BASENAME
    manifest = json.loads(path.read_bytes())
    if mutation == "reordered":
        manifest["files"].reverse()
    elif mutation == "duplicate":
        manifest["files"][1] = dict(manifest["files"][0])
    elif mutation == "traversal":
        manifest["files"][0]["basename"] = "../escape.json"
    elif mutation == "size":
        manifest["files"][0]["size_bytes"] += 1
    else:
        manifest["files"][0]["sha256"] = "0" * 64
    path.write_bytes(canonical_json(manifest))

    with pytest.raises(ValueError):
        read_phase2b3c_bundle(directory)


@pytest.mark.parametrize("mutation", ("fifo", "oversized"))
def test_verifier_rejects_nonregular_or_oversized_file(tmp_path: Path, mutation: str) -> None:
    directory = tmp_path / "bundle"
    documents = _documents()
    _write(directory, documents)
    target = directory / next(iter(documents))
    target.unlink()
    if mutation == "fifo":
        os.mkfifo(target)
    else:
        target.write_bytes(b"x" * (5 * 1024 * 1024 + 1))

    with pytest.raises(ValueError, match="bounded regular file"):
        read_phase2b3c_bundle(directory)


def test_rejects_bundle_directory_with_symlink_component(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="canonical"):
        _write(linked_parent / "bundle", _documents())


def test_failed_publish_leaves_no_bundle_or_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trustsr.evaluation.phase2b3c_bundle as module

    calls = 0
    real_write = module.atomic_write_bytes

    def fail_second(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected")
        real_write(path, payload)

    monkeypatch.setattr(module, "atomic_write_bytes", fail_second)
    directory = tmp_path / "bundle"

    with pytest.raises(OSError, match="injected"):
        _write(directory, _documents())

    assert not directory.exists()
    assert list(tmp_path.iterdir()) == []


def test_corrupt_staging_is_rejected_before_publish_and_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "bundle"
    document_name = next(iter(BUNDLE_DOCUMENT_SCHEMAS))
    real_fsync = os.fsync
    tampered = False

    def tamper_before_directory_sync(descriptor: int) -> None:
        nonlocal tampered
        if not tampered and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            tampered = True
            staging = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
            (staging / document_name).write_bytes(canonical_json({}))
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", tamper_before_directory_sync)

    with pytest.raises(ValueError):
        _write(directory, _documents())

    assert not directory.exists()
    assert list(tmp_path.iterdir()) == []


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _semantic_documents() -> dict[str, dict[str, object]]:
    sample_ids = [f"test-{index:03d}" for index in range(120)]
    samples = []
    map_entries = []
    for index, sample_id in enumerate(sample_ids):
        score_sha256 = _sha(f"score:{index}")
        risk_sha256 = _sha(f"risk:{index}")
        predictions = [
            {
                "model_name": "ldsr-s2-x4",
                "seed": seed,
                "cache_key": _sha(f"cache:{index}:{seed}"),
                "identity": {"sample_id": sample_id},
                "prediction_sha256": _sha(f"prediction:{index}:{seed}"),
            }
            for seed in range(3407, 3412)
        ]
        samples.append(
            {
                "sample_id": sample_id,
                "predictions": predictions,
                "score": {
                    "name": "ldsr_variance_k5",
                    "cache_key": _sha(f"score-cache:{index}"),
                    "identity": {"sample_id": sample_id},
                    "score_sha256": score_sha256,
                },
                "risk": {
                    "name": "local_l1_risk",
                    "window": 9,
                    "risk_sha256": risk_sha256,
                },
            }
        )
        map_entries.append(
            {
                "sample_id": sample_id,
                "score_sha256": score_sha256,
                "risk_sha256": risk_sha256,
            }
        )
    audit = {
        "schema": "trustsr.phase2b3c-evaluation-cache-audit.v1",
        "split": "internal_test",
        "ordered_sample_ids_sha256": hashlib.sha256(
            canonical_json(sample_ids)
        ).hexdigest(),
        "sample_count": 120,
        "prediction_count": 600,
        "score_count": 120,
        "samples": samples,
    }
    rois = tuple(
        ROIEvaluation(
            sample_id,
            (-1, 0, 1)[(index % 12) // 4],
            index % 4,
            index // 12 + 1,
            20,
            100,
            0.2,
            0.0,
        )
        for index, sample_id in enumerate(sample_ids)
    )
    result = build_phase2b3c_result(
        statistics=build_phase2b3c_statistics(rois),
        radiometry={
            kind: {
                "raw_crop_minimum": 0,
                "raw_crop_maximum": 10000,
                "clipped_high_count": 0,
                "clipped_high_by_band": [0, 0, 0, 0],
            }
            for kind in ("lr", "hr")
        },
        evaluation_id=_sha("evaluation"),
        permit_sha256=_sha("permit"),
        ledger_event_sha256=_sha("ledger"),
        input_receipt_sha256=_sha("receipt"),
        ordered_inputs_sha256=_sha("inputs"),
        ordered_sample_ids_sha256=audit["ordered_sample_ids_sha256"],
        ordered_membership_sha256=_sha("membership"),
        cache_audit_sha256=hashlib.sha256(canonical_json(audit)).hexdigest(),
        map_evidence_sha256=hashlib.sha256(canonical_json(map_entries)).hexdigest(),
        producer_revision="1" * 40,
    )
    provenance = build_cache_provenance(
        {
            "name": "ldsr-s2-x4",
            "scale": 4,
            "implementation_schema_version": 1,
            "opensr_model_version": OPENSR_MODEL_VERSION,
            "torch_version": "2.7.1+cu128",
            "cuda_runtime": "12.8",
            "checkpoint_name": CHECKPOINT_NAME,
            "checkpoint_url": CHECKPOINT_URL,
            "checkpoint_size": CHECKPOINT_SIZE,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "config_sha256": CONFIG_SHA256,
            "device": "cuda",
            "seed": 3407,
            "sampling_steps": 100,
            "sampling_eta": 0.95,
            "sampling_temperature": 1.0,
            "histogram_matching": True,
            "output_policy": "clip_to_[0,1]",
        }
    )
    runtime = build_phase2b3c_runtime_manifest(
        result=result,
        model_provenance=provenance,
        dependencies={
            "python": "3.12",
            "uv_lock_sha256": _sha("lock"),
            "packages": {
                "numpy": "2.2.6",
                "opensr-model": "0.3.1",
                "rasterio": "1.4.3",
                "torch": "2.7.1+cu128",
                "trustsr": "0.1.0",
            },
        },
    )
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
        evaluation_id=_sha("evaluation"),
        permit_sha256=_sha("permit"),
        state="caches_complete",
        sequence=2,
        event_sha256=_sha("ledger"),
        previous_event_sha256=_sha("previous ledger"),
    )
    return dict(
        zip(
            BUNDLE_DOCUMENT_SCHEMAS,
            (result, audit, runtime, replay, ledger),
            strict=True,
        )
    )


def test_semantic_bundle_documents_close_the_acyclic_digest_graph() -> None:
    documents = _semantic_documents()

    verified = verify_phase2b3c_bundle_documents(documents)

    assert verified.phase_decision == "confirmed"
    assert verified.cache_computation_verified is False
    assert verified.acceptance_authorized is False
    assert verified.evaluation_id == _sha("evaluation")


def test_semantic_bundle_rejects_ledger_or_map_digest_drift() -> None:
    documents = _semantic_documents()
    ledger_name = "phase2b3c-access-ledger-snapshot.json"
    documents[ledger_name]["event_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="ledger"):
        verify_phase2b3c_bundle_documents(documents)

    documents = _semantic_documents()
    audit_name = "phase2b3c-evaluation-cache-audit.json"
    documents[audit_name]["samples"][0]["risk"]["risk_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="audit|map"):
        verify_phase2b3c_bundle_documents(documents)
