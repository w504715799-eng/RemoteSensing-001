"""Tests for the Phase 2B3-B canonical bundle boundary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from trustsr.evaluation.phase2b3b_bundle import (
    BUNDLE_DOCUMENT_SCHEMAS,
    BUNDLE_MANIFEST_BASENAME,
    build_phase2b3b_bundle_manifest,
    read_phase2b3b_bundle,
    write_phase2b3b_bundle,
)
from trustsr.jsonio import canonical_json


def _documents() -> dict[str, dict[str, object]]:
    return {
        name: {"schema": schema, "payload": {"value": index}}
        for index, (name, schema) in enumerate(BUNDLE_DOCUMENT_SCHEMAS.items())
    }


def _write(directory: Path, documents: dict[str, dict[str, object]]) -> object:
    values = tuple(documents[name] for name in BUNDLE_DOCUMENT_SCHEMAS)
    return write_phase2b3b_bundle(
        directory,
        result=values[0],
        cache_audit=values[1],
        runtime=values[2],
        replay=values[3],
    )


def test_builds_sorted_manifest_over_exact_canonical_documents() -> None:
    documents = _documents()

    manifest = build_phase2b3b_bundle_manifest(documents)

    assert manifest["schema"] == "trustsr.phase2b3b-bundle-manifest.v1"
    assert manifest["phase"] == "calibration"
    assert [entry["basename"] for entry in manifest["files"]] == sorted(documents)
    for entry in manifest["files"]:
        payload = canonical_json(documents[entry["basename"]])
        assert entry["size_bytes"] == len(payload)
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()


def test_uses_frozen_five_file_allowlist() -> None:
    assert tuple(BUNDLE_DOCUMENT_SCHEMAS) == (
        "phase2b3b-calibration-result.json",
        "phase2b3b-calibration-cache-audit.json",
        "phase2b3b-calibration-runtime.json",
        "phase2b3b-calibration-replay.json",
    )
    assert BUNDLE_MANIFEST_BASENAME == "phase2b3b-bundle-manifest.json"


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
        build_phase2b3b_bundle_manifest(documents)  # type: ignore[arg-type]


def test_writes_and_verifies_bundle_idempotently(tmp_path: Path) -> None:
    directory = tmp_path / "bundle"
    documents = _documents()

    first = _write(directory, documents)
    second = _write(directory, documents)
    verified = read_phase2b3b_bundle(directory)

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


def test_refuses_to_overwrite_existing_different_bundle(tmp_path: Path) -> None:
    directory = tmp_path / "bundle"
    _write(directory, _documents())
    changed = _documents()
    changed[next(iter(changed))]["payload"] = {"value": 999}

    with pytest.raises(ValueError, match="different bytes"):
        _write(directory, changed)


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
        read_phase2b3b_bundle(directory)


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
        read_phase2b3b_bundle(directory)


@pytest.mark.parametrize("mutation", ("fifo", "oversized"))
def test_verifier_rejects_nonregular_or_oversized_file(
    tmp_path: Path, mutation: str
) -> None:
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
        read_phase2b3b_bundle(directory)


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
    import trustsr.evaluation.phase2b3b_bundle as module

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
