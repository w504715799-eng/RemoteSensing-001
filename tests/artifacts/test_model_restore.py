"""Verified model-tree copying for unprivileged Phase 2B3-A restores."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest

from trustsr.artifacts.model_restore import ModelRestoreError, copy_model_trees
from trustsr.jsonio import canonical_json


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    persistent = tmp_path / "persistent"
    sen2 = persistent / "sen2srlite"
    ldsr = persistent / "ldsr-s2"
    (sen2 / "config").mkdir(parents=True)
    ldsr.mkdir()
    (sen2 / "weights.bin").write_bytes(b"sen2-weights")
    (sen2 / "config" / "model.json").write_bytes(b'{"scale":4}\n')
    (ldsr / "weights.ckpt").write_bytes(b"ldsr-weights")
    sen2.chmod(0o750)
    (sen2 / "config").chmod(0o710)
    (sen2 / "weights.bin").chmod(0o640)
    (sen2 / "config" / "model.json").chmod(0o600)
    ldsr.chmod(0o755)
    (ldsr / "weights.ckpt").chmod(0o644)
    return sen2, ldsr


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def _digest(entries: list[dict[str, object]]) -> str:
    return hashlib.sha256(canonical_json({"entries": entries})).hexdigest()


def test_copy_model_trees_publishes_verified_nonwritable_copies(tmp_path: Path) -> None:
    sen2, ldsr = _sources(tmp_path)
    target = tmp_path / "work" / "model-mounts"

    result = copy_model_trees(
        target,
        sen2srlite_source=sen2,
        ldsr_source=ldsr,
    )

    assert result.mode == "copy"
    assert result.sen2srlite_inventory_sha256 == _digest(
        [
            {"mode": 0o750, "path": ".", "type": "directory"},
            {"mode": 0o710, "path": "config", "type": "directory"},
            {
                "mode": 0o600,
                "path": "config/model.json",
                "sha256": hashlib.sha256(b'{"scale":4}\n').hexdigest(),
                "size_bytes": 12,
                "type": "file",
            },
            {
                "mode": 0o640,
                "path": "weights.bin",
                "sha256": hashlib.sha256(b"sen2-weights").hexdigest(),
                "size_bytes": 12,
                "type": "file",
            },
        ]
    )
    assert result.ldsr_inventory_sha256 == _digest(
        [
            {"mode": 0o755, "path": ".", "type": "directory"},
            {
                "mode": 0o644,
                "path": "weights.ckpt",
                "sha256": hashlib.sha256(b"ldsr-weights").hexdigest(),
                "size_bytes": 12,
                "type": "file",
            },
        ]
    )
    assert (target / "sen2srlite/weights.bin").read_bytes() == b"sen2-weights"
    assert (target / "sen2srlite/config/model.json").read_bytes() == b'{"scale":4}\n'
    assert (target / "ldsr-s2/weights.ckpt").read_bytes() == b"ldsr-weights"
    assert _mode(target) == 0o500
    assert _mode(target / "sen2srlite") == 0o550
    assert _mode(target / "sen2srlite/config") == 0o510
    assert _mode(target / "sen2srlite/weights.bin") == 0o440
    assert _mode(target / "sen2srlite/config/model.json") == 0o400
    assert _mode(target / "ldsr-s2") == 0o555
    assert _mode(target / "ldsr-s2/weights.ckpt") == 0o444
    assert _mode(sen2) == 0o750
    assert _mode(sen2 / "weights.bin") == 0o640
    assert _mode(ldsr) == 0o755
    assert not list((tmp_path / "work").glob(".phase2b3a-model-copy.*"))


@pytest.mark.parametrize("unsafe_kind", ["file-symlink", "directory-symlink", "hard-link", "fifo"])
def test_copy_model_trees_rejects_unsafe_source_and_cleans_staging(
    tmp_path: Path, unsafe_kind: str
) -> None:
    sen2, ldsr = _sources(tmp_path)
    if unsafe_kind == "file-symlink":
        (sen2 / "linked.bin").symlink_to(sen2 / "weights.bin")
    elif unsafe_kind == "directory-symlink":
        (sen2 / "linked-config").symlink_to(sen2 / "config", target_is_directory=True)
    elif unsafe_kind == "hard-link":
        os.link(sen2 / "weights.bin", sen2 / "weights-copy.bin")
    else:
        os.mkfifo(sen2 / "pipe")
    target = tmp_path / "work" / "model-mounts"

    with pytest.raises(ModelRestoreError):
        copy_model_trees(
            target,
            sen2srlite_source=sen2,
            ldsr_source=ldsr,
        )

    assert not target.exists()
    assert not list((tmp_path / "work").glob(".phase2b3a-model-copy.*"))


def test_copy_model_trees_rejects_existing_target_without_changing_it(tmp_path: Path) -> None:
    sen2, ldsr = _sources(tmp_path)
    target = tmp_path / "work" / "model-mounts"
    target.mkdir(parents=True)
    marker = target / "owned-by-operator"
    marker.write_bytes(b"keep")

    with pytest.raises(ModelRestoreError, match="destination"):
        copy_model_trees(
            target,
            sen2srlite_source=sen2,
            ldsr_source=ldsr,
        )

    assert marker.read_bytes() == b"keep"


def test_copy_model_trees_detects_corrupt_copy_before_publication(tmp_path: Path) -> None:
    sen2, ldsr = _sources(tmp_path)
    target = tmp_path / "work" / "model-mounts"

    def corrupt_copy(source: object, destination: object) -> None:
        while block := source.read(1024 * 1024):  # type: ignore[attr-defined]
            destination.write(block)  # type: ignore[attr-defined]
        destination.seek(0)  # type: ignore[attr-defined]
        destination.write(b"x")  # type: ignore[attr-defined]

    with pytest.raises(ModelRestoreError, match="inventory"):
        copy_model_trees(
            target,
            sen2srlite_source=sen2,
            ldsr_source=ldsr,
            copy_file=corrupt_copy,
        )

    assert not target.exists()
    assert not list((tmp_path / "work").glob(".phase2b3a-model-copy.*"))
