from __future__ import annotations

import ctypes
import hashlib
import json
import os
import select
import socket
import tarfile
import threading
import time
from pathlib import Path

import pytest

import trustsr.artifacts.workspace_checkpoint as checkpoint
from trustsr.artifacts.workspace_checkpoint import (
    ARCHIVE_ROOTS,
    INPUT_AUDIT_SHA256,
    PROTOCOL_VERSION,
    SCHEMA,
    SELECTION_MANIFEST_SHA256,
    CheckpointError,
    build_checkpoint,
    load_manifest,
    verify_checkpoint,
)
from trustsr.jsonio import canonical_json

POST_BYTES = b'{"fixture":"selection"}\n'
INPUT_AUDIT_BYTES = b'{"fixture":"input-audit"}\n'
POST_SHA256 = hashlib.sha256(POST_BYTES).hexdigest()
FIXTURE_INPUT_AUDIT_SHA256 = hashlib.sha256(INPUT_AUDIT_BYTES).hexdigest()
EXPECTED_MEMBER_NAMES = [
    "trustsr/phase2b1b",
    "trustsr/phase2b1b/selections",
    f"trustsr/phase2b1b/selections/{POST_SHA256}",
    f"trustsr/phase2b1b/selections/{POST_SHA256}/samples.jsonl",
    "trustsr/phase2b2a",
    "trustsr/phase2b2a/input-audits",
    f"trustsr/phase2b2a/input-audits/{POST_SHA256}",
    f"trustsr/phase2b2a/input-audits/{POST_SHA256}/phase2b2a-input-audit.json",
    "trustsr/phase2b3a",
    "trustsr/phase2b3a/cache-z.bin",
    "trustsr/phase2b3a/cache.bin",
]


@pytest.fixture
def frozen_fixture_digests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checkpoint, "SELECTION_MANIFEST_SHA256", POST_SHA256)
    monkeypatch.setattr(checkpoint, "INPUT_AUDIT_SHA256", FIXTURE_INPUT_AUDIT_SHA256)


def _valid_workspace(workspace: Path) -> Path:
    selection = workspace / "trustsr/phase2b1b/selections" / POST_SHA256
    selection.mkdir(parents=True)
    (selection / "samples.jsonl").write_bytes(POST_BYTES)
    audit = workspace / "trustsr/phase2b2a/input-audits" / POST_SHA256
    audit.mkdir(parents=True)
    (audit / "phase2b2a-input-audit.json").write_bytes(INPUT_AUDIT_BYTES)
    cache = workspace / "trustsr/phase2b3a"
    cache.mkdir(parents=True)
    (cache / "cache.bin").write_bytes(b"cache")
    (cache / "cache-z.bin").write_bytes(b"cache-z")
    return workspace


def _build_valid_checkpoint(tmp_path: Path) -> checkpoint.BuiltCheckpoint:
    return build_checkpoint(
        _valid_workspace(tmp_path / "workspace"),
        tmp_path / "out",
        completed_stage="a0",
        reviewed_commit="a" * 40,
    )


def _manifest_payload(built: checkpoint.BuiltCheckpoint) -> dict[str, object]:
    return json.loads(built.manifest_path.read_bytes())


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(canonical_json(payload))


def _mutate_after_archive_starts(
    output_directory: Path, mutation: callable[[], None], *, delay_seconds: float = 0.0
) -> tuple[threading.Thread, threading.Event]:
    completed = threading.Event()

    def mutate() -> None:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if any(output_directory.glob(".phase2b3a-archive-*")):
                if delay_seconds:
                    time.sleep(delay_seconds)
                mutation()
                completed.set()
                return
            time.sleep(0.001)

    worker = threading.Thread(target=mutate)
    worker.start()
    return worker, completed


def _mutate_after_file_opens(
    path: Path, mutation: callable[[], None]
) -> tuple[threading.Thread, threading.Event]:
    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = libc.inotify_init1(os.O_CLOEXEC)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1 failed")
    watch = libc.inotify_add_watch(descriptor, os.fsencode(path), 0x00000020)
    if watch < 0:
        os.close(descriptor)
        raise OSError(ctypes.get_errno(), "inotify_add_watch failed")
    completed = threading.Event()

    def mutate() -> None:
        try:
            readable, _, _ = select.select([descriptor], [], [], 10)
            if readable:
                os.read(descriptor, 4096)
                mutation()
                completed.set()
        finally:
            os.close(descriptor)

    worker = threading.Thread(target=mutate)
    worker.start()
    return worker, completed


def test_production_frozen_digests_match_the_approved_spec() -> None:
    assert SELECTION_MANIFEST_SHA256 == (
        "c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a"
    )
    assert INPUT_AUDIT_SHA256 == "fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b"


def test_build_checkpoint_is_deterministic_and_has_exact_members(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    workspace = _valid_workspace(tmp_path / "workspace")
    first = build_checkpoint(
        workspace,
        tmp_path / "first",
        completed_stage="a0",
        reviewed_commit="a" * 40,
    )
    os.utime(
        workspace / "trustsr/phase2b1b/selections" / POST_SHA256 / "samples.jsonl",
        ns=(1_800_000_000_000_000_000, 1_800_000_000_000_000_000),
    )
    cache = workspace / "trustsr/phase2b3a"
    (cache / "cache.bin").unlink()
    (cache / "cache-z.bin").unlink()
    (cache / "cache-z.bin").write_bytes(b"cache-z")
    (cache / "cache.bin").write_bytes(b"cache")
    second = build_checkpoint(
        workspace,
        tmp_path / "second",
        completed_stage="a0",
        reviewed_commit="a" * 40,
    )

    assert first.archive_path.read_bytes() == second.archive_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert (
        first.manifest.archive_sha256 == hashlib.sha256(first.archive_path.read_bytes()).hexdigest()
    )
    with tarfile.open(first.archive_path, mode="r:") as archive:
        assert [member.name for member in archive.getmembers()] == EXPECTED_MEMBER_NAMES
        assert all(member.uid == member.gid == member.mtime == 0 for member in archive.getmembers())

    assert json.loads(first.manifest_path.read_bytes()) == {
        "archive_basename": first.archive_path.name,
        "archive_sha256": first.manifest.archive_sha256,
        "archive_size_bytes": first.archive_path.stat().st_size,
        "completed_stage": "a0",
        "input_audit_sha256": FIXTURE_INPUT_AUDIT_SHA256,
        "protocol_version": PROTOCOL_VERSION,
        "reviewed_commit": "a" * 40,
        "roots": list(ARCHIVE_ROOTS),
        "schema": SCHEMA,
        "selection_manifest_sha256": POST_SHA256,
    }
    assert first.manifest_path.read_bytes() == canonical_json(
        json.loads(first.manifest_path.read_bytes())
    )


def test_build_checkpoint_manifest_uses_active_frozen_digests(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    built = _build_valid_checkpoint(tmp_path)

    assert built.manifest.selection_manifest_sha256 == POST_SHA256
    assert built.manifest.input_audit_sha256 == FIXTURE_INPUT_AUDIT_SHA256
    assert verify_checkpoint(built.archive_path, built.manifest_path) == built.manifest


def test_build_checkpoint_rejects_missing_archive_root(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    workspace = _valid_workspace(tmp_path / "workspace")
    (workspace / "trustsr/phase2b2a").rename(workspace / "trustsr/not-phase2b2a")

    with pytest.raises(CheckpointError, match="missing archive root"):
        build_checkpoint(
            workspace, tmp_path / "out", completed_stage="a0", reviewed_commit="a" * 40
        )


@pytest.mark.parametrize(("kind", "target"), [("file", "cache.bin"), ("directory", "dir-link")])
def test_build_checkpoint_rejects_symlink_sources(
    tmp_path: Path, frozen_fixture_digests: None, kind: str, target: str
) -> None:
    workspace = _valid_workspace(tmp_path / "workspace")
    phase = workspace / "trustsr/phase2b3a"
    if kind == "file":
        (phase / "cache.bin").unlink()
        (phase / target).symlink_to("target.bin")
    else:
        (phase / target).symlink_to(phase, target_is_directory=True)

    with pytest.raises(CheckpointError, match="symlink"):
        build_checkpoint(
            workspace, tmp_path / "out", completed_stage="a0", reviewed_commit="a" * 40
        )


def test_build_checkpoint_rejects_fifo_source(tmp_path: Path, frozen_fixture_digests: None) -> None:
    workspace = _valid_workspace(tmp_path / "workspace")
    os.mkfifo(workspace / "trustsr/phase2b3a/pipe")

    with pytest.raises(CheckpointError, match="regular files and directories"):
        build_checkpoint(
            workspace, tmp_path / "out", completed_stage="a0", reviewed_commit="a" * 40
        )


def test_build_checkpoint_rejects_socket_source(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    workspace = _valid_workspace(tmp_path / "workspace")
    socket_path = workspace / "trustsr/phase2b3a/socket"
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(socket_path))
    try:
        with pytest.raises(CheckpointError, match="regular files and directories"):
            build_checkpoint(
                workspace, tmp_path / "out", completed_stage="a0", reviewed_commit="a" * 40
            )
    finally:
        listener.close()


def test_build_checkpoint_rejects_source_hard_link(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    workspace = _valid_workspace(tmp_path / "workspace")
    source = workspace / "trustsr/phase2b3a/cache.bin"
    os.link(source, workspace / "trustsr/phase2b3a/cache-copy.bin")

    with pytest.raises(CheckpointError, match="hard link"):
        build_checkpoint(
            workspace, tmp_path / "out", completed_stage="a0", reviewed_commit="a" * 40
        )


def test_build_checkpoint_rejects_uppercase_reviewed_commit(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    with pytest.raises(CheckpointError, match="reviewed commit"):
        build_checkpoint(
            _valid_workspace(tmp_path / "workspace"),
            tmp_path / "out",
            completed_stage="a0",
            reviewed_commit="A" * 40,
        )


def test_build_checkpoint_rejects_changed_frozen_input(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    workspace = _valid_workspace(tmp_path / "workspace")
    (workspace / "trustsr/phase2b1b/selections" / POST_SHA256 / "samples.jsonl").write_bytes(
        b"changed"
    )

    with pytest.raises(CheckpointError, match="frozen selection"):
        build_checkpoint(
            workspace, tmp_path / "out", completed_stage="a0", reviewed_commit="a" * 40
        )


def test_build_checkpoint_hashes_the_frozen_stream_written_to_the_archive(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    workspace = _valid_workspace(tmp_path / "workspace")
    before_selection = workspace / "trustsr/phase2b1b/000-before-selection.bin"
    before_selection.write_bytes(b"x" * (16 * 1024 * 1024))
    selection = workspace / "trustsr/phase2b1b/selections" / POST_SHA256 / "samples.jsonl"
    output_directory = tmp_path / "out"
    output_directory.mkdir()
    worker, mutated = _mutate_after_archive_starts(
        output_directory, lambda: selection.write_bytes(b"mutated after validation")
    )
    try:
        with pytest.raises(CheckpointError, match="frozen selection"):
            build_checkpoint(
                workspace,
                output_directory,
                completed_stage="a0",
                reviewed_commit="a" * 40,
            )
    finally:
        worker.join(timeout=10)
    assert mutated.is_set()


def test_build_checkpoint_rejects_intermediate_source_symlink(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    workspace = _valid_workspace(tmp_path / "workspace")
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "trustsr").rename(outside / "trustsr")
    (workspace / "trustsr").symlink_to(outside / "trustsr", target_is_directory=True)

    with pytest.raises(CheckpointError, match="source|archive root|symlink"):
        build_checkpoint(
            workspace, tmp_path / "out", completed_stage="a0", reviewed_commit="a" * 40
        )


def test_build_checkpoint_rejects_directory_mutation_after_scandir_snapshot(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    workspace = _valid_workspace(tmp_path / "workspace")
    phase_root = workspace / "trustsr/phase2b3a"
    (phase_root / "000-slow.bin").write_bytes(b"x" * (64 * 1024 * 1024))
    output_directory = tmp_path / "out"
    output_directory.mkdir()
    worker, mutated = _mutate_after_file_opens(
        phase_root / "000-slow.bin",
        lambda: (phase_root / "new-after-snapshot.bin").write_bytes(b"new"),
    )
    try:
        with pytest.raises(CheckpointError, match="source directory changed"):
            build_checkpoint(
                workspace,
                output_directory,
                completed_stage="a0",
                reviewed_commit="a" * 40,
            )
    finally:
        worker.join(timeout=10)
    assert mutated.is_set()


def test_load_manifest_rejects_noncanonical_json(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    built.manifest_path.write_bytes(built.manifest_path.read_bytes() + b"\n")

    with pytest.raises(CheckpointError, match="canonical"):
        load_manifest(built.manifest_path)


def test_load_manifest_rejects_extra_key(tmp_path: Path, frozen_fixture_digests: None) -> None:
    built = _build_valid_checkpoint(tmp_path)
    payload = _manifest_payload(built)
    payload["unexpected"] = "value"
    _write_manifest(built.manifest_path, payload)

    with pytest.raises(CheckpointError, match="exact key set"):
        load_manifest(built.manifest_path)


@pytest.mark.parametrize("field", ["selection_manifest_sha256", "input_audit_sha256"])
def test_load_manifest_rejects_nonfrozen_input_digest(
    tmp_path: Path, frozen_fixture_digests: None, field: str
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    payload = _manifest_payload(built)
    payload[field] = "b" * 64
    _write_manifest(built.manifest_path, payload)

    with pytest.raises(CheckpointError, match="frozen"):
        load_manifest(built.manifest_path)


def test_load_manifest_rejects_invalid_schema_values(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    payload = _manifest_payload(built)
    payload["protocol_version"] = True
    _write_manifest(built.manifest_path, payload)

    with pytest.raises(CheckpointError, match="protocol version"):
        load_manifest(built.manifest_path)


def test_verify_checkpoint_rejects_wrong_declared_size(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    payload = _manifest_payload(built)
    payload["archive_size_bytes"] = int(payload["archive_size_bytes"]) + 1
    _write_manifest(built.manifest_path, payload)

    with pytest.raises(CheckpointError, match="size"):
        verify_checkpoint(built.archive_path, built.manifest_path)


def test_verify_checkpoint_rejects_wrong_declared_digest(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    payload = _manifest_payload(built)
    payload["archive_sha256"] = "b" * 64
    payload["archive_basename"] = f"phase2b3a-workspace-a0-{'b' * 64}.tar"
    _write_manifest(built.manifest_path, payload)
    archive_path = built.archive_path.with_name(str(payload["archive_basename"]))
    built.archive_path.rename(archive_path)

    with pytest.raises(CheckpointError, match="digest"):
        verify_checkpoint(archive_path, built.manifest_path)


def test_load_manifest_rejects_wrong_stage(tmp_path: Path, frozen_fixture_digests: None) -> None:
    built = _build_valid_checkpoint(tmp_path)
    payload = _manifest_payload(built)
    payload["completed_stage"] = "a3"
    payload["archive_basename"] = f"phase2b3a-workspace-a3-{payload['archive_sha256']}.tar"
    _write_manifest(built.manifest_path, payload)

    with pytest.raises(CheckpointError, match="completed stage"):
        load_manifest(built.manifest_path)


@pytest.mark.parametrize("basename", ["archive.tar", "nested/archive.tar", "../archive.tar"])
def test_load_manifest_rejects_unsafe_archive_basename(
    tmp_path: Path, frozen_fixture_digests: None, basename: str
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    payload = _manifest_payload(built)
    payload["archive_basename"] = basename
    _write_manifest(built.manifest_path, payload)

    with pytest.raises(CheckpointError, match="digest-bound"):
        load_manifest(built.manifest_path)
