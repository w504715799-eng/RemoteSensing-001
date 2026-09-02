from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import os
import select
import shutil
import socket
import tarfile
import threading
import time
from pathlib import Path
from typing import BinaryIO

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
    publish_checkpoint,
    restore_checkpoint,
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
PHASE_EVIDENCE_FILES = {
    stage: tuple(
        f"phase2b3a-{stage}-{suffix}.json"
        for suffix in ("result", "cache-audit", "runtime", "replay")
    )
    for stage in ("a1", "a2")
}


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


def _write_stage_evidence(workspace: Path, stage: str) -> Path:
    result_directory = workspace / "trustsr/phase2b3a/results" / POST_SHA256
    result_directory.mkdir(parents=True, exist_ok=True)
    payloads = {
        name: canonical_json(
            {
                "byte_identical": True,
                "name": name,
                "phase": stage,
            }
        )
        for name in PHASE_EVIDENCE_FILES[stage]
    }
    for name, payload in payloads.items():
        (result_directory / name).write_bytes(payload)
    manifest = {
        "schema": "trustsr.phase2b3a-bundle-manifest.v1",
        "phase": stage,
        "files": [
            {
                "basename": name,
                "size_bytes": len(payloads[name]),
                "sha256": hashlib.sha256(payloads[name]).hexdigest(),
            }
            for name in sorted(payloads)
        ],
    }
    manifest_path = result_directory / "phase2b3a-bundle-manifest.json"
    manifest_path.write_bytes(canonical_json(manifest))
    return manifest_path


def _stage_evidence_members(
    stage: str, *, wrong_digest: bool = False
) -> list[tuple[tarfile.TarInfo, bytes]]:
    result_root = f"trustsr/phase2b3a/results/{POST_SHA256}"
    payloads = {
        name: canonical_json({"name": name, "phase": stage})
        for name in PHASE_EVIDENCE_FILES[stage]
    }
    entries = [
        {
            "basename": name,
            "size_bytes": len(payloads[name]),
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
        }
        for name in sorted(payloads)
    ]
    if wrong_digest:
        entries[0]["sha256"] = "b" * 64
    manifest = canonical_json(
        {
            "schema": "trustsr.phase2b3a-bundle-manifest.v1",
            "phase": stage,
            "files": entries,
        }
    )
    return [
        _tar_directory("trustsr/phase2b3a/results"),
        _tar_directory(result_root),
        *[
            _tar_file(f"{result_root}/{name}", payloads[name])
            for name in sorted(payloads)
        ],
        _tar_file(f"{result_root}/phase2b3a-bundle-manifest.json", manifest),
    ]


def _tree_file_digests(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


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


def _tar_directory(name: str) -> tuple[tarfile.TarInfo, bytes]:
    member = tarfile.TarInfo(name)
    member.type = tarfile.DIRTYPE
    member.mode = 0o755
    return member, b""


def _tar_file(name: str, payload: bytes) -> tuple[tarfile.TarInfo, bytes]:
    member = tarfile.TarInfo(name)
    member.type = tarfile.REGTYPE
    member.mode = 0o644
    member.size = len(payload)
    return member, payload


def _minimal_checkpoint_members() -> list[tuple[tarfile.TarInfo, bytes]]:
    return [
        _tar_directory("trustsr/phase2b1b"),
        _tar_directory("trustsr/phase2b1b/selections"),
        _tar_directory(f"trustsr/phase2b1b/selections/{POST_SHA256}"),
        _tar_file(
            f"trustsr/phase2b1b/selections/{POST_SHA256}/samples.jsonl", POST_BYTES
        ),
        _tar_directory("trustsr/phase2b2a"),
        _tar_directory("trustsr/phase2b2a/input-audits"),
        _tar_directory(f"trustsr/phase2b2a/input-audits/{POST_SHA256}"),
        _tar_file(
            (
                f"trustsr/phase2b2a/input-audits/{POST_SHA256}/"
                "phase2b2a-input-audit.json"
            ),
            INPUT_AUDIT_BYTES,
        ),
        _tar_directory("trustsr/phase2b3a"),
    ]


def _write_tar_checkpoint(
    directory: Path,
    members: list[tuple[tarfile.TarInfo, bytes]],
    *,
    completed_stage: str = "a0",
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "checkpoint.tar"
    with tarfile.open(temporary, mode="w:", format=tarfile.USTAR_FORMAT) as archive:
        for member, payload in members:
            archive.addfile(member, io.BytesIO(payload) if member.isreg() else None)
    archive_bytes = temporary.read_bytes()
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    archive_path = directory / f"phase2b3a-workspace-{completed_stage}-{archive_sha256}.tar"
    temporary.rename(archive_path)
    manifest_path = archive_path.with_suffix(".json")
    _write_manifest(
        manifest_path,
        {
            "archive_basename": archive_path.name,
            "archive_sha256": archive_sha256,
            "archive_size_bytes": len(archive_bytes),
            "completed_stage": completed_stage,
            "input_audit_sha256": checkpoint.INPUT_AUDIT_SHA256,
            "protocol_version": PROTOCOL_VERSION,
            "reviewed_commit": "a" * 40,
            "roots": list(ARCHIVE_ROOTS),
            "schema": SCHEMA,
            "selection_manifest_sha256": checkpoint.SELECTION_MANIFEST_SHA256,
        },
    )
    return archive_path, manifest_path


def _malicious_checkpoint_members(kind: str) -> list[tuple[tarfile.TarInfo, bytes]]:
    members = _minimal_checkpoint_members()
    if kind == "absolute":
        members.append(_tar_file("/trustsr/phase2b3a/escape", b"escape"))
    elif kind == "traversal":
        members.append(_tar_file("trustsr/phase2b3a/../escape", b"escape"))
    elif kind in {"symlink", "hardlink", "fifo"}:
        member = tarfile.TarInfo(f"trustsr/phase2b3a/{kind}")
        member.type = {
            "symlink": tarfile.SYMTYPE,
            "hardlink": tarfile.LNKTYPE,
            "fifo": tarfile.FIFOTYPE,
        }[kind]
        member.linkname = "trustsr/phase2b3a" if kind != "fifo" else ""
        members.append((member, b""))
    elif kind == "unexpected-root":
        members.append(_tar_directory("trustsr/unexpected"))
    elif kind == "duplicate":
        members.append(_tar_directory("trustsr/phase2b3a"))
    elif kind == "missing-root":
        members = [member for member in members if member[0].name != "trustsr/phase2b3a"]
    elif kind == "file-child-conflict":
        members.extend(
            [
                _tar_file("trustsr/phase2b3a/conflict", b"file"),
                _tar_file("trustsr/phase2b3a/conflict/child", b"child"),
            ]
        )
    else:
        raise AssertionError(f"unknown malicious checkpoint kind: {kind}")
    return members


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
    manifest_path = archive_path.with_suffix(".json")
    built.archive_path.rename(archive_path)
    built.manifest_path.rename(manifest_path)

    with pytest.raises(CheckpointError, match="digest"):
        verify_checkpoint(archive_path, manifest_path)


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


def test_publish_checkpoint_copies_archive_before_accepting_manifest(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    copy_started = False

    def observe_copy(source: BinaryIO, target: BinaryIO) -> None:
        nonlocal copy_started
        copy_started = True
        assert not (persistent / built.archive_path.name).exists()
        assert not (persistent / built.manifest_path.name).exists()
        shutil.copyfileobj(source, target)

    archive_path, manifest_path = publish_checkpoint(
        built, persistent, copy_file=observe_copy
    )

    assert copy_started
    assert archive_path.read_bytes() == built.archive_path.read_bytes()
    assert manifest_path.read_bytes() == built.manifest_path.read_bytes()
    assert sorted(path.name for path in persistent.iterdir()) == sorted(
        [built.archive_path.name, built.manifest_path.name]
    )
    assert not any("latest" in path.name for path in persistent.iterdir())


def test_publish_checkpoint_is_idempotent_but_never_overwrites(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()

    first = publish_checkpoint(built, persistent)
    second = publish_checkpoint(built, persistent)

    assert first == second
    assert first[0].read_bytes() == built.archive_path.read_bytes()
    first[1].write_bytes(b"different")
    with pytest.raises(CheckpointError, match="collision"):
        publish_checkpoint(built, persistent)
    assert first[1].read_bytes() == b"different"
    assert first[0].read_bytes() == built.archive_path.read_bytes()


def test_publish_checkpoint_rejects_different_archive_collision(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    collision = persistent / built.archive_path.name
    collision.write_bytes(b"different")

    with pytest.raises(CheckpointError, match="collision"):
        publish_checkpoint(built, persistent)

    assert collision.read_bytes() == b"different"
    assert not (persistent / built.manifest_path.name).exists()


def test_publish_checkpoint_rejects_symlink_collision(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    collision = persistent / built.archive_path.name
    collision.symlink_to(outside)

    with pytest.raises(CheckpointError, match="collision"):
        publish_checkpoint(built, persistent)

    assert collision.is_symlink()
    assert outside.read_bytes() == b"outside"


def test_publish_checkpoint_rejects_stale_partial(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    stale = persistent / f".{built.archive_path.name}.part"
    stale.write_bytes(b"stale")

    with pytest.raises(CheckpointError, match="collision"):
        publish_checkpoint(built, persistent)

    assert stale.read_bytes() == b"stale"


def test_publish_checkpoint_rejects_unexpected_persistent_entry(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    (persistent / "unrelated").mkdir()

    with pytest.raises(CheckpointError, match="unexpected"):
        publish_checkpoint(built, persistent)


def test_publish_checkpoint_tolerates_only_real_checkpoint_lock_directory(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    (persistent / ".checkpoint.lock").mkdir()

    archive_path, manifest_path = publish_checkpoint(built, persistent)

    assert archive_path.is_file()
    assert manifest_path.is_file()
    assert (persistent / ".checkpoint.lock").is_dir()


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_publish_checkpoint_rejects_non_directory_checkpoint_lock(
    tmp_path: Path, frozen_fixture_digests: None, kind: str
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    lock = persistent / ".checkpoint.lock"
    if kind == "file":
        lock.write_bytes(b"lock")
    else:
        lock.symlink_to(tmp_path, target_is_directory=True)

    with pytest.raises(CheckpointError, match="lock"):
        publish_checkpoint(built, persistent)


def test_publish_checkpoint_rejects_multiply_linked_final(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    final = persistent / built.archive_path.name
    os.link(built.archive_path, final)

    with pytest.raises(CheckpointError, match="collision"):
        publish_checkpoint(built, persistent)

    assert final.stat().st_nlink == 2


def test_publish_checkpoint_interrupted_copy_leaves_only_hidden_partial(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()

    def interrupt_copy(source: BinaryIO, target: BinaryIO) -> None:
        target.write(source.read(1))
        target.flush()
        raise RuntimeError("simulated interrupted copy")

    with pytest.raises(CheckpointError, match="copy"):
        publish_checkpoint(built, persistent, copy_file=interrupt_copy)

    entries = list(persistent.iterdir())
    assert len(entries) == 1
    assert entries[0].name.startswith(".")
    assert entries[0].name.endswith(".part")
    assert entries[0].read_bytes() == built.archive_path.read_bytes()[:1]
    assert not (persistent / built.archive_path.name).exists()
    assert not (persistent / built.manifest_path.name).exists()


def test_publish_checkpoint_rejects_local_manifest_mutation_during_archive_copy(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    mutated = _manifest_payload(built)
    mutated["reviewed_commit"] = "b" * 40

    def copy_then_mutate_manifest(source: BinaryIO, target: BinaryIO) -> None:
        shutil.copyfileobj(source, target)
        _write_manifest(built.manifest_path, mutated)

    with pytest.raises(CheckpointError, match="manifest changed"):
        publish_checkpoint(built, persistent, copy_file=copy_then_mutate_manifest)

    assert (persistent / built.archive_path.name).is_file()
    assert not (persistent / built.manifest_path.name).exists()


@pytest.mark.parametrize("kind", ["different", "identical", "symlink", "multiply-linked"])
def test_publish_checkpoint_rejects_manifest_only_state_before_archive_publication(
    tmp_path: Path, frozen_fixture_digests: None, kind: str
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    manifest_collision = persistent / built.manifest_path.name
    if kind == "different":
        payload = _manifest_payload(built)
        payload["reviewed_commit"] = "b" * 40
        _write_manifest(manifest_collision, payload)
    elif kind == "identical":
        manifest_collision.write_bytes(built.manifest_path.read_bytes())
    elif kind == "symlink":
        outside = tmp_path / "outside-manifest"
        outside.write_bytes(built.manifest_path.read_bytes())
        manifest_collision.symlink_to(outside)
    else:
        outside = tmp_path / "outside-manifest"
        outside.write_bytes(built.manifest_path.read_bytes())
        os.link(outside, manifest_collision)
    before = manifest_collision.lstat()

    with pytest.raises(CheckpointError, match="collision|unpaired|unsafe"):
        publish_checkpoint(built, persistent)

    after = manifest_collision.lstat()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert not (persistent / built.archive_path.name).exists()


def test_publish_checkpoint_rolls_back_new_archive_after_manifest_race(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    manifest_collision = persistent / built.manifest_path.name
    payload = _manifest_payload(built)
    payload["reviewed_commit"] = "b" * 40
    collision_bytes = canonical_json(payload)

    def copy_then_race_manifest(source: BinaryIO, target: BinaryIO) -> None:
        shutil.copyfileobj(source, target)
        manifest_collision.write_bytes(collision_bytes)

    with pytest.raises(CheckpointError, match="collision"):
        publish_checkpoint(built, persistent, copy_file=copy_then_race_manifest)

    assert manifest_collision.read_bytes() == collision_bytes
    assert not (persistent / built.archive_path.name).exists()


def test_publish_checkpoint_completes_identical_archive_only_state(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    archive_path = persistent / built.archive_path.name
    archive_path.write_bytes(built.archive_path.read_bytes())

    published_archive, published_manifest = publish_checkpoint(built, persistent)

    assert published_archive == archive_path
    assert published_manifest.read_bytes() == built.manifest_path.read_bytes()


@pytest.mark.parametrize(
    "kind",
    [
        "unrelated",
        "partial",
        "unpaired",
        "lock-file",
        "lock-symlink",
        "selected-archive-multiply-linked",
        "other-pair-multiply-linked",
    ],
)
def test_verify_and_restore_reject_unsafe_persistent_hygiene(
    tmp_path: Path,
    frozen_fixture_digests: None,
    capsys: pytest.CaptureFixture[str],
    kind: str,
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    archive_path, manifest_path = publish_checkpoint(built, persistent)
    if kind == "unrelated":
        (persistent / "unrelated").write_bytes(b"unexpected")
    elif kind == "partial":
        (persistent / ".stale.part").write_bytes(b"partial")
    elif kind == "unpaired":
        (persistent / f"phase2b3a-workspace-a2-{'b' * 64}.tar").write_bytes(b"unpaired")
    elif kind == "lock-file":
        (persistent / ".checkpoint.lock").write_bytes(b"unsafe lock")
    elif kind == "lock-symlink":
        outside = tmp_path / "outside-lock"
        outside.mkdir()
        (persistent / ".checkpoint.lock").symlink_to(outside, target_is_directory=True)
    elif kind == "selected-archive-multiply-linked":
        os.link(archive_path, tmp_path / "selected-archive-hardlink")
    else:
        other_archive = persistent / f"phase2b3a-workspace-a2-{'b' * 64}.tar"
        outside = tmp_path / "other-archive-hardlink"
        outside.write_bytes(b"other")
        os.link(outside, other_archive)
        (other_archive.with_suffix(".json")).write_bytes(b"paired but multiply linked")

    assert checkpoint.main(["verify", str(persistent), manifest_path.name]) == 2
    verify_output = capsys.readouterr()
    assert verify_output.out == ""
    assert "error:" in verify_output.err

    live = tmp_path / "live"
    live.mkdir()
    with pytest.raises(CheckpointError, match="persistent|entry|partial|pair|lock|link"):
        restore_checkpoint(
            persistent,
            manifest_path.name,
            live,
            expected_reviewed_commit="a" * 40,
        )
    assert list(live.iterdir()) == []


def test_verify_and_restore_accept_real_checkpoint_lock_directory(
    tmp_path: Path,
    frozen_fixture_digests: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    _, manifest_path = publish_checkpoint(built, persistent)
    (persistent / ".checkpoint.lock").mkdir()

    assert checkpoint.main(["verify", str(persistent), manifest_path.name]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verify"
    live = tmp_path / "live"
    live.mkdir()
    assert restore_checkpoint(
        persistent,
        manifest_path.name,
        live,
        expected_reviewed_commit="a" * 40,
    ) == live / "trustsr"


def test_multiple_syntactic_checkpoint_pairs_coexist_without_unselected_verification(
    tmp_path: Path,
    frozen_fixture_digests: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    other_archive = persistent / f"phase2b3a-workspace-a2-{'b' * 64}.tar"
    other_manifest = other_archive.with_suffix(".json")
    other_archive.write_bytes(b"syntactically paired but deliberately not a tar")
    other_manifest.write_bytes(b"syntactically paired but deliberately not JSON")

    _, manifest_path = publish_checkpoint(built, persistent)

    assert other_archive.read_bytes().startswith(b"syntactically paired")
    assert other_manifest.read_bytes().startswith(b"syntactically paired")
    assert checkpoint.main(["verify", str(persistent), manifest_path.name]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verify"
    live = tmp_path / "live"
    live.mkdir()
    assert restore_checkpoint(
        persistent,
        manifest_path.name,
        live,
        expected_reviewed_commit="a" * 40,
    ) == live / "trustsr"


def test_verify_checkpoint_accepts_complete_safe_inventory(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    archive_path, manifest_path = _write_tar_checkpoint(
        tmp_path / "checkpoint", _minimal_checkpoint_members()
    )

    manifest = verify_checkpoint(archive_path, manifest_path)

    assert manifest.archive_basename == archive_path.name


@pytest.mark.parametrize(
    "kind",
    [
        "absolute",
        "traversal",
        "symlink",
        "hardlink",
        "fifo",
        "unexpected-root",
        "duplicate",
        "missing-root",
        "file-child-conflict",
    ],
)
def test_verify_checkpoint_rejects_unsafe_full_member_inventory(
    tmp_path: Path, frozen_fixture_digests: None, kind: str
) -> None:
    archive_path, manifest_path = _write_tar_checkpoint(
        tmp_path / kind, _malicious_checkpoint_members(kind)
    )

    with pytest.raises(CheckpointError, match="archive"):
        verify_checkpoint(archive_path, manifest_path)


def test_verify_checkpoint_rejects_missing_directory_parent_record(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    members = _minimal_checkpoint_members()
    members = [
        member
        for member in members
        if member[0].name != "trustsr/phase2b1b/selections"
    ]
    archive_path, manifest_path = _write_tar_checkpoint(tmp_path / "checkpoint", members)

    with pytest.raises(CheckpointError, match="archive.*parent"):
        verify_checkpoint(archive_path, manifest_path)


def test_verify_checkpoint_rejects_wrong_manifest_basename(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    archive_path, manifest_path = _write_tar_checkpoint(
        tmp_path / "checkpoint", _minimal_checkpoint_members()
    )
    wrong_manifest = manifest_path.with_name("wrong.json")
    manifest_path.rename(wrong_manifest)

    with pytest.raises(CheckpointError, match="manifest basename"):
        verify_checkpoint(archive_path, wrong_manifest)


def test_load_manifest_rejects_oversized_archive_declaration(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    _, manifest_path = _write_tar_checkpoint(
        tmp_path / "checkpoint", _minimal_checkpoint_members()
    )
    payload = json.loads(manifest_path.read_bytes())
    payload["archive_size_bytes"] = 1 << 63
    _write_manifest(manifest_path, payload)

    with pytest.raises(CheckpointError, match="archive size is invalid"):
        load_manifest(manifest_path)


@pytest.mark.parametrize("stage", ["a1", "a2"])
def test_build_checkpoint_requires_digest_bound_stage_evidence(
    tmp_path: Path, frozen_fixture_digests: None, stage: str
) -> None:
    workspace = _valid_workspace(tmp_path / "workspace")
    _write_stage_evidence(workspace, stage)

    built = build_checkpoint(
        workspace,
        tmp_path / "out",
        completed_stage=stage,
        reviewed_commit="a" * 40,
    )

    assert built.manifest.completed_stage == stage
    assert verify_checkpoint(built.archive_path, built.manifest_path) == built.manifest


def test_build_checkpoint_rejects_missing_stage_evidence_before_archive_open(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    output = tmp_path / "out"
    output.mkdir()

    with pytest.raises(CheckpointError, match="evidence manifest"):
        build_checkpoint(
            _valid_workspace(tmp_path / "workspace"),
            output,
            completed_stage="a1",
            reviewed_commit="a" * 40,
        )

    assert list(output.iterdir()) == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("phase", "phase"),
        ("missing-entry", "four expected"),
        ("missing-file", "evidence file"),
        ("size", "size"),
        ("digest", "digest"),
        ("oversized", "size"),
        ("noncanonical", "canonical"),
    ],
)
def test_build_checkpoint_rejects_unbound_stage_evidence(
    tmp_path: Path,
    frozen_fixture_digests: None,
    mutation: str,
    message: str,
) -> None:
    workspace = _valid_workspace(tmp_path / "workspace")
    manifest_path = _write_stage_evidence(workspace, "a1")
    manifest = json.loads(manifest_path.read_bytes())
    if mutation == "phase":
        manifest["phase"] = "a2"
    elif mutation == "missing-entry":
        manifest["files"].pop()
    elif mutation == "missing-file":
        (manifest_path.parent / manifest["files"][0]["basename"]).unlink()
    elif mutation == "size":
        manifest["files"][0]["size_bytes"] += 1
    elif mutation == "digest":
        manifest["files"][0]["sha256"] = "b" * 64
    elif mutation == "oversized":
        manifest["files"][0]["size_bytes"] = 5 * 1024**2 + 1
    elif mutation != "noncanonical":
        raise AssertionError(f"unknown mutation: {mutation}")
    manifest_path.write_bytes(canonical_json(manifest))
    if mutation == "noncanonical":
        manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")

    with pytest.raises(CheckpointError, match=message):
        build_checkpoint(
            workspace,
            tmp_path / "out",
            completed_stage="a1",
            reviewed_commit="a" * 40,
        )


def test_restore_checkpoint_publishes_only_after_full_validation(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    _, manifest_path = publish_checkpoint(built, persistent)
    live = tmp_path / "new-session"
    live.mkdir()

    restored = restore_checkpoint(
        persistent,
        manifest_path.name,
        live,
        expected_reviewed_commit="a" * 40,
    )

    assert restored == live / "trustsr"
    assert _tree_file_digests(restored) == _tree_file_digests(
        tmp_path / "workspace/trustsr"
    )
    assert not any(path.name.startswith(".phase2b3a-restore.") for path in live.iterdir())
    assert all(
        (path.stat().st_mode & 0o777) == (0o700 if path.is_dir() else 0o600)
        for path in [restored, *restored.rglob("*")]
    )


@pytest.mark.parametrize("kind", ["nonempty", "empty", "symlink"])
def test_restore_checkpoint_never_merges_or_replaces_live_trustsr(
    tmp_path: Path, frozen_fixture_digests: None, kind: str
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    _, manifest_path = publish_checkpoint(built, persistent)
    live = tmp_path / "new-session"
    live.mkdir()
    destination = live / "trustsr"
    if kind == "symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        destination.symlink_to(outside, target_is_directory=True)
    else:
        destination.mkdir()
        if kind == "nonempty":
            (destination / "keep").write_bytes(b"keep")

    with pytest.raises(CheckpointError, match="collision"):
        restore_checkpoint(
            persistent,
            manifest_path.name,
            live,
            expected_reviewed_commit="a" * 40,
        )

    if kind == "nonempty":
        assert (destination / "keep").read_bytes() == b"keep"
    elif kind == "symlink":
        assert destination.is_symlink()


def test_restore_checkpoint_rejects_mismatched_reviewed_commit(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    _, manifest_path = publish_checkpoint(built, persistent)
    live = tmp_path / "new-session"
    live.mkdir()

    with pytest.raises(CheckpointError, match="reviewed commit"):
        restore_checkpoint(
            persistent,
            manifest_path.name,
            live,
            expected_reviewed_commit="b" * 40,
        )

    assert not (live / "trustsr").exists()
    assert list(live.iterdir()) == []


def test_restore_checkpoint_rejects_unsafe_inventory_before_publication(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    persistent = tmp_path / "persistent"
    _, manifest_path = _write_tar_checkpoint(
        persistent, _malicious_checkpoint_members("traversal")
    )
    live = tmp_path / "new-session"
    live.mkdir()

    with pytest.raises(CheckpointError, match="archive"):
        restore_checkpoint(
            persistent,
            manifest_path.name,
            live,
            expected_reviewed_commit="a" * 40,
        )

    assert list(live.iterdir()) == []


def test_restore_checkpoint_revalidates_frozen_inputs_after_extraction(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    members = _minimal_checkpoint_members()
    selection_name = f"trustsr/phase2b1b/selections/{POST_SHA256}/samples.jsonl"
    members = [
        _tar_file(selection_name, b"changed") if member[0].name == selection_name else member
        for member in members
    ]
    persistent = tmp_path / "persistent"
    _, manifest_path = _write_tar_checkpoint(persistent, members)
    live = tmp_path / "new-session"
    live.mkdir()

    with pytest.raises(CheckpointError, match="frozen selection"):
        restore_checkpoint(
            persistent,
            manifest_path.name,
            live,
            expected_reviewed_commit="a" * 40,
        )

    assert list(live.iterdir()) == []


def test_restore_checkpoint_revalidates_stage_evidence_after_extraction(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    members = [
        *_minimal_checkpoint_members(),
        *_stage_evidence_members("a1", wrong_digest=True),
    ]
    persistent = tmp_path / "persistent"
    _, manifest_path = _write_tar_checkpoint(
        persistent, members, completed_stage="a1"
    )
    live = tmp_path / "new-session"
    live.mkdir()
    preserved = live / ".phase2b3a-restore.preserved"
    preserved.write_bytes(b"preserved")

    with pytest.raises(CheckpointError, match="evidence.*digest"):
        restore_checkpoint(
            persistent,
            manifest_path.name,
            live,
            expected_reviewed_commit="a" * 40,
        )

    assert preserved.read_bytes() == b"preserved"
    assert not (live / "trustsr").exists()
    assert sorted(path.name for path in live.iterdir()) == [preserved.name]


def test_restore_checkpoint_independently_reverifies_staged_archive_copy(
    tmp_path: Path, frozen_fixture_digests: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    archive_path, manifest_path = publish_checkpoint(built, persistent)
    live = tmp_path / "new-session"
    live.mkdir()
    original_verify = checkpoint.verify_checkpoint
    calls = 0

    def mutate_after_first_verification(
        candidate_archive: Path, candidate_manifest: Path
    ) -> checkpoint.CheckpointManifest:
        nonlocal calls
        manifest = original_verify(candidate_archive, candidate_manifest)
        calls += 1
        if calls == 1:
            archive_path.write_bytes(archive_path.read_bytes() + b"changed")
        return manifest

    monkeypatch.setattr(checkpoint, "verify_checkpoint", mutate_after_first_verification)

    with pytest.raises(CheckpointError, match="size|digest"):
        restore_checkpoint(
            persistent,
            manifest_path.name,
            live,
            expected_reviewed_commit="a" * 40,
        )

    assert calls == 1
    assert list(live.iterdir()) == []


def test_restore_checkpoint_rejects_unsafe_manifest_basename(
    tmp_path: Path, frozen_fixture_digests: None
) -> None:
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    live = tmp_path / "new-session"
    live.mkdir()

    with pytest.raises(CheckpointError, match="manifest basename"):
        restore_checkpoint(
            persistent,
            "../outside.json",
            live,
            expected_reviewed_commit="a" * 40,
        )


@pytest.mark.parametrize(
    ("error_number", "message"),
    [(errno.EEXIST, "collision"), (errno.ENOSYS, "unsupported")],
)
def test_rename_noreplace_fails_closed_without_rename_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
    message: str,
) -> None:
    class FakeRenameat2:
        argtypes: object = None
        restype: object = None

        def __call__(self, *_args: object) -> int:
            ctypes.set_errno(error_number)
            return -1

    class FakeLibc:
        renameat2 = FakeRenameat2()

    monkeypatch.setattr(checkpoint.ctypes, "CDLL", lambda *_args, **_kwargs: FakeLibc())
    monkeypatch.setattr(
        checkpoint.os,
        "rename",
        lambda *_args, **_kwargs: pytest.fail("replacement-capable rename fallback used"),
    )

    with pytest.raises(CheckpointError, match=message):
        checkpoint._rename_noreplace(tmp_path / "source", tmp_path / "destination")


def test_module_cli_round_trip_emits_only_safe_canonical_payloads(
    tmp_path: Path,
    frozen_fixture_digests: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = _valid_workspace(tmp_path / "workspace")
    scratch = tmp_path / "scratch"
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    live = tmp_path / "live"
    live.mkdir()

    assert checkpoint.main(
        ["build", str(workspace), str(scratch), "a0", "a" * 40]
    ) == 0
    build_output = capsys.readouterr()
    build_payload = json.loads(build_output.out)
    manifest_basename = str(build_payload["manifest_basename"])
    archive_basename = str(build_payload["archive_basename"])

    assert checkpoint.main(
        ["publish", str(scratch), manifest_basename, str(persistent)]
    ) == 0
    publish_output = capsys.readouterr()
    publish_payload = json.loads(publish_output.out)
    assert checkpoint.main(["verify", str(persistent), manifest_basename]) == 0
    verify_output = capsys.readouterr()
    verify_payload = json.loads(verify_output.out)
    assert checkpoint.main(
        ["restore", str(persistent), manifest_basename, str(live), "a" * 40]
    ) == 0
    restore_output = capsys.readouterr()
    restore_payload = json.loads(restore_output.out)

    expected_keys = {
        "archive_basename",
        "archive_sha256",
        "archive_size_bytes",
        "completed_stage",
        "manifest_basename",
        "reviewed_commit",
        "status",
    }
    for status, payload, output in (
        ("build", build_payload, build_output),
        ("publish", publish_payload, publish_output),
        ("verify", verify_payload, verify_output),
        ("restore", restore_payload, restore_output),
    ):
        assert set(payload) == expected_keys
        assert payload["status"] == status
        assert payload["archive_basename"] == archive_basename
        assert payload["manifest_basename"] == manifest_basename
        assert payload["reviewed_commit"] == "a" * 40
        assert output.err == ""
        assert output.out.encode() == canonical_json(payload) + b"\n"
        assert str(tmp_path) not in output.out


def test_module_cli_returns_two_without_success_payload_for_contract_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = checkpoint.main(
        ["verify", str(tmp_path / "missing"), "missing.json"]
    )

    output = capsys.readouterr()
    assert result == 2
    assert output.out == ""
    assert "error:" in output.err
