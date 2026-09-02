"""Deterministic local Phase 2B3-A workspace checkpoint artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tarfile
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from trustsr.jsonio import canonical_json

SCHEMA = "trustsr.phase2b3a-workspace-checkpoint.v1"
PROTOCOL_VERSION = 1
ARCHIVE_ROOTS = ("trustsr/phase2b1b", "trustsr/phase2b2a", "trustsr/phase2b3a")
COMPLETED_STAGES = frozenset({"a0", "a1", "a2"})
SELECTION_MANIFEST_SHA256 = "c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a"
INPUT_AUDIT_SHA256 = "fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b"
_MAX_MANIFEST_BYTES = 1024 * 1024
_HASH_CHUNK_SIZE = 1024 * 1024
_MANIFEST_KEYS = {
    "archive_basename",
    "archive_sha256",
    "archive_size_bytes",
    "completed_stage",
    "input_audit_sha256",
    "protocol_version",
    "reviewed_commit",
    "roots",
    "schema",
    "selection_manifest_sha256",
}


class CheckpointError(RuntimeError):
    """A checkpoint source or artifact violates the checkpoint contract."""


@dataclass(frozen=True)
class CheckpointManifest:
    """Canonical metadata that binds one archive to a completed stage."""

    schema: str
    protocol_version: int
    completed_stage: str
    reviewed_commit: str
    roots: tuple[str, ...]
    selection_manifest_sha256: str
    input_audit_sha256: str
    archive_basename: str
    archive_size_bytes: int
    archive_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "archive_basename": self.archive_basename,
            "archive_sha256": self.archive_sha256,
            "archive_size_bytes": self.archive_size_bytes,
            "completed_stage": self.completed_stage,
            "input_audit_sha256": self.input_audit_sha256,
            "protocol_version": self.protocol_version,
            "reviewed_commit": self.reviewed_commit,
            "roots": list(self.roots),
            "schema": self.schema,
            "selection_manifest_sha256": self.selection_manifest_sha256,
        }


@dataclass(frozen=True)
class BuiltCheckpoint:
    """A locally built archive and its canonical manifest."""

    archive_path: Path
    manifest_path: Path
    manifest: CheckpointManifest


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_stage_and_commit(completed_stage: object, reviewed_commit: object) -> tuple[str, str]:
    if type(completed_stage) is not str or completed_stage not in COMPLETED_STAGES:
        raise CheckpointError("completed stage is invalid")
    if not _is_lower_hex(reviewed_commit, 40):
        raise CheckpointError("reviewed commit must be a lowercase 40-character digest")
    return completed_stage, reviewed_commit


def _lstat(path: Path, description: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as exc:
        raise CheckpointError(f"{description} is unavailable") from exc


def _require_directory(path: Path, description: str) -> os.stat_result:
    result = _lstat(path, description)
    if stat.S_ISLNK(result.st_mode) or not stat.S_ISDIR(result.st_mode):
        raise CheckpointError(f"{description} must be a non-symlink directory")
    return result


def _require_source_file(path: Path, description: str) -> os.stat_result:
    result = _lstat(path, description)
    if stat.S_ISLNK(result.st_mode):
        raise CheckpointError(f"{description} must not be a symlink")
    if not stat.S_ISREG(result.st_mode):
        raise CheckpointError(f"{description} must be a regular file")
    if result.st_nlink != 1:
        raise CheckpointError(f"{description} must not be a hard link")
    return result


def _open_source_file(path: Path, expected: os.stat_result, description: str) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CheckpointError(f"{description} could not be opened safely") from exc
    current = os.fstat(descriptor)
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_dev != expected.st_dev
        or current.st_ino != expected.st_ino
        or current.st_size != expected.st_size
        or current.st_nlink != 1
    ):
        os.close(descriptor)
        raise CheckpointError(f"{description} changed while being opened")
    return descriptor


def _hash_source_file(path: Path, description: str) -> str:
    expected = _require_source_file(path, description)
    descriptor = _open_source_file(path, expected, description)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb") as stream:
            for block in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
                digest.update(block)
            current = os.fstat(stream.fileno())
            if current.st_size != expected.st_size or current.st_mtime_ns != expected.st_mtime_ns:
                raise CheckpointError(f"{description} changed while being read")
    except OSError as exc:
        raise CheckpointError(f"{description} could not be read") from exc
    return digest.hexdigest()


def _validate_frozen_inputs(workspace_root: Path) -> None:
    selection = (
        workspace_root
        / "trustsr/phase2b1b/selections"
        / SELECTION_MANIFEST_SHA256
        / "samples.jsonl"
    )
    if _hash_source_file(selection, "frozen selection input") != SELECTION_MANIFEST_SHA256:
        raise CheckpointError("frozen selection input digest mismatch")
    audit = (
        workspace_root
        / "trustsr/phase2b2a/input-audits"
        / SELECTION_MANIFEST_SHA256
        / "phase2b2a-input-audit.json"
    )
    if _hash_source_file(audit, "frozen input audit") != INPUT_AUDIT_SHA256:
        raise CheckpointError("frozen input audit digest mismatch")


def _validated_member_name(name: str) -> str:
    try:
        name.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CheckpointError("archive member name is not valid UTF-8") from exc
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise CheckpointError("archive member name is not a confined relative path")
    if not any(name == root or name.startswith(f"{root}/") for root in ARCHIVE_ROOTS):
        raise CheckpointError("archive member falls outside allowed roots")
    return name


def _normalized_info(name: str, source: os.stat_result) -> tarfile.TarInfo:
    info = tarfile.TarInfo(_validated_member_name(name))
    info.type = tarfile.DIRTYPE if stat.S_ISDIR(source.st_mode) else tarfile.REGTYPE
    info.size = 0 if info.isdir() else source.st_size
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if info.isdir() else 0o644
    info.pax_headers = {}
    return info


def _scan_directory(
    directory: Path, archive_name: str, expected: os.stat_result
) -> Iterator[tuple[Path, str, os.stat_result]]:
    yield directory, archive_name, expected
    try:
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise CheckpointError("source filename is not valid UTF-8") from exc
    except OSError as exc:
        raise CheckpointError("source directory could not be scanned") from exc
    for entry in ordered:
        path = directory / entry.name
        name = _validated_member_name(f"{archive_name}/{entry.name}")
        try:
            source = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise CheckpointError("source entry is unavailable") from exc
        if stat.S_ISLNK(source.st_mode):
            raise CheckpointError("source tree contains a symlink")
        if stat.S_ISDIR(source.st_mode):
            yield from _scan_directory(path, name, source)
        elif stat.S_ISREG(source.st_mode):
            if source.st_nlink != 1:
                raise CheckpointError("source tree contains a hard link")
            yield path, name, source
        else:
            raise CheckpointError("source tree permits only regular files and directories")
    current = _lstat(directory, "source directory")
    if current.st_dev != expected.st_dev or current.st_ino != expected.st_ino:
        raise CheckpointError("source directory changed while being scanned")


def _write_archive(workspace_root: Path, temporary_path: Path) -> None:
    with tarfile.open(
        temporary_path,
        mode="w:",
        format=tarfile.USTAR_FORMAT,
        dereference=False,
    ) as archive:
        for root in ARCHIVE_ROOTS:
            root_path = workspace_root / root
            try:
                root_stat = _require_directory(root_path, f"archive root {root}")
            except CheckpointError as exc:
                if not root_path.exists() and not root_path.is_symlink():
                    raise CheckpointError(f"missing archive root: {root}") from exc
                raise
            for path, name, source in _scan_directory(root_path, root, root_stat):
                info = _normalized_info(name, source)
                if info.isdir():
                    archive.addfile(info)
                    continue
                descriptor = _open_source_file(path, source, "source file")
                try:
                    with os.fdopen(descriptor, "rb") as stream:
                        archive.addfile(info, stream)
                        current = os.fstat(stream.fileno())
                        if (
                            current.st_size != source.st_size
                            or current.st_mtime_ns != source.st_mtime_ns
                        ):
                            raise CheckpointError("source file changed while being archived")
                except OSError as exc:
                    raise CheckpointError("source file could not be archived") from exc


def _fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hash_regular_file(path: Path, description: str) -> tuple[str, int]:
    source = _lstat(path, description)
    if stat.S_ISLNK(source.st_mode) or not stat.S_ISREG(source.st_mode):
        raise CheckpointError(f"{description} must be a regular non-symlink file")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CheckpointError(f"{description} could not be opened safely") from exc
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if opened.st_dev != source.st_dev or opened.st_ino != source.st_ino:
                raise CheckpointError(f"{description} changed while being opened")
            for block in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
                digest.update(block)
            current = os.fstat(stream.fileno())
            if current.st_size != opened.st_size or current.st_mtime_ns != opened.st_mtime_ns:
                raise CheckpointError(f"{description} changed while being read")
            return digest.hexdigest(), current.st_size
    except OSError as exc:
        raise CheckpointError(f"{description} could not be read") from exc


def _link_without_replacement(temporary_path: Path, final_path: Path) -> None:
    try:
        os.link(temporary_path, final_path, follow_symlinks=False)
    except FileExistsError as exc:
        raise CheckpointError(f"checkpoint output collision: {final_path.name}") from exc
    except OSError as exc:
        raise CheckpointError("checkpoint output could not be finalized") from exc
    temporary_path.unlink()


def _write_manifest(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".phase2b3a-manifest-", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _link_without_replacement(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_checkpoint(
    workspace_root: Path,
    output_directory: Path,
    *,
    completed_stage: str,
    reviewed_commit: str,
) -> BuiltCheckpoint:
    """Build a deterministic local archive and canonical manifest from allowed source roots."""
    completed_stage, reviewed_commit = _validate_stage_and_commit(completed_stage, reviewed_commit)
    workspace_root = Path(workspace_root)
    output_directory = Path(output_directory)
    _require_directory(workspace_root, "workspace root")
    for root in ARCHIVE_ROOTS:
        root_path = workspace_root / root
        if not root_path.exists() and not root_path.is_symlink():
            raise CheckpointError(f"missing archive root: {root}")
        _require_directory(root_path, f"archive root {root}")
    _validate_frozen_inputs(workspace_root)
    output_directory.mkdir(parents=True, exist_ok=True)
    _require_directory(output_directory, "output directory")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".phase2b3a-archive-", dir=output_directory
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        _write_archive(workspace_root, temporary_path)
        _fsync_path(temporary_path)
        archive_sha256, archive_size_bytes = _hash_regular_file(temporary_path, "local archive")
        archive_basename = f"phase2b3a-workspace-{completed_stage}-{archive_sha256}.tar"
        archive_path = output_directory / archive_basename
        _link_without_replacement(temporary_path, archive_path)
        manifest = CheckpointManifest(
            schema=SCHEMA,
            protocol_version=PROTOCOL_VERSION,
            completed_stage=completed_stage,
            reviewed_commit=reviewed_commit,
            roots=ARCHIVE_ROOTS,
            selection_manifest_sha256=SELECTION_MANIFEST_SHA256,
            input_audit_sha256=INPUT_AUDIT_SHA256,
            archive_basename=archive_basename,
            archive_size_bytes=archive_size_bytes,
            archive_sha256=archive_sha256,
        )
        manifest_path = archive_path.with_suffix(".json")
        _write_manifest(manifest_path, canonical_json(manifest.as_dict()))
        _fsync_directory(output_directory)
        return BuiltCheckpoint(
            archive_path=archive_path,
            manifest_path=manifest_path,
            manifest=manifest,
        )
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_manifest_bytes(path: Path) -> bytes:
    source = _lstat(path, "manifest")
    if stat.S_ISLNK(source.st_mode) or not stat.S_ISREG(source.st_mode):
        raise CheckpointError("manifest must be a regular non-symlink file")
    if source.st_size > _MAX_MANIFEST_BYTES:
        raise CheckpointError("manifest exceeds 1 MiB")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise CheckpointError("manifest could not be opened safely") from exc
    try:
        with os.fdopen(descriptor, "rb") as stream:
            payload = stream.read(source.st_size)
            current = os.fstat(stream.fileno())
            if current.st_size != source.st_size or current.st_mtime_ns != source.st_mtime_ns:
                raise CheckpointError("manifest changed while being read")
            return payload
    except OSError as exc:
        raise CheckpointError("manifest could not be read") from exc


def _parse_manifest_value(value: object) -> CheckpointManifest:
    if type(value) is not dict or set(value) != _MANIFEST_KEYS:
        raise CheckpointError("manifest must use the exact key set")
    if value["schema"] != SCHEMA:
        raise CheckpointError("manifest schema is invalid")
    if type(value["protocol_version"]) is not int or value["protocol_version"] != PROTOCOL_VERSION:
        raise CheckpointError("manifest protocol version is invalid")
    completed_stage, reviewed_commit = _validate_stage_and_commit(
        value["completed_stage"], value["reviewed_commit"]
    )
    if type(value["roots"]) is not list or tuple(value["roots"]) != ARCHIVE_ROOTS:
        raise CheckpointError("manifest roots are invalid")
    if not _is_lower_hex(value["selection_manifest_sha256"], 64):
        raise CheckpointError("selection manifest digest is invalid")
    if not _is_lower_hex(value["input_audit_sha256"], 64):
        raise CheckpointError("input audit digest is invalid")
    if value["selection_manifest_sha256"] != SELECTION_MANIFEST_SHA256:
        raise CheckpointError("selection manifest digest does not match the frozen input")
    if value["input_audit_sha256"] != INPUT_AUDIT_SHA256:
        raise CheckpointError("input audit digest does not match the frozen input")
    if not _is_lower_hex(value["archive_sha256"], 64):
        raise CheckpointError("archive digest is invalid")
    if type(value["archive_size_bytes"]) is not int or value["archive_size_bytes"] <= 0:
        raise CheckpointError("archive size is invalid")
    if type(value["archive_basename"]) is not str:
        raise CheckpointError("archive basename is invalid")
    manifest = CheckpointManifest(
        schema=value["schema"],
        protocol_version=value["protocol_version"],
        completed_stage=completed_stage,
        reviewed_commit=reviewed_commit,
        roots=tuple(value["roots"]),
        selection_manifest_sha256=value["selection_manifest_sha256"],
        input_audit_sha256=value["input_audit_sha256"],
        archive_basename=value["archive_basename"],
        archive_size_bytes=value["archive_size_bytes"],
        archive_sha256=value["archive_sha256"],
    )
    expected_basename = (
        f"phase2b3a-workspace-{manifest.completed_stage}-{manifest.archive_sha256}.tar"
    )
    if manifest.archive_basename != expected_basename:
        raise CheckpointError("archive basename is not digest-bound")
    return manifest


def load_manifest(path: Path) -> CheckpointManifest:
    """Load only a bounded, canonical manifest with the exact checkpoint schema."""
    payload = _read_manifest_bytes(Path(path))
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError("manifest is not valid JSON") from exc
    if payload != canonical_json(value):
        raise CheckpointError("manifest is not canonical JSON")
    return _parse_manifest_value(value)


def verify_checkpoint(archive_path: Path, manifest_path: Path) -> CheckpointManifest:
    """Verify a local archive's name, size, and SHA-256 against its manifest."""
    manifest = load_manifest(Path(manifest_path))
    archive_path = Path(archive_path)
    if archive_path.name != manifest.archive_basename:
        raise CheckpointError("archive basename does not match manifest")
    archive_sha256, archive_size_bytes = _hash_regular_file(archive_path, "archive")
    if archive_size_bytes != manifest.archive_size_bytes:
        raise CheckpointError("archive size does not match manifest")
    if archive_sha256 != manifest.archive_sha256:
        raise CheckpointError("archive digest does not match manifest")
    return manifest
