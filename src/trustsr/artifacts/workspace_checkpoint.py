"""Deterministic local Phase 2B3-A workspace checkpoint artifacts."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tarfile
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from trustsr.jsonio import canonical_json

SCHEMA = "trustsr.phase2b3a-workspace-checkpoint.v1"
PROTOCOL_VERSION = 1
ARCHIVE_ROOTS = ("trustsr/phase2b1b", "trustsr/phase2b2a", "trustsr/phase2b3a")
COMPLETED_STAGES = frozenset({"a0", "a1", "a2"})
SELECTION_MANIFEST_SHA256 = "c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a"
INPUT_AUDIT_SHA256 = "fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b"
LEGACY_A1_PRODUCER_COMMIT = "4df5195e0a28701391c3951659a42409f81a11c2"
LEGACY_A1_ARCHIVE_SHA256 = "623535c33fee50e7d05b83386158b349c4056d1f4aa256efda1189933e9993f8"
LEGACY_A1_ARCHIVE_SIZE = 933_263_360
_NORMALIZATION_POLICY = "uint16_saturate_10000_divide_10000_v2"
_RAW_RADIOMETRIC_MAX = 32767
_SATURATION_THRESHOLD = 10000
_RADIOMETRIC_BANDS = ["B04", "B03", "B02", "B08"]
SELECTION_RELATIVE = Path(
    "trustsr/phase2b1b/selections/"
    f"{SELECTION_MANIFEST_SHA256}/samples.jsonl"
)
INPUT_AUDIT_RELATIVE = Path(
    "trustsr/phase2b2a/input-audits/"
    f"{SELECTION_MANIFEST_SHA256}/phase2b2a-input-audit.json"
)
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_ARCHIVE_BYTES = (1 << 63) - 1
_MAX_EVIDENCE_BYTES = 5 * 1024**2
_HASH_CHUNK_SIZE = 1024 * 1024
_BUNDLE_MANIFEST_BASENAME = "phase2b3a-bundle-manifest.json"
_PERSISTENT_FINAL_PATTERN = re.compile(
    r"phase2b3a-workspace-(?:a0|a1|a2)-[0-9a-f]{64}\.(?:tar|json)\Z"
)
_STAGE_EVIDENCE_BASENAMES = {
    stage: tuple(
        f"phase2b3a-{stage}-{suffix}.json"
        for suffix in ("result", "cache-audit", "runtime", "replay")
    )
    for stage in ("a1", "a2")
}
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


@dataclass(frozen=True)
class _ArchiveMember:
    name: str
    is_directory: bool
    size: int


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


def _frozen_input_digests() -> dict[str, str]:
    return {
        (
            f"trustsr/phase2b1b/selections/{SELECTION_MANIFEST_SHA256}/samples.jsonl"
        ): SELECTION_MANIFEST_SHA256,
        (
            f"trustsr/phase2b2a/input-audits/{SELECTION_MANIFEST_SHA256}/phase2b2a-input-audit.json"
        ): INPUT_AUDIT_SHA256,
    }


def _validated_member_name(name: str) -> str:
    try:
        name.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CheckpointError("archive member name is not valid UTF-8") from exc
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != name
    ):
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


class _DigestingReader:
    """File reader that hashes exactly the bytes consumed by ``TarFile``."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        payload = self._stream.read(size)
        self.digest.update(payload)
        return payload


def _open_workspace_root(workspace_root: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        return os.open(workspace_root, flags)
    except OSError as exc:
        raise CheckpointError("workspace root must be a non-symlink directory") from exc


def _open_child_directory(parent_descriptor: int, name: str, description: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        return os.open(name, flags, dir_fd=parent_descriptor)
    except FileNotFoundError as exc:
        raise CheckpointError(f"missing archive root: {description}") from exc
    except OSError as exc:
        raise CheckpointError(f"source path component {description} is unsafe") from exc


def _open_archive_root(workspace_descriptor: int, root: str) -> int:
    current = os.dup(workspace_descriptor)
    try:
        for component in PurePosixPath(root).parts:
            child = _open_child_directory(current, component, root)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _directory_changed(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    )


def _add_source_file(
    archive: tarfile.TarFile,
    directory_descriptor: int,
    entry_name: str,
    archive_name: str,
    source: os.stat_result,
    frozen_inputs: dict[str, str],
) -> None:
    if source.st_nlink != 1:
        raise CheckpointError("source tree contains a hard link")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(entry_name, flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise CheckpointError("source file could not be opened safely") from exc
    try:
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != source.st_dev
                or opened.st_ino != source.st_ino
                or opened.st_size != source.st_size
                or opened.st_nlink != 1
            ):
                raise CheckpointError("source file changed while being opened")
            reader = _DigestingReader(stream)
            archive.addfile(_normalized_info(archive_name, source), reader)
            current = os.fstat(stream.fileno())
            if current.st_size != source.st_size or current.st_mtime_ns != source.st_mtime_ns:
                raise CheckpointError("source file changed while being archived")
    except OSError as exc:
        raise CheckpointError("source file could not be archived") from exc
    expected_digest = frozen_inputs.pop(archive_name, None)
    if expected_digest is not None and reader.digest.hexdigest() != expected_digest:
        raise CheckpointError("frozen selection or input audit digest mismatch")


def _write_directory(
    archive: tarfile.TarFile,
    directory_descriptor: int,
    archive_name: str,
    frozen_inputs: dict[str, str],
) -> None:
    before = os.fstat(directory_descriptor)
    if not stat.S_ISDIR(before.st_mode):
        raise CheckpointError("source tree permits only regular files and directories")
    archive.addfile(_normalized_info(archive_name, before))
    try:
        with os.scandir(directory_descriptor) as entries:
            names = sorted((entry.name for entry in entries), key=lambda name: name.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise CheckpointError("source filename is not valid UTF-8") from exc
    except OSError as exc:
        raise CheckpointError("source directory could not be scanned") from exc
    for entry_name in names:
        name = _validated_member_name(f"{archive_name}/{entry_name}")
        try:
            source = os.stat(entry_name, dir_fd=directory_descriptor, follow_symlinks=False)
        except OSError as exc:
            raise CheckpointError("source entry is unavailable") from exc
        if stat.S_ISLNK(source.st_mode):
            raise CheckpointError("source tree contains a symlink")
        if stat.S_ISDIR(source.st_mode):
            child_descriptor = _open_child_directory(directory_descriptor, entry_name, name)
            try:
                opened = os.fstat(child_descriptor)
                if opened.st_dev != source.st_dev or opened.st_ino != source.st_ino:
                    raise CheckpointError("source directory changed while being opened")
                _write_directory(archive, child_descriptor, name, frozen_inputs)
            finally:
                os.close(child_descriptor)
        elif stat.S_ISREG(source.st_mode):
            _add_source_file(
                archive,
                directory_descriptor,
                entry_name,
                name,
                source,
                frozen_inputs,
            )
        else:
            raise CheckpointError("source tree permits only regular files and directories")
    if _directory_changed(before, os.fstat(directory_descriptor)):
        raise CheckpointError("source directory changed while being scanned")


def _write_archive(workspace_root: Path, temporary_path: Path) -> None:
    workspace_descriptor = _open_workspace_root(workspace_root)
    frozen_inputs = _frozen_input_digests()
    try:
        with tarfile.open(
            temporary_path,
            mode="w:",
            format=tarfile.USTAR_FORMAT,
            dereference=False,
        ) as archive:
            for root in ARCHIVE_ROOTS:
                root_descriptor = _open_archive_root(workspace_descriptor, root)
                try:
                    _write_directory(archive, root_descriptor, root, frozen_inputs)
                finally:
                    os.close(root_descriptor)
        if frozen_inputs:
            raise CheckpointError("frozen source input is missing from archive")
    finally:
        os.close(workspace_descriptor)


def _active_frozen_relatives() -> tuple[tuple[Path, str, str], ...]:
    return (
        (
            Path(
                "trustsr/phase2b1b/selections/"
                f"{SELECTION_MANIFEST_SHA256}/samples.jsonl"
            ),
            SELECTION_MANIFEST_SHA256,
            "frozen selection",
        ),
        (
            Path(
                "trustsr/phase2b2a/input-audits/"
                f"{SELECTION_MANIFEST_SHA256}/phase2b2a-input-audit.json"
            ),
            INPUT_AUDIT_SHA256,
            "frozen input audit",
        ),
    )


def _read_relative_regular_file(
    root: Path,
    relative: Path,
    description: str,
    *,
    max_bytes: int | None = None,
    collect: bool = False,
) -> tuple[bytes | None, str, int]:
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise CheckpointError(f"{description} relative path is invalid")
    root_descriptor = _open_workspace_root(root)
    current = root_descriptor
    try:
        for component in relative.parts[:-1]:
            try:
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=current,
                )
            except OSError as exc:
                raise CheckpointError(f"{description} source is missing or unsafe") from exc
            if current != root_descriptor:
                os.close(current)
            current = child
        try:
            source = os.stat(relative.name, dir_fd=current, follow_symlinks=False)
        except OSError as exc:
            raise CheckpointError(f"{description} file is missing or unsafe") from exc
        if (
            stat.S_ISLNK(source.st_mode)
            or not stat.S_ISREG(source.st_mode)
            or source.st_nlink != 1
        ):
            raise CheckpointError(f"{description} file must be a one-link regular file")
        if max_bytes is not None and source.st_size > max_bytes:
            raise CheckpointError(f"{description} size exceeds the allowed limit")
        try:
            descriptor = os.open(
                relative.name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=current,
            )
        except OSError as exc:
            raise CheckpointError(f"{description} file could not be opened safely") from exc
        digest = hashlib.sha256()
        payload = bytearray() if collect else None
        try:
            with os.fdopen(descriptor, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if (
                    opened.st_dev != source.st_dev
                    or opened.st_ino != source.st_ino
                    or opened.st_nlink != 1
                ):
                    raise CheckpointError(f"{description} file changed while being opened")
                for block in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
                    digest.update(block)
                    if payload is not None:
                        payload.extend(block)
                current_source = os.fstat(stream.fileno())
                if (
                    current_source.st_size != opened.st_size
                    or current_source.st_mtime_ns != opened.st_mtime_ns
                    or current_source.st_nlink != 1
                ):
                    raise CheckpointError(f"{description} file changed while being read")
        except OSError as exc:
            raise CheckpointError(f"{description} file could not be read") from exc
        return bytes(payload) if payload is not None else None, digest.hexdigest(), source.st_size
    finally:
        if current != root_descriptor:
            os.close(current)
        os.close(root_descriptor)


def _validate_preflight_evidence(trustsr_root: Path, reviewed_commit: str) -> None:
    log_relative = Path("trustsr/phase2b3a/logs/preflight.jsonl")
    runtime_relative = (
        Path("trustsr/phase2b3a/results")
        / SELECTION_MANIFEST_SHA256
        / "phase2b3a-preflight-runtime.json"
    )
    try:
        _, _, log_size = _read_relative_regular_file(
            trustsr_root,
            log_relative,
            "preflight log",
            max_bytes=_MAX_EVIDENCE_BYTES,
        )
        runtime_bytes, _, _ = _read_relative_regular_file(
            trustsr_root,
            runtime_relative,
            "preflight runtime",
            max_bytes=_MAX_EVIDENCE_BYTES,
            collect=True,
        )
    except CheckpointError as exc:
        raise CheckpointError("preflight evidence is missing or unsafe") from exc
    if log_size == 0:
        raise CheckpointError("preflight log is empty")
    assert runtime_bytes is not None
    try:
        runtime = json.loads(runtime_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError("preflight runtime is not valid JSON") from exc
    if canonical_json(runtime) != runtime_bytes:
        raise CheckpointError("preflight runtime is not canonical JSON")
    producer_commit = runtime.get("git_commit") if type(runtime) is dict else None
    if not _is_lower_hex(producer_commit, 40) or producer_commit != reviewed_commit:
        raise CheckpointError("preflight runtime producer commit does not match checkpoint")


def _parse_canonical_evidence(payload: bytes, description: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"{description} is not valid JSON") from exc
    if type(value) is not dict or canonical_json(value) != payload:
        raise CheckpointError(f"{description} is not canonical JSON")
    return value


def _checkpoint_radiometric_policy(
    samples: object, *, expected_sample_count: int
) -> dict[str, object]:
    if type(samples) is not list or len(samples) != expected_sample_count:
        raise CheckpointError("stage radiometric sample count is invalid")
    lr_total = 0
    hr_total = 0
    affected_samples = 0
    affected_assets = 0
    maxima: list[int] = []
    for sample in samples:
        saturation = sample.get("radiometric_saturation") if type(sample) is dict else None
        if type(saturation) is not dict or set(saturation) != {"lr", "hr"}:
            raise CheckpointError("stage radiometric saturation schema is invalid")
        sample_affected = False
        for asset_name in ("lr", "hr"):
            asset = saturation[asset_name]
            if type(asset) is not dict or set(asset) != {
                "raw_crop_minimum",
                "raw_crop_maximum",
                "clipped_high_count",
                "clipped_high_by_band",
            }:
                raise CheckpointError("stage radiometric asset schema is invalid")
            minimum = asset["raw_crop_minimum"]
            maximum = asset["raw_crop_maximum"]
            clipped = asset["clipped_high_count"]
            by_band = asset["clipped_high_by_band"]
            if any(type(value) is not int for value in (minimum, maximum, clipped)):
                raise CheckpointError("stage radiometric values must be built-in integers")
            if (
                type(by_band) is not list
                or len(by_band) != 4
                or any(type(value) is not int for value in by_band)
            ):
                raise CheckpointError("stage radiometric band counts are invalid")
            if (
                minimum < 0
                or maximum < minimum
                or maximum > _RAW_RADIOMETRIC_MAX
                or clipped < 0
                or any(value < 0 for value in by_band)
                or sum(by_band) != clipped
                or (maximum > _SATURATION_THRESHOLD) != (clipped > 0)
            ):
                raise CheckpointError("stage radiometric saturation values are invalid")
            maxima.append(maximum)
            if clipped:
                affected_assets += 1
                sample_affected = True
            if asset_name == "lr":
                lr_total += clipped
            else:
                hr_total += clipped
        affected_samples += int(sample_affected)
    return {
        "normalization_policy": _NORMALIZATION_POLICY,
        "raw_radiometric_max": _RAW_RADIOMETRIC_MAX,
        "saturation_threshold": _SATURATION_THRESHOLD,
        "bands": _RADIOMETRIC_BANDS,
        "sample_count": expected_sample_count,
        "affected_sample_count": affected_samples,
        "affected_asset_count": affected_assets,
        "lr_clipped_high_count": lr_total,
        "hr_clipped_high_count": hr_total,
        "raw_crop_maximum": max(maxima),
    }


def _require_checkpoint_policy(value: object, expected: dict[str, object]) -> None:
    integer_keys = set(expected) - {"normalization_policy", "bands"}
    if (
        type(value) is not dict
        or set(value) != set(expected)
        or any(type(value.get(key)) is not int for key in integer_keys)
        or type(value.get("normalization_policy")) is not str
        or type(value.get("bands")) is not list
        or value != expected
    ):
        raise CheckpointError("stage radiometric policy is invalid")


def _validate_current_stage_policy(
    documents: dict[str, dict[str, object]], completed_stage: str
) -> None:
    result, audit, runtime, replay = (
        documents[name] for name in _STAGE_EVIDENCE_BASENAMES[completed_stage]
    )
    expected_schemas = (
        (
            "trustsr.phase2b3a-development-smoke.v2",
            "trustsr.phase2b3a-development-smoke-cache-audit.v2",
            "trustsr.phase2b3a-a1-runtime.v2",
            "trustsr.phase2b3a-a1-replay.v2",
        )
        if completed_stage == "a1"
        else (
            "trustsr.phase2b3a-development-score-audit.v1",
            "trustsr.phase2b3a-development-score-cache-audit.v1",
            "trustsr.phase2b3a-a2-runtime.v1",
            "trustsr.phase2b3a-a2-replay.v1",
        )
    )
    if tuple(item.get("schema") for item in (result, audit, runtime, replay)) != expected_schemas:
        raise CheckpointError("stage evidence schema is not current")
    if any(
        item.get("normalization_policy") != _NORMALIZATION_POLICY
        for item in (result, audit, runtime)
    ):
        raise CheckpointError("stage normalization policy is invalid")
    sample_count = 4 if completed_stage == "a1" else 120
    expected_policy = _checkpoint_radiometric_policy(
        result.get("samples"), expected_sample_count=sample_count
    )
    _require_checkpoint_policy(result.get("radiometric_policy"), expected_policy)
    _require_checkpoint_policy(runtime.get("radiometric_policy"), expected_policy)


def _is_exact_legacy_a1_checkpoint(manifest: CheckpointManifest) -> bool:
    return (
        manifest.completed_stage == "a1"
        and manifest.reviewed_commit == LEGACY_A1_PRODUCER_COMMIT
        and manifest.archive_sha256 == LEGACY_A1_ARCHIVE_SHA256
        and manifest.archive_size_bytes == LEGACY_A1_ARCHIVE_SIZE
        and manifest.selection_manifest_sha256 == SELECTION_MANIFEST_SHA256
        and manifest.input_audit_sha256 == INPUT_AUDIT_SHA256
    )


def _validate_workspace_evidence(
    trustsr_root: Path,
    completed_stage: str,
    reviewed_commit: str,
    *,
    allow_legacy_a1: bool = False,
) -> None:
    for relative, expected_digest, description in _active_frozen_relatives():
        try:
            _, observed_digest, _ = _read_relative_regular_file(
                trustsr_root, relative, description
            )
        except CheckpointError as exc:
            raise CheckpointError(
                f"missing archive root or unsafe {description} source"
            ) from exc
        if observed_digest != expected_digest:
            raise CheckpointError(f"{description} digest mismatch")
    if completed_stage in {"a0", "a1"}:
        _validate_preflight_evidence(trustsr_root, reviewed_commit)
    if completed_stage == "a0":
        return

    result_relative = Path(
        "trustsr/phase2b3a/results"
    ) / SELECTION_MANIFEST_SHA256
    manifest_relative = result_relative / _BUNDLE_MANIFEST_BASENAME
    manifest_bytes, _, _ = _read_relative_regular_file(
        trustsr_root,
        manifest_relative,
        "stage evidence manifest",
        max_bytes=_MAX_MANIFEST_BYTES,
        collect=True,
    )
    assert manifest_bytes is not None
    try:
        value = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError("stage evidence manifest is not valid JSON") from exc
    if canonical_json(value) != manifest_bytes:
        raise CheckpointError("stage evidence manifest is not canonical JSON")
    expected_manifest_schema = (
        "trustsr.phase2b3a-bundle-manifest.v1"
        if completed_stage == "a2" or allow_legacy_a1
        else "trustsr.phase2b3a-bundle-manifest.v2"
    )
    if (
        type(value) is not dict
        or set(value) != {"schema", "phase", "files"}
        or value["schema"] != expected_manifest_schema
    ):
        raise CheckpointError("stage evidence manifest schema is invalid")
    if value["phase"] != completed_stage:
        raise CheckpointError("stage evidence manifest phase does not match checkpoint")
    expected_basenames = _STAGE_EVIDENCE_BASENAMES[completed_stage]
    entries = value["files"]
    if type(entries) is not list or len(entries) != 4:
        raise CheckpointError("stage evidence manifest must declare four expected files")
    observed_basenames: list[str] = []
    documents: dict[str, dict[str, object]] = {}
    for entry in entries:
        if type(entry) is not dict or set(entry) != {
            "basename",
            "size_bytes",
            "sha256",
        }:
            raise CheckpointError("stage evidence manifest file entry is invalid")
        basename = entry["basename"]
        size_bytes = entry["size_bytes"]
        declared_digest = entry["sha256"]
        if (
            type(basename) is not str
            or basename not in expected_basenames
            or type(size_bytes) is not int
            or size_bytes < 0
            or size_bytes > _MAX_EVIDENCE_BYTES
            or not _is_lower_hex(declared_digest, 64)
        ):
            raise CheckpointError("stage evidence basename, size, or digest is invalid")
        payload, observed_digest, observed_size = _read_relative_regular_file(
            trustsr_root,
            result_relative / basename,
            "stage evidence file",
            max_bytes=_MAX_EVIDENCE_BYTES,
            collect=True,
        )
        if observed_size != size_bytes:
            raise CheckpointError("stage evidence file size does not match manifest")
        if observed_digest != declared_digest:
            raise CheckpointError("stage evidence file digest does not match manifest")
        assert payload is not None
        documents[basename] = _parse_canonical_evidence(payload, "stage evidence file")
        observed_basenames.append(basename)
    if observed_basenames != sorted(expected_basenames):
        raise CheckpointError("stage evidence manifest must declare four expected files")
    if not allow_legacy_a1:
        _validate_current_stage_policy(documents, completed_stage)


def _fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(
        directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
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
    _validate_workspace_evidence(workspace_root, completed_stage, reviewed_commit)
    output_directory = Path(output_directory)
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
    if (
        stat.S_ISLNK(source.st_mode)
        or not stat.S_ISREG(source.st_mode)
        or source.st_nlink != 1
    ):
        raise CheckpointError("manifest must be a one-link regular non-symlink file")
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
    if (
        type(value["archive_size_bytes"]) is not int
        or value["archive_size_bytes"] <= 0
        or value["archive_size_bytes"] > _MAX_ARCHIVE_BYTES
    ):
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


def _inspect_archive(archive_path: Path) -> tuple[_ArchiveMember, ...]:
    source = _lstat(archive_path, "archive")
    if stat.S_ISLNK(source.st_mode) or not stat.S_ISREG(source.st_mode):
        raise CheckpointError("archive must be a regular non-symlink file")
    try:
        descriptor = os.open(archive_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise CheckpointError("archive could not be opened safely") from exc
    inventory: list[_ArchiveMember] = []
    seen_names: set[str] = set()
    seen_directories: set[str] = set()
    try:
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if opened.st_dev != source.st_dev or opened.st_ino != source.st_ino:
                raise CheckpointError("archive changed while being opened")
            with tarfile.open(fileobj=stream, mode="r:") as archive:
                for member in archive:
                    name = _validated_member_name(member.name)
                    if name in seen_names:
                        raise CheckpointError("archive contains a duplicate member name")
                    if not (member.isdir() or member.isreg()):
                        raise CheckpointError(
                            "archive permits only regular files and directories"
                        )
                    if member.size < 0 or member.size > opened.st_size:
                        raise CheckpointError("archive member size is invalid")
                    root = next(
                        (
                            candidate
                            for candidate in ARCHIVE_ROOTS
                            if name == candidate or name.startswith(f"{candidate}/")
                        ),
                        None,
                    )
                    if root is None:
                        raise CheckpointError("archive member falls outside allowed roots")
                    if name != root:
                        parent = PurePosixPath(name).parent.as_posix()
                        if parent not in seen_directories:
                            raise CheckpointError(
                                "archive member parent directory is missing or out of order"
                            )
                    seen_names.add(name)
                    if member.isdir():
                        seen_directories.add(name)
                    inventory.append(
                        _ArchiveMember(
                            name=name,
                            is_directory=member.isdir(),
                            size=member.size,
                        )
                    )
            current = os.fstat(stream.fileno())
            if (
                current.st_size != opened.st_size
                or current.st_mtime_ns != opened.st_mtime_ns
            ):
                raise CheckpointError("archive changed while being inspected")
    except CheckpointError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise CheckpointError("archive could not be fully inspected") from exc
    missing_roots = set(ARCHIVE_ROOTS) - seen_directories
    if missing_roots:
        raise CheckpointError("archive is missing a required root directory record")
    return tuple(inventory)


def verify_checkpoint(archive_path: Path, manifest_path: Path) -> CheckpointManifest:
    """Verify a local archive's name, bytes, and complete inventory against its manifest."""
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    archive_path = Path(archive_path)
    expected_manifest_basename = (
        manifest.archive_basename.removesuffix(".tar") + ".json"
    )
    if manifest_path.name != expected_manifest_basename:
        raise CheckpointError("manifest basename does not match archive")
    _verify_archive_against_manifest(archive_path, manifest)
    return manifest


def _verify_archive_against_manifest(
    archive_path: Path, manifest: CheckpointManifest
) -> tuple[_ArchiveMember, ...]:
    archive_path = Path(archive_path)
    if archive_path.name != manifest.archive_basename:
        raise CheckpointError("archive basename does not match manifest")
    archive_sha256, archive_size_bytes = _hash_regular_file(archive_path, "archive")
    if archive_size_bytes != manifest.archive_size_bytes:
        raise CheckpointError("archive size does not match manifest")
    if archive_sha256 != manifest.archive_sha256:
        raise CheckpointError("archive digest does not match manifest")
    return _inspect_archive(archive_path)


def _matching_immutable_final(final_path: Path, expected_path: Path, description: str) -> bool:
    try:
        final_source = final_path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise CheckpointError(f"checkpoint {description} collision") from exc
    if (
        stat.S_ISLNK(final_source.st_mode)
        or not stat.S_ISREG(final_source.st_mode)
        or final_source.st_nlink != 1
    ):
        raise CheckpointError(f"checkpoint {description} collision")
    try:
        final_descriptor = os.open(
            final_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        expected_descriptor = os.open(
            expected_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
    except OSError as exc:
        if "final_descriptor" in locals():
            os.close(final_descriptor)
        raise CheckpointError(f"checkpoint {description} collision") from exc
    matches = True
    try:
        with (
            os.fdopen(final_descriptor, "rb") as final_stream,
            os.fdopen(expected_descriptor, "rb") as expected_stream,
        ):
            opened_final = os.fstat(final_stream.fileno())
            if (
                opened_final.st_dev != final_source.st_dev
                or opened_final.st_ino != final_source.st_ino
                or opened_final.st_nlink != 1
            ):
                raise CheckpointError(f"checkpoint {description} collision")
            while True:
                final_block = final_stream.read(_HASH_CHUNK_SIZE)
                expected_block = expected_stream.read(_HASH_CHUNK_SIZE)
                if final_block != expected_block:
                    matches = False
                    break
                if not final_block:
                    break
            current_final = os.fstat(final_stream.fileno())
            if (
                current_final.st_size != opened_final.st_size
                or current_final.st_mtime_ns != opened_final.st_mtime_ns
                or current_final.st_nlink != 1
            ):
                raise CheckpointError(f"checkpoint {description} collision")
    except OSError as exc:
        raise CheckpointError(f"checkpoint {description} collision") from exc
    if not matches:
        raise CheckpointError(f"checkpoint {description} collision")
    return True


def _matching_immutable_manifest(final_path: Path, expected_bytes: bytes) -> bool:
    try:
        final_path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise CheckpointError("checkpoint manifest collision") from exc
    try:
        observed_bytes = _read_manifest_bytes(final_path)
    except CheckpointError as exc:
        raise CheckpointError("checkpoint manifest collision") from exc
    if observed_bytes != expected_bytes:
        raise CheckpointError("checkpoint manifest collision")
    return True


def _validate_persistent_entries(
    persistent_directory: Path,
    archive_basename: str,
    manifest_basename: str,
    *,
    allow_selected_archive_only: bool,
) -> None:
    selected_stem = archive_basename.removesuffix(".tar")
    observed_suffixes: dict[str, set[str]] = {}
    try:
        entries = list(os.scandir(persistent_directory))
    except OSError as exc:
        raise CheckpointError("persistent directory could not be scanned") from exc
    for entry in entries:
        if entry.name == ".checkpoint.lock":
            try:
                lock = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise CheckpointError("checkpoint lock is unsafe") from exc
            if stat.S_ISLNK(lock.st_mode) or not stat.S_ISDIR(lock.st_mode):
                raise CheckpointError("checkpoint lock must be a non-symlink directory")
            continue
        if entry.name.endswith(".part"):
            raise CheckpointError(f"checkpoint partial collision: {entry.name}")
        if _PERSISTENT_FINAL_PATTERN.fullmatch(entry.name) is None:
            raise CheckpointError(f"unexpected persistent-directory entry: {entry.name}")
        try:
            source = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise CheckpointError("persistent checkpoint final is unsafe") from exc
        if (
            stat.S_ISLNK(source.st_mode)
            or not stat.S_ISREG(source.st_mode)
            or source.st_nlink != 1
        ):
            raise CheckpointError(
                f"persistent checkpoint final collision: unsafe link for {entry.name}"
            )
        stem, suffix = entry.name.rsplit(".", 1)
        observed_suffixes.setdefault(stem, set()).add(suffix)

    for stem, suffixes in observed_suffixes.items():
        if suffixes == {"tar", "json"}:
            continue
        if (
            allow_selected_archive_only
            and stem == selected_stem
            and suffixes == {"tar"}
        ):
            continue
        raise CheckpointError(f"persistent checkpoint pair is unpaired: {stem}")

    selected_suffixes = observed_suffixes.get(selected_stem, set())
    if allow_selected_archive_only:
        if selected_suffixes not in (set(), {"tar"}, {"tar", "json"}):
            raise CheckpointError("selected persistent checkpoint pair is unpaired")
    elif selected_suffixes != {"tar", "json"}:
        raise CheckpointError("selected persistent checkpoint pair is incomplete")

    if manifest_basename != f"{selected_stem}.json":
        raise CheckpointError("selected persistent checkpoint basenames do not match")


def _create_partial(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        return os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise CheckpointError(f"checkpoint partial collision: {path.name}") from exc
    except OSError as exc:
        raise CheckpointError("checkpoint partial could not be created safely") from exc


def _link_partial(partial_path: Path, final_path: Path, persistent_directory: Path) -> None:
    try:
        _rename_noreplace(partial_path, final_path)
    except CheckpointError as exc:
        if final_path.exists() or final_path.is_symlink():
            partial_path.unlink(missing_ok=True)
            raise CheckpointError(f"checkpoint output collision: {final_path.name}") from exc
        partial_path.unlink(missing_ok=True)
        raise CheckpointError("checkpoint output could not be published") from exc
    _fsync_directory(persistent_directory)


def _rollback_created_final(
    final_path: Path,
    created: os.stat_result,
    persistent_directory: Path,
) -> None:
    try:
        current = final_path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise CheckpointError("new archive final could not be inspected for rollback") from exc
    if current.st_dev != created.st_dev or current.st_ino != created.st_ino:
        raise CheckpointError("new archive final changed before rollback")
    try:
        final_path.unlink()
        _fsync_directory(persistent_directory)
    except OSError as exc:
        raise CheckpointError("new archive final could not be rolled back") from exc


def publish_checkpoint(
    built: BuiltCheckpoint,
    persistent_directory: Path,
    *,
    copy_file: Callable[[BinaryIO, BinaryIO], None] = shutil.copyfileobj,
) -> tuple[Path, Path]:
    """Immutably publish a local checkpoint across filesystem boundaries."""
    persistent_directory = Path(persistent_directory)
    _require_directory(persistent_directory, "persistent directory")
    manifest = verify_checkpoint(built.archive_path, built.manifest_path)
    if manifest != built.manifest:
        raise CheckpointError("built checkpoint metadata does not match its manifest")
    manifest_bytes = canonical_json(manifest.as_dict())
    if _read_manifest_bytes(built.manifest_path) != manifest_bytes:
        raise CheckpointError("built checkpoint manifest changed after verification")
    manifest_basename = manifest.archive_basename.removesuffix(".tar") + ".json"
    if built.manifest_path.name != manifest_basename:
        raise CheckpointError("built manifest basename does not match archive")
    _validate_persistent_entries(
        persistent_directory,
        manifest.archive_basename,
        manifest_basename,
        allow_selected_archive_only=True,
    )

    archive_path = persistent_directory / manifest.archive_basename
    manifest_path = persistent_directory / manifest_basename
    archive_exists = _matching_immutable_final(
        archive_path, built.archive_path, "archive"
    )
    manifest_exists = _matching_immutable_manifest(manifest_path, manifest_bytes)
    if manifest_exists and not archive_exists:
        raise CheckpointError("checkpoint manifest-only collision")

    created_archive: os.stat_result | None = None
    if not archive_exists:
        archive_partial = persistent_directory / f".{manifest.archive_basename}.part"
        destination_descriptor: int | None = _create_partial(archive_partial)
        source_descriptor: int | None = None
        try:
            source_descriptor = os.open(
                built.archive_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            )
            with os.fdopen(source_descriptor, "rb") as source:
                source_descriptor = None
                with os.fdopen(destination_descriptor, "wb") as target:
                    destination_descriptor = None
                    copy_file(source, target)
                    target.flush()
                    os.fsync(target.fileno())
        except Exception as exc:
            raise CheckpointError("checkpoint archive copy failed") from exc
        finally:
            if source_descriptor is not None:
                os.close(source_descriptor)
            if destination_descriptor is not None:
                os.close(destination_descriptor)
        archive_sha256, archive_size_bytes = _hash_regular_file(
            archive_partial, "checkpoint archive partial"
        )
        if archive_size_bytes != manifest.archive_size_bytes:
            raise CheckpointError("checkpoint archive partial size mismatch")
        if archive_sha256 != manifest.archive_sha256:
            raise CheckpointError("checkpoint archive partial digest mismatch")
        partial_source = _lstat(archive_partial, "checkpoint archive partial")
        _link_partial(archive_partial, archive_path, persistent_directory)
        linked_archive = _lstat(archive_path, "new checkpoint archive final")
        if (
            linked_archive.st_dev != partial_source.st_dev
            or linked_archive.st_ino != partial_source.st_ino
        ):
            raise CheckpointError("new checkpoint archive final changed after linking")
        created_archive = partial_source

    _verify_archive_against_manifest(archive_path, manifest)
    if _read_manifest_bytes(built.manifest_path) != manifest_bytes:
        raise CheckpointError("built checkpoint manifest changed during publication")

    try:
        if not manifest_exists:
            manifest_partial = persistent_directory / f".{manifest_basename}.part"
            descriptor = _create_partial(manifest_partial)
            try:
                with os.fdopen(descriptor, "wb") as target:
                    target.write(manifest_bytes)
                    target.flush()
                    os.fsync(target.fileno())
            except OSError as exc:
                raise CheckpointError("checkpoint manifest copy failed") from exc
            _link_partial(manifest_partial, manifest_path, persistent_directory)

        if _read_manifest_bytes(manifest_path) != manifest_bytes:
            raise CheckpointError("published checkpoint manifest changed")
        if load_manifest(manifest_path) != manifest:
            raise CheckpointError("published checkpoint manifest does not match source")
    except BaseException:
        if created_archive is not None:
            _rollback_created_final(
                archive_path, created_archive, persistent_directory
            )
        raise

    return archive_path, manifest_path


def _copy_archive_to_staging(source_path: Path, destination_path: Path) -> None:
    source = _lstat(source_path, "persistent archive")
    if (
        stat.S_ISLNK(source.st_mode)
        or not stat.S_ISREG(source.st_mode)
        or source.st_nlink != 1
    ):
        raise CheckpointError("persistent archive must be a one-link regular file")
    source_descriptor: int | None = None
    destination_descriptor: int | None = None
    try:
        source_descriptor = os.open(
            source_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        destination_descriptor = os.open(
            destination_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        with (
            os.fdopen(source_descriptor, "rb") as source_stream,
            os.fdopen(destination_descriptor, "wb") as destination_stream,
        ):
            source_descriptor = None
            destination_descriptor = None
            opened = os.fstat(source_stream.fileno())
            if (
                opened.st_dev != source.st_dev
                or opened.st_ino != source.st_ino
                or opened.st_nlink != 1
            ):
                raise CheckpointError("persistent archive changed while being opened")
            shutil.copyfileobj(source_stream, destination_stream)
            destination_stream.flush()
            os.fsync(destination_stream.fileno())
            current = os.fstat(source_stream.fileno())
            if (
                current.st_size != opened.st_size
                or current.st_mtime_ns != opened.st_mtime_ns
                or current.st_nlink != 1
            ):
                raise CheckpointError("persistent archive changed while being copied")
    except CheckpointError:
        raise
    except OSError as exc:
        raise CheckpointError("persistent archive could not be copied to staging") from exc
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        if destination_descriptor is not None:
            os.close(destination_descriptor)


def _extract_archive_exclusively(
    archive_path: Path,
    inventory: tuple[_ArchiveMember, ...],
    staging_directory: Path,
) -> Path:
    trustsr_path = staging_directory / "trustsr"
    try:
        os.mkdir(trustsr_path, 0o700)
        _fsync_directory(staging_directory)
        descriptor = os.open(
            archive_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        with os.fdopen(descriptor, "rb") as stream:
            with tarfile.open(fileobj=stream, mode="r:") as archive:
                archive_members = iter(archive)
                observed_count = 0
                for member, expected in zip(archive_members, inventory, strict=False):
                    observed_count += 1
                    if (
                        member.name != expected.name
                        or member.isdir() != expected.is_directory
                        or member.size != expected.size
                        or not (member.isdir() or member.isreg())
                    ):
                        raise CheckpointError("archive inventory changed before extraction")
                    destination = staging_directory / PurePosixPath(expected.name)
                    if expected.is_directory:
                        os.mkdir(destination, 0o700)
                        _fsync_directory(destination.parent)
                        continue
                    source = archive.extractfile(member)
                    if source is None:
                        raise CheckpointError("archive regular file could not be extracted")
                    output_descriptor: int | None = os.open(
                        destination,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | os.O_CLOEXEC
                        | os.O_NOFOLLOW,
                        0o600,
                    )
                    try:
                        with source:
                            with os.fdopen(output_descriptor, "wb") as output:
                                output_descriptor = None
                                shutil.copyfileobj(source, output)
                                output.flush()
                                os.fsync(output.fileno())
                    finally:
                        if output_descriptor is not None:
                            os.close(output_descriptor)
                    _fsync_directory(destination.parent)
                if observed_count != len(inventory):
                    raise CheckpointError("archive inventory changed before extraction")
                try:
                    next(archive_members)
                except StopIteration:
                    pass
                else:
                    raise CheckpointError("archive inventory changed before extraction")
    except CheckpointError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise CheckpointError("archive could not be extracted exclusively") from exc
    return trustsr_path


def _normalize_staged_permissions(trustsr_path: Path) -> None:
    paths = [trustsr_path, *trustsr_path.rglob("*")]
    directories: list[Path] = []
    for path in paths:
        source = _lstat(path, "restored workspace entry")
        if stat.S_ISDIR(source.st_mode):
            os.chmod(path, 0o700, follow_symlinks=False)
            directories.append(path)
        elif stat.S_ISREG(source.st_mode) and source.st_nlink == 1:
            os.chmod(path, 0o600, follow_symlinks=False)
            _fsync_path(path)
        else:
            raise CheckpointError("restored workspace contains an unsafe entry")
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        _fsync_directory(directory)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise CheckpointError("renameat2 is unsupported; restore fails closed") from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise CheckpointError("live trustsr publication collision")
    if error_number == errno.ENOSYS:
        raise CheckpointError("renameat2 is unsupported; restore fails closed")
    raise CheckpointError(
        f"live trustsr publication failed: {os.strerror(error_number)}"
    )


def _require_absent_destination(destination: Path) -> None:
    try:
        destination.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise CheckpointError("live trustsr destination collision") from exc
    raise CheckpointError("live trustsr destination collision")


def restore_checkpoint(
    persistent_directory: Path,
    manifest_basename: str,
    workspace_root: Path,
    *,
    expected_reviewed_commit: str,
) -> Path:
    """Restore one explicit checkpoint through private staging and no-replace publication."""
    persistent_directory = Path(persistent_directory)
    workspace_root = Path(workspace_root)
    _require_directory(persistent_directory, "persistent directory")
    _require_directory(workspace_root, "workspace root")
    if (
        type(manifest_basename) is not str
        or PurePosixPath(manifest_basename).name != manifest_basename
        or "\\" in manifest_basename
    ):
        raise CheckpointError("manifest basename is unsafe")
    if not _is_lower_hex(expected_reviewed_commit, 40):
        raise CheckpointError("expected reviewed commit must be a lowercase digest")

    manifest_path = persistent_directory / manifest_basename
    declared_manifest = load_manifest(manifest_path)
    expected_manifest_basename = (
        declared_manifest.archive_basename.removesuffix(".tar") + ".json"
    )
    if manifest_basename != expected_manifest_basename:
        raise CheckpointError("manifest basename does not match archive")
    _validate_persistent_entries(
        persistent_directory,
        declared_manifest.archive_basename,
        manifest_basename,
        allow_selected_archive_only=False,
    )
    archive_path = persistent_directory / declared_manifest.archive_basename
    manifest = verify_checkpoint(archive_path, manifest_path)
    if manifest.reviewed_commit != expected_reviewed_commit:
        raise CheckpointError("checkpoint reviewed commit does not match expected commit")

    destination = workspace_root / "trustsr"
    _require_absent_destination(destination)
    staging_directory = Path(
        tempfile.mkdtemp(prefix=".phase2b3a-restore.", dir=workspace_root)
    )
    try:
        os.chmod(staging_directory, 0o700)
        staged_archive = staging_directory / manifest.archive_basename
        _copy_archive_to_staging(archive_path, staged_archive)
        inventory = _verify_archive_against_manifest(staged_archive, manifest)
        _fsync_directory(staging_directory)
        staged_trustsr = _extract_archive_exclusively(
            staged_archive, inventory, staging_directory
        )
        _validate_workspace_evidence(
            staging_directory,
            manifest.completed_stage,
            manifest.reviewed_commit,
            allow_legacy_a1=_is_exact_legacy_a1_checkpoint(manifest),
        )
        _normalize_staged_permissions(staged_trustsr)
        _require_absent_destination(destination)
        _rename_noreplace(staged_trustsr, destination)
        _fsync_directory(workspace_root)
        return destination
    finally:
        shutil.rmtree(staging_directory)


def _manifest_basename_is_safe(manifest_basename: object) -> bool:
    return (
        type(manifest_basename) is str
        and PurePosixPath(manifest_basename).name == manifest_basename
        and "\\" not in manifest_basename
    )


def _checkpoint_from_directory(
    directory: Path,
    manifest_basename: str,
    description: str,
    *,
    require_persistent_hygiene: bool = False,
) -> BuiltCheckpoint:
    directory = Path(directory)
    _require_directory(directory, description)
    if not _manifest_basename_is_safe(manifest_basename):
        raise CheckpointError("manifest basename is unsafe")
    manifest_path = directory / manifest_basename
    manifest = load_manifest(manifest_path)
    expected_manifest_basename = (
        manifest.archive_basename.removesuffix(".tar") + ".json"
    )
    if manifest_basename != expected_manifest_basename:
        raise CheckpointError("manifest basename does not match archive")
    if require_persistent_hygiene:
        _validate_persistent_entries(
            directory,
            manifest.archive_basename,
            manifest_basename,
            allow_selected_archive_only=False,
        )
    archive_path = directory / manifest.archive_basename
    verified = verify_checkpoint(archive_path, manifest_path)
    return BuiltCheckpoint(
        archive_path=archive_path,
        manifest_path=manifest_path,
        manifest=verified,
    )


def _success_payload(
    manifest: CheckpointManifest, command_name: str
) -> dict[str, object]:
    return {
        "archive_basename": manifest.archive_basename,
        "archive_sha256": manifest.archive_sha256,
        "archive_size_bytes": manifest.archive_size_bytes,
        "completed_stage": manifest.completed_stage,
        "manifest_basename": (
            manifest.archive_basename.removesuffix(".tar") + ".json"
        ),
        "reviewed_commit": manifest.reviewed_commit,
        "status": command_name,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("workspace_root", type=Path)
    build.add_argument("local_scratch", type=Path)
    build.add_argument("completed_stage")
    build.add_argument("reviewed_commit")

    publish = subparsers.add_parser("publish")
    publish.add_argument("local_scratch", type=Path)
    publish.add_argument("manifest_basename")
    publish.add_argument("persistent_directory", type=Path)

    verify = subparsers.add_parser("verify")
    verify.add_argument("persistent_directory", type=Path)
    verify.add_argument("manifest_basename")

    restore = subparsers.add_parser("restore")
    restore.add_argument("persistent_directory", type=Path)
    restore.add_argument("manifest_basename")
    restore.add_argument("workspace_root", type=Path)
    restore.add_argument("expected_reviewed_commit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            built = build_checkpoint(
                args.workspace_root,
                args.local_scratch,
                completed_stage=args.completed_stage,
                reviewed_commit=args.reviewed_commit,
            )
            manifest = built.manifest
        elif args.command == "publish":
            built = _checkpoint_from_directory(
                args.local_scratch, args.manifest_basename, "local scratch directory"
            )
            publish_checkpoint(built, args.persistent_directory)
            manifest = built.manifest
        elif args.command == "verify":
            built = _checkpoint_from_directory(
                args.persistent_directory,
                args.manifest_basename,
                "persistent directory",
                require_persistent_hygiene=True,
            )
            manifest = built.manifest
        elif args.command == "restore":
            built = _checkpoint_from_directory(
                args.persistent_directory,
                args.manifest_basename,
                "persistent directory",
                require_persistent_hygiene=True,
            )
            restore_checkpoint(
                args.persistent_directory,
                args.manifest_basename,
                args.workspace_root,
                expected_reviewed_commit=args.expected_reviewed_commit,
            )
            manifest = built.manifest
        else:  # pragma: no cover - argparse enforces the command choices.
            raise CheckpointError("unknown checkpoint command")
    except (CheckpointError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(canonical_json(_success_payload(manifest, args.command)).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
