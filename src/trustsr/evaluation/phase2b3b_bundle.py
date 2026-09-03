"""Atomic canonical bundle I/O for Phase 2B3-B calibration evidence."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from trustsr.jsonio import atomic_write_bytes, canonical_json

BUNDLE_MANIFEST_BASENAME = "phase2b3b-bundle-manifest.json"
BUNDLE_DOCUMENT_SCHEMAS = {
    "phase2b3b-calibration-result.json": "trustsr.phase2b3b-calibration.v1",
    "phase2b3b-calibration-cache-audit.json": ("trustsr.phase2b3b-calibration-cache-audit.v1"),
    "phase2b3b-calibration-runtime.json": "trustsr.phase2b3b-calibration-runtime.v1",
    "phase2b3b-calibration-replay.json": "trustsr.phase2b3b-calibration-replay.v1",
}

_MAX_FILE_BYTES = 5 * 1024 * 1024
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_MANIFEST_SCHEMA = "trustsr.phase2b3b-bundle-manifest.v1"
_RENAME_NOREPLACE = 1
_LIBC = ctypes.CDLL(None, use_errno=True)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _document_payload(name: str, value: object) -> bytes:
    if type(value) is not dict:
        raise TypeError(f"bundle document must be an exact JSON object: {name}")
    if value.get("schema") != BUNDLE_DOCUMENT_SCHEMAS[name]:
        raise ValueError(f"bundle document schema is invalid: {name}")
    payload = canonical_json(value)
    if len(payload) > _MAX_FILE_BYTES:
        raise ValueError(f"bundle document exceeds the 5 MiB limit: {name}")
    return payload


def _validated_payloads(documents: Mapping[str, object]) -> dict[str, bytes]:
    if not isinstance(documents, Mapping) or set(documents) != set(BUNDLE_DOCUMENT_SCHEMAS):
        raise ValueError("bundle documents must match the exact file allowlist")
    return {name: _document_payload(name, documents[name]) for name in BUNDLE_DOCUMENT_SCHEMAS}


def _manifest_from_payloads(payloads: Mapping[str, bytes]) -> dict[str, object]:
    if type(payloads) is not dict or tuple(payloads) != tuple(BUNDLE_DOCUMENT_SCHEMAS):
        raise ValueError("bundle payloads must use the exact canonical order")
    if any(type(payload) is not bytes for payload in payloads.values()):
        raise TypeError("bundle payloads must be immutable bytes")
    return {
        "schema": _MANIFEST_SCHEMA,
        "phase": "calibration",
        "files": [
            {
                "basename": name,
                "size_bytes": len(payloads[name]),
                "sha256": _sha256(payloads[name]),
            }
            for name in sorted(payloads)
        ],
    }


def build_phase2b3b_bundle_manifest(
    documents: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact four-file manifest without reading or writing paths."""

    return _manifest_from_payloads(_validated_payloads(documents))


def _canonical_object(payload: bytes, name: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"bundle file is not valid JSON: {name}") from exc
    if type(value) is not dict or canonical_json(value) != payload:
        raise ValueError(f"bundle file is not canonical JSON: {name}")
    return value


def _validate_snapshot(payloads: tuple[tuple[str, bytes], ...], manifest_payload: bytes) -> None:
    if type(payloads) is not tuple or tuple(name for name, _ in payloads) != tuple(
        BUNDLE_DOCUMENT_SCHEMAS
    ):
        raise ValueError("loaded bundle payload order is invalid")
    if any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not bytes
        for item in payloads
    ):
        raise TypeError("loaded bundle payloads must be immutable named bytes")
    if type(manifest_payload) is not bytes:
        raise TypeError("loaded bundle manifest must be immutable bytes")

    manifest = _canonical_object(manifest_payload, BUNDLE_MANIFEST_BASENAME)
    if set(manifest) != {"schema", "phase", "files"} or (
        manifest["schema"] != _MANIFEST_SCHEMA or manifest["phase"] != "calibration"
    ):
        raise ValueError("bundle manifest schema is invalid")
    entries = manifest["files"]
    if type(entries) is not list or len(entries) != len(BUNDLE_DOCUMENT_SCHEMAS):
        raise ValueError("bundle manifest file entries are invalid")

    payload_by_name = dict(payloads)
    for entry, expected_name in zip(entries, sorted(BUNDLE_DOCUMENT_SCHEMAS), strict=True):
        if type(entry) is not dict or set(entry) != {
            "basename",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("bundle manifest file entry schema is invalid")
        size = entry["size_bytes"]
        digest = entry["sha256"]
        if (
            entry["basename"] != expected_name
            or type(size) is not int
            or size < 0
            or size > _MAX_FILE_BYTES
            or type(digest) is not str
            or _DIGEST_PATTERN.fullmatch(digest) is None
        ):
            raise ValueError("bundle manifest file entry identity is invalid")
        payload = payload_by_name[expected_name]
        if len(payload) != size or _sha256(payload) != digest:
            raise ValueError("bundle file size or digest differs from the manifest")
        value = _canonical_object(payload, expected_name)
        if value.get("schema") != BUNDLE_DOCUMENT_SCHEMAS[expected_name]:
            raise ValueError(f"bundle document schema is invalid: {expected_name}")


@dataclass(frozen=True)
class LoadedPhase2B3BBundle:
    """Immutable, internally descriptor-consistent bytes from one bundle read."""

    payloads: tuple[tuple[str, bytes], ...] = field(repr=False)
    manifest_payload: bytes = field(repr=False)
    manifest_sha256: str

    def __post_init__(self) -> None:
        _validate_snapshot(self.payloads, self.manifest_payload)
        if (
            type(self.manifest_sha256) is not str
            or _DIGEST_PATTERN.fullmatch(self.manifest_sha256) is None
            or self.manifest_sha256 != _sha256(self.manifest_payload)
        ):
            raise ValueError("loaded bundle manifest identity is invalid")

    def documents(self) -> dict[str, dict[str, object]]:
        """Return freshly parsed documents so callers cannot mutate verified state."""

        return {name: json.loads(payload.decode("utf-8")) for name, payload in self.payloads}


@dataclass(frozen=True, init=False)
class BundleWriteReceipt:
    """Host-free identity returned only after descriptor-based publication checks."""

    manifest_sha256: str
    file_sha256s: tuple[tuple[str, str], ...]

    def __init__(self) -> None:
        raise TypeError("bundle write receipts are created only after publication checks")

    @classmethod
    def _from_loaded(cls, bundle: LoadedPhase2B3BBundle) -> BundleWriteReceipt:
        if type(bundle) is not LoadedPhase2B3BBundle:
            raise TypeError("bundle receipt requires an exact loaded bundle")
        bundle.__post_init__()
        value = object.__new__(cls)
        object.__setattr__(value, "manifest_sha256", bundle.manifest_sha256)
        object.__setattr__(
            value,
            "file_sha256s",
            tuple((name, _sha256(payload)) for name, payload in bundle.payloads),
        )
        value.__post_init__()
        return value

    def __post_init__(self) -> None:
        if (
            type(self.manifest_sha256) is not str
            or _DIGEST_PATTERN.fullmatch(self.manifest_sha256) is None
            or type(self.file_sha256s) is not tuple
            or tuple(name for name, _ in self.file_sha256s) != tuple(BUNDLE_DOCUMENT_SCHEMAS)
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[1]) is not str
                or _DIGEST_PATTERN.fullmatch(item[1]) is None
                for item in self.file_sha256s
            )
        ):
            raise ValueError("bundle write receipt identity is invalid")


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_canonical_directory(path: Path, label: str) -> int:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError(f"{label} must be an absolute canonical directory")
    try:
        if path.resolve(strict=True) != path.absolute():
            raise ValueError(f"{label} must be an absolute canonical directory")
    except OSError as exc:
        raise ValueError(f"{label} must be an absolute canonical directory") from exc

    descriptor = os.open(path.anchor, _directory_flags())
    try:
        for component in path.parts[1:]:
            if component in {"", ".", ".."}:
                raise ValueError(f"{label} must be an absolute canonical directory")
            next_descriptor = os.open(component, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except (OSError, ValueError) as exc:
        os.close(descriptor)
        raise ValueError(f"{label} must be an absolute canonical directory") from exc
    return descriptor


def _open_bundle_parent(bundle_dir: Path) -> tuple[int, str]:
    if (
        not isinstance(bundle_dir, Path)
        or not bundle_dir.is_absolute()
        or bundle_dir.name in {"", ".", ".."}
    ):
        raise ValueError("bundle directory must be an absolute canonical child path")
    return _open_canonical_directory(bundle_dir.parent, "bundle parent"), bundle_dir.name


def _read_regular_at(directory_fd: int, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise ValueError(f"bundle file is missing or unsafe: {name}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_FILE_BYTES:
            raise ValueError(f"bundle file is not a bounded regular file: {name}")
        chunks: list[bytes] = []
        remaining = _MAX_FILE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) != metadata.st_size or len(payload) > _MAX_FILE_BYTES:
            raise ValueError(f"bundle file size changed or exceeds the limit: {name}")
        return payload
    finally:
        os.close(descriptor)


def _read_bundle_at(directory_fd: int) -> LoadedPhase2B3BBundle:
    expected_names = {BUNDLE_MANIFEST_BASENAME, *BUNDLE_DOCUMENT_SCHEMAS}
    if set(os.listdir(directory_fd)) != expected_names:
        raise ValueError("bundle directory must contain the exact file allowlist")
    manifest_payload = _read_regular_at(directory_fd, BUNDLE_MANIFEST_BASENAME)
    payloads = tuple(
        (name, _read_regular_at(directory_fd, name)) for name in BUNDLE_DOCUMENT_SCHEMAS
    )
    if set(os.listdir(directory_fd)) != expected_names:
        raise ValueError("bundle directory changed during verification")
    return LoadedPhase2B3BBundle(
        payloads=payloads,
        manifest_payload=manifest_payload,
        manifest_sha256=_sha256(manifest_payload),
    )


def _read_child_bundle_at(parent_fd: int, name: str) -> LoadedPhase2B3BBundle:
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError("bundle directory cannot be opened safely") from exc
    try:
        return _read_bundle_at(descriptor)
    finally:
        os.close(descriptor)


def _read_child_if_present(parent_fd: int, name: str) -> LoadedPhase2B3BBundle | None:
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError("bundle directory cannot be opened safely") from exc
    try:
        return _read_bundle_at(descriptor)
    finally:
        os.close(descriptor)


def read_phase2b3b_bundle(bundle_dir: Path) -> LoadedPhase2B3BBundle:
    """Read exactly five allowlisted files through no-follow descriptors."""

    descriptor = _open_canonical_directory(bundle_dir, "bundle directory")
    try:
        return _read_bundle_at(descriptor)
    finally:
        os.close(descriptor)


def _receipt(bundle: LoadedPhase2B3BBundle) -> BundleWriteReceipt:
    return BundleWriteReceipt._from_loaded(bundle)


def _new_staging_at(parent_fd: int, target_name: str) -> tuple[str, int]:
    for _ in range(128):
        name = f".{target_name}.{secrets.token_hex(12)}.tmp"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        try:
            return name, os.open(name, _directory_flags(), dir_fd=parent_fd)
        except OSError:
            os.rmdir(name, dir_fd=parent_fd)
            raise
    raise FileExistsError("unable to allocate a unique bundle staging directory")


def _write_regular_at(directory_fd: int, name: str, payload: bytes) -> None:
    descriptor_path = Path(f"/proc/self/fd/{directory_fd}")
    atomic_write_bytes(descriptor_path / name, payload)


def _remove_staging_at(parent_fd: int, staging_fd: int, staging_name: str) -> None:
    for name in os.listdir(staging_fd):
        os.unlink(name, dir_fd=staging_fd)
    os.rmdir(staging_name, dir_fd=parent_fd)


def _rename_noreplace_at(parent_fd: int, source: str, target: str) -> None:
    renameat2 = getattr(_LIBC, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is required for fail-closed publication")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            parent_fd,
            os.fsencode(source),
            parent_fd,
            os.fsencode(target),
            _RENAME_NOREPLACE,
        )
        != 0
    ):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), target)


def _matches_expected(
    bundle: LoadedPhase2B3BBundle,
    payloads: Mapping[str, bytes],
    manifest_payload: bytes,
) -> bool:
    return dict(bundle.payloads) == payloads and bundle.manifest_payload == manifest_payload


def write_phase2b3b_bundle(
    bundle_dir: Path,
    *,
    result: Mapping[str, object],
    cache_audit: Mapping[str, object],
    runtime: Mapping[str, object],
    replay: Mapping[str, object],
) -> BundleWriteReceipt:
    """Atomically publish one exact bundle, or reuse byte-identical evidence."""

    documents = dict(
        zip(
            BUNDLE_DOCUMENT_SCHEMAS,
            (result, cache_audit, runtime, replay),
            strict=True,
        )
    )
    payloads = _validated_payloads(documents)
    manifest_payload = canonical_json(_manifest_from_payloads(payloads))
    parent_fd, target_name = _open_bundle_parent(bundle_dir)
    try:
        existing = _read_child_if_present(parent_fd, target_name)
        if existing is not None:
            if not _matches_expected(existing, payloads, manifest_payload):
                raise ValueError("existing bundle has different bytes")
            return _receipt(existing)

        staging_name, staging_fd = _new_staging_at(parent_fd, target_name)
        published = False
        try:
            try:
                for name, payload in payloads.items():
                    _write_regular_at(staging_fd, name, payload)
                _write_regular_at(staging_fd, BUNDLE_MANIFEST_BASENAME, manifest_payload)
                os.fsync(staging_fd)
                staged = _read_bundle_at(staging_fd)
                if not _matches_expected(staged, payloads, manifest_payload):
                    raise ValueError("staged bundle has different bytes")
                try:
                    _rename_noreplace_at(parent_fd, staging_name, target_name)
                except FileExistsError:
                    existing = _read_child_bundle_at(parent_fd, target_name)
                else:
                    published = True
                    os.fsync(parent_fd)
                    existing = _read_child_bundle_at(parent_fd, target_name)
            finally:
                if not published:
                    _remove_staging_at(parent_fd, staging_fd, staging_name)
        finally:
            os.close(staging_fd)

        if not _matches_expected(existing, payloads, manifest_payload):
            raise ValueError("existing bundle has different bytes")
        return _receipt(existing)
    finally:
        os.close(parent_fd)
