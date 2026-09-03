"""Atomic canonical bundle I/O for Phase 2B3-B calibration evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from trustsr.jsonio import atomic_write_bytes, canonical_json

BUNDLE_MANIFEST_BASENAME = "phase2b3b-bundle-manifest.json"
BUNDLE_DOCUMENT_SCHEMAS = {
    "phase2b3b-calibration-result.json": "trustsr.phase2b3b-calibration.v1",
    "phase2b3b-calibration-cache-audit.json": (
        "trustsr.phase2b3b-calibration-cache-audit.v1"
    ),
    "phase2b3b-calibration-runtime.json": "trustsr.phase2b3b-calibration-runtime.v1",
    "phase2b3b-calibration-replay.json": "trustsr.phase2b3b-calibration-replay.v1",
}

_MAX_FILE_BYTES = 5 * 1024 * 1024
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


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
    return {
        name: _document_payload(name, documents[name])
        for name in BUNDLE_DOCUMENT_SCHEMAS
    }


def build_phase2b3b_bundle_manifest(
    documents: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact four-file manifest without reading or writing paths."""

    payloads = _validated_payloads(documents)
    return {
        "schema": "trustsr.phase2b3b-bundle-manifest.v1",
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


@dataclass(frozen=True)
class LoadedPhase2B3BBundle:
    """Descriptor-verified immutable bytes from one complete bundle."""

    payloads: tuple[tuple[str, bytes], ...] = field(repr=False)
    manifest_payload: bytes = field(repr=False)
    manifest_sha256: str

    def __post_init__(self) -> None:
        if tuple(name for name, _ in self.payloads) != tuple(BUNDLE_DOCUMENT_SCHEMAS):
            raise ValueError("loaded bundle payload order is invalid")
        if any(type(payload) is not bytes for _, payload in self.payloads):
            raise TypeError("loaded bundle payloads must be immutable bytes")
        if type(self.manifest_payload) is not bytes or (
            type(self.manifest_sha256) is not str
            or _DIGEST_PATTERN.fullmatch(self.manifest_sha256) is None
            or self.manifest_sha256 != _sha256(self.manifest_payload)
        ):
            raise ValueError("loaded bundle manifest identity is invalid")

    def documents(self) -> dict[str, dict[str, object]]:
        """Return freshly parsed documents so callers cannot mutate verified state."""

        result: dict[str, dict[str, object]] = {}
        for name, payload in self.payloads:
            value = json.loads(payload.decode("utf-8"))
            if type(value) is not dict:
                raise AssertionError("verified bundle document stopped being a JSON object")
            result[name] = value
        return result


@dataclass(frozen=True)
class BundleWriteReceipt:
    """Host-free identity of an atomically published or reused bundle."""

    manifest_sha256: str
    file_sha256s: tuple[tuple[str, str], ...]


def _canonical_bundle_parent(bundle_dir: Path) -> Path:
    if not isinstance(bundle_dir, Path) or not bundle_dir.is_absolute() or not bundle_dir.name:
        raise ValueError("bundle directory must be an absolute canonical child path")
    parent = bundle_dir.parent
    try:
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("bundle parent must be an existing canonical directory")
        resolved = parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError("bundle parent must be an existing canonical directory") from exc
    if resolved != parent.absolute():
        raise ValueError("bundle parent must be an existing canonical directory")
    return resolved


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


def _canonical_object(payload: bytes, name: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"bundle file is not valid JSON: {name}") from exc
    if type(value) is not dict or canonical_json(value) != payload:
        raise ValueError(f"bundle file is not canonical JSON: {name}")
    return value


def read_phase2b3b_bundle(bundle_dir: Path) -> LoadedPhase2B3BBundle:
    """Read exactly five allowlisted files through no-follow descriptors."""

    if not isinstance(bundle_dir, Path) or not bundle_dir.is_absolute():
        raise ValueError("bundle directory must be an absolute canonical directory")
    try:
        if bundle_dir.is_symlink() or not bundle_dir.is_dir():
            raise ValueError("bundle directory must be an existing canonical directory")
        resolved = bundle_dir.resolve(strict=True)
    except OSError as exc:
        raise ValueError("bundle directory must be an existing canonical directory") from exc
    if resolved != bundle_dir.absolute():
        raise ValueError("bundle directory must be an existing canonical directory")

    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(bundle_dir, directory_flags)
    except OSError as exc:
        raise ValueError("bundle directory cannot be opened safely") from exc
    try:
        expected_names = {BUNDLE_MANIFEST_BASENAME, *BUNDLE_DOCUMENT_SCHEMAS}
        if set(os.listdir(directory_fd)) != expected_names:
            raise ValueError("bundle directory must contain the exact file allowlist")
        manifest_payload = _read_regular_at(directory_fd, BUNDLE_MANIFEST_BASENAME)
        manifest = _canonical_object(manifest_payload, BUNDLE_MANIFEST_BASENAME)
        if set(manifest) != {"schema", "phase", "files"} or (
            manifest["schema"] != "trustsr.phase2b3b-bundle-manifest.v1"
            or manifest["phase"] != "calibration"
        ):
            raise ValueError("bundle manifest schema is invalid")
        entries = manifest["files"]
        if type(entries) is not list or len(entries) != len(BUNDLE_DOCUMENT_SCHEMAS):
            raise ValueError("bundle manifest file entries are invalid")

        expected_order = sorted(BUNDLE_DOCUMENT_SCHEMAS)
        payload_by_name: dict[str, bytes] = {}
        for entry, expected_name in zip(entries, expected_order, strict=True):
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
            payload = _read_regular_at(directory_fd, expected_name)
            if len(payload) != size or _sha256(payload) != digest:
                raise ValueError("bundle file size or digest differs from the manifest")
            value = _canonical_object(payload, expected_name)
            if value.get("schema") != BUNDLE_DOCUMENT_SCHEMAS[expected_name]:
                raise ValueError(f"bundle document schema is invalid: {expected_name}")
            payload_by_name[expected_name] = payload
        if set(os.listdir(directory_fd)) != expected_names:
            raise ValueError("bundle directory changed during verification")
    finally:
        os.close(directory_fd)

    ordered_payloads = tuple(
        (name, payload_by_name[name]) for name in BUNDLE_DOCUMENT_SCHEMAS
    )
    return LoadedPhase2B3BBundle(
        payloads=ordered_payloads,
        manifest_payload=manifest_payload,
        manifest_sha256=_sha256(manifest_payload),
    )


def _receipt(bundle: LoadedPhase2B3BBundle) -> BundleWriteReceipt:
    return BundleWriteReceipt(
        manifest_sha256=bundle.manifest_sha256,
        file_sha256s=tuple((name, _sha256(payload)) for name, payload in bundle.payloads),
    )


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
    manifest_payload = canonical_json(build_phase2b3b_bundle_manifest(documents))
    parent = _canonical_bundle_parent(bundle_dir)

    if bundle_dir.exists() or bundle_dir.is_symlink():
        loaded = read_phase2b3b_bundle(bundle_dir)
        if dict(loaded.payloads) != payloads or loaded.manifest_payload != manifest_payload:
            raise ValueError("existing bundle has different bytes")
        return _receipt(loaded)

    staging = Path(tempfile.mkdtemp(dir=parent, prefix=f".{bundle_dir.name}.", suffix=".tmp"))
    published = False
    try:
        for name, payload in payloads.items():
            atomic_write_bytes(staging / name, payload)
        atomic_write_bytes(staging / BUNDLE_MANIFEST_BASENAME, manifest_payload)
        staging_fd = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        try:
            os.replace(staging, bundle_dir)
            published = True
        except OSError:
            if not bundle_dir.exists():
                raise
        parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)

    loaded = read_phase2b3b_bundle(bundle_dir)
    if dict(loaded.payloads) != payloads or loaded.manifest_payload != manifest_payload:
        raise ValueError("existing bundle has different bytes")
    return _receipt(loaded)
