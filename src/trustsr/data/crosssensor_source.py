"""Resumable, integrity-checked acquisition of the frozen crosssensor source."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from trustsr.data.provenance import DatasetSource, LfsObject

_CROSSSENSOR_OBJECT = "sen2naipv2-crosssensor.taco"
_MINIMUM_FREE_BYTES = 15 * 1024**3
_SHA256 = re.compile(r"[0-9a-f]{64}")


class SourceIntegrityError(RuntimeError):
    """Raised when an acquired source object differs from its frozen specification."""


@dataclass(frozen=True)
class VerifiedSourceObject:
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SourcePaths:
    final: Path
    partial: Path
    quarantine: Path


def require_crosssensor_object(source: DatasetSource) -> LfsObject:
    """Return the sole inventory entry permitted for crosssensor acquisition."""
    if not isinstance(source, DatasetSource):
        raise TypeError("source must be a validated DatasetSource")
    matches = tuple(item for item in source.objects if item.path == _CROSSSENSOR_OBJECT)
    if len(matches) != 1:
        raise ValueError("source must contain exactly one crosssensor object")
    object_spec = matches[0]
    if _SHA256.fullmatch(object_spec.sha256) is None or object_spec.size_bytes <= 0:
        raise ValueError("crosssensor object specification is invalid")
    return object_spec


def require_cloud_confirmation(storage_root: Path, confirmed_cloud_storage: bool) -> Path:
    """Validate the explicit storage root and require destructive-write confirmation."""
    if not isinstance(storage_root, Path):
        raise TypeError("storage_root must be a pathlib.Path")
    if not confirmed_cloud_storage:
        raise ValueError("explicit cloud storage confirmation is required")
    if storage_root.is_symlink():
        raise ValueError("storage_root must not be a symlink")
    if not storage_root.is_dir():
        raise ValueError("storage_root must be an existing directory")
    root = storage_root.resolve(strict=True)
    if root == Path("/") or root == Path.home().resolve():
        raise ValueError("storage_root must not be the filesystem or home directory")
    return root


def require_free_space(storage_root: Path, *, minimum_bytes: int) -> None:
    """Fail closed unless the explicit storage root has more than the required capacity."""
    if shutil.disk_usage(storage_root).free <= minimum_bytes:
        raise ValueError("storage_root must have more than 15 GiB free")


def _confined(root: Path, relative: Path) -> Path:
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError("derived source path escapes storage_root") from None
    return candidate


def source_paths(storage_root: Path, object_spec: LfsObject) -> SourcePaths:
    """Derive digest-qualified source and quarantine paths confined below storage root."""
    if _SHA256.fullmatch(object_spec.sha256) is None:
        raise ValueError("object SHA-256 must be a lowercase 64-character digest")
    if object_spec.path != _CROSSSENSOR_OBJECT:
        raise ValueError("only the crosssensor object may be acquired")
    final = _confined(
        storage_root,
        Path("trustsr", "phase2b1a", "source", object_spec.sha256, object_spec.path),
    )
    partial = _confined(
        storage_root,
        final.relative_to(storage_root).with_name(f"{final.name}.part"),
    )
    quarantine = _confined(
        storage_root,
        Path("trustsr", "phase2b1a", "quarantine", object_spec.sha256),
    )
    return SourcePaths(final=final, partial=partial, quarantine=quarantine)


def _require_https_url(transport_url: str) -> None:
    if not isinstance(transport_url, str):
        raise TypeError("transport_url must be a string")
    try:
        parsed = urlsplit(transport_url)
        hostname = parsed.hostname
    except ValueError as error:
        raise ValueError("transport_url must be an HTTPS transport URL") from error
    if (
        parsed.scheme.lower() != "https"
        or not parsed.netloc
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError(
            "transport_url must be an HTTPS transport URL without user info or fragment"
        )


def curl_arguments(transport_url: str, partial_path: Path) -> tuple[str, ...]:
    """Build the fixed non-shell curl invocation for a resumable partial download."""
    _require_https_url(transport_url)
    return (
        "curl",
        "--fail",
        "--location",
        "--retry",
        "5",
        "--continue-at",
        "-",
        "--output",
        str(partial_path),
        transport_url,
    )


def verify_crosssensor(path: Path, object_spec: LfsObject) -> VerifiedSourceObject:
    """Stream-verify exact byte count and SHA-256 against the frozen inventory entry."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    try:
        size_bytes = 0
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                size_bytes += len(block)
                digest.update(block)
    except OSError as error:
        raise SourceIntegrityError(f"source object cannot be read: {path}") from error
    sha256 = digest.hexdigest()
    if size_bytes != object_spec.size_bytes:
        raise SourceIntegrityError(
            f"source object size {size_bytes} does not match expected size {object_spec.size_bytes}"
        )
    if sha256 != object_spec.sha256:
        raise SourceIntegrityError(
            f"source object SHA-256 {sha256} does not match expected SHA-256 {object_spec.sha256}"
        )
    return VerifiedSourceObject(path=path, size_bytes=size_bytes, sha256=sha256)


def quarantine_completed_partial(partial_path: Path, quarantine_directory: Path) -> Path | None:
    """Move a completed invalid partial to a fresh quarantine name without overwriting data."""
    if not partial_path.exists() or partial_path.is_symlink():
        return None
    quarantine_directory.mkdir(parents=True, exist_ok=True)
    base = quarantine_directory / partial_path.name
    index = 0
    while True:
        candidate = base if index == 0 else base.with_name(f"{base.name}.{index}")
        try:
            os.link(partial_path, candidate)
        except FileExistsError:
            index += 1
            continue
        break
    partial_path.unlink()
    return candidate


def acquire_crosssensor(
    source: DatasetSource,
    storage_root: Path,
    transport_url: str,
    *,
    confirmed_cloud_storage: bool,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> VerifiedSourceObject:
    """Acquire the one permitted source object, preserving resumable failure state."""
    object_spec = require_crosssensor_object(source)
    _require_https_url(transport_url)
    root = require_cloud_confirmation(storage_root, confirmed_cloud_storage)
    require_free_space(root, minimum_bytes=_MINIMUM_FREE_BYTES)
    paths = source_paths(root, object_spec)
    if paths.final.exists() or paths.final.is_symlink():
        return verify_crosssensor(paths.final, object_spec)

    paths.partial.parent.mkdir(parents=True, exist_ok=True)
    runner(curl_arguments(transport_url, paths.partial), check=True, text=True)
    try:
        verified = verify_crosssensor(paths.partial, object_spec)
    except SourceIntegrityError:
        quarantine_completed_partial(paths.partial, paths.quarantine)
        raise
    os.replace(paths.partial, paths.final)
    return VerifiedSourceObject(paths.final, verified.size_bytes, verified.sha256)
