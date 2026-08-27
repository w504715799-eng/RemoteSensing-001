"""Strict, offline provenance types for the SEN2NAIPv2 source inventory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

_SCHEMA = "trustsr.sen2naipv2-source.v1"
_EXPECTED_BANDS = ("B04", "B03", "B02", "B08")
_TOP_LEVEL_KEYS = {
    "schema",
    "repository",
    "revision",
    "license_claim",
    "card_sha256",
    "bands",
    "scale",
    "lr_shape",
    "hr_shape",
    "declared_total_bytes",
    "objects",
}
_OBJECT_KEYS = {"path", "sha256", "size_bytes"}


@dataclass(frozen=True)
class LfsObject:
    """A single Git LFS object recorded by path, digest, and byte size."""

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class DatasetSource:
    """Validated metadata describing the pinned SEN2NAIPv2 source."""

    schema: str
    repository: str
    revision: str
    license_claim: str
    card_sha256: str
    bands: tuple[str, ...]
    scale: int
    lr_shape: tuple[int, int]
    hr_shape: tuple[int, int]
    declared_total_bytes: int
    objects: tuple[LfsObject, ...]

    @property
    def total_bytes(self) -> int:
        """Return the sum of all recorded object sizes."""
        return sum(item.size_bytes for item in self.objects)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _check_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    actual = set(value)
    unknown = actual - expected
    missing = expected - actual
    if unknown:
        raise ValueError(f"unknown {label} keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing {label} keys: {sorted(missing)}")


def _require_string(value: Any, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be a string")
    return value


def _require_integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _validate_digest(value: Any, label: str, length: int) -> str:
    digest = _require_string(value, label)
    if re.fullmatch(rf"[0-9a-f]{{{length}}}", digest) is None:
        raise ValueError(f"{label} must be {length} lowercase hexadecimal characters")
    return digest


def _validate_relative_path(value: Any) -> str:
    path = _require_string(value, "object path")
    if not path or "\\" in path:
        raise ValueError("object path must be a non-empty relative path")
    posix = PurePosixPath(path)
    windows = PureWindowsPath(path)
    if posix.is_absolute() or windows.is_absolute():
        raise ValueError("object path must be relative")
    if ".." in posix.parts or ".." in windows.parts:
        raise ValueError("object path must not contain '..' components")
    return path


def _validate_shape(value: Any, label: str) -> tuple[int, int]:
    if type(value) is not list or len(value) != 2:
        raise ValueError(f"{label} must contain two positive integers")
    dimensions = tuple(_require_integer(item, f"{label} dimensions") for item in value)
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(f"{label} must contain positive integers")
    return dimensions


def load_dataset_source(path: Path) -> DatasetSource:
    """Load and strictly validate one local SEN2NAIPv2 provenance JSON file."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    _check_keys(payload, _TOP_LEVEL_KEYS, "top-level")

    schema = _require_string(payload["schema"], "schema")
    if schema != _SCHEMA:
        raise ValueError(f"schema must be {_SCHEMA}")
    repository = _require_string(payload["repository"], "repository")
    revision = _validate_digest(payload["revision"], "revision", 40)
    license_claim = _require_string(payload["license_claim"], "license_claim")
    card_sha256 = _validate_digest(payload["card_sha256"], "card_sha256", 64)

    bands = payload["bands"]
    if type(bands) is not list or tuple(bands) != _EXPECTED_BANDS:
        raise ValueError(f"bands must equal {_EXPECTED_BANDS!r}")
    if any(type(band) is not str for band in bands):
        raise ValueError("bands must contain strings")

    scale = _require_integer(payload["scale"], "scale")
    if scale != 4:
        raise ValueError("scale must equal 4")
    lr_shape = _validate_shape(payload["lr_shape"], "lr_shape")
    hr_shape = _validate_shape(payload["hr_shape"], "hr_shape")
    if hr_shape != tuple(dimension * scale for dimension in lr_shape):
        raise ValueError("hr_shape must equal lr_shape multiplied by scale")

    declared_total_bytes = _require_integer(
        payload["declared_total_bytes"], "declared_total_bytes"
    )
    if declared_total_bytes < 0:
        raise ValueError("declared_total_bytes must not be negative")

    raw_objects = payload["objects"]
    if type(raw_objects) is not list or not raw_objects:
        raise ValueError("objects must not be empty")
    objects: list[LfsObject] = []
    paths: set[str] = set()
    for raw_object in raw_objects:
        _check_keys(raw_object, _OBJECT_KEYS, "object")
        object_path = _validate_relative_path(raw_object["path"])
        if object_path in paths:
            raise ValueError(f"duplicate object path: {object_path}")
        paths.add(object_path)
        object_sha256 = _validate_digest(raw_object["sha256"], "object sha256", 64)
        size_bytes = _require_integer(raw_object["size_bytes"], "size_bytes")
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        objects.append(LfsObject(object_path, object_sha256, size_bytes))

    object_tuple = tuple(objects)
    total_bytes = sum(item.size_bytes for item in object_tuple)
    if declared_total_bytes != total_bytes:
        raise ValueError("declared_total_bytes must equal object sizes")

    return DatasetSource(
        schema=schema,
        repository=repository,
        revision=revision,
        license_claim=license_claim,
        card_sha256=card_sha256,
        bands=tuple(bands),
        scale=scale,
        lr_shape=lr_shape,
        hr_shape=hr_shape,
        declared_total_bytes=declared_total_bytes,
        objects=object_tuple,
    )
