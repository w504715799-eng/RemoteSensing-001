"""Cloud-only adapter for extracting raw GeoTIFF pairs from TACO v1 files."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from rasterio.io import MemoryFile

from trustsr.data.crosssensor_manifest import ExtractedAsset
from trustsr.data.crosssensor_schema import AcquisitionTimes
from trustsr.jsonio import atomic_write_bytes

_EXPECTED_BANDS = ("B04", "B03", "B02", "B08")
_READER_VERSION = "0.4.5"
_FORMAT_VERSION = "0.4.0"
_FORMAT_KEYS = ("taco_version", "version")
_LR_DIMENSIONS = (130, 130)
_HR_DIMENSIONS = (520, 520)
_RESOLUTION_TOLERANCE_M = 1e-6
_BOUNDS_TOLERANCE_M = 1e-3
_VSI_SUBFILE_RE = re.compile(r"/vsisubfile/(\d+)_(\d+),(.+)")


@dataclass(frozen=True)
class _Raster:
    payload: bytes
    time_start: str
    shape: tuple[int, int, int]
    dtype: str
    crs: str
    transform: tuple[float, float, float, float, float, float]
    nodata: float | int | None
    minimum: float
    maximum: float
    bounds: tuple[float, float, float, float]
    resolution: tuple[float, float]


def require_tacoreader_v1() -> ModuleType:
    """Load the pinned legacy reader only for explicit cloud extraction."""
    try:
        installed_version = importlib.metadata.version("tacoreader")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "tacoreader is unavailable; run the cloud bootstrap with tacoreader==0.4.5"
        ) from exc
    if installed_version != _READER_VERSION:
        raise RuntimeError(
            f"tacoreader must be exactly {_READER_VERSION}, observed {installed_version!r}"
        )
    try:
        reader = importlib.import_module("tacoreader")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "tacoreader is unavailable; run the cloud bootstrap with tacoreader==0.4.5"
        ) from exc
    return reader


def _require_taco_path(taco_path: Path) -> Path:
    if not isinstance(taco_path, Path):
        raise TypeError("taco_path must be a pathlib.Path")
    return taco_path


def _validate_collection(metadata: object) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError("TACO collection metadata must be a mapping")
    values = [metadata[key] for key in _FORMAT_KEYS if key in metadata]
    if not values:
        raise ValueError("TACO collection must provide exactly one legacy version field")
    if len(values) == 2 and values[0] != values[1]:
        raise ValueError("TACO collection version fields are ambiguous")
    if len(values) > 2 or any(type(value) is not str for value in values):
        raise ValueError("TACO collection version fields must be unambiguous strings")
    if values[0] != _FORMAT_VERSION:
        raise ValueError(f"TACO collection version must equal {_FORMAT_VERSION}")


def _load_verified_top(taco_path: Path) -> Any:
    reader = require_tacoreader_v1()
    source = str(_require_taco_path(taco_path))
    _validate_collection(reader.load_metadata(source))
    return reader.load(source)


def _records_from_top(top: object) -> tuple[Mapping[str, object], ...]:
    to_dict = getattr(top, "to_dict", None)
    if not callable(to_dict):
        raise ValueError("TACO top-level table does not provide records")
    records = to_dict(orient="records")
    if not isinstance(records, list) or not all(isinstance(record, Mapping) for record in records):
        raise ValueError("TACO top-level records must be mappings")
    normalized: list[Mapping[str, object]] = []
    for record in records:
        normalized_record = dict(record)
        if "stac:time_start" in normalized_record:
            normalized_record["stac:time_start"] = _canonical_time_start(
                normalized_record["stac:time_start"], "top-level stac:time_start"
            )
        normalized.append(normalized_record)
    return tuple(normalized)


def load_top_level_records(taco_path: Path) -> tuple[Mapping[str, object], ...]:
    """Load top-level records after validating the legacy collection format."""
    return _records_from_top(_load_verified_top(taco_path))


def _require_source_index(source_index: int) -> int:
    if type(source_index) is not int or source_index < 0:
        raise ValueError("source_index must be a non-negative integer")
    return source_index


def _require_output_root(output_root: Path) -> Path:
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a pathlib.Path")
    return output_root


def _require_bands(bands: tuple[str, ...]) -> None:
    if bands != _EXPECTED_BANDS:
        raise ValueError(f"bands must equal {_EXPECTED_BANDS!r}")


def _nested_row(nested: object, index: int) -> Mapping[str, object]:
    iloc = getattr(nested, "iloc", None)
    if iloc is None:
        raise ValueError("TACO nested table does not expose rows")
    row = iloc[index]
    to_dict = getattr(row, "to_dict", None)
    if callable(to_dict):
        row = to_dict()
    if not isinstance(row, Mapping):
        raise ValueError("TACO nested asset row must be a mapping")
    return row


def _canonical_time_start(value: object, label: str) -> str:
    if type(value) is str and value:
        return value
    if type(value) in {int, float} and math.isfinite(value):
        try:
            return datetime.fromtimestamp(value, UTC).isoformat().replace("+00:00", "Z")
        except (OverflowError, OSError, ValueError) as exc:
            raise ValueError(f"{label} must be a valid timestamp") from exc
    raise ValueError(f"{label} must be a non-empty string or finite Unix epoch seconds")


def _time_start(nested: object, index: int) -> str:
    value = _nested_row(nested, index).get("stac:time_start")
    return _canonical_time_start(value, "nested asset stac:time_start")


def _metadata_integer(row: Mapping[str, object], key: str, *, minimum: int) -> int:
    value = row.get(key)
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) is not int or value < minimum:
        raise ValueError(f"nested asset {key} must be an integer no smaller than {minimum}")
    return value


def _read_asset_payload(taco_path: Path, nested: object, index: int) -> bytes:
    value = nested.read(index)
    if isinstance(value, bytes):
        return value
    if type(value) is not str:
        raise ValueError("TACO nested asset must be bytes or a local VSI subfile descriptor")

    row = _nested_row(nested, index)
    if row.get("tortilla:file_format") != "GTiff":
        raise ValueError("VSI nested asset tortilla:file_format must equal GTiff")
    if row.get("internal:subfile") != value:
        raise ValueError("VSI nested asset must equal its internal:subfile metadata")
    match = _VSI_SUBFILE_RE.fullmatch(value)
    if match is None:
        raise ValueError("TACO nested asset is not a valid local VSI subfile descriptor")

    offset = int(match.group(1))
    length = int(match.group(2))
    if offset != _metadata_integer(row, "tortilla:offset", minimum=0):
        raise ValueError("VSI offset does not match nested asset metadata")
    if length != _metadata_integer(row, "tortilla:length", minimum=1):
        raise ValueError("VSI length does not match nested asset metadata")
    try:
        descriptor_source = Path(match.group(3)).resolve(strict=True)
        verified_source = taco_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("VSI descriptor source must resolve to the verified TACO source") from exc
    if descriptor_source != verified_source:
        raise ValueError("VSI descriptor source must equal the verified TACO source")
    if offset + length > verified_source.stat().st_size:
        raise ValueError("VSI byte range exceeds the verified TACO source")

    with verified_source.open("rb") as stream:
        stream.seek(offset)
        payload = stream.read(length)
    if len(payload) != length:
        raise ValueError("VSI byte range could not be read completely")
    return payload


def load_crosssensor_metadata(
    taco_path: Path,
) -> tuple[tuple[Mapping[str, object], ...], tuple[AcquisitionTimes, ...]]:
    """Load top records and source-ordered LR/HR times without reading pixels."""
    top = _load_verified_top(taco_path)
    records = _records_from_top(top)
    read = getattr(top, "read", None)
    if not callable(read):
        raise ValueError("TACO top-level table does not expose nested metadata")
    times: list[AcquisitionTimes] = []
    for source_index in range(len(records)):
        nested = read(source_index)
        if len(nested) != 2:
            raise ValueError("each sample must contain exactly two nested metadata rows")
        times.append(
            AcquisitionTimes(
                lr_time_start=_time_start(nested, 0),
                hr_time_start=_time_start(nested, 1),
            )
        )
    return records, tuple(times)


def _validate_descriptions(descriptions: tuple[str | None, ...]) -> None:
    for expected, description in zip(_EXPECTED_BANDS, descriptions, strict=True):
        if description is not None and description != expected:
            raise ValueError("GeoTIFF band descriptions contradict the frozen source bands")


def _inspect_raster(payload: object, time_start: str) -> _Raster:
    if not isinstance(payload, bytes):
        raise ValueError("TACO nested assets must contain raw bytes")
    try:
        with MemoryFile(payload) as memory:
            with memory.open() as dataset:
                if dataset.driver != "GTiff":
                    raise ValueError("nested asset must use the GTiff driver")
                if dataset.count != len(_EXPECTED_BANDS):
                    raise ValueError("GeoTIFF assets must contain exactly four bands")
                if len(set(dataset.dtypes)) != 1:
                    raise ValueError("GeoTIFF bands must use one uniform dtype")
                _validate_descriptions(dataset.descriptions)
                pixels = dataset.read()
                if not np.isfinite(pixels).all():
                    raise ValueError("GeoTIFF pixels must be finite")
                if dataset.crs is None:
                    raise ValueError("GeoTIFF assets must declare a CRS")
                nodata = dataset.nodata
                if nodata is not None and not math.isfinite(nodata):
                    raise ValueError("GeoTIFF nodata must be finite when present")
                transform = dataset.transform
                return _Raster(
                    payload=payload,
                    time_start=time_start,
                    shape=(dataset.count, dataset.height, dataset.width),
                    dtype=dataset.dtypes[0],
                    crs=dataset.crs.to_string(),
                    transform=(
                        float(transform.a),
                        float(transform.b),
                        float(transform.c),
                        float(transform.d),
                        float(transform.e),
                        float(transform.f),
                    ),
                    nodata=nodata,
                    minimum=float(pixels.min()),
                    maximum=float(pixels.max()),
                    bounds=tuple(float(value) for value in dataset.bounds),
                    resolution=tuple(float(value) for value in dataset.res),
                )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("nested asset is not a readable GeoTIFF") from exc


def _raster_kind(raster: _Raster) -> str:
    dimensions = raster.shape[1:]
    if dimensions == _LR_DIMENSIONS:
        expected_resolution = 10.0
        kind = "lr"
    elif dimensions == _HR_DIMENSIONS:
        expected_resolution = 2.5
        kind = "hr"
    else:
        raise ValueError("GeoTIFF dimensions must be either 130x130 or 520x520")
    if not all(
        math.isclose(
            value, expected_resolution, rel_tol=0.0, abs_tol=_RESOLUTION_TOLERANCE_M
        )
        for value in raster.resolution
    ):
        raise ValueError(f"{kind.upper()} GeoTIFF resolution is not {expected_resolution:g} m")
    return kind


def _pair_by_dimensions(rasters: tuple[_Raster, _Raster]) -> tuple[_Raster, _Raster]:
    grouped: dict[str, list[_Raster]] = {"lr": [], "hr": []}
    for raster in rasters:
        grouped[_raster_kind(raster)].append(raster)
    if len(grouped["lr"]) != 1 or len(grouped["hr"]) != 1:
        raise ValueError("nested assets must contain exactly one LR and one HR GeoTIFF")
    return grouped["lr"][0], grouped["hr"][0]


def _validate_pair_geometry(lr: _Raster, hr: _Raster) -> None:
    if lr.crs != hr.crs:
        raise ValueError("LR and HR GeoTIFF CRS values must match")
    if not all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=_BOUNDS_TOLERANCE_M)
        for left, right in zip(lr.bounds, hr.bounds, strict=True)
    ):
        raise ValueError("LR and HR GeoTIFF bounds must match")


def _asset(relative_path: str, raster: _Raster) -> ExtractedAsset:
    return ExtractedAsset(
        relative_path=relative_path,
        size_bytes=len(raster.payload),
        sha256=hashlib.sha256(raster.payload).hexdigest(),
        shape=raster.shape,
        dtype=raster.dtype,
        crs=raster.crs,
        transform=raster.transform,
        nodata=raster.nodata,
        minimum=raster.minimum,
        maximum=raster.maximum,
        time_start=raster.time_start,
    )


def _write_or_reuse(output_root: Path, assets: tuple[tuple[str, _Raster], ...]) -> None:
    targets = tuple(
        (output_root / relative_path, raster.payload) for relative_path, raster in assets
    )
    for target, payload in targets:
        if target.exists():
            if not target.is_file() or target.read_bytes() != payload:
                raise ValueError(f"existing asset {target} has different bytes")
    for target, payload in targets:
        if not target.exists():
            atomic_write_bytes(target, payload)


def extract_pair(
    taco_path: Path,
    source_index: int,
    output_root: Path,
    bands: tuple[str, ...],
) -> tuple[ExtractedAsset, ExtractedAsset]:
    """Extract and validate one raw LR/HR pair into its selected sample directory."""
    _require_bands(bands)
    source_index = _require_source_index(source_index)
    output_root = _require_output_root(output_root)
    top = _load_verified_top(taco_path)
    nested = top.read(source_index)
    if len(nested) != 2:
        raise ValueError("selected sample must contain exactly two nested assets")
    rasters = tuple(
        _inspect_raster(
            _read_asset_payload(taco_path, nested, index), _time_start(nested, index)
        )
        for index in range(2)
    )
    lr, hr = _pair_by_dimensions(rasters)  # type: ignore[arg-type]
    _validate_pair_geometry(lr, hr)
    _write_or_reuse(output_root, (("lr.tif", lr), ("hr.tif", hr)))
    return _asset("lr.tif", lr), _asset("hr.tif", hr)
