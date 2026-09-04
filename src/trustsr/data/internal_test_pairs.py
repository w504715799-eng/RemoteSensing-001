"""One-time-access boundary for loading Phase 2B3-C test pixels."""

from __future__ import annotations

import hashlib
import math
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.io import MemoryFile
from rasterio.windows import Window, bounds, transform

from trustsr.contracts import SRPair
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    RAW_DTYPE,
    RAW_NODATA,
    RAW_RADIOMETRIC_MAX,
    REFLECTANCE_SCALE,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
    RadiometricSaturation,
)

INTERNAL_TEST_SIZE = 120
_DAYS = (-1, 0, 1)
_BINS = (0, 1, 2, 3)
_ROUNDS = tuple(range(1, 11))
_IDENTITIES = ("sample_id", "selection_sha256", "spatial_group_id")
_GUARD_AUTHORITY = object()
_BANDS = ("B04", "B03", "B02", "B08")
_MAX_ASSET_BYTES = 16 * 1024**2
_BOUNDS_TOLERANCE_M = 1e-3
type PairLoader = Callable[..., LoadedCrosssensorPair]


@dataclass(frozen=True)
class PixelsOpenedAccessGuard:
    """Capability issued only after the immutable ledger records pixel access."""

    evaluation_id: str
    ledger_event_sha256: str
    _authority: object = field(repr=False, compare=False)


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _issue_pixels_opened_access_guard(
    evaluation_id: str, ledger_event_sha256: str
) -> PixelsOpenedAccessGuard:
    """Issue a process-local capability from the ledger implementation."""

    return PixelsOpenedAccessGuard(
        evaluation_id=_digest(evaluation_id, "evaluation ID"),
        ledger_event_sha256=_digest(ledger_event_sha256, "ledger event"),
        _authority=_GUARD_AUTHORITY,
    )


def _validate_guard(value: object) -> PixelsOpenedAccessGuard:
    if type(value) is not PixelsOpenedAccessGuard or value._authority is not _GUARD_AUTHORITY:
        raise ValueError("an authenticated pixels_opened access guard is required")
    _digest(value.evaluation_id, "access guard evaluation ID")
    _digest(value.ledger_event_sha256, "access guard ledger event")
    return value


def _string(record: Mapping[str, object], field_name: str) -> str:
    value = record.get(field_name)
    if type(value) is not str or not value:
        raise ValueError(f"internal_test record {field_name} must be a non-empty string")
    return value


def _integer(record: Mapping[str, object], field_name: str) -> int:
    value = record.get(field_name)
    if type(value) is not int:
        raise ValueError(f"internal_test record {field_name} must be an integer")
    return value


def _asset_digest(record: Mapping[str, object], kind: str) -> str:
    asset = record.get(f"{kind}_asset")
    if not isinstance(asset, Mapping) or not asset:
        raise ValueError(f"internal_test record requires a non-empty {kind}_asset")
    return _digest(asset.get("sha256"), f"internal_test {kind}_asset")


def validate_internal_test_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Require the exact 12-stratum by 10-round internal-test design."""

    if isinstance(records, str | bytes) or not isinstance(records, Sequence):
        raise TypeError("internal_test records must be a stable sequence")
    if len(records) != INTERNAL_TEST_SIZE:
        raise ValueError("internal_test records require exactly 120 rows")
    values: list[Mapping[str, object]] = []
    strata = {(day, bin_index): [] for day in _DAYS for bin_index in _BINS}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("internal_test records must be mappings")
        if _string(record, "split") != "internal_test":
            raise ValueError("only internal_test records may cross the pixel boundary")
        _string(record, "sample_id")
        _digest(record.get("selection_sha256"), "internal_test selection_sha256")
        _digest(record.get("spatial_group_id"), "internal_test spatial_group_id")
        _asset_digest(record, "lr")
        _asset_digest(record, "hr")
        day = _integer(record, "days_between")
        bin_index = _integer(record, "correlation_bin")
        selection_round = _integer(record, "selection_round")
        if (day, bin_index) not in strata or selection_round not in _ROUNDS:
            raise ValueError("internal_test record has an invalid stratum or selection round")
        strata[(day, bin_index)].append(selection_round)
        values.append(record)
    for identity in _IDENTITIES:
        observed = [_string(record, identity) for record in values]
        if len(set(observed)) != INTERNAL_TEST_SIZE:
            raise ValueError(f"internal_test records require unique {identity} values")
    if any(tuple(sorted(rounds)) != _ROUNDS for rounds in strata.values()):
        raise ValueError("internal_test records require 12 strata with rounds 1 through 10")
    return tuple(values)


def _validate_loaded(
    loaded: object, record: Mapping[str, object]
) -> LoadedCrosssensorPair:
    if type(loaded) is not LoadedCrosssensorPair:
        raise TypeError("internal_test pair loader must return LoadedCrosssensorPair")
    if type(loaded.pair) is not SRPair or type(loaded.metadata) is not CrosssensorPairMetadata:
        raise TypeError("internal_test pair loader returned forged pair state")
    loaded.pair.validate()
    metadata = loaded.metadata
    sample_id = _string(record, "sample_id")
    if loaded.pair.sample_id != sample_id or metadata.sample_id != sample_id:
        raise ValueError("internal_test pair loader output is out of input order")
    if loaded.pair.source != f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}":
        raise ValueError("internal_test pair loader output has the wrong source")
    if (
        metadata.split != "internal_test"
        or metadata.manifest_sha256 != POST_MANIFEST_SHA256
        or metadata.crop_policy != CROP_POLICY
        or metadata.normalization_policy != PHASE2B3A_NORMALIZATION_POLICY
    ):
        raise ValueError("internal_test pair loader output has invalid split, manifest, or policy")
    for saturation in (metadata.lr_saturation, metadata.hr_saturation):
        if type(saturation) is not RadiometricSaturation:
            raise ValueError("internal_test pair loader output requires saturation records")
        try:
            saturation.__post_init__()
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("internal_test pair loader output has invalid saturation") from exc
    if (
        metadata.lr_asset_sha256 != _asset_digest(record, "lr")
        or metadata.hr_asset_sha256 != _asset_digest(record, "hr")
    ):
        raise ValueError("internal_test pair loader output asset identity differs")
    expected = {
        "spatial_group_id": _string(record, "spatial_group_id"),
        "days_between": _integer(record, "days_between"),
        "correlation_bin": _integer(record, "correlation_bin"),
        "selection_round": _integer(record, "selection_round"),
    }
    if any(getattr(metadata, name) != value for name, value in expected.items()):
        raise ValueError("internal_test pair loader output metadata differs")
    return loaded


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_storage_root(storage_root: Path) -> int:
    if storage_root.is_symlink() or not storage_root.is_dir():
        raise ValueError("storage_root must be an existing non-symlink directory")
    try:
        if storage_root.resolve(strict=True) != storage_root.absolute():
            raise ValueError("storage_root must not contain symlink components")
        descriptor = os.open(storage_root, _directory_flags())
    except OSError as exc:
        raise ValueError("storage_root cannot be opened safely") from exc
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):  # pragma: no cover - O_DIRECTORY
        os.close(descriptor)
        raise ValueError("storage_root descriptor is not a directory")
    return descriptor


def _open_child_directory(parent: int, name: str) -> int:
    if not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError("asset directory component is not canonical")
    try:
        return os.open(name, _directory_flags(), dir_fd=parent)
    except OSError as exc:
        raise ValueError("asset directory tree is missing or contains a symlink") from exc


def _read_asset_at(directory: int, name: str, expected_size: object) -> bytes:
    if type(expected_size) is not int or not 0 < expected_size <= _MAX_ASSET_BYTES:
        raise ValueError("asset size_bytes is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory)
    except OSError as exc:
        raise ValueError("asset must be a confined regular non-symlink file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise ValueError("asset descriptor type or size differs from the sidecar")
        chunks: list[bytes] = []
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("asset descriptor ended before its declared size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("asset descriptor exceeds its declared size")
        after = os.fstat(descriptor)
        stable = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if not stable:
            raise ValueError("asset changed while its descriptor was being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _transform_tuple(value: object) -> tuple[float, float, float, float, float, float]:
    return (
        float(value.a),  # type: ignore[attr-defined]
        float(value.b),  # type: ignore[attr-defined]
        float(value.c),  # type: ignore[attr-defined]
        float(value.d),  # type: ignore[attr-defined]
        float(value.e),  # type: ignore[attr-defined]
        float(value.f),  # type: ignore[attr-defined]
    )


@dataclass(frozen=True)
class _RasterInput:
    crop: np.ndarray
    crop_transform: tuple[float, float, float, float, float, float]
    crop_bounds: tuple[float, float, float, float]
    asset_sha256: str
    saturation: RadiometricSaturation


def _load_raster_bytes(
    payload: bytes, asset: Mapping[str, object], kind: str
) -> _RasterInput:
    digest = hashlib.sha256(payload).hexdigest()
    if digest != asset.get("sha256"):
        raise ValueError("asset bytes do not match the sidecar digest")
    try:
        with MemoryFile(payload) as memory, memory.open() as dataset:
            if dataset.driver != "GTiff" or dataset.count != 4:
                raise ValueError("asset must be a four-band GTiff")
            if set(dataset.dtypes) != {RAW_DTYPE}:
                raise ValueError("asset bands must all use uint16")
            if any(description is not None for description in dataset.descriptions) and (
                dataset.descriptions != _BANDS
            ):
                raise ValueError("asset band descriptions are invalid")
            if dataset.crs is None or dataset.nodata != RAW_NODATA:
                raise ValueError("asset CRS or nodata sentinel is invalid")
            pixels = dataset.read()
            if np.any(dataset.read_masks() == 0) or not np.isfinite(pixels).all():
                raise ValueError("asset contains invalid or non-finite pixels")
            minimum = float(pixels.min())
            maximum = float(pixels.max())
            raster_transform = _transform_tuple(dataset.transform)
            observed = {
                "shape": [dataset.count, dataset.height, dataset.width],
                "dtype": dataset.dtypes[0],
                "crs": dataset.crs.to_string(),
                "transform": list(raster_transform),
                "nodata": dataset.nodata,
                "minimum": minimum,
                "maximum": maximum,
            }
            if any(asset.get(key) != value for key, value in observed.items()):
                raise ValueError("asset GeoTIFF metadata does not match its sidecar")
            if minimum < 0.0 or maximum > RAW_RADIOMETRIC_MAX:
                raise ValueError("asset raw reflectance must be in [0, 32767]")
            window = Window(1, 1, 128, 128) if kind == "lr" else Window(4, 4, 512, 512)
            crop = pixels[
                :,
                int(window.row_off) : int(window.row_off + window.height),
                int(window.col_off) : int(window.col_off + window.width),
            ]
            clipped = crop > REFLECTANCE_SCALE
            saturation = RadiometricSaturation(
                raw_crop_minimum=int(crop.min()),
                raw_crop_maximum=int(crop.max()),
                clipped_high_count=int(np.count_nonzero(clipped)),
                clipped_high_by_band=tuple(
                    int(np.count_nonzero(clipped[index])) for index in range(4)
                ),
            )
            crop = np.minimum(crop, int(REFLECTANCE_SCALE))
            crop_transform = _transform_tuple(transform(window, dataset.transform))
            crop_bounds = tuple(float(value) for value in bounds(window, dataset.transform))
    except (OSError, rasterio.errors.RasterioError) as exc:
        raise ValueError("asset bytes are not a readable GeoTIFF") from exc
    return _RasterInput(crop, crop_transform, crop_bounds, digest, saturation)


def _load_internal_test_pair(
    storage_root: Path,
    record: Mapping[str, object],
    *,
    manifest_sha256: str,
    normalization_policy: str,
) -> LoadedCrosssensorPair:
    if manifest_sha256 != POST_MANIFEST_SHA256:
        raise ValueError("expected the frozen post-manifest SHA-256")
    if normalization_policy != PHASE2B3A_NORMALIZATION_POLICY:
        raise ValueError("expected the frozen saturation normalization policy")
    sample_id = _string(record, "sample_id")
    root_fd = _open_storage_root(storage_root)
    current = root_fd
    try:
        for component in ("trustsr", "phase2b1b", "subset-v1", "internal_test", sample_id):
            child = _open_child_directory(current, component)
            if current != root_fd:
                os.close(current)
            current = child
        rasters: dict[str, _RasterInput] = {}
        for kind in ("lr", "hr"):
            asset = record.get(f"{kind}_asset")
            if not isinstance(asset, Mapping):  # validation invariant, fail closed
                raise ValueError("internal_test asset metadata is missing")
            expected_relative = f"subset-v1/internal_test/{sample_id}/{kind}.tif"
            if asset.get("relative_path") != expected_relative:
                raise ValueError("asset relative_path is not the exact frozen layout")
            payload = _read_asset_at(current, f"{kind}.tif", asset.get("size_bytes"))
            rasters[kind] = _load_raster_bytes(payload, asset, kind)
    finally:
        if current != root_fd:
            os.close(current)
        os.close(root_fd)
    lr = rasters["lr"]
    hr = rasters["hr"]
    if any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=_BOUNDS_TOLERANCE_M)
        for left, right in zip(lr.crop_bounds, hr.crop_bounds, strict=True)
    ):
        raise ValueError("cropped LR and HR bounds do not align")
    lr_tensor = torch.from_numpy(np.array(lr.crop, copy=True)).to(torch.float32)
    hr_tensor = torch.from_numpy(np.array(hr.crop, copy=True)).to(torch.float32)
    pair = SRPair(
        sample_id=sample_id,
        source=f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}",
        lr=lr_tensor.div_(REFLECTANCE_SCALE).contiguous(),
        hr=hr_tensor.div_(REFLECTANCE_SCALE).contiguous(),
        scale=4,
    )
    pair.validate()
    metadata = CrosssensorPairMetadata(
        manifest_sha256=POST_MANIFEST_SHA256,
        sample_id=sample_id,
        split="internal_test",
        spatial_group_id=_string(record, "spatial_group_id"),
        days_between=_integer(record, "days_between"),
        correlation_bin=_integer(record, "correlation_bin"),
        selection_round=_integer(record, "selection_round"),
        lr_asset_sha256=lr.asset_sha256,
        hr_asset_sha256=hr.asset_sha256,
        lr_crop_transform=lr.crop_transform,
        hr_crop_transform=hr.crop_transform,
        crop_bounds=lr.crop_bounds,
        crop_policy=CROP_POLICY,
        normalization_policy=PHASE2B3A_NORMALIZATION_POLICY,
        lr_saturation=lr.saturation,
        hr_saturation=hr.saturation,
    )
    return LoadedCrosssensorPair(pair=pair, metadata=metadata)


def load_internal_test_pairs(
    storage_root: Path,
    records: Sequence[Mapping[str, object]],
    access_guard: object,
    *,
    pair_loader: PairLoader | None = None,
) -> tuple[LoadedCrosssensorPair, ...]:
    """Validate metadata and access capability before the first asset open."""

    _validate_guard(access_guard)
    if not isinstance(storage_root, Path):
        raise TypeError("storage_root must be a pathlib.Path")
    if pair_loader is not None and not callable(pair_loader):
        raise TypeError("pair_loader must be callable")
    validated = validate_internal_test_records(records)
    loader = _load_internal_test_pair if pair_loader is None else pair_loader
    loaded = tuple(
        _validate_loaded(
            loader(
                storage_root,
                record,
                manifest_sha256=POST_MANIFEST_SHA256,
                normalization_policy=PHASE2B3A_NORMALIZATION_POLICY,
            ),
            record,
        )
        for record in validated
    )
    if tuple(item.pair.sample_id for item in loaded) != tuple(
        _string(record, "sample_id") for record in validated
    ):
        raise ValueError("internal_test pair loader output is not in input order")
    return loaded
