"""Strict Phase 2B2-A inputs derived from the frozen crosssensor subset."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window, bounds, transform

from trustsr.contracts import SRPair
from trustsr.data.subset_manifest import load_subset_manifest

POST_MANIFEST_SHA256 = "c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a"
PHASE2B1B_AUDIT_SHA256 = "d8964033958594a23ac7056519894d508977bfd2cc13da50a5833024274f3e90"
REFLECTANCE_SCALE = 10_000.0
RAW_DTYPE = "uint16"
CROP_POLICY = "center_crop_lr_1_hr_4_v1"
NORMALIZATION_POLICY = "uint16_divide_10000_no_clip_v1"
SMOKE_SPLITS = ("calibration", "development", "internal_test")
SMOKE_BINS = (0, 1, 2, 3)
_BANDS = ("B04", "B03", "B02", "B08")
_BOUNDS_TOLERANCE_M = 1e-3


@dataclass(frozen=True)
class CrosssensorPairMetadata:
    manifest_sha256: str
    sample_id: str
    split: str
    spatial_group_id: str
    days_between: int
    correlation_bin: int
    selection_round: int
    lr_asset_sha256: str
    hr_asset_sha256: str
    lr_crop_transform: tuple[float, float, float, float, float, float]
    hr_crop_transform: tuple[float, float, float, float, float, float]
    crop_bounds: tuple[float, float, float, float]
    crop_policy: str
    normalization_policy: str


@dataclass(frozen=True)
class LoadedCrosssensorPair:
    pair: SRPair
    metadata: CrosssensorPairMetadata


@dataclass(frozen=True)
class _LoadedRaster:
    crop: np.ndarray
    crop_transform: tuple[float, float, float, float, float, float]
    crop_bounds: tuple[float, float, float, float]
    asset_sha256: str


def _require_unique_strings(
    records: Sequence[Mapping[str, object]], field: str
) -> None:
    values = [record.get(field) for record in records]
    if any(type(value) is not str or not value for value in values):
        raise ValueError(f"smoke record {field} must be a non-empty string")
    if len(set(values)) != len(values):
        raise ValueError(f"smoke records require unique {field} values")


def select_input_smoke_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Select the frozen 12-cell CPU input smoke set."""

    cells: dict[tuple[str, int], list[Mapping[str, object]]] = {
        (split, bin_index): []
        for split in SMOKE_SPLITS
        for bin_index in SMOKE_BINS
    }
    for record in records:
        if record.get("selection_round") != 1 or record.get("days_between") != -1:
            continue
        key = (record.get("split"), record.get("correlation_bin"))
        if key in cells:
            cells[key].append(record)

    for key, candidates in cells.items():
        if len(candidates) != 1:
            raise ValueError(
                "smoke selection requires exactly one record for "
                f"split={key[0]}, correlation_bin={key[1]}"
            )

    selected = tuple(
        cells[(split, bin_index)][0]
        for split in SMOKE_SPLITS
        for bin_index in SMOKE_BINS
    )
    _require_unique_strings(selected, "sample_id")
    _require_unique_strings(selected, "spatial_group_id")
    return selected


def load_crosssensor_records(
    storage_root: Path,
    manifest_path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, object], ...]:
    """Load the frozen all-assets Phase 2B1B post-manifest."""

    if not isinstance(storage_root, Path) or not isinstance(manifest_path, Path):
        raise TypeError("storage_root and manifest_path must be pathlib.Path values")
    if expected_sha256 != POST_MANIFEST_SHA256:
        raise ValueError("expected the frozen post-manifest SHA-256")
    if storage_root.is_symlink() or not storage_root.is_dir():
        raise ValueError("storage_root must be an existing non-symlink directory")
    resolved_root = storage_root.resolve(strict=True)
    if resolved_root != storage_root.absolute():
        raise ValueError("storage_root must not contain symlink path components")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("post-manifest must be an existing regular file")
    resolved_manifest = manifest_path.resolve(strict=True)
    expected_path = (
        resolved_root
        / "trustsr"
        / "phase2b1b"
        / "selections"
        / expected_sha256
        / "samples.jsonl"
    )
    if resolved_manifest != expected_path:
        raise ValueError("manifest must be the digest-addressed frozen post-manifest")
    records = load_subset_manifest(
        resolved_manifest,
        expected_sha256=expected_sha256,
    )
    if any(
        record["lr_asset"] is None or record["hr_asset"] is None
        for record in records
    ):
        raise ValueError("Phase 2B2-A requires the all-assets post-manifest")
    return records


def _require_string(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"record {field} must be a non-empty string")
    return value


def _require_integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int:
        raise ValueError(f"record {field} must be an integer")
    return value


def _require_asset(record: Mapping[str, object], kind: str) -> Mapping[str, object]:
    asset = record.get(f"{kind}_asset")
    if not isinstance(asset, Mapping):
        raise ValueError(f"record {kind}_asset must be present")
    return asset


def _sha256(path: Path) -> tuple[int, str]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            hasher.update(chunk)
    return size, hasher.hexdigest()


def _transform_tuple(value: object) -> tuple[float, float, float, float, float, float]:
    return (
        float(value.a),
        float(value.b),
        float(value.c),
        float(value.d),
        float(value.e),
        float(value.f),
    )


def _require_asset_metadata(
    asset: Mapping[str, object],
    *,
    shape: tuple[int, int, int],
    dtype: str,
    crs: str,
    raster_transform: tuple[float, float, float, float, float, float],
    nodata: float | int | None,
    minimum: float,
    maximum: float,
) -> None:
    observed = {
        "shape": list(shape),
        "dtype": dtype,
        "crs": crs,
        "transform": list(raster_transform),
        "nodata": nodata,
        "minimum": minimum,
        "maximum": maximum,
    }
    if any(asset.get(field) != value for field, value in observed.items()):
        raise ValueError("asset GeoTIFF metadata does not match the sidecar")


def _load_asset(
    storage_root: Path,
    record: Mapping[str, object],
    kind: str,
) -> _LoadedRaster:
    sample_id = _require_string(record, "sample_id")
    split = _require_string(record, "split")
    asset = _require_asset(record, kind)
    expected_relative = f"subset-v1/{split}/{sample_id}/{kind}.tif"
    if asset.get("relative_path") != expected_relative:
        raise ValueError("asset must use the canonical Phase 2B1B path")
    phase_root = (storage_root / "trustsr" / "phase2b1b").resolve(strict=True)
    path = phase_root / expected_relative
    if path.is_symlink() or not path.is_file():
        raise ValueError("asset must be a confined regular GeoTIFF")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(phase_root)
    except ValueError:
        raise ValueError("asset path escapes the Phase 2B1B pixel tree") from None

    size_bytes, digest = _sha256(resolved)
    if asset.get("size_bytes") != size_bytes or asset.get("sha256") != digest:
        raise ValueError("asset bytes do not match the sidecar")

    with rasterio.open(resolved) as dataset:
        if dataset.driver != "GTiff":
            raise ValueError("asset must use the GTiff driver")
        if dataset.count != 4:
            raise ValueError("asset must contain exactly four bands")
        if set(dataset.dtypes) != {RAW_DTYPE}:
            raise ValueError("asset bands must all use uint16")
        if any(description is not None for description in dataset.descriptions) and (
            dataset.descriptions != _BANDS
        ):
            raise ValueError("asset band descriptions must equal B04, B03, B02, B08")
        if dataset.crs is None:
            raise ValueError("asset must declare a CRS")
        if dataset.nodata is not None:
            raise ValueError("asset nodata must be None")
        pixels = dataset.read()
        if not np.isfinite(pixels).all():
            raise ValueError("asset pixels must be finite")
        minimum = float(pixels.min())
        maximum = float(pixels.max())
        raster_transform = _transform_tuple(dataset.transform)
        _require_asset_metadata(
            asset,
            shape=(dataset.count, dataset.height, dataset.width),
            dtype=dataset.dtypes[0],
            crs=dataset.crs.to_string(),
            raster_transform=raster_transform,
            nodata=dataset.nodata,
            minimum=minimum,
            maximum=maximum,
        )
        if minimum < 0.0 or maximum > REFLECTANCE_SCALE:
            raise ValueError("asset raw reflectance must be in [0, 10000]")
        window = Window(1, 1, 128, 128) if kind == "lr" else Window(4, 4, 512, 512)
        row_start = int(window.row_off)
        column_start = int(window.col_off)
        crop = pixels[
            :,
            row_start : row_start + int(window.height),
            column_start : column_start + int(window.width),
        ]
        crop_transform = _transform_tuple(transform(window, dataset.transform))
        crop_bounds = tuple(float(value) for value in bounds(window, dataset.transform))
    return _LoadedRaster(
        crop=crop,
        crop_transform=crop_transform,
        crop_bounds=crop_bounds,
        asset_sha256=digest,
    )


def _to_reflectance(value: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.array(value, copy=True)).to(torch.float32)
    return tensor.div_(REFLECTANCE_SCALE).contiguous()


def load_crosssensor_pair(
    storage_root: Path,
    record: Mapping[str, object],
    *,
    manifest_sha256: str,
) -> LoadedCrosssensorPair:
    """Load, align and normalize one frozen Phase 2B1B pair."""

    if manifest_sha256 != POST_MANIFEST_SHA256:
        raise ValueError("expected the frozen post-manifest SHA-256")
    if not isinstance(storage_root, Path) or storage_root.is_symlink() or not storage_root.is_dir():
        raise ValueError("storage_root must be an existing non-symlink directory")
    resolved_root = storage_root.resolve(strict=True)
    if resolved_root != storage_root.absolute():
        raise ValueError("storage_root must not contain symlink path components")
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")

    lr = _load_asset(resolved_root, record, "lr")
    hr = _load_asset(resolved_root, record, "hr")
    if any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=_BOUNDS_TOLERANCE_M)
        for left, right in zip(lr.crop_bounds, hr.crop_bounds, strict=True)
    ):
        raise ValueError("cropped LR/HR bounds do not align")

    sample_id = _require_string(record, "sample_id")
    pair = SRPair(
        sample_id=sample_id,
        source=f"sen2naipv2-crosssensor/{manifest_sha256}",
        lr=_to_reflectance(lr.crop),
        hr=_to_reflectance(hr.crop),
        scale=4,
    )
    pair.validate()
    metadata = CrosssensorPairMetadata(
        manifest_sha256=manifest_sha256,
        sample_id=sample_id,
        split=_require_string(record, "split"),
        spatial_group_id=_require_string(record, "spatial_group_id"),
        days_between=_require_integer(record, "days_between"),
        correlation_bin=_require_integer(record, "correlation_bin"),
        selection_round=_require_integer(record, "selection_round"),
        lr_asset_sha256=lr.asset_sha256,
        hr_asset_sha256=hr.asset_sha256,
        lr_crop_transform=lr.crop_transform,
        hr_crop_transform=hr.crop_transform,
        crop_bounds=lr.crop_bounds,
        crop_policy=CROP_POLICY,
        normalization_policy=NORMALIZATION_POLICY,
    )
    return LoadedCrosssensorPair(pair=pair, metadata=metadata)
