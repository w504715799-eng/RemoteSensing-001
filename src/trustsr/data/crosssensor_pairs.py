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
RAW_NODATA = 65_535.0
CROP_POLICY = "center_crop_lr_1_hr_4_v1"
LEGACY_NORMALIZATION_POLICY = "uint16_divide_10000_no_clip_v1"
PHASE2B3A_NORMALIZATION_POLICY = "uint16_saturate_10000_divide_10000_v2"
NORMALIZATION_POLICY = LEGACY_NORMALIZATION_POLICY
NODATA_POLICY = "uint16_sentinel_65535_reject_invalid_v1"
RAW_RADIOMETRIC_MAX = 32_767
SMOKE_SPLITS = ("calibration", "development", "internal_test")
SMOKE_BINS = (0, 1, 2, 3)
DEVELOPMENT_DAYS = (-1, 0, 1)
DEVELOPMENT_BINS = (0, 1, 2, 3)
DEVELOPMENT_ROUNDS = tuple(range(1, 11))
_BANDS = ("B04", "B03", "B02", "B08")
_BOUNDS_TOLERANCE_M = 1e-3


@dataclass(frozen=True)
class RadiometricSaturation:
    raw_crop_minimum: int
    raw_crop_maximum: int
    clipped_high_count: int
    clipped_high_by_band: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if type(self.clipped_high_by_band) is not tuple:
            raise TypeError("radiometric saturation band counts must use an exact tuple")
        values = (
            self.raw_crop_minimum,
            self.raw_crop_maximum,
            self.clipped_high_count,
            *self.clipped_high_by_band,
        )
        if any(type(value) is not int for value in values):
            raise TypeError("radiometric saturation statistics must use built-in integers")
        if len(self.clipped_high_by_band) != len(_BANDS):
            raise ValueError("radiometric saturation requires one count per ordered band")
        if any(value < 0 for value in values):
            raise ValueError("radiometric saturation statistics must be non-negative")
        if self.raw_crop_minimum > self.raw_crop_maximum:
            raise ValueError("radiometric saturation crop minimum exceeds maximum")
        if self.raw_crop_maximum > RAW_RADIOMETRIC_MAX:
            raise ValueError("radiometric saturation crop maximum exceeds 32767")
        if self.clipped_high_count != sum(self.clipped_high_by_band):
            raise ValueError("radiometric saturation band counts do not match total")
        if (self.raw_crop_maximum > REFLECTANCE_SCALE) != (
            self.clipped_high_count > 0
        ):
            raise ValueError(
                "radiometric saturation maximum and clipped count are inconsistent"
            )


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
    lr_saturation: RadiometricSaturation | None = None
    hr_saturation: RadiometricSaturation | None = None


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
    saturation: RadiometricSaturation | None


def _require_unique_strings(
    records: Sequence[Mapping[str, object]], field: str
) -> None:
    values = [record.get(field) for record in records]
    if any(type(value) is not str or not value for value in values):
        raise ValueError(f"smoke record {field} must be a non-empty string")
    if len(set(values)) != len(values):
        raise ValueError(f"smoke records require unique {field} values")


def select_development_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Select and validate the frozen 120-record development set."""

    selected = tuple(record for record in records if record.get("split") == "development")
    if len(selected) != 120:
        raise ValueError("development selection must contain exactly 120 records")
    try:
        _require_unique_strings(selected, "sample_id")
        _require_unique_strings(selected, "spatial_group_id")
    except ValueError as error:
        raise ValueError(f"development selection is invalid: {error}") from error
    cells: dict[tuple[int, int], list[int]] = {
        (day, bin_index): []
        for day in DEVELOPMENT_DAYS
        for bin_index in DEVELOPMENT_BINS
    }
    for record in selected:
        day = _require_integer(record, "days_between")
        bin_index = _require_integer(record, "correlation_bin")
        selection_round = _require_integer(record, "selection_round")
        if (day, bin_index) not in cells:
            raise ValueError("development record has an invalid stratum")
        cells[(day, bin_index)].append(selection_round)
    if any(tuple(sorted(rounds)) != DEVELOPMENT_ROUNDS for rounds in cells.values()):
        raise ValueError("development strata must each contain selection rounds 1 through 10")
    return selected


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


def select_development_smoke_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Filter the canonical 12-cell input smoke set to development only."""

    selected = tuple(
        record
        for record in select_input_smoke_records(records)
        if record["split"] == "development"
    )
    if len(selected) != 4 or [
        record["correlation_bin"] for record in selected
    ] != list(SMOKE_BINS):
        raise ValueError("development smoke selection must contain the four canonical bins")
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


def _saturate_crop_v2(crop: np.ndarray) -> tuple[np.ndarray, RadiometricSaturation]:
    clipped = crop > REFLECTANCE_SCALE
    saturation = RadiometricSaturation(
        raw_crop_minimum=int(crop.min()),
        raw_crop_maximum=int(crop.max()),
        clipped_high_count=int(np.count_nonzero(clipped)),
        clipped_high_by_band=tuple(
            int(np.count_nonzero(clipped[index])) for index in range(len(_BANDS))
        ),
    )
    return np.minimum(crop, int(REFLECTANCE_SCALE)), saturation


def _load_asset(
    storage_root: Path,
    record: Mapping[str, object],
    kind: str,
    normalization_policy: str,
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
        if dataset.nodata != RAW_NODATA:
            raise ValueError("asset nodata must equal the frozen uint16 sentinel 65535")
        pixels = dataset.read()
        if np.any(dataset.read_masks() == 0):
            raise ValueError("asset contains invalid nodata pixels")
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
        if minimum < 0.0:
            raise ValueError("asset raw reflectance must be in [0, 10000]")
        if (
            normalization_policy == LEGACY_NORMALIZATION_POLICY
            and maximum > REFLECTANCE_SCALE
        ):
            raise ValueError("asset raw reflectance must be in [0, 10000]")
        if (
            normalization_policy == PHASE2B3A_NORMALIZATION_POLICY
            and maximum > RAW_RADIOMETRIC_MAX
        ):
            raise ValueError("asset raw reflectance must be no greater than 32767")
        window = Window(1, 1, 128, 128) if kind == "lr" else Window(4, 4, 512, 512)
        row_start = int(window.row_off)
        column_start = int(window.col_off)
        crop = pixels[
            :,
            row_start : row_start + int(window.height),
            column_start : column_start + int(window.width),
        ]
        saturation = None
        if normalization_policy == PHASE2B3A_NORMALIZATION_POLICY:
            crop, saturation = _saturate_crop_v2(crop)
        crop_transform = _transform_tuple(transform(window, dataset.transform))
        crop_bounds = tuple(float(value) for value in bounds(window, dataset.transform))
    return _LoadedRaster(
        crop=crop,
        crop_transform=crop_transform,
        crop_bounds=crop_bounds,
        asset_sha256=digest,
        saturation=saturation,
    )


def _to_reflectance(value: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.array(value, copy=True)).to(torch.float32)
    return tensor.div_(REFLECTANCE_SCALE).contiguous()


def load_crosssensor_pair(
    storage_root: Path,
    record: Mapping[str, object],
    *,
    manifest_sha256: str,
    normalization_policy: str = LEGACY_NORMALIZATION_POLICY,
) -> LoadedCrosssensorPair:
    """Load, align and normalize one frozen Phase 2B1B pair."""

    if manifest_sha256 != POST_MANIFEST_SHA256:
        raise ValueError("expected the frozen post-manifest SHA-256")
    if type(normalization_policy) is not str or normalization_policy not in {
        LEGACY_NORMALIZATION_POLICY,
        PHASE2B3A_NORMALIZATION_POLICY,
    }:
        raise ValueError("unknown normalization policy")
    if not isinstance(storage_root, Path) or storage_root.is_symlink() or not storage_root.is_dir():
        raise ValueError("storage_root must be an existing non-symlink directory")
    resolved_root = storage_root.resolve(strict=True)
    if resolved_root != storage_root.absolute():
        raise ValueError("storage_root must not contain symlink path components")
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")

    lr = _load_asset(resolved_root, record, "lr", normalization_policy)
    hr = _load_asset(resolved_root, record, "hr", normalization_policy)
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
        normalization_policy=normalization_policy,
        lr_saturation=lr.saturation,
        hr_saturation=hr.saturation,
    )
    return LoadedCrosssensorPair(pair=pair, metadata=metadata)
