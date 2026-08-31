"""Canonical manifests for the Phase 2B1B crosssensor research subset."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from trustsr.data.crosssensor_manifest import (
    SOURCE_OBJECT_SHA256,
    SOURCE_REVISION,
    ExtractedAsset,
    ManifestArtifact,
)
from trustsr.data.crosssensor_schema import (
    AcquisitionTimes,
    CrosssensorSample,
    validate_acquisition_times,
)
from trustsr.data.pilot_sampling import correlation_bin
from trustsr.data.research_subset import (
    CORRELATION_BINS,
    DAYS_BETWEEN,
    SELECTION_ROUNDS,
    SPLITS,
    SubsetChoice,
    select_research_subset,
    selection_sha256,
)
from trustsr.data.spatial_split import AssignedSample
from trustsr.jsonio import atomic_write_bytes, canonical_json

SUBSET_SCHEMA = "trustsr.phase2b1b-selection.v1"
AUDIT_SCHEMA = "trustsr.phase2b1b-audit.v1"
BASE_MANIFEST_SHA256 = "7487b0af2ebef86910e918d5d6b2fb927a6f5e46bac7c2e30be7ffb2ce994482"
BANDS = ("B04", "B03", "B02", "B08")
_SUBSET_FIELDS = frozenset(
    {
        "schema",
        "base_manifest_sha256",
        "source",
        "source_index",
        "sample_id",
        "centroid",
        "crs",
        "geotransform",
        "raster_shape",
        "time_start",
        "lr_time_start",
        "hr_time_start",
        "days_between",
        "correlation",
        "correlation_bin",
        "scale_factor",
        "bands",
        "spatial_group_id",
        "split",
        "selection_round",
        "selection_sha256",
        "lr_asset",
        "hr_asset",
    }
)
_ASSET_FIELDS = frozenset(
    {
        "relative_path",
        "size_bytes",
        "sha256",
        "shape",
        "dtype",
        "crs",
        "transform",
        "nodata",
        "minimum",
        "maximum",
        "time_start",
    }
)


def _require_mapping(
    value: object, label: str, fields: frozenset[str]
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} has invalid fields")
    return value


def _require_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_integer(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        qualifier = f" no smaller than {minimum}" if minimum is not None else ""
        raise ValueError(f"{label} must be an integer{qualifier}")
    return value


def _require_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite float")
    return value


def _require_sha256(value: object, label: str) -> str:
    value = _require_string(value, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_float_sequence(value: object, label: str, length: int) -> tuple[float, ...]:
    if type(value) not in {tuple, list} or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} finite floats")
    return tuple(_require_float(item, label) for item in value)


def _require_integer_sequence(
    value: object, label: str, length: int
) -> tuple[int, ...]:
    if type(value) not in {tuple, list} or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} positive integers")
    return tuple(_require_integer(item, label, minimum=1) for item in value)


def _require_relative_path(value: object) -> str:
    value = _require_string(value, "asset relative_path")
    path = PurePosixPath(value)
    if (
        "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or path == PurePosixPath(".")
        or path.as_posix() != value
    ):
        raise ValueError("asset relative_path must be a canonical relative POSIX path")
    return value


def _validate_asset(value: object, *, kind: str, record: Mapping[str, object]) -> None:
    asset = _require_mapping(value, f"{kind} asset", _ASSET_FIELDS)
    expected_path = PurePosixPath(
        "subset-v1", str(record["split"]), str(record["sample_id"]), f"{kind}.tif"
    ).as_posix()
    if _require_relative_path(asset["relative_path"]) != expected_path:
        raise ValueError("asset relative_path must use the exact Phase 2B1B layout")
    _require_integer(asset["size_bytes"], "asset size_bytes", minimum=1)
    _require_sha256(asset["sha256"], "asset sha256")
    expected_shape = (4, 130, 130) if kind == "lr" else (4, 520, 520)
    if _require_integer_sequence(asset["shape"], "asset shape", 3) != expected_shape:
        raise ValueError(f"{kind} asset has an invalid shape")
    _require_string(asset["dtype"], "asset dtype")
    if _require_string(asset["crs"], "asset crs") != record["crs"]:
        raise ValueError("asset CRS must match the sidecar record")
    _require_float_sequence(asset["transform"], "asset transform", 6)
    nodata = asset["nodata"]
    if nodata is not None and type(nodata) is not int:
        _require_float(nodata, "asset nodata")
    minimum = _require_float(asset["minimum"], "asset minimum")
    maximum = _require_float(asset["maximum"], "asset maximum")
    if minimum > maximum:
        raise ValueError("asset minimum must not exceed its maximum")
    expected_time = record[f"{kind}_time_start"]
    if _require_string(asset["time_start"], "asset time_start") != expected_time:
        raise ValueError(f"{kind} asset time_start must match the sidecar record")


def _asset_record(asset: ExtractedAsset) -> dict[str, object]:
    if not isinstance(asset, ExtractedAsset):
        raise ValueError("asset must be an ExtractedAsset")
    return asdict(asset)


def _validate_record(value: object) -> dict[str, object]:
    record = _require_mapping(value, "selection record", _SUBSET_FIELDS)
    if record["schema"] != SUBSET_SCHEMA:
        raise ValueError("unknown Phase 2B1B selection schema")
    if record["base_manifest_sha256"] != BASE_MANIFEST_SHA256:
        raise ValueError("selection record has the wrong base manifest SHA-256")
    source = _require_mapping(
        record["source"], "source", frozenset({"revision", "object_sha256"})
    )
    if source["revision"] != SOURCE_REVISION or source["object_sha256"] != SOURCE_OBJECT_SHA256:
        raise ValueError("selection source does not match the frozen crosssensor object")
    _require_integer(record["source_index"], "source_index", minimum=0)
    sample_id = _require_string(record["sample_id"], "sample_id")
    centroid = _require_mapping(
        record["centroid"], "centroid", frozenset({"longitude", "latitude"})
    )
    longitude = _require_float(centroid["longitude"], "centroid longitude")
    latitude = _require_float(centroid["latitude"], "centroid latitude")
    if not -180.0 <= longitude <= 180.0 or not -90.0 <= latitude <= 90.0:
        raise ValueError("selection centroid is outside longitude/latitude bounds")
    _require_string(record["crs"], "crs")
    _require_float_sequence(record["geotransform"], "geotransform", 6)
    _require_integer_sequence(record["raster_shape"], "raster_shape", 2)
    days_between = _require_integer(record["days_between"], "days_between")
    if days_between not in DAYS_BETWEEN:
        raise ValueError("days_between is outside the fixed strata")
    validate_acquisition_times(
        record["time_start"],
        AcquisitionTimes(
            lr_time_start=record["lr_time_start"],  # type: ignore[arg-type]
            hr_time_start=record["hr_time_start"],  # type: ignore[arg-type]
        ),
        days_between,
        top_label="time_start",
    )
    correlation = _require_float(record["correlation"], "correlation")
    bin_index = _require_integer(record["correlation_bin"], "correlation_bin")
    if bin_index not in CORRELATION_BINS or bin_index != correlation_bin(correlation):
        raise ValueError("correlation_bin must match the fixed correlation cuts")
    if _require_integer(record["scale_factor"], "scale_factor") != 4:
        raise ValueError("scale_factor must equal 4")
    if type(record["bands"]) not in {tuple, list} or tuple(record["bands"]) != BANDS:
        raise ValueError(f"bands must equal {BANDS!r}")
    _require_sha256(record["spatial_group_id"], "spatial_group_id")
    if record["split"] not in SPLITS:
        raise ValueError("split is outside the fixed Phase 2B1B splits")
    selection_round = _require_integer(
        record["selection_round"], "selection_round", minimum=1
    )
    if selection_round > SELECTION_ROUNDS:
        raise ValueError("selection_round must be between 1 and 10")
    if _require_sha256(record["selection_sha256"], "selection_sha256") != selection_sha256(
        sample_id
    ):
        raise ValueError("selection_sha256 must match sample_id")
    lr_asset, hr_asset = record["lr_asset"], record["hr_asset"]
    if (lr_asset is None) != (hr_asset is None):
        raise ValueError("lr_asset and hr_asset must both be null or both be present")
    if lr_asset is not None:
        _validate_asset(lr_asset, kind="lr", record=record)
        _validate_asset(hr_asset, kind="hr", record=record)
    return dict(record)


def _validate_collection(records: Sequence[Mapping[str, object]]) -> None:
    if len(records) != 360:
        raise ValueError("selection manifest must contain exactly 360 records")
    sample_ids = [record["sample_id"] for record in records]
    if len(set(sample_ids)) != 360:
        raise ValueError("selection manifest must contain 360 unique sample IDs")
    if len({record["selection_sha256"] for record in records}) != 360:
        raise ValueError("selection manifest must contain 360 unique selection hashes")
    asset_states = {record["lr_asset"] is not None for record in records}
    if len(asset_states) != 1:
        raise ValueError("selection manifest assets must be uniformly all-null or all-present")
    split_counts = Counter(record["split"] for record in records)
    if split_counts != Counter({split: 120 for split in SPLITS}):
        raise ValueError("selection manifest must contain exactly 120 records per split")
    for split in SPLITS:
        groups = {
            record["spatial_group_id"] for record in records if record["split"] == split
        }
        if len(groups) != 120:
            raise ValueError("each split must contain 120 distinct spatial groups")
    strata = Counter(
        (record["split"], record["days_between"], record["correlation_bin"])
        for record in records
    )
    expected_strata = {
        (split, days_between, bin_index)
        for split in SPLITS
        for days_between in DAYS_BETWEEN
        for bin_index in CORRELATION_BINS
    }
    if set(strata) != expected_strata or set(strata.values()) != {10}:
        raise ValueError("every Phase 2B1B stratum must contain exactly ten records")
    rounds = Counter(record["selection_round"] for record in records)
    if rounds != Counter({selection_round: 36 for selection_round in range(1, 11)}):
        raise ValueError("every selection round must contain exactly 36 records")
    round_strata = Counter(
        (
            record["split"],
            record["selection_round"],
            record["days_between"],
            record["correlation_bin"],
        )
        for record in records
    )
    if len(round_strata) != 360 or set(round_strata.values()) != {1}:
        raise ValueError("every round must contain one record from every stratum")


def _assignment_from_base_record(record: Mapping[str, object]) -> AssignedSample:
    centroid = record["centroid"]
    admin = record["admin"]
    if not isinstance(centroid, Mapping) or not isinstance(admin, Mapping):
        raise ValueError("base manifest record has invalid nested metadata")
    return AssignedSample(
        sample=CrosssensorSample(
            source_index=record["source_index"],  # type: ignore[arg-type]
            sample_id=record["sample_id"],  # type: ignore[arg-type]
            longitude=centroid["longitude"],  # type: ignore[arg-type]
            latitude=centroid["latitude"],  # type: ignore[arg-type]
            crs=record["crs"],  # type: ignore[arg-type]
            geotransform=tuple(record["geotransform"]),  # type: ignore[arg-type]
            raster_shape=tuple(record["raster_shape"]),  # type: ignore[arg-type]
            time_start=record["time_start"],  # type: ignore[arg-type]
            lr_time_start=record["lr_time_start"],  # type: ignore[arg-type]
            hr_time_start=record["hr_time_start"],  # type: ignore[arg-type]
            admin0=admin["admin0"],  # type: ignore[arg-type]
            admin1=admin["admin1"],  # type: ignore[arg-type]
            admin2=admin["admin2"],  # type: ignore[arg-type]
            days_between=record["days_between"],  # type: ignore[arg-type]
            correlation=record["correlation"],  # type: ignore[arg-type]
            scale_factor=record["scale_factor"],  # type: ignore[arg-type]
        ),
        spatial_group_id=record["spatial_group_id"],  # type: ignore[arg-type]
        split=record["split"],  # type: ignore[arg-type]
    )


def _base_by_id(
    base_records: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for record in base_records:
        if not isinstance(record, Mapping):
            raise ValueError("base manifest records must be mappings")
        sample_id = _require_string(record.get("sample_id"), "base sample_id")
        if sample_id in result:
            raise ValueError("base manifest sample IDs must be unique")
        result[sample_id] = record
    return result


def select_from_base_manifest(
    base_records: Sequence[Mapping[str, object]],
) -> tuple[SubsetChoice, ...]:
    """Select the fixed research subset directly from Phase 2B1A manifest records."""
    _base_by_id(base_records)
    return select_research_subset(
        tuple(_assignment_from_base_record(record) for record in base_records)
    )


def _require_record_matches_base(
    record: Mapping[str, object], base: Mapping[str, object]
) -> None:
    copied_fields = (
        "source",
        "source_index",
        "sample_id",
        "centroid",
        "crs",
        "geotransform",
        "raster_shape",
        "time_start",
        "lr_time_start",
        "hr_time_start",
        "days_between",
        "correlation",
        "scale_factor",
        "spatial_group_id",
        "split",
    )
    if any(record[field] != base[field] for field in copied_fields):
        raise ValueError("selection record does not match its base manifest record")


def _require_record_matches_choice(
    record: Mapping[str, object], choice: SubsetChoice
) -> None:
    expected = {
        "sample_id": choice.sample_id,
        "split": choice.split,
        "days_between": choice.days_between,
        "correlation_bin": choice.correlation_bin,
        "selection_round": choice.selection_round,
        "spatial_group_id": choice.spatial_group_id,
        "selection_sha256": choice.selection_sha256,
    }
    if any(record[field] != value for field, value in expected.items()):
        raise ValueError("selection record does not match its deterministic choice")


def validate_subset_against_base(
    records: Sequence[Mapping[str, object]],
    base_records: Sequence[Mapping[str, object]],
) -> tuple[SubsetChoice, ...]:
    """Require sidecar records to equal the deterministic selection from the base."""
    validated = tuple(_validate_record(record) for record in records)
    _validate_collection(validated)
    base_by_id = _base_by_id(base_records)
    expected = select_from_base_manifest(base_records)
    expected_by_id = {choice.sample_id: choice for choice in expected}
    if {record["sample_id"] for record in validated} != set(expected_by_id):
        raise ValueError("sidecar does not equal the deterministic research subset")
    for record in validated:
        sample_id = str(record["sample_id"])
        _require_record_matches_base(record, base_by_id[sample_id])
        _require_record_matches_choice(record, expected_by_id[sample_id])
    return expected


def _subset_record(
    base: Mapping[str, object],
    choice: SubsetChoice,
    asset_pair: tuple[ExtractedAsset, ExtractedAsset] | None,
) -> dict[str, object]:
    lr_asset: dict[str, object] | None = None
    hr_asset: dict[str, object] | None = None
    if asset_pair is not None:
        if type(asset_pair) is not tuple or len(asset_pair) != 2:
            raise ValueError("each selected sample must map to one LR/HR asset pair")
        lr_asset, hr_asset = _asset_record(asset_pair[0]), _asset_record(asset_pair[1])
    return {
        "schema": SUBSET_SCHEMA,
        "base_manifest_sha256": BASE_MANIFEST_SHA256,
        "source": base["source"],
        "source_index": base["source_index"],
        "sample_id": base["sample_id"],
        "centroid": base["centroid"],
        "crs": base["crs"],
        "geotransform": base["geotransform"],
        "raster_shape": base["raster_shape"],
        "time_start": base["time_start"],
        "lr_time_start": base["lr_time_start"],
        "hr_time_start": base["hr_time_start"],
        "days_between": base["days_between"],
        "correlation": base["correlation"],
        "correlation_bin": choice.correlation_bin,
        "scale_factor": base["scale_factor"],
        "bands": list(BANDS),
        "spatial_group_id": base["spatial_group_id"],
        "split": base["split"],
        "selection_round": choice.selection_round,
        "selection_sha256": choice.selection_sha256,
        "lr_asset": lr_asset,
        "hr_asset": hr_asset,
    }


def write_subset_manifest(
    path: Path,
    base_records: Sequence[Mapping[str, object]],
    choices: Sequence[SubsetChoice],
    assets: Mapping[str, tuple[ExtractedAsset, ExtractedAsset]],
) -> ManifestArtifact:
    """Write one canonical record per selected sample and return its content identity."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if not isinstance(assets, Mapping):
        raise ValueError("assets must be a mapping")
    base_by_id = _base_by_id(base_records)
    expected = select_from_base_manifest(base_records)
    if tuple(choices) != expected:
        raise ValueError("choices do not equal the deterministic research subset")
    selected_ids = {choice.sample_id for choice in expected}
    asset_ids = set(assets)
    if any(type(sample_id) is not str for sample_id in asset_ids):
        raise ValueError("asset mapping keys must be sample IDs")
    if asset_ids not in (set(), selected_ids):
        raise ValueError("assets must contain exactly every selected sample or be empty")
    choice_by_id = {choice.sample_id: choice for choice in expected}
    records = tuple(
        _validate_record(
            _subset_record(
                base_by_id[sample_id],
                choice_by_id[sample_id],
                assets.get(sample_id),
            )
        )
        for sample_id in sorted(selected_ids)
    )
    validate_subset_against_base(records, base_records)
    payload = b"".join(canonical_json(record) + b"\n" for record in records)
    digest = hashlib.sha256(payload).hexdigest()
    atomic_write_bytes(path, payload)
    return ManifestArtifact(path=path, size_bytes=len(payload), sha256=digest)


def load_subset_manifest(
    path: Path, *, expected_sha256: str
) -> tuple[dict[str, object], ...]:
    """Verify and parse one canonical, complete Phase 2B1B sidecar."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    _require_sha256(expected_sha256, "expected selection manifest SHA-256")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("selection manifest SHA-256 does not match")
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("selection manifest must end with one newline")
    records: list[dict[str, object]] = []
    for line in payload.splitlines(keepends=True):
        if not line.endswith(b"\n") or line == b"\n":
            raise ValueError("selection manifest contains an invalid JSONL line")
        try:
            decoded = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("selection manifest contains invalid JSON") from exc
        record = _validate_record(decoded)
        if canonical_json(record) + b"\n" != line:
            raise ValueError("selection manifest record is not canonical JSON")
        records.append(record)
    sample_ids = [record["sample_id"] for record in records]
    if sample_ids != sorted(sample_ids):
        raise ValueError("selection manifest must be sorted by sample_id")
    _validate_collection(records)
    return tuple(records)
