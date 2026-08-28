"""Canonical, reader-independent crosssensor manifests and audit summaries."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from trustsr.data.crosssensor_schema import (
    AcquisitionTimes,
    CrosssensorSample,
    validate_acquisition_times,
)
from trustsr.data.pilot_sampling import PilotChoice, correlation_bin, select_pilot
from trustsr.data.spatial_split import AssignedSample
from trustsr.jsonio import atomic_write_bytes, canonical_json

SAMPLE_SCHEMA = "trustsr.sen2naipv2-sample.v1"
AUDIT_SCHEMA = "trustsr.phase2b1a-audit.v1"
SOURCE_REVISION = "c370504201072fdb1dd388013ab8c0fc7d00a57e"
SOURCE_OBJECT_NAME = "sen2naipv2-crosssensor.taco"
SOURCE_OBJECT_SIZE_BYTES = 9_717_583_850
SOURCE_OBJECT_SHA256 = "c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5"
_SPLITS = ("development", "calibration", "internal_test")
_DISTANCE_KEYS = (
    "calibration:development",
    "calibration:internal_test",
    "development:internal_test",
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
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
        "admin",
        "days_between",
        "correlation",
        "scale_factor",
        "spatial_group_id",
        "split",
        "pilot",
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


@dataclass(frozen=True)
class ExpectedCounts:
    """Frozen expected sample and connected-component totals for one manifest."""

    samples: int
    components: int
    development_samples: int
    calibration_samples: int
    internal_test_samples: int
    development_components: int
    calibration_components: int
    internal_test_components: int


PRODUCTION_EXPECTED_COUNTS = ExpectedCounts(
    samples=8_000,
    components=6_695,
    development_samples=3_967,
    calibration_samples=2_070,
    internal_test_samples=1_963,
    development_components=3_317,
    calibration_components=1_719,
    internal_test_components=1_659,
)


@dataclass(frozen=True)
class ExtractedAsset:
    """Metadata for an extracted LR or HR GeoTIFF, without any pixel payload."""

    relative_path: str
    size_bytes: int
    sha256: str
    shape: tuple[int, int, int]
    dtype: str
    crs: str
    transform: tuple[float, float, float, float, float, float]
    nodata: float | int | None
    minimum: float
    maximum: float
    time_start: str


@dataclass(frozen=True)
class ManifestArtifact:
    """The path and content identity of a manifest written atomically."""

    path: Path
    size_bytes: int
    sha256: str


def _require_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_administrative_label(value: object, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be a string")
    return value


def _require_integer(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _require_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite float")
    return value


def _require_sha256(value: object, label: str) -> str:
    value = _require_string(value, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256")
    return value


def _require_mapping(value: object, label: str, fields: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} has invalid fields")
    return value


def _require_finite_float_sequence(value: object, label: str, length: int) -> tuple[float, ...]:
    if type(value) not in {tuple, list} or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} finite floats")
    return tuple(_require_float(item, label) for item in value)


def _require_positive_integer_sequence(value: object, label: str, length: int) -> tuple[int, ...]:
    if type(value) not in {tuple, list} or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} positive integers")
    return tuple(_require_integer(item, label, minimum=1) for item in value)


def _validate_relative_posix_path(value: object) -> str:
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


def _asset_record(asset: ExtractedAsset) -> dict[str, object]:
    if not isinstance(asset, ExtractedAsset):
        raise ValueError("asset must be an ExtractedAsset")
    record = asdict(asset)
    _validate_asset_record(record)
    return record


def _validate_asset_record(value: object) -> None:
    asset = _require_mapping(value, "asset", _ASSET_FIELDS)
    _validate_relative_posix_path(asset["relative_path"])
    _require_integer(asset["size_bytes"], "asset size_bytes", minimum=0)
    _require_sha256(asset["sha256"], "asset sha256")
    _require_positive_integer_sequence(asset["shape"], "asset shape", 3)
    _require_string(asset["dtype"], "asset dtype")
    _require_string(asset["crs"], "asset crs")
    _require_finite_float_sequence(asset["transform"], "asset transform", 6)
    nodata = asset["nodata"]
    if nodata is not None:
        if type(nodata) is int:
            pass
        else:
            _require_float(nodata, "asset nodata")
    _require_float(asset["minimum"], "asset minimum")
    _require_float(asset["maximum"], "asset maximum")
    _require_string(asset["time_start"], "asset time_start")


def _pilot_record(choice: PilotChoice | None) -> dict[str, object] | None:
    if choice is None:
        return None
    return {
        "days_between": choice.days_between,
        "correlation_bin": choice.correlation_bin,
        "selection_sha256": choice.selection_sha256,
    }


def _validate_pilot_record(
    value: object, sample_id: str, days_between: int, correlation: float
) -> None:
    if value is None:
        return
    pilot = _require_mapping(
        value,
        "pilot",
        frozenset({"days_between", "correlation_bin", "selection_sha256"}),
    )
    if _require_integer(pilot["days_between"], "pilot days_between") != days_between:
        raise ValueError("pilot days_between must match the sample")
    if _require_integer(pilot["correlation_bin"], "pilot correlation_bin") != correlation_bin(
        correlation
    ):
        raise ValueError("pilot correlation_bin must match the sample")
    expected_selection = hashlib.sha256(
        b"trustsr-pilot-v1\n" + sample_id.encode("utf-8")
    ).hexdigest()
    if _require_sha256(pilot["selection_sha256"], "pilot selection_sha256") != expected_selection:
        raise ValueError("pilot selection_sha256 must match the sample")


def _validate_manifest_record(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("manifest record must be an object")
    if value.get("schema") != SAMPLE_SCHEMA:
        raise ValueError("unknown manifest schema")
    record = _require_mapping(value, "manifest record", _MANIFEST_FIELDS)
    _require_mapping(
        record["source"],
        "source",
        frozenset({"revision", "object_sha256"}),
    )
    source = record["source"]
    if source["revision"] != SOURCE_REVISION or source["object_sha256"] != SOURCE_OBJECT_SHA256:
        raise ValueError("manifest source does not match the frozen crosssensor object")
    _require_integer(record["source_index"], "source_index", minimum=0)
    sample_id = _require_string(record["sample_id"], "sample_id")
    centroid = _require_mapping(
        record["centroid"], "centroid", frozenset({"longitude", "latitude"})
    )
    longitude = _require_float(centroid["longitude"], "centroid longitude")
    latitude = _require_float(centroid["latitude"], "centroid latitude")
    if not -180.0 <= longitude <= 180.0 or not -90.0 <= latitude <= 90.0:
        raise ValueError("manifest centroid is outside longitude/latitude bounds")
    _require_string(record["crs"], "crs")
    _require_finite_float_sequence(record["geotransform"], "geotransform", 6)
    _require_positive_integer_sequence(record["raster_shape"], "raster_shape", 2)
    admin = _require_mapping(record["admin"], "admin", frozenset({"admin0", "admin1", "admin2"}))
    for label, value in admin.items():
        if value is not None:
            _require_administrative_label(value, f"admin {label}")
    days_between = _require_integer(record["days_between"], "days_between")
    if days_between not in {-1, 0, 1}:
        raise ValueError("days_between must be one of {-1, 0, 1}")
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
    if _require_integer(record["scale_factor"], "scale_factor") != 4:
        raise ValueError("scale_factor must equal 4")
    _require_string(record["spatial_group_id"], "spatial_group_id")
    if record["split"] not in _SPLITS:
        raise ValueError("split is not a supported crosssensor split")
    _validate_pilot_record(record["pilot"], sample_id, days_between, correlation)
    lr_asset, hr_asset = record["lr_asset"], record["hr_asset"]
    if (lr_asset is None) != (hr_asset is None):
        raise ValueError("lr_asset and hr_asset must both be null or both be present")
    if lr_asset is not None:
        if record["pilot"] is None:
            raise ValueError("only a pilot record may contain extracted assets")
        _validate_asset_record(lr_asset)
        _validate_asset_record(hr_asset)
        if lr_asset["time_start"] != record["lr_time_start"]:
            raise ValueError("lr_asset time_start must equal lr_time_start")
        if hr_asset["time_start"] != record["hr_time_start"]:
            raise ValueError("hr_asset time_start must equal hr_time_start")
    return dict(record)


def _choices_by_sample_id(
    assignments_by_sample_id: Mapping[str, AssignedSample], choices: Sequence[PilotChoice]
) -> dict[str, PilotChoice]:
    expected_choices = select_pilot(tuple(assignments_by_sample_id.values()))
    if tuple(choices) != expected_choices:
        raise ValueError("pilot choices must equal the deterministic pilot choices")
    return {choice.sample_id: choice for choice in expected_choices}


def _assignment_from_manifest_record(record: Mapping[str, object]) -> AssignedSample:
    centroid = record["centroid"]
    admin = record["admin"]
    return AssignedSample(
        sample=CrosssensorSample(
            source_index=record["source_index"],  # type: ignore[arg-type]
            sample_id=record["sample_id"],  # type: ignore[arg-type]
            longitude=centroid["longitude"],  # type: ignore[index, arg-type]
            latitude=centroid["latitude"],  # type: ignore[index, arg-type]
            crs=record["crs"],  # type: ignore[arg-type]
            geotransform=tuple(record["geotransform"]),  # type: ignore[arg-type]
            raster_shape=tuple(record["raster_shape"]),  # type: ignore[arg-type]
            time_start=record["time_start"],  # type: ignore[arg-type]
            lr_time_start=record["lr_time_start"],  # type: ignore[arg-type]
            hr_time_start=record["hr_time_start"],  # type: ignore[arg-type]
            admin0=admin["admin0"],  # type: ignore[index, arg-type]
            admin1=admin["admin1"],  # type: ignore[index, arg-type]
            admin2=admin["admin2"],  # type: ignore[index, arg-type]
            days_between=record["days_between"],  # type: ignore[arg-type]
            correlation=record["correlation"],  # type: ignore[arg-type]
            scale_factor=record["scale_factor"],  # type: ignore[arg-type]
        ),
        spatial_group_id=record["spatial_group_id"],  # type: ignore[arg-type]
        split=record["split"],  # type: ignore[arg-type]
    )


def _validate_manifest_stream(records: Sequence[Mapping[str, object]]) -> None:
    expected_choices = select_pilot(
        tuple(_assignment_from_manifest_record(record) for record in records)
    )
    expected_pilots = {
        choice.sample_id: _pilot_record(choice)
        for choice in expected_choices
    }
    for record in records:
        if record["pilot"] != expected_pilots.get(record["sample_id"]):
            raise ValueError(
                "manifest pilot records do not equal the deterministic pilot selection"
            )
    asset_sample_ids = {
        record["sample_id"]
        for record in records
        if record["lr_asset"] is not None
    }
    if asset_sample_ids and asset_sample_ids != set(expected_pilots):
        raise ValueError("manifest extraction state must be all-null or contain every pilot pair")


def _manifest_record(
    assignment: AssignedSample,
    choice: PilotChoice | None,
    asset_pair: tuple[ExtractedAsset, ExtractedAsset] | None,
) -> dict[str, object]:
    sample = assignment.sample
    lr_asset: dict[str, object] | None = None
    hr_asset: dict[str, object] | None = None
    if asset_pair is not None:
        if type(asset_pair) is not tuple or len(asset_pair) != 2:
            raise ValueError("each selected sample must map to an LR/HR asset pair")
        lr_asset, hr_asset = (_asset_record(asset_pair[0]), _asset_record(asset_pair[1]))
    return {
        "schema": SAMPLE_SCHEMA,
        "source": {"revision": SOURCE_REVISION, "object_sha256": SOURCE_OBJECT_SHA256},
        "source_index": sample.source_index,
        "sample_id": sample.sample_id,
        "centroid": {"longitude": sample.longitude, "latitude": sample.latitude},
        "crs": sample.crs,
        "geotransform": list(sample.geotransform),
        "raster_shape": list(sample.raster_shape),
        "time_start": sample.time_start,
        "lr_time_start": sample.lr_time_start,
        "hr_time_start": sample.hr_time_start,
        "admin": {"admin0": sample.admin0, "admin1": sample.admin1, "admin2": sample.admin2},
        "days_between": sample.days_between,
        "correlation": sample.correlation,
        "scale_factor": sample.scale_factor,
        "spatial_group_id": assignment.spatial_group_id,
        "split": assignment.split,
        "pilot": _pilot_record(choice),
        "lr_asset": lr_asset,
        "hr_asset": hr_asset,
    }


def write_manifest(
    path: Path,
    assignments: Sequence[AssignedSample],
    choices: Sequence[PilotChoice],
    assets: Mapping[str, tuple[ExtractedAsset, ExtractedAsset]],
) -> ManifestArtifact:
    """Write one canonical JSONL record per assigned sample and return its content digest."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not isinstance(assets, Mapping):
        raise ValueError("assets must be a mapping")
    assignments_by_sample_id: dict[str, AssignedSample] = {}
    for assignment in assignments:
        if not isinstance(assignment, AssignedSample):
            raise ValueError("assignments must be AssignedSample records")
        sample_id = assignment.sample.sample_id
        if sample_id in assignments_by_sample_id:
            raise ValueError("manifest sample IDs must be unique")
        assignments_by_sample_id[sample_id] = assignment
    choices_by_sample_id = _choices_by_sample_id(assignments_by_sample_id, choices)
    asset_sample_ids = set(assets)
    if asset_sample_ids and asset_sample_ids != set(choices_by_sample_id):
        raise ValueError("assets must contain exactly one pair for every selected sample")
    if any(type(sample_id) is not str for sample_id in asset_sample_ids):
        raise ValueError("asset mapping keys must be sample IDs")
    records = tuple(
        _validate_manifest_record(
            _manifest_record(
                assignment,
                choices_by_sample_id.get(assignment.sample.sample_id),
                assets.get(assignment.sample.sample_id),
            )
        )
        for assignment in sorted(assignments, key=lambda assignment: assignment.sample.sample_id)
    )
    _validate_manifest_stream(records)
    payload = b"".join(canonical_json(record) + b"\n" for record in records)
    digest = hashlib.sha256(payload).hexdigest()
    atomic_write_bytes(path, payload)
    return ManifestArtifact(path=path, size_bytes=len(payload), sha256=digest)


def load_manifest(path: Path, *, expected_sha256: str) -> tuple[dict[str, object], ...]:
    """Verify and parse a canonical JSONL manifest without accepting alternate schemas."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    _require_sha256(expected_sha256, "expected manifest SHA-256")
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("manifest SHA-256 does not match the expected digest")
    if payload and not payload.endswith(b"\n"):
        raise ValueError("manifest JSONL records must end with one newline")
    records: list[dict[str, object]] = []
    for line in payload.splitlines(keepends=True):
        if not line.endswith(b"\n") or line == b"\n":
            raise ValueError("manifest JSONL records must end with one newline")
        try:
            decoded = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("manifest contains invalid JSON") from exc
        record = _validate_manifest_record(decoded)
        if canonical_json(record) + b"\n" != line:
            raise ValueError("manifest record is not canonical JSON")
        records.append(record)
    sample_ids = [record["sample_id"] for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("manifest sample IDs must be unique")
    if sample_ids != sorted(sample_ids):
        raise ValueError("manifest records must be sorted by sample_id")
    _validate_manifest_stream(records)
    return tuple(records)


def _validate_expected_counts(expected: ExpectedCounts) -> None:
    if not isinstance(expected, ExpectedCounts):
        raise ValueError("expected must be an ExpectedCounts record")
    for label, value in asdict(expected).items():
        _require_integer(value, f"expected {label}", minimum=0)


def _count_by_split(records: Sequence[Mapping[str, object]], field: str) -> dict[str, int]:
    return {
        split: sum(record["split"] == split for record in records)
        if field == "samples"
        else len({record["spatial_group_id"] for record in records if record["split"] == split})
        for split in _SPLITS
    }


def _validate_minimum_distances(minimum_distances: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(minimum_distances, Mapping) or set(minimum_distances) != set(_DISTANCE_KEYS):
        raise ValueError("minimum cross-split distances must contain every split pair")
    result: dict[str, float] = {}
    for key in _DISTANCE_KEYS:
        distance = _require_float(minimum_distances[key], f"minimum cross-split distance {key}")
        if distance <= 5.0:
            raise ValueError("minimum cross-split distance must be strictly greater than 5 km")
        result[key] = distance
    return result


def build_audit(
    records: Sequence[Mapping[str, object]],
    *,
    manifest_sha256: str,
    minimum_distances: Mapping[str, float],
    expected: ExpectedCounts,
) -> dict[str, object]:
    """Build a strict, host-free summary after validating the full manifest contract."""
    _require_sha256(manifest_sha256, "manifest SHA-256")
    _validate_expected_counts(expected)
    validated_records = tuple(_validate_manifest_record(record) for record in records)
    sample_ids = [record["sample_id"] for record in validated_records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("manifest sample IDs must be unique")
    _validate_manifest_stream(validated_records)
    groups_to_splits: dict[str, set[str]] = {}
    for record in validated_records:
        groups_to_splits.setdefault(str(record["spatial_group_id"]), set()).add(
            str(record["split"])
        )
    if any(len(splits) != 1 for splits in groups_to_splits.values()):
        raise ValueError("a shared spatial group crosses splits")
    split_sample_counts = _count_by_split(validated_records, "samples")
    split_component_counts = _count_by_split(validated_records, "components")
    actual_counts = ExpectedCounts(
        samples=len(validated_records),
        components=len(groups_to_splits),
        development_samples=split_sample_counts["development"],
        calibration_samples=split_sample_counts["calibration"],
        internal_test_samples=split_sample_counts["internal_test"],
        development_components=split_component_counts["development"],
        calibration_components=split_component_counts["calibration"],
        internal_test_components=split_component_counts["internal_test"],
    )
    if actual_counts != expected:
        raise ValueError(f"manifest counts do not match expected counts: {actual_counts}")
    distances = _validate_minimum_distances(minimum_distances)
    pilot_pair_count = sum(record["pilot"] is not None for record in validated_records)
    pilot_geotiff_count = 2 * sum(record["lr_asset"] is not None for record in validated_records)
    return {
        "schema": AUDIT_SCHEMA,
        "source_revision": SOURCE_REVISION,
        "source_object_name": SOURCE_OBJECT_NAME,
        "source_object_size_bytes": SOURCE_OBJECT_SIZE_BYTES,
        "source_object_sha256": SOURCE_OBJECT_SHA256,
        "manifest_sha256": manifest_sha256,
        "sample_count": actual_counts.samples,
        "component_count": actual_counts.components,
        "split_sample_counts": split_sample_counts,
        "split_component_counts": split_component_counts,
        "minimum_cross_split_distances": distances,
        "pilot_pair_count": pilot_pair_count,
        "pilot_geotiff_count": pilot_geotiff_count,
        "real_pixels_local": False,
        "gpu_used": False,
    }
