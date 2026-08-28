"""Immutable, reader-independent metadata types for crosssensor samples."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

REQUIRED_COLUMNS = frozenset(
    {
        "tortilla:id",
        "stac:crs",
        "stac:geotransform",
        "stac:raster_shape",
        "stac:time_start",
        "stac:centroid",
        "rai:admin0",
        "rai:admin1",
        "rai:admin2",
        "days_between",
        "correlation",
        "scale_factor",
        "tortilla:data_split",
    }
)

_NUMBER = r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?"
_POINT_RE = re.compile(rf"POINT\s*\(\s*({_NUMBER})\s+({_NUMBER})\s*\)", re.IGNORECASE)
_RFC3339_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})"
)


@dataclass(frozen=True)
class AcquisitionTimes:
    """Source-ordered LR Sentinel-2 and HR NAIP acquisition timestamps."""

    lr_time_start: str
    hr_time_start: str


@dataclass(frozen=True)
class CrosssensorSample:
    """A validated row from the crosssensor top-level metadata table."""

    source_index: int
    sample_id: str
    longitude: float
    latitude: float
    crs: str
    geotransform: tuple[float, float, float, float, float, float]
    raster_shape: tuple[int, int]
    time_start: str
    lr_time_start: str
    hr_time_start: str
    admin0: str | None
    admin1: str | None
    admin2: str | None
    days_between: int
    correlation: float
    scale_factor: int


def _python_scalar(value: object) -> object:
    """Convert a NumPy scalar to its corresponding built-in Python scalar."""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return value


def _require_string(value: object, label: str, *, nonempty: bool = True) -> str:
    value = _python_scalar(value)
    if type(value) is not str or (nonempty and not value):
        suffix = " non-empty" if nonempty else ""
        raise ValueError(f"{label} must be a{suffix} string")
    return value


def _require_integer(value: object, label: str) -> int:
    value = _python_scalar(value)
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _require_float(value: object, label: str) -> float:
    value = _python_scalar(value)
    if type(value) is not float:
        raise ValueError(f"{label} must be a float")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _sequence_items(value: object, label: str, length: int) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"{label} must contain exactly {length} values")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{label} must contain exactly {length} values") from exc
    if len(items) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    return items


def _validate_geotransform(value: object) -> tuple[float, float, float, float, float, float]:
    items = _sequence_items(value, "stac:geotransform", 6)
    return tuple(_require_float(item, "stac:geotransform values") for item in items)  # type: ignore[return-value]


def _validate_raster_shape(value: object) -> tuple[int, int]:
    items = _sequence_items(value, "stac:raster_shape", 2)
    shape = tuple(_require_integer(item, "stac:raster_shape dimensions") for item in items)
    if shape != (130, 130):
        raise ValueError("stac:raster_shape must equal (130, 130)")
    return shape  # type: ignore[return-value]


def _validate_centroid(value: object) -> tuple[float, float]:
    value = _python_scalar(value)
    if type(value) is not str:
        raise ValueError("stac:centroid must be a complete POINT WKT")
    match = _POINT_RE.fullmatch(value)
    if match is None:
        raise ValueError("stac:centroid must be a complete POINT WKT")
    longitude = float(match.group(1))
    latitude = float(match.group(2))
    if not math.isfinite(longitude) or not -180.0 <= longitude <= 180.0:
        raise ValueError("stac:centroid longitude must be within [-180, 180]")
    if not math.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
        raise ValueError("stac:centroid latitude must be within [-90, 90]")
    return longitude, latitude


def _normalize_admin(value: object, label: str) -> str | None:
    if value is None:
        return None
    value = _python_scalar(value)
    if value is None:
        return None
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return _require_string(value, label, nonempty=False)


def _rfc3339_datetime(value: object, label: str) -> tuple[str, datetime]:
    value = _python_scalar(value)
    if type(value) is not str or _RFC3339_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a non-empty RFC 3339 timestamp with timezone")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise ValueError(
            f"{label} must be a non-empty RFC 3339 timestamp with timezone"
        ) from exc
    if parsed.utcoffset() is None:
        raise ValueError(f"{label} must be a non-empty RFC 3339 timestamp with timezone")
    return value, parsed


def validate_acquisition_times(
    time_start: object,
    acquisition_times: object,
    days_between: int,
    *,
    top_label: str = "stac:time_start",
) -> tuple[str, AcquisitionTimes]:
    """Validate top/LR/HR timestamps and the frozen signed UTC-date relation."""
    top_value, _ = _rfc3339_datetime(time_start, top_label)
    if not isinstance(acquisition_times, AcquisitionTimes):
        raise ValueError("acquisition time pair must be an AcquisitionTimes record")
    lr_value, lr_datetime = _rfc3339_datetime(
        acquisition_times.lr_time_start, "lr_time_start"
    )
    hr_value, hr_datetime = _rfc3339_datetime(
        acquisition_times.hr_time_start, "hr_time_start"
    )
    signed_days = (
        hr_datetime.astimezone(UTC).date()
        - lr_datetime.astimezone(UTC).date()
    ).days
    if signed_days != days_between:
        raise ValueError("HR minus LR UTC calendar dates must equal signed days_between")
    return top_value, AcquisitionTimes(lr_value, hr_value)


def _normalize_row(
    index: int, row: Mapping[str, object], acquisition_times: AcquisitionTimes
) -> CrosssensorSample:
    if not isinstance(row, Mapping):
        raise ValueError(f"row {index} must be a mapping")
    if row.get("tortilla:data_split") != "train":
        raise ValueError("upstream split must equal exact string train")
    missing = REQUIRED_COLUMNS - set(row)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    sample_id = _require_string(row["tortilla:id"], "tortilla:id")
    longitude, latitude = _validate_centroid(row["stac:centroid"])
    geotransform = _validate_geotransform(row["stac:geotransform"])
    raster_shape = _validate_raster_shape(row["stac:raster_shape"])
    days_between = _require_integer(row["days_between"], "days_between")
    if days_between not in {-1, 0, 1}:
        raise ValueError("days_between must be one of {-1, 0, 1}")
    scale_factor = _require_integer(row["scale_factor"], "scale_factor")
    if scale_factor != 4:
        raise ValueError("scale_factor must equal 4")
    time_start, validated_times = validate_acquisition_times(
        row["stac:time_start"], acquisition_times, days_between
    )

    return CrosssensorSample(
        source_index=index,
        sample_id=sample_id,
        longitude=longitude,
        latitude=latitude,
        crs=_require_string(row["stac:crs"], "stac:crs"),
        geotransform=geotransform,
        raster_shape=raster_shape,
        time_start=time_start,
        lr_time_start=validated_times.lr_time_start,
        hr_time_start=validated_times.hr_time_start,
        admin0=_normalize_admin(row["rai:admin0"], "rai:admin0"),
        admin1=_normalize_admin(row["rai:admin1"], "rai:admin1"),
        admin2=_normalize_admin(row["rai:admin2"], "rai:admin2"),
        days_between=days_between,
        correlation=_require_float(row["correlation"], "correlation"),
        scale_factor=scale_factor,
    )


def normalize_top_level(
    records: Sequence[Mapping[str, object]],
    *,
    acquisition_times: Sequence[AcquisitionTimes],
    expected_count: int = 8_000,
) -> tuple[CrosssensorSample, ...]:
    """Normalize and validate the required fields in a crosssensor table."""
    if len(acquisition_times) != len(records):
        raise ValueError("expected one acquisition time pair per crosssensor row")
    normalized = tuple(
        _normalize_row(index, row, acquisition_times[index])
        for index, row in enumerate(records)
    )
    if len(normalized) != expected_count:
        raise ValueError(f"expected {expected_count} crosssensor rows, observed {len(normalized)}")
    identifiers = [sample.sample_id for sample in normalized]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("tortilla:id values must be unique")
    return normalized
