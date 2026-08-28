"""Immutable, reader-independent metadata types for crosssensor samples."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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
    }
)

_NUMBER = r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?"
_POINT_RE = re.compile(rf"POINT\s*\(\s*({_NUMBER})\s+({_NUMBER})\s*\)", re.IGNORECASE)


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
    # DataFrame adapters commonly represent a missing string as NaN.
    if type(value) is float and math.isnan(value):
        return None
    return _require_string(value, label, nonempty=False)


def _normalize_row(index: int, row: Mapping[str, object]) -> CrosssensorSample:
    if not isinstance(row, Mapping):
        raise ValueError(f"row {index} must be a mapping")
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

    return CrosssensorSample(
        source_index=index,
        sample_id=sample_id,
        longitude=longitude,
        latitude=latitude,
        crs=_require_string(row["stac:crs"], "stac:crs"),
        geotransform=geotransform,
        raster_shape=raster_shape,
        time_start=_require_string(row["stac:time_start"], "stac:time_start"),
        admin0=_normalize_admin(row["rai:admin0"], "rai:admin0"),
        admin1=_normalize_admin(row["rai:admin1"], "rai:admin1"),
        admin2=_normalize_admin(row["rai:admin2"], "rai:admin2"),
        days_between=days_between,
        correlation=_require_float(row["correlation"], "correlation"),
        scale_factor=scale_factor,
    )


def normalize_top_level(
    records: Sequence[Mapping[str, object]], *, expected_count: int = 8_000
) -> tuple[CrosssensorSample, ...]:
    """Normalize and validate the required fields in a crosssensor table."""
    normalized = tuple(_normalize_row(index, row) for index, row in enumerate(records))
    if len(normalized) != expected_count:
        raise ValueError(f"expected {expected_count} crosssensor rows, observed {len(normalized)}")
    identifiers = [sample.sample_id for sample in normalized]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("tortilla:id values must be unique")
    return normalized
