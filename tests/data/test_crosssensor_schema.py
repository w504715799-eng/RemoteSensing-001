"""Tests for strict normalization of crosssensor top-level metadata."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from trustsr.data.crosssensor_schema import (
    REQUIRED_COLUMNS,
    CrosssensorSample,
    normalize_top_level,
)


def _row(
    sample_id: str = "sample-1", longitude: float = -76.5, latitude: float = 3.5
) -> dict[str, object]:
    return {
        "tortilla:id": sample_id,
        "stac:crs": "EPSG:32618",
        "stac:geotransform": (10.0, 0.0, 500000.0, 0.0, -10.0, 400000.0),
        "stac:raster_shape": (130, 130),
        "stac:time_start": "2020-01-02T10:00:00Z",
        "stac:centroid": f"POINT ({longitude} {latitude})",
        "rai:admin0": "Colombia",
        "rai:admin1": None,
        "rai:admin2": "Cali",
        "days_between": 0,
        "correlation": 0.91,
        "scale_factor": 4,
    }


def test_normalizes_a_row_to_an_immutable_crosssensor_sample() -> None:
    assert REQUIRED_COLUMNS == frozenset(
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
    assert normalize_top_level([_row()], expected_count=1) == (
        CrosssensorSample(
            source_index=0,
            sample_id="sample-1",
            longitude=-76.5,
            latitude=3.5,
            crs="EPSG:32618",
            geotransform=(10.0, 0.0, 500000.0, 0.0, -10.0, 400000.0),
            raster_shape=(130, 130),
            time_start="2020-01-02T10:00:00Z",
            admin0="Colombia",
            admin1=None,
            admin2="Cali",
            days_between=0,
            correlation=0.91,
            scale_factor=4,
        ),
    )


def test_preserves_extra_metadata_and_normalizes_numpy_scalars() -> None:
    row = _row()
    row["extra:metadata"] = "preserved upstream"
    row["days_between"] = np.int64(0)
    row["scale_factor"] = np.int64(4)
    row["correlation"] = np.float64(0.91)
    row["stac:geotransform"] = tuple(np.float64(value) for value in row["stac:geotransform"])
    row["stac:raster_shape"] = tuple(np.int64(value) for value in row["stac:raster_shape"])

    normalized = normalize_top_level([row], expected_count=1)[0]

    assert normalized.days_between == 0
    assert type(normalized.days_between) is int
    assert normalized.scale_factor == 4
    assert type(normalized.scale_factor) is int
    assert normalized.correlation == 0.91
    assert type(normalized.correlation) is float


def test_normalized_sample_cannot_be_mutated() -> None:
    sample = normalize_top_level([_row()], expected_count=1)[0]

    with pytest.raises(FrozenInstanceError):
        sample.sample_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda rows: rows.append(_row(sample_id="sample-1")),
            "tortilla:id values must be unique",
        ),
        (
            lambda rows: rows[0].__setitem__("stac:centroid", "LINESTRING (-76.5 3.5, -76 4)"),
            "POINT",
        ),
        (lambda rows: rows[0].__setitem__("stac:centroid", "POINT (-181 3.5)"), "longitude"),
        (lambda rows: rows[0].__setitem__("stac:centroid", "POINT (-76.5 91)"), "latitude"),
        (lambda rows: rows[0].__setitem__("correlation", float("nan")), "correlation"),
        (lambda rows: rows[0].__setitem__("days_between", 2), "days_between"),
        (lambda rows: rows[0].__setitem__("scale_factor", 2), "scale_factor"),
        (lambda rows: rows[0].__setitem__("stac:raster_shape", (129, 130)), "raster_shape"),
    ],
)
def test_rejects_invalid_required_values(mutation: object, message: str) -> None:
    rows = [_row()]
    mutation(rows)

    with pytest.raises(ValueError, match=message):
        normalize_top_level(rows, expected_count=len(rows))


def test_rejects_missing_required_field() -> None:
    row = _row()
    del row["correlation"]

    with pytest.raises(ValueError, match="missing required columns.*correlation"):
        normalize_top_level([row], expected_count=1)


def test_rejects_wrong_input_length() -> None:
    with pytest.raises(ValueError, match="expected 8000 crosssensor rows, observed 1"):
        normalize_top_level([_row()])
