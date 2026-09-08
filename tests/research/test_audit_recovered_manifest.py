import copy

import pytest

from research.trustmask.audit_recovered_manifest import audit_rows


def fixture():
    row = {
        "sample_id": "a",
        "source_index": "0",
        "source": {
            "revision": "c370504201072fdb1dd388013ab8c0fc7d00a57e",
            "object_sha256": "c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5",
        },
        "centroid": {"longitude": "1.0", "latitude": "2.0"},
        "crs": "EPSG:32610",
        "geotransform": ["0", "2.5", "0", "100", "0", "-2.5"],
        "raster_shape": ["520", "520"],
        "time_start": "2020-01-02T22:00:00Z",
        "lr_time_start": "2020-01-02T22:00:00Z",
        "hr_time_start": "2020-01-01T23:00:00Z",
        "days_between": "-1",
    }
    return [row], [["a", 1.0, 2.0]], [["a", "EPSG:32610", [0, 2.5, 0, 100, 0, -2.5], [520, 520]]]


def test_signed_calendar_dates_not_elapsed_days():
    report = audit_rows(*fixture())
    assert report["top_equals_lr_count"] == 1
    assert report["signed_day_counts"] == {"-1": 1}


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_index", "1"),
        ("days_between", "0"),
        ("crs", "EPSG:4326"),
        ("lr_time_start", "2020-01-02T22:00:00"),
        ("centroid", {"longitude": "nan", "latitude": "2"}),
    ],
)
def test_corrupt_metadata_rejected(field, value):
    rows, members, geometry = fixture()
    rows[0][field] = value
    with pytest.raises(ValueError):
        audit_rows(rows, members, geometry)


def test_duplicate_identity_rejected():
    rows, members, geometry = fixture()
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError):
        audit_rows(rows, members, geometry)
