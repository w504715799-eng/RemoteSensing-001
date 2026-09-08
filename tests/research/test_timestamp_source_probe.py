import io
from email.message import Message

import pytest

from research.trustmask.timestamp_source_probe import candidate_projection, query_for, time_audit


def test_summer_and_winter_midnight_hypothesis():
    rows = [
        {
            "sample_id": "x_20220710",
            "lr_time_start": "2022-07-10T22:00:00Z",
            "hr_time_start": "2022-07-09T22:00:00Z",
        },
        {
            "sample_id": "x_20220110",
            "lr_time_start": "2022-01-09T23:00:00Z",
            "hr_time_start": "2022-01-09T23:00:00Z",
        },
    ]
    result = time_audit(rows)
    assert result["hr_filename_date_match_in_hypothesis_zone"] == 2
    assert result["local_midnight_counts"] == {"lr_time_start": 2, "hr_time_start": 2}


def test_window_covers_raw_and_next_date_without_quality_filter():
    query = query_for(
        {
            "lr_time_start": "2022-07-10T22:00:00Z",
            "centroid": {"longitude": "-123", "latitude": "39"},
        }
    )
    assert query["datetime"] == "2022-07-10T00:00:00Z/2022-07-11T23:59:59.999999Z"
    assert query["bbox"] == "-123.02,38.98,-122.98,39.02"
    assert "query" not in query
    assert "-assets" in query["fields"]


def test_projection_drops_assets_and_quality():
    body = {
        "type": "FeatureCollection",
        "features": [
            {
                "id": "candidate",
                "collection": "sentinel-2-l2a",
                "properties": {
                    "datetime": "2022-07-11T19:00:00Z",
                    "eo:cloud_cover": 42,
                    "s2:product_uri": "product",
                },
                "assets": {"red": {"href": "https://never-follow"}},
            }
        ],
        "links": [],
    }
    result = candidate_projection(body)
    assert result[0]["properties"]["s2:product_uri"] == "product"
    assert "assets" not in result[0]
    assert "eo:cloud_cover" not in result[0]["properties"]


@pytest.mark.parametrize(
    "body",
    [
        {"type": "FeatureCollection", "features": [], "links": [{"rel": "next"}]},
        {"type": "FeatureCollection", "features": [], "numberMatched": 1},
        {"type": "FeatureCollection", "features": [{}] * 51},
    ],
)
def test_incomplete_catalog_rejected(body):
    with pytest.raises(ValueError):
        candidate_projection(body)


def test_partial_mode_retains_candidates_without_claiming_completeness():
    body = {
        "type": "FeatureCollection",
        "features": [],
        "numberMatched": 1,
        "links": [{"rel": "next"}],
    }
    assert candidate_projection(body, require_complete=False) == []


@pytest.mark.parametrize(
    "size,encoding,status",
    [
        (1024 * 1024 + 1, "identity", 200),
        (2, "gzip", 200),
        (2, "identity", 206),
    ],
)
def test_transport_rejects_overflow_compression_and_wrong_status(
    monkeypatch, size, encoding, status
):
    from research.trustmask import timestamp_source_probe as probe

    response = io.BytesIO(b"x" * size)
    response.status = status
    response.headers = Message()
    response.headers["Content-Type"] = "application/json"
    response.headers["Content-Encoding"] = encoding

    class Opener:
        def open(self, request, timeout):
            assert request.full_url.startswith(probe.ENDPOINT + "?")
            assert timeout == 20
            return response

    monkeypatch.setattr(probe, "build_opener", lambda handler: Opener())
    with pytest.raises(ValueError):
        probe.fetch({"limit": "50"})


def test_redirects_are_not_followed():
    from research.trustmask.timestamp_source_probe import NoRedirect

    assert NoRedirect().redirect_request(None, None, 302, "", {}, "https://asset") is None
