"""Tests for leak-safe spatial groups and deterministic split assignments."""

import math

import pytest

from trustsr.data.crosssensor_schema import CrosssensorSample
from trustsr.data.spatial_split import (
    EARTH_RADIUS_KM,
    AssignedSample,
    assign_spatial_splits,
    minimum_cross_split_distances,
)


def _sample(sample_id: str, longitude: float, latitude: float = 0.0) -> CrosssensorSample:
    return CrosssensorSample(
        source_index=0,
        sample_id=sample_id,
        longitude=longitude,
        latitude=latitude,
        crs="EPSG:4326",
        geotransform=(10.0, 0.0, 500000.0, 0.0, -10.0, 400000.0),
        raster_shape=(130, 130),
        time_start="2020-01-02T10:00:00Z",
        admin0="Colombia",
        admin1=None,
        admin2="Cali",
        days_between=0,
        correlation=0.91,
        scale_factor=4,
    )


def _groups_by_sample_id(assignments: tuple[AssignedSample, ...]) -> dict[str, str]:
    return {assignment.sample.sample_id: assignment.spatial_group_id for assignment in assignments}


def test_inclusive_boundary_and_transitive_component_membership() -> None:
    boundary_degrees = math.degrees(5.0 / EARTH_RADIUS_KM)
    samples = (
        _sample("boundary-a", 0.0),
        _sample("boundary-b", boundary_degrees),
        _sample("beyond", -math.degrees(5.01 / EARTH_RADIUS_KM)),
        _sample("chain-a", 20.0),
        _sample("chain-b", 20.0 + math.degrees(4.99 / EARTH_RADIUS_KM)),
        _sample("chain-c", 20.0 + math.degrees(9.98 / EARTH_RADIUS_KM)),
    )

    groups = _groups_by_sample_id(assign_spatial_splits(samples))

    assert groups["boundary-a"] == groups["boundary-b"]
    assert groups["boundary-a"] != groups["beyond"]
    assert groups["chain-a"] == groups["chain-b"] == groups["chain-c"]


def test_unit_sphere_search_keeps_dateline_neighbors_together() -> None:
    assignments = assign_spatial_splits(
        (_sample("dateline-east", 179.99), _sample("dateline-west", -179.99))
    )

    assert assignments[0].spatial_group_id == assignments[1].spatial_group_id


def test_high_latitude_haversine_boundary_is_never_a_cross_split_pair() -> None:
    samples = (
        _sample("a", -170.0, -88.24019444444444),
        _sample("b", -168.53672139438012, -88.24252237316091),
    )

    assignments = assign_spatial_splits(samples)
    separated_assignments = assign_spatial_splits(samples, threshold_km=4.0)
    separated_distances = minimum_cross_split_distances(separated_assignments)

    assert assignments[0].spatial_group_id == assignments[1].spatial_group_id
    assert minimum_cross_split_distances(assignments) == {}
    assert separated_distances == {
        "development:internal_test": pytest.approx(5.0, abs=1e-12)
    }
    assert all(distance <= 5.0 for distance in separated_distances.values())


def test_assignments_are_sorted_deterministic_and_have_one_split_per_group() -> None:
    samples = (
        _sample("zeta", 0.0),
        _sample("alpha", math.degrees(2.0 / EARTH_RADIUS_KM)),
        _sample("middle", 20.0),
    )

    first = assign_spatial_splits(samples)
    second = assign_spatial_splits(samples)
    splits_by_group: dict[str, set[str]] = {}
    for assignment in first:
        splits_by_group.setdefault(assignment.spatial_group_id, set()).add(assignment.split)

    assert [assignment.sample.sample_id for assignment in first] == ["alpha", "middle", "zeta"]
    assert first == second
    assert all(len(splits) == 1 for splits in splits_by_group.values())


def test_cross_split_minimum_distances_exceed_the_grouping_threshold() -> None:
    assignments = assign_spatial_splits(
        (
            _sample("split-3", 0.0),
            _sample("split-0", math.degrees(8.0 / EARTH_RADIUS_KM)),
            _sample("split-1", math.degrees(16.0 / EARTH_RADIUS_KM)),
        )
    )

    distances = minimum_cross_split_distances(assignments)

    assert distances == pytest.approx(
        {
            "calibration:development": 8.0,
            "calibration:internal_test": 8.0,
            "development:internal_test": 16.0,
        }
    )
    assert all(distance > 5.0 for distance in distances.values())
