"""Tests for deterministic, group-distinct crosssensor pilot selection."""

from __future__ import annotations

import hashlib

import pytest

from trustsr.data.crosssensor_schema import CrosssensorSample
from trustsr.data.pilot_sampling import correlation_bin, select_pilot
from trustsr.data.spatial_split import AssignedSample


def _assignment(
    sample_id: str,
    split: str,
    days_between: int,
    correlation: float,
    group_id: str,
) -> AssignedSample:
    return AssignedSample(
        sample=CrosssensorSample(
            source_index=0,
            sample_id=sample_id,
            longitude=-76.5,
            latitude=3.5,
            crs="EPSG:32618",
            geotransform=(10.0, 0.0, 500000.0, 0.0, -10.0, 400000.0),
            raster_shape=(130, 130),
            time_start="2020-01-02T10:00:00Z",
            admin0="Colombia",
            admin1=None,
            admin2="Cali",
            days_between=days_between,
            correlation=correlation,
            scale_factor=4,
        ),
        spatial_group_id=group_id,
        split=split,  # type: ignore[arg-type]
    )


def _complete_assignments() -> tuple[AssignedSample, ...]:
    correlations = (0.8, 0.89, 0.91, 0.94)
    assignments: list[AssignedSample] = []
    for split in ("development", "calibration", "internal_test"):
        for days_between in (-1, 0, 1):
            for bin_index, correlation in enumerate(correlations):
                stratum = f"{split}-{days_between}-{bin_index}"
                assignments.extend(
                    (
                        _assignment(
                            f"{stratum}-first",
                            split,
                            days_between,
                            correlation,
                            f"{stratum}-group-first",
                        ),
                        _assignment(
                            f"{stratum}-second",
                            split,
                            days_between,
                            correlation,
                            f"{stratum}-group-second",
                        ),
                    )
                )
    return tuple(assignments)


def test_select_pilot_covers_every_stratum_with_distinct_groups_per_split() -> None:
    choices = select_pilot(_complete_assignments())

    assert len(choices) == 36
    assert {choice.split for choice in choices} == {
        "development",
        "calibration",
        "internal_test",
    }
    strata = {(choice.split, choice.days_between, choice.correlation_bin) for choice in choices}
    assert len(strata) == 36
    for split in ("development", "calibration", "internal_test"):
        selected = [choice for choice in choices if choice.split == split]
        assert len({choice.spatial_group_id for choice in selected}) == 12

    expected_first = min(
        ("development--1-0-first", "development--1-0-second"),
        key=lambda sample_id: hashlib.sha256(
            b"trustsr-pilot-v1\n" + sample_id.encode("utf-8")
        ).hexdigest(),
    )
    choice = next(
        choice
        for choice in choices
        if (choice.split, choice.days_between, choice.correlation_bin) == ("development", -1, 0)
    )
    assert choice.sample_id == expected_first
    assert choice.selection_sha256 == hashlib.sha256(
        b"trustsr-pilot-v1\n" + choice.sample_id.encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize(
    ("value", "expected_bin"),
    [
        (0.8842208864, 1),
        (0.9041984739, 2),
        (0.9265462586, 3),
    ],
)
def test_correlation_bin_places_cut_boundaries_in_higher_bin(
    value: float, expected_bin: int
) -> None:
    assert correlation_bin(value) == expected_bin


def test_select_pilot_rejects_a_later_stratum_that_can_only_reuse_a_group() -> None:
    assignments = list(_complete_assignments())
    duplicated_group = "development--1-0-group-first"
    assignments = [
        assignment
        for assignment in assignments
        if not (
            assignment.split == "development"
            and assignment.sample.days_between == -1
            and assignment.sample.correlation == 0.89
            and assignment.sample.sample_id.endswith("-second")
        )
    ]
    assignments = [
        AssignedSample(
            sample=assignment.sample,
            spatial_group_id=duplicated_group
            if (
                assignment.split == "development"
                and assignment.sample.days_between == -1
                and assignment.sample.correlation == 0.89
            )
            else assignment.spatial_group_id,
            split=assignment.split,
        )
        for assignment in assignments
    ]

    with pytest.raises(ValueError, match="distinct spatial group"):
        select_pilot(assignments)
