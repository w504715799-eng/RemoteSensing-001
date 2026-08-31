"""Tests for the deterministic Phase 2B1B research-subset selector."""

from __future__ import annotations

from collections import Counter

import pytest

from trustsr.data.crosssensor_schema import CrosssensorSample
from trustsr.data.pilot_sampling import CORRELATION_CUTS, correlation_bin, select_pilot
from trustsr.data.research_subset import select_research_subset
from trustsr.data.spatial_split import AssignedSample


def _assignment(
    sample_id: str,
    split: str,
    days_between: int,
    correlation: float,
    group_id: str,
    source_index: int,
) -> AssignedSample:
    lr_time_start = {
        -1: "2020-01-03T10:00:00Z",
        0: "2020-01-02T10:00:00Z",
        1: "2020-01-01T10:00:00Z",
    }[days_between]
    return AssignedSample(
        sample=CrosssensorSample(
            source_index=source_index,
            sample_id=sample_id,
            longitude=-76.5,
            latitude=3.5,
            crs="EPSG:32618",
            geotransform=(10.0, 0.0, 500000.0, 0.0, -10.0, 400000.0),
            raster_shape=(130, 130),
            time_start="2020-01-02T10:00:00Z",
            lr_time_start=lr_time_start,
            hr_time_start="2020-01-02T10:00:00Z",
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


def _complete_assignments(candidates_per_stratum: int = 10) -> tuple[AssignedSample, ...]:
    correlations = (0.8, 0.89, 0.91, 0.94)
    assignments: list[AssignedSample] = []
    for split in ("development", "calibration", "internal_test"):
        for days_between in (-1, 0, 1):
            for bin_index, correlation in enumerate(correlations):
                for rank in range(candidates_per_stratum):
                    label = f"{split}-{days_between}-{bin_index}-{rank}"
                    assignments.append(
                        _assignment(
                            sample_id=label,
                            split=split,
                            days_between=days_between,
                            correlation=correlation,
                            group_id=f"group-{label}",
                            source_index=len(assignments),
                        )
                    )
    return tuple(assignments)


def test_research_subset_has_ten_rounds_per_stratum_and_unique_groups() -> None:
    choices = select_research_subset(_complete_assignments())

    assert len(choices) == 360
    assert Counter(choice.split for choice in choices) == {
        "development": 120,
        "calibration": 120,
        "internal_test": 120,
    }
    assert set(
        Counter(
            (choice.split, choice.days_between, choice.correlation_bin)
            for choice in choices
        ).values()
    ) == {10}
    assert Counter(choice.selection_round for choice in choices) == {
        selection_round: 36 for selection_round in range(1, 11)
    }
    for split in ("development", "calibration", "internal_test"):
        selected = [choice for choice in choices if choice.split == split]
        assert len({choice.spatial_group_id for choice in selected}) == 120


def test_round_one_exactly_preserves_phase2b1a_pilot() -> None:
    assignments = _complete_assignments()
    expected = {choice.sample_id for choice in select_pilot(assignments)}

    observed = {
        choice.sample_id
        for choice in select_research_subset(assignments)
        if choice.selection_round == 1
    }

    assert observed == expected


def test_selection_is_independent_of_input_order() -> None:
    assignments = _complete_assignments()

    assert select_research_subset(assignments) == select_research_subset(
        tuple(reversed(assignments))
    )


@pytest.mark.parametrize(
    ("cut", "expected_bin"), zip(CORRELATION_CUTS, (1, 2, 3), strict=True)
)
def test_exact_correlation_cuts_enter_the_higher_bin(
    cut: float, expected_bin: int
) -> None:
    assert correlation_bin(cut) == expected_bin


def test_selection_rejects_a_tenth_round_without_a_distinct_group() -> None:
    assignments = list(_complete_assignments())
    target = [
        assignment
        for assignment in assignments
        if assignment.split == "development"
        and assignment.sample.days_between == -1
        and assignment.sample.correlation == 0.8
    ]
    assignments.remove(target[-1])

    with pytest.raises(ValueError, match=r"selection_round=10.*correlation_bin=0"):
        select_research_subset(assignments)


def test_selection_skips_a_candidate_whose_group_was_used_by_an_earlier_stratum() -> None:
    assignments = list(_complete_assignments(candidates_per_stratum=11))
    first_stratum = [
        item
        for item in assignments
        if item.split == "development"
        and item.sample.days_between == -1
        and item.sample.correlation == 0.8
    ]
    second_stratum = [
        item
        for item in assignments
        if item.split == "development"
        and item.sample.days_between == -1
        and item.sample.correlation == 0.89
    ]
    pilot_first = next(
        choice
        for choice in select_pilot(assignments)
        if choice.split == "development"
        and choice.days_between == -1
        and choice.correlation_bin == 0
    )
    reused_group = next(
        item.spatial_group_id
        for item in first_stratum
        if item.sample.sample_id == pilot_first.sample_id
    )
    original = second_stratum[0]
    replacement = AssignedSample(
        sample=original.sample,
        spatial_group_id=reused_group,
        split=original.split,
    )
    assignments[assignments.index(original)] = replacement

    choices = select_research_subset(assignments)

    development_groups = [
        choice.spatial_group_id for choice in choices if choice.split == "development"
    ]
    assert len(development_groups) == len(set(development_groups)) == 120
