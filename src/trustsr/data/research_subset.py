"""Deterministic selection for the Phase 2B1B crosssensor research subset."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from trustsr.data.pilot_sampling import correlation_bin, select_pilot
from trustsr.data.spatial_split import AssignedSample

SPLITS = ("development", "calibration", "internal_test")
DAYS_BETWEEN = (-1, 0, 1)
CORRELATION_BINS = (0, 1, 2, 3)
SELECTION_ROUNDS = 10
SELECTION_PREFIX = b"trustsr-pilot-v1\n"


@dataclass(frozen=True)
class SubsetChoice:
    """One deterministic choice from a split, stratum, and selection round."""

    sample_id: str
    split: str
    days_between: int
    correlation_bin: int
    selection_round: int
    spatial_group_id: str
    selection_sha256: str


def selection_sha256(sample_id: str) -> str:
    """Return the frozen candidate-order digest for one sample identifier."""
    if type(sample_id) is not str or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    return hashlib.sha256(SELECTION_PREFIX + sample_id.encode("utf-8")).hexdigest()


def select_research_subset(
    assignments: Sequence[AssignedSample],
) -> tuple[SubsetChoice, ...]:
    """Choose ten rounds from every fixed stratum without reusing a split group."""
    choices: list[SubsetChoice] = []
    for split in SPLITS:
        used_groups: set[str] = set()
        strata = {
            (days_between, bin_index): sorted(
                (
                    assignment
                    for assignment in assignments
                    if assignment.split == split
                    and assignment.sample.days_between == days_between
                    and correlation_bin(assignment.sample.correlation) == bin_index
                ),
                key=lambda assignment: selection_sha256(assignment.sample.sample_id),
            )
            for days_between in DAYS_BETWEEN
            for bin_index in CORRELATION_BINS
        }
        for selection_round in range(1, SELECTION_ROUNDS + 1):
            for days_between in DAYS_BETWEEN:
                for bin_index in CORRELATION_BINS:
                    choice = next(
                        (
                            assignment
                            for assignment in strata[(days_between, bin_index)]
                            if assignment.spatial_group_id not in used_groups
                        ),
                        None,
                    )
                    if choice is None:
                        raise ValueError(
                            "research subset stratum cannot select a distinct spatial group: "
                            f"split={split}, selection_round={selection_round}, "
                            f"days_between={days_between}, correlation_bin={bin_index}"
                        )
                    used_groups.add(choice.spatial_group_id)
                    choices.append(
                        SubsetChoice(
                            sample_id=choice.sample.sample_id,
                            split=split,
                            days_between=days_between,
                            correlation_bin=bin_index,
                            selection_round=selection_round,
                            spatial_group_id=choice.spatial_group_id,
                            selection_sha256=selection_sha256(choice.sample.sample_id),
                        )
                    )

    result = tuple(
        sorted(
            choices,
            key=lambda choice: (
                choice.split,
                choice.selection_round,
                choice.days_between,
                choice.correlation_bin,
            ),
        )
    )
    if {choice.sample_id for choice in result if choice.selection_round == 1} != {
        choice.sample_id for choice in select_pilot(assignments)
    }:
        raise ValueError("research subset round one must equal the Phase 2B1A pilot")
    return result
