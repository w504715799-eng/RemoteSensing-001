"""Deterministic stratified selection for the Phase 2B1A pilot."""

from __future__ import annotations

import bisect
import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

from trustsr.data.spatial_split import AssignedSample

CORRELATION_CUTS = (0.8842208864, 0.9041984739, 0.9265462586)
_SPLITS = ("development", "calibration", "internal_test")
_DAYS_BETWEEN = (-1, 0, 1)
_CORRELATION_BINS = (0, 1, 2, 3)
_SELECTION_PREFIX = b"trustsr-pilot-v1\n"


@dataclass(frozen=True)
class PilotChoice:
    """One deterministic pilot selection from a split, day, and correlation stratum."""

    sample_id: str
    split: str
    days_between: int
    correlation_bin: int
    spatial_group_id: str
    selection_sha256: str


def correlation_bin(value: float) -> int:
    """Return the fixed correlation bin, assigning each cut to the higher bin."""
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("correlation must be a finite float")
    return bisect.bisect_right(CORRELATION_CUTS, value)


def _selection_sha256(sample_id: str) -> str:
    return hashlib.sha256(_SELECTION_PREFIX + sample_id.encode("utf-8")).hexdigest()


def select_pilot(assignments: Sequence[AssignedSample]) -> tuple[PilotChoice, ...]:
    """Choose one candidate from every fixed pilot stratum without reusing a group."""
    choices: list[PilotChoice] = []
    for split in _SPLITS:
        used_groups: set[str] = set()
        for days_between in _DAYS_BETWEEN:
            for bin_index in _CORRELATION_BINS:
                candidates = sorted(
                    (
                        assignment
                        for assignment in assignments
                        if assignment.split == split
                        and assignment.sample.days_between == days_between
                        and correlation_bin(assignment.sample.correlation) == bin_index
                    ),
                    key=lambda assignment: _selection_sha256(assignment.sample.sample_id),
                )
                choice = next(
                    (
                        assignment
                        for assignment in candidates
                        if assignment.spatial_group_id not in used_groups
                    ),
                    None,
                )
                if choice is None:
                    raise ValueError(
                        "pilot stratum cannot select a distinct spatial group: "
                        f"split={split}, days_between={days_between}, correlation_bin={bin_index}"
                    )
                used_groups.add(choice.spatial_group_id)
                choices.append(
                    PilotChoice(
                        sample_id=choice.sample.sample_id,
                        split=split,
                        days_between=days_between,
                        correlation_bin=bin_index,
                        spatial_group_id=choice.spatial_group_id,
                        selection_sha256=_selection_sha256(choice.sample.sample_id),
                    )
                )
    return tuple(
        sorted(
            choices,
            key=lambda choice: (choice.split, choice.days_between, choice.correlation_bin),
        )
    )
