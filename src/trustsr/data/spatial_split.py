"""Spatially leak-safe grouping and deterministic split assignment."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree

from trustsr.data.crosssensor_schema import CrosssensorSample

EARTH_RADIUS_KM = 6_371.0088
_SPLITS = ("development", "calibration", "internal_test")


@dataclass(frozen=True)
class AssignedSample:
    """A sample with its spatial connected-component group and split."""

    sample: CrosssensorSample
    spatial_group_id: str
    split: Literal["development", "calibration", "internal_test"]


def component_split(spatial_group_id: str) -> str:
    """Assign a reproducible split from the leading bits of a component hash."""
    value = int(spatial_group_id[:16], 16) / 2**64
    if value < 0.50:
        return "development"
    if value < 0.75:
        return "calibration"
    return "internal_test"


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, item: int) -> int:
        while self.parents[item] != item:
            self.parents[item] = self.parents[self.parents[item]]
            item = self.parents[item]
        return item

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parents[second_root] = first_root


def _unit_vectors(samples: Sequence[CrosssensorSample]) -> np.ndarray:
    latitudes = np.radians([sample.latitude for sample in samples])
    longitudes = np.radians([sample.longitude for sample in samples])
    return np.column_stack(
        (
            np.cos(latitudes) * np.cos(longitudes),
            np.cos(latitudes) * np.sin(longitudes),
            np.sin(latitudes),
        )
    )


def _haversine_km(first: CrosssensorSample, second: CrosssensorSample) -> float:
    latitude_delta = math.radians(second.latitude - first.latitude)
    longitude_delta = math.radians(second.longitude - first.longitude)
    first_latitude = math.radians(first.latitude)
    second_latitude = math.radians(second.latitude)
    haversine = math.sin(latitude_delta / 2) ** 2 + math.cos(first_latitude) * math.cos(
        second_latitude
    ) * math.sin(longitude_delta / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, max(0.0, haversine))))


def _component_id(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(sample_ids)).encode()).hexdigest()


def assign_spatial_splits(
    samples: Sequence[CrosssensorSample], threshold_km: float = 5.0
) -> tuple[AssignedSample, ...]:
    """Group samples connected within an inclusive geodesic threshold, then split groups."""
    if not math.isfinite(threshold_km) or threshold_km < 0:
        raise ValueError("threshold_km must be finite and non-negative")

    ordered_samples = tuple(sorted(samples, key=lambda sample: sample.sample_id))
    if not ordered_samples:
        return ()

    angular_threshold = threshold_km / EARTH_RADIUS_KM
    chord_threshold = 2 * math.sin(angular_threshold / 2)
    vectors = _unit_vectors(ordered_samples)
    components = _UnionFind(len(ordered_samples))
    candidate_pairs = cKDTree(vectors).query_pairs(chord_threshold, output_type="ndarray")
    for first_index, second_index in candidate_pairs:
        distance_km = _haversine_km(
            ordered_samples[first_index], ordered_samples[second_index]
        )
        if distance_km <= threshold_km:
            components.union(int(first_index), int(second_index))

    sample_ids_by_root: dict[int, list[str]] = {}
    for index, sample in enumerate(ordered_samples):
        sample_ids_by_root.setdefault(components.find(index), []).append(sample.sample_id)
    group_ids_by_root = {
        root: _component_id(sample_ids) for root, sample_ids in sample_ids_by_root.items()
    }

    return tuple(
        AssignedSample(
            sample=sample,
            spatial_group_id=group_ids_by_root[components.find(index)],
            split=component_split(group_ids_by_root[components.find(index)]),
        )
        for index, sample in enumerate(ordered_samples)
    )


def _chord_to_arc_km(chord: float) -> float:
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, max(0.0, chord / 2)))


def minimum_cross_split_distances(assignments: Sequence[AssignedSample]) -> dict[str, float]:
    """Return the nearest geodesic distance for each populated pair of splits."""
    samples_by_split: dict[str, list[CrosssensorSample]] = {split: [] for split in _SPLITS}
    for assignment in assignments:
        samples_by_split[assignment.split].append(assignment.sample)

    distances: dict[str, float] = {}
    for first_index, first_split in enumerate(_SPLITS):
        first_samples = samples_by_split[first_split]
        if not first_samples:
            continue
        first_tree = cKDTree(_unit_vectors(first_samples))
        for second_split in _SPLITS[first_index + 1 :]:
            second_samples = samples_by_split[second_split]
            if not second_samples:
                continue
            nearest_chords, _ = first_tree.query(_unit_vectors(second_samples), k=1)
            pair_key = ":".join(sorted((first_split, second_split)))
            distances[pair_key] = _chord_to_arc_km(
                float(np.min(nearest_chords))
            )
    return distances
