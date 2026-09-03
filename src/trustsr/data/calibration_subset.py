"""Metadata-only calibration selection for Phase 2B3-B."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from trustsr.data.crosssensor_pairs import POST_MANIFEST_SHA256, load_crosssensor_records

_SPLITS = ("development", "calibration", "internal_test")
_CALIBRATION_DAYS = (-1, 0, 1)
_CALIBRATION_BINS = (0, 1, 2, 3)
_CALIBRATION_ROUNDS = tuple(range(1, 11))


def _require_non_empty_string(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int:
        raise ValueError(f"{field} must be a built-in integer")
    return value


def _validate_record(record: object) -> Mapping[str, object]:
    if not isinstance(record, Mapping):
        raise ValueError("post-manifest record must be a mapping")
    split = _require_non_empty_string(record, "split")
    if split not in _SPLITS:
        raise ValueError("post-manifest record has an invalid split")
    for field in ("sample_id", "selection_sha256", "spatial_group_id"):
        _require_non_empty_string(record, field)
    for field in ("days_between", "correlation_bin", "selection_round"):
        _require_integer(record, field)
    for field in ("lr_asset", "hr_asset"):
        if not isinstance(record.get(field), Mapping) or not record[field]:
            raise ValueError(f"post-manifest record is missing {field}")
    return record


def _require_unique_identities(records: Sequence[Mapping[str, object]]) -> None:
    for field in ("sample_id", "selection_sha256", "spatial_group_id"):
        values = [_require_non_empty_string(record, field) for record in records]
        if len(set(values)) != len(values):
            raise ValueError(f"post-manifest records require unique {field} values")


def select_calibration_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Validate a frozen all-assets manifest and select its calibration rows."""

    if not isinstance(records, Sequence) or len(records) != 360:
        raise ValueError("calibration selection requires exactly 360 post-manifest records")
    validated = tuple(_validate_record(record) for record in records)
    split_counts = Counter(record["split"] for record in validated)
    if split_counts != Counter({split: 120 for split in _SPLITS}):
        raise ValueError("post-manifest must contain exactly 120 records for each split")
    _require_unique_identities(validated)

    selected = tuple(record for record in validated if record["split"] == "calibration")
    cells: dict[tuple[int, int], list[int]] = {
        (day, bin_index): []
        for day in _CALIBRATION_DAYS
        for bin_index in _CALIBRATION_BINS
    }
    for record in selected:
        day = _require_integer(record, "days_between")
        bin_index = _require_integer(record, "correlation_bin")
        selection_round = _require_integer(record, "selection_round")
        if (day, bin_index) not in cells:
            raise ValueError("calibration record has an invalid stratum")
        cells[(day, bin_index)].append(selection_round)
    if any(tuple(sorted(rounds)) != _CALIBRATION_ROUNDS for rounds in cells.values()):
        raise ValueError(
            "calibration strata must each contain selection rounds 1 through 10"
        )
    return selected


def load_calibration_records(
    storage_root: Path,
    manifest_path: Path,
) -> tuple[Mapping[str, object], ...]:
    """Load the fixed post-manifest, then select calibration metadata only."""

    return select_calibration_records(
        load_crosssensor_records(
            storage_root,
            manifest_path,
            expected_sha256=POST_MANIFEST_SHA256,
        )
    )
