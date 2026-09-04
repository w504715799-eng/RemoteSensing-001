"""Metadata-only internal-test selection for Phase 2B3-C."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from trustsr.data.crosssensor_pairs import POST_MANIFEST_SHA256, load_crosssensor_records

_SPLITS = ("development", "calibration", "internal_test")
_DAYS = (-1, 0, 1)
_BINS = (0, 1, 2, 3)
_ROUNDS = tuple(range(1, 11))
_IDENTITIES = ("sample_id", "selection_sha256", "spatial_group_id")


def _string(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"post-manifest {field} must be a non-empty string")
    return value


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int:
        raise ValueError(f"post-manifest {field} must be a built-in integer")
    return value


def _digest(record: Mapping[str, object], field: str) -> str:
    value = _string(record, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"post-manifest {field} must be a lowercase SHA-256 digest")
    return value


def _validate_record(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("post-manifest record must be a mapping")
    if _string(value, "split") not in _SPLITS:
        raise ValueError("post-manifest record has an invalid split")
    _string(value, "sample_id")
    _digest(value, "selection_sha256")
    _digest(value, "spatial_group_id")
    _integer(value, "days_between")
    _integer(value, "correlation_bin")
    _integer(value, "selection_round")
    for kind in ("lr", "hr"):
        asset = value.get(f"{kind}_asset")
        if not isinstance(asset, Mapping) or not asset:
            raise ValueError(f"post-manifest record requires a non-empty {kind}_asset")
        digest = asset.get("sha256")
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"post-manifest {kind}_asset requires a lowercase SHA-256")
    return value


def select_internal_test_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Validate all 360 frozen metadata rows and select the 120 test rows."""

    if isinstance(records, str | bytes) or not isinstance(records, Sequence):
        raise TypeError("internal-test selection requires a stable record sequence")
    if len(records) != 360:
        raise ValueError("internal-test selection requires exactly 360 post-manifest records")
    validated = tuple(_validate_record(record) for record in records)
    sample_ids = [_string(record, "sample_id") for record in validated]
    if sample_ids != sorted(sample_ids):
        raise ValueError("post-manifest records must remain in canonical sample_id order")
    for field in _IDENTITIES:
        identities = [_string(record, field) for record in validated]
        if len(set(identities)) != len(identities):
            raise ValueError(f"post-manifest records require unique {field} values")
    if Counter(record["split"] for record in validated) != Counter(
        {split: 120 for split in _SPLITS}
    ):
        raise ValueError("post-manifest must contain exactly 120 records for each split")

    strata: dict[tuple[str, int, int], list[int]] = {
        (split, day, bin_index): []
        for split in _SPLITS
        for day in _DAYS
        for bin_index in _BINS
    }
    for record in validated:
        key = (
            _string(record, "split"),
            _integer(record, "days_between"),
            _integer(record, "correlation_bin"),
        )
        if key not in strata:
            raise ValueError("post-manifest contains an invalid stratum")
        strata[key].append(_integer(record, "selection_round"))
    if any(tuple(sorted(rounds)) != _ROUNDS for rounds in strata.values()):
        raise ValueError("post-manifest strata must contain selection rounds 1 through 10")
    return tuple(record for record in validated if record["split"] == "internal_test")


def load_internal_test_records(
    storage_root: Path, manifest_path: Path
) -> tuple[Mapping[str, object], ...]:
    """Load the digest-addressed manifest without opening any image asset."""

    return select_internal_test_records(
        load_crosssensor_records(
            storage_root,
            manifest_path,
            expected_sha256=POST_MANIFEST_SHA256,
        )
    )
