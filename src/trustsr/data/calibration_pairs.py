"""Strict calibration-only boundary before any crosssensor pixel loading."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from trustsr.contracts import SRPair
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
    RadiometricSaturation,
    load_crosssensor_pair,
)

CALIBRATION_SIZE = 120
_DAYS = (-1, 0, 1)
_BINS = (0, 1, 2, 3)
_ROUNDS = tuple(range(1, 11))
_IDENTITY_FIELDS = ("sample_id", "selection_sha256", "spatial_group_id")
type PairLoader = Callable[..., LoadedCrosssensorPair]


def _required_string(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"calibration record {field} must be a non-empty string")
    return value


def _required_integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int:
        raise ValueError(f"calibration record {field} must be an integer")
    return value


def _validate_record(record: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(record, Mapping):
        raise TypeError("calibration records must be mappings")
    if _required_string(record, "split") != "calibration":
        raise ValueError("calibration records must use only the calibration split")
    for field in _IDENTITY_FIELDS:
        _required_string(record, field)
    for kind in ("lr", "hr"):
        asset = record.get(f"{kind}_asset")
        if not isinstance(asset, Mapping) or not asset:
            raise ValueError(f"calibration record {kind}_asset must be a non-empty asset mapping")
    day = _required_integer(record, "days_between")
    bin_index = _required_integer(record, "correlation_bin")
    selection_round = _required_integer(record, "selection_round")
    if day not in _DAYS or bin_index not in _BINS or selection_round not in _ROUNDS:
        raise ValueError("calibration record has an invalid stratum or selection round")
    return record


def validate_calibration_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Validate exactly the frozen calibration 12-stratum by 10-round input."""

    if isinstance(records, str | bytes) or not isinstance(records, Sequence):
        raise TypeError("calibration records must be a stable sequence")
    if len(records) != CALIBRATION_SIZE:
        raise ValueError("calibration records require exactly 120 rows")
    values = tuple(_validate_record(record) for record in records)
    for field in _IDENTITY_FIELDS:
        identities = tuple(_required_string(record, field) for record in values)
        if len(set(identities)) != CALIBRATION_SIZE:
            raise ValueError(f"calibration records require unique {field} values")
    strata: dict[tuple[int, int], list[int]] = {
        (day, bin_index): [] for day in _DAYS for bin_index in _BINS
    }
    for record in values:
        day = _required_integer(record, "days_between")
        bin_index = _required_integer(record, "correlation_bin")
        strata[(day, bin_index)].append(_required_integer(record, "selection_round"))
    if any(tuple(sorted(rounds)) != _ROUNDS for rounds in strata.values()):
        raise ValueError("calibration records require 12 strata with rounds 1 through 10")
    return values


def _validate_loaded_pair(
    loaded: object, record: Mapping[str, object]
) -> LoadedCrosssensorPair:
    if not isinstance(loaded, LoadedCrosssensorPair):
        raise TypeError("calibration pair loader must return LoadedCrosssensorPair")
    if not isinstance(loaded.pair, SRPair) or not isinstance(
        loaded.metadata, CrosssensorPairMetadata
    ):
        raise TypeError("calibration pair loader returned forged pair state")
    loaded.pair.validate()
    metadata: CrosssensorPairMetadata = loaded.metadata
    sample_id = _required_string(record, "sample_id")
    if loaded.pair.sample_id != sample_id or metadata.sample_id != sample_id:
        raise ValueError("calibration pair loader output is out of input order")
    if loaded.pair.source != f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}":
        raise ValueError("calibration pair loader output has the wrong source")
    if (
        metadata.split != "calibration"
        or metadata.manifest_sha256 != POST_MANIFEST_SHA256
        or metadata.crop_policy != CROP_POLICY
        or metadata.normalization_policy != PHASE2B3A_NORMALIZATION_POLICY
    ):
        raise ValueError("calibration pair loader output has invalid split, manifest, or policy")
    if not isinstance(metadata.lr_saturation, RadiometricSaturation) or not isinstance(
        metadata.hr_saturation, RadiometricSaturation
    ):
        raise ValueError("calibration pair loader output requires saturation records")
    expected_metadata = {
        "spatial_group_id": _required_string(record, "spatial_group_id"),
        "days_between": _required_integer(record, "days_between"),
        "correlation_bin": _required_integer(record, "correlation_bin"),
        "selection_round": _required_integer(record, "selection_round"),
    }
    if any(getattr(metadata, field) != value for field, value in expected_metadata.items()):
        raise ValueError("calibration pair loader output metadata differs from the input record")
    return loaded


def load_calibration_pairs(
    storage_root: Path,
    records: Sequence[Mapping[str, object]],
    *,
    pair_loader: PairLoader | None = None,
) -> tuple[LoadedCrosssensorPair, ...]:
    """Validate all calibration metadata before crossing the pixel-loading boundary."""

    if not isinstance(storage_root, Path):
        raise TypeError("storage_root must be a pathlib.Path")
    if pair_loader is not None and not callable(pair_loader):
        raise TypeError("pair_loader must be callable")
    validated = validate_calibration_records(records)
    loader = load_crosssensor_pair if pair_loader is None else pair_loader
    loaded = tuple(
        _validate_loaded_pair(
            loader(
                storage_root,
                record,
                manifest_sha256=POST_MANIFEST_SHA256,
                normalization_policy=PHASE2B3A_NORMALIZATION_POLICY,
            ),
            record,
        )
        for record in validated
    )
    if len(loaded) != CALIBRATION_SIZE:
        raise RuntimeError("calibration pair loader returned the wrong number of pairs")
    if tuple(item.pair.sample_id for item in loaded) != tuple(
        _required_string(record, "sample_id") for record in validated
    ):
        raise ValueError("calibration pair loader output is not in input order")
    return loaded
