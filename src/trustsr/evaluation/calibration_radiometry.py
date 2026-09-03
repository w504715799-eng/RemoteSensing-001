"""Host-free Phase 2B3-B radiometry receipt built without reading tensor values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from trustsr.contracts import SRPair
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    RAW_RADIOMETRIC_MAX,
    REFLECTANCE_SCALE,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
    RadiometricSaturation,
)
from trustsr.data.subset_manifest import BANDS
from trustsr.evaluation.phase2b3b_preflight import ordered_sample_ids_sha256

_DAYS = (-1, 0, 1)
_BINS = (0, 1, 2, 3)
_ROUNDS = tuple(range(1, 11))
_SOURCE = f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}"


def _serialize_saturation(value: object) -> dict[str, object]:
    if type(value) is not RadiometricSaturation:
        raise TypeError("calibration pair requires an exact radiometric saturation record")
    RadiometricSaturation.__post_init__(value)
    return {
        "raw_crop_minimum": value.raw_crop_minimum,
        "raw_crop_maximum": value.raw_crop_maximum,
        "clipped_high_count": value.clipped_high_count,
        "clipped_high_by_band": list(value.clipped_high_by_band),
    }


def _serialize_pair(value: object) -> dict[str, object]:
    if type(value) is not LoadedCrosssensorPair:
        raise TypeError("calibration radiometry requires exact LoadedCrosssensorPair values")
    pair = value.pair
    metadata = value.metadata
    if type(pair) is not SRPair or type(metadata) is not CrosssensorPairMetadata:
        raise TypeError("calibration radiometry pair or metadata type is invalid")
    if type(pair.sample_id) is not str or not pair.sample_id:
        raise ValueError("calibration pair sample identity must be a non-empty string")
    if metadata.sample_id != pair.sample_id:
        raise ValueError("calibration pair and metadata sample identities differ")
    if metadata.split != "calibration":
        raise ValueError("calibration radiometry accepts the calibration split only")
    if metadata.manifest_sha256 != POST_MANIFEST_SHA256:
        raise ValueError("calibration pair has the wrong frozen manifest")
    if pair.source != _SOURCE:
        raise ValueError("calibration pair has the wrong source identity")
    if type(pair.scale) is not int or pair.scale != 4:
        raise ValueError("calibration pair must use scale 4")
    if (
        metadata.crop_policy != CROP_POLICY
        or metadata.normalization_policy != PHASE2B3A_NORMALIZATION_POLICY
    ):
        raise ValueError("calibration pair has the wrong crop or normalization policy")
    if (
        type(metadata.days_between) is not int
        or type(metadata.correlation_bin) is not int
        or type(metadata.selection_round) is not int
        or metadata.days_between not in _DAYS
        or metadata.correlation_bin not in _BINS
        or metadata.selection_round not in _ROUNDS
    ):
        raise ValueError("calibration pair has invalid stratum metadata")
    return {
        "sample_id": pair.sample_id,
        "days_between": metadata.days_between,
        "correlation_bin": metadata.correlation_bin,
        "selection_round": metadata.selection_round,
        "radiometric_saturation": {
            "lr": _serialize_saturation(metadata.lr_saturation),
            "hr": _serialize_saturation(metadata.hr_saturation),
        },
    }


def _asset_saturation(
    sample: Mapping[str, object], asset_name: str
) -> Mapping[str, object]:
    saturation = sample["radiometric_saturation"]
    if not isinstance(saturation, Mapping):
        raise AssertionError("serialized radiometric saturation must be a mapping")
    asset = saturation[asset_name]
    if not isinstance(asset, Mapping):
        raise AssertionError("serialized asset saturation must be a mapping")
    return asset


def _aggregate(
    samples: Sequence[Mapping[str, object]], asset_name: str
) -> dict[str, object]:
    records = [_asset_saturation(sample, asset_name) for sample in samples]
    band_totals = [
        sum(record["clipped_high_by_band"][index] for record in records)  # type: ignore[index]
        for index in range(len(BANDS))
    ]
    return {
        "raw_crop_minimum": min(record["raw_crop_minimum"] for record in records),
        "raw_crop_maximum": max(record["raw_crop_maximum"] for record in records),
        "clipped_high_count": sum(record["clipped_high_count"] for record in records),
        "clipped_high_by_band": band_totals,
    }


def build_calibration_radiometry(
    pairs: Sequence[LoadedCrosssensorPair],
) -> dict[str, object]:
    """Validate exact calibration metadata and derive a fresh JSON-native receipt."""

    if not isinstance(pairs, Sequence) or len(pairs) != 120:
        raise ValueError("calibration radiometry requires exactly 120 pairs")
    samples = [_serialize_pair(pair) for pair in pairs]
    sample_ids = [sample["sample_id"] for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("calibration radiometry requires unique ordered sample IDs")

    cells: dict[tuple[int, int], list[int]] = {
        (day, bin_index): [] for day in _DAYS for bin_index in _BINS
    }
    for sample in samples:
        key = (sample["days_between"], sample["correlation_bin"])
        if key not in cells:
            raise ValueError("calibration radiometry contains an invalid stratum")
        cells[key].append(sample["selection_round"])  # type: ignore[arg-type]
    if any(tuple(sorted(rounds)) != _ROUNDS for rounds in cells.values()):
        raise ValueError("calibration radiometry strata must each contain rounds 1 through 10")

    affected_sample_count = sum(
        int(
            any(
                _asset_saturation(sample, name)["clipped_high_count"] > 0  # type: ignore[operator]
                for name in ("lr", "hr")
            )
        )
        for sample in samples
    )
    return {
        "schema": "trustsr.phase2b3b-calibration-radiometry.v1",
        "split": "calibration",
        "ordered_sample_ids_sha256": ordered_sample_ids_sha256(sample_ids),
        "policy": {
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "raw_radiometric_max": RAW_RADIOMETRIC_MAX,
            "saturation_threshold": int(REFLECTANCE_SCALE),
            "saturation_operation": "minimum(raw,10000)",
            "saturation_scope": "aligned_crop_only",
            "reflectance_divisor": REFLECTANCE_SCALE,
            "crop_policy": CROP_POLICY,
            "bands": list(BANDS),
        },
        "sample_count": len(samples),
        "affected_sample_count": affected_sample_count,
        "lr": _aggregate(samples, "lr"),
        "hr": _aggregate(samples, "hr"),
        "samples": samples,
    }
