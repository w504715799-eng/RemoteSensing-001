"""Independent in-memory verifier for Phase 2B3-B radiometry receipts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from trustsr.evaluation.phase2b3b_preflight import ordered_sample_ids_sha256
from trustsr.jsonio import canonical_json

_SCHEMA = "trustsr.phase2b3b-calibration-radiometry.v1"
_NORMALIZATION_POLICY = "uint16_saturate_10000_divide_10000_v2"
_CROP_POLICY = "center_crop_lr_1_hr_4_v1"
_RAW_RADIOMETRIC_MAX = 32_767
_SATURATION_THRESHOLD = 10_000
_BANDS = ("B04", "B03", "B02", "B08")
_DAYS = (-1, 0, 1)
_BINS = (0, 1, 2, 3)
_ROUNDS = tuple(range(1, 11))
_TOP_KEYS = {
    "schema",
    "split",
    "ordered_sample_ids_sha256",
    "policy",
    "sample_count",
    "affected_sample_count",
    "lr",
    "hr",
    "samples",
}
_POLICY_KEYS = {
    "normalization_policy",
    "raw_radiometric_max",
    "saturation_threshold",
    "saturation_operation",
    "saturation_scope",
    "reflectance_divisor",
    "crop_policy",
    "bands",
}
_SAMPLE_KEYS = {
    "sample_id",
    "days_between",
    "correlation_bin",
    "selection_round",
    "radiometric_saturation",
}
_SATURATION_KEYS = {
    "raw_crop_minimum",
    "raw_crop_maximum",
    "clipped_high_count",
    "clipped_high_by_band",
}


@dataclass(frozen=True)
class VerifiedCalibrationRadiometry:
    """Small immutable verification receipt without sample or host data."""

    source_sha256: str
    aggregate_sha256: str
    split: str
    ordered_sample_ids_sha256: str
    sample_count: int
    affected_sample_count: int
    aggregates: Mapping[str, Mapping[str, object]]


def _exact_dict(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} must be an exact parsed JSON object")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be a built-in integer")
    return value


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_policy(value: object) -> None:
    policy = _exact_dict(value, _POLICY_KEYS, "radiometric policy")
    if (
        policy["normalization_policy"] != _NORMALIZATION_POLICY
        or type(policy["normalization_policy"]) is not str
        or _integer(policy["raw_radiometric_max"], "radiometric policy raw maximum")
        != _RAW_RADIOMETRIC_MAX
        or _integer(policy["saturation_threshold"], "radiometric policy threshold")
        != _SATURATION_THRESHOLD
        or policy["saturation_operation"] != "minimum(raw,10000)"
        or type(policy["saturation_operation"]) is not str
        or policy["saturation_scope"] != "aligned_crop_only"
        or type(policy["saturation_scope"]) is not str
        or type(policy["reflectance_divisor"]) is not float
        or policy["reflectance_divisor"] != 10_000.0
        or policy["crop_policy"] != _CROP_POLICY
        or type(policy["crop_policy"]) is not str
        or type(policy["bands"]) is not list
        or policy["bands"] != list(_BANDS)
        or any(type(band) is not str for band in policy["bands"])
    ):
        raise ValueError("radiometric policy does not match the frozen contract")


def _validate_saturation(value: object) -> dict[str, object]:
    saturation = _exact_dict(value, _SATURATION_KEYS, "radiometric saturation")
    minimum = _integer(saturation["raw_crop_minimum"], "radiometric saturation minimum")
    maximum = _integer(saturation["raw_crop_maximum"], "radiometric saturation maximum")
    clipped = _integer(saturation["clipped_high_count"], "radiometric saturation count")
    by_band = saturation["clipped_high_by_band"]
    if (
        type(by_band) is not list
        or len(by_band) != len(_BANDS)
        or any(type(count) is not int for count in by_band)
    ):
        raise TypeError("radiometric saturation band counts must be a four-integer JSON list")
    if minimum < 0 or maximum < 0 or clipped < 0 or any(count < 0 for count in by_band):
        raise ValueError("radiometric saturation values must be non-negative")
    if minimum > maximum:
        raise ValueError("radiometric saturation minimum exceeds maximum")
    if maximum > _RAW_RADIOMETRIC_MAX:
        raise ValueError("radiometric saturation maximum exceeds 32767")
    if clipped != sum(by_band):
        raise ValueError("radiometric saturation count does not equal its band sum")
    if (maximum > _SATURATION_THRESHOLD) != (clipped > 0):
        raise ValueError("radiometric saturation maximum and clipped count are inconsistent")
    return {
        "raw_crop_minimum": minimum,
        "raw_crop_maximum": maximum,
        "clipped_high_count": clipped,
        "clipped_high_by_band": list(by_band),
    }


def _validate_sample(value: object) -> dict[str, object]:
    sample = _exact_dict(value, _SAMPLE_KEYS, "radiometric sample")
    sample_id = sample["sample_id"]
    day = _integer(sample["days_between"], "radiometric sample stratum day")
    bin_index = _integer(sample["correlation_bin"], "radiometric sample stratum bin")
    round_index = _integer(sample["selection_round"], "radiometric sample round")
    if type(sample_id) is not str or not sample_id:
        raise ValueError("radiometric sample identity must be a non-empty string")
    if day not in _DAYS or bin_index not in _BINS or round_index not in _ROUNDS:
        raise ValueError("radiometric sample stratum or round is invalid")
    saturation = _exact_dict(
        sample["radiometric_saturation"], {"lr", "hr"}, "sample radiometric saturation"
    )
    return {
        "sample_id": sample_id,
        "days_between": day,
        "correlation_bin": bin_index,
        "selection_round": round_index,
        "radiometric_saturation": {
            "lr": _validate_saturation(saturation["lr"]),
            "hr": _validate_saturation(saturation["hr"]),
        },
    }


def _asset(sample: Mapping[str, object], name: str) -> Mapping[str, object]:
    saturation = sample["radiometric_saturation"]
    if not isinstance(saturation, Mapping):
        raise AssertionError("validated sample saturation must be a mapping")
    asset = saturation[name]
    if not isinstance(asset, Mapping):
        raise AssertionError("validated asset saturation must be a mapping")
    return asset


def _aggregate(samples: Sequence[Mapping[str, object]], name: str) -> dict[str, object]:
    assets = [_asset(sample, name) for sample in samples]
    return {
        "raw_crop_minimum": min(asset["raw_crop_minimum"] for asset in assets),
        "raw_crop_maximum": max(asset["raw_crop_maximum"] for asset in assets),
        "clipped_high_count": sum(asset["clipped_high_count"] for asset in assets),
        "clipped_high_by_band": [
            sum(asset["clipped_high_by_band"][index] for asset in assets)  # type: ignore[index]
            for index in range(len(_BANDS))
        ],
    }


def _freeze_aggregate(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "raw_crop_minimum": value["raw_crop_minimum"],
            "raw_crop_maximum": value["raw_crop_maximum"],
            "clipped_high_count": value["clipped_high_count"],
            "clipped_high_by_band": tuple(value["clipped_high_by_band"]),  # type: ignore[arg-type]
        }
    )


def verify_calibration_radiometry(value: object) -> VerifiedCalibrationRadiometry:
    """Validate and independently re-aggregate a parsed JSON-native radiometry receipt."""

    receipt = _exact_dict(value, _TOP_KEYS, "radiometry receipt")
    if receipt["schema"] != _SCHEMA or type(receipt["schema"]) is not str:
        raise ValueError("radiometry receipt schema is invalid")
    if receipt["split"] != "calibration" or type(receipt["split"]) is not str:
        raise ValueError("radiometry receipt split is invalid")
    _validate_policy(receipt["policy"])
    sample_count = _integer(receipt["sample_count"], "radiometry receipt sample count")
    affected_count = _integer(
        receipt["affected_sample_count"], "radiometry receipt affected sample count"
    )
    if sample_count != 120:
        raise ValueError("radiometry receipt requires exactly 120 samples")
    samples_value = receipt["samples"]
    if type(samples_value) is not list or len(samples_value) != sample_count:
        raise ValueError("radiometry receipt requires exactly 120 JSON sample records")
    samples = [_validate_sample(sample) for sample in samples_value]
    sample_ids = [sample["sample_id"] for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("radiometry receipt requires unique sample identities")
    ordered_ids_digest = _digest(
        receipt["ordered_sample_ids_sha256"], "radiometry ordered sample IDs digest"
    )
    if ordered_ids_digest != ordered_sample_ids_sha256(sample_ids):
        raise ValueError("radiometry ordered sample IDs digest is inconsistent")
    cells: dict[tuple[int, int], list[int]] = {
        (day, bin_index): [] for day in _DAYS for bin_index in _BINS
    }
    for sample in samples:
        key = (sample["days_between"], sample["correlation_bin"])
        cells[key].append(sample["selection_round"])  # type: ignore[arg-type]
    if any(tuple(sorted(rounds)) != _ROUNDS for rounds in cells.values()):
        raise ValueError("radiometry receipt strata must each contain rounds 1 through 10")

    recomputed_affected = sum(
        int(any(_asset(sample, name)["clipped_high_count"] > 0 for name in ("lr", "hr")))  # type: ignore[operator]
        for sample in samples
    )
    if affected_count != recomputed_affected:
        raise ValueError("radiometry receipt affected sample aggregate is inconsistent")
    aggregates = {name: _aggregate(samples, name) for name in ("lr", "hr")}
    for name, aggregate in aggregates.items():
        try:
            claimed = _validate_saturation(receipt[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"radiometry receipt {name} aggregate is invalid") from exc
        if claimed != aggregate:
            raise ValueError(f"radiometry receipt {name} aggregate is inconsistent")

    aggregate_payload = {
        "sample_count": sample_count,
        "affected_sample_count": affected_count,
        "lr": aggregates["lr"],
        "hr": aggregates["hr"],
    }
    source_payload = canonical_json(receipt)
    aggregate_payload_bytes = canonical_json(aggregate_payload)
    return VerifiedCalibrationRadiometry(
        source_sha256=hashlib.sha256(source_payload).hexdigest(),
        aggregate_sha256=hashlib.sha256(aggregate_payload_bytes).hexdigest(),
        split="calibration",
        ordered_sample_ids_sha256=ordered_ids_digest,
        sample_count=sample_count,
        affected_sample_count=affected_count,
        aggregates=MappingProxyType(
            {
                "lr": _freeze_aggregate(aggregates["lr"]),
                "hr": _freeze_aggregate(aggregates["hr"]),
            }
        ),
    )
