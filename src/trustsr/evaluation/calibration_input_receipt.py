"""Frozen-membership receipt for loaded Phase 2B3-B calibration inputs."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from trustsr.artifacts.predictions import tensor_sha256
from trustsr.contracts import SRPair
from trustsr.data.calibration_pairs import validate_calibration_records
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
)
from trustsr.evaluation.phase2b3b_preflight import ordered_sample_ids_sha256
from trustsr.jsonio import canonical_json

SCHEMA = "trustsr.phase2b3b-calibration-input-receipt.v1"
CALIBRATION_SIZE = 120
_SOURCE = f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}"
_BANDS = ("B04", "B03", "B02", "B08")
_DAYS = (-1, 0, 1)
_BINS = (0, 1, 2, 3)
_ROUNDS = tuple(range(1, 11))
_PREFLIGHT_KEYS = {"schema", "upstream", "calibration", "score", "risk", "input"}
_CALIBRATION_KEYS = {
    "split",
    "sample_count",
    "ordered_sample_ids_sha256",
    "ordered_membership_sha256",
    "input_receipt_sha256s",
    "strata",
}
_INPUT_KEYS = {
    "manifest_sha256",
    "source",
    "normalization_policy",
    "crop_policy",
    "bands",
    "scale",
}
_RECEIPT_KEYS = {
    "schema",
    "split",
    "sample_count",
    "input",
    "ordered_sample_ids_sha256",
    "ordered_membership_sha256",
    "input_receipt_sha256s",
    "samples",
    "ordered_inputs_sha256",
}
_SAMPLE_KEYS = {"membership", "lr", "hr"}
_MEMBERSHIP_KEYS = {
    "sample_id",
    "selection_sha256",
    "spatial_group_id",
    "lr_asset_sha256",
    "hr_asset_sha256",
    "days_between",
    "correlation_bin",
    "selection_round",
}
_ASSET_KEYS = {"asset_sha256", "tensor_sha256", "shape", "dtype"}


@dataclass(frozen=True)
class VerifiedCalibrationInputReceipt:
    """Host-free proof that one parsed input receipt is internally consistent."""

    source_sha256: str
    ordered_inputs_sha256: str
    ordered_sample_ids_sha256: str
    ordered_membership_sha256: str
    sample_count: int


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _exact(actual: object, expected: object) -> bool:
    return type(actual) is type(expected) and actual == expected


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _mapping(value: object, keys: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys are invalid")
    return value


def _json_mapping(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} must be an exact parsed JSON object")
    return value


def _sequence(value: object, *, length: int, label: str) -> Sequence[object]:
    if (
        isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
        or len(value) != length
    ):
        raise ValueError(f"{label} must contain exactly {length} items")
    return value


def _json_list(value: object, *, length: int, label: str) -> list[object]:
    if type(value) is not list or len(value) != length:
        raise ValueError(f"{label} must be an exact {length}-item JSON array")
    return value


def _membership(record: Mapping[str, object]) -> dict[str, object]:
    lr_asset = record["lr_asset"]
    hr_asset = record["hr_asset"]
    if not isinstance(lr_asset, Mapping) or not isinstance(hr_asset, Mapping):
        raise ValueError("calibration membership requires LR and HR asset mappings")
    return {
        "sample_id": record["sample_id"],
        "selection_sha256": record["selection_sha256"],
        "spatial_group_id": record["spatial_group_id"],
        "lr_asset_sha256": lr_asset["sha256"],
        "hr_asset_sha256": hr_asset["sha256"],
        "days_between": record["days_between"],
        "correlation_bin": record["correlation_bin"],
        "selection_round": record["selection_round"],
    }


def _validate_preflight(
    preflight: object, membership: Sequence[Mapping[str, object]]
) -> tuple[str, str, list[str]]:
    value = _mapping(preflight, _PREFLIGHT_KEYS, "Phase 2B3-B preflight")
    if value["schema"] != "trustsr.phase2b3b-preflight.v1":
        raise ValueError("Phase 2B3-B preflight schema is invalid")
    upstream = value["upstream"]
    if (
        not isinstance(upstream, Mapping)
        or upstream.get("post_manifest_sha256") != POST_MANIFEST_SHA256
    ):
        raise ValueError("Phase 2B3-B preflight manifest authority is invalid")
    input_identity = _mapping(value["input"], {
        "normalization_policy", "crop_policy", "bands", "scale"
    }, "Phase 2B3-B preflight input")
    bands = _sequence(input_identity["bands"], length=4, label="preflight bands")
    if (
        input_identity["normalization_policy"] != PHASE2B3A_NORMALIZATION_POLICY
        or input_identity["crop_policy"] != CROP_POLICY
        or tuple(bands) != _BANDS
        or type(input_identity["scale"]) is not int
        or input_identity["scale"] != 4
    ):
        raise ValueError("Phase 2B3-B preflight input identity is invalid")
    calibration = _mapping(
        value["calibration"], _CALIBRATION_KEYS, "preflight calibration"
    )
    if (
        not _exact(calibration["split"], "calibration")
        or type(calibration["sample_count"]) is not int
        or calibration["sample_count"] != CALIBRATION_SIZE
    ):
        raise ValueError("preflight calibration split or count is invalid")
    sample_ids = [str(record["sample_id"]) for record in membership]
    ordered_ids_digest = ordered_sample_ids_sha256(sample_ids)
    ordered_membership_digest = _sha256(list(membership))
    receipt_digests = [
        _digest(digest, "preflight input receipt digest")
        for digest in _sequence(
            calibration["input_receipt_sha256s"],
            length=CALIBRATION_SIZE,
            label="preflight input receipt digests",
        )
    ]
    expected_receipts = [_sha256(record) for record in membership]
    if (
        _digest(
            calibration["ordered_sample_ids_sha256"],
            "preflight ordered sample IDs digest",
        )
        != ordered_ids_digest
        or _digest(
            calibration["ordered_membership_sha256"],
            "preflight ordered membership digest",
        )
        != ordered_membership_digest
        or receipt_digests != expected_receipts
    ):
        raise ValueError("preflight authoritative calibration membership differs")
    return ordered_ids_digest, ordered_membership_digest, receipt_digests


def _tensor_entry(tensor: object, asset_sha256: str, label: str) -> dict[str, object]:
    if type(tensor) is not torch.Tensor:
        raise TypeError(f"{label} tensor must be an exact torch.Tensor")
    if tensor.device.type != "cpu":
        raise ValueError(f"{label} tensor must be on CPU")
    if tensor.layout != torch.strided:
        raise ValueError(f"{label} tensor must use a strided layout")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{label} tensor must use torch.float32")
    if tensor.ndim != 3 or tensor.shape[0] != 4 or any(size <= 0 for size in tensor.shape):
        raise ValueError(f"{label} tensor shape must be positive four-channel CHW")
    if not tensor.is_contiguous():
        raise ValueError(f"{label} tensor must be contiguous")
    if tensor.requires_grad:
        raise ValueError(f"{label} tensor must not require gradients")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} tensor must contain finite reflectance")
    return {
        "asset_sha256": _digest(asset_sha256, f"{label} asset digest"),
        "tensor_sha256": tensor_sha256(tensor),
        "shape": list(tensor.shape),
        "dtype": "torch.float32",
    }


def _sample(
    record: Mapping[str, object], loaded: object
) -> dict[str, object]:
    if type(loaded) is not LoadedCrosssensorPair:
        raise TypeError("calibration input receipt requires exact loaded pairs")
    pair = loaded.pair
    metadata = loaded.metadata
    if type(pair) is not SRPair or type(metadata) is not CrosssensorPairMetadata:
        raise TypeError("calibration input receipt pair state is invalid")
    membership = _membership(record)
    sample_id = membership["sample_id"]
    if (
        not _exact(pair.sample_id, sample_id)
        or not _exact(metadata.sample_id, sample_id)
        or not _exact(pair.source, _SOURCE)
        or type(pair.scale) is not int
        or pair.scale != 4
    ):
        raise ValueError("loaded calibration pair is out of order or has invalid identity")
    if (
        not _exact(metadata.split, "calibration")
        or not _exact(metadata.manifest_sha256, POST_MANIFEST_SHA256)
        or not _exact(metadata.normalization_policy, PHASE2B3A_NORMALIZATION_POLICY)
        or not _exact(metadata.crop_policy, CROP_POLICY)
    ):
        raise ValueError("loaded calibration pair has invalid split, manifest, or policy")
    expected_metadata = {
        "spatial_group_id": membership["spatial_group_id"],
        "days_between": membership["days_between"],
        "correlation_bin": membership["correlation_bin"],
        "selection_round": membership["selection_round"],
        "lr_asset_sha256": membership["lr_asset_sha256"],
        "hr_asset_sha256": membership["hr_asset_sha256"],
    }
    if any(
        not _exact(getattr(metadata, key), expected)
        for key, expected in expected_metadata.items()
    ):
        raise ValueError("loaded calibration pair metadata or asset membership differs")
    lr = _tensor_entry(pair.lr, metadata.lr_asset_sha256, "LR")
    hr = _tensor_entry(pair.hr, metadata.hr_asset_sha256, "HR")
    if hr["shape"][1:] != [lr["shape"][1] * 4, lr["shape"][2] * 4]:
        raise ValueError("HR tensor shape must be exactly four times the LR shape")
    pair.validate()
    return {"membership": membership, "lr": lr, "hr": hr}


def build_calibration_input_receipt(
    calibration_records: Sequence[Mapping[str, object]],
    loaded_pairs: Sequence[LoadedCrosssensorPair],
    preflight: Mapping[str, object],
) -> dict[str, object]:
    """Bind frozen manifest membership to exact loaded, normalized CPU tensors."""

    records = validate_calibration_records(calibration_records)
    membership = [_membership(record) for record in records]
    ordered_ids_digest, ordered_membership_digest, receipt_digests = (
        _validate_preflight(preflight, membership)
    )
    if isinstance(loaded_pairs, str | bytes) or not isinstance(loaded_pairs, Sequence):
        raise TypeError("loaded calibration pairs must be a stable sequence")
    if len(loaded_pairs) != CALIBRATION_SIZE:
        raise ValueError("calibration input receipt requires exactly 120 loaded pairs")
    samples = [
        _sample(record, loaded)
        for record, loaded in zip(records, loaded_pairs, strict=True)
    ]
    return {
        "schema": SCHEMA,
        "split": "calibration",
        "sample_count": CALIBRATION_SIZE,
        "input": {
            "manifest_sha256": POST_MANIFEST_SHA256,
            "source": _SOURCE,
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "crop_policy": CROP_POLICY,
            "bands": list(_BANDS),
            "scale": 4,
        },
        "ordered_sample_ids_sha256": ordered_ids_digest,
        "ordered_membership_sha256": ordered_membership_digest,
        "input_receipt_sha256s": list(receipt_digests),
        "samples": samples,
        "ordered_inputs_sha256": _sha256(samples),
    }


def _verified_membership(value: object) -> dict[str, object]:
    membership = _json_mapping(value, _MEMBERSHIP_KEYS, "receipt membership")
    for field in ("sample_id", "spatial_group_id"):
        if type(membership[field]) is not str or not membership[field]:
            raise ValueError(f"receipt membership {field} must be a non-empty string")
    for field in (
        "selection_sha256",
        "lr_asset_sha256",
        "hr_asset_sha256",
    ):
        _digest(membership[field], f"receipt membership {field}")
    for field in ("days_between", "correlation_bin", "selection_round"):
        if type(membership[field]) is not int:
            raise TypeError(f"receipt membership {field} must be an integer")
    if (
        membership["days_between"] not in _DAYS
        or membership["correlation_bin"] not in _BINS
        or membership["selection_round"] not in _ROUNDS
    ):
        raise ValueError("receipt membership stratum or round is invalid")
    return dict(membership)


def _verified_asset(value: object, *, kind: str) -> dict[str, object]:
    asset = _json_mapping(value, _ASSET_KEYS, f"receipt {kind} input")
    shape = _json_list(asset["shape"], length=3, label=f"receipt {kind} shape")
    if (
        any(type(size) is not int or size <= 0 for size in shape)
        or shape[0] != 4
        or asset["dtype"] != "torch.float32"
        or type(asset["dtype"]) is not str
    ):
        raise ValueError(f"receipt {kind} tensor shape or dtype is invalid")
    return {
        "asset_sha256": _digest(asset["asset_sha256"], f"receipt {kind} asset digest"),
        "tensor_sha256": _digest(asset["tensor_sha256"], f"receipt {kind} tensor digest"),
        "shape": list(shape),
        "dtype": "torch.float32",
    }


def _verified_sample(value: object) -> dict[str, object]:
    sample = _json_mapping(value, _SAMPLE_KEYS, "calibration input sample")
    membership = _verified_membership(sample["membership"])
    lr = _verified_asset(sample["lr"], kind="LR")
    hr = _verified_asset(sample["hr"], kind="HR")
    if (
        lr["asset_sha256"] != membership["lr_asset_sha256"]
        or hr["asset_sha256"] != membership["hr_asset_sha256"]
        or hr["shape"][1:] != [lr["shape"][1] * 4, lr["shape"][2] * 4]
    ):
        raise ValueError("receipt assets or tensor shapes differ from membership")
    return {"membership": membership, "lr": lr, "hr": hr}


def _validate_receipt_input(value: object) -> None:
    input_identity = _json_mapping(value, _INPUT_KEYS, "receipt input identity")
    if (
        input_identity["manifest_sha256"] != POST_MANIFEST_SHA256
        or input_identity["source"] != _SOURCE
        or input_identity["normalization_policy"] != PHASE2B3A_NORMALIZATION_POLICY
        or input_identity["crop_policy"] != CROP_POLICY
        or input_identity["bands"] != list(_BANDS)
        or any(type(band) is not str for band in input_identity["bands"])
        or type(input_identity["scale"]) is not int
        or input_identity["scale"] != 4
    ):
        raise ValueError("receipt input identity is invalid")


def verify_calibration_input_receipt(
    receipt: object,
) -> VerifiedCalibrationInputReceipt:
    """Independently verify a parsed receipt without loading inputs or calling builder."""

    value = _json_mapping(receipt, _RECEIPT_KEYS, "calibration input receipt")
    canonical_receipt = canonical_json(value)
    if value["schema"] != SCHEMA or type(value["schema"]) is not str:
        raise ValueError("calibration input receipt schema is invalid")
    if value["split"] != "calibration" or type(value["split"]) is not str:
        raise ValueError("calibration input receipt split is invalid")
    if type(value["sample_count"]) is not int or value["sample_count"] != CALIBRATION_SIZE:
        raise ValueError("calibration input receipt requires exactly 120 samples")
    _validate_receipt_input(value["input"])
    samples = [
        _verified_sample(sample)
        for sample in _json_list(
            value["samples"], length=CALIBRATION_SIZE, label="receipt samples"
        )
    ]
    membership = [sample["membership"] for sample in samples]
    sample_ids = [record["sample_id"] for record in membership]
    for field in ("sample_id", "selection_sha256", "spatial_group_id"):
        identities = [record[field] for record in membership]
        if len(set(identities)) != CALIBRATION_SIZE:
            raise ValueError(f"receipt membership requires unique {field}")
    cells: dict[tuple[int, int], list[int]] = {
        (day, bin_index): [] for day in _DAYS for bin_index in _BINS
    }
    for record in membership:
        cells[(record["days_between"], record["correlation_bin"])].append(
            record["selection_round"]
        )
    if any(tuple(sorted(rounds)) != _ROUNDS for rounds in cells.values()):
        raise ValueError("receipt membership requires complete calibration strata")
    ordered_ids_digest = ordered_sample_ids_sha256(sample_ids)
    ordered_membership_digest = _sha256(membership)
    receipt_digests = [
        _digest(digest, "receipt per-record membership digest")
        for digest in _json_list(
            value["input_receipt_sha256s"],
            length=CALIBRATION_SIZE,
            label="receipt per-record digests",
        )
    ]
    if receipt_digests != [_sha256(record) for record in membership]:
        raise ValueError("receipt per-record membership digests are inconsistent")
    if (
        _digest(value["ordered_sample_ids_sha256"], "receipt ordered sample IDs digest")
        != ordered_ids_digest
        or _digest(
            value["ordered_membership_sha256"], "receipt ordered membership digest"
        )
        != ordered_membership_digest
    ):
        raise ValueError("receipt ordered membership aggregate is inconsistent")
    ordered_inputs_digest = _sha256(samples)
    if (
        _digest(value["ordered_inputs_sha256"], "receipt ordered inputs digest")
        != ordered_inputs_digest
    ):
        raise ValueError("receipt ordered input aggregate is inconsistent")
    return VerifiedCalibrationInputReceipt(
        source_sha256=hashlib.sha256(canonical_receipt).hexdigest(),
        ordered_inputs_sha256=ordered_inputs_digest,
        ordered_sample_ids_sha256=ordered_ids_digest,
        ordered_membership_sha256=ordered_membership_digest,
        sample_count=CALIBRATION_SIZE,
    )
