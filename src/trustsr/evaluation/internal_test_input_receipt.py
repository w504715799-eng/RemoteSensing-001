"""Host-free identity receipt for loaded Phase 2B3-C inputs."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from trustsr.artifacts.predictions import tensor_sha256
from trustsr.contracts import SRPair
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
)
from trustsr.data.internal_test_pairs import (
    INTERNAL_TEST_SIZE,
    validate_internal_test_records,
)
from trustsr.jsonio import canonical_json

SCHEMA = "trustsr.phase2b3c-internal-test-input-receipt.v1"
_SOURCE = f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}"
_BANDS = ("B04", "B03", "B02", "B08")
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
_INPUT_KEYS = {
    "manifest_sha256",
    "source",
    "normalization_policy",
    "crop_policy",
    "bands",
    "scale",
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
class VerifiedInternalTestInputReceipt:
    """Validated host-free identity of all 120 normalized input pairs."""

    source_sha256: str
    ordered_inputs_sha256: str
    ordered_sample_ids_sha256: str
    ordered_membership_sha256: str
    sample_count: int


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _membership(record: Mapping[str, object]) -> dict[str, object]:
    lr_asset = record["lr_asset"]
    hr_asset = record["hr_asset"]
    if not isinstance(lr_asset, Mapping) or not isinstance(hr_asset, Mapping):
        raise ValueError("internal_test membership requires LR and HR assets")
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


def _tensor_entry(tensor: object, asset_sha256: str, label: str) -> dict[str, object]:
    if type(tensor) is not torch.Tensor:
        raise TypeError(f"{label} tensor must be an exact torch.Tensor")
    if tensor.device.type != "cpu" or tensor.layout != torch.strided:
        raise ValueError(f"{label} tensor must be a CPU strided tensor")
    if tensor.dtype != torch.float32 or not tensor.is_contiguous():
        raise ValueError(f"{label} tensor must be contiguous torch.float32")
    if tensor.ndim != 3 or tensor.shape[0] != 4 or any(size <= 0 for size in tensor.shape):
        raise ValueError(f"{label} tensor must have positive four-channel CHW shape")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{label} tensor values must be finite")
    if bool(torch.any(tensor < 0).item()) or bool(torch.any(tensor > 1).item()):
        raise ValueError(f"{label} tensor values must be in [0, 1]")
    return {
        "asset_sha256": asset_sha256,
        "tensor_sha256": tensor_sha256(tensor),
        "shape": list(tensor.shape),
        "dtype": "torch.float32",
    }


def _sample(
    record: Mapping[str, object], loaded: LoadedCrosssensorPair
) -> dict[str, object]:
    if type(loaded) is not LoadedCrosssensorPair:
        raise TypeError("internal_test input receipt requires exact loaded pairs")
    pair = loaded.pair
    metadata = loaded.metadata
    if type(pair) is not SRPair or type(metadata) is not CrosssensorPairMetadata:
        raise TypeError("internal_test input receipt pair state is invalid")
    membership = _membership(record)
    if pair.sample_id != membership["sample_id"] or metadata.sample_id != membership["sample_id"]:
        raise ValueError("loaded internal_test pair is out of order")
    if (
        pair.source != _SOURCE
        or pair.scale != 4
        or metadata.split != "internal_test"
        or metadata.manifest_sha256 != POST_MANIFEST_SHA256
        or metadata.normalization_policy != PHASE2B3A_NORMALIZATION_POLICY
        or metadata.crop_policy != CROP_POLICY
    ):
        raise ValueError("loaded internal_test pair has invalid frozen identity")
    expected = {
        "spatial_group_id": membership["spatial_group_id"],
        "days_between": membership["days_between"],
        "correlation_bin": membership["correlation_bin"],
        "selection_round": membership["selection_round"],
        "lr_asset_sha256": membership["lr_asset_sha256"],
        "hr_asset_sha256": membership["hr_asset_sha256"],
    }
    if any(getattr(metadata, key) != value for key, value in expected.items()):
        raise ValueError("loaded internal_test pair metadata differs from membership")
    lr = _tensor_entry(pair.lr, metadata.lr_asset_sha256, "LR")
    hr = _tensor_entry(pair.hr, metadata.hr_asset_sha256, "HR")
    if hr["shape"][1:] != [lr["shape"][1] * 4, lr["shape"][2] * 4]:
        raise ValueError("HR tensor shape must be exactly four times the LR shape")
    pair.validate()
    return {"membership": membership, "lr": lr, "hr": hr}


def build_internal_test_input_receipt(
    records: Sequence[Mapping[str, object]],
    loaded_pairs: Sequence[LoadedCrosssensorPair],
) -> dict[str, object]:
    """Bind ordered frozen membership to exact normalized CPU tensors."""

    validated = validate_internal_test_records(records)
    if isinstance(loaded_pairs, str | bytes) or not isinstance(loaded_pairs, Sequence):
        raise TypeError("loaded internal_test pairs must be a stable sequence")
    if len(loaded_pairs) != INTERNAL_TEST_SIZE:
        raise ValueError("internal_test input receipt requires exactly 120 loaded pairs")
    memberships = [_membership(record) for record in validated]
    samples = [
        _sample(record, loaded)
        for record, loaded in zip(validated, loaded_pairs, strict=True)
    ]
    return {
        "schema": SCHEMA,
        "split": "internal_test",
        "sample_count": INTERNAL_TEST_SIZE,
        "input": {
            "manifest_sha256": POST_MANIFEST_SHA256,
            "source": _SOURCE,
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "crop_policy": CROP_POLICY,
            "bands": list(_BANDS),
            "scale": 4,
        },
        "ordered_sample_ids_sha256": _sha256(
            [membership["sample_id"] for membership in memberships]
        ),
        "ordered_membership_sha256": _sha256(memberships),
        "input_receipt_sha256s": [_sha256(membership) for membership in memberships],
        "samples": samples,
        "ordered_inputs_sha256": _sha256(samples),
    }


def _mapping(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} keys are invalid")
    return value


def _verify_asset(value: object, membership_digest: str, kind: str) -> dict[str, object]:
    asset = _mapping(value, _ASSET_KEYS, f"receipt {kind} asset")
    if asset["asset_sha256"] != membership_digest:
        raise ValueError(f"receipt {kind} asset differs from membership")
    _digest(asset["asset_sha256"], f"receipt {kind} asset")
    _digest(asset["tensor_sha256"], f"receipt {kind} tensor")
    shape = asset["shape"]
    if (
        type(shape) is not list
        or len(shape) != 3
        or any(type(size) is not int or size <= 0 for size in shape)
        or shape[0] != 4
        or asset["dtype"] != "torch.float32"
    ):
        raise ValueError(f"receipt {kind} tensor shape or dtype is invalid")
    return asset


def verify_internal_test_input_receipt(value: object) -> VerifiedInternalTestInputReceipt:
    """Strictly validate a parsed canonical receipt and all internal bindings."""

    receipt = _mapping(value, _RECEIPT_KEYS, "internal_test input receipt")
    if (
        receipt["schema"] != SCHEMA
        or receipt["split"] != "internal_test"
        or type(receipt["sample_count"]) is not int
        or receipt["sample_count"] != INTERNAL_TEST_SIZE
    ):
        raise ValueError("internal_test input receipt identity is invalid")
    expected_input = {
        "manifest_sha256": POST_MANIFEST_SHA256,
        "source": _SOURCE,
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "crop_policy": CROP_POLICY,
        "bands": list(_BANDS),
        "scale": 4,
    }
    if _mapping(receipt["input"], _INPUT_KEYS, "receipt input") != expected_input:
        raise ValueError("internal_test input receipt policy is invalid")
    samples = receipt["samples"]
    if type(samples) is not list or len(samples) != INTERNAL_TEST_SIZE:
        raise ValueError("internal_test input receipt requires exactly 120 samples")
    memberships: list[dict[str, object]] = []
    sample_ids: list[str] = []
    for sample_value in samples:
        sample = _mapping(sample_value, _SAMPLE_KEYS, "receipt sample")
        membership = _mapping(sample["membership"], _MEMBERSHIP_KEYS, "membership")
        sample_id = membership["sample_id"]
        if type(sample_id) is not str or not sample_id:
            raise ValueError("receipt sample_id must be a non-empty string")
        _digest(membership["selection_sha256"], "receipt selection_sha256")
        _digest(membership["spatial_group_id"], "receipt spatial_group_id")
        lr_digest = _digest(membership["lr_asset_sha256"], "receipt LR asset")
        hr_digest = _digest(membership["hr_asset_sha256"], "receipt HR asset")
        lr = _verify_asset(sample["lr"], lr_digest, "LR")
        hr = _verify_asset(sample["hr"], hr_digest, "HR")
        if hr["shape"][1:] != [lr["shape"][1] * 4, lr["shape"][2] * 4]:
            raise ValueError("receipt HR shape must be exactly four times LR")
        memberships.append(dict(membership))
        sample_ids.append(sample_id)
    if len(set(sample_ids)) != INTERNAL_TEST_SIZE:
        raise ValueError("receipt sample IDs must be unique")
    validate_internal_test_records(
        tuple(
            {
                **membership,
                "split": "internal_test",
                "lr_asset": {"sha256": membership["lr_asset_sha256"]},
                "hr_asset": {"sha256": membership["hr_asset_sha256"]},
            }
            for membership in memberships
        )
    )
    ordered_ids = _digest(receipt["ordered_sample_ids_sha256"], "ordered sample IDs")
    ordered_membership = _digest(
        receipt["ordered_membership_sha256"], "ordered membership"
    )
    ordered_inputs = _digest(receipt["ordered_inputs_sha256"], "ordered inputs")
    per_record = receipt["input_receipt_sha256s"]
    if (
        type(per_record) is not list
        or len(per_record) != INTERNAL_TEST_SIZE
        or [
            _digest(item, "per-record input receipt") for item in per_record
        ]
        != [_sha256(membership) for membership in memberships]
    ):
        raise ValueError("receipt per-record membership digests are invalid")
    if ordered_ids != _sha256(sample_ids):
        raise ValueError("receipt ordered sample IDs digest is invalid")
    if ordered_membership != _sha256(memberships):
        raise ValueError("receipt ordered membership digest is invalid")
    if ordered_inputs != _sha256(samples):
        raise ValueError("receipt ordered inputs digest is invalid")
    return VerifiedInternalTestInputReceipt(
        source_sha256=hashlib.sha256(canonical_json(receipt)).hexdigest(),
        ordered_inputs_sha256=ordered_inputs,
        ordered_sample_ids_sha256=ordered_ids,
        ordered_membership_sha256=ordered_membership,
        sample_count=INTERNAL_TEST_SIZE,
    )
