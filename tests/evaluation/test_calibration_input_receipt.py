"""CPU-only contracts for frozen calibration input receipts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from trustsr.artifacts.predictions import tensor_sha256
from trustsr.contracts import SRPair
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
    RadiometricSaturation,
)
from trustsr.evaluation import calibration_input_receipt
from trustsr.jsonio import canonical_json

_SOURCE = f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}"


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "sample_id": f"calibration-{index:03d}",
            "selection_sha256": _sha(f"selection:{index}"),
            "spatial_group_id": f"group-{index:03d}",
            "split": "calibration",
            "days_between": day,
            "correlation_bin": bin_index,
            "selection_round": round_index,
            "lr_asset": {
                "relative_path": f"secret/calibration-{index:03d}/lr.tif",
                "sha256": _sha(f"lr-asset:{index}"),
            },
            "hr_asset": {
                "relative_path": f"secret/calibration-{index:03d}/hr.tif",
                "sha256": _sha(f"hr-asset:{index}"),
            },
        }
        for index, (day, bin_index, round_index) in enumerate(
            (day, bin_index, round_index)
            for day in (-1, 0, 1)
            for bin_index in range(4)
            for round_index in range(1, 11)
        )
    )


def _membership(record: dict[str, object]) -> dict[str, object]:
    return {
        "sample_id": record["sample_id"],
        "selection_sha256": record["selection_sha256"],
        "spatial_group_id": record["spatial_group_id"],
        "lr_asset_sha256": record["lr_asset"]["sha256"],
        "hr_asset_sha256": record["hr_asset"]["sha256"],
        "days_between": record["days_between"],
        "correlation_bin": record["correlation_bin"],
        "selection_round": record["selection_round"],
    }


def _preflight(records: tuple[dict[str, object], ...]) -> dict[str, object]:
    membership = [_membership(record) for record in records]
    sample_ids = [record["sample_id"] for record in records]
    return {
        "schema": "trustsr.phase2b3b-preflight.v1",
        "upstream": {"post_manifest_sha256": POST_MANIFEST_SHA256},
        "calibration": {
            "split": "calibration",
            "sample_count": 120,
            "ordered_sample_ids_sha256": _sha(sample_ids),
            "ordered_membership_sha256": _sha(membership),
            "input_receipt_sha256s": [_sha(record) for record in membership],
            "strata": [],
        },
        "score": {},
        "risk": {},
        "input": {
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "crop_policy": CROP_POLICY,
            "bands": ["B04", "B03", "B02", "B08"],
            "scale": 4,
        },
    }


def _pair(record: dict[str, object], *, value: float = 0.25) -> LoadedCrosssensorPair:
    sample_id = str(record["sample_id"])
    lr = torch.full((4, 2, 3), value, dtype=torch.float32)
    hr = torch.full((4, 8, 12), value + 0.25, dtype=torch.float32)
    return LoadedCrosssensorPair(
        pair=SRPair(sample_id=sample_id, source=_SOURCE, lr=lr, hr=hr, scale=4),
        metadata=CrosssensorPairMetadata(
            manifest_sha256=POST_MANIFEST_SHA256,
            sample_id=sample_id,
            split="calibration",
            spatial_group_id=str(record["spatial_group_id"]),
            days_between=int(record["days_between"]),
            correlation_bin=int(record["correlation_bin"]),
            selection_round=int(record["selection_round"]),
            lr_asset_sha256=str(record["lr_asset"]["sha256"]),
            hr_asset_sha256=str(record["hr_asset"]["sha256"]),
            lr_crop_transform=(10.0, 0.0, 10.0, 0.0, -10.0, -10.0),
            hr_crop_transform=(2.5, 0.0, 10.0, 0.0, -2.5, -10.0),
            crop_bounds=(10.0, -30.0, 40.0, -10.0),
            crop_policy=CROP_POLICY,
            normalization_policy=PHASE2B3A_NORMALIZATION_POLICY,
            lr_saturation=RadiometricSaturation(2500, 2500, 0, (0, 0, 0, 0)),
            hr_saturation=RadiometricSaturation(5000, 5000, 0, (0, 0, 0, 0)),
        ),
    )


@pytest.fixture
def inputs() -> tuple[
    tuple[dict[str, object], ...],
    tuple[LoadedCrosssensorPair, ...],
    dict[str, object],
]:
    records = _records()
    return records, tuple(_pair(record) for record in records), _preflight(records)


def test_builds_and_independently_verifies_canonical_host_free_receipt(
    inputs: tuple[
        tuple[dict[str, object], ...],
        tuple[LoadedCrosssensorPair, ...],
        dict[str, object],
    ],
) -> None:
    records, pairs, preflight = inputs

    first = calibration_input_receipt.build_calibration_input_receipt(
        records, pairs, preflight
    )
    second = calibration_input_receipt.build_calibration_input_receipt(
        records, pairs, preflight
    )
    verified = calibration_input_receipt.verify_calibration_input_receipt(
        json.loads(canonical_json(first))
    )

    assert set(first) == {
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
    assert first["schema"] == "trustsr.phase2b3b-calibration-input-receipt.v1"
    assert first["split"] == "calibration"
    assert first["sample_count"] == 120
    assert first["ordered_sample_ids_sha256"] == preflight["calibration"][
        "ordered_sample_ids_sha256"
    ]
    assert first["ordered_membership_sha256"] == preflight["calibration"][
        "ordered_membership_sha256"
    ]
    assert first["input_receipt_sha256s"] == preflight["calibration"][
        "input_receipt_sha256s"
    ]
    assert first["ordered_inputs_sha256"] == _sha(first["samples"])
    assert first["samples"][0] == {
        "membership": _membership(records[0]),
        "lr": {
            "asset_sha256": pairs[0].metadata.lr_asset_sha256,
            "tensor_sha256": tensor_sha256(pairs[0].pair.lr),
            "shape": [4, 2, 3],
            "dtype": "torch.float32",
        },
        "hr": {
            "asset_sha256": pairs[0].metadata.hr_asset_sha256,
            "tensor_sha256": tensor_sha256(pairs[0].pair.hr),
            "shape": [4, 8, 12],
            "dtype": "torch.float32",
        },
    }
    assert verified.source_sha256 == _sha(first)
    assert verified.ordered_inputs_sha256 == first["ordered_inputs_sha256"]
    assert verified.ordered_membership_sha256 == first["ordered_membership_sha256"]
    assert verified.sample_count == 120
    assert canonical_json(first) == canonical_json(second)
    assert first is not second
    with pytest.raises(FrozenInstanceError):
        verified.sample_count = 0  # type: ignore[misc]

    encoded = canonical_json(first).decode()
    for forbidden in ("secret/", "relative_path", "hostname", "timestamp", "cuda"):
        assert forbidden not in encoded


@pytest.mark.parametrize(
    "fault", ("order", "lr_asset", "hr_asset", "split", "metadata_type", "count")
)
def test_builder_rejects_pair_membership_or_order_mismatch(
    inputs: tuple[
        tuple[dict[str, object], ...],
        tuple[LoadedCrosssensorPair, ...],
        dict[str, object],
    ],
    fault: str,
) -> None:
    records, source_pairs, preflight = inputs
    pairs = list(source_pairs)
    if fault == "order":
        pairs[0], pairs[1] = pairs[1], pairs[0]
    elif fault == "lr_asset":
        pairs[0] = replace(
            pairs[0], metadata=replace(pairs[0].metadata, lr_asset_sha256="0" * 64)
        )
    elif fault == "hr_asset":
        pairs[0] = replace(
            pairs[0], metadata=replace(pairs[0].metadata, hr_asset_sha256="0" * 64)
        )
    elif fault == "split":
        pairs[0] = replace(
            pairs[0], metadata=replace(pairs[0].metadata, split="internal_test")
        )
    elif fault == "metadata_type":
        pairs[0] = replace(
            pairs[0], metadata=replace(pairs[0].metadata, selection_round=True)
        )
    else:
        pairs.pop()

    with pytest.raises((TypeError, ValueError), match="120|order|asset|split|membership"):
        calibration_input_receipt.build_calibration_input_receipt(
            records, pairs, preflight
        )


@pytest.mark.parametrize(
    "fault", ("nan", "shape", "dtype", "noncontiguous", "requires_grad", "device")
)
def test_builder_rejects_ambiguous_or_invalid_normalized_tensors(
    inputs: tuple[
        tuple[dict[str, object], ...],
        tuple[LoadedCrosssensorPair, ...],
        dict[str, object],
    ],
    fault: str,
) -> None:
    records, source_pairs, preflight = inputs
    pairs = list(source_pairs)
    original = pairs[0].pair
    if fault == "nan":
        lr = original.lr.clone()
        lr[0, 0, 0] = float("nan")
        changed = replace(original, lr=lr)
    elif fault == "shape":
        changed = replace(original, hr=torch.zeros((4, 7, 12), dtype=torch.float32))
    elif fault == "dtype":
        changed = replace(original, lr=original.lr.to(torch.float64))
    elif fault == "noncontiguous":
        changed = replace(original, lr=original.lr.transpose(1, 2))
    elif fault == "requires_grad":
        changed = replace(original, hr=original.hr.clone().requires_grad_())
    else:
        changed = replace(original, lr=torch.empty((4, 2, 3), device="meta"))
    pairs[0] = replace(pairs[0], pair=changed)

    with pytest.raises((TypeError, ValueError), match="tensor|CPU|float32|finite|shape|contiguous"):
        calibration_input_receipt.build_calibration_input_receipt(
            records, pairs, preflight
        )


def test_valid_tensor_replacement_changes_the_ordered_input_commitment(
    inputs: tuple[
        tuple[dict[str, object], ...],
        tuple[LoadedCrosssensorPair, ...],
        dict[str, object],
    ],
) -> None:
    records, source_pairs, preflight = inputs
    first = calibration_input_receipt.build_calibration_input_receipt(
        records, source_pairs, preflight
    )
    pairs = list(source_pairs)
    replacement = replace(
        pairs[0].pair,
        lr=torch.full_like(pairs[0].pair.lr, 0.3),
        hr=torch.full_like(pairs[0].pair.hr, 0.6),
    )
    pairs[0] = replace(pairs[0], pair=replacement)

    second = calibration_input_receipt.build_calibration_input_receipt(
        records, pairs, preflight
    )

    assert first["samples"][0]["lr"]["tensor_sha256"] != second["samples"][0][
        "lr"
    ]["tensor_sha256"]
    assert first["ordered_inputs_sha256"] != second["ordered_inputs_sha256"]


@pytest.mark.parametrize(
    "fault",
    (
        "extra",
        "swap",
        "membership",
        "asset",
        "tensor_digest",
        "shape",
        "ordered_samples",
        "ordered_membership",
        "ordered_inputs",
    ),
)
def test_independent_verifier_rejects_hostile_receipt_mutations(
    inputs: tuple[
        tuple[dict[str, object], ...],
        tuple[LoadedCrosssensorPair, ...],
        dict[str, object],
    ],
    fault: str,
) -> None:
    records, pairs, preflight = inputs
    receipt = json.loads(
        canonical_json(
            calibration_input_receipt.build_calibration_input_receipt(
                records, pairs, preflight
            )
        )
    )
    if fault == "extra":
        receipt["extra"] = True
    elif fault == "swap":
        receipt["samples"][0], receipt["samples"][1] = (
            receipt["samples"][1],
            receipt["samples"][0],
        )
    elif fault == "membership":
        receipt["samples"][0]["membership"]["selection_round"] = 2
    elif fault == "asset":
        receipt["samples"][0]["lr"]["asset_sha256"] = "0" * 64
    elif fault == "tensor_digest":
        receipt["samples"][0]["hr"]["tensor_sha256"] = "0" * 64
    elif fault == "shape":
        receipt["samples"][0]["hr"]["shape"] = [4, 7, 12]
    elif fault == "ordered_samples":
        receipt["ordered_sample_ids_sha256"] = "0" * 64
    elif fault == "ordered_membership":
        receipt["ordered_membership_sha256"] = "0" * 64
    elif fault == "ordered_inputs":
        receipt["ordered_inputs_sha256"] = "0" * 64

    with pytest.raises((TypeError, ValueError)):
        calibration_input_receipt.verify_calibration_input_receipt(receipt)


@pytest.mark.parametrize(
    "fault", ("records_order", "sample_digest", "membership_digest", "per_record")
)
def test_builder_rejects_preflight_authority_mismatch(
    inputs: tuple[
        tuple[dict[str, object], ...],
        tuple[LoadedCrosssensorPair, ...],
        dict[str, object],
    ],
    fault: str,
) -> None:
    source_records, pairs, source_preflight = inputs
    records = list(source_records)
    preflight = json.loads(canonical_json(source_preflight))
    if fault == "records_order":
        records[0], records[1] = records[1], records[0]
        pairs = (pairs[1], pairs[0], *pairs[2:])
    elif fault == "sample_digest":
        preflight["calibration"]["ordered_sample_ids_sha256"] = "0" * 64
    elif fault == "membership_digest":
        preflight["calibration"]["ordered_membership_sha256"] = "0" * 64
    else:
        preflight["calibration"]["input_receipt_sha256s"][0] = "0" * 64

    with pytest.raises(ValueError, match="preflight|authoritative|membership|receipt"):
        calibration_input_receipt.build_calibration_input_receipt(
            records, pairs, preflight
        )


def test_verifier_does_not_call_builder(
    inputs: tuple[
        tuple[dict[str, object], ...],
        tuple[LoadedCrosssensorPair, ...],
        dict[str, object],
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, pairs, preflight = inputs
    receipt = calibration_input_receipt.build_calibration_input_receipt(
        records, pairs, preflight
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("verifier called builder")

    monkeypatch.setattr(
        calibration_input_receipt, "build_calibration_input_receipt", forbidden
    )
    assert calibration_input_receipt.verify_calibration_input_receipt(
        json.loads(canonical_json(receipt))
    ).sample_count == 120
