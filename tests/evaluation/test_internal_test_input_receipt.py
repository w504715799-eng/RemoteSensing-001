"""Host-free receipt for synthetic Phase 2B3-C input pairs."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from trustsr.contracts import SRPair
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
    RadiometricSaturation,
)
from trustsr.evaluation.internal_test_input_receipt import (
    build_internal_test_input_receipt,
    verify_internal_test_input_receipt,
)
from trustsr.jsonio import canonical_json


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "sample_id": f"internal-test-{index:03d}",
            "selection_sha256": f"{index:064x}",
            "spatial_group_id": f"{index + 1000:064x}",
            "split": "internal_test",
            "days_between": day,
            "correlation_bin": bin_index,
            "selection_round": round_index,
            "lr_asset": {"sha256": "a" * 64},
            "hr_asset": {"sha256": "b" * 64},
        }
        for index, (day, bin_index, round_index) in enumerate(
            (day, bin_index, round_index)
            for day in (-1, 0, 1)
            for bin_index in range(4)
            for round_index in range(1, 11)
        )
    )


def _loaded(record: dict[str, object]) -> LoadedCrosssensorPair:
    sample_id = str(record["sample_id"])
    return LoadedCrosssensorPair(
        pair=SRPair(
            sample_id=sample_id,
            source=f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}",
            lr=torch.full((4, 3, 3), 0.25, dtype=torch.float32),
            hr=torch.full((4, 12, 12), 0.5, dtype=torch.float32),
            scale=4,
        ),
        metadata=CrosssensorPairMetadata(
            manifest_sha256=POST_MANIFEST_SHA256,
            sample_id=sample_id,
            split="internal_test",
            spatial_group_id=str(record["spatial_group_id"]),
            days_between=int(record["days_between"]),
            correlation_bin=int(record["correlation_bin"]),
            selection_round=int(record["selection_round"]),
            lr_asset_sha256="a" * 64,
            hr_asset_sha256="b" * 64,
            lr_crop_transform=(10.0, 0.0, 10.0, 0.0, -10.0, -10.0),
            hr_crop_transform=(2.5, 0.0, 10.0, 0.0, -2.5, -10.0),
            crop_bounds=(10.0, -30.0, 40.0, -10.0),
            crop_policy=CROP_POLICY,
            normalization_policy=PHASE2B3A_NORMALIZATION_POLICY,
            lr_saturation=RadiometricSaturation(2500, 2500, 0, (0, 0, 0, 0)),
            hr_saturation=RadiometricSaturation(5000, 5000, 0, (0, 0, 0, 0)),
        ),
    )


def test_builds_deterministic_host_free_receipt() -> None:
    records = _records()
    pairs = tuple(_loaded(record) for record in records)

    first = build_internal_test_input_receipt(records, pairs)
    second = build_internal_test_input_receipt(records, pairs)

    assert canonical_json(first) == canonical_json(second)
    assert first["schema"] == "trustsr.phase2b3c-internal-test-input-receipt.v1"
    assert first["split"] == "internal_test"
    assert first["sample_count"] == 120
    assert len(first["samples"]) == 120
    assert "storage_root" not in canonical_json(first).decode("utf-8")
    verified = verify_internal_test_input_receipt(first)
    assert verified.sample_count == 120
    assert verified.source_sha256 == hashlib.sha256(canonical_json(first)).hexdigest()


def test_receipt_binds_manifest_membership_order_and_exact_tensors() -> None:
    records = _records()
    pairs = tuple(_loaded(record) for record in records)
    first = build_internal_test_input_receipt(records, pairs)
    mutated_pair = replace(
        pairs[0], pair=replace(pairs[0].pair, lr=pairs[0].pair.lr.clone().fill_(0.75))
    )
    second = build_internal_test_input_receipt(records, (mutated_pair, *pairs[1:]))

    assert first["ordered_membership_sha256"] == second["ordered_membership_sha256"]
    assert first["ordered_inputs_sha256"] != second["ordered_inputs_sha256"]


def test_rejects_wrong_pair_order_and_nonfinite_tensor() -> None:
    records = _records()
    pairs = tuple(_loaded(record) for record in records)
    with pytest.raises(ValueError, match="order"):
        build_internal_test_input_receipt(records, (*pairs[1:], pairs[0]))
    pairs[0].pair.lr[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_internal_test_input_receipt(records, pairs)


@pytest.mark.parametrize(
    "path",
    (
        ("split",),
        ("sample_count",),
        ("ordered_membership_sha256",),
        ("samples", 0, "membership", "sample_id"),
        ("samples", 0, "lr", "tensor_sha256"),
    ),
)
def test_verifier_rejects_any_receipt_identity_mutation(path: tuple[object, ...]) -> None:
    receipt = deepcopy(
        build_internal_test_input_receipt(
            _records(), tuple(_loaded(record) for record in _records())
        )
    )
    target: object = receipt
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = "wrong"  # type: ignore[index]

    with pytest.raises((TypeError, ValueError)):
        verify_internal_test_input_receipt(receipt)
