"""Canonical evidence for repeated Phase 2B2-A input loading."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from trustsr.contracts import SRPair
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    NORMALIZATION_POLICY,
    PHASE2B1B_AUDIT_SHA256,
    POST_MANIFEST_SHA256,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
)
from trustsr.data.input_audit import build_input_audit


def _loaded_pairs() -> tuple[LoadedCrosssensorPair, ...]:
    lr = torch.full((4, 128, 128), 0.25, dtype=torch.float32)
    hr = torch.full((4, 512, 512), 0.5, dtype=torch.float32)
    result: list[LoadedCrosssensorPair] = []
    for split in ("calibration", "development", "internal_test"):
        for bin_index in range(4):
            sample_id = f"sample-{split}-{bin_index}"
            pair = SRPair(
                sample_id=sample_id,
                source=f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}",
                lr=lr.clone(),
                hr=hr.clone(),
                scale=4,
            )
            metadata = CrosssensorPairMetadata(
                manifest_sha256=POST_MANIFEST_SHA256,
                sample_id=sample_id,
                split=split,
                spatial_group_id=f"group-{split}-{bin_index}",
                days_between=-1,
                correlation_bin=bin_index,
                selection_round=1,
                lr_asset_sha256=f"{bin_index + 1:x}" * 64,
                hr_asset_sha256=f"{bin_index + 5:x}" * 64,
                lr_crop_transform=(10.0, 0.0, 500010.0, 0.0, -10.0, 399990.0),
                hr_crop_transform=(2.5, 0.0, 500010.0, 0.0, -2.5, 399990.0),
                crop_bounds=(500010.0, 398710.0, 501290.0, 399990.0),
                crop_policy=CROP_POLICY,
                normalization_policy=NORMALIZATION_POLICY,
            )
            result.append(LoadedCrosssensorPair(pair=pair, metadata=metadata))
    return tuple(result)


def test_input_audit_records_exact_counts_and_repeatable_tensor_digests() -> None:
    audit = build_input_audit(_loaded_pairs(), _loaded_pairs())

    assert audit["schema"] == "trustsr.phase2b2a-input-audit.v1"
    assert audit["post_manifest_sha256"] == POST_MANIFEST_SHA256
    assert audit["phase2b1b_audit_sha256"] == PHASE2B1B_AUDIT_SHA256
    assert audit["smoke_pair_count"] == 12
    assert audit["smoke_geotiff_count"] == 24
    assert audit["split_counts"] == {
        "calibration": 4,
        "development": 4,
        "internal_test": 4,
    }
    assert audit["correlation_bin_counts"] == {"0": 3, "1": 3, "2": 3, "3": 3}
    assert audit["raw_nodata"] == 65535.0
    assert audit["nodata_policy"] == "uint16_sentinel_65535_reject_invalid_v1"
    assert len(audit["pairs"]) == 12
    assert audit["pairs"][0]["crop_bounds"] == [
        500010.0,
        398710.0,
        501290.0,
        399990.0,
    ]
    assert audit["repeated_load_equal"] is True
    assert audit["model_inference_run"] is False
    assert audit["gpu_used"] is False
    assert audit["real_pixels_local"] is False


def test_input_audit_rejects_reordered_or_changed_second_load() -> None:
    first = _loaded_pairs()
    with pytest.raises(ValueError, match="repeated load"):
        build_input_audit(first, tuple(reversed(_loaded_pairs())))

    second = list(_loaded_pairs())
    changed_hr = second[0].pair.hr.clone()
    changed_hr[0, 0, 0] = 0.75
    second[0] = replace(second[0], pair=replace(second[0].pair, hr=changed_hr))
    with pytest.raises(ValueError, match="repeated load"):
        build_input_audit(first, second)


def test_input_audit_rejects_duplicate_identity_or_wrong_stratum_counts() -> None:
    duplicate = list(_loaded_pairs())
    duplicate[1] = replace(
        duplicate[1],
        pair=replace(duplicate[1].pair, sample_id=duplicate[0].pair.sample_id),
        metadata=replace(duplicate[1].metadata, sample_id=duplicate[0].metadata.sample_id),
    )
    with pytest.raises(ValueError, match="unique sample"):
        build_input_audit(duplicate, duplicate)

    wrong_bin = list(_loaded_pairs())
    wrong_bin[0] = replace(
        wrong_bin[0],
        metadata=replace(wrong_bin[0].metadata, correlation_bin=1),
    )
    with pytest.raises(ValueError, match="correlation-bin counts"):
        build_input_audit(wrong_bin, wrong_bin)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("crop_policy", "different", "crop policy"),
        ("normalization_policy", "different", "normalization policy"),
        ("selection_round", 2, "round one"),
        ("days_between", 0, "days_between=-1"),
    ],
)
def test_input_audit_rejects_non_frozen_metadata_policy(
    field: str, value: object, message: str
) -> None:
    pairs = list(_loaded_pairs())
    pairs[0] = replace(
        pairs[0],
        metadata=replace(pairs[0].metadata, **{field: value}),
    )

    with pytest.raises(ValueError, match=message):
        build_input_audit(pairs, pairs)
