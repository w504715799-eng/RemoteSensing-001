"""Metadata-only tests for the Phase 2B3-B calibration radiometry receipt."""

from __future__ import annotations

import importlib
from dataclasses import replace
from types import ModuleType

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
from trustsr.jsonio import canonical_json


def _module() -> ModuleType:
    return importlib.import_module("trustsr.evaluation.calibration_radiometry")


def _pair(
    *,
    sample_id: str,
    day: int,
    bin_index: int,
    round_index: int,
    lr_saturation: RadiometricSaturation,
    hr_saturation: RadiometricSaturation,
) -> LoadedCrosssensorPair:
    return LoadedCrosssensorPair(
        pair=SRPair(
            sample_id=sample_id,
            source=f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}",
            lr=torch.zeros((4, 1, 1), dtype=torch.float32),
            hr=torch.zeros((4, 4, 4), dtype=torch.float32),
            scale=4,
        ),
        metadata=CrosssensorPairMetadata(
            manifest_sha256=POST_MANIFEST_SHA256,
            sample_id=sample_id,
            split="calibration",
            spatial_group_id=f"group-{sample_id}",
            days_between=day,
            correlation_bin=bin_index,
            selection_round=round_index,
            lr_asset_sha256="a" * 64,
            hr_asset_sha256="b" * 64,
            lr_crop_transform=(10.0, 0.0, 10.0, 0.0, -10.0, -10.0),
            hr_crop_transform=(2.5, 0.0, 10.0, 0.0, -2.5, -10.0),
            crop_bounds=(10.0, -30.0, 20.0, -20.0),
            crop_policy=CROP_POLICY,
            normalization_policy=PHASE2B3A_NORMALIZATION_POLICY,
            lr_saturation=lr_saturation,
            hr_saturation=hr_saturation,
        ),
    )


def _pairs() -> tuple[LoadedCrosssensorPair, ...]:
    pairs: list[LoadedCrosssensorPair] = []
    for index, (day, bin_index, round_index) in enumerate(
        (day, bin_index, round_index)
        for day in (-1, 0, 1)
        for bin_index in range(4)
        for round_index in range(1, 11)
    ):
        if index == 0:
            lr = RadiometricSaturation(208, 11968, 8, (4, 0, 0, 4))
            hr = RadiometricSaturation(208, 11968, 117, (56, 0, 0, 61))
        elif index == 1:
            lr = RadiometricSaturation(100, 10000, 0, (0, 0, 0, 0))
            hr = RadiometricSaturation(50, 9000, 0, (0, 0, 0, 0))
        else:
            lr = RadiometricSaturation(120, 9572, 0, (0, 0, 0, 0))
            hr = RadiometricSaturation(136, 9572, 0, (0, 0, 0, 0))
        pairs.append(
            _pair(
                sample_id=f"calibration-{index:03d}",
                day=day,
                bin_index=bin_index,
                round_index=round_index,
                lr_saturation=lr,
                hr_saturation=hr,
            )
        )
    return tuple(pairs)


def _forged_saturation(
    *,
    minimum: object = 0,
    maximum: object = 10000,
    clipped: object = 0,
    by_band: object = (0, 0, 0, 0),
) -> RadiometricSaturation:
    value = object.__new__(RadiometricSaturation)
    object.__setattr__(value, "raw_crop_minimum", minimum)
    object.__setattr__(value, "raw_crop_maximum", maximum)
    object.__setattr__(value, "clipped_high_count", clipped)
    object.__setattr__(value, "clipped_high_by_band", by_band)
    return value


def test_builds_hand_checked_host_free_radiometric_receipt() -> None:
    result = _module().build_calibration_radiometry(_pairs())

    assert set(result) == {
        "schema",
        "policy",
        "sample_count",
        "affected_sample_count",
        "lr",
        "hr",
        "samples",
    }
    assert result["schema"] == "trustsr.phase2b3b-calibration-radiometry.v1"
    assert result["policy"] == {
        "normalization_policy": "uint16_saturate_10000_divide_10000_v2",
        "raw_radiometric_max": 32767,
        "saturation_threshold": 10000,
        "saturation_operation": "minimum(raw,10000)",
        "saturation_scope": "aligned_crop_only",
        "reflectance_divisor": 10000.0,
        "crop_policy": "center_crop_lr_1_hr_4_v1",
        "bands": ["B04", "B03", "B02", "B08"],
    }
    assert result["sample_count"] == 120
    assert result["affected_sample_count"] == 1
    assert result["lr"] == {
        "raw_crop_minimum": 100,
        "raw_crop_maximum": 11968,
        "clipped_high_count": 8,
        "clipped_high_by_band": [4, 0, 0, 4],
    }
    assert result["hr"] == {
        "raw_crop_minimum": 50,
        "raw_crop_maximum": 11968,
        "clipped_high_count": 117,
        "clipped_high_by_band": [56, 0, 0, 61],
    }
    assert [sample["sample_id"] for sample in result["samples"]] == [
        pair.pair.sample_id for pair in _pairs()
    ]
    assert result["samples"][0] == {
        "sample_id": "calibration-000",
        "days_between": -1,
        "correlation_bin": 0,
        "selection_round": 1,
        "radiometric_saturation": {
            "lr": {
                "raw_crop_minimum": 208,
                "raw_crop_maximum": 11968,
                "clipped_high_count": 8,
                "clipped_high_by_band": [4, 0, 0, 4],
            },
            "hr": {
                "raw_crop_minimum": 208,
                "raw_crop_maximum": 11968,
                "clipped_high_count": 117,
                "clipped_high_by_band": [56, 0, 0, 61],
            },
        },
    }


def test_output_is_fresh_json_native_and_canonically_repeatable() -> None:
    module = _module()
    pairs = _pairs()

    first = module.build_calibration_radiometry(pairs)
    second = module.build_calibration_radiometry(pairs)

    assert canonical_json(first) == canonical_json(second)
    assert first is not second
    assert type(first) is dict
    assert type(first["policy"]) is dict
    assert type(first["policy"]["bands"]) is list
    assert type(first["samples"]) is list
    assert type(first["samples"][0]) is dict
    assert type(first["samples"][0]["radiometric_saturation"]["lr"]["clipped_high_by_band"]) is list
    first["samples"][0]["radiometric_saturation"]["lr"]["clipped_high_by_band"][0] = 99
    assert second["samples"][0]["radiometric_saturation"]["lr"]["clipped_high_by_band"][0] == 4


def test_receipt_contains_no_tensor_path_time_or_internal_test_data() -> None:
    payload = canonical_json(_module().build_calibration_radiometry(_pairs())).decode("utf-8")

    for forbidden in ("tensor", "asset", "path", "timestamp", "internal_test"):
        assert forbidden not in payload


def test_does_not_read_tensor_values(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("radiometry receipt read a tensor value")

    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)

    assert _module().build_calibration_radiometry(_pairs())["sample_count"] == 120


def test_rejects_119_or_duplicate_sample_ids() -> None:
    module = _module()
    pairs = list(_pairs())
    with pytest.raises(ValueError, match="120"):
        module.build_calibration_radiometry(pairs[:-1])

    duplicate = replace(
        pairs[-1],
        pair=replace(pairs[-1].pair, sample_id=pairs[0].pair.sample_id),
        metadata=replace(pairs[-1].metadata, sample_id=pairs[0].pair.sample_id),
    )
    pairs[-1] = duplicate
    with pytest.raises(ValueError, match="unique.*sample"):
        module.build_calibration_radiometry(pairs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("split", "development", "calibration"),
        ("split", "internal_test", "calibration"),
        ("manifest_sha256", "0" * 64, "manifest"),
        ("crop_policy", "wrong", "policy"),
        ("normalization_policy", "wrong", "policy"),
        ("days_between", 2, "stratum"),
        ("correlation_bin", 4, "stratum"),
        ("selection_round", 11, "stratum|round"),
    ),
)
def test_rejects_wrong_calibration_metadata_policy_or_stratum(
    field: str, value: object, message: str
) -> None:
    module = _module()
    pairs = list(_pairs())
    pairs[0] = replace(pairs[0], metadata=replace(pairs[0].metadata, **{field: value}))

    with pytest.raises(ValueError, match=message):
        module.build_calibration_radiometry(pairs)


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ("source", "source"),
        ("scale", "scale"),
        ("pair_sample", "identit"),
        ("duplicate_round", "round"),
    ),
)
def test_rejects_wrong_pair_identity_source_scale_or_incomplete_strata(
    change: str, message: str
) -> None:
    module = _module()
    pairs = list(_pairs())
    if change == "source":
        pairs[0] = replace(pairs[0], pair=replace(pairs[0].pair, source="wrong"))
    elif change == "scale":
        pairs[0] = replace(pairs[0], pair=replace(pairs[0].pair, scale=2))
    elif change == "pair_sample":
        pairs[0] = replace(pairs[0], pair=replace(pairs[0].pair, sample_id="wrong"))
    else:
        pairs[0] = replace(pairs[0], metadata=replace(pairs[0].metadata, selection_round=2))

    with pytest.raises(ValueError, match=message):
        module.build_calibration_radiometry(pairs)


@pytest.mark.parametrize(
    "saturation",
    (
        _forged_saturation(by_band=[0, 0, 0, 0]),
        _forged_saturation(maximum=10001, clipped=2, by_band=(1, 0, 0, 0)),
        _forged_saturation(maximum=32768, clipped=1, by_band=(1, 0, 0, 0)),
        _forged_saturation(maximum=10000, clipped=1, by_band=(1, 0, 0, 0)),
    ),
)
def test_reruns_radiometric_contract_against_forged_tuple_count_or_maximum(
    saturation: RadiometricSaturation,
) -> None:
    module = _module()
    pairs = list(_pairs())
    pairs[0] = replace(pairs[0], metadata=replace(pairs[0].metadata, lr_saturation=saturation))

    with pytest.raises((TypeError, ValueError), match="radiometric|saturation|maximum|band|count"):
        module.build_calibration_radiometry(pairs)
