"""Calibration-only pixel-loading boundary contracts with synthetic pairs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from trustsr.contracts import SRPair
from trustsr.data.calibration_pairs import (
    load_calibration_pairs,
    validate_calibration_records,
)
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
    RadiometricSaturation,
)


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "sample_id": f"calibration-{index:03d}",
            "selection_sha256": f"{index:064x}",
            "spatial_group_id": f"group-{index:03d}",
            "split": "calibration",
            "days_between": day,
            "correlation_bin": bin_index,
            "selection_round": round_index,
            "lr_asset": {"asset": "lr"},
            "hr_asset": {"asset": "hr"},
        }
        for index, (day, bin_index, round_index) in enumerate(
            (day, bin_index, round_index)
            for day in (-1, 0, 1)
            for bin_index in range(4)
            for round_index in range(1, 11)
        )
    )


def _loaded(record: dict[str, object]) -> LoadedCrosssensorPair:
    sample_id = record["sample_id"]
    assert isinstance(sample_id, str)
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
            split="calibration",
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


def test_validate_calibration_records_requires_exact_frozen_ordered_120() -> None:
    records = _records()

    validated = validate_calibration_records(records)

    assert validated == records
    assert [record["sample_id"] for record in validated] == [
        record["sample_id"] for record in records
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split", "development", "calibration"),
        ("split", "internal_test", "calibration"),
        ("sample_id", "calibration-000", "unique"),
        ("lr_asset", None, "asset"),
        ("selection_round", 11, "round"),
    ],
)
def test_hostile_record_fails_before_any_pixel_loader_call(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    records = list(_records())
    records[-1] = {**records[-1], field: value}
    calls: list[object] = []

    def forbidden_loader(*args: object, **kwargs: object) -> LoadedCrosssensorPair:
        calls.append((args, kwargs))
        raise AssertionError("invalid metadata reached the pixel boundary")

    with pytest.raises(ValueError, match=message):
        load_calibration_pairs(tmp_path, records, pair_loader=forbidden_loader)
    assert calls == []


def test_loader_starts_after_full_validation_in_input_order_with_fixed_context(
    tmp_path: Path,
) -> None:
    records = _records()
    calls: list[str] = []

    def fake_loader(
        storage_root: Path,
        record: dict[str, object],
        *,
        manifest_sha256: str,
        normalization_policy: str,
    ) -> LoadedCrosssensorPair:
        assert storage_root == tmp_path
        assert manifest_sha256 == POST_MANIFEST_SHA256
        assert normalization_policy == PHASE2B3A_NORMALIZATION_POLICY
        calls.append(str(record["sample_id"]))
        return _loaded(record)

    loaded = load_calibration_pairs(tmp_path, records, pair_loader=fake_loader)

    assert calls == [record["sample_id"] for record in records]
    assert [item.pair.sample_id for item in loaded] == calls
    assert all(item.metadata.split == "calibration" for item in loaded)


@pytest.mark.parametrize("wrong", ["split", "manifest", "normalization", "saturation", "order"])
def test_rejects_forged_or_wrong_order_loader_output(tmp_path: Path, wrong: str) -> None:
    records = _records()

    def fake_loader(
        _storage_root: Path,
        record: dict[str, object],
        **_kwargs: object,
    ) -> LoadedCrosssensorPair:
        loaded = _loaded(record)
        if wrong == "split":
            return replace(loaded, metadata=replace(loaded.metadata, split="development"))
        if wrong == "manifest":
            return replace(loaded, metadata=replace(loaded.metadata, manifest_sha256="0" * 64))
        if wrong == "normalization":
            return replace(loaded, metadata=replace(loaded.metadata, normalization_policy="wrong"))
        if wrong == "saturation":
            return replace(loaded, metadata=replace(loaded.metadata, lr_saturation=None))
        if record is records[-1]:
            return _loaded(records[0])
        return loaded

    with pytest.raises(ValueError, match="loader|calibration|manifest|policy|saturation|order"):
        load_calibration_pairs(tmp_path, records, pair_loader=fake_loader)
