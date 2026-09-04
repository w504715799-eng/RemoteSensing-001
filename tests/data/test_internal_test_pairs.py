"""Permit-gated Phase 2B3-C pair loading with synthetic tensors only."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from trustsr.contracts import SRPair
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
    RadiometricSaturation,
)
from trustsr.data.internal_test_pairs import (
    _issue_pixels_opened_access_guard,
    _load_internal_test_pair,
    load_internal_test_pairs,
    validate_internal_test_records,
)


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


def _guard():
    return _issue_pixels_opened_access_guard("e" * 64, "a" * 64)


def _write_asset(path: Path, kind: str) -> dict[str, object]:
    size = 130 if kind == "lr" else 520
    resolution = 10.0 if kind == "lr" else 2.5
    value = 2500 if kind == "lr" else 5000
    transform = from_origin(0.0, 0.0, resolution, resolution)
    pixels = np.full((4, size, size), value, dtype=np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=4,
        dtype="uint16",
        crs="EPSG:32618",
        transform=transform,
        nodata=65535,
    ) as dataset:
        dataset.write(pixels)
    payload = path.read_bytes()
    return {
        "relative_path": (
            f"subset-v1/internal_test/internal-test-000/{kind}.tif"
        ),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "shape": [4, size, size],
        "dtype": "uint16",
        "crs": "EPSG:32618",
        "transform": [resolution, 0.0, 0.0, 0.0, -resolution, 0.0],
        "nodata": 65535.0,
        "minimum": float(value),
        "maximum": float(value),
    }


def test_validates_exact_internal_test_design() -> None:
    assert validate_internal_test_records(_records()) == _records()


@pytest.mark.parametrize("guard", (None, object()))
def test_rejects_missing_or_forged_guard_before_loader(
    tmp_path: Path, guard: object
) -> None:
    calls: list[object] = []

    def forbidden(*args: object, **kwargs: object) -> LoadedCrosssensorPair:
        calls.append((args, kwargs))
        raise AssertionError("pixel loader crossed the access boundary")

    with pytest.raises(ValueError, match="pixels_opened access guard"):
        load_internal_test_pairs(tmp_path, _records(), guard, pair_loader=forbidden)
    assert calls == []


def test_invalid_metadata_fails_before_loader_even_with_guard(tmp_path: Path) -> None:
    records = list(_records())
    records[-1] = {**records[-1], "split": "calibration"}
    calls: list[object] = []

    def forbidden(*args: object, **kwargs: object) -> LoadedCrosssensorPair:
        calls.append((args, kwargs))
        raise AssertionError("invalid metadata reached pixels")

    with pytest.raises(ValueError, match="internal_test"):
        load_internal_test_pairs(tmp_path, records, _guard(), pair_loader=forbidden)
    assert calls == []


def test_loads_in_order_only_after_full_validation(tmp_path: Path) -> None:
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

    loaded = load_internal_test_pairs(
        tmp_path, records, _guard(), pair_loader=fake_loader
    )

    assert calls == [str(record["sample_id"]) for record in records]
    assert [item.pair.sample_id for item in loaded] == calls


def test_default_asset_reader_uses_confined_nofollow_descriptors(tmp_path: Path) -> None:
    sample_root = (
        tmp_path / "trustsr" / "phase2b1b" / "subset-v1" / "internal_test"
        / "internal-test-000"
    )
    record = dict(_records()[0])
    record["lr_asset"] = _write_asset(sample_root / "lr.tif", "lr")
    record["hr_asset"] = _write_asset(sample_root / "hr.tif", "hr")

    loaded = _load_internal_test_pair(
        tmp_path,
        record,
        manifest_sha256=POST_MANIFEST_SHA256,
        normalization_policy=PHASE2B3A_NORMALIZATION_POLICY,
    )

    assert loaded.pair.lr.shape == (4, 128, 128)
    assert loaded.pair.hr.shape == (4, 512, 512)
    outside = tmp_path / "outside.tif"
    outside.write_bytes((sample_root / "lr.tif").read_bytes())
    (sample_root / "lr.tif").unlink()
    (sample_root / "lr.tif").symlink_to(outside)
    with pytest.raises(ValueError, match="non-symlink"):
        _load_internal_test_pair(
            tmp_path,
            record,
            manifest_sha256=POST_MANIFEST_SHA256,
            normalization_policy=PHASE2B3A_NORMALIZATION_POLICY,
        )


@pytest.mark.parametrize("wrong", ("split", "order", "asset", "saturation"))
def test_rejects_forged_loader_output(tmp_path: Path, wrong: str) -> None:
    records = _records()

    def fake_loader(
        _root: Path, record: dict[str, object], **_kwargs: object
    ) -> LoadedCrosssensorPair:
        loaded = _loaded(record)
        if wrong == "split":
            return replace(loaded, metadata=replace(loaded.metadata, split="calibration"))
        if wrong == "asset":
            return replace(
                loaded, metadata=replace(loaded.metadata, lr_asset_sha256="0" * 64)
            )
        if wrong == "saturation":
            return replace(loaded, metadata=replace(loaded.metadata, lr_saturation=None))
        if record is records[-1]:
            return _loaded(records[0])
        return loaded

    with pytest.raises(ValueError, match="loader|internal_test|asset|saturation|order"):
        load_internal_test_pairs(tmp_path, records, _guard(), pair_loader=fake_loader)
