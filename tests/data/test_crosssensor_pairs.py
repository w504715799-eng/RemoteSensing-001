"""Contracts for Phase 2B2-A crosssensor model inputs."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from affine import Affine

from trustsr.data import crosssensor_pairs
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    load_crosssensor_pair,
    load_crosssensor_records,
    select_development_smoke_records,
    select_input_smoke_records,
)

SPLITS = ("development", "calibration", "internal_test")


def _eligible_records() -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"sample-{split}-{bin_index}",
            "split": split,
            "spatial_group_id": f"group-{split}-{bin_index}",
            "days_between": -1,
            "correlation_bin": bin_index,
            "selection_round": 1,
        }
        for split in SPLITS
        for bin_index in range(4)
    ]


def test_smoke_selection_has_four_bins_per_split_in_canonical_order() -> None:
    records = _eligible_records()
    records.extend(
        {
            **deepcopy(records[0]),
            "sample_id": f"not-eligible-{index}",
            "spatial_group_id": f"not-eligible-group-{index}",
            "selection_round": 2,
        }
        for index in range(3)
    )

    selected = select_input_smoke_records(tuple(reversed(records)))

    assert [(record["split"], record["correlation_bin"]) for record in selected] == [
        (split, bin_index) for split in sorted(SPLITS) for bin_index in range(4)
    ]
    assert len({record["sample_id"] for record in selected}) == 12
    assert len({record["spatial_group_id"] for record in selected}) == 12


def test_smoke_selection_rejects_a_missing_or_duplicate_required_cell() -> None:
    records = _eligible_records()

    with pytest.raises(ValueError, match="exactly one record"):
        select_input_smoke_records(records[:-1])

    duplicate = records + [{**records[0], "sample_id": "duplicate"}]
    with pytest.raises(ValueError, match="exactly one record"):
        select_input_smoke_records(duplicate)


def test_development_smoke_selection_filters_after_canonical_selection() -> None:
    selected = select_development_smoke_records(_eligible_records())

    assert len(selected) == 4
    assert [record["split"] for record in selected] == ["development"] * 4
    assert [record["correlation_bin"] for record in selected] == [0, 1, 2, 3]


def test_development_smoke_selection_is_input_order_independent() -> None:
    records = _eligible_records()

    assert select_development_smoke_records(tuple(reversed(records))) == (
        select_development_smoke_records(records)
    )


def test_development_smoke_selection_rejects_unvalidated_four_row_shortcut() -> None:
    development_only = [
        record for record in _eligible_records() if record["split"] == "development"
    ]

    with pytest.raises(ValueError, match="exactly one record"):
        select_development_smoke_records(development_only)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sample_id", "", "sample_id"),
        ("sample_id", 3, "sample_id"),
        ("spatial_group_id", "", "spatial_group_id"),
        ("spatial_group_id", None, "spatial_group_id"),
    ],
)
def test_smoke_selection_rejects_invalid_or_duplicate_identities(
    field: str, value: object, message: str
) -> None:
    records = _eligible_records()
    records[0][field] = value

    with pytest.raises(ValueError, match=message):
        select_input_smoke_records(records)

    records = _eligible_records()
    records[1][field] = records[0][field]
    with pytest.raises(ValueError, match=f"unique {message}"):
        select_input_smoke_records(records)


def _post_records(*, all_assets: bool = True) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "sample_id": f"sample-{index:03d}",
            "lr_asset": {"sha256": "a" * 64} if all_assets else None,
            "hr_asset": {"sha256": "b" * 64} if all_assets else None,
        }
        for index in range(360)
    )


def _digest_manifest(tmp_path: Path, payload: bytes = b"post-manifest\n") -> tuple[Path, str]:
    digest = hashlib.sha256(payload).hexdigest()
    manifest = (
        tmp_path
        / "trustsr"
        / "phase2b1b"
        / "selections"
        / digest
        / "samples.jsonl"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(payload)
    return manifest, digest


def test_load_records_requires_digest_addressed_all_assets_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, digest = _digest_manifest(tmp_path)
    calls: list[tuple[Path, str]] = []

    def load(path: Path, *, expected_sha256: str) -> tuple[dict[str, object], ...]:
        calls.append((path, expected_sha256))
        return _post_records()

    monkeypatch.setattr(crosssensor_pairs, "POST_MANIFEST_SHA256", digest)
    monkeypatch.setattr(crosssensor_pairs, "load_subset_manifest", load)

    records = load_crosssensor_records(tmp_path, manifest, expected_sha256=digest)

    assert len(records) == 360
    assert calls == [(manifest.resolve(), digest)]


def test_load_records_rejects_wrong_digest_or_layout_before_schema_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, digest = _digest_manifest(tmp_path)
    calls: list[Path] = []

    def load(path: Path, *, expected_sha256: str) -> tuple[dict[str, object], ...]:
        calls.append(path)
        return _post_records()

    monkeypatch.setattr(crosssensor_pairs, "POST_MANIFEST_SHA256", digest)
    monkeypatch.setattr(crosssensor_pairs, "load_subset_manifest", load)
    misplaced = tmp_path / "misplaced.jsonl"
    misplaced.write_bytes(manifest.read_bytes())

    with pytest.raises(ValueError, match="frozen post-manifest"):
        load_crosssensor_records(tmp_path, misplaced, expected_sha256=digest)
    with pytest.raises(ValueError, match="frozen post-manifest SHA-256"):
        load_crosssensor_records(tmp_path, manifest, expected_sha256="0" * 64)
    assert calls == []


def test_load_records_rejects_null_assets_after_schema_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, digest = _digest_manifest(tmp_path)
    monkeypatch.setattr(crosssensor_pairs, "POST_MANIFEST_SHA256", digest)
    monkeypatch.setattr(
        crosssensor_pairs,
        "load_subset_manifest",
        lambda path, *, expected_sha256: _post_records(all_assets=False),
    )

    with pytest.raises(ValueError, match="all-assets post-manifest"):
        load_crosssensor_records(tmp_path, manifest, expected_sha256=digest)


def test_load_records_rejects_symlink_root_or_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    manifest, digest = _digest_manifest(real_root)
    monkeypatch.setattr(crosssensor_pairs, "POST_MANIFEST_SHA256", digest)
    monkeypatch.setattr(
        crosssensor_pairs,
        "load_subset_manifest",
        lambda path, *, expected_sha256: _post_records(),
    )
    root_link = tmp_path / "root-link"
    root_link.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="storage_root"):
        load_crosssensor_records(root_link, manifest, expected_sha256=digest)

    manifest_link = manifest.with_name("manifest-link.jsonl")
    manifest_link.symlink_to(manifest)
    with pytest.raises(ValueError, match="regular file"):
        load_crosssensor_records(real_root, manifest_link, expected_sha256=digest)


_BANDS = ("B04", "B03", "B02", "B08")
_FROZEN_NODATA = 65535
_LR_TRANSFORM = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 400000.0)
_HR_TRANSFORM = Affine(2.5, 0.0, 500000.0, 0.0, -2.5, 400000.0)


def _write_geotiff(
    path: Path,
    pixels: np.ndarray,
    transform: Affine,
    *,
    nodata: int | None = None,
    descriptions: tuple[str, ...] | None = _BANDS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=pixels.shape[2],
        height=pixels.shape[1],
        count=pixels.shape[0],
        dtype=pixels.dtype,
        crs="EPSG:32618",
        transform=transform,
        nodata=nodata,
    ) as dataset:
        dataset.write(pixels)
        if descriptions is not None:
            dataset.descriptions = descriptions


def _asset(path: Path, relative_path: str) -> dict[str, object]:
    with rasterio.open(path) as dataset:
        pixels = dataset.read()
        transform = dataset.transform
        return {
            "relative_path": relative_path,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": [dataset.count, dataset.height, dataset.width],
            "dtype": dataset.dtypes[0],
            "crs": dataset.crs.to_string(),
            "transform": [
                float(transform.a),
                float(transform.b),
                float(transform.c),
                float(transform.d),
                float(transform.e),
                float(transform.f),
            ],
            "nodata": dataset.nodata,
            "minimum": float(pixels.min()),
            "maximum": float(pixels.max()),
            "time_start": "2020-01-02T10:00:00Z",
        }


def _real_pair_fixture(tmp_path: Path, damage: str | None = None) -> dict[str, object]:
    sample_id = "sample-development-0"
    prefix = f"subset-v1/development/{sample_id}"
    pair_root = tmp_path / "trustsr" / "phase2b1b" / prefix
    dtype = np.int16 if damage == "dtype" else np.uint16
    lr = np.full((4, 130, 130), 5000, dtype=dtype)
    hr = np.full((4, 520, 520), 5000, dtype=dtype)
    lr[0, 0, 0] = 9999
    lr[0, 1, 1] = 1234
    hr[0, 0, 0] = 9999
    hr[0, 4, 4] = 1234
    if damage == "maximum":
        lr[0, 2, 2] = 10001
    if damage == "band-count":
        lr = lr[:3]
    nodata = (
        -32768
        if damage == "dtype"
        else 0
        if damage == "nodata"
        else _FROZEN_NODATA
    )
    if damage == "nodata-pixel":
        lr[0, 2, 2] = _FROZEN_NODATA
    descriptions = (
        ("WRONG", "B03", "B02", "B08")
        if damage == "descriptions"
        else _BANDS[: lr.shape[0]]
    )
    hr_transform = (
        Affine(2.5, 0.0, 500001.0, 0.0, -2.5, 400000.0)
        if damage == "bounds"
        else _HR_TRANSFORM
    )
    lr_path = pair_root / "lr.tif"
    hr_path = pair_root / "hr.tif"
    _write_geotiff(lr_path, lr, _LR_TRANSFORM, nodata=nodata, descriptions=descriptions)
    _write_geotiff(hr_path, hr, hr_transform, nodata=nodata)
    record: dict[str, object] = {
        "sample_id": sample_id,
        "split": "development",
        "spatial_group_id": "a" * 64,
        "days_between": -1,
        "correlation_bin": 0,
        "selection_round": 1,
        "lr_asset": _asset(lr_path, f"{prefix}/lr.tif"),
        "hr_asset": _asset(hr_path, f"{prefix}/hr.tif"),
    }
    if damage == "hash":
        record["lr_asset"]["sha256"] = "0" * 64  # type: ignore[index]
    elif damage == "transform":
        record["lr_asset"]["transform"][0] = 11.0  # type: ignore[index]
    elif damage == "minimum":
        record["lr_asset"]["minimum"] = 1.0  # type: ignore[index]
    elif damage == "path":
        record["lr_asset"]["relative_path"] = "../lr.tif"  # type: ignore[index]
    elif damage == "symlink":
        original = lr_path.with_name("original-lr.tif")
        lr_path.rename(original)
        lr_path.symlink_to(original)
    return record


def test_load_pair_center_crops_aligns_and_normalizes_without_clipping(
    tmp_path: Path,
) -> None:
    record = _real_pair_fixture(tmp_path)

    loaded = load_crosssensor_pair(
        tmp_path,
        record,
        manifest_sha256=POST_MANIFEST_SHA256,
    )

    assert loaded.pair.lr.shape == (4, 128, 128)
    assert loaded.pair.hr.shape == (4, 512, 512)
    assert loaded.pair.lr.dtype == torch.float32
    assert loaded.pair.hr.dtype == torch.float32
    assert loaded.pair.lr.device.type == "cpu"
    assert loaded.pair.lr.is_contiguous()
    assert loaded.pair.hr.is_contiguous()
    assert loaded.pair.lr[0, 0, 0].item() == pytest.approx(0.1234)
    assert loaded.pair.hr[0, 0, 0].item() == pytest.approx(0.1234)
    assert loaded.pair.lr.max().item() == pytest.approx(0.5)
    assert loaded.pair.hr.max().item() == pytest.approx(0.5)
    loaded.pair.validate()
    assert loaded.metadata.crop_policy == CROP_POLICY
    assert loaded.metadata.normalization_policy == NORMALIZATION_POLICY
    assert loaded.metadata.lr_crop_transform == (10.0, 0.0, 500010.0, 0.0, -10.0, 399990.0)
    assert loaded.metadata.hr_crop_transform == (2.5, 0.0, 500010.0, 0.0, -2.5, 399990.0)
    assert loaded.metadata.crop_bounds == (500010.0, 398710.0, 501290.0, 399990.0)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("hash", "asset bytes"),
        ("symlink", "regular GeoTIFF"),
        ("dtype", "uint16"),
        ("nodata", "65535"),
        ("nodata-pixel", "invalid nodata pixels"),
        ("maximum", r"\[0, 10000\]"),
        ("transform", "metadata"),
        ("bounds", "cropped LR/HR bounds"),
        ("band-count", "four bands"),
        ("descriptions", "band descriptions"),
        ("minimum", "metadata"),
        ("path", "canonical Phase 2B1B path"),
    ],
)
def test_load_pair_rejects_integrity_or_reflectance_contract_violation(
    tmp_path: Path, damage: str, message: str
) -> None:
    record = _real_pair_fixture(tmp_path, damage)

    with pytest.raises(ValueError, match=message):
        load_crosssensor_pair(
            tmp_path,
            record,
            manifest_sha256=POST_MANIFEST_SHA256,
        )


def test_load_pair_rejects_wrong_manifest_digest_before_reading(tmp_path: Path) -> None:
    record = _real_pair_fixture(tmp_path)

    with pytest.raises(ValueError, match="frozen post-manifest SHA-256"):
        load_crosssensor_pair(tmp_path, record, manifest_sha256="0" * 64)
