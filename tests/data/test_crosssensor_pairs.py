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
    select_development_records,
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


def _complete_research_records() -> tuple[dict[str, object], ...]:
    rows = []
    for split in ("development", "calibration", "internal_test"):
        for days_between in (-1, 0, 1):
            for bin_index in range(4):
                for selection_round in range(1, 11):
                    sample_id = f"{split}-{days_between}-{bin_index}-{selection_round}"
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "split": split,
                            "spatial_group_id": f"group-{sample_id}",
                            "days_between": days_between,
                            "correlation_bin": bin_index,
                            "selection_round": selection_round,
                        }
                    )
    return tuple(rows)


def test_select_development_records_preserves_manifest_order_and_strata() -> None:
    records = _complete_research_records()
    selected = select_development_records(records)
    assert len(selected) == 120
    assert selected == tuple(row for row in records if row["split"] == "development")
    assert len({row["spatial_group_id"] for row in selected}) == 120


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate_sample", "duplicate_group", "bad_day", "bad_bin", "bad_round"],
)
def test_select_development_records_rejects_broken_frozen_cells(mutation: str) -> None:
    records = [dict(row) for row in _complete_research_records()]
    development_index = next(
        i for i, row in enumerate(records) if row["split"] == "development"
    )
    if mutation == "missing":
        records.pop(development_index)
    elif mutation == "duplicate_sample":
        records[development_index + 1]["sample_id"] = records[development_index]["sample_id"]
    elif mutation == "duplicate_group":
        records[development_index + 1]["spatial_group_id"] = records[development_index][
            "spatial_group_id"
        ]
    elif mutation == "bad_day":
        records[development_index]["days_between"] = 2
    elif mutation == "bad_bin":
        records[development_index]["correlation_bin"] = 4
    else:
        records[development_index]["selection_round"] = 11
    with pytest.raises(ValueError, match="development"):
        select_development_records(records)


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


def _saturation_pair_fixture(
    tmp_path: Path, *, raw_out_of_range: bool = False
) -> dict[str, object]:
    """Build a real aligned pair with crop-local B04/B08 saturation."""

    sample_id = "sample-development-saturation"
    prefix = f"subset-v1/development/{sample_id}"
    pair_root = tmp_path / "trustsr" / "phase2b1b" / prefix
    lr = np.full((4, 130, 130), 5000, dtype=np.uint16)
    hr = np.full((4, 520, 520), 5000, dtype=np.uint16)

    # (0, 0) is deliberately outside both aligned crops.
    lr[0, 0, 0] = 11968
    hr[0, 0, 0] = 11968
    if raw_out_of_range:
        lr[0, 0, 0] = 32768

    # The aligned LR crop starts at (1, 1), HR at (4, 4).  The saturation
    # counts are deliberately asymmetric in every ordered B04/B03/B02/B08 band.
    lr[0, 1, 1] = 10001
    lr[0, 2, 2] = 11288
    lr[1, 4, 4] = 11968
    lr[2, 5, 5] = 10001
    lr[2, 6, 6] = 11288
    lr[2, 7, 7] = 11968
    lr[3, 3, 3] = 11968
    lr[3, 4, 4] = 10001
    lr[3, 5, 5] = 11288
    lr[3, 6, 6] = 11968
    hr[0, 4, 4] = 10001
    hr[0, 5, 5] = 11288
    hr[1, 7, 7] = 11968
    hr[2, 8, 8] = 10001
    hr[2, 9, 9] = 11288
    hr[2, 10, 10] = 11968
    hr[3, 6, 6] = 11968
    hr[3, 7, 7] = 10001
    hr[3, 8, 8] = 11288
    hr[3, 9, 9] = 11968

    lr_path = pair_root / "lr.tif"
    hr_path = pair_root / "hr.tif"
    _write_geotiff(lr_path, lr, _LR_TRANSFORM, nodata=_FROZEN_NODATA)
    _write_geotiff(hr_path, hr, _HR_TRANSFORM, nodata=_FROZEN_NODATA)
    return {
        "sample_id": sample_id,
        "split": "development",
        "spatial_group_id": "b" * 64,
        "days_between": -1,
        "correlation_bin": 0,
        "selection_round": 1,
        "lr_asset": _asset(lr_path, f"{prefix}/lr.tif"),
        "hr_asset": _asset(hr_path, f"{prefix}/hr.tif"),
    }


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


def test_v2_saturates_only_aligned_crops_and_records_ordered_band_statistics(
    tmp_path: Path,
) -> None:
    """Fails if v2 is removed, clips the wrong window, or reorders B04/B03/B02/B08."""

    record = _saturation_pair_fixture(tmp_path)

    loaded = load_crosssensor_pair(
        tmp_path,
        record,
        manifest_sha256=POST_MANIFEST_SHA256,
        normalization_policy=crosssensor_pairs.PHASE2B3A_NORMALIZATION_POLICY,
    )

    assert loaded.metadata.normalization_policy == "uint16_saturate_10000_divide_10000_v2"
    assert loaded.pair.lr[0, 0, 0].item() == pytest.approx(1.0)
    assert loaded.pair.lr[0, 1, 1].item() == pytest.approx(1.0)
    assert loaded.pair.lr[3, 2, 2].item() == pytest.approx(1.0)
    assert loaded.pair.lr[1, 0, 0].item() == pytest.approx(0.5)
    assert loaded.pair.hr[0, 0, 0].item() == pytest.approx(1.0)
    assert loaded.pair.hr[0, 1, 1].item() == pytest.approx(1.0)
    assert loaded.pair.hr[3, 2, 2].item() == pytest.approx(1.0)
    assert loaded.pair.hr[1, 0, 0].item() == pytest.approx(0.5)
    assert loaded.metadata.lr_saturation.raw_crop_minimum == 5000
    assert loaded.metadata.lr_saturation.raw_crop_maximum == 11968
    assert loaded.metadata.lr_saturation.clipped_high_count == 10
    assert loaded.metadata.lr_saturation.clipped_high_by_band == (2, 1, 3, 4)
    assert loaded.metadata.hr_saturation.raw_crop_minimum == 5000
    assert loaded.metadata.hr_saturation.raw_crop_maximum == 11968
    assert loaded.metadata.hr_saturation.clipped_high_count == 10
    assert loaded.metadata.hr_saturation.clipped_high_by_band == (2, 1, 3, 4)


def test_v2_preserves_raw_geotiff_pixels_while_saturating_tensor_copy(tmp_path: Path) -> None:
    """Fails if v2 clips the source array before the crop tensor is copied."""

    record = _saturation_pair_fixture(tmp_path)

    load_crosssensor_pair(
        tmp_path,
        record,
        manifest_sha256=POST_MANIFEST_SHA256,
        normalization_policy=crosssensor_pairs.PHASE2B3A_NORMALIZATION_POLICY,
    )

    for asset_key, index, expected in (
        ("lr_asset", (0, 1, 1), 10001),
        ("hr_asset", (3, 6, 6), 11968),
    ):
        relative_path = record[asset_key]["relative_path"]  # type: ignore[index]
        with rasterio.open(tmp_path / "trustsr" / "phase2b1b" / relative_path) as dataset:
            assert int(dataset.read()[index]) == expected


def test_v2_crop_transform_does_not_mutate_its_owned_source_array() -> None:
    """Fails if v2 clips in place with ``out=crop`` rather than creating a copy."""

    source = np.array(
        [
            [[5000, 10001], [5000, 5000]],
            [[5000, 11288], [5000, 5000]],
            [[5000, 11968], [5000, 5000]],
            [[5000, 10001], [5000, 5000]],
        ],
        dtype=np.uint16,
    )
    original = source.copy()

    saturated, _ = crosssensor_pairs._saturate_crop_v2(source)

    assert source.tobytes() == original.tobytes()
    assert saturated is not source
    assert saturated[:, 0, 1].tolist() == [10000, 10000, 10000, 10000]


def test_legacy_v1_rejects_saturated_fixture_by_default(tmp_path: Path) -> None:
    """Fails if the historical default stops rejecting raw values above 10000."""

    record = _saturation_pair_fixture(tmp_path)

    with pytest.raises(ValueError, match=r"\[0, 10000\]"):
        load_crosssensor_pair(tmp_path, record, manifest_sha256=POST_MANIFEST_SHA256)


def test_v2_rejects_full_raw_raster_value_above_32767_even_outside_crop(
    tmp_path: Path,
) -> None:
    """Fails if v2 accepts 32768 or validates only the aligned crop."""

    record = _saturation_pair_fixture(tmp_path, raw_out_of_range=True)

    with pytest.raises(ValueError, match="32767"):
        load_crosssensor_pair(
            tmp_path,
            record,
            manifest_sha256=POST_MANIFEST_SHA256,
            normalization_policy=crosssensor_pairs.PHASE2B3A_NORMALIZATION_POLICY,
        )


def test_loader_rejects_unknown_normalization_policy(tmp_path: Path) -> None:
    """Fails if a typo can silently select a normalization branch."""

    record = _real_pair_fixture(tmp_path)

    with pytest.raises(ValueError, match="normalization policy"):
        load_crosssensor_pair(
            tmp_path,
            record,
            manifest_sha256=POST_MANIFEST_SHA256,
            normalization_policy="unknown-policy",
        )


@pytest.mark.parametrize("invalid_count", [True, np.int64(3)])
def test_radiometric_saturation_rejects_boolean_or_numpy_statistics(
    invalid_count: object,
) -> None:
    """Fails if provenance can contain non-built-in integer statistics."""

    with pytest.raises(TypeError, match="built-in integers"):
        crosssensor_pairs.RadiometricSaturation(
            raw_crop_minimum=5000,
            raw_crop_maximum=11968,
            clipped_high_count=invalid_count,
            clipped_high_by_band=(2, 0, 0, 1),
        )


class _TupleSubclass(tuple):
    pass


@pytest.mark.parametrize(
    "band_counts",
    [[0, 0, 0, 0], True, _TupleSubclass((0, 0, 0, 0))],
)
def test_radiometric_saturation_requires_an_exact_band_count_tuple(
    band_counts: object,
) -> None:
    """Fails if mutable or subclassed band-count containers enter provenance."""

    with pytest.raises(TypeError, match="exact tuple"):
        crosssensor_pairs.RadiometricSaturation(
            raw_crop_minimum=5000,
            raw_crop_maximum=10000,
            clipped_high_count=0,
            clipped_high_by_band=band_counts,
        )


@pytest.mark.parametrize(
    ("maximum", "clipped"),
    [(10000, 1), (10001, 0)],
)
def test_radiometric_saturation_rejects_threshold_mismatches(
    maximum: int, clipped: int
) -> None:
    """Fails if a constructed record can disagree about clipping at 10000."""

    with pytest.raises(ValueError, match="maximum and clipped count are inconsistent"):
        crosssensor_pairs.RadiometricSaturation(
            raw_crop_minimum=5000,
            raw_crop_maximum=maximum,
            clipped_high_count=clipped,
            clipped_high_by_band=(clipped, 0, 0, 0),
        )
