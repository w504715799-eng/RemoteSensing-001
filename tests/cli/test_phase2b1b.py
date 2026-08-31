"""Tests for restartable Phase 2B1B cloud data stages."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from trustsr.cli import phase2b1b
from trustsr.data.crosssensor_manifest import ExtractedAsset, load_manifest, write_manifest
from trustsr.data.crosssensor_schema import CrosssensorSample
from trustsr.data.crosssensor_source import VerifiedSourceObject
from trustsr.data.pilot_sampling import select_pilot
from trustsr.data.spatial_split import AssignedSample
from trustsr.data.subset_manifest import (
    BASE_MANIFEST_SHA256,
    load_subset_manifest,
    select_from_base_manifest,
    write_subset_manifest,
)

REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "artifacts" / "datasets" / "sen2naipv2-source-v1.json"


def _assignments() -> tuple[AssignedSample, ...]:
    correlations = (0.8, 0.89, 0.91, 0.94)
    assignments: list[AssignedSample] = []
    for split in ("development", "calibration", "internal_test"):
        for days_between in (-1, 0, 1):
            for bin_index, correlation in enumerate(correlations):
                for rank in range(10):
                    sample_id = f"{split}-{days_between}-{bin_index}-{rank}"
                    lr_time_start = {
                        -1: "2020-01-03T10:00:00Z",
                        0: "2020-01-02T10:00:00Z",
                        1: "2020-01-01T10:00:00Z",
                    }[days_between]
                    assignments.append(
                        AssignedSample(
                            sample=CrosssensorSample(
                                source_index=len(assignments),
                                sample_id=sample_id,
                                longitude=-76.5 + len(assignments) * 0.001,
                                latitude=3.5,
                                crs="EPSG:32618",
                                geotransform=(
                                    10.0,
                                    0.0,
                                    500000.0,
                                    0.0,
                                    -10.0,
                                    400000.0,
                                ),
                                raster_shape=(130, 130),
                                time_start="2020-01-02T10:00:00Z",
                                lr_time_start=lr_time_start,
                                hr_time_start="2020-01-02T10:00:00Z",
                                admin0="Colombia",
                                admin1=None,
                                admin2="Cali",
                                days_between=days_between,
                                correlation=correlation,
                                scale_factor=4,
                            ),
                            spatial_group_id=hashlib.sha256(
                                f"group-{sample_id}".encode()
                            ).hexdigest(),
                            split=split,  # type: ignore[arg-type]
                        )
                    )
    return tuple(assignments)


def _base_records(tmp_path: Path) -> tuple[dict, ...]:
    assignments = _assignments()
    path = tmp_path / "fixture-base.jsonl"
    artifact = write_manifest(path, assignments, select_pilot(assignments), {})
    return load_manifest(path, expected_sha256=artifact.sha256)


def _base_path(storage_root: Path) -> Path:
    return (
        storage_root
        / "trustsr"
        / "phase2b1a"
        / "manifests"
        / BASE_MANIFEST_SHA256
        / "samples.jsonl"
    )


def _patch_verified_source(monkeypatch: pytest.MonkeyPatch, storage_root: Path) -> None:
    source_path = (
        storage_root
        / "trustsr"
        / "phase2b1a"
        / "source"
        / phase2b1b.SOURCE_OBJECT_SHA256
        / phase2b1b.SOURCE_OBJECT_NAME
    )
    monkeypatch.setattr(
        phase2b1b,
        "verify_crosssensor",
        lambda path, spec: VerifiedSourceObject(
            path=source_path,
            size_bytes=phase2b1b.SOURCE_OBJECT_SIZE_BYTES,
            sha256=phase2b1b.SOURCE_OBJECT_SHA256,
        ),
    )


def _pre_selection(
    storage_root: Path, base_records: tuple[dict, ...]
) -> tuple[Path, str]:
    choices = select_from_base_manifest(base_records)
    candidate = storage_root / "candidate-selection.jsonl"
    artifact = write_subset_manifest(candidate, base_records, choices, {})
    path = (
        storage_root
        / "trustsr"
        / "phase2b1b"
        / "selections"
        / artifact.sha256
        / "samples.jsonl"
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(candidate.read_bytes())
    candidate.unlink()
    return path, artifact.sha256


@lru_cache(maxsize=2)
def _geotiff_payload(kind: str) -> bytes:
    size, resolution = (130, 10.0) if kind == "lr" else (520, 2.5)
    pixels = np.zeros((4, size, size), dtype=np.uint16)
    pixels[0, 0, 0] = 10_000
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=size,
            height=size,
            count=4,
            dtype="uint16",
            crs="EPSG:32618",
            transform=Affine(resolution, 0.0, 500000.0, 0.0, -resolution, 400000.0),
            compress="deflate",
        ) as dataset:
            dataset.write(pixels)
            for band_index, description in enumerate(
                ("B04", "B03", "B02", "B08"), start=1
            ):
                dataset.set_band_description(band_index, description)
        return memory.read()


class _FakeExtractor:
    def __init__(self, base_records: tuple[dict, ...]) -> None:
        self.by_index = {record["source_index"]: record for record in base_records}
        self.calls: list[int] = []

    def __call__(
        self,
        taco_path: Path,
        source_index: int,
        output_root: Path,
        bands: tuple[str, ...],
    ) -> tuple[ExtractedAsset, ExtractedAsset]:
        assert bands == ("B04", "B03", "B02", "B08")
        record = self.by_index[source_index]
        self.calls.append(source_index)
        output_root.mkdir(parents=True, exist_ok=True)
        lr_payload = _geotiff_payload("lr")
        hr_payload = _geotiff_payload("hr")
        (output_root / "lr.tif").write_bytes(lr_payload)
        (output_root / "hr.tif").write_bytes(hr_payload)
        return (
            ExtractedAsset(
                relative_path="lr.tif",
                size_bytes=len(lr_payload),
                sha256=hashlib.sha256(lr_payload).hexdigest(),
                shape=(4, 130, 130),
                dtype="uint16",
                crs=record["crs"],
                transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 400000.0),
                nodata=None,
                minimum=0.0,
                maximum=10000.0,
                time_start=record["lr_time_start"],
            ),
            ExtractedAsset(
                relative_path="hr.tif",
                size_bytes=len(hr_payload),
                sha256=hashlib.sha256(hr_payload).hexdigest(),
                shape=(4, 520, 520),
                dtype="uint16",
                crs=record["crs"],
                transform=(2.5, 0.0, 500000.0, 0.0, -2.5, 400000.0),
                nodata=None,
                minimum=0.0,
                maximum=10000.0,
                time_start=record["hr_time_start"],
            ),
        )
def test_parser_has_exact_stage_specific_manifest_arguments() -> None:
    parser = phase2b1b.build_parser()

    selected = parser.parse_args(
        [
            "select",
            "--source",
            "source.json",
            "--storage-root",
            "/persistent",
            "--base-manifest",
            "base.jsonl",
            "--confirm-cloud-storage",
        ]
    )
    extracted = parser.parse_args(
        [
            "extract",
            "--source",
            "source.json",
            "--storage-root",
            "/persistent",
            "--selection-manifest",
            "selection.jsonl",
            "--confirm-cloud-storage",
        ]
    )

    assert selected.stage == "select"
    assert selected.base_manifest == Path("base.jsonl")
    assert not hasattr(selected, "selection_manifest")
    assert extracted.stage == "extract"
    assert extracted.selection_manifest == Path("selection.jsonl")
    assert not hasattr(extracted, "base_manifest")


def test_selection_commit_failure_does_not_publish_an_empty_digest_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_records = _base_records(tmp_path)
    candidate = tmp_path / "candidate.jsonl"
    artifact = write_subset_manifest(
        candidate, base_records, select_from_base_manifest(base_records), {}
    )
    selection_root = tmp_path / "trustsr" / "phase2b1b" / "selections"
    selection_root.mkdir(parents=True)

    def fail_write(path: Path, payload: bytes) -> None:
        raise OSError("injected write failure")

    monkeypatch.setattr(phase2b1b, "atomic_write_bytes", fail_write)

    with pytest.raises(OSError, match="injected write failure"):
        phase2b1b._commit_selection(selection_root, artifact)

    assert not (selection_root / artifact.sha256).exists()


def test_audit_commit_failure_does_not_publish_an_empty_digest_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    phase_root = tmp_path / "trustsr" / "phase2b1b"
    phase_root.mkdir(parents=True)
    manifest_sha256 = "a" * 64

    def fail_write(path: Path, payload: bytes) -> None:
        raise OSError("injected write failure")

    monkeypatch.setattr(phase2b1b, "atomic_write_bytes", fail_write)

    with pytest.raises(OSError, match="injected write failure"):
        phase2b1b._commit_audit(phase_root, manifest_sha256, b"{}")

    assert not (phase_root / "audits" / manifest_sha256).exists()


def test_reusable_selection_scan_ignores_an_abandoned_legacy_candidate(
    tmp_path: Path,
) -> None:
    base_records = _base_records(tmp_path)
    pre_manifest, pre_digest = _pre_selection(tmp_path, base_records)
    records = load_subset_manifest(pre_manifest, expected_sha256=pre_digest)
    abandoned = pre_manifest.parent.parent / ".candidate-interrupted"
    abandoned.mkdir()
    (abandoned / "partial").write_bytes(b"incomplete")

    assert phase2b1b._find_reusable_post_selection(
        tmp_path, pre_digest, records
    ) is None


def test_frozen_source_loader_accepts_the_committed_provenance() -> None:
    source, object_spec = phase2b1b._load_frozen_source(SOURCE)

    assert source.repository == "tacofoundation/SEN2NAIPv2"
    assert object_spec.path == "sen2naipv2-crosssensor.taco"
    assert object_spec.sha256 == phase2b1b.SOURCE_OBJECT_SHA256
    assert object_spec.size_bytes == phase2b1b.SOURCE_OBJECT_SIZE_BYTES


def test_base_manifest_loader_rejects_any_path_outside_the_frozen_digest(
    tmp_path: Path,
) -> None:
    wrong = tmp_path / "wrong" / "samples.jsonl"
    wrong.parent.mkdir()
    wrong.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frozen Phase 2B1A manifest"):
        phase2b1b._load_base_manifest(tmp_path, wrong)


def test_select_writes_and_reuses_a_digest_addressed_pre_extraction_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_records = _base_records(tmp_path)
    base_path = _base_path(tmp_path)
    monkeypatch.setattr(phase2b1b, "_load_base_manifest", lambda root, path: base_records)
    _patch_verified_source(monkeypatch, tmp_path)

    first = phase2b1b.run_select(
        SOURCE,
        tmp_path,
        base_path,
        confirmed_cloud_storage=True,
    )
    digest = first["digests"]["selection_manifest_sha256"]
    output = (
        tmp_path
        / "trustsr"
        / "phase2b1b"
        / "selections"
        / digest
        / "samples.jsonl"
    )
    records = load_subset_manifest(output, expected_sha256=digest)
    second = phase2b1b.run_select(
        SOURCE,
        tmp_path,
        base_path,
        confirmed_cloud_storage=True,
    )

    assert first == {
        "stage": "select",
        "digests": {
            "base_manifest_sha256": BASE_MANIFEST_SHA256,
            "selection_manifest_sha256": digest,
            "source_sha256": phase2b1b.SOURCE_OBJECT_SHA256,
        },
        "counts": {"subset_pairs": 360, "subset_geotiffs": 0},
        "reused": False,
    }
    assert len(records) == 360
    assert all(record["lr_asset"] is None for record in records)
    assert second["digests"] == first["digests"]
    assert second["reused"] is True


def test_select_rejects_wrong_choice_count_before_creating_phase_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_records = _base_records(tmp_path)
    monkeypatch.setattr(phase2b1b, "_load_base_manifest", lambda root, path: base_records)
    monkeypatch.setattr(
        phase2b1b,
        "select_from_base_manifest",
        lambda records: tuple(range(359)),
    )
    _patch_verified_source(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="exactly 360"):
        phase2b1b.run_select(
            SOURCE,
            tmp_path,
            _base_path(tmp_path),
            confirmed_cloud_storage=True,
        )

    assert not (tmp_path / "trustsr" / "phase2b1b").exists()


def test_extract_writes_all_720_assets_and_reuses_the_post_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_records = _base_records(tmp_path)
    pre_manifest, pre_digest = _pre_selection(tmp_path, base_records)
    extractor = _FakeExtractor(base_records)
    monkeypatch.setattr(
        phase2b1b, "_load_frozen_base_from_storage", lambda root: base_records
    )
    monkeypatch.setattr(phase2b1b, "extract_pair", extractor)
    _patch_verified_source(monkeypatch, tmp_path)

    first = phase2b1b.run_extract(
        SOURCE,
        tmp_path,
        pre_manifest,
        confirmed_cloud_storage=True,
    )
    post_digest = first["digests"]["selection_manifest_sha256"]
    post_manifest = (
        tmp_path
        / "trustsr"
        / "phase2b1b"
        / "selections"
        / post_digest
        / "samples.jsonl"
    )
    records = load_subset_manifest(post_manifest, expected_sha256=post_digest)
    second = phase2b1b.run_extract(
        SOURCE,
        tmp_path,
        pre_manifest,
        confirmed_cloud_storage=True,
    )

    assert first == {
        "stage": "extract",
        "digests": {
            "base_manifest_sha256": BASE_MANIFEST_SHA256,
            "input_selection_manifest_sha256": pre_digest,
            "selection_manifest_sha256": post_digest,
            "source_sha256": phase2b1b.SOURCE_OBJECT_SHA256,
        },
        "counts": {"subset_pairs": 360, "subset_geotiffs": 720},
        "reused": False,
    }
    assert len(extractor.calls) == 360
    assert all(record["lr_asset"] is not None for record in records)
    assert all(
        record["lr_asset"]["relative_path"]
        == f"subset-v1/{record['split']}/{record['sample_id']}/lr.tif"
        for record in records
    )
    assert second["digests"] == first["digests"]
    assert second["reused"] is True
    assert len(extractor.calls) == 360


def test_extract_rejects_a_partial_pair_before_calling_the_reader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_records = _base_records(tmp_path)
    pre_manifest, _ = _pre_selection(tmp_path, base_records)
    choices = select_from_base_manifest(base_records)
    partial = (
        tmp_path
        / "trustsr"
        / "phase2b1b"
        / "subset-v1"
        / choices[0].split
        / choices[0].sample_id
    )
    partial.mkdir(parents=True)
    (partial / "lr.tif").write_bytes(b"partial")
    extractor = _FakeExtractor(base_records)
    monkeypatch.setattr(
        phase2b1b, "_load_frozen_base_from_storage", lambda root: base_records
    )
    monkeypatch.setattr(phase2b1b, "extract_pair", extractor)
    _patch_verified_source(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="partial or invalid"):
        phase2b1b.run_extract(
            SOURCE,
            tmp_path,
            pre_manifest,
            confirmed_cloud_storage=True,
        )

    assert extractor.calls == []


def test_extract_requires_an_all_null_pre_extraction_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_records = _base_records(tmp_path)
    choices = select_from_base_manifest(base_records)
    extractor = _FakeExtractor(base_records)
    by_id = {record["sample_id"]: record for record in base_records}
    assets: dict[str, tuple[ExtractedAsset, ExtractedAsset]] = {}
    for choice in choices:
        pair = extractor(
            Path("unused"),
            by_id[choice.sample_id]["source_index"],
            tmp_path / "asset-fixtures" / choice.sample_id,
            ("B04", "B03", "B02", "B08"),
        )
        prefix = f"subset-v1/{choice.split}/{choice.sample_id}"
        assets[choice.sample_id] = (
            replace(pair[0], relative_path=f"{prefix}/lr.tif"),
            replace(pair[1], relative_path=f"{prefix}/hr.tif"),
        )
    candidate = tmp_path / "post-candidate.jsonl"
    artifact = write_subset_manifest(candidate, base_records, choices, assets)
    post_manifest = (
        tmp_path
        / "trustsr"
        / "phase2b1b"
        / "selections"
        / artifact.sha256
        / "samples.jsonl"
    )
    post_manifest.parent.mkdir(parents=True)
    post_manifest.write_bytes(candidate.read_bytes())
    extractor.calls.clear()
    monkeypatch.setattr(
        phase2b1b, "_load_frozen_base_from_storage", lambda root: base_records
    )
    monkeypatch.setattr(phase2b1b, "extract_pair", extractor)
    _patch_verified_source(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="all-null pre-extraction"):
        phase2b1b.run_extract(
            SOURCE,
            tmp_path,
            post_manifest,
            confirmed_cloud_storage=True,
        )

    assert extractor.calls == []


def _completed_extraction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[tuple[dict, ...], Path, str]:
    base_records = _base_records(tmp_path)
    pre_manifest, _ = _pre_selection(tmp_path, base_records)
    extractor = _FakeExtractor(base_records)
    monkeypatch.setattr(
        phase2b1b, "_load_frozen_base_from_storage", lambda root: base_records
    )
    monkeypatch.setattr(phase2b1b, "extract_pair", extractor)
    _patch_verified_source(monkeypatch, tmp_path)
    extracted = phase2b1b.run_extract(
        SOURCE,
        tmp_path,
        pre_manifest,
        confirmed_cloud_storage=True,
    )
    post_digest = extracted["digests"]["selection_manifest_sha256"]
    post_manifest = (
        tmp_path
        / "trustsr"
        / "phase2b1b"
        / "selections"
        / post_digest
        / "samples.jsonl"
    )
    return base_records, post_manifest, post_digest


def test_extract_rejects_insufficient_inodes_before_reading_taco(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_records = _base_records(tmp_path)
    pre_manifest, _ = _pre_selection(tmp_path, base_records)
    extractor = _FakeExtractor(base_records)
    monkeypatch.setattr(
        phase2b1b, "_load_frozen_base_from_storage", lambda root: base_records
    )
    monkeypatch.setattr(phase2b1b, "extract_pair", extractor)
    monkeypatch.setattr(os, "statvfs", lambda path: SimpleNamespace(f_favail=1_095))
    _patch_verified_source(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="free inodes"):
        phase2b1b.run_extract(
            SOURCE,
            tmp_path,
            pre_manifest,
            confirmed_cloud_storage=True,
        )

    assert extractor.calls == []


def test_extract_inode_gate_accounts_for_completed_pairs_at_exact_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_records = _base_records(tmp_path)
    choices = select_from_base_manifest(base_records)
    pre_manifest, _ = _pre_selection(tmp_path, base_records)
    extractor = _FakeExtractor(base_records)
    by_id = {record["sample_id"]: record for record in base_records}
    first = choices[0]
    extractor(
        Path("unused"),
        by_id[first.sample_id]["source_index"],
        tmp_path / "trustsr" / "phase2b1b" / "subset-v1" / first.split / first.sample_id,
        ("B04", "B03", "B02", "B08"),
    )
    extractor.calls.clear()
    monkeypatch.setattr(
        phase2b1b, "_load_frozen_base_from_storage", lambda root: base_records
    )
    monkeypatch.setattr(phase2b1b, "extract_pair", extractor)
    monkeypatch.setattr(os, "statvfs", lambda path: SimpleNamespace(f_favail=1_093))
    _patch_verified_source(monkeypatch, tmp_path)

    result = phase2b1b.run_extract(
        SOURCE,
        tmp_path,
        pre_manifest,
        confirmed_cloud_storage=True,
    )

    assert result["counts"] == {"subset_pairs": 360, "subset_geotiffs": 720}
    assert len(extractor.calls) == 360


def test_audit_rehashes_all_720_files_and_reuses_identical_canonical_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, post_manifest, post_digest = _completed_extraction(monkeypatch, tmp_path)
    real_inspect_pair = phase2b1b.inspect_extracted_pair
    inspection_calls: list[tuple[Path, Path]] = []

    def counting_inspection(
        lr_path: Path,
        hr_path: Path,
        *,
        lr_time_start: str,
        hr_time_start: str,
    ) -> tuple[ExtractedAsset, ExtractedAsset]:
        inspection_calls.append((lr_path, hr_path))
        return real_inspect_pair(
            lr_path,
            hr_path,
            lr_time_start=lr_time_start,
            hr_time_start=hr_time_start,
        )

    monkeypatch.setattr(phase2b1b, "inspect_extracted_pair", counting_inspection)

    first = phase2b1b.run_audit(
        SOURCE,
        tmp_path,
        post_manifest,
        confirmed_cloud_storage=True,
    )
    audit_path = (
        tmp_path
        / "trustsr"
        / "phase2b1b"
        / "audits"
        / post_digest
        / "phase2b1b-audit.json"
    )
    audit = json.loads(audit_path.read_bytes())
    second = phase2b1b.run_audit(
        SOURCE,
        tmp_path,
        post_manifest,
        confirmed_cloud_storage=True,
    )

    assert len(inspection_calls) == 720
    assert first == {
        "stage": "audit",
        "digests": {
            "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "base_manifest_sha256": BASE_MANIFEST_SHA256,
            "selection_manifest_sha256": post_digest,
            "source_sha256": phase2b1b.SOURCE_OBJECT_SHA256,
        },
        "counts": {"subset_pairs": 360, "subset_geotiffs": 720},
        "reused": False,
    }
    assert audit["manifest_sha256"] == post_digest
    assert audit["round_one_matches_phase2b1a"] is True
    assert second["digests"] == first["digests"]
    assert second["reused"] is True


@pytest.mark.parametrize("damage", ["symlink", "changed-bytes"])
def test_audit_rejects_a_symlink_or_hash_mismatch_before_writing_an_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, damage: str
) -> None:
    _, post_manifest, post_digest = _completed_extraction(monkeypatch, tmp_path)
    records = load_subset_manifest(post_manifest, expected_sha256=post_digest)
    first = records[0]
    target = (
        tmp_path
        / "trustsr"
        / "phase2b1b"
        / first["lr_asset"]["relative_path"]
    )
    if damage == "symlink":
        original = target.with_name("original-lr.tif")
        target.rename(original)
        target.symlink_to(original)
        message = "regular assets"
    else:
        target.write_bytes(b"changed")
        message = "readable GeoTIFF"

    with pytest.raises(ValueError, match=message):
        phase2b1b.run_audit(
            SOURCE,
            tmp_path,
            post_manifest,
            confirmed_cloud_storage=True,
        )

    assert not (tmp_path / "trustsr" / "phase2b1b" / "audits").exists()


@pytest.mark.parametrize("damage", ["non-geotiff", "falsified-metadata"])
def test_asset_verification_reconstructs_metadata_from_real_geotiff_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, damage: str
) -> None:
    _, post_manifest, post_digest = _completed_extraction(monkeypatch, tmp_path)
    records = [
        dict(record)
        for record in load_subset_manifest(post_manifest, expected_sha256=post_digest)
    ]
    first = dict(records[0])
    lr_asset = dict(first["lr_asset"])
    target = (
        tmp_path / "trustsr" / "phase2b1b" / lr_asset["relative_path"]
    )
    if damage == "non-geotiff":
        payload = b"self-consistent but not a GeoTIFF"
        target.write_bytes(payload)
        lr_asset["size_bytes"] = len(payload)
        lr_asset["sha256"] = hashlib.sha256(payload).hexdigest()
        message = "readable GeoTIFF"
    else:
        lr_asset["minimum"] = 1.0
        message = "metadata"
    first["lr_asset"] = lr_asset
    records[0] = first

    with pytest.raises(ValueError, match=message):
        phase2b1b._verify_post_selection_assets(tmp_path, records)


def test_audit_rejects_a_symlinked_audit_parent_before_writing_outside_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    _, post_manifest, _ = _completed_extraction(monkeypatch, storage_root)
    audit_parent = storage_root / "trustsr" / "phase2b1b" / "audits"
    external = tmp_path / "external-audits"
    external.mkdir()
    audit_parent.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes storage_root"):
        phase2b1b.run_audit(
            SOURCE,
            storage_root,
            post_manifest,
            confirmed_cloud_storage=True,
        )

    assert tuple(external.iterdir()) == ()
