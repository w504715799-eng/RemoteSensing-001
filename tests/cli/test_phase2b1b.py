"""Tests for restartable Phase 2B1B cloud data stages."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from trustsr.cli import phase2b1b
from trustsr.data.crosssensor_manifest import load_manifest, write_manifest
from trustsr.data.crosssensor_schema import CrosssensorSample
from trustsr.data.crosssensor_source import VerifiedSourceObject
from trustsr.data.pilot_sampling import select_pilot
from trustsr.data.spatial_split import AssignedSample
from trustsr.data.subset_manifest import BASE_MANIFEST_SHA256, load_subset_manifest

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
