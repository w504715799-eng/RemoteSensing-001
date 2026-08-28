"""Tests for canonical crosssensor manifest and audit records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from trustsr.data.crosssensor_manifest import (
    ExpectedCounts,
    ExtractedAsset,
    build_audit,
    load_manifest,
    write_manifest,
)
from trustsr.data.crosssensor_schema import CrosssensorSample
from trustsr.data.pilot_sampling import select_pilot
from trustsr.data.spatial_split import AssignedSample
from trustsr.jsonio import canonical_json


def _assignment(
    sample_id: str,
    split: str,
    days_between: int,
    correlation: float,
    group_id: str,
) -> AssignedSample:
    return AssignedSample(
        sample=CrosssensorSample(
            source_index=int(sample_id.rsplit("-", maxsplit=1)[-1]),
            sample_id=sample_id,
            longitude=-76.5,
            latitude=3.5,
            crs="EPSG:32618",
            geotransform=(10.0, 0.0, 500000.0, 0.0, -10.0, 400000.0),
            raster_shape=(130, 130),
            time_start="2020-01-02T10:00:00Z",
            admin0="Colombia",
            admin1=None,
            admin2="Cali",
            days_between=days_between,
            correlation=correlation,
            scale_factor=4,
        ),
        spatial_group_id=group_id,
        split=split,  # type: ignore[arg-type]
    )


def _assignments() -> tuple[AssignedSample, ...]:
    correlations = (0.8, 0.89, 0.91, 0.94)
    result: list[AssignedSample] = []
    index = 0
    for split in ("development", "calibration", "internal_test"):
        for days_between in (-1, 0, 1):
            for bin_index, correlation in enumerate(correlations):
                result.append(
                    _assignment(
                        f"sample-{index}",
                        split,
                        days_between,
                        correlation,
                        f"group-{split}-{days_between}-{bin_index}",
                    )
                )
                index += 1
    return tuple(reversed(result))


def _asset(relative_path: str = "pilot-v1/development/sample-0/lr.tif") -> ExtractedAsset:
    return ExtractedAsset(
        relative_path=relative_path,
        size_bytes=123,
        sha256="a" * 64,
        shape=(4, 130, 130),
        dtype="uint16",
        crs="EPSG:32618",
        transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 400000.0),
        nodata=None,
        minimum=100.0,
        maximum=120.0,
        time_start="2020-01-02T10:00:00Z",
    )


def test_write_manifest_is_canonical_sample_sorted_and_preextraction_assets_are_null(
    tmp_path: Path,
) -> None:
    assignments = _assignments() + (
        _assignment("sample-36", "development", -1, 0.8, "unselected-candidate-group"),
    )
    choices = select_pilot(assignments)
    first = write_manifest(tmp_path / "first.jsonl", assignments, choices, {})
    second = write_manifest(tmp_path / "second.jsonl", assignments, choices, {})

    payload = first.path.read_bytes()
    assert payload == second.path.read_bytes()
    assert first.size_bytes == len(payload)
    assert first.sha256 == second.sha256 == hashlib.sha256(payload).hexdigest()
    assert payload.endswith(b"\n")
    records = [json.loads(line) for line in payload.splitlines()]
    assert [record["sample_id"] for record in records] == sorted(
        record["sample_id"] for record in records
    )
    assert all(
        set(record) == {
            "schema",
            "source",
            "source_index",
            "sample_id",
            "centroid",
            "crs",
            "geotransform",
            "raster_shape",
            "time_start",
            "admin",
            "days_between",
            "correlation",
            "scale_factor",
            "spatial_group_id",
            "split",
            "pilot",
            "lr_asset",
            "hr_asset",
        }
        for record in records
    )
    assert all(record["lr_asset"] is None and record["hr_asset"] is None for record in records)
    assert all(record["schema"] == "trustsr.sen2naipv2-sample.v1" for record in records)
    assert all(
        record["source"]
        == {
            "revision": "c370504201072fdb1dd388013ab8c0fc7d00a57e",
            "object_sha256": "c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5",
        }
        for record in records
    )
    selected_ids = {choice.sample_id for choice in choices}
    assert sum(record["pilot"] is not None for record in records) == 36
    assert all(
        (record["pilot"] is not None) == (record["sample_id"] in selected_ids)
        for record in records
    )
    assert load_manifest(first.path, expected_sha256=first.sha256) == tuple(records)


def test_write_manifest_serializes_a_pair_for_every_selected_sample(tmp_path: Path) -> None:
    assignments = _assignments()
    choices = select_pilot(assignments)
    assets = {
        choice.sample_id: (
            _asset(f"pilot-v1/{choice.split}/{choice.sample_id}/lr.tif"),
            _asset(f"pilot-v1/{choice.split}/{choice.sample_id}/hr.tif"),
        )
        for choice in choices
    }

    artifact = write_manifest(tmp_path / "assets.jsonl", assignments, choices, assets)
    records = load_manifest(artifact.path, expected_sha256=artifact.sha256)

    assert all(
        record["lr_asset"] is not None and record["hr_asset"] is not None for record in records
    )
    assert records[0]["lr_asset"] == {
        "relative_path": "pilot-v1/development/sample-0/lr.tif",
        "size_bytes": 123,
        "sha256": "a" * 64,
        "shape": [4, 130, 130],
        "dtype": "uint16",
        "crs": "EPSG:32618",
        "transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 400000.0],
        "nodata": None,
        "minimum": 100.0,
        "maximum": 120.0,
        "time_start": "2020-01-02T10:00:00Z",
    }


def test_write_manifest_preserves_empty_administrative_labels(tmp_path: Path) -> None:
    assignments = list(_assignments())
    assignments[0] = replace(
        assignments[0], sample=replace(assignments[0].sample, admin1="")
    )
    choices = select_pilot(assignments)

    artifact = write_manifest(tmp_path / "manifest.jsonl", assignments, choices, {})

    records = load_manifest(artifact.path, expected_sha256=artifact.sha256)
    changed_record = next(record for record in records if record["sample_id"] == "sample-35")
    assert changed_record["admin"]["admin1"] == ""


@pytest.mark.parametrize("relative_path", ["/pilot/lr.tif", "pilot/../lr.tif"])
def test_write_manifest_rejects_nonrelative_asset_paths(tmp_path: Path, relative_path: str) -> None:
    assignments = _assignments()
    choices = select_pilot(assignments)
    assets = {
        choice.sample_id: (_asset(relative_path), _asset("pilot-v1/valid/hr.tif"))
        for choice in choices
    }

    with pytest.raises(ValueError, match="relative POSIX path"):
        write_manifest(tmp_path / "manifest.jsonl", assignments, choices, assets)


def test_build_audit_reports_exact_synthetic_counts_and_rejects_leakage(tmp_path: Path) -> None:
    assignments = _assignments()
    choices = select_pilot(assignments)
    artifact = write_manifest(tmp_path / "manifest.jsonl", assignments, choices, {})
    records = load_manifest(artifact.path, expected_sha256=artifact.sha256)
    expected = ExpectedCounts(
        samples=36,
        components=36,
        development_samples=12,
        calibration_samples=12,
        internal_test_samples=12,
        development_components=12,
        calibration_components=12,
        internal_test_components=12,
    )

    audit = build_audit(
        records,
        manifest_sha256=artifact.sha256,
        minimum_distances={
            "calibration:development": 5.1,
            "calibration:internal_test": 5.2,
            "development:internal_test": 5.3,
        },
        expected=expected,
    )

    assert audit == {
        "schema": "trustsr.phase2b1a-audit.v1",
        "source_revision": "c370504201072fdb1dd388013ab8c0fc7d00a57e",
        "source_object_sha256": "c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5",
        "manifest_sha256": artifact.sha256,
        "sample_count": 36,
        "component_count": 36,
        "split_sample_counts": {
            "development": 12,
            "calibration": 12,
            "internal_test": 12,
        },
        "split_component_counts": {
            "development": 12,
            "calibration": 12,
            "internal_test": 12,
        },
        "minimum_cross_split_distances": {
            "calibration:development": 5.1,
            "calibration:internal_test": 5.2,
            "development:internal_test": 5.3,
        },
        "pilot_pair_count": 36,
        "pilot_geotiff_count": 0,
        "real_pixels_local": False,
        "gpu_used": False,
    }

    with pytest.raises(ValueError, match="minimum cross-split distance"):
        build_audit(
            records,
            manifest_sha256=artifact.sha256,
            minimum_distances={"calibration:development": 5.0},
            expected=expected,
        )

    records_with_shared_group = [dict(record) for record in records]
    records_with_shared_group[0]["spatial_group_id"] = records_with_shared_group[1][
        "spatial_group_id"
    ]
    records_with_shared_group[0]["split"] = "development"
    records_with_shared_group[1]["split"] = "calibration"
    with pytest.raises(ValueError, match="shared spatial group"):
        build_audit(
            records_with_shared_group,
            manifest_sha256=artifact.sha256,
            minimum_distances={"calibration:development": 5.1},
            expected=expected,
        )


def test_load_manifest_rejects_digest_mismatch_and_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    record = {
        "schema": "trustsr.unknown.v1",
        "source": {
            "revision": "c370504201072fdb1dd388013ab8c0fc7d00a57e",
            "object_sha256": "c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5",
        },
    }
    payload = canonical_json(record) + b"\n"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="manifest SHA-256"):
        load_manifest(path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="unknown manifest schema"):
        load_manifest(path, expected_sha256=hashlib.sha256(payload).hexdigest())
