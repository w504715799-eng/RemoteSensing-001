"""Tests for the canonical Phase 2B1B research-subset sidecar."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from trustsr.data.crosssensor_manifest import (
    ExtractedAsset,
    load_manifest,
    write_manifest,
)
from trustsr.data.crosssensor_schema import CrosssensorSample
from trustsr.data.pilot_sampling import select_pilot
from trustsr.data.research_subset import SubsetChoice, select_research_subset
from trustsr.data.spatial_split import AssignedSample
from trustsr.data.subset_manifest import (
    BASE_MANIFEST_SHA256,
    FROZEN_MINIMUM_CROSS_SPLIT_DISTANCES,
    build_subset_audit,
    load_subset_manifest,
    validate_subset_against_base,
    write_subset_manifest,
)


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


def _base_records(tmp_path: Path, assignments: tuple[AssignedSample, ...]) -> tuple[dict, ...]:
    path = tmp_path / "base.jsonl"
    artifact = write_manifest(path, assignments, select_pilot(assignments), {})
    return load_manifest(path, expected_sha256=artifact.sha256)


def _asset(
    sample_id: str,
    split: str,
    kind: str,
    time_start: str,
) -> ExtractedAsset:
    is_lr = kind == "lr"
    return ExtractedAsset(
        relative_path=f"subset-v1/{split}/{sample_id}/{kind}.tif",
        size_bytes=100 if is_lr else 200,
        sha256=hashlib.sha256(f"{sample_id}-{kind}".encode()).hexdigest(),
        shape=(4, 130, 130) if is_lr else (4, 520, 520),
        dtype="uint16",
        crs="EPSG:32618",
        transform=(
            10.0 if is_lr else 2.5,
            0.0,
            500000.0,
            0.0,
            -10.0 if is_lr else -2.5,
            400000.0,
        ),
        nodata=None,
        minimum=0.0,
        maximum=10000.0,
        time_start=time_start,
    )


def _assets(
    base_records: tuple[dict, ...], choices: tuple[SubsetChoice, ...]
) -> dict[str, tuple[ExtractedAsset, ExtractedAsset]]:
    by_id = {record["sample_id"]: record for record in base_records}
    return {
        choice.sample_id: (
            _asset(
                choice.sample_id,
                choice.split,
                "lr",
                by_id[choice.sample_id]["lr_time_start"],
            ),
            _asset(
                choice.sample_id,
                choice.split,
                "hr",
                by_id[choice.sample_id]["hr_time_start"],
            ),
        )
        for choice in choices
    }


def test_subset_manifest_round_trip_is_canonical_and_digest_addressed(
    tmp_path: Path,
) -> None:
    assignments = _assignments()
    base_records = _base_records(tmp_path, assignments)
    choices = select_research_subset(assignments)
    path = tmp_path / "samples.jsonl"

    artifact = write_subset_manifest(path, base_records, choices, {})
    records = load_subset_manifest(path, expected_sha256=artifact.sha256)

    assert len(records) == 360
    assert artifact.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(
        record["schema"] == "trustsr.phase2b1b-selection.v1" for record in records
    )
    assert all(
        record["base_manifest_sha256"] == BASE_MANIFEST_SHA256 for record in records
    )
    assert all(
        record["lr_asset"] is None and record["hr_asset"] is None for record in records
    )
    assert path.read_bytes().endswith(b"\n")


def test_subset_manifest_round_trip_accepts_exactly_all_asset_pairs(tmp_path: Path) -> None:
    assignments = _assignments()
    base_records = _base_records(tmp_path, assignments)
    choices = select_research_subset(assignments)
    path = tmp_path / "post.jsonl"

    artifact = write_subset_manifest(path, base_records, choices, _assets(base_records, choices))
    records = load_subset_manifest(path, expected_sha256=artifact.sha256)

    assert sum(record["lr_asset"] is not None for record in records) == 360
    assert sum(record["hr_asset"] is not None for record in records) == 360


def test_subset_manifest_requires_all_or_none_assets(tmp_path: Path) -> None:
    assignments = _assignments()
    base_records = _base_records(tmp_path, assignments)
    choices = select_research_subset(assignments)
    one_sample = choices[0].sample_id
    partial = {one_sample: _assets(base_records, choices)[one_sample]}

    with pytest.raises(ValueError, match="exactly every selected sample"):
        write_subset_manifest(tmp_path / "samples.jsonl", base_records, choices, partial)


def test_subset_manifest_requires_the_deterministic_choice_records(tmp_path: Path) -> None:
    assignments = _assignments()
    base_records = _base_records(tmp_path, assignments)
    choices = list(select_research_subset(assignments))
    choices[0] = replace(choices[0], selection_round=2)

    with pytest.raises(ValueError, match="deterministic research subset"):
        write_subset_manifest(tmp_path / "samples.jsonl", base_records, choices, {})


def test_subset_cross_check_rejects_metadata_changed_from_the_base(tmp_path: Path) -> None:
    assignments = _assignments()
    base_records = _base_records(tmp_path, assignments)
    choices = select_research_subset(assignments)
    path = tmp_path / "samples.jsonl"
    artifact = write_subset_manifest(path, base_records, choices, {})
    records = list(load_subset_manifest(path, expected_sha256=artifact.sha256))
    records[0]["centroid"]["longitude"] = -75.0

    with pytest.raises(ValueError, match="does not match its base manifest record"):
        validate_subset_against_base(records, base_records)


def test_subset_loader_rejects_a_duplicate_group_within_one_split(tmp_path: Path) -> None:
    assignments = _assignments()
    base_records = _base_records(tmp_path, assignments)
    choices = select_research_subset(assignments)
    path = tmp_path / "samples.jsonl"
    artifact = write_subset_manifest(path, base_records, choices, {})
    records = list(load_subset_manifest(path, expected_sha256=artifact.sha256))
    same_split = [record for record in records if record["split"] == records[0]["split"]]
    same_split[1]["spatial_group_id"] = same_split[0]["spatial_group_id"]
    payload = b"".join(
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
        for record in records
    )
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="120 distinct spatial groups"):
        load_subset_manifest(path, expected_sha256=hashlib.sha256(payload).hexdigest())


def _post_records(tmp_path: Path) -> tuple[tuple[dict, ...], tuple[dict, ...], str]:
    assignments = _assignments()
    base_records = _base_records(tmp_path, assignments)
    choices = select_research_subset(assignments)
    path = tmp_path / "post.jsonl"
    artifact = write_subset_manifest(path, base_records, choices, _assets(base_records, choices))
    records = load_subset_manifest(path, expected_sha256=artifact.sha256)
    return records, base_records, artifact.sha256


def test_subset_audit_records_exact_counts_and_round_one_evidence(tmp_path: Path) -> None:
    records, base_records, digest = _post_records(tmp_path)

    audit = build_subset_audit(
        records,
        manifest_sha256=digest,
        base_records=base_records,
        minimum_distances=FROZEN_MINIMUM_CROSS_SPLIT_DISTANCES,
    )

    assert audit["schema"] == "trustsr.phase2b1b-audit.v1"
    assert audit["base_manifest_sha256"] == BASE_MANIFEST_SHA256
    assert audit["manifest_sha256"] == digest
    assert audit["subset_pair_count"] == 360
    assert audit["subset_geotiff_count"] == 720
    assert audit["split_sample_counts"] == {
        "development": 120,
        "calibration": 120,
        "internal_test": 120,
    }
    assert audit["split_spatial_group_counts"] == audit["split_sample_counts"]
    assert len(audit["stratum_counts"]) == 36
    assert set(audit["stratum_counts"].values()) == {10}
    assert audit["selection_round_counts"] == {
        str(selection_round): 36 for selection_round in range(1, 11)
    }
    assert audit["round_one_matches_phase2b1a"] is True
    assert audit["minimum_cross_split_distances"] == FROZEN_MINIMUM_CROSS_SPLIT_DISTANCES
    assert audit["real_pixels_local"] is False
    assert audit["gpu_used"] is False


def test_subset_audit_rejects_an_all_null_sidecar(tmp_path: Path) -> None:
    assignments = _assignments()
    base_records = _base_records(tmp_path, assignments)
    choices = select_research_subset(assignments)
    path = tmp_path / "pre.jsonl"
    artifact = write_subset_manifest(path, base_records, choices, {})
    records = load_subset_manifest(path, expected_sha256=artifact.sha256)

    with pytest.raises(ValueError, match="all 720 GeoTIFF assets"):
        build_subset_audit(
            records,
            manifest_sha256=artifact.sha256,
            base_records=base_records,
            minimum_distances=FROZEN_MINIMUM_CROSS_SPLIT_DISTANCES,
        )


def test_subset_audit_rejects_changed_distance_evidence(tmp_path: Path) -> None:
    records, base_records, digest = _post_records(tmp_path)
    distances = dict(FROZEN_MINIMUM_CROSS_SPLIT_DISTANCES)
    distances["calibration:development"] += 0.001

    with pytest.raises(ValueError, match="frozen Phase 2B1A evidence"):
        build_subset_audit(
            records,
            manifest_sha256=digest,
            base_records=base_records,
            minimum_distances=distances,
        )
