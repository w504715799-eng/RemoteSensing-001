"""Offline tests for the restartable Phase 2B1A stage controller."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import trustsr.cli.phase2b1a as phase2b1a
from trustsr.data.crosssensor_manifest import (
    PRODUCTION_EXPECTED_COUNTS,
    ExtractedAsset,
    ManifestArtifact,
)
from trustsr.data.crosssensor_schema import CrosssensorSample
from trustsr.data.crosssensor_source import VerifiedSourceObject
from trustsr.data.pilot_sampling import select_pilot
from trustsr.data.provenance import DatasetSource, LfsObject
from trustsr.data.spatial_split import AssignedSample

_SOURCE_REVISION = "c370504201072fdb1dd388013ab8c0fc7d00a57e"
_SOURCE_SHA256 = "c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5"
_SOURCE_SIZE = 9_717_583_850


def _source(
    *, revision: str = _SOURCE_REVISION, object_sha256: str = _SOURCE_SHA256
) -> DatasetSource:
    return DatasetSource(
        schema="trustsr.sen2naipv2-source.v1",
        repository="tacofoundation/SEN2NAIPv2",
        revision=revision,
        license_claim="cc0-1.0",
        card_sha256="b" * 64,
        bands=("B04", "B03", "B02", "B08"),
        scale=4,
        lr_shape=(130, 130),
        hr_shape=(520, 520),
        declared_total_bytes=_SOURCE_SIZE,
        objects=(
            LfsObject(
                path="sen2naipv2-crosssensor.taco",
                sha256=object_sha256,
                size_bytes=_SOURCE_SIZE,
            ),
        ),
    )


def _argv(stage: str, tmp_path: Path) -> list[str]:
    arguments = [
        stage,
        "--source",
        str(tmp_path / "source.json"),
        "--storage-root",
        str(tmp_path),
    ]
    if stage == "download":
        arguments.extend(("--transport-url", "https://download.invalid/source.taco"))
    if stage in {"pilot", "audit"}:
        arguments.extend(("--manifest", str(tmp_path / "manifest.jsonl")))
    arguments.append("--confirm-cloud-storage")
    return arguments


@pytest.mark.parametrize(
    "argv",
    [
        ["download", "--source", "source.json", "--transport-url", "https://invalid"],
        ["download", "--source", "source.json", "--storage-root", "/cloud"],
        ["manifest", "--source", "source.json"],
        ["pilot", "--source", "source.json", "--manifest", "samples.jsonl"],
        ["audit", "--source", "source.json", "--manifest", "samples.jsonl"],
    ],
)
def test_parser_has_no_storage_root_or_transport_url_default(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        phase2b1a.build_parser().parse_args(argv)


@pytest.mark.parametrize("selected_stage", ["download", "manifest", "pilot", "audit"])
def test_main_dispatches_only_the_selected_stage_and_emits_safe_canonical_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    selected_stage: str,
) -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    def service(stage: str):  # type: ignore[no-untyped-def]
        def run(*arguments: object, **keywords: object) -> dict[str, object]:
            calls.append((stage, arguments + (keywords,)))
            return {
                "stage": stage,
                "digests": {"artifact_sha256": "a" * 64},
                "counts": {"items": 1},
                "reused": False,
            }

        return run

    for stage in ("download", "manifest", "pilot", "audit"):
        monkeypatch.setattr(phase2b1a, f"run_{stage}", service(stage))

    assert phase2b1a.main(_argv(selected_stage, tmp_path)) == 0

    assert [stage for stage, _ in calls] == [selected_stage]
    line = capsys.readouterr().out
    payload = json.loads(line)
    assert line == json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ) + "\n"
    assert payload == {
        "counts": {"items": 1},
        "digests": {"artifact_sha256": "a" * 64},
        "reused": False,
        "stage": selected_stage,
    }
    assert str(tmp_path) not in line
    assert "https://" not in line
    assert "download.invalid" not in line
    assert "password" not in line
    assert "timestamp" not in line

    arguments = calls[0][1]
    assert arguments[0] == tmp_path / "source.json"
    assert arguments[1] == tmp_path
    if selected_stage == "download":
        assert arguments[2] == "https://download.invalid/source.taco"
    elif selected_stage in {"pilot", "audit"}:
        assert arguments[2] == tmp_path / "manifest.jsonl"
    assert arguments[-1] == {"confirmed_cloud_storage": True}


def test_download_delegates_the_frozen_source_and_reports_only_digest_count_and_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _source()
    expected_path = (
        tmp_path
        / "trustsr"
        / "phase2b1a"
        / "source"
        / _SOURCE_SHA256
        / "sen2naipv2-crosssensor.taco"
    )
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(phase2b1a, "load_dataset_source", lambda _: source)

    def acquire(*args: object, **kwargs: object) -> VerifiedSourceObject:
        calls.append(args + (kwargs,))
        return VerifiedSourceObject(expected_path, _SOURCE_SIZE, _SOURCE_SHA256)

    monkeypatch.setattr(phase2b1a, "acquire_crosssensor", acquire)

    payload = phase2b1a.run_download(
        tmp_path / "source.json",
        tmp_path,
        "https://download.invalid/source.taco",
        confirmed_cloud_storage=True,
    )

    assert calls == [
        (
            source,
            tmp_path,
            "https://download.invalid/source.taco",
            {"confirmed_cloud_storage": True},
        )
    ]
    assert payload == {
        "stage": "download",
        "digests": {"source_sha256": _SOURCE_SHA256},
        "counts": {"source_bytes": _SOURCE_SIZE},
        "reused": False,
    }


@pytest.mark.parametrize("confirmed", [False, None, 1, "true"])
def test_manifest_requires_exact_true_confirmation_before_loading_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    confirmed: object,
) -> None:
    def unexpected(_: Path) -> DatasetSource:
        raise AssertionError("source loading must follow confirmation")

    monkeypatch.setattr(phase2b1a, "load_dataset_source", unexpected)

    with pytest.raises(ValueError, match="explicit cloud storage confirmation"):
        phase2b1a.run_manifest(
            tmp_path / "source.json",
            tmp_path,
            confirmed_cloud_storage=confirmed,  # type: ignore[arg-type]
        )


def test_download_rejects_a_source_object_mismatch_before_acquisition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        phase2b1a,
        "load_dataset_source",
        lambda _: _source(object_sha256="a" * 64),
    )

    def unexpected(*_: object, **__: object) -> VerifiedSourceObject:
        raise AssertionError("mismatched source must not reach acquisition")

    monkeypatch.setattr(phase2b1a, "acquire_crosssensor", unexpected)

    with pytest.raises(ValueError, match="frozen crosssensor source"):
        phase2b1a.run_download(
            tmp_path / "source.json",
            tmp_path,
            "https://download.invalid/source.taco",
            confirmed_cloud_storage=True,
        )


def _top_records(column_count: int = 26) -> tuple[dict[str, object], ...]:
    columns = {
        "tortilla:id": "sample",
        "stac:crs": "EPSG:32618",
        "stac:geotransform": (10.0, 0.0, 500000.0, 0.0, -10.0, 400000.0),
        "stac:raster_shape": (130, 130),
        "stac:time_start": "2020-01-02T10:00:00Z",
        "stac:centroid": "POINT (-76.5 3.5)",
        "rai:admin0": "Colombia",
        "rai:admin1": None,
        "rai:admin2": "Cali",
        "days_between": 0,
        "correlation": 0.91,
        "scale_factor": 4,
        **{f"extra:{index}": index for index in range(14)},
    }
    selected = dict(list(columns.items())[:column_count])
    return tuple(selected for _ in range(8_000))


def _patch_manifest_services(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    column_count: int = 26,
    pilot_count: int = 36,
    audit_error: str | None = None,
) -> tuple[str, dict[str, object]]:
    candidate_payload = b'{"synthetic":"pre-extraction"}\n'
    candidate_digest = hashlib.sha256(candidate_payload).hexdigest()
    calls: dict[str, object] = {}
    source = _source()
    monkeypatch.setattr(phase2b1a, "load_dataset_source", lambda _: source)

    def verify(path: Path, object_spec: LfsObject) -> VerifiedSourceObject:
        calls["verify"] = (path, object_spec)
        return VerifiedSourceObject(path, _SOURCE_SIZE, _SOURCE_SHA256)

    monkeypatch.setattr(phase2b1a, "verify_crosssensor", verify)

    def load_records(path: Path) -> tuple[dict[str, object], ...]:
        calls["reader_path"] = path
        return _top_records(column_count)

    monkeypatch.setattr(phase2b1a, "load_top_level_records", load_records)

    def normalize(records: object, *, expected_count: int) -> tuple[str, ...]:
        calls["normalize"] = (records, expected_count)
        return ("normalized",)

    monkeypatch.setattr(phase2b1a, "normalize_top_level", normalize)
    monkeypatch.setattr(
        phase2b1a,
        "assign_spatial_splits",
        lambda samples: calls.setdefault("assign", samples) and ("assigned",),
    )
    distances = {
        "calibration:development": 5.1,
        "calibration:internal_test": 5.2,
        "development:internal_test": 5.3,
    }
    monkeypatch.setattr(
        phase2b1a,
        "minimum_cross_split_distances",
        lambda assignments: calls.setdefault("distance_assignments", assignments) and distances,
    )
    monkeypatch.setattr(
        phase2b1a,
        "select_pilot",
        lambda assignments: calls.setdefault("choice_assignments", assignments)
        and tuple(range(pilot_count)),
    )

    def write(
        path: Path, assignments: object, choices: object, assets: object
    ) -> ManifestArtifact:
        calls["write"] = (assignments, choices, assets)
        path.write_bytes(candidate_payload)
        return ManifestArtifact(path, len(candidate_payload), candidate_digest)

    monkeypatch.setattr(phase2b1a, "write_manifest", write)

    def load(path: Path, *, expected_sha256: str) -> tuple[dict[str, object], ...]:
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("manifest SHA-256 does not match the expected digest")
        calls.setdefault("loads", []).append((path, expected_sha256))  # type: ignore[union-attr]
        return ({"synthetic": True},)

    monkeypatch.setattr(phase2b1a, "load_manifest", load)

    def audit(
        records: object,
        *,
        manifest_sha256: str,
        minimum_distances: object,
        expected: object,
    ) -> dict[str, object]:
        calls["audit"] = (records, manifest_sha256, minimum_distances, expected)
        if audit_error is not None:
            raise ValueError(audit_error)
        return {
            "schema": "trustsr.phase2b1a-audit.v1",
            "source_revision": _SOURCE_REVISION,
            "source_object_sha256": _SOURCE_SHA256,
            "manifest_sha256": manifest_sha256,
            "sample_count": 8_000,
            "component_count": 6_695,
            "split_sample_counts": {
                "development": 3_967,
                "calibration": 2_070,
                "internal_test": 1_963,
            },
            "split_component_counts": {
                "development": 3_317,
                "calibration": 1_719,
                "internal_test": 1_659,
            },
            "minimum_cross_split_distances": distances,
            "pilot_pair_count": 36,
            "pilot_geotiff_count": 0,
            "real_pixels_local": False,
            "gpu_used": False,
        }

    monkeypatch.setattr(phase2b1a, "build_audit", audit)
    return candidate_digest, calls


def test_manifest_commits_digest_addressed_output_reuses_it_and_rejects_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    digest, calls = _patch_manifest_services(monkeypatch, tmp_path)

    first = phase2b1a.run_manifest(
        tmp_path / "source.json", tmp_path, confirmed_cloud_storage=True
    )
    manifest = (
        tmp_path
        / "trustsr"
        / "phase2b1a"
        / "manifests"
        / digest
        / "samples.jsonl"
    )

    assert manifest.read_bytes() == b'{"synthetic":"pre-extraction"}\n'
    assert first == {
        "stage": "manifest",
        "digests": {
            "manifest_sha256": digest,
            "source_sha256": _SOURCE_SHA256,
        },
        "counts": {"samples": 8_000, "components": 6_695, "pilot_pairs": 36},
        "reused": False,
    }
    assert calls["verify"] == (
        tmp_path
        / "trustsr"
        / "phase2b1a"
        / "source"
        / _SOURCE_SHA256
        / "sen2naipv2-crosssensor.taco",
        _source().objects[0],
    )
    assert calls["normalize"][1] == 8_000  # type: ignore[index]
    assert calls["write"] == (("assigned",), tuple(range(36)), {})
    assert calls["audit"][3] == PRODUCTION_EXPECTED_COUNTS  # type: ignore[index]

    second = phase2b1a.run_manifest(
        tmp_path / "source.json", tmp_path, confirmed_cloud_storage=True
    )
    assert second == {**first, "reused": True}

    manifest.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        phase2b1a.run_manifest(
            tmp_path / "source.json", tmp_path, confirmed_cloud_storage=True
        )
    assert manifest.read_bytes() == b"tampered"


def test_manifest_rejects_any_top_level_table_width_other_than_26_before_normalizing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_manifest_services(monkeypatch, tmp_path, column_count=25)

    with pytest.raises(ValueError, match="exactly 26 top-level columns"):
        phase2b1a.run_manifest(
            tmp_path / "source.json", tmp_path, confirmed_cloud_storage=True
        )


def test_manifest_rejects_a_pilot_other_than_36_before_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_manifest_services(monkeypatch, tmp_path, pilot_count=35)

    with pytest.raises(ValueError, match="exactly 36"):
        phase2b1a.run_manifest(
            tmp_path / "source.json", tmp_path, confirmed_cloud_storage=True
        )


def test_manifest_rejects_wrong_production_counts_before_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    digest, _ = _patch_manifest_services(
        monkeypatch,
        tmp_path,
        audit_error="manifest counts do not match expected counts",
    )

    with pytest.raises(ValueError, match="manifest counts do not match expected counts"):
        phase2b1a.run_manifest(
            tmp_path / "source.json", tmp_path, confirmed_cloud_storage=True
        )

    assert not (
        tmp_path / "trustsr" / "phase2b1a" / "manifests" / digest
    ).exists()


def _pilot_assignments() -> tuple[AssignedSample, ...]:
    correlations = (0.8, 0.89, 0.91, 0.94)
    assignments: list[AssignedSample] = []
    source_index = 0
    for split in ("development", "calibration", "internal_test"):
        for days_between in (-1, 0, 1):
            for bin_index, correlation in enumerate(correlations):
                sample_id = f"{split}-{days_between}-{bin_index}"
                assignments.append(
                    AssignedSample(
                        sample=CrosssensorSample(
                            source_index=source_index,
                            sample_id=sample_id,
                            longitude=-76.5 + source_index * 0.1,
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
                            admin0="Colombia",
                            admin1=None,
                            admin2="Cali",
                            days_between=days_between,
                            correlation=correlation,
                            scale_factor=4,
                        ),
                        spatial_group_id=hashlib.sha256(sample_id.encode()).hexdigest(),
                        split=split,  # type: ignore[arg-type]
                    )
                )
                source_index += 1
    return tuple(assignments)


def _asset(relative_path: str, payload: bytes) -> ExtractedAsset:
    dimensions = (4, 130, 130) if relative_path == "lr.tif" else (4, 520, 520)
    resolution = 10.0 if relative_path == "lr.tif" else 2.5
    return ExtractedAsset(
        relative_path=relative_path,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        shape=dimensions,
        dtype="uint16",
        crs="EPSG:32618",
        transform=(resolution, 0.0, 500000.0, 0.0, -resolution, 400000.0),
        nodata=None,
        minimum=100.0,
        maximum=120.0,
        time_start="2020-01-02T10:00:00Z",
    )


def _asset_record(asset: ExtractedAsset) -> dict[str, object]:
    return {
        "relative_path": asset.relative_path,
        "size_bytes": asset.size_bytes,
        "sha256": asset.sha256,
        "shape": list(asset.shape),
        "dtype": asset.dtype,
        "crs": asset.crs,
        "transform": list(asset.transform),
        "nodata": asset.nodata,
        "minimum": asset.minimum,
        "maximum": asset.maximum,
        "time_start": asset.time_start,
    }


def _manifest_records(
    assignments: tuple[AssignedSample, ...],
    assets: dict[str, tuple[ExtractedAsset, ExtractedAsset]] | None = None,
) -> tuple[dict[str, object], ...]:
    choices = {choice.sample_id: choice for choice in select_pilot(assignments)}
    result: list[dict[str, object]] = []
    for assignment in assignments:
        sample = assignment.sample
        choice = choices[sample.sample_id]
        pair = None if assets is None else assets[sample.sample_id]
        result.append(
            {
                "schema": "trustsr.sen2naipv2-sample.v1",
                "source": {
                    "revision": _SOURCE_REVISION,
                    "object_sha256": _SOURCE_SHA256,
                },
                "source_index": sample.source_index,
                "sample_id": sample.sample_id,
                "centroid": {
                    "longitude": sample.longitude,
                    "latitude": sample.latitude,
                },
                "crs": sample.crs,
                "geotransform": list(sample.geotransform),
                "raster_shape": list(sample.raster_shape),
                "time_start": sample.time_start,
                "admin": {
                    "admin0": sample.admin0,
                    "admin1": sample.admin1,
                    "admin2": sample.admin2,
                },
                "days_between": sample.days_between,
                "correlation": sample.correlation,
                "scale_factor": sample.scale_factor,
                "spatial_group_id": assignment.spatial_group_id,
                "split": assignment.split,
                "pilot": {
                    "days_between": choice.days_between,
                    "correlation_bin": choice.correlation_bin,
                    "selection_sha256": choice.selection_sha256,
                },
                "lr_asset": None if pair is None else _asset_record(pair[0]),
                "hr_asset": None if pair is None else _asset_record(pair[1]),
            }
        )
    return tuple(sorted(result, key=lambda record: str(record["sample_id"])))


def _digest_manifest(tmp_path: Path, payload: bytes = b"synthetic pre-manifest\n") -> Path:
    digest = hashlib.sha256(payload).hexdigest()
    path = (
        tmp_path
        / "trustsr"
        / "phase2b1a"
        / "manifests"
        / digest
        / "samples.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _patch_pilot_services(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    records: tuple[dict[str, object], ...],
    *,
    preaudit_error: str | None = None,
) -> tuple[Path, str, dict[str, object]]:
    input_manifest = _digest_manifest(tmp_path)
    input_digest = input_manifest.parent.name
    post_payload = b'{"synthetic":"post-extraction"}\n'
    post_digest = hashlib.sha256(post_payload).hexdigest()
    calls: dict[str, object] = {"extract": []}
    monkeypatch.setattr(phase2b1a, "load_dataset_source", lambda _: _source())
    monkeypatch.setattr(
        phase2b1a,
        "verify_crosssensor",
        lambda path, spec: VerifiedSourceObject(path, spec.size_bytes, spec.sha256),
    )

    def load(path: Path, *, expected_sha256: str) -> tuple[dict[str, object], ...]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("manifest SHA-256 does not match the expected digest")
        return records

    monkeypatch.setattr(phase2b1a, "load_manifest", load)
    distances = {
        "calibration:development": 5.1,
        "calibration:internal_test": 5.2,
        "development:internal_test": 5.3,
    }
    monkeypatch.setattr(
        phase2b1a,
        "minimum_cross_split_distances",
        lambda assignments: calls.setdefault("preaudit_assignments", assignments)
        and distances,
    )

    def build_pre_audit(
        received_records: object,
        *,
        manifest_sha256: str,
        minimum_distances: object,
        expected: object,
    ) -> dict[str, object]:
        calls["preaudit"] = (
            received_records,
            manifest_sha256,
            minimum_distances,
            expected,
        )
        if preaudit_error is not None:
            raise ValueError(preaudit_error)
        return {
            "sample_count": 8_000,
            "component_count": 6_695,
            "pilot_pair_count": 36,
            "pilot_geotiff_count": 0,
        }

    monkeypatch.setattr(phase2b1a, "build_audit", build_pre_audit)

    def extract(
        taco_path: Path,
        source_index: int,
        output_root: Path,
        bands: tuple[str, ...],
    ) -> tuple[ExtractedAsset, ExtractedAsset]:
        calls["extract"].append((taco_path, source_index, output_root, bands))  # type: ignore[union-attr]
        lr_payload = f"lr-{source_index}".encode()
        hr_payload = f"hr-{source_index}".encode()
        for name, payload in (("lr.tif", lr_payload), ("hr.tif", hr_payload)):
            target = output_root / name
            if target.exists() and target.read_bytes() != payload:
                raise ValueError(f"existing asset {target} has different bytes")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(payload)
        return _asset("lr.tif", lr_payload), _asset("hr.tif", hr_payload)

    monkeypatch.setattr(phase2b1a, "extract_pair", extract)

    def write(
        path: Path,
        assignments: object,
        choices: object,
        assets: dict[str, tuple[ExtractedAsset, ExtractedAsset]],
    ) -> ManifestArtifact:
        calls["write"] = (assignments, choices, assets)
        path.write_bytes(post_payload)
        return ManifestArtifact(path, len(post_payload), post_digest)

    monkeypatch.setattr(phase2b1a, "write_manifest", write)
    return input_manifest, input_digest, {"post_digest": post_digest, "calls": calls}


def test_pilot_rebases_all_72_assets_and_reuses_the_digest_addressed_post_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assignments = _pilot_assignments()
    records = _manifest_records(assignments)
    input_manifest, input_digest, state = _patch_pilot_services(
        monkeypatch, tmp_path, records
    )

    first = phase2b1a.run_pilot(
        tmp_path / "source.json",
        tmp_path,
        input_manifest,
        confirmed_cloud_storage=True,
    )
    post_digest = state["post_digest"]
    post_manifest = (
        tmp_path
        / "trustsr"
        / "phase2b1a"
        / "manifests"
        / post_digest
        / "samples.jsonl"
    )
    written_assets = state["calls"]["write"][2]  # type: ignore[index]

    assert post_manifest.read_bytes() == b'{"synthetic":"post-extraction"}\n'
    assert len(written_assets) == 36
    for assignment in assignments:
        sample_id = assignment.sample.sample_id
        lr, hr = written_assets[sample_id]
        prefix = f"pilot-v1/{assignment.split}/{sample_id}"
        assert lr.relative_path == f"{prefix}/lr.tif"
        assert hr.relative_path == f"{prefix}/hr.tif"
        assert (tmp_path / "trustsr" / "phase2b1a" / lr.relative_path).is_file()
        assert (tmp_path / "trustsr" / "phase2b1a" / hr.relative_path).is_file()
    assert first == {
        "stage": "pilot",
        "digests": {
            "input_manifest_sha256": input_digest,
            "manifest_sha256": post_digest,
            "source_sha256": _SOURCE_SHA256,
        },
        "counts": {"samples": 36, "pilot_pairs": 36, "pilot_geotiffs": 72},
        "reused": False,
    }

    second = phase2b1a.run_pilot(
        tmp_path / "source.json",
        tmp_path,
        input_manifest,
        confirmed_cloud_storage=True,
    )
    assert second == {**first, "reused": True}

    first_output = state["calls"]["extract"][0][2] / "lr.tif"  # type: ignore[index]
    first_output.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="different bytes"):
        phase2b1a.run_pilot(
            tmp_path / "source.json",
            tmp_path,
            input_manifest,
            confirmed_cloud_storage=True,
        )
    assert first_output.read_bytes() == b"tampered"


def test_pilot_rejects_a_manifest_whose_parent_is_not_its_actual_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _digest_manifest(tmp_path)
    wrong = manifest.parent.parent / ("a" * 64) / "samples.jsonl"
    wrong.parent.mkdir()
    wrong.write_bytes(manifest.read_bytes())
    monkeypatch.setattr(phase2b1a, "load_dataset_source", lambda _: _source())
    monkeypatch.setattr(
        phase2b1a,
        "verify_crosssensor",
        lambda path, spec: VerifiedSourceObject(path, spec.size_bytes, spec.sha256),
    )

    with pytest.raises(ValueError, match="parent.*actual SHA-256"):
        phase2b1a.run_pilot(
            tmp_path / "source.json",
            tmp_path,
            wrong,
            confirmed_cloud_storage=True,
        )


def test_pilot_rejects_any_selection_other_than_36_before_extraction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records = _manifest_records(_pilot_assignments())[:-1]
    manifest, _, state = _patch_pilot_services(monkeypatch, tmp_path, records)

    with pytest.raises(ValueError, match="exactly 36"):
        phase2b1a.run_pilot(
            tmp_path / "source.json",
            tmp_path,
            manifest,
            confirmed_cloud_storage=True,
        )
    assert state["calls"]["extract"] == []  # type: ignore[index]


def test_pilot_rejects_a_pre_manifest_with_wrong_production_counts_before_extraction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records = _manifest_records(_pilot_assignments())
    manifest, _, state = _patch_pilot_services(
        monkeypatch,
        tmp_path,
        records,
        preaudit_error="manifest counts do not match expected counts",
    )

    with pytest.raises(ValueError, match="manifest counts do not match expected counts"):
        phase2b1a.run_pilot(
            tmp_path / "source.json",
            tmp_path,
            manifest,
            confirmed_cloud_storage=True,
        )
    assert state["calls"]["preaudit"][3] == PRODUCTION_EXPECTED_COUNTS  # type: ignore[index]
    assert state["calls"]["extract"] == []  # type: ignore[index]


def test_pilot_rejects_unsafe_sample_id_before_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records = list(_manifest_records(_pilot_assignments()))
    records[0] = {**records[0], "sample_id": "../escape"}
    manifest, _, state = _patch_pilot_services(monkeypatch, tmp_path, tuple(records))

    with pytest.raises(ValueError, match="safe path component"):
        phase2b1a.run_pilot(
            tmp_path / "source.json",
            tmp_path,
            manifest,
            confirmed_cloud_storage=True,
        )
    assert state["calls"]["extract"] == []  # type: ignore[index]
    assert not (tmp_path / "trustsr" / "phase2b1a" / "escape").exists()


def test_pilot_rejects_a_partial_existing_pair_before_any_extraction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assignments = _pilot_assignments()
    records = _manifest_records(assignments)
    manifest, _, state = _patch_pilot_services(monkeypatch, tmp_path, records)
    first = assignments[0]
    partial = (
        tmp_path
        / "trustsr"
        / "phase2b1a"
        / "pilot-v1"
        / first.split
        / first.sample.sample_id
    )
    partial.mkdir(parents=True)
    (partial / "lr.tif").write_bytes(b"partial")

    with pytest.raises(ValueError, match="partial or invalid"):
        phase2b1a.run_pilot(
            tmp_path / "source.json",
            tmp_path,
            manifest,
            confirmed_cloud_storage=True,
        )
    assert state["calls"]["extract"] == []  # type: ignore[index]
    assert (partial / "lr.tif").read_bytes() == b"partial"
    assert not (partial / "hr.tif").exists()


def _post_assets(
    tmp_path: Path, assignments: tuple[AssignedSample, ...]
) -> dict[str, tuple[ExtractedAsset, ExtractedAsset]]:
    assets: dict[str, tuple[ExtractedAsset, ExtractedAsset]] = {}
    for assignment in assignments:
        sample = assignment.sample
        prefix = f"pilot-v1/{assignment.split}/{sample.sample_id}"
        lr_payload = f"lr-{sample.source_index}".encode()
        hr_payload = f"hr-{sample.source_index}".encode()
        lr = replace(_asset("lr.tif", lr_payload), relative_path=f"{prefix}/lr.tif")
        hr = replace(_asset("hr.tif", hr_payload), relative_path=f"{prefix}/hr.tif")
        assets[sample.sample_id] = (lr, hr)
        output = tmp_path / "trustsr" / "phase2b1a" / prefix
        output.mkdir(parents=True)
        (output / "lr.tif").write_bytes(lr_payload)
        (output / "hr.tif").write_bytes(hr_payload)
    return assets


def _patch_audit_services(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    records: tuple[dict[str, object], ...],
) -> tuple[Path, dict[str, object], dict[str, object]]:
    manifest = _digest_manifest(tmp_path, b"synthetic post-manifest\n")
    calls: dict[str, object] = {}
    distances = {
        "calibration:development": 5.1,
        "calibration:internal_test": 5.2,
        "development:internal_test": 5.3,
    }
    audit_payload = {
        "schema": "trustsr.phase2b1a-audit.v1",
        "source_revision": _SOURCE_REVISION,
        "source_object_sha256": _SOURCE_SHA256,
        "manifest_sha256": manifest.parent.name,
        "sample_count": 8_000,
        "component_count": 6_695,
        "split_sample_counts": {
            "development": 3_967,
            "calibration": 2_070,
            "internal_test": 1_963,
        },
        "split_component_counts": {
            "development": 3_317,
            "calibration": 1_719,
            "internal_test": 1_659,
        },
        "minimum_cross_split_distances": distances,
        "pilot_pair_count": 36,
        "pilot_geotiff_count": 72,
        "real_pixels_local": False,
        "gpu_used": False,
    }
    monkeypatch.setattr(phase2b1a, "load_dataset_source", lambda _: _source())
    monkeypatch.setattr(
        phase2b1a,
        "verify_crosssensor",
        lambda path, spec: VerifiedSourceObject(path, spec.size_bytes, spec.sha256),
    )

    def load(path: Path, *, expected_sha256: str) -> tuple[dict[str, object], ...]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("manifest SHA-256 does not match the expected digest")
        return records

    monkeypatch.setattr(phase2b1a, "load_manifest", load)
    monkeypatch.setattr(
        phase2b1a,
        "minimum_cross_split_distances",
        lambda assignments: calls.setdefault("distance_assignments", assignments) and distances,
    )

    def build(
        received_records: object,
        *,
        manifest_sha256: str,
        minimum_distances: object,
        expected: object,
    ) -> dict[str, object]:
        calls["build"] = (
            received_records,
            manifest_sha256,
            minimum_distances,
            expected,
        )
        return audit_payload

    monkeypatch.setattr(phase2b1a, "build_audit", build)
    return manifest, audit_payload, calls


def test_audit_rehashes_72_files_writes_canonical_digest_addressed_output_and_reuses_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assignments = _pilot_assignments()
    records = _manifest_records(assignments, _post_assets(tmp_path, assignments))
    manifest, expected_audit, calls = _patch_audit_services(
        monkeypatch, tmp_path, records
    )

    first = phase2b1a.run_audit(
        tmp_path / "source.json",
        tmp_path,
        manifest,
        confirmed_cloud_storage=True,
    )
    expected_bytes = json.dumps(
        expected_audit,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    audit_digest = hashlib.sha256(expected_bytes).hexdigest()
    audit_path = (
        tmp_path
        / "trustsr"
        / "phase2b1a"
        / "audits"
        / manifest.parent.name
        / "phase2b1a-audit.json"
    )

    assert audit_path.read_bytes() == expected_bytes
    assert calls["build"][1:] == (  # type: ignore[index]
        manifest.parent.name,
        expected_audit["minimum_cross_split_distances"],
        PRODUCTION_EXPECTED_COUNTS,
    )
    assert first == {
        "stage": "audit",
        "digests": {
            "audit_sha256": audit_digest,
            "manifest_sha256": manifest.parent.name,
            "source_sha256": _SOURCE_SHA256,
        },
        "counts": {"samples": 8_000, "components": 6_695, "pilot_pairs": 36, "pilot_geotiffs": 72},
        "reused": False,
    }

    second = phase2b1a.run_audit(
        tmp_path / "source.json",
        tmp_path,
        manifest,
        confirmed_cloud_storage=True,
    )
    assert second == {**first, "reused": True}

    audit_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="existing audit"):
        phase2b1a.run_audit(
            tmp_path / "source.json",
            tmp_path,
            manifest,
            confirmed_cloud_storage=True,
        )
    assert audit_path.read_bytes() == b"tampered"


def test_audit_rejects_execution_before_all_72_files_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assignments = _pilot_assignments()
    assets = _post_assets(tmp_path, assignments)
    records = _manifest_records(assignments, assets)
    missing = tmp_path / "trustsr" / "phase2b1a" / next(iter(assets.values()))[0].relative_path
    missing.unlink()
    manifest, _, calls = _patch_audit_services(monkeypatch, tmp_path, records)

    with pytest.raises(ValueError, match="all 72.*GeoTIFF"):
        phase2b1a.run_audit(
            tmp_path / "source.json",
            tmp_path,
            manifest,
            confirmed_cloud_storage=True,
        )
    assert "build" not in calls


def test_audit_rejects_an_asset_hash_mismatch_without_rewriting_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assignments = _pilot_assignments()
    assets = _post_assets(tmp_path, assignments)
    records = _manifest_records(assignments, assets)
    target = tmp_path / "trustsr" / "phase2b1a" / next(iter(assets.values()))[0].relative_path
    replacement = b"x" * len(target.read_bytes())
    target.write_bytes(replacement)
    manifest, _, calls = _patch_audit_services(monkeypatch, tmp_path, records)

    with pytest.raises(ValueError, match="SHA-256"):
        phase2b1a.run_audit(
            tmp_path / "source.json",
            tmp_path,
            manifest,
            confirmed_cloud_storage=True,
        )
    assert target.read_bytes() == replacement
    assert "build" not in calls


def test_audit_rejects_an_asset_outside_its_exact_sample_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assignments = _pilot_assignments()
    assets = _post_assets(tmp_path, assignments)
    records = list(_manifest_records(assignments, assets))
    first = dict(records[0])
    first["lr_asset"] = {
        **first["lr_asset"],  # type: ignore[dict-item]
        "relative_path": "source/not-a-pilot/lr.tif",
    }
    records[0] = first
    manifest, _, calls = _patch_audit_services(monkeypatch, tmp_path, tuple(records))

    with pytest.raises(ValueError, match="exact pilot sample layout"):
        phase2b1a.run_audit(
            tmp_path / "source.json",
            tmp_path,
            manifest,
            confirmed_cloud_storage=True,
        )
    assert "build" not in calls


def test_pilot_rejects_a_digest_manifest_reached_through_an_outside_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    storage_root = tmp_path / "storage"
    phase_root = storage_root / "trustsr" / "phase2b1a"
    phase_root.mkdir(parents=True)
    outside_manifests = tmp_path / "outside" / "manifests"
    payload = b"outside post-manifest\n"
    digest = hashlib.sha256(payload).hexdigest()
    outside_manifest = outside_manifests / digest / "samples.jsonl"
    outside_manifest.parent.mkdir(parents=True)
    outside_manifest.write_bytes(payload)
    (phase_root / "manifests").symlink_to(outside_manifests, target_is_directory=True)
    supplied = phase_root / "manifests" / digest / "samples.jsonl"
    monkeypatch.setattr(phase2b1a, "load_dataset_source", lambda _: _source())
    monkeypatch.setattr(
        phase2b1a,
        "verify_crosssensor",
        lambda path, spec: VerifiedSourceObject(path, spec.size_bytes, spec.sha256),
    )

    with pytest.raises(ValueError, match="escapes storage_root"):
        phase2b1a.run_pilot(
            storage_root / "source.json",
            storage_root,
            supplied,
            confirmed_cloud_storage=True,
        )


def test_pilot_rejects_an_output_tree_symlinked_outside_storage_before_extraction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    records = _manifest_records(_pilot_assignments())
    manifest, _, state = _patch_pilot_services(monkeypatch, storage_root, records)
    outside = tmp_path / "outside"
    outside.mkdir()
    pilot_root = storage_root / "trustsr" / "phase2b1a" / "pilot-v1"
    pilot_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes storage_root"):
        phase2b1a.run_pilot(
            storage_root / "source.json",
            storage_root,
            manifest,
            confirmed_cloud_storage=True,
        )
    assert state["calls"]["extract"] == []  # type: ignore[index]
    assert tuple(outside.iterdir()) == ()
