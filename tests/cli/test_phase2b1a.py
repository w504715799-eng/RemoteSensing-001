"""Offline tests for the restartable Phase 2B1A stage controller."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

import trustsr.cli.phase2b1a as phase2b1a
import trustsr.data.crosssensor_source as crosssensor_source
import trustsr.data.taco_v1_adapter as taco_v1_adapter
from trustsr.data.crosssensor_manifest import (
    PRODUCTION_EXPECTED_COUNTS,
    ExpectedCounts,
    ExtractedAsset,
    ManifestArtifact,
)
from trustsr.data.crosssensor_schema import AcquisitionTimes, CrosssensorSample
from trustsr.data.crosssensor_source import VerifiedSourceObject
from trustsr.data.pilot_sampling import select_pilot
from trustsr.data.provenance import DatasetSource, LfsObject
from trustsr.data.spatial_split import AssignedSample

_SOURCE_REVISION = "c370504201072fdb1dd388013ab8c0fc7d00a57e"
_SOURCE_NAME = "sen2naipv2-crosssensor.taco"
_SOURCE_SHA256 = "c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5"
_SOURCE_SIZE = 9_717_583_850

_STAGE_SEQUENCE_STRATA = (
    ("synthetic-development--1-0-0", "development", -1, 0.8),
    ("synthetic-development--1-1-1", "development", -1, 0.89),
    ("synthetic-development--1-2-0", "development", -1, 0.91),
    ("synthetic-development--1-3-1", "development", -1, 0.94),
    ("synthetic-development-0-0-0", "development", 0, 0.8),
    ("synthetic-development-0-1-2", "development", 0, 0.89),
    ("synthetic-development-0-2-0", "development", 0, 0.91),
    ("synthetic-development-0-3-2", "development", 0, 0.94),
    ("synthetic-development-1-0-0", "development", 1, 0.8),
    ("synthetic-development-1-1-1", "development", 1, 0.89),
    ("synthetic-development-1-2-5", "development", 1, 0.91),
    ("synthetic-development-1-3-0", "development", 1, 0.94),
    ("synthetic-calibration--1-0-8", "calibration", -1, 0.8),
    ("synthetic-calibration--1-1-7", "calibration", -1, 0.89),
    ("synthetic-calibration--1-2-4", "calibration", -1, 0.91),
    ("synthetic-calibration--1-3-2", "calibration", -1, 0.94),
    ("synthetic-calibration-0-0-5", "calibration", 0, 0.8),
    ("synthetic-calibration-0-1-4", "calibration", 0, 0.89),
    ("synthetic-calibration-0-2-5", "calibration", 0, 0.91),
    ("synthetic-calibration-0-3-1", "calibration", 0, 0.94),
    ("synthetic-calibration-1-0-2", "calibration", 1, 0.8),
    ("synthetic-calibration-1-1-6", "calibration", 1, 0.89),
    ("synthetic-calibration-1-2-0", "calibration", 1, 0.91),
    ("synthetic-calibration-1-3-1", "calibration", 1, 0.94),
    ("synthetic-internal_test--1-0-3", "internal_test", -1, 0.8),
    ("synthetic-internal_test--1-1-4", "internal_test", -1, 0.89),
    ("synthetic-internal_test--1-2-9", "internal_test", -1, 0.91),
    ("synthetic-internal_test--1-3-4", "internal_test", -1, 0.94),
    ("synthetic-internal_test-0-0-0", "internal_test", 0, 0.8),
    ("synthetic-internal_test-0-1-2", "internal_test", 0, 0.89),
    ("synthetic-internal_test-0-2-0", "internal_test", 0, 0.91),
    ("synthetic-internal_test-0-3-8", "internal_test", 0, 0.94),
    ("synthetic-internal_test-1-0-2", "internal_test", 1, 0.8),
    ("synthetic-internal_test-1-1-1", "internal_test", 1, 0.89),
    ("synthetic-internal_test-1-2-2", "internal_test", 1, 0.91),
    ("synthetic-internal_test-1-3-0", "internal_test", 1, 0.94),
)


class _StageSequenceRows:
    def __init__(self, rows: tuple[dict[str, object], ...]) -> None:
        self._rows = rows

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._rows[index]


class _StageSequenceNested:
    def __init__(
        self, lr_payload: bytes, hr_payload: bytes, days_between: int
    ) -> None:
        self._payloads = (lr_payload, hr_payload)
        lr_time_start = {
            -1: "2020-01-03T10:00:00Z",
            0: "2020-01-02T10:00:00Z",
            1: "2020-01-01T10:00:00Z",
        }[days_between]
        self.iloc = _StageSequenceRows(
            (
                {"stac:time_start": lr_time_start},
                {"stac:time_start": "2020-01-02T10:00:00Z"},
            )
        )

    def __len__(self) -> int:
        return len(self._payloads)

    def read(self, index: int) -> bytes:
        return self._payloads[index]


class _StageSequenceTop:
    def __init__(
        self,
        records: tuple[dict[str, object], ...],
        nested: tuple[_StageSequenceNested, ...],
    ) -> None:
        self._records = records
        self._nested = nested
        self.read_calls: list[int] = []

    def to_dict(self, *, orient: str) -> list[dict[str, object]]:
        assert orient == "records"
        return [dict(record) for record in self._records]

    def read(self, source_index: int) -> _StageSequenceNested:
        assert 0 <= source_index < len(self._records)
        self.read_calls.append(source_index)
        return self._nested[source_index]


class _StageSequenceReader:
    def __init__(self, top: _StageSequenceTop) -> None:
        self.top = top
        self.load_metadata_calls: list[str] = []
        self.load_calls: list[str] = []

    def load_metadata(self, path: str) -> dict[str, object]:
        self.load_metadata_calls.append(path)
        return {"taco_version": "0.4.0"}

    def load(self, path: str) -> _StageSequenceTop:
        self.load_calls.append(path)
        return self.top

    def clear_extraction_calls(self) -> None:
        self.top.read_calls.clear()
        self.load_metadata_calls.clear()
        self.load_calls.clear()


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
        "tortilla:data_split": "train",
        **{f"extra:{index}": index for index in range(13)},
    }
    selected = dict(list(columns.items())[:column_count])
    return tuple(selected for _ in range(8_000))


def _stage_sequence_top_records() -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for index, (sample_id, _, days_between, correlation) in enumerate(
        _STAGE_SEQUENCE_STRATA
    ):
        longitude = float(-170 + (index % 12) * 30)
        latitude = float(-70 + (index // 12) * 70)
        records.append(
            {
                "tortilla:id": sample_id,
                "stac:crs": "EPSG:32618",
                "stac:geotransform": (
                    10.0,
                    0.0,
                    500000.0,
                    0.0,
                    -10.0,
                    400000.0,
                ),
                "stac:raster_shape": (130, 130),
                "stac:time_start": "2020-01-02T10:00:00Z",
                "stac:centroid": f"POINT ({longitude} {latitude})",
                "rai:admin0": "Synthetic",
                "rai:admin1": None,
                "rai:admin2": None,
                "days_between": days_between,
                "correlation": correlation,
                "scale_factor": 4,
                "tortilla:data_split": "train",
                **{f"extra:{extra_index}": extra_index for extra_index in range(13)},
            }
        )
    return tuple(records)


def _synthetic_geotiff(dimension: int, resolution: float, value: int) -> bytes:
    transform = from_origin(500000.0, 400000.0, resolution, resolution)
    pixels = np.full((4, dimension, dimension), value, dtype=np.uint16)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=dimension,
            height=dimension,
            count=4,
            dtype="uint16",
            crs="EPSG:32618",
            transform=transform,
            compress="deflate",
        ) as dataset:
            dataset.write(pixels)
            dataset.descriptions = ("B04", "B03", "B02", "B08")
        return memory.read()


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

    def load_records(
        path: Path,
    ) -> tuple[tuple[dict[str, object], ...], tuple[AcquisitionTimes, ...]]:
        calls["reader_path"] = path
        records = _top_records(column_count)
        return records, tuple(
            AcquisitionTimes(
                "2020-01-02T10:00:00Z", "2020-01-02T10:00:00Z"
            )
            for _ in records
        )

    monkeypatch.setattr(phase2b1a, "load_crosssensor_metadata", load_records)

    def normalize(
        records: object, *, acquisition_times: object, expected_count: int
    ) -> tuple[str, ...]:
        calls["normalize"] = (records, acquisition_times, expected_count)
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
            "source_object_name": _SOURCE_NAME,
            "source_object_size_bytes": _SOURCE_SIZE,
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
    assert len(calls["normalize"][1]) == 8_000  # type: ignore[arg-type, index]
    assert calls["normalize"][2] == 8_000  # type: ignore[index]
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


def test_manifest_rejects_inconsistent_audit_evidence_before_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catch a source-evidence preflight placed after manifest writes or commits."""
    digest, calls = _patch_manifest_services(monkeypatch, tmp_path)
    monkeypatch.setattr(
        phase2b1a,
        "audit_source_identity",
        lambda: {
            "source_revision": _SOURCE_REVISION,
            "source_object_name": _SOURCE_NAME,
            "source_object_size_bytes": _SOURCE_SIZE - 1,
            "source_object_sha256": _SOURCE_SHA256,
        },
        raising=False,
    )

    with pytest.raises(ValueError, match="audit source identity"):
        phase2b1a.run_manifest(
            tmp_path / "source.json", tmp_path, confirmed_cloud_storage=True
        )
    assert "write" not in calls
    assert not (tmp_path / "trustsr" / "phase2b1a" / "manifests" / digest).exists()


def _pilot_assignments() -> tuple[AssignedSample, ...]:
    correlations = (0.8, 0.89, 0.91, 0.94)
    assignments: list[AssignedSample] = []
    source_index = 0
    for split in ("development", "calibration", "internal_test"):
        for days_between in (-1, 0, 1):
            for bin_index, correlation in enumerate(correlations):
                sample_id = f"{split}-{days_between}-{bin_index}"
                lr_time_start = {
                    -1: "2020-01-03T10:00:00Z",
                    0: "2020-01-02T10:00:00Z",
                    1: "2020-01-01T10:00:00Z",
                }[days_between]
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
                            lr_time_start=lr_time_start,
                            hr_time_start="2020-01-02T10:00:00Z",
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


def _asset(
    relative_path: str,
    payload: bytes,
    *,
    time_start: str = "2020-01-02T10:00:00Z",
) -> ExtractedAsset:
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
        time_start=time_start,
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
                "lr_time_start": sample.lr_time_start,
                "hr_time_start": sample.hr_time_start,
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
    calls: dict[str, object] = {"extract": [], "post_records": None}
    monkeypatch.setattr(phase2b1a, "load_dataset_source", lambda _: _source())
    monkeypatch.setattr(
        phase2b1a,
        "verify_crosssensor",
        lambda path, spec: VerifiedSourceObject(path, spec.size_bytes, spec.sha256),
    )

    def load(path: Path, *, expected_sha256: str) -> tuple[dict[str, object], ...]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("manifest SHA-256 does not match the expected digest")
        if expected_sha256 == post_digest and calls["post_records"] is not None:
            return calls["post_records"]  # type: ignore[return-value]
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
            "source_revision": _SOURCE_REVISION,
            "source_object_name": _SOURCE_NAME,
            "source_object_size_bytes": _SOURCE_SIZE,
            "source_object_sha256": _SOURCE_SHA256,
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
        record = next(
            record for record in records if record["source_index"] == source_index
        )
        return (
            _asset(
                "lr.tif", lr_payload, time_start=str(record["lr_time_start"])
            ),
            _asset(
                "hr.tif", hr_payload, time_start=str(record["hr_time_start"])
            ),
        )

    monkeypatch.setattr(phase2b1a, "extract_pair", extract)

    def write(
        path: Path,
        assignments: object,
        choices: object,
        assets: dict[str, tuple[ExtractedAsset, ExtractedAsset]],
    ) -> ManifestArtifact:
        calls["write"] = (assignments, choices, assets)
        calls["post_records"] = _manifest_records(
            tuple(assignments), assets  # type: ignore[arg-type]
        )
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
    assert len(state["calls"]["extract"]) == 36  # type: ignore[index]

    first_output = state["calls"]["extract"][0][2] / "lr.tif"  # type: ignore[index]
    first_output.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="byte size|SHA-256"):
        phase2b1a.run_pilot(
            tmp_path / "source.json",
            tmp_path,
            input_manifest,
            confirmed_cloud_storage=True,
        )
    assert first_output.read_bytes() == b"tampered"
    assert len(state["calls"]["extract"]) == 36  # type: ignore[index]


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


def test_pilot_rejects_extracted_times_that_contradict_the_complete_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assignments = _pilot_assignments()
    records = _manifest_records(assignments)
    manifest, _, state = _patch_pilot_services(monkeypatch, tmp_path, records)

    def inconsistent_extract(
        taco_path: Path,
        source_index: int,
        output_root: Path,
        bands: tuple[str, ...],
    ) -> tuple[ExtractedAsset, ExtractedAsset]:
        del taco_path, source_index, output_root, bands
        return (
            _asset("lr.tif", b"lr", time_start="2020-01-09T10:00:00Z"),
            _asset("hr.tif", b"hr", time_start="2020-01-02T10:00:00Z"),
        )

    monkeypatch.setattr(phase2b1a, "extract_pair", inconsistent_extract)

    with pytest.raises(ValueError, match="LR time_start.*manifest lr_time_start"):
        phase2b1a.run_pilot(
            tmp_path / "source.json",
            tmp_path,
            manifest,
            confirmed_cloud_storage=True,
        )
    assert "write" not in state["calls"]  # type: ignore[operator]


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


def test_pilot_rejects_inconsistent_audit_evidence_before_output_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catch an audit-evidence preflight placed after pilot output side effects."""
    records = _manifest_records(_pilot_assignments())
    manifest, _, state = _patch_pilot_services(monkeypatch, tmp_path, records)
    output_preflights: list[Path] = []
    monkeypatch.setattr(
        phase2b1a,
        "audit_source_identity",
        lambda: {
            "source_revision": _SOURCE_REVISION,
            "source_object_name": _SOURCE_NAME,
            "source_object_size_bytes": _SOURCE_SIZE - 1,
            "source_object_sha256": _SOURCE_SHA256,
        },
        raising=False,
    )
    monkeypatch.setattr(
        phase2b1a,
        "_require_absent_or_complete_pair",
        output_preflights.append,
    )

    with pytest.raises(ValueError, match="audit source identity"):
        phase2b1a.run_pilot(
            tmp_path / "source.json",
            tmp_path,
            manifest,
            confirmed_cloud_storage=True,
        )
    assert output_preflights == []
    assert state["calls"]["extract"] == []  # type: ignore[index]
    assert "write" not in state["calls"]  # type: ignore[operator]
    assert not (tmp_path / "trustsr" / "phase2b1a" / "pilot-v1").exists()


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


def test_pilot_resumes_complete_pairs_before_the_post_manifest_is_committed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assignments = _pilot_assignments()
    records = _manifest_records(assignments)
    manifest, _, state = _patch_pilot_services(monkeypatch, tmp_path, records)
    completed = assignments[:3]
    for assignment in completed:
        sample = assignment.sample
        output_root = (
            tmp_path
            / "trustsr"
            / "phase2b1a"
            / "pilot-v1"
            / assignment.split
            / sample.sample_id
        )
        output_root.mkdir(parents=True)
        (output_root / "lr.tif").write_bytes(f"lr-{sample.source_index}".encode())
        (output_root / "hr.tif").write_bytes(f"hr-{sample.source_index}".encode())

    result = phase2b1a.run_pilot(
        tmp_path / "source.json",
        tmp_path,
        manifest,
        confirmed_cloud_storage=True,
    )

    assert result["reused"] is False
    assert len(state["calls"]["extract"]) == 36  # type: ignore[index]
    for assignment in assignments:
        sample = assignment.sample
        output_root = (
            tmp_path
            / "trustsr"
            / "phase2b1a"
            / "pilot-v1"
            / assignment.split
            / sample.sample_id
        )
        assert (output_root / "lr.tif").read_bytes() == (
            f"lr-{sample.source_index}".encode()
        )
        assert (output_root / "hr.tif").read_bytes() == (
            f"hr-{sample.source_index}".encode()
        )


def test_pilot_never_replaces_a_different_complete_pair_without_a_post_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assignments = _pilot_assignments()
    records = _manifest_records(assignments)
    manifest, _, state = _patch_pilot_services(monkeypatch, tmp_path, records)
    for assignment in assignments:
        sample = assignment.sample
        output_root = (
            tmp_path
            / "trustsr"
            / "phase2b1a"
            / "pilot-v1"
            / assignment.split
            / sample.sample_id
        )
        output_root.mkdir(parents=True)
        (output_root / "lr.tif").write_bytes(f"lr-{sample.source_index}".encode())
        (output_root / "hr.tif").write_bytes(f"hr-{sample.source_index}".encode())
    tampered_root = (
        tmp_path
        / "trustsr"
        / "phase2b1a"
        / "pilot-v1"
        / assignments[0].split
        / assignments[0].sample.sample_id
    )
    tampered_payload = b"tampered complete pair"
    (tampered_root / "lr.tif").write_bytes(tampered_payload)

    with pytest.raises(ValueError, match="different bytes"):
        phase2b1a.run_pilot(
            tmp_path / "source.json",
            tmp_path,
            manifest,
            confirmed_cloud_storage=True,
        )

    assert (tampered_root / "lr.tif").read_bytes() == tampered_payload
    assert "write" not in state["calls"]  # type: ignore[operator]


def _post_assets(
    tmp_path: Path, assignments: tuple[AssignedSample, ...]
) -> dict[str, tuple[ExtractedAsset, ExtractedAsset]]:
    assets: dict[str, tuple[ExtractedAsset, ExtractedAsset]] = {}
    for assignment in assignments:
        sample = assignment.sample
        prefix = f"pilot-v1/{assignment.split}/{sample.sample_id}"
        lr_payload = f"lr-{sample.source_index}".encode()
        hr_payload = f"hr-{sample.source_index}".encode()
        lr = replace(
            _asset("lr.tif", lr_payload, time_start=sample.lr_time_start),
            relative_path=f"{prefix}/lr.tif",
        )
        hr = replace(
            _asset("hr.tif", hr_payload, time_start=sample.hr_time_start),
            relative_path=f"{prefix}/hr.tif",
        )
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
        "source_object_name": _SOURCE_NAME,
        "source_object_size_bytes": _SOURCE_SIZE,
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


@pytest.mark.parametrize(
    "source_size",
    (_SOURCE_SIZE - 1, float(_SOURCE_SIZE), True, False),
    ids=("wrong-value", "float", "true", "false"),
)
def test_audit_rejects_a_payload_with_inconsistent_frozen_source_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_size: object,
) -> None:
    """Catch an audit writer that records invalid source bytes or representation."""
    assignments = _pilot_assignments()
    records = _manifest_records(assignments, _post_assets(tmp_path, assignments))
    manifest, expected_audit, _ = _patch_audit_services(monkeypatch, tmp_path, records)
    expected_audit["source_object_size_bytes"] = source_size

    with pytest.raises(ValueError, match="audit source identity"):
        phase2b1a.run_audit(
            tmp_path / "source.json",
            tmp_path,
            manifest,
            confirmed_cloud_storage=True,
        )
    assert not (
        tmp_path
        / "trustsr"
        / "phase2b1a"
        / "audits"
        / manifest.parent.name
    ).exists()


@pytest.mark.parametrize(
    "source_size",
    (_SOURCE_SIZE - 1, float(_SOURCE_SIZE), True, False),
    ids=("wrong-value", "float", "true", "false"),
)
def test_audit_rejects_a_verified_source_that_disagrees_with_frozen_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_size: object,
) -> None:
    """Catch a future verifier that returns bytes inconsistent with the frozen audit."""
    assignments = _pilot_assignments()
    records = _manifest_records(assignments, _post_assets(tmp_path, assignments))
    manifest, _, _ = _patch_audit_services(monkeypatch, tmp_path, records)
    monkeypatch.setattr(
        phase2b1a,
        "verify_crosssensor",
        lambda path, spec: VerifiedSourceObject(path, source_size, spec.sha256),
    )

    with pytest.raises(ValueError, match="audit source identity"):
        phase2b1a.run_audit(
            tmp_path / "source.json",
            tmp_path,
            manifest,
            confirmed_cloud_storage=True,
        )
    assert not (
        tmp_path
        / "trustsr"
        / "phase2b1a"
        / "audits"
        / manifest.parent.name
    ).exists()


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


def test_phase2b1a_synthetic_stage_sequence_preserves_restart_and_integrity_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    source_path = storage_root / "source.json"
    source_path.write_bytes(
        (Path(__file__).parents[2] / "artifacts/datasets/sen2naipv2-source-v1.json").read_bytes()
    )
    source_payload = b"tiny synthetic TACO v1 boundary\n"
    transport_calls: list[tuple[str, ...]] = []

    def verify_tiny_source(path: Path, object_spec: LfsObject) -> VerifiedSourceObject:
        if path.read_bytes() != source_payload:
            raise ValueError("synthetic source bytes do not match")
        return VerifiedSourceObject(path, object_spec.size_bytes, object_spec.sha256)

    def fake_transport(arguments: tuple[str, ...], **options: object) -> None:
        assert options == {"check": True, "text": True}
        partial_path = Path(arguments[arguments.index("--output") + 1])
        partial_path.write_bytes(source_payload)
        transport_calls.append(arguments)

    real_acquire = phase2b1a.acquire_crosssensor

    def acquire_tiny_source(
        source: DatasetSource,
        root: Path,
        transport_url: str,
        *,
        confirmed_cloud_storage: bool,
    ) -> VerifiedSourceObject:
        return real_acquire(
            source,
            root,
            transport_url,
            confirmed_cloud_storage=confirmed_cloud_storage,
            runner=fake_transport,
        )

    monkeypatch.setattr(crosssensor_source, "verify_crosssensor", verify_tiny_source)
    monkeypatch.setattr(
        crosssensor_source,
        "require_free_space",
        lambda _root, *, minimum_bytes: None,
    )
    monkeypatch.setattr(phase2b1a, "acquire_crosssensor", acquire_tiny_source)
    monkeypatch.setattr(phase2b1a, "verify_crosssensor", verify_tiny_source)

    real_normalize_top_level = phase2b1a.normalize_top_level

    def normalize_scaled_top_level(
        records: object, *, acquisition_times: object, expected_count: int
    ) -> tuple[CrosssensorSample, ...]:
        assert expected_count == 8_000
        return real_normalize_top_level(
            records,
            acquisition_times=acquisition_times,  # type: ignore[arg-type]
            expected_count=36,
        )  # type: ignore[arg-type]

    monkeypatch.setattr(phase2b1a, "normalize_top_level", normalize_scaled_top_level)

    real_build_audit = phase2b1a.build_audit
    audit_source_identities: list[tuple[object, object, object, object]] = []
    synthetic_expected = ExpectedCounts(
        samples=36,
        components=36,
        development_samples=12,
        calibration_samples=12,
        internal_test_samples=12,
        development_components=12,
        calibration_components=12,
        internal_test_components=12,
    )

    def build_scaled_audit(
        records: object,
        *,
        manifest_sha256: str,
        minimum_distances: object,
        expected: object,
    ) -> dict[str, object]:
        assert expected == PRODUCTION_EXPECTED_COUNTS
        audit = real_build_audit(
            records,  # type: ignore[arg-type]
            manifest_sha256=manifest_sha256,
            minimum_distances=minimum_distances,  # type: ignore[arg-type]
            expected=synthetic_expected,
        )
        audit_source_identities.append(
            (
                audit["source_revision"],
                audit["source_object_name"],
                audit["source_object_size_bytes"],
                audit["source_object_sha256"],
            )
        )
        return audit

    monkeypatch.setattr(phase2b1a, "build_audit", build_scaled_audit)

    payloads = {
        "lr.tif": _synthetic_geotiff(130, 10.0, 100),
        "hr.tif": _synthetic_geotiff(520, 2.5, 120),
    }
    stage_records = _stage_sequence_top_records()
    nested = tuple(
        _StageSequenceNested(
            payloads["lr.tif"], payloads["hr.tif"], int(record["days_between"])
        )
        for record in stage_records
    )
    reader = _StageSequenceReader(_StageSequenceTop(stage_records, nested))
    monkeypatch.setattr(taco_v1_adapter, "require_tacoreader_v1", lambda: reader)
    real_atomic_write = taco_v1_adapter.atomic_write_bytes
    adapter_write_calls: list[Path] = []

    def counting_atomic_write(path: Path, payload: bytes) -> None:
        adapter_write_calls.append(path)
        real_atomic_write(path, payload)

    monkeypatch.setattr(taco_v1_adapter, "atomic_write_bytes", counting_atomic_write)

    first_download = phase2b1a.run_download(
        source_path,
        storage_root,
        "https://download.invalid/synthetic.taco",
        confirmed_cloud_storage=True,
    )
    first_manifest = phase2b1a.run_manifest(
        source_path, storage_root, confirmed_cloud_storage=True
    )
    pre_manifest_digest = str(first_manifest["digests"]["manifest_sha256"])  # type: ignore[index]
    pre_manifest = (
        storage_root
        / "trustsr"
        / "phase2b1a"
        / "manifests"
        / pre_manifest_digest
        / "samples.jsonl"
    )
    taco_path = (
        storage_root
        / "trustsr"
        / "phase2b1a"
        / "source"
        / _SOURCE_SHA256
        / _SOURCE_NAME
    )
    pre_records = phase2b1a.load_manifest(
        pre_manifest, expected_sha256=pre_manifest_digest
    )
    pre_times = {
        record["sample_id"]: (record["lr_time_start"], record["hr_time_start"])
        for record in pre_records
    }
    assert len(pre_times) == 36
    interrupted_records = tuple(
        record for record in pre_records if record["pilot"] is not None
    )[:3]
    interrupted_payloads: dict[Path, bytes] = {}
    for record in interrupted_records:
        output_root = (
            storage_root
            / "trustsr"
            / "phase2b1a"
            / "pilot-v1"
            / str(record["split"])
            / str(record["sample_id"])
        )
        taco_v1_adapter.extract_pair(
            taco_path,
            int(record["source_index"]),
            output_root,
            ("B04", "B03", "B02", "B08"),
        )
        interrupted_payloads.update(
            (output_root / name, (output_root / name).read_bytes()) for name in payloads
        )
    assert len(interrupted_payloads) == 6
    reader.clear_extraction_calls()
    adapter_write_calls.clear()

    first_pilot = phase2b1a.run_pilot(
        source_path,
        storage_root,
        pre_manifest,
        confirmed_cloud_storage=True,
    )
    post_manifest_digest = str(first_pilot["digests"]["manifest_sha256"])  # type: ignore[index]
    post_manifest = pre_manifest.parent.parent / post_manifest_digest / "samples.jsonl"
    first_audit = phase2b1a.run_audit(
        source_path,
        storage_root,
        post_manifest,
        confirmed_cloud_storage=True,
    )
    audit_path = (
        storage_root
        / "trustsr"
        / "phase2b1a"
        / "audits"
        / post_manifest_digest
        / "phase2b1a-audit.json"
    )
    first_audit_bytes = audit_path.read_bytes()
    first_audit_digest = hashlib.sha256(first_audit_bytes).hexdigest()
    first_audit_payload = json.loads(first_audit_bytes)
    assert {
        key: first_audit_payload[key]
        for key in (
            "source_revision",
            "source_object_name",
            "source_object_size_bytes",
            "source_object_sha256",
        )
    } == {
        "source_revision": _SOURCE_REVISION,
        "source_object_name": _SOURCE_NAME,
        "source_object_size_bytes": _SOURCE_SIZE,
        "source_object_sha256": _SOURCE_SHA256,
    }
    assert type(first_audit_payload["source_object_size_bytes"]) is int

    post_records = phase2b1a.load_manifest(
        post_manifest, expected_sha256=post_manifest_digest
    )
    selected = tuple(record for record in post_records if record["pilot"] is not None)
    assert len(selected) == 36
    assert {record["sample_id"] for record in selected} == {
        sample_id for sample_id, _, _, _ in _STAGE_SEQUENCE_STRATA
    }
    for record in selected:
        assert (record["lr_time_start"], record["hr_time_start"]) == pre_times[
            record["sample_id"]
        ]
        for kind in ("lr", "hr"):
            asset = record[f"{kind}_asset"]  # type: ignore[assignment]
            relative_path = str(asset["relative_path"])  # type: ignore[index]
            assert relative_path == (
                f"pilot-v1/{record['split']}/{record['sample_id']}/{kind}.tif"
            )
            asset_path = storage_root / "trustsr" / "phase2b1a" / relative_path
            assert asset_path.read_bytes() == payloads[f"{kind}.tif"]
            expected_dimension = 130 if kind == "lr" else 520
            expected_resolution = 10.0 if kind == "lr" else 2.5
            expected_value = 100.0 if kind == "lr" else 120.0
            expected_time = str(record[f"{kind}_time_start"])
            assert asset["size_bytes"] == len(payloads[f"{kind}.tif"])  # type: ignore[index]
            assert asset["sha256"] == hashlib.sha256(  # type: ignore[index]
                payloads[f"{kind}.tif"]
            ).hexdigest()
            assert asset["shape"] == [4, expected_dimension, expected_dimension]  # type: ignore[index]
            assert asset["dtype"] == "uint16"  # type: ignore[index]
            assert asset["crs"] == "EPSG:32618"  # type: ignore[index]
            assert asset["transform"] == [  # type: ignore[index]
                expected_resolution,
                0.0,
                500000.0,
                0.0,
                -expected_resolution,
                400000.0,
            ]
            assert asset["nodata"] is None  # type: ignore[index]
            assert asset["minimum"] == expected_value  # type: ignore[index]
            assert asset["maximum"] == expected_value  # type: ignore[index]
            assert asset["time_start"] == expected_time  # type: ignore[index]

    second_download = phase2b1a.run_download(
        source_path,
        storage_root,
        "https://download.invalid/synthetic.taco",
        confirmed_cloud_storage=True,
    )
    second_manifest = phase2b1a.run_manifest(
        source_path, storage_root, confirmed_cloud_storage=True
    )
    second_pilot = phase2b1a.run_pilot(
        source_path,
        storage_root,
        pre_manifest,
        confirmed_cloud_storage=True,
    )
    second_audit = phase2b1a.run_audit(
        source_path,
        storage_root,
        post_manifest,
        confirmed_cloud_storage=True,
    )

    assert [
        first_download["stage"],
        first_manifest["stage"],
        first_pilot["stage"],
        first_audit["stage"],
    ] == ["download", "manifest", "pilot", "audit"]
    assert [
        first_download["reused"],
        first_manifest["reused"],
        first_pilot["reused"],
        first_audit["reused"],
    ] == [False, False, False, False]
    assert [
        second_download["reused"],
        second_manifest["reused"],
        second_pilot["reused"],
        second_audit["reused"],
    ] == [True, True, True, True]
    assert len(transport_calls) == 1
    assert sorted(reader.top.read_calls) == [index for index in range(36) for _ in range(2)]
    assert len(adapter_write_calls) == 66
    assert all(
        path.resolve().is_relative_to(storage_root.resolve())
        for path in adapter_write_calls
    )
    assert all(path.read_bytes() == payload for path, payload in interrupted_payloads.items())
    assert audit_path.read_bytes() == first_audit_bytes
    assert second_audit["digests"]["audit_sha256"] == first_audit_digest  # type: ignore[index]
    assert second_audit["digests"] == first_audit["digests"]
    assert audit_source_identities == [
        (_SOURCE_REVISION, _SOURCE_NAME, _SOURCE_SIZE, _SOURCE_SHA256)
    ] * 6

    tampered_manifest = b"tampered manifest\n"
    pre_manifest.write_bytes(tampered_manifest)
    with pytest.raises(ValueError, match="parent.*actual SHA-256"):
        phase2b1a.run_pilot(
            source_path,
            storage_root,
            pre_manifest,
            confirmed_cloud_storage=True,
        )
    assert pre_manifest.read_bytes() == tampered_manifest

    first_asset = storage_root / "trustsr" / "phase2b1a" / str(
        selected[0]["lr_asset"]["relative_path"]  # type: ignore[index]
    )
    tampered_asset = b"x" * len(first_asset.read_bytes())
    first_asset.write_bytes(tampered_asset)
    with pytest.raises(ValueError, match="SHA-256"):
        phase2b1a.run_audit(
            source_path,
            storage_root,
            post_manifest,
            confirmed_cloud_storage=True,
        )
    assert first_asset.read_bytes() == tampered_asset

    created_paths = tuple(tmp_path.rglob("*"))
    assert created_paths
    assert all(
        path.resolve().is_relative_to(storage_root.resolve()) for path in created_paths
    )
