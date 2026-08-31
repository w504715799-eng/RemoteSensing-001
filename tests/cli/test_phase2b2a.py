"""Restartable CPU-only Phase 2B2-A input audit command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from trustsr.cli import phase2b2a
from trustsr.contracts import SRPair
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
)


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "sample_id": f"sample-{split}-{bin_index}",
            "split": split,
            "spatial_group_id": f"group-{split}-{bin_index}",
            "days_between": -1,
            "correlation_bin": bin_index,
            "selection_round": 1,
        }
        for split in ("calibration", "development", "internal_test")
        for bin_index in range(4)
    )


def _loaded(record: dict[str, object]) -> LoadedCrosssensorPair:
    sample_id = str(record["sample_id"])
    split = str(record["split"])
    bin_index = int(record["correlation_bin"])
    return LoadedCrosssensorPair(
        pair=SRPair(
            sample_id=sample_id,
            source=f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}",
            lr=torch.full((4, 128, 128), 0.25, dtype=torch.float32),
            hr=torch.full((4, 512, 512), 0.5, dtype=torch.float32),
            scale=4,
        ),
        metadata=CrosssensorPairMetadata(
            manifest_sha256=POST_MANIFEST_SHA256,
            sample_id=sample_id,
            split=split,
            spatial_group_id=str(record["spatial_group_id"]),
            days_between=-1,
            correlation_bin=bin_index,
            selection_round=1,
            lr_asset_sha256=f"{bin_index + 1:x}" * 64,
            hr_asset_sha256=f"{bin_index + 5:x}" * 64,
            lr_crop_transform=(10.0, 0.0, 500010.0, 0.0, -10.0, 399990.0),
            hr_crop_transform=(2.5, 0.0, 500010.0, 0.0, -2.5, 399990.0),
            crop_bounds=(500010.0, 398710.0, 501290.0, 399990.0),
            crop_policy=CROP_POLICY,
            normalization_policy=NORMALIZATION_POLICY,
        ),
    )


@dataclass
class _State:
    manifest: Path
    expected_ids: list[str]
    loaded_sample_ids: list[str]


def _patch_valid_services(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> _State:
    records = _records()
    expected_ids = [str(record["sample_id"]) for record in records]
    loaded_sample_ids: list[str] = []
    manifest = tmp_path / "samples.jsonl"

    monkeypatch.setattr(
        phase2b2a,
        "require_cloud_confirmation",
        lambda root, confirmed: root,
    )
    monkeypatch.setattr(
        phase2b2a,
        "load_crosssensor_records",
        lambda root, path, *, expected_sha256: records,
    )

    def load_pair(
        root: Path, record: dict[str, object], *, manifest_sha256: str
    ) -> LoadedCrosssensorPair:
        assert root == tmp_path
        assert manifest_sha256 == POST_MANIFEST_SHA256
        loaded_sample_ids.append(str(record["sample_id"]))
        return _loaded(record)

    monkeypatch.setattr(phase2b2a, "load_crosssensor_pair", load_pair)
    return _State(manifest, expected_ids, loaded_sample_ids)


def test_parser_requires_explicit_frozen_manifest_and_confirmation() -> None:
    args = phase2b2a.build_parser().parse_args(
        [
            "audit-inputs",
            "--storage-root",
            "/persistent",
            "--selection-manifest",
            "/persistent/samples.jsonl",
            "--selection-manifest-sha256",
            POST_MANIFEST_SHA256,
            "--confirm-cloud-storage",
        ]
    )

    assert args.stage == "audit-inputs"
    assert args.selection_manifest_sha256 == POST_MANIFEST_SHA256
    assert args.confirm_cloud_storage is True


def test_invalid_manifest_stops_before_audit_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        phase2b2a,
        "require_cloud_confirmation",
        lambda root, confirmed: root,
    )

    with pytest.raises(ValueError, match="frozen post-manifest"):
        phase2b2a.run_audit_inputs(
            tmp_path,
            tmp_path / "missing.jsonl",
            "0" * 64,
            confirmed_cloud_storage=True,
        )

    assert not (tmp_path / "trustsr" / "phase2b2a").exists()


def test_run_audit_inputs_loads_12_pairs_twice_and_reuses_identical_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = _patch_valid_services(monkeypatch, tmp_path)

    first = phase2b2a.run_audit_inputs(
        tmp_path,
        state.manifest,
        POST_MANIFEST_SHA256,
        confirmed_cloud_storage=True,
    )
    second = phase2b2a.run_audit_inputs(
        tmp_path,
        state.manifest,
        POST_MANIFEST_SHA256,
        confirmed_cloud_storage=True,
    )

    assert state.loaded_sample_ids == state.expected_ids * 4
    assert first["counts"] == {"smoke_pairs": 12, "smoke_geotiffs": 24}
    assert first["digests"]["audit_sha256"] == second["digests"]["audit_sha256"]
    assert first["reused"] is False
    assert second["reused"] is True
    audit_path = (
        tmp_path
        / "trustsr"
        / "phase2b2a"
        / "input-audits"
        / POST_MANIFEST_SHA256
        / "phase2b2a-input-audit.json"
    )
    assert audit_path.is_file()


@pytest.mark.parametrize("damage", ["extra", "different-bytes", "symlink"])
def test_existing_audit_must_be_the_identical_single_regular_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, damage: str
) -> None:
    state = _patch_valid_services(monkeypatch, tmp_path)
    phase2b2a.run_audit_inputs(
        tmp_path,
        state.manifest,
        POST_MANIFEST_SHA256,
        confirmed_cloud_storage=True,
    )
    audit_dir = (
        tmp_path
        / "trustsr"
        / "phase2b2a"
        / "input-audits"
        / POST_MANIFEST_SHA256
    )
    audit_path = audit_dir / "phase2b2a-input-audit.json"
    if damage == "extra":
        (audit_dir / "extra").write_text("unexpected", encoding="utf-8")
    elif damage == "different-bytes":
        audit_path.write_bytes(b"different")
    else:
        original = audit_path.with_name("original.json")
        audit_path.rename(original)
        audit_path.symlink_to(original)

    with pytest.raises(ValueError, match="existing Phase 2B2-A audit"):
        phase2b2a.run_audit_inputs(
            tmp_path,
            state.manifest,
            POST_MANIFEST_SHA256,
            confirmed_cloud_storage=True,
        )
