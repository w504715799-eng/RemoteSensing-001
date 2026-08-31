"""Contracts for Phase 2B2-A crosssensor model inputs."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from trustsr.data import crosssensor_pairs
from trustsr.data.crosssensor_pairs import (
    load_crosssensor_records,
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
