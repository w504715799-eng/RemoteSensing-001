"""Metadata-only selection contracts for Phase 2B3-C."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from trustsr.data import internal_test_subset
from trustsr.data.crosssensor_pairs import POST_MANIFEST_SHA256
from trustsr.data.internal_test_subset import (
    load_internal_test_records,
    select_internal_test_records,
)


def _complete_records() -> tuple[dict[str, object], ...]:
    records = [
        {
            "sample_id": f"{split}-{day + 1}-{bin_index}-{round_index:02d}",
            "selection_sha256": f"{index:064x}",
            "spatial_group_id": f"{index + 1000:064x}",
            "split": split,
            "days_between": day,
            "correlation_bin": bin_index,
            "selection_round": round_index,
            "lr_asset": {"sha256": "a" * 64},
            "hr_asset": {"sha256": "b" * 64},
        }
        for index, (split, day, bin_index, round_index) in enumerate(
            (split, day, bin_index, round_index)
            for split in ("calibration", "development", "internal_test")
            for day in (-1, 0, 1)
            for bin_index in range(4)
            for round_index in range(1, 11)
        )
    ]
    return tuple(sorted(records, key=lambda record: str(record["sample_id"])))


def test_selects_exact_balanced_test_membership_in_manifest_order() -> None:
    records = _complete_records()

    selected = select_internal_test_records(records)

    assert len(selected) == 120
    assert all(record["split"] == "internal_test" for record in selected)
    assert selected == tuple(
        record for record in records if record["split"] == "internal_test"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "360"),
        ("duplicate_id", "unique sample_id"),
        ("duplicate_group", "unique spatial_group_id"),
        ("wrong_split", "120"),
        ("bad_round", "rounds 1 through 10"),
        ("reordered", "canonical"),
    ],
)
def test_rejects_incomplete_duplicate_or_reordered_manifest(
    mutation: str, message: str
) -> None:
    records = [deepcopy(record) for record in _complete_records()]
    test_indices = [
        index for index, record in enumerate(records) if record["split"] == "internal_test"
    ]
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate_id":
        records[test_indices[1]]["sample_id"] = records[test_indices[0]]["sample_id"]
    elif mutation == "duplicate_group":
        records[test_indices[1]]["spatial_group_id"] = records[test_indices[0]][
            "spatial_group_id"
        ]
    elif mutation == "wrong_split":
        records[test_indices[0]]["split"] = "calibration"
    elif mutation == "bad_round":
        records[test_indices[0]]["selection_round"] = 2
    else:
        records[test_indices[0]], records[test_indices[1]] = (
            records[test_indices[1]],
            records[test_indices[0]],
        )

    with pytest.raises(ValueError, match=message):
        select_internal_test_records(records)


def test_rejects_calibration_or_development_shortcut() -> None:
    for split in ("calibration", "development"):
        shortcut = tuple(record for record in _complete_records() if record["split"] == split)
        with pytest.raises(ValueError, match="360"):
            select_internal_test_records(shortcut)


def test_loader_uses_only_the_frozen_manifest_metadata_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records = _complete_records()

    def fake_load(
        storage_root: Path, manifest_path: Path, *, expected_sha256: str
    ) -> tuple[dict[str, object], ...]:
        assert storage_root == tmp_path
        assert manifest_path == tmp_path / "samples.jsonl"
        assert expected_sha256 == POST_MANIFEST_SHA256
        return records

    monkeypatch.setattr(internal_test_subset, "load_crosssensor_records", fake_load)

    selected = load_internal_test_records(tmp_path, tmp_path / "samples.jsonl")

    assert len(selected) == 120
    assert {record["split"] for record in selected} == {"internal_test"}
