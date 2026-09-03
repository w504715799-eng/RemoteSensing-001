"""Metadata-only contracts for the Phase 2B3-B calibration subset."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from trustsr.data import calibration_subset
from trustsr.data.calibration_subset import (
    load_calibration_records,
    select_calibration_records,
)
from trustsr.data.crosssensor_pairs import POST_MANIFEST_SHA256


def _complete_records() -> tuple[dict[str, object], ...]:
    """Hand-built metadata only; this fixture never creates image files."""

    return tuple(
        {
            "sample_id": f"{split}-{day}-{bin_index}-{round_index}",
            "selection_sha256": (
                f"selection-{split}-{day}-{bin_index}-{round_index}"
            ),
            "spatial_group_id": f"group-{split}-{day}-{bin_index}-{round_index}",
            "split": split,
            "days_between": day,
            "correlation_bin": bin_index,
            "selection_round": round_index,
            "lr_asset": {"metadata": "lr"},
            "hr_asset": {"metadata": "hr"},
        }
        for split in ("development", "calibration", "internal_test")
        for day in (-1, 0, 1)
        for bin_index in range(4)
        for round_index in range(1, 11)
    )


def _calibration_index(records: list[dict[str, object]]) -> int:
    return next(index for index, record in enumerate(records) if record["split"] == "calibration")


def test_select_calibration_records_preserves_full_manifest_order() -> None:
    records = _complete_records()

    selected = select_calibration_records(records)

    assert len(selected) == 120
    assert [record["sample_id"] for record in selected[:3]] == [
        "calibration--1-0-1",
        "calibration--1-0-2",
        "calibration--1-0-3",
    ]
    assert [record["sample_id"] for record in selected[-3:]] == [
        "calibration-1-3-8",
        "calibration-1-3-9",
        "calibration-1-3-10",
    ]
    assert {record["split"] for record in selected} == {"calibration"}


def test_select_calibration_records_keeps_reversed_input_calibration_order() -> None:
    selected = select_calibration_records(tuple(reversed(_complete_records())))

    assert [record["sample_id"] for record in selected[:3]] == [
        "calibration-1-3-10",
        "calibration-1-3-9",
        "calibration-1-3-8",
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("calibration_shortcut", "360"),
        ("wrong_total", "360"),
        ("wrong_split_count", "120"),
        ("unknown_split", "split"),
        ("missing_sample_id", "sample_id"),
        ("empty_selection_sha256", "selection_sha256"),
        ("non_string_spatial_group_id", "spatial_group_id"),
        ("duplicate_sample_id", "unique sample_id"),
        ("duplicate_selection_sha256", "unique selection_sha256"),
        ("duplicate_spatial_group_id", "unique spatial_group_id"),
        ("missing_lr_asset", "asset"),
        ("missing_hr_asset", "asset"),
        ("bad_day", "stratum"),
        ("boolean_bin", "correlation_bin"),
        ("bad_round", "round"),
    ],
)
def test_select_calibration_records_rejects_mutated_frozen_manifest(
    mutation: str, message: str
) -> None:
    records = [deepcopy(record) for record in _complete_records()]
    calibration_index = _calibration_index(records)

    if mutation == "calibration_shortcut":
        records = [record for record in records if record["split"] == "calibration"]
    elif mutation == "wrong_total":
        records.pop()
    elif mutation == "wrong_split_count":
        records[calibration_index]["split"] = "development"
    elif mutation == "unknown_split":
        records[calibration_index]["split"] = "other"
    elif mutation == "missing_sample_id":
        records[calibration_index].pop("sample_id")
    elif mutation == "empty_selection_sha256":
        records[calibration_index]["selection_sha256"] = ""
    elif mutation == "non_string_spatial_group_id":
        records[calibration_index]["spatial_group_id"] = 1
    elif mutation == "duplicate_sample_id":
        records[calibration_index + 1]["sample_id"] = records[calibration_index]["sample_id"]
    elif mutation == "duplicate_selection_sha256":
        records[calibration_index + 1]["selection_sha256"] = records[calibration_index][
            "selection_sha256"
        ]
    elif mutation == "duplicate_spatial_group_id":
        records[calibration_index + 1]["spatial_group_id"] = records[calibration_index][
            "spatial_group_id"
        ]
    elif mutation == "missing_lr_asset":
        records[calibration_index]["lr_asset"] = None
    elif mutation == "missing_hr_asset":
        records[calibration_index]["hr_asset"] = None
    elif mutation == "bad_day":
        records[calibration_index]["days_between"] = 2
    elif mutation == "boolean_bin":
        records[calibration_index]["correlation_bin"] = True
    else:
        records[calibration_index]["selection_round"] = 11

    with pytest.raises(ValueError, match=message):
        select_calibration_records(records)


def test_loader_returns_only_calibration_after_fixed_schema_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records = _complete_records()

    def load_frozen_manifest(
        storage_root: Path, manifest_path: Path, *, expected_sha256: str
    ) -> tuple[dict[str, object], ...]:
        if expected_sha256 != POST_MANIFEST_SHA256:
            raise ValueError("fixed digest was not supplied")
        return records

    monkeypatch.setattr(calibration_subset, "load_crosssensor_records", load_frozen_manifest)

    selected = load_calibration_records(tmp_path, tmp_path / "samples.jsonl")

    assert len(selected) == 120
    assert selected[0]["sample_id"] == "calibration--1-0-1"
    assert selected[-1]["sample_id"] == "calibration-1-3-10"


def test_loader_propagates_schema_loader_error_before_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def reject_invalid_schema(
        storage_root: Path, manifest_path: Path, *, expected_sha256: str
    ) -> tuple[dict[str, object], ...]:
        raise ValueError("schema rejected malformed post-manifest")

    monkeypatch.setattr(calibration_subset, "load_crosssensor_records", reject_invalid_schema)

    with pytest.raises(ValueError, match="schema rejected"):
        load_calibration_records(tmp_path, tmp_path / "samples.jsonl")
