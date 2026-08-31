"""Contracts for Phase 2B2-A crosssensor model inputs."""

from __future__ import annotations

from copy import deepcopy

import pytest

from trustsr.data.crosssensor_pairs import select_input_smoke_records

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
