"""Strict Phase 2B2-A inputs derived from the frozen crosssensor subset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

POST_MANIFEST_SHA256 = "c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a"
PHASE2B1B_AUDIT_SHA256 = "d8964033958594a23ac7056519894d508977bfd2cc13da50a5833024274f3e90"
REFLECTANCE_SCALE = 10_000.0
RAW_DTYPE = "uint16"
CROP_POLICY = "center_crop_lr_1_hr_4_v1"
NORMALIZATION_POLICY = "uint16_divide_10000_no_clip_v1"
SMOKE_SPLITS = ("calibration", "development", "internal_test")
SMOKE_BINS = (0, 1, 2, 3)


def _require_unique_strings(
    records: Sequence[Mapping[str, object]], field: str
) -> None:
    values = [record.get(field) for record in records]
    if any(type(value) is not str or not value for value in values):
        raise ValueError(f"smoke record {field} must be a non-empty string")
    if len(set(values)) != len(values):
        raise ValueError(f"smoke records require unique {field} values")


def select_input_smoke_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Select the frozen 12-cell CPU input smoke set."""

    cells: dict[tuple[str, int], list[Mapping[str, object]]] = {
        (split, bin_index): []
        for split in SMOKE_SPLITS
        for bin_index in SMOKE_BINS
    }
    for record in records:
        if record.get("selection_round") != 1 or record.get("days_between") != -1:
            continue
        key = (record.get("split"), record.get("correlation_bin"))
        if key in cells:
            cells[key].append(record)

    for key, candidates in cells.items():
        if len(candidates) != 1:
            raise ValueError(
                "smoke selection requires exactly one record for "
                f"split={key[0]}, correlation_bin={key[1]}"
            )

    selected = tuple(
        cells[(split, bin_index)][0]
        for split in SMOKE_SPLITS
        for bin_index in SMOKE_BINS
    )
    _require_unique_strings(selected, "sample_id")
    _require_unique_strings(selected, "spatial_group_id")
    return selected
