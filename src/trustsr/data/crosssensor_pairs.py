"""Strict Phase 2B2-A inputs derived from the frozen crosssensor subset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from trustsr.data.subset_manifest import load_subset_manifest

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


def load_crosssensor_records(
    storage_root: Path,
    manifest_path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, object], ...]:
    """Load the frozen all-assets Phase 2B1B post-manifest."""

    if not isinstance(storage_root, Path) or not isinstance(manifest_path, Path):
        raise TypeError("storage_root and manifest_path must be pathlib.Path values")
    if expected_sha256 != POST_MANIFEST_SHA256:
        raise ValueError("expected the frozen post-manifest SHA-256")
    if storage_root.is_symlink() or not storage_root.is_dir():
        raise ValueError("storage_root must be an existing non-symlink directory")
    resolved_root = storage_root.resolve(strict=True)
    if resolved_root != storage_root.absolute():
        raise ValueError("storage_root must not contain symlink path components")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("post-manifest must be an existing regular file")
    resolved_manifest = manifest_path.resolve(strict=True)
    expected_path = (
        resolved_root
        / "trustsr"
        / "phase2b1b"
        / "selections"
        / expected_sha256
        / "samples.jsonl"
    )
    if resolved_manifest != expected_path:
        raise ValueError("manifest must be the digest-addressed frozen post-manifest")
    records = load_subset_manifest(
        resolved_manifest,
        expected_sha256=expected_sha256,
    )
    if any(
        record["lr_asset"] is None or record["hr_asset"] is None
        for record in records
    ):
        raise ValueError("Phase 2B2-A requires the all-assets post-manifest")
    return records
