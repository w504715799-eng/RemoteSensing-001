"""Canonical evidence for deterministic Phase 2B2-A model inputs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from trustsr.artifacts.predictions import tensor_sha256
from trustsr.data.crosssensor_manifest import SOURCE_OBJECT_SHA256
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    NODATA_POLICY,
    NORMALIZATION_POLICY,
    PHASE2B1B_AUDIT_SHA256,
    POST_MANIFEST_SHA256,
    RAW_DTYPE,
    RAW_NODATA,
    REFLECTANCE_SCALE,
    SMOKE_BINS,
    SMOKE_SPLITS,
    LoadedCrosssensorPair,
)
from trustsr.data.subset_manifest import BASE_MANIFEST_SHA256
from trustsr.jsonio import canonical_json

AUDIT_SCHEMA = "trustsr.phase2b2a-input-audit.v1"


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _pair_record(loaded: LoadedCrosssensorPair) -> dict[str, object]:
    if not isinstance(loaded, LoadedCrosssensorPair):
        raise TypeError("audit inputs must be LoadedCrosssensorPair values")
    loaded.pair.validate()
    metadata = loaded.metadata
    if metadata.manifest_sha256 != POST_MANIFEST_SHA256:
        raise ValueError("audit input must use the frozen post-manifest")
    if loaded.pair.sample_id != metadata.sample_id:
        raise ValueError("pair and metadata sample identities must match")
    if loaded.pair.source != f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}":
        raise ValueError("pair source must use the frozen post-manifest identity")
    if metadata.crop_policy != CROP_POLICY:
        raise ValueError("audit input has the wrong crop policy")
    if metadata.normalization_policy != NORMALIZATION_POLICY:
        raise ValueError("audit input has the wrong normalization policy")
    if metadata.selection_round != 1:
        raise ValueError("audit input must come from selection round one")
    if metadata.days_between != -1:
        raise ValueError("audit input must use days_between=-1")
    _require_digest(metadata.lr_asset_sha256, "LR asset SHA-256")
    _require_digest(metadata.hr_asset_sha256, "HR asset SHA-256")
    return {
        "sample_id": metadata.sample_id,
        "split": metadata.split,
        "correlation_bin": metadata.correlation_bin,
        "lr_asset_sha256": metadata.lr_asset_sha256,
        "hr_asset_sha256": metadata.hr_asset_sha256,
        "lr_crop_transform": list(metadata.lr_crop_transform),
        "hr_crop_transform": list(metadata.hr_crop_transform),
        "crop_bounds": list(metadata.crop_bounds),
        "lr_tensor_sha256": tensor_sha256(loaded.pair.lr),
        "hr_tensor_sha256": tensor_sha256(loaded.pair.hr),
    }


def _validated_records(
    loaded_pairs: Sequence[LoadedCrosssensorPair],
) -> tuple[dict[str, object], ...]:
    if len(loaded_pairs) != 12:
        raise ValueError("input audit requires exactly 12 smoke pairs")
    records = tuple(_pair_record(loaded) for loaded in loaded_pairs)
    sample_ids = [record["sample_id"] for record in records]
    if len(set(sample_ids)) != 12:
        raise ValueError("input audit requires 12 unique sample IDs")
    split_counts = Counter(record["split"] for record in records)
    expected_splits = {split: 4 for split in SMOKE_SPLITS}
    if dict(split_counts) != expected_splits:
        raise ValueError("input audit has the wrong split counts")
    bin_counts = Counter(record["correlation_bin"] for record in records)
    expected_bins = {bin_index: 3 for bin_index in SMOKE_BINS}
    if dict(bin_counts) != expected_bins:
        raise ValueError("input audit has the wrong correlation-bin counts")
    expected_order = [
        (split, bin_index)
        for split in SMOKE_SPLITS
        for bin_index in SMOKE_BINS
    ]
    observed_order = [
        (record["split"], record["correlation_bin"])
        for record in records
    ]
    if observed_order != expected_order:
        raise ValueError("repeated load must use canonical smoke order")
    return records


def build_input_audit(
    first: Sequence[LoadedCrosssensorPair],
    second: Sequence[LoadedCrosssensorPair],
) -> dict[str, object]:
    """Build host-free evidence that two independent loads are identical."""

    first_records = _validated_records(first)
    second_records = _validated_records(second)
    if first_records != second_records:
        raise ValueError("repeated load metadata or tensor digests differ")
    result: dict[str, object] = {
        "schema": AUDIT_SCHEMA,
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "phase2b1b_audit_sha256": PHASE2B1B_AUDIT_SHA256,
        "base_manifest_sha256": BASE_MANIFEST_SHA256,
        "source_object_sha256": SOURCE_OBJECT_SHA256,
        "smoke_pair_count": 12,
        "smoke_geotiff_count": 24,
        "split_counts": {split: 4 for split in SMOKE_SPLITS},
        "correlation_bin_counts": {str(bin_index): 3 for bin_index in SMOKE_BINS},
        "raw_shapes": {"lr": [4, 130, 130], "hr": [4, 520, 520]},
        "cropped_shapes": {"lr": [4, 128, 128], "hr": [4, 512, 512]},
        "raw_dtype": RAW_DTYPE,
        "raw_nodata": RAW_NODATA,
        "nodata_policy": NODATA_POLICY,
        "reflectance_scale": REFLECTANCE_SCALE,
        "crop_policy": CROP_POLICY,
        "normalization_policy": NORMALIZATION_POLICY,
        "pairs": list(first_records),
        "repeated_load_equal": True,
        "model_inference_run": False,
        "gpu_used": False,
        "real_pixels_local": False,
    }
    canonical_json(result)
    return result
