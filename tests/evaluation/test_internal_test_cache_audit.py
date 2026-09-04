"""Identity-only audit for the fixed Phase 2B3-C cache set."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from trustsr.artifacts.predictions import build_identity, tensor_sha256
from trustsr.artifacts.scores import ScoreCache
from trustsr.contracts import SRPair
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
    RadiometricSaturation,
)
from trustsr.evaluation.internal_test_cache_audit import (
    build_internal_test_cache_audit,
)
from trustsr.evaluation.internal_test_maps import (
    InternalTestMaps,
    load_or_compute_internal_test_maps,
)
from trustsr.evaluation.internal_test_predictions import (
    MODEL_NAME,
    SEEDS,
    CachedInternalTestPrediction,
    InternalTestPredictionBundle,
    build_cache_provenance,
)
from trustsr.jsonio import canonical_json
from trustsr.models.ldsr_assets import (
    CHECKPOINT_NAME,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    CHECKPOINT_URL,
    CONFIG_SHA256,
)
from trustsr.models.versions import OPENSR_MODEL_VERSION


def _pair(index: int) -> LoadedCrosssensorPair:
    round_index = index // 12 + 1
    cell = index % 12
    sample_id = f"test-{index:03d}"
    return LoadedCrosssensorPair(
        pair=SRPair(
            sample_id=sample_id,
            source=f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}",
            lr=torch.full((4, 3, 3), 0.25, dtype=torch.float32),
            hr=torch.full((4, 12, 12), 0.5, dtype=torch.float32),
            scale=4,
        ),
        metadata=CrosssensorPairMetadata(
            manifest_sha256=POST_MANIFEST_SHA256,
            sample_id=sample_id,
            split="internal_test",
            spatial_group_id=f"{index + 1000:064x}",
            days_between=(-1, 0, 1)[cell // 4],
            correlation_bin=cell % 4,
            selection_round=round_index,
            lr_asset_sha256="a" * 64,
            hr_asset_sha256="b" * 64,
            lr_crop_transform=(10.0, 0.0, 10.0, 0.0, -10.0, -10.0),
            hr_crop_transform=(2.5, 0.0, 10.0, 0.0, -2.5, -10.0),
            crop_bounds=(10.0, -30.0, 40.0, -10.0),
            crop_policy=CROP_POLICY,
            normalization_policy=PHASE2B3A_NORMALIZATION_POLICY,
            lr_saturation=RadiometricSaturation(2500, 2500, 0, (0, 0, 0, 0)),
            hr_saturation=RadiometricSaturation(5000, 5000, 0, (0, 0, 0, 0)),
        ),
    )


def _provenance(seed: int, torch_version: str) -> dict[str, object]:
    return build_cache_provenance(
        {
            "name": MODEL_NAME,
            "scale": 4,
            "implementation_schema_version": 1,
            "opensr_model_version": OPENSR_MODEL_VERSION,
            "torch_version": torch_version,
            "cuda_runtime": "12.8",
            "checkpoint_name": CHECKPOINT_NAME,
            "checkpoint_url": CHECKPOINT_URL,
            "checkpoint_size": CHECKPOINT_SIZE,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "config_sha256": CONFIG_SHA256,
            "device": "cuda",
            "seed": seed,
            "sampling_steps": 100,
            "sampling_eta": 0.95,
            "sampling_temperature": 1.0,
            "histogram_matching": True,
            "output_policy": "clip_to_[0,1]",
        }
    )


def _artifacts(
    tmp_path: Path, index: int, *, torch_version: str = "2.7.1+cu128"
) -> tuple[InternalTestPredictionBundle, InternalTestMaps]:
    pair = _pair(index)
    items = []
    for seed in SEEDS:
        tensor = torch.full((4, 12, 12), 0.5, dtype=torch.float32)
        identity = build_identity(
            _provenance(seed, torch_version),
            pair.pair.source,
            pair.pair.sample_id,
            pair.pair.lr,
        )
        items.append(
            CachedInternalTestPrediction(
                MODEL_NAME, seed, identity, tensor_sha256(tensor), tensor
            )
        )
    bundle = InternalTestPredictionBundle(pair.pair.sample_id, tuple(items))
    maps = load_or_compute_internal_test_maps(pair, bundle, ScoreCache(tmp_path))
    return bundle, maps


def _all_artifacts(
    tmp_path: Path,
) -> tuple[tuple[InternalTestPredictionBundle, ...], tuple[InternalTestMaps, ...]]:
    values = tuple(_artifacts(tmp_path, index) for index in range(120))
    return tuple(value[0] for value in values), tuple(value[1] for value in values)


def test_audit_enumerates_exact_cache_identities_without_roi_metrics(
    tmp_path: Path,
) -> None:
    bundles, maps = _all_artifacts(tmp_path)

    audit = build_internal_test_cache_audit(bundles, maps)

    assert audit["schema"] == "trustsr.phase2b3c-evaluation-cache-audit.v1"
    assert audit["split"] == "internal_test"
    assert audit["sample_count"] == 120
    assert audit["prediction_count"] == 600
    assert audit["score_count"] == 120
    assert len(audit["samples"]) == 120
    payload = canonical_json(audit).decode("utf-8")
    assert '"loss"' not in payload
    assert '"coverage"' not in payload


def test_audit_rejects_map_order_or_global_model_identity_drift(tmp_path: Path) -> None:
    bundles, maps = _all_artifacts(tmp_path)
    with pytest.raises(ValueError, match="order"):
        build_internal_test_cache_audit(bundles, tuple(reversed(maps)))

    changed_bundle, changed_maps = _artifacts(
        tmp_path / "changed", 119, torch_version="2.8.0+cu129"
    )
    with pytest.raises(ValueError, match="model scientific identities"):
        build_internal_test_cache_audit(
            (*bundles[:-1], changed_bundle), (*maps[:-1], changed_maps)
        )


def test_audit_rejects_score_prediction_digest_mismatch(tmp_path: Path) -> None:
    bundles, maps = _all_artifacts(tmp_path)
    forged = object.__new__(InternalTestMaps)
    for field, value in maps[-1].__dict__.items():
        object.__setattr__(forged, field, value)
    object.__setattr__(forged, "score_prediction_sha256s", ("0" * 64,) * 5)

    with pytest.raises(ValueError, match="prediction digests"):
        build_internal_test_cache_audit(bundles, (*maps[:-1], forged))
