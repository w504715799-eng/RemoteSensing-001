"""ROI metrics derived only from frozen Phase 2B3-C maps."""

from __future__ import annotations

from dataclasses import replace
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
from trustsr.evaluation.internal_test_maps import (
    CachedInternalTestScore,
    InternalTestMaps,
    load_or_compute_internal_test_maps,
)
from trustsr.evaluation.internal_test_metrics import (
    build_internal_test_metrics,
    evaluate_internal_test_metric,
)
from trustsr.evaluation.internal_test_predictions import (
    MODEL_NAME,
    SEEDS,
    CachedInternalTestPrediction,
    InternalTestPredictionBundle,
    build_cache_provenance,
)
from trustsr.evaluation.phase2b3c_policy import PHASE2B3C_THRESHOLD
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
    day = (-1, 0, 1)[cell // 4]
    bin_index = cell % 4
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
            days_between=day,
            correlation_bin=bin_index,
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


def _provenance(seed: int) -> dict[str, object]:
    return build_cache_provenance(
        {
            "name": MODEL_NAME,
            "scale": 4,
            "implementation_schema_version": 1,
            "opensr_model_version": OPENSR_MODEL_VERSION,
            "torch_version": "2.7.1+cu128",
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


def _bundle(pair: LoadedCrosssensorPair) -> InternalTestPredictionBundle:
    items = []
    for seed in SEEDS:
        tensor = torch.full((4, 12, 12), 0.5, dtype=torch.float32)
        identity = build_identity(
            _provenance(seed), pair.pair.source, pair.pair.sample_id, pair.pair.lr
        )
        items.append(
            CachedInternalTestPrediction(
                MODEL_NAME, seed, identity, tensor_sha256(tensor), tensor
            )
        )
    return InternalTestPredictionBundle(pair.pair.sample_id, tuple(items))


def _maps(
    tmp_path: Path,
    pair: LoadedCrosssensorPair,
    *,
    score: torch.Tensor,
    risk: torch.Tensor,
) -> InternalTestMaps:
    base = load_or_compute_internal_test_maps(pair, _bundle(pair), ScoreCache(tmp_path))
    cached_score = CachedInternalTestScore(
        base.score.name,
        base.score.identity,
        tensor_sha256(score),
        score,
    )
    return replace(
        base,
        score=cached_score,
        risk_sha256=tensor_sha256(risk),
        risk=risk,
    )


def test_metric_uses_inclusive_threshold_and_maximum_trusted_risk(
    tmp_path: Path,
) -> None:
    pair = _pair(0)
    score = torch.full((12, 12), PHASE2B3C_THRESHOLD * 2, dtype=torch.float64)
    score[0, 0] = PHASE2B3C_THRESHOLD
    score[0, 1] = PHASE2B3C_THRESHOLD / 2
    risk = torch.zeros((12, 12), dtype=torch.float64)
    risk[0, 0] = 0.02
    risk[0, 1] = 0.04

    metric = evaluate_internal_test_metric(
        pair,
        _maps(tmp_path, pair, score=score, risk=risk),
        threshold=PHASE2B3C_THRESHOLD,
    )

    assert metric.trusted_pixels == 2
    assert metric.total_pixels == 144
    assert metric.coverage == 2 / 144
    assert metric.loss == 0.04


def test_empty_trusted_mask_has_zero_roi_loss(tmp_path: Path) -> None:
    pair = _pair(0)
    score = torch.full((12, 12), PHASE2B3C_THRESHOLD * 2, dtype=torch.float64)
    risk = torch.ones((12, 12), dtype=torch.float64)

    metric = evaluate_internal_test_metric(
        pair,
        _maps(tmp_path, pair, score=score, risk=risk),
        threshold=PHASE2B3C_THRESHOLD,
    )

    assert metric.trusted_pixels == 0
    assert metric.loss == 0.0


def test_builds_exact_120_metrics_in_input_order_with_map_digest(tmp_path: Path) -> None:
    pairs = tuple(_pair(index) for index in range(120))
    maps = tuple(
        _maps(
            tmp_path,
            pair,
            score=torch.zeros((12, 12), dtype=torch.float64),
            risk=torch.full((12, 12), index / 1000, dtype=torch.float64),
        )
        for index, pair in enumerate(pairs)
    )

    metrics = build_internal_test_metrics(
        pairs, maps, threshold=PHASE2B3C_THRESHOLD
    )

    assert tuple(value.sample_id for value in metrics.rois) == tuple(
        pair.pair.sample_id for pair in pairs
    )
    assert metrics.rois[-1].loss == pytest.approx(0.119)
    assert len(metrics.map_evidence_sha256) == 64


def test_rejects_wrong_threshold_pair_order_and_map_range(tmp_path: Path) -> None:
    pair = _pair(0)
    maps = _maps(
        tmp_path,
        pair,
        score=torch.zeros((12, 12), dtype=torch.float64),
        risk=torch.zeros((12, 12), dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="approved"):
        evaluate_internal_test_metric(pair, maps, threshold=0.1)
    with pytest.raises(ValueError, match="sample"):
        evaluate_internal_test_metric(_pair(1), maps, threshold=PHASE2B3C_THRESHOLD)
    forged_risk = torch.full((12, 12), 1.1, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        replace(maps, risk_sha256=tensor_sha256(forged_risk), risk=forged_risk)
