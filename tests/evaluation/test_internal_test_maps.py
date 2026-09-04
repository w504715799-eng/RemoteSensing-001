"""Fixed Phase 2B3-C score and risk map contracts."""

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
    RISK_NAME,
    RISK_WINDOW,
    SCORE_NAME,
    CachedInternalTestScore,
    load_or_compute_internal_test_maps,
)
from trustsr.evaluation.internal_test_predictions import (
    MODEL_NAME,
    SEEDS,
    CachedInternalTestPrediction,
    InternalTestPredictionBundle,
    build_cache_provenance,
)
from trustsr.models.ldsr_assets import (
    CHECKPOINT_NAME,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    CHECKPOINT_URL,
    CONFIG_SHA256,
)
from trustsr.models.versions import OPENSR_MODEL_VERSION


def _pair() -> LoadedCrosssensorPair:
    sample_id = "test-000"
    return LoadedCrosssensorPair(
        pair=SRPair(
            source=f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}",
            sample_id=sample_id,
            lr=torch.full((4, 3, 3), 0.25, dtype=torch.float32),
            hr=torch.full((4, 12, 12), 0.5, dtype=torch.float32),
            scale=4,
        ),
        metadata=CrosssensorPairMetadata(
            manifest_sha256=POST_MANIFEST_SHA256,
            sample_id=sample_id,
            split="internal_test",
            spatial_group_id="a" * 64,
            days_between=-1,
            correlation_bin=0,
            selection_round=1,
            lr_asset_sha256="b" * 64,
            hr_asset_sha256="c" * 64,
            lr_crop_transform=(10.0, 0.0, 10.0, 0.0, -10.0, -10.0),
            hr_crop_transform=(2.5, 0.0, 10.0, 0.0, -2.5, -10.0),
            crop_bounds=(10.0, -30.0, 30.0, -10.0),
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


def _bundle(pair: LoadedCrosssensorPair | None = None) -> InternalTestPredictionBundle:
    loaded = _pair() if pair is None else pair
    items = []
    for index, seed in enumerate(SEEDS):
        tensor = torch.full((4, 12, 12), index / 4, dtype=torch.float32)
        identity = build_identity(
            _provenance(seed),
            loaded.pair.source,
            loaded.pair.sample_id,
            loaded.pair.lr,
        )
        items.append(
            CachedInternalTestPrediction(
                MODEL_NAME,
                seed,
                identity,
                tensor_sha256(tensor),
                tensor,
            )
        )
    return InternalTestPredictionBundle(loaded.pair.sample_id, tuple(items))


def test_computes_population_variance_band_mean_and_seed3407_r9(tmp_path: Path) -> None:
    pair = _pair()

    maps = load_or_compute_internal_test_maps(pair, _bundle(pair), ScoreCache(tmp_path))

    assert maps.score.name == SCORE_NAME
    assert maps.risk_name == RISK_NAME
    assert maps.risk_window == RISK_WINDOW == 9
    assert maps.score.tensor.dtype == torch.float64
    assert maps.risk.dtype == torch.float64
    assert torch.equal(
        maps.score.tensor, torch.full((12, 12), 0.125, dtype=torch.float64)
    )
    assert torch.equal(maps.risk, torch.full((12, 12), 0.5, dtype=torch.float64))


def test_score_cache_replay_is_byte_identical(tmp_path: Path) -> None:
    pair = _pair()
    bundle = _bundle(pair)
    cache = ScoreCache(tmp_path)
    first = load_or_compute_internal_test_maps(pair, bundle, cache)
    second = load_or_compute_internal_test_maps(pair, bundle, cache)

    assert first == second
    assert first.score.score_sha256 == tensor_sha256(first.score.tensor)
    assert first.risk_sha256 == tensor_sha256(first.risk)


def test_rejects_calibration_pair_and_reordered_seed_bundle(tmp_path: Path) -> None:
    pair = _pair()
    calibration = replace(pair, metadata=replace(pair.metadata, split="calibration"))
    with pytest.raises(ValueError, match="internal_test"):
        load_or_compute_internal_test_maps(
            calibration, _bundle(pair), ScoreCache(tmp_path / "a")
        )
    with pytest.raises(ValueError, match="fixed ordered K5"):
        InternalTestPredictionBundle(
            pair.pair.sample_id, tuple(reversed(_bundle(pair).items))
        )


@pytest.mark.parametrize(
    ("key", "value"), (("correction", 1), ("seed_first", SEEDS[1]))
)
def test_cached_score_rejects_fixed_operator_mutation(
    tmp_path: Path, key: str, value: int
) -> None:
    maps = load_or_compute_internal_test_maps(
        _pair(), _bundle(), ScoreCache(tmp_path)
    )
    parameters = dict(maps.score.identity.operator_parameters)
    parameters[key] = value
    forged_identity = replace(maps.score.identity, operator_parameters=parameters)

    with pytest.raises(ValueError, match="fixed B3-C contract"):
        CachedInternalTestScore(
            maps.score.name,
            forged_identity,
            maps.score.score_sha256,
            maps.score.tensor,
        )


def test_maps_reject_risk_window_and_float32_map_mutations(tmp_path: Path) -> None:
    maps = load_or_compute_internal_test_maps(
        _pair(), _bundle(), ScoreCache(tmp_path)
    )

    with pytest.raises(ValueError, match="risk configuration"):
        replace(maps, risk_window=7)
    float32_score = maps.score.tensor.to(dtype=torch.float32)
    with pytest.raises(ValueError, match="float64"):
        CachedInternalTestScore(
            maps.score.name,
            maps.score.identity,
            tensor_sha256(float32_score),
            float32_score,
        )
