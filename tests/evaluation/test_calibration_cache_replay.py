"""Inference-free reconstruction of fixed calibration caches using tiny CPU tensors."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch

from trustsr.artifacts.predictions import (
    CacheIntegrityError,
    PredictionCache,
    build_identity,
    tensor_sha256,
)
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
from trustsr.evaluation.calibration_cache_audit import build_calibration_cache_audit
from trustsr.evaluation.calibration_cache_replay import ReplayInputs, replay_calibration_caches
from trustsr.evaluation.calibration_maps import (
    CachedCalibrationScore,
    CalibrationMaps,
    load_or_compute_calibration_maps,
)
from trustsr.evaluation.calibration_predictions import (
    MODEL_NAME,
    SEEDS,
    CachedCalibrationPrediction,
    CalibrationPredictionBundle,
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


def _raw_model_provenance(seed: int) -> dict[str, object]:
    return {
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


def _pair(index: int) -> LoadedCrosssensorPair:
    sample_id = f"calibration-{index:03d}"
    day = (-1, 0, 1)[index // 40]
    bin_index = (index // 10) % 4
    selection_round = index % 10 + 1
    return LoadedCrosssensorPair(
        pair=SRPair(
            sample_id=sample_id,
            source=f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}",
            lr=torch.full((4, 3, 3), 0.25 + (index % 2) / 100, dtype=torch.float32),
            hr=torch.full((4, 12, 12), 0.45, dtype=torch.float32),
            scale=4,
        ),
        metadata=CrosssensorPairMetadata(
            manifest_sha256=POST_MANIFEST_SHA256,
            sample_id=sample_id,
            split="calibration",
            spatial_group_id=f"group-{index:03d}",
            days_between=day,
            correlation_bin=bin_index,
            selection_round=selection_round,
            lr_asset_sha256="a" * 64,
            hr_asset_sha256="b" * 64,
            lr_crop_transform=(10.0, 0.0, 10.0, 0.0, -10.0, -10.0),
            hr_crop_transform=(2.5, 0.0, 10.0, 0.0, -2.5, -10.0),
            crop_bounds=(10.0, -30.0, 40.0, -10.0),
            crop_policy=CROP_POLICY,
            normalization_policy=PHASE2B3A_NORMALIZATION_POLICY,
            lr_saturation=RadiometricSaturation(2500, 2500, 0, (0, 0, 0, 0)),
            hr_saturation=RadiometricSaturation(4500, 4500, 0, (0, 0, 0, 0)),
        ),
    )


def _prepared(
    tmp_path: Path,
) -> tuple[dict[str, object], tuple[LoadedCrosssensorPair, ...], PredictionCache, ScoreCache]:
    prediction_cache = PredictionCache(tmp_path / "predictions")
    score_cache = ScoreCache(tmp_path / "scores")
    pairs = tuple(_pair(index) for index in range(120))
    bundles: list[CalibrationPredictionBundle] = []
    maps = []
    for index, pair in enumerate(pairs):
        items = []
        for seed_index, seed in enumerate(SEEDS):
            identity = build_identity(
                build_cache_provenance(_raw_model_provenance(seed)),
                pair.pair.source,
                pair.pair.sample_id,
                pair.pair.lr,
            )
            prediction = torch.full(
                (4, 12, 12), 0.1 + seed_index / 100 + index / 10_000, dtype=torch.float32
            )
            prediction_cache.put(identity, prediction)
            items.append(
                CachedCalibrationPrediction(
                    model_name=MODEL_NAME,
                    seed=seed,
                    identity=identity,
                    prediction_sha256=tensor_sha256(prediction),
                    tensor=prediction_cache.get(identity),
                )
            )
        bundle = CalibrationPredictionBundle(pair.pair.sample_id, tuple(items))
        bundles.append(bundle)
        maps.append(load_or_compute_calibration_maps(pair, bundle, score_cache))
    audit = json.loads(canonical_json(build_calibration_cache_audit(tuple(bundles), tuple(maps))))
    return audit, pairs, prediction_cache, score_cache


def test_replays_real_cache_entries_without_model_or_prediction_access(tmp_path: Path) -> None:
    audit, pairs, prediction_cache, score_cache = _prepared(tmp_path)

    replayed = replay_calibration_caches(audit, pairs, prediction_cache, score_cache)

    assert isinstance(replayed, ReplayInputs)
    assert len(replayed.bundles) == len(replayed.maps) == 120
    assert replayed.bundles[0].items[0].seed == SEEDS[0]
    assert replayed.maps[-1].sample_id == pairs[-1].pair.sample_id
    with pytest.raises(FrozenInstanceError):
        replayed.maps = ()  # type: ignore[misc]


def test_replay_fails_closed_for_missing_cache_or_mutated_audit(tmp_path: Path) -> None:
    audit, pairs, prediction_cache, score_cache = _prepared(tmp_path)
    sample = audit["samples"][0]
    key = sample["predictions"][0]["cache_key"]
    (prediction_cache.root / f"{key}.json").unlink()
    with pytest.raises(CacheIntegrityError):
        replay_calibration_caches(audit, pairs, prediction_cache, score_cache)

    audit, pairs, prediction_cache, score_cache = _prepared(tmp_path / "mutated")
    changed = deepcopy(audit)
    changed["samples"][0]["risk"]["risk_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="risk|digest"):
        replay_calibration_caches(changed, pairs, prediction_cache, score_cache)

    wrong_split = deepcopy(audit)
    wrong_split["split"] = "internal_test"
    with pytest.raises(ValueError, match="split|schema"):
        replay_calibration_caches(wrong_split, pairs, prediction_cache, score_cache)


def test_replay_rejects_pair_order_and_risk_mismatches(tmp_path: Path) -> None:
    audit, pairs, prediction_cache, score_cache = _prepared(tmp_path)
    with pytest.raises(ValueError, match="order|sample"):
        replay_calibration_caches(
            audit, (pairs[1], pairs[0], *pairs[2:]), prediction_cache, score_cache
        )

    changed_pair = replace(
        pairs[0], pair=replace(pairs[0].pair, hr=torch.zeros_like(pairs[0].pair.hr))
    )
    with pytest.raises(ValueError, match="risk"):
        replay_calibration_caches(audit, (changed_pair, *pairs[1:]), prediction_cache, score_cache)


def test_replay_inputs_rejects_same_sample_with_a_different_k5_identity(tmp_path: Path) -> None:
    audit, pairs, prediction_cache, score_cache = _prepared(tmp_path)
    replayed = replay_calibration_caches(audit, pairs, prediction_cache, score_cache)
    expected_bundle = replayed.bundles[0]
    foreign_maps = replayed.maps[1]
    sample_id = expected_bundle.sample_id
    foreign_identity = replace(foreign_maps.score.identity, sample_id=sample_id)
    forged_score = CachedCalibrationScore(
        name=foreign_maps.score.name,
        identity=foreign_identity,
        score_sha256=foreign_maps.score.score_sha256,
        tensor=foreign_maps.score.tensor,
    )
    forged_maps = CalibrationMaps(
        sample_id=sample_id,
        score=forged_score,
        score_prediction_sha256s=foreign_maps.score_prediction_sha256s,
        risk_name=foreign_maps.risk_name,
        risk_window=foreign_maps.risk_window,
        risk_sha256=foreign_maps.risk_sha256,
        risk=foreign_maps.risk,
    )
    maps = (forged_maps, *replayed.maps[1:])

    with pytest.raises(ValueError, match="prediction|LR|source|bundle"):
        ReplayInputs(bundles=replayed.bundles, maps=maps)


def test_replay_inputs_rejects_one_bundle_with_another_legal_runtime(
    tmp_path: Path,
) -> None:
    audit, pairs, prediction_cache, score_cache = _prepared(tmp_path)
    replayed = replay_calibration_caches(audit, pairs, prediction_cache, score_cache)
    original = replayed.bundles[1]
    runtime = {
        "torch_version": "2.12.1+cu130",
        "cuda_runtime": "13.0",
    }
    items = tuple(
        CachedCalibrationPrediction(
            model_name=item.model_name,
            seed=item.seed,
            identity=build_identity(
                {**dict(item.identity.model_provenance), **runtime},
                item.identity.source,
                item.identity.sample_id,
                pairs[1].pair.lr,
            ),
            prediction_sha256=item.prediction_sha256,
            tensor=item.tensor,
        )
        for item in original.items
    )
    alternate = CalibrationPredictionBundle(sample_id=original.sample_id, items=items)

    with pytest.raises(ValueError, match="model scientific identit"):
        ReplayInputs(
            bundles=(replayed.bundles[0], alternate, *replayed.bundles[2:]),
            maps=replayed.maps,
        )
