"""Fixed internal-test LDSR K5 prediction-cache contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from trustsr.artifacts.predictions import PredictionCache, tensor_sha256
from trustsr.contracts import SRPair
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
    RadiometricSaturation,
)
from trustsr.evaluation.internal_test_predictions import (
    MODEL_NAME,
    SEEDS,
    IncompleteInternalTestPredictionCache,
    load_complete_cached_internal_test_bundles,
    load_or_generate_internal_test_bundle,
    probe_cached_internal_test_bundles,
    require_complete_cached_internal_test_bundles,
)
from trustsr.evaluation.phase2b3c_evidence import (
    PHASE2B3B_FILE_SHA256S,
    PHASE2B3B_PUBLICATION_COMMIT,
)
from trustsr.models.ldsr_assets import (
    CHECKPOINT_NAME,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    CHECKPOINT_URL,
    CONFIG_SHA256,
)
from trustsr.models.versions import OPENSR_MODEL_VERSION


def _raw_provenance(seed: int = 3407) -> dict[str, object]:
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


def _pair(index: int = 0) -> LoadedCrosssensorPair:
    position = index % 120
    selection_round = position // 12 + 1
    within_round = position % 12
    day = (-1, 0, 1)[within_round // 4]
    bin_index = within_round % 4
    sample_id = f"test-{index:03d}"
    return LoadedCrosssensorPair(
        pair=SRPair(
            source=f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}",
            sample_id=sample_id,
            lr=torch.full((4, 1, 1), 0.25, dtype=torch.float32),
            hr=torch.full((4, 4, 4), 0.5, dtype=torch.float32),
            scale=4,
        ),
        metadata=CrosssensorPairMetadata(
            manifest_sha256=POST_MANIFEST_SHA256,
            sample_id=sample_id,
            split="internal_test",
            spatial_group_id=f"{index + 1000:064x}",
            days_between=day,
            correlation_bin=bin_index,
            selection_round=selection_round,
            lr_asset_sha256="a" * 64,
            hr_asset_sha256="b" * 64,
            lr_crop_transform=(10.0, 0.0, 10.0, 0.0, -10.0, -10.0),
            hr_crop_transform=(2.5, 0.0, 10.0, 0.0, -2.5, -10.0),
            crop_bounds=(10.0, -30.0, 20.0, -20.0),
            crop_policy=CROP_POLICY,
            normalization_policy=PHASE2B3A_NORMALIZATION_POLICY,
            lr_saturation=RadiometricSaturation(2500, 2500, 0, (0, 0, 0, 0)),
            hr_saturation=RadiometricSaturation(5000, 5000, 0, (0, 0, 0, 0)),
        ),
    )


class _SeedModel:
    name = MODEL_NAME
    scale = 4

    def __init__(self, owner: _FakeLDSR, seed: int) -> None:
        self.owner = owner
        self.seed = seed

    def provenance(self) -> dict[str, object]:
        return _raw_provenance(self.seed)

    def predict(self, lr: torch.Tensor) -> torch.Tensor:
        if self.owner.fail_on_predict:
            raise AssertionError("warm cache invoked inference")
        self.owner.predictions += 1
        return torch.full(
            (4, lr.shape[1] * 4, lr.shape[2] * 4),
            self.seed / 10_000,
            dtype=torch.float32,
        )


class _FakeLDSR:
    name = MODEL_NAME
    scale = 4

    def __init__(self, *, fail_on_predict: bool = False) -> None:
        self.fail_on_predict = fail_on_predict
        self.predictions = 0

    def provenance(self) -> dict[str, object]:
        return _raw_provenance()

    def for_seed(self, seed: int) -> _SeedModel:
        return _SeedModel(self, seed)


def test_k5_cache_populates_and_reuses_without_inference(tmp_path: Path) -> None:
    cache = PredictionCache(tmp_path)
    pair = _pair()
    cold = _FakeLDSR()
    first = load_or_generate_internal_test_bundle(pair, ldsr=cold, cache=cache)
    warm = _FakeLDSR(fail_on_predict=True)
    second = load_or_generate_internal_test_bundle(pair, ldsr=warm, cache=cache)

    assert tuple(item.seed for item in first.items) == SEEDS
    assert cold.predictions == 5
    assert warm.predictions == 0
    assert first == second
    assert tuple(item.prediction_sha256 for item in first.items) == tuple(
        tensor_sha256(item.tensor) for item in first.items
    )


def test_prediction_identity_binds_test_split_and_frozen_b_publication(
    tmp_path: Path,
) -> None:
    bundle = load_or_generate_internal_test_bundle(
        _pair(), ldsr=_FakeLDSR(), cache=PredictionCache(tmp_path)
    )
    provenance = dict(bundle.for_seed(3407).identity.model_provenance)

    assert provenance["experiment_schema"] == "trustsr.phase2b3c-predictions.v1"
    assert provenance["split"] == "internal_test"
    assert provenance["phase2b3b_publication_commit"] == PHASE2B3B_PUBLICATION_COMMIT
    assert provenance["phase2b3b_acceptance_sha256"] == PHASE2B3B_FILE_SHA256S[
        "sen2naipv2-calibration-conformal-acceptance-v1.json"
    ]


def test_rejects_calibration_pair_before_model_use(tmp_path: Path) -> None:
    pair = _pair()
    pair = replace(pair, metadata=replace(pair.metadata, split="calibration"))
    model = _FakeLDSR()

    with pytest.raises(ValueError, match="internal_test"):
        load_or_generate_internal_test_bundle(
            pair, ldsr=model, cache=PredictionCache(tmp_path)
        )
    assert model.predictions == 0


def test_complete_120_by_5_probe_and_typed_missing_cache_stop(tmp_path: Path) -> None:
    cache = PredictionCache(tmp_path)
    pairs = tuple(_pair(index) for index in range(120))
    model = _FakeLDSR()
    expected = tuple(
        load_or_generate_internal_test_bundle(pair, ldsr=model, cache=cache)
        for pair in pairs
    )

    loaded = load_complete_cached_internal_test_bundles(pairs, cache=cache)

    assert loaded == expected
    complete_probe = probe_cached_internal_test_bundles(pairs, cache=cache)
    assert complete_probe.present_count == 600
    assert complete_probe.missing_count == 0
    assert complete_probe.bundles == expected
    assert model.predictions == 600
    missing_tensor = next(tmp_path.glob("*.safetensors"))
    missing_tensor.with_suffix(".json").unlink()
    missing_tensor.unlink()
    assert load_complete_cached_internal_test_bundles(pairs, cache=cache) is None
    incomplete_probe = probe_cached_internal_test_bundles(pairs, cache=cache)
    assert incomplete_probe.present_count == 599
    assert incomplete_probe.missing_count == 1
    assert incomplete_probe.bundles is None
    with pytest.raises(IncompleteInternalTestPredictionCache, match="incomplete"):
        require_complete_cached_internal_test_bundles(pairs, cache=cache)


def test_complete_probe_rejects_extra_malformed_cache_metadata(tmp_path: Path) -> None:
    cache = PredictionCache(tmp_path)
    pairs = tuple(_pair(index) for index in range(120))
    model = _FakeLDSR()
    for pair in pairs:
        load_or_generate_internal_test_bundle(pair, ldsr=model, cache=cache)
    (tmp_path / f"{'f' * 64}.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="metadata schema"):
        load_complete_cached_internal_test_bundles(pairs, cache=cache)
