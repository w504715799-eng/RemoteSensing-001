"""Fixed calibration-only LDSR K5 score and R9 risk maps."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch

from trustsr.artifacts.predictions import PredictionCache, tensor_sha256
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
from trustsr.evaluation.calibration_maps import (
    A2_RESULT_SHA256,
    PUBLICATION_COMMIT,
    RISK_NAME,
    RISK_WINDOW,
    SCORE_NAME,
    SCORE_SCHEMA_VERSION,
    CachedCalibrationScore,
    CalibrationMaps,
    load_or_compute_calibration_maps,
)
from trustsr.evaluation.calibration_predictions import (
    MODEL_NAME,
    SEEDS,
    CalibrationPredictionBundle,
    load_or_generate_calibration_bundle,
)
from trustsr.evaluation.phase2b3b_evidence import INPUT_AUDIT_SHA256, PRODUCER_REVISION


def _pair(*, sample_id: str = "calibration-0") -> LoadedCrosssensorPair:
    lr = torch.full((4, 3, 3), 0.25, dtype=torch.float32)
    return LoadedCrosssensorPair(
        pair=SRPair(
            source=f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}",
            sample_id=sample_id,
            lr=lr,
            hr=torch.zeros((4, 12, 12), dtype=torch.float32),
            scale=4,
        ),
        metadata=CrosssensorPairMetadata(
            manifest_sha256=POST_MANIFEST_SHA256,
            sample_id=sample_id,
            split="calibration",
            spatial_group_id=f"group-{sample_id}",
            days_between=0,
            correlation_bin=0,
            selection_round=1,
            lr_asset_sha256="a" * 64,
            hr_asset_sha256="b" * 64,
            lr_crop_transform=(10.0, 0.0, 10.0, 0.0, -10.0, -10.0),
            hr_crop_transform=(2.5, 0.0, 10.0, 0.0, -2.5, -10.0),
            crop_bounds=(10.0, -30.0, 40.0, -10.0),
            crop_policy=CROP_POLICY,
            normalization_policy=PHASE2B3A_NORMALIZATION_POLICY,
            lr_saturation=RadiometricSaturation(2500, 2500, 0, (0, 0, 0, 0)),
            hr_saturation=RadiometricSaturation(0, 0, 0, (0, 0, 0, 0)),
        ),
    )


class _SeedModel:
    name = MODEL_NAME
    scale = 4

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def provenance(self) -> dict[str, object]:
        return {"name": self.name, "scale": self.scale, "seed": self.seed, "backend": "cpu-test"}

    def predict(self, lr: torch.Tensor) -> torch.Tensor:
        value = 0.1 * (self.seed - SEEDS[0] + 1)
        return torch.full((4, lr.shape[1] * 4, lr.shape[2] * 4), value, dtype=torch.float32)


class _LDSR:
    name = MODEL_NAME
    scale = 4

    def provenance(self) -> dict[str, object]:
        return {"name": self.name, "scale": self.scale, "seed": SEEDS[0], "backend": "cpu-test"}

    def for_seed(self, seed: int) -> _SeedModel:
        return _SeedModel(seed)


def _bundle(pair: LoadedCrosssensorPair, root: Path) -> CalibrationPredictionBundle:
    return load_or_generate_calibration_bundle(pair, ldsr=_LDSR(), cache=PredictionCache(root))


def test_fixed_k5_score_and_r9_risk_match_hand_calculation(tmp_path: Path) -> None:
    pair = _pair()
    maps = load_or_compute_calibration_maps(
        pair, _bundle(pair, tmp_path / "predictions"), ScoreCache(tmp_path / "scores")
    )

    # Values are 0.1 ... 0.5, whose population variance is 0.02 in every band/pixel.
    torch.testing.assert_close(maps.score.tensor, torch.full((12, 12), 0.02, dtype=torch.float64))
    # The central (seed 3407) prediction is 0.1 and HR is zero, so every reflected R9 mean is 0.1.
    torch.testing.assert_close(maps.risk, torch.full((12, 12), 0.1, dtype=torch.float64))
    assert maps.risk_sha256
    assert maps.risk_name == RISK_NAME == "local_l1_risk"
    assert maps.risk_window == RISK_WINDOW == 9


def test_score_cache_hit_reuses_exact_identity_and_bytes_without_recomputing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = _pair()
    bundle = _bundle(pair, tmp_path / "predictions")
    cache = ScoreCache(tmp_path / "scores")
    first = load_or_compute_calibration_maps(pair, bundle, cache)
    paths = tuple((tmp_path / "scores").glob(f"{first.score.identity.key}.*"))
    before = {path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in paths}

    monkeypatch.setattr(
        "trustsr.evaluation.calibration_maps.ensemble_variance_score",
        lambda _samples: (_ for _ in ()).throw(AssertionError("warm score cache recomputed")),
    )
    second = load_or_compute_calibration_maps(pair, bundle, cache)

    assert second.score.identity.as_dict() == first.score.identity.as_dict()
    assert second.score.score_sha256 == first.score.score_sha256
    assert {path.name: (path.stat().st_mtime_ns, path.read_bytes()) for path in paths} == before


def test_score_identity_has_exact_fixed_operator_and_frozen_context(tmp_path: Path) -> None:
    pair = _pair()
    maps = load_or_compute_calibration_maps(
        pair, _bundle(pair, tmp_path / "predictions"), ScoreCache(tmp_path / "scores")
    )

    identity = maps.score.identity
    assert identity.score_name == SCORE_NAME == "ldsr_variance_k5"
    assert identity.score_schema_version == SCORE_SCHEMA_VERSION == 1
    assert identity.input_sha256s == maps.score_prediction_sha256s
    assert dict(identity.operator_parameters) == {
        "algorithm": "ensemble_variance_score",
        "band_reduction": "mean",
        "correction": 0,
        "seed_first": 3407,
        "seed_last": 3411,
        "seed_count": 5,
        "lr_sha256": tensor_sha256(pair.pair.lr),
        "source": pair.pair.source,
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": INPUT_AUDIT_SHA256,
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "crop_policy": CROP_POLICY,
        "phase2b3a_publication_commit": PUBLICATION_COMMIT,
        "phase2b3a_a2_result_sha256": A2_RESULT_SHA256,
        "phase2b3a_producer_revision": PRODUCER_REVISION,
    }


@pytest.mark.parametrize("split", ["development", "internal_test"])
def test_rejects_non_calibration_pair_before_maps(tmp_path: Path, split: str) -> None:
    pair = _pair()
    rejected = replace(pair, metadata=replace(pair.metadata, split=split))
    with pytest.raises(ValueError, match="calibration"):
        load_or_compute_calibration_maps(
            rejected, _bundle(pair, tmp_path / "predictions"), ScoreCache(tmp_path / "scores")
        )


def test_rejects_bundle_sample_source_lr_and_k5_order_mismatches(tmp_path: Path) -> None:
    pair = _pair()
    bundle = _bundle(pair, tmp_path / "predictions")
    cache = ScoreCache(tmp_path / "scores")

    other = _pair(sample_id="calibration-other")
    other_bundle = _bundle(other, tmp_path / "other-predictions")
    with pytest.raises(ValueError, match="sample"):
        load_or_compute_calibration_maps(pair, other_bundle, cache)
    with pytest.raises(ValueError, match="source"):
        load_or_compute_calibration_maps(
            replace(pair, pair=replace(pair.pair, source="wrong-source")), bundle, cache
        )
    changed_lr = pair.pair.lr.clone()
    changed_lr[0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="LR|input"):
        load_or_compute_calibration_maps(
            replace(pair, pair=replace(pair.pair, lr=changed_lr)), bundle, cache
        )

    forged = object.__new__(CalibrationPredictionBundle)
    object.__setattr__(forged, "sample_id", bundle.sample_id)
    object.__setattr__(forged, "items", tuple(reversed(bundle.items)))
    with pytest.raises(ValueError, match="ordered|K5|seeds"):
        load_or_compute_calibration_maps(pair, forged, cache)


def test_rejects_wrong_pair_type_and_forged_or_mutable_result_state(tmp_path: Path) -> None:
    pair = _pair()
    bundle = _bundle(pair, tmp_path / "predictions")
    cache = ScoreCache(tmp_path / "scores")
    with pytest.raises(TypeError, match="LoadedCrosssensorPair"):
        load_or_compute_calibration_maps(object(), bundle, cache)  # type: ignore[arg-type]

    maps = load_or_compute_calibration_maps(pair, bundle, cache)
    with pytest.raises(ValueError, match="digest"):
        CachedCalibrationScore(
            name=maps.score.name,
            identity=maps.score.identity,
            score_sha256="0" * 64,
            tensor=maps.score.tensor,
        )
    forged_identity = replace(
        maps.score.identity,
        operator_parameters={**maps.score.identity.operator_parameters, "correction": 1},
    )
    with pytest.raises(ValueError, match="fixed B3-B"):
        CachedCalibrationScore(
            name=maps.score.name,
            identity=forged_identity,
            score_sha256=maps.score.score_sha256,
            tensor=maps.score.tensor,
        )
    with pytest.raises(ValueError, match="risk"):
        CalibrationMaps(
            sample_id=maps.sample_id,
            score=maps.score,
            score_prediction_sha256s=maps.score_prediction_sha256s,
            risk_name=maps.risk_name,
            risk_window=maps.risk_window,
            risk_sha256="0" * 64,
            risk=maps.risk,
        )
    with pytest.raises(FrozenInstanceError):
        maps.sample_id = "mutated"  # type: ignore[misc]
    with pytest.raises(TypeError):
        maps.score.identity.operator_parameters["correction"] = 1  # type: ignore[index]


def test_calibration_maps_rejects_risk_above_reflectance_bound(tmp_path: Path) -> None:
    pair = _pair()
    maps = load_or_compute_calibration_maps(
        pair, _bundle(pair, tmp_path / "predictions"), ScoreCache(tmp_path / "scores")
    )
    risk = torch.full_like(maps.risk, 1.1)

    with pytest.raises(ValueError, match=r"risk.*\[0, 1\]"):
        CalibrationMaps(
            sample_id=maps.sample_id,
            score=maps.score,
            score_prediction_sha256s=maps.score_prediction_sha256s,
            risk_name=maps.risk_name,
            risk_window=maps.risk_window,
            risk_sha256=tensor_sha256(risk),
            risk=risk,
        )


def test_cached_calibration_score_rejects_value_above_k5_variance_bound(tmp_path: Path) -> None:
    pair = _pair()
    maps = load_or_compute_calibration_maps(
        pair, _bundle(pair, tmp_path / "predictions"), ScoreCache(tmp_path / "scores")
    )
    score = torch.full_like(maps.score.tensor, 0.2500001)

    with pytest.raises(ValueError, match="score.*0.25"):
        CachedCalibrationScore(
            name=maps.score.name,
            identity=maps.score.identity,
            score_sha256=tensor_sha256(score),
            tensor=score,
        )


def test_calibration_maps_rejects_forged_cached_score_state(tmp_path: Path) -> None:
    pair = _pair()
    maps = load_or_compute_calibration_maps(
        pair, _bundle(pair, tmp_path / "predictions"), ScoreCache(tmp_path / "scores")
    )
    forged_score = object.__new__(CachedCalibrationScore)
    object.__setattr__(forged_score, "name", maps.score.name)
    object.__setattr__(forged_score, "identity", maps.score.identity)
    object.__setattr__(forged_score, "score_sha256", "0" * 64)
    object.__setattr__(forged_score, "tensor", maps.score.tensor)

    with pytest.raises(ValueError, match="score.*digest"):
        CalibrationMaps(
            sample_id=maps.sample_id,
            score=forged_score,
            score_prediction_sha256s=maps.score_prediction_sha256s,
            risk_name=maps.risk_name,
            risk_window=maps.risk_window,
            risk_sha256=maps.risk_sha256,
            risk=maps.risk,
        )
