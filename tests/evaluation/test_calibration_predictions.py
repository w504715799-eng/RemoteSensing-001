"""Calibration-only LDSR K5 prediction cache contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

from trustsr.artifacts.predictions import PredictionCache, build_identity, tensor_sha256
from trustsr.contracts import SRPair
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
    RadiometricSaturation,
)
from trustsr.evaluation.calibration_predictions import (
    A2_RESULT_SHA256,
    EXPERIMENT_SCHEMA,
    MODEL_NAME,
    PUBLICATION_COMMIT,
    SEEDS,
    CachedCalibrationPrediction,
    CalibrationPredictionBundle,
    build_cache_provenance,
    load_or_generate_calibration_bundle,
)
from trustsr.evaluation.phase2b3b_evidence import INPUT_AUDIT_SHA256


def _pair() -> LoadedCrosssensorPair:
    sample_id = "calibration-0"
    return LoadedCrosssensorPair(
        pair=SRPair(
            source=f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}",
            sample_id=sample_id,
            lr=torch.full((4, 2, 3), 0.25, dtype=torch.float32),
            hr=torch.full((4, 8, 12), 0.5, dtype=torch.float32),
            scale=4,
        ),
        metadata=CrosssensorPairMetadata(
            manifest_sha256=POST_MANIFEST_SHA256,
            sample_id=sample_id,
            split="calibration",
            spatial_group_id="group-0",
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
        return {
            "name": self.name,
            "scale": self.scale,
            "seed": self.seed,
            "backend": "tiny-cpu-fake",
        }

    def predict(self, lr: torch.Tensor) -> torch.Tensor:
        if self.owner.fail_on_predict:
            raise AssertionError("warm cache invoked prediction")
        self.owner.calls_by_seed[self.seed] = self.owner.calls_by_seed.get(self.seed, 0) + 1
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
        self.calls_by_seed: dict[int, int] = {}
        self.requested_seeds: list[int] = []

    def provenance(self) -> dict[str, object]:
        return {"name": self.name, "scale": self.scale, "seed": 3407}

    def for_seed(self, seed: int) -> _SeedModel:
        self.requested_seeds.append(seed)
        return _SeedModel(self, seed)


def _generate(
    tmp_path: Path,
    *,
    pair: LoadedCrosssensorPair | None = None,
    ldsr: _FakeLDSR | None = None,
    cache: PredictionCache | None = None,
) -> CalibrationPredictionBundle:
    return load_or_generate_calibration_bundle(
        pair or _pair(), ldsr=ldsr or _FakeLDSR(), cache=cache or PredictionCache(tmp_path)
    )


def test_fixed_k5_bundle_populates_then_reuses_real_cache_without_inference(
    tmp_path: Path,
) -> None:
    pair = _pair()
    first_model = _FakeLDSR()
    first = _generate(tmp_path, pair=pair, ldsr=first_model)
    warm_model = _FakeLDSR(fail_on_predict=True)
    second = _generate(tmp_path, pair=pair, ldsr=warm_model)

    assert SEEDS == (3407, 3408, 3409, 3410, 3411)
    assert first.sample_id == pair.pair.sample_id
    assert tuple(item.seed for item in first.items) == SEEDS
    assert first_model.requested_seeds == list(SEEDS)
    assert first_model.calls_by_seed == {seed: 1 for seed in SEEDS}
    assert warm_model.requested_seeds == list(SEEDS)
    assert warm_model.calls_by_seed == {}
    assert second == first
    assert tuple(item.prediction_sha256 for item in second.items) == tuple(
        tensor_sha256(item.tensor) for item in first.items
    )
    assert first.for_seed(3407).identity == second.for_seed(3407).identity


def test_k5_identity_binds_calibration_context_and_upstream_evidence(tmp_path: Path) -> None:
    pair = _pair()
    bundle = _generate(tmp_path, pair=pair)

    assert dict(bundle.for_seed(3407).identity.model_provenance) == {
        "name": "ldsr-s2-x4",
        "scale": 4,
        "seed": 3407,
        "backend": "tiny-cpu-fake",
        "experiment_schema": "trustsr.phase2b3b-predictions.v1",
        "post_manifest_sha256": "c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a",
        "input_audit_sha256": "fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b",
        "normalization_policy": "uint16_saturate_10000_divide_10000_v2",
        "phase2b3a_publication_commit": "b386d4b38c9f3725107eed178829955d442f5601",
        "phase2b3a_a2_result_sha256": (
            "5bb0e5138d6ed1df6c65744556be02ccd48b77d3288df39630d16fbd9cd2dce9"
        ),
    }
    assert bundle.for_seed(3407).identity.source == pair.pair.source
    assert bundle.for_seed(3407).identity.lr_sha256 == tensor_sha256(pair.pair.lr)
    assert bundle.for_seed(3407).identity.key != bundle.for_seed(3408).identity.key
    assert EXPERIMENT_SCHEMA == "trustsr.phase2b3b-predictions.v1"
    assert PUBLICATION_COMMIT == "b386d4b38c9f3725107eed178829955d442f5601"
    assert A2_RESULT_SHA256 == "5bb0e5138d6ed1df6c65744556be02ccd48b77d3288df39630d16fbd9cd2dce9"
    assert INPUT_AUDIT_SHA256 == "fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split", "development", "calibration"),
        ("split", "internal_test", "calibration"),
        ("manifest_sha256", "0" * 64, "manifest"),
        ("sample_id", "other", "identit"),
        ("crop_policy", "wrong", "policy"),
        ("normalization_policy", "wrong", "policy"),
    ],
)
def test_rejects_non_calibration_or_unfrozen_pair_metadata(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    pair = _pair()
    object.__setattr__(pair.metadata, field, value)
    model = _FakeLDSR()

    with pytest.raises(ValueError, match=message):
        _generate(tmp_path, pair=pair, ldsr=model)

    assert model.requested_seeds == []


def test_rejects_wrong_source_type_range_or_missing_radiometric_records(tmp_path: Path) -> None:
    cases: list[LoadedCrosssensorPair | object] = []
    wrong_source = _pair()
    object.__setattr__(wrong_source.pair, "source", "wrong")
    cases.append(wrong_source)
    wrong_day = _pair()
    object.__setattr__(wrong_day.metadata, "days_between", 2)
    cases.append(wrong_day)
    missing_radiometric = _pair()
    object.__setattr__(missing_radiometric.metadata, "lr_saturation", None)
    cases.append(missing_radiometric)
    cases.append(object())

    for value in cases:
        with pytest.raises(
            (TypeError, ValueError),
            match="LoadedCrosssensorPair|source|selection|radiometric",
        ):
            _generate(tmp_path, pair=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["name", "scale", "seed"])
def test_rejects_model_or_seed_provenance_conflicts(tmp_path: Path, field: str) -> None:
    model = _FakeLDSR()
    original = model.for_seed

    def conflicting(seed: int) -> _SeedModel:
        view = original(seed)
        if field == "name":
            view.name = "wrong"
        elif field == "scale":
            view.scale = 2
        else:
            view.seed = seed + 1
        return view

    model.for_seed = conflicting  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="model|scale|seed|provenance"):
        _generate(tmp_path, ldsr=model)


def test_rejects_ldsr_factory_provenance_conflict_before_seed_views(tmp_path: Path) -> None:
    model = _FakeLDSR()
    model.provenance = lambda: {"name": MODEL_NAME, "scale": 4, "seed": 9999}  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="seed"):
        _generate(tmp_path, ldsr=model)

    assert model.requested_seeds == []


def test_rejects_reserved_context_provenance_keys() -> None:
    reserved = {
        "experiment_schema": EXPERIMENT_SCHEMA,
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": INPUT_AUDIT_SHA256,
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "phase2b3a_publication_commit": PUBLICATION_COMMIT,
        "phase2b3a_a2_result_sha256": A2_RESULT_SHA256,
    }
    for key, value in reserved.items():
        with pytest.raises(ValueError, match="reserved"):
            build_cache_provenance({"name": MODEL_NAME, "scale": 4, key: value})


def test_wrong_prediction_tensor_is_rejected_by_real_cache_boundary(tmp_path: Path) -> None:
    model = _FakeLDSR()
    original = model.for_seed

    def wrong_tensor(seed: int) -> _SeedModel:
        view = original(seed)
        view.predict = lambda lr: torch.zeros((4, 8, 12), dtype=torch.float64)  # type: ignore[method-assign]
        return view

    model.for_seed = wrong_tensor  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="float32"):
        _generate(tmp_path, ldsr=model)


def test_cached_item_and_bundle_membership_cannot_be_forged(tmp_path: Path) -> None:
    bundle = _generate(tmp_path)
    item = bundle.for_seed(3407)
    with pytest.raises(ValueError, match="seed|identity"):
        CachedCalibrationPrediction(
            model_name=item.model_name,
            seed=3408,
            identity=item.identity,
            prediction_sha256=item.prediction_sha256,
            tensor=item.tensor,
        )
    with pytest.raises(ValueError, match="seeds"):
        CalibrationPredictionBundle(sample_id=bundle.sample_id, items=(item,) * 5)
    with pytest.raises(FrozenInstanceError):
        item.seed = 0  # type: ignore[misc]


def test_bundle_for_seed_rejects_absent_seed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        _generate(tmp_path).for_seed(9999)


def test_item_constructor_rejects_identity_with_wrong_calibration_context(tmp_path: Path) -> None:
    item = _generate(tmp_path).for_seed(3407)
    wrong_identity = build_identity(
        {"name": MODEL_NAME, "scale": 4, "seed": 3407},
        item.identity.source,
        item.identity.sample_id,
        _pair().pair.lr,
    )
    with pytest.raises(ValueError, match="provenance"):
        CachedCalibrationPrediction(
            model_name=MODEL_NAME,
            seed=3407,
            identity=wrong_identity,
            prediction_sha256=item.prediction_sha256,
            tensor=item.tensor,
        )
