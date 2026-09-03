"""Prediction-grid generation for the Phase 2B3-A development audit."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

from trustsr.artifacts.predictions import (
    CacheIntegrityError,
    PredictionCache,
    build_identity,
    tensor_sha256,
)
from trustsr.contracts import SRPair
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
    RadiometricSaturation,
)
from trustsr.evaluation.crosssensor_smoke import INPUT_AUDIT_SHA256
from trustsr.evaluation.development_predictions import (
    A1_SEEDS,
    K5A_SEEDS,
    K5B_SEEDS,
    CachedDevelopmentPrediction,
    build_cache_provenance,
    load_or_generate_prediction_bundle,
)


def _loaded_development_pair() -> LoadedCrosssensorPair:
    sample_id = "development-0"
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
            split="development",
            spatial_group_id="group-0",
            days_between=0,
            correlation_bin=0,
            selection_round=5,
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


class _FakeModel:
    scale = 4

    def __init__(
        self,
        name: str,
        value: float,
        *,
        fail_on_predict: bool = False,
        provenance_extra: dict[str, object] | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self.fail_on_predict = fail_on_predict
        self.provenance_extra = provenance_extra or {}
        self.calls = 0

    def provenance(self) -> dict[str, object]:
        return {
            "name": self.name,
            "scale": self.scale,
            "value": self.value,
            **self.provenance_extra,
        }

    def predict(self, lr: torch.Tensor) -> torch.Tensor:
        if self.fail_on_predict:
            raise AssertionError("warm cache invoked a model")
        self.calls += 1
        return torch.full(
            (4, lr.shape[1] * 4, lr.shape[2] * 4),
            self.value,
            dtype=torch.float32,
        )


class _FakeSeedModel(_FakeModel):
    def __init__(self, owner: _FakeLDSR, seed: int) -> None:
        super().__init__(owner.name, seed / 10_000, fail_on_predict=owner.fail_on_predict)
        self.owner = owner
        self.seed = seed
        self.backend = owner.backend

    def provenance(self) -> dict[str, object]:
        return {**super().provenance(), "seed": self.seed, "backend": self.backend}

    def predict(self, lr: torch.Tensor) -> torch.Tensor:
        result = super().predict(lr)
        self.owner.calls_by_seed[self.seed] = self.owner.calls_by_seed.get(self.seed, 0) + 1
        return result


class _FakeLDSR:
    name = "ldsr-s2-x4"
    scale = 4

    def __init__(self, *, fail_on_predict: bool = False) -> None:
        self.fail_on_predict = fail_on_predict
        self.backend = "shared-cpu-backend"
        self.calls_by_seed: dict[int, int] = {}
        self.requested_seeds: list[int] = []
        self.views: list[_FakeSeedModel] = []

    def for_seed(self, seed: int) -> _FakeSeedModel:
        self.requested_seeds.append(seed)
        view = _FakeSeedModel(self, seed)
        self.views.append(view)
        return view


def _fake_models(
    *, fail_on_predict: bool = False
) -> tuple[_FakeModel, _FakeModel, _FakeLDSR]:
    return (
        _FakeModel("bicubic-x4", 0.1, fail_on_predict=fail_on_predict),
        _FakeModel("sen2srlite-x4", 0.2, fail_on_predict=fail_on_predict),
        _FakeLDSR(fail_on_predict=fail_on_predict),
    )


def _generate(
    tmp_path: Path,
    *,
    pair: LoadedCrosssensorPair | None = None,
    models: tuple[_FakeModel, _FakeModel, _FakeLDSR] | None = None,
    seeds: tuple[int, ...] = (3407, 3408),
    cache: object | None = None,
):
    bicubic, sen2srlite, ldsr = models or _fake_models()
    return load_or_generate_prediction_bundle(
        pair or _loaded_development_pair(),
        bicubic=bicubic,
        sen2srlite=sen2srlite,
        ldsr=ldsr,
        ldsr_seeds=seeds,
        cache=cache or PredictionCache(tmp_path),
    )


def test_prediction_grid_computes_central_models_once_and_all_requested_seeds(
    tmp_path: Path,
) -> None:
    pair = _loaded_development_pair()
    bicubic, sen2srlite, ldsr = _fake_models()

    bundle = _generate(
        tmp_path,
        pair=pair,
        models=(bicubic, sen2srlite, ldsr),
        seeds=(3407, 3408, 3409),
    )

    assert bundle.sample_id == pair.pair.sample_id
    assert (bundle.bicubic.model_name, bundle.sen2srlite.model_name) == (
        "bicubic-x4",
        "sen2srlite-x4",
    )
    assert (bundle.bicubic.seed, bundle.sen2srlite.seed) == (None, None)
    assert tuple(item.seed for item in bundle.ldsr) == (3407, 3408, 3409)
    assert bicubic.calls == 1
    assert sen2srlite.calls == 1
    assert ldsr.calls_by_seed == {3407: 1, 3408: 1, 3409: 1}
    assert ldsr.requested_seeds == [3407, 3408, 3409]
    assert {id(view.backend) for view in ldsr.views} == {id(ldsr.backend)}
    assert bundle.ldsr[0].prediction_sha256 == tensor_sha256(bundle.ldsr[0].tensor)
    assert bundle.ldsr_for_seed(3408).seed == 3408


def test_prediction_grid_warm_run_invokes_no_model(tmp_path: Path) -> None:
    first = _generate(tmp_path)
    warm = _fake_models(fail_on_predict=True)

    second = _generate(tmp_path, models=warm)

    assert second == first
    assert warm[0].calls == warm[1].calls == 0
    assert warm[2].calls_by_seed == {}


def test_prediction_identities_bind_model_seed_and_frozen_input_context(tmp_path: Path) -> None:
    pair = _loaded_development_pair()
    bundle = _generate(tmp_path, pair=pair)

    bicubic_provenance = dict(bundle.bicubic.identity.model_provenance)
    seed_provenance = dict(bundle.ldsr_for_seed(3408).identity.model_provenance)
    assert bicubic_provenance == {
        "name": "bicubic-x4",
        "scale": 4,
        "value": 0.1,
        "experiment_schema": "trustsr.phase2b3a-predictions.v1",
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": INPUT_AUDIT_SHA256,
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
    }
    assert seed_provenance["seed"] == 3408
    assert bundle.ldsr[0].identity.key != bundle.ldsr[1].identity.key
    assert bundle.bicubic.identity.source == pair.pair.source
    assert bundle.bicubic.identity.sample_id == pair.pair.sample_id
    assert bundle.bicubic.identity.lr_sha256 == tensor_sha256(pair.pair.lr)


def test_prediction_cache_identity_invalidates_legacy_policy_provenance(
    tmp_path: Path,
) -> None:
    bundle = _generate(tmp_path)
    current = dict(bundle.bicubic.identity.model_provenance)
    legacy = dict(current)
    legacy["normalization_policy"] = "uint16_divide_10000_no_clip_v1"

    legacy_identity = build_identity(
        legacy,
        bundle.bicubic.identity.source,
        bundle.bicubic.identity.sample_id,
        _loaded_development_pair().pair.lr,
    )

    assert legacy_identity.key != bundle.bicubic.identity.key


def test_fixed_seed_constants_are_disjoint_where_required() -> None:
    assert A1_SEEDS == tuple(range(3407, 3432))
    assert K5A_SEEDS == tuple(range(3407, 3412))
    assert K5B_SEEDS == tuple(range(3412, 3417))
    assert set(K5A_SEEDS).isdisjoint(K5B_SEEDS)


@pytest.mark.parametrize(
    "seeds",
    [
        (),
        (3408,),
        (3407, 3407),
        (3408, 3407),
        [3407, 3408],
        (3407, True),
    ],
)
def test_prediction_grid_rejects_noncanonical_seed_tuple_before_models(
    tmp_path: Path, seeds: object
) -> None:
    models = _fake_models()

    with pytest.raises((TypeError, ValueError), match="seed"):
        _generate(tmp_path, models=models, seeds=seeds)  # type: ignore[arg-type]

    assert models[0].calls == models[1].calls == 0
    assert models[2].requested_seeds == []


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (_FakeModel("sen2srlite-x4", 0.1), "order"),
        (_FakeModel("wrong-name", 0.1), "order"),
    ],
)
def test_prediction_grid_rejects_wrong_central_model_name_or_order(
    tmp_path: Path, replacement: _FakeModel, message: str
) -> None:
    _, sen2srlite, ldsr = _fake_models()

    with pytest.raises(ValueError, match=message):
        _generate(tmp_path, models=(replacement, sen2srlite, ldsr))


def test_prediction_grid_rejects_wrong_model_scale(tmp_path: Path) -> None:
    bicubic, sen2srlite, ldsr = _fake_models()
    sen2srlite.scale = 2

    with pytest.raises(ValueError, match="scale 4"):
        _generate(tmp_path, models=(bicubic, sen2srlite, ldsr))


@pytest.mark.parametrize("field", ["name", "scale"])
def test_prediction_grid_rejects_provenance_that_does_not_identify_model(
    tmp_path: Path, field: str
) -> None:
    bicubic, sen2srlite, ldsr = _fake_models()
    bicubic.provenance_extra[field] = "wrong" if field == "name" else 2

    with pytest.raises(ValueError, match="provenance"):
        _generate(tmp_path, models=(bicubic, sen2srlite, ldsr))


def test_prediction_grid_rejects_seed_view_with_wrong_seed_provenance(tmp_path: Path) -> None:
    bicubic, sen2srlite, ldsr = _fake_models()
    original_for_seed = ldsr.for_seed

    def wrong_seed_view(seed: int) -> _FakeSeedModel:
        view = original_for_seed(seed)
        view.seed = seed + 1
        return view

    ldsr.for_seed = wrong_seed_view  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="seed"):
        _generate(tmp_path, models=(bicubic, sen2srlite, ldsr))


@pytest.mark.parametrize(
    ("metadata_field", "value"),
    [
        ("split", "calibration"),
        ("manifest_sha256", "0" * 64),
        ("sample_id", "other-sample"),
        ("crop_policy", "wrong"),
        ("normalization_policy", "wrong"),
        ("lr_saturation", None),
        ("hr_saturation", None),
    ],
)
def test_prediction_grid_rejects_non_development_or_unfrozen_metadata(
    tmp_path: Path, metadata_field: str, value: object
) -> None:
    loaded = _loaded_development_pair()
    object.__setattr__(loaded.metadata, metadata_field, value)

    with pytest.raises(ValueError, match="development|manifest|identit|policy"):
        _generate(tmp_path, pair=loaded)


def test_prediction_grid_rejects_wrong_pair_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="LoadedCrosssensorPair"):
        _generate(tmp_path, pair=object())  # type: ignore[arg-type]


def test_cache_provenance_rejects_every_reserved_context_collision() -> None:
    for key, value in (
        ("experiment_schema", "trustsr.phase2b3a-predictions.v1"),
        ("post_manifest_sha256", POST_MANIFEST_SHA256),
        ("input_audit_sha256", INPUT_AUDIT_SHA256),
        ("normalization_policy", PHASE2B3A_NORMALIZATION_POLICY),
    ):
        with pytest.raises(ValueError, match="reserved"):
            build_cache_provenance({"name": "bicubic-x4", "scale": 4, key: value})


def test_cached_integrity_failure_is_not_recomputed(tmp_path: Path) -> None:
    first = _generate(tmp_path)
    metadata_path = tmp_path / f"{first.bicubic.identity.key}.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["prediction"]["sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))
    models = _fake_models(fail_on_predict=True)

    with pytest.raises(CacheIntegrityError):
        _generate(tmp_path, models=models)

    assert models[0].calls == models[1].calls == 0
    assert models[2].calls_by_seed == {}


class _DifferingCommitCache:
    def __init__(self) -> None:
        self.committed = False

    def get(self, _identity):
        if not self.committed:
            return None
        return torch.full((4, 8, 12), 0.9, dtype=torch.float32)

    def put(self, _identity, _prediction) -> None:
        self.committed = True


def test_post_commit_reload_must_match_model_output(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="prediction differs after cache commit"):
        _generate(tmp_path, cache=_DifferingCommitCache())


def test_prediction_records_are_frozen_and_ignore_tensor_in_comparison(tmp_path: Path) -> None:
    record = _generate(tmp_path).bicubic
    same_identity_and_digest = CachedDevelopmentPrediction(
        model_name=record.model_name,
        seed=record.seed,
        identity=record.identity,
        prediction_sha256=record.prediction_sha256,
        tensor=torch.ones_like(record.tensor),
    )

    assert same_identity_and_digest == record
    with pytest.raises(FrozenInstanceError):
        record.seed = 3407  # type: ignore[misc]


def test_bundle_lookup_rejects_unrequested_seed(tmp_path: Path) -> None:
    bundle = _generate(tmp_path)

    with pytest.raises(ValueError, match="exactly one requested LDSR seed"):
        bundle.ldsr_for_seed(9999)
