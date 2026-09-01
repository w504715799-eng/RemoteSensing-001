"""Verified prediction grids for the Phase 2B3-A development audit."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import torch

from trustsr.artifacts.predictions import (
    PredictionCache,
    PredictionIdentity,
    build_identity,
    tensor_sha256,
)
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    DEVELOPMENT_BINS,
    DEVELOPMENT_DAYS,
    DEVELOPMENT_ROUNDS,
    NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    LoadedCrosssensorPair,
)
from trustsr.evaluation.crosssensor_smoke import INPUT_AUDIT_SHA256
from trustsr.models.protocols import JsonScalar, SRModel

EXPERIMENT_SCHEMA = "trustsr.phase2b3a-predictions.v1"
A1_SEEDS = tuple(range(3407, 3432))
K5A_SEEDS = tuple(range(3407, 3412))
K5B_SEEDS = tuple(range(3412, 3417))
MODEL_NAMES = ("bicubic-x4", "sen2srlite-x4", "ldsr-s2-x4")
_CONTEXT_KEYS = (
    "experiment_schema",
    "post_manifest_sha256",
    "input_audit_sha256",
)


@dataclass(frozen=True)
class CachedDevelopmentPrediction:
    model_name: str
    seed: int | None
    identity: PredictionIdentity
    prediction_sha256: str
    tensor: torch.Tensor = field(compare=False, repr=False)


@dataclass(frozen=True)
class DevelopmentPredictionBundle:
    sample_id: str
    bicubic: CachedDevelopmentPrediction
    sen2srlite: CachedDevelopmentPrediction
    ldsr: tuple[CachedDevelopmentPrediction, ...]

    def ldsr_for_seed(self, seed: int) -> CachedDevelopmentPrediction:
        matches = tuple(item for item in self.ldsr if item.seed == seed)
        if len(matches) != 1:
            raise ValueError("prediction bundle does not contain exactly one requested LDSR seed")
        return matches[0]


def build_cache_provenance(
    model_provenance: Mapping[str, JsonScalar],
) -> dict[str, JsonScalar]:
    """Bind model provenance to the frozen Phase 2B3-A input context."""

    if not isinstance(model_provenance, Mapping):
        raise TypeError("model provenance must be a mapping")
    if any(key in model_provenance for key in _CONTEXT_KEYS):
        raise ValueError("model provenance contains a reserved experiment context key")
    result = dict(model_provenance)
    result.update(
        {
            "experiment_schema": EXPERIMENT_SCHEMA,
            "post_manifest_sha256": POST_MANIFEST_SHA256,
            "input_audit_sha256": INPUT_AUDIT_SHA256,
        }
    )
    return result


def _validate_pair(loaded: LoadedCrosssensorPair) -> LoadedCrosssensorPair:
    if not isinstance(loaded, LoadedCrosssensorPair):
        raise TypeError("prediction input must be a LoadedCrosssensorPair")
    loaded.pair.validate()
    metadata = loaded.metadata
    if metadata.split != "development":
        raise ValueError("prediction input must use development metadata")
    if metadata.manifest_sha256 != POST_MANIFEST_SHA256:
        raise ValueError("development pair has the wrong manifest")
    if metadata.sample_id != loaded.pair.sample_id:
        raise ValueError("development pair and metadata identities differ")
    if loaded.pair.source != f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}":
        raise ValueError("development pair has the wrong source identity")
    if (
        metadata.crop_policy != CROP_POLICY
        or metadata.normalization_policy != NORMALIZATION_POLICY
    ):
        raise ValueError("development pair has the wrong input policy")
    if (
        metadata.days_between not in DEVELOPMENT_DAYS
        or metadata.correlation_bin not in DEVELOPMENT_BINS
        or metadata.selection_round not in DEVELOPMENT_ROUNDS
    ):
        raise ValueError("development pair metadata is outside the frozen selection")
    return loaded


def _validate_seeds(seeds: tuple[int, ...]) -> tuple[int, ...]:
    if type(seeds) is not tuple:
        raise TypeError("LDSR seeds must be an immutable tuple")
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("LDSR seeds must be integers")
    if 3407 not in seeds:
        raise ValueError("LDSR seeds must contain seed 3407")
    if any(left >= right for left, right in pairwise(seeds)):
        raise ValueError("LDSR seeds must be strictly increasing and unique")
    return seeds


def _validate_model_slot(model: Any, expected_name: str) -> None:
    if getattr(model, "name", None) != expected_name:
        raise ValueError("development prediction model order must match the frozen model order")
    if getattr(model, "scale", None) != 4:
        raise ValueError(f"model {expected_name!r} must use scale 4")


def _model_provenance(
    model: SRModel, expected_name: str, *, seed: int | None
) -> Mapping[str, JsonScalar]:
    _validate_model_slot(model, expected_name)
    provenance = model.provenance()
    if provenance.get("name") != expected_name or provenance.get("scale") != 4:
        raise ValueError(f"model {expected_name!r} provenance does not identify the model")
    if seed is not None and provenance.get("seed") != seed:
        raise ValueError("LDSR seed provenance does not match the requested seed")
    build_cache_provenance(provenance)
    return provenance


def _load_or_generate(
    loaded: LoadedCrosssensorPair,
    model: SRModel,
    *,
    expected_name: str,
    seed: int | None,
    cache: PredictionCache,
) -> CachedDevelopmentPrediction:
    provenance = _model_provenance(model, expected_name, seed=seed)
    pair = loaded.pair
    identity = build_identity(
        build_cache_provenance(provenance), pair.source, pair.sample_id, pair.lr
    )
    prediction = cache.get(identity)
    if prediction is None:
        produced = model.predict(pair.lr)
        produced_sha256 = tensor_sha256(produced)
        cache.put(identity, produced)
        prediction = cache.get(identity)
        if prediction is None or tensor_sha256(prediction) != produced_sha256:
            raise RuntimeError("prediction differs after cache commit")
    return CachedDevelopmentPrediction(
        model_name=model.name,
        seed=seed,
        identity=identity,
        prediction_sha256=tensor_sha256(prediction),
        tensor=prediction,
    )


def load_or_generate_prediction_bundle(
    pair: LoadedCrosssensorPair,
    *,
    bicubic: SRModel,
    sen2srlite: SRModel,
    ldsr: Any,
    ldsr_seeds: tuple[int, ...],
    cache: PredictionCache,
) -> DevelopmentPredictionBundle:
    """Load or atomically generate one fixed three-model prediction grid."""

    loaded = _validate_pair(pair)
    seeds = _validate_seeds(ldsr_seeds)
    for model, expected_name in zip(
        (bicubic, sen2srlite, ldsr), MODEL_NAMES, strict=True
    ):
        _validate_model_slot(model, expected_name)
    central_bicubic = _load_or_generate(
        loaded, bicubic, expected_name=MODEL_NAMES[0], seed=None, cache=cache
    )
    central_sen2srlite = _load_or_generate(
        loaded, sen2srlite, expected_name=MODEL_NAMES[1], seed=None, cache=cache
    )
    ldsr_predictions = tuple(
        _load_or_generate(
            loaded,
            ldsr.for_seed(seed),
            expected_name=MODEL_NAMES[2],
            seed=seed,
            cache=cache,
        )
        for seed in seeds
    )
    return DevelopmentPredictionBundle(
        sample_id=loaded.pair.sample_id,
        bicubic=central_bicubic,
        sen2srlite=central_sen2srlite,
        ldsr=ldsr_predictions,
    )
