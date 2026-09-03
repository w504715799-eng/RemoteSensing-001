"""Calibration-only fixed LDSR K5 prediction bundles for Phase 2B3-B."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from trustsr.artifacts.predictions import (
    PredictionCache,
    PredictionIdentity,
    build_identity,
    tensor_sha256,
)
from trustsr.data.crosssensor_pairs import (
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
    RadiometricSaturation,
)
from trustsr.evaluation import phase2b3b_evidence
from trustsr.models.protocols import JsonScalar, SRModel

EXPERIMENT_SCHEMA = "trustsr.phase2b3b-predictions.v1"
SEEDS = (3407, 3408, 3409, 3410, 3411)
MODEL_NAME = "ldsr-s2-x4"
SCALE = 4
PUBLICATION_COMMIT = phase2b3b_evidence.PUBLICATION_COMMIT
A2_RESULT_SHA256 = phase2b3b_evidence.PUBLISHED_EVIDENCE_SHA256S[
    "sen2naipv2-development-score-audit-v1.json"
]
_DAYS = (-1, 0, 1)
_BINS = (0, 1, 2, 3)
_ROUNDS = tuple(range(1, 11))
_CONTEXT_KEYS = (
    "experiment_schema",
    "post_manifest_sha256",
    "input_audit_sha256",
    "normalization_policy",
    "phase2b3a_publication_commit",
    "phase2b3a_a2_result_sha256",
)


def build_cache_provenance(
    model_provenance: Mapping[str, JsonScalar],
) -> dict[str, JsonScalar]:
    """Bind one LDSR seed provenance to immutable B3-B input evidence."""

    if not isinstance(model_provenance, Mapping):
        raise TypeError("model provenance must be a mapping")
    if any(key in model_provenance for key in _CONTEXT_KEYS):
        raise ValueError("model provenance contains a reserved experiment context key")
    result = dict(model_provenance)
    result.update(
        {
            "experiment_schema": EXPERIMENT_SCHEMA,
            "post_manifest_sha256": phase2b3b_evidence.POST_MANIFEST_SHA256,
            "input_audit_sha256": phase2b3b_evidence.INPUT_AUDIT_SHA256,
            "normalization_policy": phase2b3b_evidence.NORMALIZATION_POLICY,
            "phase2b3a_publication_commit": PUBLICATION_COMMIT,
            "phase2b3a_a2_result_sha256": A2_RESULT_SHA256,
        }
    )
    return result


def _validate_pair(loaded: LoadedCrosssensorPair) -> LoadedCrosssensorPair:
    if not isinstance(loaded, LoadedCrosssensorPair):
        raise TypeError("prediction input must be a LoadedCrosssensorPair")
    loaded.pair.validate()
    metadata: CrosssensorPairMetadata = loaded.metadata
    if type(metadata.split) is not str or metadata.split != "calibration":
        raise ValueError("prediction input must use calibration metadata")
    if metadata.manifest_sha256 != phase2b3b_evidence.POST_MANIFEST_SHA256:
        raise ValueError("calibration pair has the wrong manifest")
    if metadata.sample_id != loaded.pair.sample_id:
        raise ValueError("calibration pair and metadata identities differ")
    if loaded.pair.source != (
        f"sen2naipv2-crosssensor/{phase2b3b_evidence.POST_MANIFEST_SHA256}"
    ):
        raise ValueError("calibration pair has the wrong source identity")
    if (
        metadata.crop_policy != phase2b3b_evidence.CROP_POLICY
        or metadata.normalization_policy != phase2b3b_evidence.NORMALIZATION_POLICY
    ):
        raise ValueError("calibration pair has the wrong input policy")
    if not isinstance(metadata.lr_saturation, RadiometricSaturation) or not isinstance(
        metadata.hr_saturation, RadiometricSaturation
    ):
        raise ValueError("calibration pair requires radiometric saturation records")
    if (
        type(metadata.days_between) is not int
        or type(metadata.correlation_bin) is not int
        or type(metadata.selection_round) is not int
        or metadata.days_between not in _DAYS
        or metadata.correlation_bin not in _BINS
        or metadata.selection_round not in _ROUNDS
    ):
        raise ValueError("calibration pair metadata is outside the frozen selection")
    return loaded


def _validate_model_slot(model: Any) -> None:
    if getattr(model, "name", None) != MODEL_NAME:
        raise ValueError("calibration prediction model must be ldsr-s2-x4")
    if getattr(model, "scale", None) != SCALE:
        raise ValueError("calibration prediction model must use scale 4")


def _validate_ldsr_factory(model: Any) -> None:
    _validate_model_slot(model)
    try:
        provenance = model.provenance()
    except AttributeError as exc:
        raise TypeError("calibration LDSR model must provide provenance") from exc
    if not isinstance(provenance, Mapping):
        raise TypeError("model provenance must be a mapping")
    if provenance.get("name") != MODEL_NAME or provenance.get("scale") != SCALE:
        raise ValueError("model provenance does not identify ldsr-s2-x4 scale 4")
    if provenance.get("seed") != SEEDS[0]:
        raise ValueError("calibration LDSR factory seed provenance must be seed 3407")


def _model_provenance(model: SRModel, *, seed: int) -> Mapping[str, JsonScalar]:
    _validate_model_slot(model)
    provenance = model.provenance()
    if not isinstance(provenance, Mapping):
        raise TypeError("model provenance must be a mapping")
    if provenance.get("name") != MODEL_NAME or provenance.get("scale") != SCALE:
        raise ValueError("model provenance does not identify ldsr-s2-x4 scale 4")
    if provenance.get("seed") != seed:
        raise ValueError("LDSR seed provenance does not match the requested seed")
    build_cache_provenance(provenance)
    return provenance


@dataclass(frozen=True)
class CachedCalibrationPrediction:
    """One cache-verified prediction belonging to the fixed calibration K5 set."""

    model_name: str
    seed: int
    identity: PredictionIdentity
    prediction_sha256: str
    tensor: torch.Tensor = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.model_name != MODEL_NAME or type(self.seed) is not int or self.seed not in SEEDS:
            raise ValueError("cached calibration prediction has an invalid model or seed")
        if not isinstance(self.identity, PredictionIdentity):
            raise TypeError("cached calibration prediction requires a PredictionIdentity")
        provenance = self.identity.model_provenance
        if (
            provenance.get("name") != MODEL_NAME
            or provenance.get("scale") != SCALE
            or provenance.get("seed") != self.seed
        ):
            raise ValueError("cached calibration prediction identity provenance is invalid")
        if dict(provenance) != build_cache_provenance(
            {key: value for key, value in provenance.items() if key not in _CONTEXT_KEYS}
        ):
            raise ValueError("cached calibration prediction identity provenance is invalid")
        if not isinstance(self.tensor, torch.Tensor) or self.prediction_sha256 != tensor_sha256(
            self.tensor
        ):
            raise ValueError("cached calibration prediction tensor digest is invalid")
        expected_shape = (
            4,
            self.identity.lr_shape[1] * SCALE,
            self.identity.lr_shape[2] * SCALE,
        )
        if (
            self.tensor.dtype != torch.float32
            or tuple(self.tensor.shape) != expected_shape
            or not torch.isfinite(self.tensor).all()
            or (self.tensor < 0).any()
            or (self.tensor > 1).any()
        ):
            raise ValueError("cached calibration prediction tensor is outside cache contract")


@dataclass(frozen=True)
class CalibrationPredictionBundle:
    """One calibration sample's strictly ordered, immutable LDSR K5 bundle."""

    sample_id: str
    items: tuple[CachedCalibrationPrediction, ...]

    def __post_init__(self) -> None:
        if type(self.sample_id) is not str or not self.sample_id:
            raise TypeError("calibration prediction bundle sample_id must be a non-empty string")
        if type(self.items) is not tuple or tuple(item.seed for item in self.items) != SEEDS:
            raise ValueError(
                "calibration prediction bundle must contain the fixed ordered K5 seeds"
            )
        if any(
            not isinstance(item, CachedCalibrationPrediction)
            or item.identity.sample_id != self.sample_id
            for item in self.items
        ):
            raise ValueError("calibration prediction bundle items have mismatched identities")

    def for_seed(self, seed: int) -> CachedCalibrationPrediction:
        matches = tuple(item for item in self.items if item.seed == seed)
        if len(matches) != 1:
            raise ValueError("prediction bundle does not contain exactly one requested LDSR seed")
        return matches[0]


def _load_or_generate(
    loaded: LoadedCrosssensorPair,
    model: SRModel,
    *,
    seed: int,
    cache: PredictionCache,
) -> CachedCalibrationPrediction:
    provenance = _model_provenance(model, seed=seed)
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
    return CachedCalibrationPrediction(
        model_name=MODEL_NAME,
        seed=seed,
        identity=identity,
        prediction_sha256=tensor_sha256(prediction),
        tensor=prediction,
    )


def load_or_generate_calibration_bundle(
    pair: LoadedCrosssensorPair,
    *,
    ldsr: Any,
    cache: PredictionCache,
) -> CalibrationPredictionBundle:
    """Load or atomically generate the sole permitted calibration LDSR K5 bundle."""

    loaded = _validate_pair(pair)
    if not isinstance(cache, PredictionCache):
        raise TypeError("calibration prediction cache must be a PredictionCache")
    _validate_ldsr_factory(ldsr)
    try:
        items = tuple(
            _load_or_generate(loaded, ldsr.for_seed(seed), seed=seed, cache=cache)
            for seed in SEEDS
        )
    except AttributeError as exc:
        raise TypeError("calibration LDSR model must provide for_seed") from exc
    return CalibrationPredictionBundle(sample_id=loaded.pair.sample_id, items=items)
