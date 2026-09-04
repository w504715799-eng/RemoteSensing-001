"""Calibration-only fixed LDSR K5 prediction bundles for Phase 2B3-B."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
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
from trustsr.evaluation.calibration_model_identity import (
    CalibrationModelIdentity,
    validate_cached_calibration_model_identity,
    validate_calibration_model_identity,
)
from trustsr.jsonio import canonical_json
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
    "split",
    "post_manifest_sha256",
    "input_audit_sha256",
    "normalization_policy",
    "phase2b3a_publication_commit",
    "phase2b3a_a2_result_sha256",
)
_CACHE_KEY = re.compile(r"[0-9a-f]{64}")
_MAX_CACHE_METADATA_BYTES = 128 * 1024


def build_cache_provenance(
    model_provenance: Mapping[str, object],
) -> dict[str, JsonScalar]:
    """Bind one LDSR seed provenance to immutable B3-B input evidence."""

    if not isinstance(model_provenance, Mapping):
        raise TypeError("model provenance must be a mapping")
    if any(key in model_provenance for key in _CONTEXT_KEYS):
        raise ValueError("model provenance contains a reserved experiment context key")
    return _cache_provenance(validate_calibration_model_identity(model_provenance))


def _cache_provenance(identity: CalibrationModelIdentity) -> dict[str, JsonScalar]:
    """Combine a normalized model identity with the immutable calibration context."""

    result = identity.as_dict()
    result.update(
        {
            "experiment_schema": EXPERIMENT_SCHEMA,
            "split": "calibration",
            "post_manifest_sha256": phase2b3b_evidence.POST_MANIFEST_SHA256,
            "input_audit_sha256": phase2b3b_evidence.INPUT_AUDIT_SHA256,
            "normalization_policy": phase2b3b_evidence.NORMALIZATION_POLICY,
            "phase2b3a_publication_commit": PUBLICATION_COMMIT,
            "phase2b3a_a2_result_sha256": A2_RESULT_SHA256,
        }
    )
    return result


def validate_cached_calibration_prediction_provenance(
    provenance: Mapping[str, object], *, seed: int
) -> dict[str, JsonScalar]:
    """Fail closed on the exact host-free prediction provenance written to cache."""

    if not isinstance(provenance, Mapping):
        raise TypeError("cached prediction provenance must be a mapping")
    identity_keys = set(CalibrationModelIdentity.__dataclass_fields__)
    if set(provenance) != identity_keys | set(_CONTEXT_KEYS):
        raise ValueError("cached prediction provenance keys are invalid")
    identity = validate_cached_calibration_model_identity(
        {key: provenance[key] for key in identity_keys}
    )
    if identity.seed != seed:
        raise ValueError("cached prediction provenance seed does not match the fixed K5 slot")
    expected = _cache_provenance(identity)
    if dict(provenance) != expected:
        raise ValueError("cached prediction provenance has the wrong calibration context")
    return expected


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
    if loaded.pair.source != (f"sen2naipv2-crosssensor/{phase2b3b_evidence.POST_MANIFEST_SHA256}"):
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
    identity = validate_calibration_model_identity(provenance)
    if identity.seed != SEEDS[0]:
        raise ValueError("calibration LDSR factory seed provenance must be seed 3407")


def _model_provenance(model: SRModel, *, seed: int) -> dict[str, JsonScalar]:
    _validate_model_slot(model)
    provenance = model.provenance()
    identity = validate_calibration_model_identity(provenance)
    if identity.seed != seed:
        raise ValueError("LDSR seed provenance does not match the requested seed")
    return _cache_provenance(identity)


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
        if self.identity.source != (
            f"sen2naipv2-crosssensor/{phase2b3b_evidence.POST_MANIFEST_SHA256}"
        ):
            raise ValueError("cached calibration prediction identity source is invalid")
        try:
            validate_cached_calibration_prediction_provenance(
                self.identity.model_provenance, seed=self.seed
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "cached calibration prediction identity provenance is invalid"
            ) from exc
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
            or self.tensor.device.type != "cpu"
            or not self.tensor.is_contiguous()
            or self.tensor.requires_grad
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
        if type(self.items) is not tuple:
            raise TypeError("calibration prediction bundle items must be an exact tuple")
        if any(not isinstance(item, CachedCalibrationPrediction) for item in self.items):
            raise TypeError(
                "calibration prediction bundle items must be CachedCalibrationPrediction"
            )
        if tuple(item.seed for item in self.items) != SEEDS:
            raise ValueError(
                "calibration prediction bundle must contain the fixed ordered K5 seeds"
            )
        if any(item.identity.sample_id != self.sample_id for item in self.items):
            raise ValueError("calibration prediction bundle items have mismatched identities")
        try:
            model_identities = tuple(
                {
                    key: value
                    for key, value in validate_cached_calibration_prediction_provenance(
                        item.identity.model_provenance, seed=item.seed
                    ).items()
                    if key not in _CONTEXT_KEYS and key != "seed"
                }
                for item in self.items
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "calibration prediction bundle items have invalid model scientific identities"
            ) from exc
        if any(identity != model_identities[0] for identity in model_identities[1:]):
            raise ValueError(
                "calibration prediction bundle items have mismatched model scientific identities"
            )
        first_identity = self.items[0].identity
        input_identity = (
            first_identity.source,
            first_identity.lr_shape,
            first_identity.lr_dtype,
            first_identity.lr_sha256,
        )
        if any(
            (
                item.identity.source,
                item.identity.lr_shape,
                item.identity.lr_dtype,
                item.identity.lr_sha256,
            )
            != input_identity
            for item in self.items[1:]
        ):
            raise ValueError("calibration prediction bundle items have mismatched inputs")

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
    identity = build_identity(provenance, pair.source, pair.sample_id, pair.lr)
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
            _load_or_generate(loaded, ldsr.for_seed(seed), seed=seed, cache=cache) for seed in SEEDS
        )
    except AttributeError as exc:
        raise TypeError("calibration LDSR model must provide for_seed") from exc
    return CalibrationPredictionBundle(sample_id=loaded.pair.sample_id, items=items)


def _cached_candidate(
    metadata_path: Path,
    *,
    pairs_by_id: Mapping[str, LoadedCrosssensorPair],
    cache: PredictionCache,
) -> CachedCalibrationPrediction | None:
    if (
        metadata_path.is_symlink()
        or not metadata_path.is_file()
        or _CACHE_KEY.fullmatch(metadata_path.stem) is None
    ):
        raise ValueError("calibration cache metadata path is unsafe")
    payload = metadata_path.read_bytes()
    if len(payload) > _MAX_CACHE_METADATA_BYTES:
        raise ValueError("calibration cache metadata exceeds the size limit")
    try:
        metadata = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("calibration cache metadata is not valid JSON") from exc
    if (
        type(metadata) is not dict
        or canonical_json(metadata) != payload
        or set(metadata)
        != {"schema_version", "cache_key", "identity", "prediction", "tensor_filename"}
        or metadata["schema_version"] != 1
        or metadata["cache_key"] != metadata_path.stem
        or metadata["tensor_filename"] != f"{metadata_path.stem}.safetensors"
    ):
        raise ValueError("calibration cache metadata schema is invalid")
    identity_value = metadata["identity"]
    if type(identity_value) is not dict or set(identity_value) != {
        "model_provenance",
        "source",
        "sample_id",
        "lr",
    }:
        raise ValueError("calibration cache identity schema is invalid")
    lr_value = identity_value["lr"]
    provenance = identity_value["model_provenance"]
    if (
        type(lr_value) is not dict
        or set(lr_value) != {"shape", "dtype", "sha256"}
        or type(provenance) is not dict
    ):
        raise ValueError("calibration cache input identity is invalid")
    seed = provenance.get("seed")
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("calibration cache seed is outside the fixed K5 set")
    validated_provenance = validate_cached_calibration_prediction_provenance(
        provenance, seed=seed
    )
    shape = lr_value["shape"]
    if type(shape) is not list or len(shape) != 3:
        raise ValueError("calibration cache LR shape is invalid")
    try:
        identity = PredictionIdentity(
            model_provenance=validated_provenance,
            source=identity_value["source"],
            sample_id=identity_value["sample_id"],
            lr_shape=tuple(shape),
            lr_dtype=lr_value["dtype"],
            lr_sha256=lr_value["sha256"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("calibration cache prediction identity is invalid") from exc
    if identity.key != metadata_path.stem:
        raise ValueError("calibration cache filename differs from its identity")
    loaded = pairs_by_id.get(identity.sample_id)
    if loaded is None:
        return None
    expected_identity = build_identity(
        validated_provenance,
        loaded.pair.source,
        loaded.pair.sample_id,
        loaded.pair.lr,
    )
    if identity != expected_identity:
        return None
    prediction = cache.get(identity)
    if prediction is None:
        raise ValueError("calibration cache metadata has no committed prediction")
    return CachedCalibrationPrediction(
        model_name=MODEL_NAME,
        seed=seed,
        identity=identity,
        prediction_sha256=tensor_sha256(prediction),
        tensor=prediction,
    )


def load_complete_cached_calibration_bundles(
    pairs: tuple[LoadedCrosssensorPair, ...],
    *,
    cache: PredictionCache,
) -> tuple[CalibrationPredictionBundle, ...] | None:
    """Return one coherent verified K5 cache set, or ``None`` when it is incomplete."""

    if type(pairs) is not tuple or not pairs:
        raise TypeError("cache completeness requires a non-empty exact pair tuple")
    if not isinstance(cache, PredictionCache):
        raise TypeError("calibration prediction cache must be a PredictionCache")
    loaded_pairs = tuple(_validate_pair(pair) for pair in pairs)
    pairs_by_id = {pair.pair.sample_id: pair for pair in loaded_pairs}
    if len(pairs_by_id) != len(loaded_pairs):
        raise ValueError("calibration cache completeness requires unique sample IDs")
    if cache.root.is_symlink() or not cache.root.is_dir():
        raise ValueError("calibration prediction cache directory is unsafe")

    groups: dict[bytes, dict[tuple[str, int], CachedCalibrationPrediction]] = {}
    for metadata_path in sorted(cache.root.glob("*.json")):
        candidate = _cached_candidate(
            metadata_path, pairs_by_id=pairs_by_id, cache=cache
        )
        if candidate is None:
            continue
        projection = dict(candidate.identity.model_provenance)
        projection.pop("seed")
        group = groups.setdefault(canonical_json(projection), {})
        slot = (candidate.identity.sample_id, candidate.seed)
        if slot in group:
            raise ValueError("calibration cache contains a duplicate K5 slot")
        group[slot] = candidate

    required_slots = {
        (pair.pair.sample_id, seed) for pair in loaded_pairs for seed in SEEDS
    }
    complete = tuple(group for group in groups.values() if set(group) == required_slots)
    if not complete:
        return None
    if len(complete) != 1:
        raise ValueError("calibration cache contains multiple complete model identities")
    selected = complete[0]
    return tuple(
        CalibrationPredictionBundle(
            sample_id=pair.pair.sample_id,
            items=tuple(selected[(pair.pair.sample_id, seed)] for seed in SEEDS),
        )
        for pair in loaded_pairs
    )
