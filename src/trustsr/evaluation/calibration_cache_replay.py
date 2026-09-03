"""Inference-free reconstruction of fixed Phase 2B3-B calibration caches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from trustsr.artifacts.predictions import PredictionCache, PredictionIdentity, tensor_sha256
from trustsr.artifacts.scores import ScoreCache, ScoreIdentity
from trustsr.contracts import SRPair
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
    RadiometricSaturation,
)
from trustsr.evaluation.calibration_cache_verify import verify_calibration_cache_audit
from trustsr.evaluation.calibration_maps import (
    RISK_NAME,
    RISK_WINDOW,
    CachedCalibrationScore,
    CalibrationMaps,
)
from trustsr.evaluation.calibration_predictions import (
    MODEL_NAME,
    SEEDS,
    CachedCalibrationPrediction,
    CalibrationPredictionBundle,
)
from trustsr.risk.local import local_l1_risk

CALIBRATION_SIZE = 120
_DAYS = (-1, 0, 1)
_BINS = (0, 1, 2, 3)
_ROUNDS = tuple(range(1, 11))


@dataclass(frozen=True)
class ReplayInputs:
    """Verified cache-only calibration inputs in their immutable audit order."""

    bundles: tuple[CalibrationPredictionBundle, ...]
    maps: tuple[CalibrationMaps, ...]

    def __post_init__(self) -> None:
        if type(self.bundles) is not tuple or type(self.maps) is not tuple:
            raise TypeError("replay inputs must use exact immutable tuples")
        if len(self.bundles) != CALIBRATION_SIZE or len(self.maps) != CALIBRATION_SIZE:
            raise ValueError("replay inputs require exactly 120 bundles and maps")
        for bundle, maps in zip(self.bundles, self.maps, strict=True):
            if not isinstance(bundle, CalibrationPredictionBundle) or not isinstance(
                maps, CalibrationMaps
            ):
                raise TypeError("replay inputs require calibration bundle and maps values")
            for item in bundle.items:
                item.__post_init__()
            bundle.__post_init__()
            maps.__post_init__()
            if bundle.sample_id != maps.sample_id:
                raise ValueError("replay bundle and maps order differs")
        sample_ids = tuple(bundle.sample_id for bundle in self.bundles)
        if len(set(sample_ids)) != CALIBRATION_SIZE:
            raise ValueError("replay inputs require unique calibration sample identities")


def _validate_pairs(
    pairs: Sequence[LoadedCrosssensorPair],
) -> tuple[LoadedCrosssensorPair, ...]:
    if isinstance(pairs, str | bytes) or not isinstance(pairs, Sequence):
        raise TypeError("replay pairs must be a stable sequence")
    values = tuple(pairs)
    if len(values) != CALIBRATION_SIZE:
        raise ValueError("replay requires exactly 120 calibration pairs")
    strata: dict[tuple[int, int], list[int]] = {
        (day, bin_index): [] for day in _DAYS for bin_index in _BINS
    }
    for pair in values:
        if not isinstance(pair, LoadedCrosssensorPair):
            raise TypeError("replay pairs must be LoadedCrosssensorPair values")
        if not isinstance(pair.pair, SRPair) or not isinstance(
            pair.metadata, CrosssensorPairMetadata
        ):
            raise TypeError("replay pair has forged loaded-pair state")
        pair.pair.validate()
        metadata: CrosssensorPairMetadata = pair.metadata
        if (
            metadata.split != "calibration"
            or metadata.manifest_sha256 != POST_MANIFEST_SHA256
            or metadata.sample_id != pair.pair.sample_id
            or pair.pair.source != f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}"
            or metadata.crop_policy != CROP_POLICY
            or metadata.normalization_policy != PHASE2B3A_NORMALIZATION_POLICY
        ):
            raise ValueError("replay pair has invalid calibration identity or policy")
        if not isinstance(metadata.lr_saturation, RadiometricSaturation) or not isinstance(
            metadata.hr_saturation, RadiometricSaturation
        ):
            raise ValueError("replay pair requires radiometric saturation records")
        try:
            metadata.lr_saturation.__post_init__()
            metadata.hr_saturation.__post_init__()
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("replay pair has invalid radiometric saturation state") from exc
        if (
            type(metadata.days_between) is not int
            or type(metadata.correlation_bin) is not int
            or type(metadata.selection_round) is not int
            or metadata.days_between not in _DAYS
            or metadata.correlation_bin not in _BINS
            or metadata.selection_round not in _ROUNDS
        ):
            raise ValueError("replay pair is outside the frozen calibration strata")
        strata[(metadata.days_between, metadata.correlation_bin)].append(metadata.selection_round)
    if len({pair.pair.sample_id for pair in values}) != CALIBRATION_SIZE or any(
        tuple(sorted(rounds)) != _ROUNDS for rounds in strata.values()
    ):
        raise ValueError("replay pairs have invalid calibration sample or stratum membership")
    return values


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"verified audit {label} is not a mapping")
    return value


def _prediction_identity(raw: object) -> PredictionIdentity:
    value = _mapping(raw, "prediction identity")
    lr = _mapping(value.get("lr"), "prediction LR identity")
    try:
        return PredictionIdentity(
            model_provenance=_mapping(value.get("model_provenance"), "prediction provenance"),
            source=value["source"],
            sample_id=value["sample_id"],
            lr_shape=tuple(lr["shape"]),
            lr_dtype=lr["dtype"],
            lr_sha256=lr["sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("verified audit prediction identity cannot be rebuilt") from exc


def _score_identity(raw: object) -> ScoreIdentity:
    value = _mapping(raw, "score identity")
    try:
        return ScoreIdentity(
            score_name=value["score_name"],
            score_schema_version=value["score_schema_version"],
            sample_id=value["sample_id"],
            input_sha256s=tuple(value["input_sha256s"]),
            operator_parameters=_mapping(value["operator_parameters"], "score parameters"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("verified audit score identity cannot be rebuilt") from exc


def _replay_sample(
    raw_sample: object,
    pair: LoadedCrosssensorPair,
    prediction_cache: PredictionCache,
    score_cache: ScoreCache,
) -> tuple[CalibrationPredictionBundle, CalibrationMaps]:
    sample = _mapping(raw_sample, "sample")
    if sample.get("sample_id") != pair.pair.sample_id:
        raise ValueError("replay audit sample order differs from calibration pairs")
    raw_predictions = sample.get("predictions")
    if type(raw_predictions) is not list or len(raw_predictions) != len(SEEDS):
        raise ValueError("verified audit prediction order is invalid")
    items: list[CachedCalibrationPrediction] = []
    for raw_prediction, seed in zip(raw_predictions, SEEDS, strict=True):
        entry = _mapping(raw_prediction, "prediction entry")
        identity = _prediction_identity(entry.get("identity"))
        if entry.get("seed") != seed or entry.get("model_name") != MODEL_NAME:
            raise ValueError("verified audit prediction model or K5 seed order is invalid")
        if entry.get("cache_key") != identity.key:
            raise ValueError("verified audit prediction cache key is invalid")
        prediction = prediction_cache.get(identity)
        if prediction is None:
            raise RuntimeError("verified calibration prediction cache entry is missing")
        if entry.get("prediction_sha256") != tensor_sha256(prediction):
            raise ValueError("prediction cache tensor digest differs from the audit")
        items.append(
            CachedCalibrationPrediction(
                model_name=MODEL_NAME,
                seed=seed,
                identity=identity,
                prediction_sha256=tensor_sha256(prediction),
                tensor=prediction,
            )
        )
    bundle = CalibrationPredictionBundle(sample_id=pair.pair.sample_id, items=tuple(items))
    score_entry = _mapping(sample.get("score"), "score entry")
    identity = _score_identity(score_entry.get("identity"))
    if score_entry.get("cache_key") != identity.key:
        raise ValueError("verified audit score cache key is invalid")
    score_tensor = score_cache.get(identity)
    if score_tensor is None:
        raise RuntimeError("verified calibration score cache entry is missing")
    if score_entry.get("score_sha256") != tensor_sha256(score_tensor):
        raise ValueError("score cache tensor digest differs from the audit")
    prediction_sha256s = tuple(item.prediction_sha256 for item in bundle.items)
    parameters = identity.operator_parameters
    if (
        identity.sample_id != pair.pair.sample_id
        or identity.input_sha256s != prediction_sha256s
        or parameters.get("lr_sha256") != tensor_sha256(pair.pair.lr)
        or parameters.get("source") != pair.pair.source
    ):
        raise ValueError("score identity differs from replayed pair or K5 bundle")
    score = CachedCalibrationScore(
        name=score_entry.get("name"),
        identity=identity,
        score_sha256=tensor_sha256(score_tensor),
        tensor=score_tensor,
    )
    risk_entry = _mapping(sample.get("risk"), "risk entry")
    if risk_entry.get("name") != RISK_NAME or risk_entry.get("window") != RISK_WINDOW:
        raise ValueError("verified audit risk configuration is invalid")
    risk = local_l1_risk(bundle.for_seed(SEEDS[0]).tensor, pair.pair.hr, window=RISK_WINDOW)
    if risk_entry.get("risk_sha256") != tensor_sha256(risk):
        raise ValueError("replayed local risk tensor digest differs from the audit")
    return bundle, CalibrationMaps(
        sample_id=pair.pair.sample_id,
        score=score,
        score_prediction_sha256s=prediction_sha256s,
        risk_name=RISK_NAME,
        risk_window=RISK_WINDOW,
        risk_sha256=tensor_sha256(risk),
        risk=risk,
    )


def replay_calibration_caches(
    audit: object,
    pairs: Sequence[LoadedCrosssensorPair],
    prediction_cache: PredictionCache,
    score_cache: ScoreCache,
) -> ReplayInputs:
    """Rebuild calibration inputs solely from a verified audit and cache entries."""

    verify_calibration_cache_audit(audit)
    if not isinstance(prediction_cache, PredictionCache) or not isinstance(score_cache, ScoreCache):
        raise TypeError("replay requires PredictionCache and ScoreCache")
    validated_pairs = _validate_pairs(pairs)
    root = _mapping(audit, "root")
    samples = root.get("samples")
    if type(samples) is not list or len(samples) != CALIBRATION_SIZE:
        raise ValueError("verified audit samples are invalid")
    replayed = tuple(
        _replay_sample(sample, pair, prediction_cache, score_cache)
        for sample, pair in zip(samples, validated_pairs, strict=True)
    )
    return ReplayInputs(
        bundles=tuple(bundle for bundle, _ in replayed),
        maps=tuple(maps for _, maps in replayed),
    )
