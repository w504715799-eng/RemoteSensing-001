"""Fixed calibration-only LDSR K5 score and R9 risk maps.

This module deliberately owns no conformal parameter or threshold fitting.  It
accepts only the frozen calibration prediction bundle and exposes one score and
one in-memory risk map for each ROI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from trustsr.artifacts.predictions import PredictionIdentity, tensor_sha256
from trustsr.artifacts.scores import ScoreCache, ScoreIdentity
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
    PUBLICATION_COMMIT,
    SEEDS,
    CachedCalibrationPrediction,
    CalibrationPredictionBundle,
)
from trustsr.evaluation.phase2b3b_evidence import (
    INPUT_AUDIT_SHA256,
    PRODUCER_REVISION,
)
from trustsr.risk.local import ensemble_variance_score, local_l1_risk

SCORE_NAME = "ldsr_variance_k5"
SCORE_SCHEMA_VERSION = 1
RISK_NAME = "local_l1_risk"
RISK_WINDOW = 9
_DAYS = (-1, 0, 1)
_BINS = (0, 1, 2, 3)
_ROUNDS = tuple(range(1, 11))


def _validate_map_tensor(tensor: object, *, label: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if (
        tensor.dtype != torch.float64
        or tensor.device.type != "cpu"
        or tensor.ndim != 2
        or any(dimension <= 0 for dimension in tensor.shape)
        or not tensor.is_contiguous()
        or tensor.requires_grad
        or not torch.isfinite(tensor).all()
        or (tensor < 0).any()
    ):
        raise ValueError(f"{label} must be a finite non-negative contiguous CPU float64 map")
    return tensor


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_fixed_score_identity(identity: ScoreIdentity) -> None:
    """Reject a directly-constructed score identity outside the B3-B contract."""

    parameters = dict(identity.operator_parameters)
    required = {
        "algorithm": "ensemble_variance_score",
        "band_reduction": "mean",
        "correction": 0,
        "seed_first": SEEDS[0],
        "seed_last": SEEDS[-1],
        "seed_count": len(SEEDS),
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": INPUT_AUDIT_SHA256,
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "crop_policy": CROP_POLICY,
        "phase2b3a_publication_commit": PUBLICATION_COMMIT,
        "phase2b3a_a2_result_sha256": A2_RESULT_SHA256,
        "phase2b3a_producer_revision": PRODUCER_REVISION,
    }
    if (
        any(parameters.get(key) != value for key, value in required.items())
        or set(parameters) != {*required, "lr_sha256", "source"}
        or not _is_sha256(parameters.get("lr_sha256"))
        or parameters.get("source") != f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}"
        or len(identity.input_sha256s) != len(SEEDS)
    ):
        raise ValueError("cached calibration score identity is outside the fixed B3-B contract")


@dataclass(frozen=True)
class CachedCalibrationScore:
    """One cache-verified fixed LDSR K5 score map."""

    name: str
    identity: ScoreIdentity
    score_sha256: str
    tensor: torch.Tensor = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.name != SCORE_NAME:
            raise ValueError("cached calibration score has the wrong score name")
        if not isinstance(self.identity, ScoreIdentity):
            raise TypeError("cached calibration score requires a ScoreIdentity")
        if (
            self.identity.score_name != SCORE_NAME
            or self.identity.score_schema_version != SCORE_SCHEMA_VERSION
        ):
            raise ValueError("cached calibration score identity is invalid")
        _validate_fixed_score_identity(self.identity)
        _validate_map_tensor(self.tensor, label="cached calibration score")
        if not _is_sha256(self.score_sha256) or self.score_sha256 != tensor_sha256(self.tensor):
            raise ValueError("cached calibration score digest is invalid")


@dataclass(frozen=True)
class CalibrationMaps:
    """The sole permitted score/risk maps for one calibration ROI."""

    sample_id: str
    score: CachedCalibrationScore
    score_prediction_sha256s: tuple[str, ...]
    risk_name: str
    risk_window: int
    risk_sha256: str
    risk: torch.Tensor = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.sample_id) is not str or not self.sample_id:
            raise TypeError("calibration maps sample_id must be a non-empty string")
        if not isinstance(self.score, CachedCalibrationScore):
            raise TypeError("calibration maps requires a CachedCalibrationScore")
        if self.score.identity.sample_id != self.sample_id:
            raise ValueError("calibration score identity does not match the sample")
        if (
            type(self.score_prediction_sha256s) is not tuple
            or len(self.score_prediction_sha256s) != len(SEEDS)
        ):
            raise ValueError("calibration maps requires the ordered fixed K5 prediction digests")
        if not all(_is_sha256(digest) for digest in self.score_prediction_sha256s):
            raise ValueError("calibration prediction digests are invalid")
        if self.score.identity.input_sha256s != self.score_prediction_sha256s:
            raise ValueError("calibration score identity prediction digests are invalid")
        if self.risk_name != RISK_NAME or self.risk_window != RISK_WINDOW:
            raise ValueError("calibration maps has the wrong fixed risk configuration")
        _validate_map_tensor(self.risk, label="calibration risk")
        if tuple(self.risk.shape) != tuple(self.score.tensor.shape):
            raise ValueError("calibration score and risk shapes differ")
        if not _is_sha256(self.risk_sha256) or self.risk_sha256 != tensor_sha256(self.risk):
            raise ValueError("calibration risk digest is invalid")


def _validate_pair(pair: LoadedCrosssensorPair) -> LoadedCrosssensorPair:
    if not isinstance(pair, LoadedCrosssensorPair):
        raise TypeError("calibration maps require a LoadedCrosssensorPair")
    pair.pair.validate()
    metadata: CrosssensorPairMetadata = pair.metadata
    if type(metadata.split) is not str or metadata.split != "calibration":
        raise ValueError("calibration maps require the calibration split")
    if metadata.manifest_sha256 != POST_MANIFEST_SHA256:
        raise ValueError("calibration pair has the wrong manifest")
    if metadata.sample_id != pair.pair.sample_id:
        raise ValueError("calibration pair and metadata identities differ")
    if pair.pair.source != f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}":
        raise ValueError("calibration pair has the wrong source identity")
    if (
        metadata.crop_policy != CROP_POLICY
        or metadata.normalization_policy != PHASE2B3A_NORMALIZATION_POLICY
    ):
        raise ValueError("calibration pair has the wrong crop or normalization policy")
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
    return pair


def _validate_bundle(
    pair: LoadedCrosssensorPair, bundle: CalibrationPredictionBundle
) -> CalibrationPredictionBundle:
    if not isinstance(bundle, CalibrationPredictionBundle):
        raise TypeError("calibration maps require a CalibrationPredictionBundle")
    if type(bundle.sample_id) is not str or bundle.sample_id != pair.pair.sample_id:
        raise ValueError("calibration prediction bundle sample does not match the pair")
    if type(bundle.items) is not tuple or tuple(item.seed for item in bundle.items) != SEEDS:
        raise ValueError("calibration prediction bundle must contain the ordered fixed K5 seeds")
    for seed, item in zip(SEEDS, bundle.items, strict=True):
        if not isinstance(item, CachedCalibrationPrediction):
            raise TypeError("calibration prediction bundle item is invalid")
        # Re-run the immutable public value contract to fail closed even if an
        # object was forged with object.__new__.
        item.__post_init__()
        identity: PredictionIdentity = item.identity
        if (
            item.seed != seed
            or identity.sample_id != pair.pair.sample_id
            or identity.source != pair.pair.source
            or identity.lr_shape != tuple(pair.pair.lr.shape)
            or identity.lr_dtype != str(pair.pair.lr.dtype)
            or identity.lr_sha256 != tensor_sha256(pair.pair.lr)
        ):
            raise ValueError("calibration prediction bundle input identity differs from the pair")
    return bundle


def _score_identity(
    pair: LoadedCrosssensorPair, bundle: CalibrationPredictionBundle
) -> ScoreIdentity:
    prediction_sha256s = tuple(item.prediction_sha256 for item in bundle.items)
    return ScoreIdentity(
        score_name=SCORE_NAME,
        score_schema_version=SCORE_SCHEMA_VERSION,
        sample_id=pair.pair.sample_id,
        input_sha256s=prediction_sha256s,
        operator_parameters={
            "algorithm": "ensemble_variance_score",
            "band_reduction": "mean",
            "correction": 0,
            "seed_first": SEEDS[0],
            "seed_last": SEEDS[-1],
            "seed_count": len(SEEDS),
            "lr_sha256": tensor_sha256(pair.pair.lr),
            "source": pair.pair.source,
            "post_manifest_sha256": POST_MANIFEST_SHA256,
            "input_audit_sha256": INPUT_AUDIT_SHA256,
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "crop_policy": CROP_POLICY,
            "phase2b3a_publication_commit": PUBLICATION_COMMIT,
            "phase2b3a_a2_result_sha256": A2_RESULT_SHA256,
            "phase2b3a_producer_revision": PRODUCER_REVISION,
        },
    )


def _load_or_compute_score(
    identity: ScoreIdentity, bundle: CalibrationPredictionBundle, score_cache: ScoreCache
) -> CachedCalibrationScore:
    score = score_cache.get(identity)
    if score is None:
        samples = torch.stack([item.tensor for item in bundle.items], dim=0)
        produced = ensemble_variance_score(samples)
        produced_sha256 = tensor_sha256(produced)
        score_cache.put(identity, produced)
        score = score_cache.get(identity)
        if score is None or tensor_sha256(score) != produced_sha256:
            raise RuntimeError("calibration score differs after cache commit")
    return CachedCalibrationScore(SCORE_NAME, identity, tensor_sha256(score), score)


def load_or_compute_calibration_maps(
    pair: LoadedCrosssensorPair,
    bundle: CalibrationPredictionBundle,
    score_cache: ScoreCache,
) -> CalibrationMaps:
    """Load or compute the only permitted calibration K5 score and R9 risk.

    The score is cache-backed.  The risk is deliberately computed in memory for
    this invocation and is neither cached nor serialized as a calibration unit.
    """

    validated_pair = _validate_pair(pair)
    validated_bundle = _validate_bundle(validated_pair, bundle)
    if not isinstance(score_cache, ScoreCache):
        raise TypeError("calibration maps require a ScoreCache")
    identity = _score_identity(validated_pair, validated_bundle)
    score = _load_or_compute_score(identity, validated_bundle, score_cache)
    central = validated_bundle.for_seed(SEEDS[0])
    risk = local_l1_risk(central.tensor, validated_pair.pair.hr, window=RISK_WINDOW)
    return CalibrationMaps(
        sample_id=validated_pair.pair.sample_id,
        score=score,
        score_prediction_sha256s=identity.input_sha256s,
        risk_name=RISK_NAME,
        risk_window=RISK_WINDOW,
        risk_sha256=tensor_sha256(risk),
        risk=risk,
    )
