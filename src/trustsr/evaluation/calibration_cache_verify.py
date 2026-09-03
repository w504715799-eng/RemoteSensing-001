"""Independent structural verifier for Phase 2B3-B calibration cache audits."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from types import MappingProxyType

from trustsr.artifacts.predictions import PredictionIdentity
from trustsr.artifacts.scores import ScoreIdentity
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
)
from trustsr.evaluation.calibration_predictions import (
    A2_RESULT_SHA256,
    EXPERIMENT_SCHEMA,
    MODEL_NAME,
    PUBLICATION_COMMIT,
    SEEDS,
)
from trustsr.evaluation.phase2b3b_evidence import (
    INPUT_AUDIT_SHA256,
    PRODUCER_REVISION,
)
from trustsr.jsonio import canonical_json

_AUDIT_SCHEMA = "trustsr.phase2b3b-calibration-cache-audit.v1"
_VERIFICATION_SCHEMA = "trustsr.phase2b3b-calibration-cache-verification.v1"
_SOURCE = f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}"
_TOP_KEYS = {"schema", "sample_count", "prediction_count", "score_count", "samples"}
_SAMPLE_KEYS = {"sample_id", "predictions", "score", "risk"}
_PREDICTION_KEYS = {
    "model_name",
    "seed",
    "cache_key",
    "identity",
    "prediction_sha256",
}
_SCORE_KEYS = {"name", "cache_key", "identity", "score_sha256"}
_RISK_KEYS = {"name", "window", "risk_sha256"}
_PREDICTION_IDENTITY_KEYS = {"model_provenance", "source", "sample_id", "lr"}
_LR_KEYS = {"shape", "dtype", "sha256"}
_SCORE_IDENTITY_KEYS = {
    "score_name",
    "score_schema_version",
    "sample_id",
    "input_sha256s",
    "operator_parameters",
}


def _mapping(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} keys are invalid")
    return value


def _list(value: object, *, length: int, label: str) -> list[object]:
    if type(value) is not list or len(value) != length:
        raise ValueError(f"{label} must be an exact {length}-item JSON array")
    return value


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _exact(actual: object, expected: object) -> bool:
    return type(actual) is type(expected) and actual == expected


def _rebuild_prediction_identity(
    raw: object, *, sample_id: str, seed: int
) -> PredictionIdentity:
    value = _mapping(raw, _PREDICTION_IDENTITY_KEYS, "prediction identity")
    provenance = value["model_provenance"]
    if type(provenance) is not dict:
        raise ValueError("prediction model provenance must be a JSON object")
    required_provenance = {
        "name": MODEL_NAME,
        "scale": 4,
        "seed": seed,
        "experiment_schema": EXPERIMENT_SCHEMA,
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": INPUT_AUDIT_SHA256,
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "phase2b3a_publication_commit": PUBLICATION_COMMIT,
        "phase2b3a_a2_result_sha256": A2_RESULT_SHA256,
    }
    if any(
        key not in provenance or not _exact(provenance[key], expected)
        for key, expected in required_provenance.items()
    ):
        raise ValueError("prediction provenance has the wrong fixed upstream identity")
    lr = _mapping(value["lr"], _LR_KEYS, "prediction LR identity")
    shape = _list(lr["shape"], length=3, label="prediction LR shape")
    if (
        value["source"] != _SOURCE
        or value["sample_id"] != sample_id
        or lr["dtype"] != "torch.float32"
    ):
        raise ValueError("prediction source, sample, or dtype identity is invalid")
    try:
        identity = PredictionIdentity(
            model_provenance=provenance,
            source=value["source"],
            sample_id=value["sample_id"],
            lr_shape=tuple(shape),
            lr_dtype=lr["dtype"],
            lr_sha256=lr["sha256"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("prediction identity is invalid") from exc
    if identity.as_dict() != value:
        raise ValueError("prediction identity is not canonical")
    return identity


def _verify_prediction(
    raw: object, *, sample_id: str, seed: int
) -> tuple[PredictionIdentity, str, dict[str, object]]:
    value = _mapping(raw, _PREDICTION_KEYS, "prediction entry")
    if value["model_name"] != MODEL_NAME or not _exact(value["seed"], seed):
        raise ValueError("prediction model or fixed seed order is invalid")
    identity = _rebuild_prediction_identity(value["identity"], sample_id=sample_id, seed=seed)
    cache_key = _digest(value["cache_key"], "prediction cache key")
    if cache_key != identity.key:
        raise ValueError("prediction cache key does not match its rebuilt identity")
    prediction_sha256 = _digest(value["prediction_sha256"], "prediction tensor digest")
    receipt = {
        "cache_key": cache_key,
        "identity": identity.as_dict(),
        "prediction_sha256": prediction_sha256,
    }
    return identity, prediction_sha256, receipt


def _expected_score_parameters(lr_sha256: str) -> dict[str, str | int]:
    return {
        "algorithm": "ensemble_variance_score",
        "band_reduction": "mean",
        "correction": 0,
        "seed_first": SEEDS[0],
        "seed_last": SEEDS[-1],
        "seed_count": len(SEEDS),
        "lr_sha256": lr_sha256,
        "source": _SOURCE,
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": INPUT_AUDIT_SHA256,
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "crop_policy": CROP_POLICY,
        "phase2b3a_publication_commit": PUBLICATION_COMMIT,
        "phase2b3a_a2_result_sha256": A2_RESULT_SHA256,
        "phase2b3a_producer_revision": PRODUCER_REVISION,
    }


def _verify_score(
    raw: object,
    *,
    sample_id: str,
    prediction_sha256s: tuple[str, ...],
    lr_sha256: str,
) -> dict[str, object]:
    value = _mapping(raw, _SCORE_KEYS, "score entry")
    if value["name"] != "ldsr_variance_k5":
        raise ValueError("score entry is not the fixed LDSR K5 score")
    raw_identity = _mapping(value["identity"], _SCORE_IDENTITY_KEYS, "score identity")
    raw_inputs = _list(
        raw_identity["input_sha256s"], length=len(SEEDS), label="score input digests"
    )
    if tuple(raw_inputs) != prediction_sha256s:
        raise ValueError("score input digests do not match the five prediction tensors")
    parameters = raw_identity["operator_parameters"]
    expected_parameters = _expected_score_parameters(lr_sha256)
    if (
        type(parameters) is not dict
        or set(parameters) != set(expected_parameters)
        or any(
            not _exact(parameters[key], expected)
            for key, expected in expected_parameters.items()
        )
    ):
        raise ValueError("score identity has the wrong fixed operator or upstream identity")
    if (
        raw_identity["score_name"] != "ldsr_variance_k5"
        or not _exact(raw_identity["score_schema_version"], 1)
        or raw_identity["sample_id"] != sample_id
    ):
        raise ValueError("score name, schema, or sample identity is invalid")
    try:
        identity = ScoreIdentity(
            score_name=raw_identity["score_name"],
            score_schema_version=raw_identity["score_schema_version"],
            sample_id=raw_identity["sample_id"],
            input_sha256s=tuple(raw_inputs),
            operator_parameters=parameters,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("score identity is invalid") from exc
    if identity.as_dict() != raw_identity:
        raise ValueError("score identity is not canonical")
    cache_key = _digest(value["cache_key"], "score cache key")
    if cache_key != identity.key:
        raise ValueError("score cache key does not match its rebuilt identity")
    score_sha256 = _digest(value["score_sha256"], "score tensor digest")
    return {
        "cache_key": cache_key,
        "identity": identity.as_dict(),
        "score_sha256": score_sha256,
    }


def _verify_risk(raw: object, *, sample_id: str) -> dict[str, object]:
    value = _mapping(raw, _RISK_KEYS, "risk entry")
    if value["name"] != "local_l1_risk" or not _exact(value["window"], 9):
        raise ValueError("risk entry must use the fixed local L1 window 9 configuration")
    return {
        "sample_id": sample_id,
        "name": "local_l1_risk",
        "window": 9,
        "risk_sha256": _digest(value["risk_sha256"], "risk tensor digest"),
    }


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def verify_calibration_cache_audit(audit: object) -> Mapping[str, object]:
    """Validate one parsed audit without calling its builder or reading cache assets."""

    value = _mapping(audit, _TOP_KEYS, "calibration cache audit")
    try:
        audit_payload = canonical_json(value)
    except ValueError as exc:
        raise ValueError("calibration cache audit is not canonical JSON data") from exc
    if (
        value["schema"] != _AUDIT_SCHEMA
        or not _exact(value["sample_count"], 120)
        or not _exact(value["prediction_count"], 600)
        or not _exact(value["score_count"], 120)
    ):
        raise ValueError("calibration cache audit schema or fixed counts are invalid")
    samples = _list(value["samples"], length=120, label="calibration audit samples")
    sample_ids: list[str] = []
    prediction_receipts: list[dict[str, object]] = []
    score_receipts: list[dict[str, object]] = []
    risk_receipts: list[dict[str, object]] = []
    for raw_sample in samples:
        sample = _mapping(raw_sample, _SAMPLE_KEYS, "calibration audit sample")
        sample_id = sample["sample_id"]
        if type(sample_id) is not str or not sample_id:
            raise ValueError("calibration audit sample_id must be a non-empty string")
        sample_ids.append(sample_id)
        raw_predictions = _list(
            sample["predictions"], length=len(SEEDS), label="sample predictions"
        )
        identities: list[PredictionIdentity] = []
        prediction_sha256s: list[str] = []
        for raw_prediction, seed in zip(raw_predictions, SEEDS, strict=True):
            identity, prediction_sha256, receipt = _verify_prediction(
                raw_prediction, sample_id=sample_id, seed=seed
            )
            identities.append(identity)
            prediction_sha256s.append(prediction_sha256)
            prediction_receipts.append(receipt)
        first_identity = identities[0]
        lr_identity = (
            first_identity.source,
            first_identity.sample_id,
            first_identity.lr_shape,
            first_identity.lr_dtype,
            first_identity.lr_sha256,
        )
        if any(
            (
                identity.source,
                identity.sample_id,
                identity.lr_shape,
                identity.lr_dtype,
                identity.lr_sha256,
            )
            != lr_identity
            for identity in identities[1:]
        ):
            raise ValueError("the fixed K5 predictions do not share one LR identity")
        score_receipts.append(
            _verify_score(
                sample["score"],
                sample_id=sample_id,
                prediction_sha256s=tuple(prediction_sha256s),
                lr_sha256=first_identity.lr_sha256,
            )
        )
        risk_receipts.append(_verify_risk(sample["risk"], sample_id=sample_id))
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("calibration cache audit requires unique sample identities")
    counts = MappingProxyType({"samples": 120, "predictions": 600, "scores": 120})
    digests = MappingProxyType(
        {
            "audit_sha256": hashlib.sha256(audit_payload).hexdigest(),
            "prediction_identities_sha256": _payload_sha256(prediction_receipts),
            "score_identities_sha256": _payload_sha256(score_receipts),
            "risk_receipts_sha256": _payload_sha256(risk_receipts),
        }
    )
    return MappingProxyType(
        {"schema": _VERIFICATION_SCHEMA, "counts": counts, "digests": digests}
    )
