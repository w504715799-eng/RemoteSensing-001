"""Pure final result composition for Phase 2B3-B calibration."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence

from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    RAW_RADIOMETRIC_MAX,
    REFLECTANCE_SCALE,
    RadiometricSaturation,
)
from trustsr.evaluation.calibration_cache_verify import verify_calibration_cache_audit
from trustsr.evaluation.calibration_fit import CalibrationFit
from trustsr.evaluation.calibration_predictions import SEEDS
from trustsr.evaluation.phase2b3b_evidence import (
    INPUT_AUDIT_SHA256,
    PRODUCER_REVISION,
    PUBLICATION_COMMIT,
    PUBLISHED_EVIDENCE_SHA256S,
)
from trustsr.evaluation.phase2b3b_revision import Phase2B3BRevision
from trustsr.jsonio import canonical_json

_REVISION = re.compile(r"[0-9a-f]{40}")
_BANDS = ("B04", "B03", "B02", "B08")
_DAYS = (-1, 0, 1)
_BINS = (0, 1, 2, 3)
_ROUNDS = tuple(range(1, 11))


def _mapping(value: object, keys: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys are invalid")
    return value


def _sequence(value: object, *, length: int, label: str) -> Sequence[object]:
    if (
        isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
        or len(value) != length
    ):
        raise ValueError(f"{label} must contain exactly {length} items")
    return value


def _exact(actual: object, expected: object) -> bool:
    return type(actual) is type(expected) and actual == expected


def _json_object(value: object, label: str) -> dict[str, object]:
    try:
        parsed = json.loads(canonical_json(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be JSON-native") from exc
    if type(parsed) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _validate_preflight(preflight: Mapping[str, object]) -> dict[str, object]:
    value = _mapping(
        preflight,
        {"schema", "upstream", "calibration", "score", "risk", "input"},
        "preflight",
    )
    if value["schema"] != "trustsr.phase2b3b-preflight.v1":
        raise ValueError("preflight schema is invalid")
    upstream = _mapping(
        value["upstream"],
        {
            "publication_commit",
            "producer_revision",
            "post_manifest_sha256",
            "input_audit_sha256",
            "evidence_sha256s",
        },
        "preflight upstream",
    )
    evidence_sha256s = upstream["evidence_sha256s"]
    if (
        not isinstance(evidence_sha256s, Mapping)
        or dict(evidence_sha256s) != dict(PUBLISHED_EVIDENCE_SHA256S)
        or upstream["publication_commit"] != PUBLICATION_COMMIT
        or upstream["producer_revision"] != PRODUCER_REVISION
        or upstream["post_manifest_sha256"] != POST_MANIFEST_SHA256
        or upstream["input_audit_sha256"] != INPUT_AUDIT_SHA256
    ):
        raise ValueError("preflight upstream identity is invalid")
    calibration = _mapping(
        value["calibration"], {"sample_count", "strata"}, "preflight calibration"
    )
    if not _exact(calibration["sample_count"], 120):
        raise ValueError("preflight calibration count is invalid")
    strata = _sequence(calibration["strata"], length=12, label="preflight strata")
    expected_cells = tuple((day, bin_index) for day in _DAYS for bin_index in _BINS)
    for raw, (day, bin_index) in zip(strata, expected_cells, strict=True):
        cell = _mapping(
            raw, {"days_between", "correlation_bin", "sample_count"}, "preflight stratum"
        )
        if not (
            _exact(cell["days_between"], day)
            and _exact(cell["correlation_bin"], bin_index)
            and _exact(cell["sample_count"], 10)
        ):
            raise ValueError("preflight calibration strata are invalid")
    score = _mapping(value["score"], {"name", "operator_parameters", "seeds"}, "score")
    expected_operator = {
        "algorithm": "ensemble_variance_score",
        "band_reduction": "mean",
        "correction": 0,
        "seed_count": 5,
        "seed_first": 3407,
        "seed_last": 3411,
    }
    operator = score["operator_parameters"]
    seeds = _sequence(score["seeds"], length=5, label="preflight seeds")
    if (
        score["name"] != "ldsr_variance_k5"
        or not isinstance(operator, Mapping)
        or set(operator) != set(expected_operator)
        or any(not _exact(operator[key], expected) for key, expected in expected_operator.items())
        or tuple(seeds) != SEEDS
        or any(type(seed) is not int for seed in seeds)
    ):
        raise ValueError("preflight score identity is invalid")
    risk = _mapping(value["risk"], {"name", "window", "upper_bound"}, "risk")
    if (
        risk["name"] != "local_l1_risk"
        or not _exact(risk["window"], 9)
        or not _exact(risk["upper_bound"], 1.0)
    ):
        raise ValueError("preflight risk identity is invalid")
    input_identity = _mapping(
        value["input"],
        {"normalization_policy", "crop_policy", "bands", "scale"},
        "input identity",
    )
    bands = _sequence(input_identity["bands"], length=4, label="input bands")
    if (
        input_identity["normalization_policy"] != PHASE2B3A_NORMALIZATION_POLICY
        or input_identity["crop_policy"] != CROP_POLICY
        or tuple(bands) != _BANDS
        or not _exact(input_identity["scale"], 4)
    ):
        raise ValueError("preflight input identity is invalid")
    return {
        "upstream": {
            "post_manifest_sha256": POST_MANIFEST_SHA256,
            "input_audit_sha256": INPUT_AUDIT_SHA256,
            "phase2b3a_publication_commit": PUBLICATION_COMMIT,
            "phase2b3a_calculation_revision": PRODUCER_REVISION,
            "evidence_sha256s": dict(PUBLISHED_EVIDENCE_SHA256S),
        },
        "score": {
            "name": "ldsr_variance_k5",
            "operator_parameters": dict(expected_operator),
            "seeds": list(SEEDS),
        },
        "risk": {"name": "local_l1_risk", "window": 9, "upper_bound": 1.0},
        "input": {
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "crop_policy": CROP_POLICY,
            "bands": list(_BANDS),
            "scale": 4,
        },
    }


def _saturation(value: object, label: str) -> dict[str, object]:
    data = _mapping(
        value,
        {
            "raw_crop_minimum",
            "raw_crop_maximum",
            "clipped_high_count",
            "clipped_high_by_band",
        },
        label,
    )
    bands = _sequence(data["clipped_high_by_band"], length=4, label=f"{label} bands")
    try:
        validated = RadiometricSaturation(
            raw_crop_minimum=data["raw_crop_minimum"],
            raw_crop_maximum=data["raw_crop_maximum"],
            clipped_high_count=data["clipped_high_count"],
            clipped_high_by_band=tuple(bands),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    return {
        "raw_crop_minimum": validated.raw_crop_minimum,
        "raw_crop_maximum": validated.raw_crop_maximum,
        "clipped_high_count": validated.clipped_high_count,
        "clipped_high_by_band": list(validated.clipped_high_by_band),
    }


def _aggregate(samples: Sequence[dict[str, object]], asset: str) -> dict[str, object]:
    values = [sample["radiometric_saturation"][asset] for sample in samples]
    return {
        "raw_crop_minimum": min(value["raw_crop_minimum"] for value in values),
        "raw_crop_maximum": max(value["raw_crop_maximum"] for value in values),
        "clipped_high_count": sum(value["clipped_high_count"] for value in values),
        "clipped_high_by_band": [
            sum(value["clipped_high_by_band"][index] for value in values)
            for index in range(4)
        ],
    }


def _validate_radiometry(radiometry: object) -> tuple[dict[str, object], list[dict[str, object]]]:
    value = _mapping(
        radiometry,
        {"schema", "policy", "sample_count", "affected_sample_count", "lr", "hr", "samples"},
        "radiometry",
    )
    policy = _mapping(
        value["policy"],
        {
            "normalization_policy",
            "raw_radiometric_max",
            "saturation_threshold",
            "saturation_operation",
            "saturation_scope",
            "reflectance_divisor",
            "crop_policy",
            "bands",
        },
        "radiometry policy",
    )
    bands = _sequence(policy["bands"], length=4, label="radiometry bands")
    expected_policy = {
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "raw_radiometric_max": RAW_RADIOMETRIC_MAX,
        "saturation_threshold": int(REFLECTANCE_SCALE),
        "saturation_operation": "minimum(raw,10000)",
        "saturation_scope": "aligned_crop_only",
        "reflectance_divisor": REFLECTANCE_SCALE,
        "crop_policy": CROP_POLICY,
    }
    if (
        value["schema"] != "trustsr.phase2b3b-calibration-radiometry.v1"
        or not _exact(value["sample_count"], 120)
        or tuple(bands) != _BANDS
        or any(not _exact(policy[key], expected) for key, expected in expected_policy.items())
    ):
        raise ValueError("radiometry schema, count, or policy is invalid")
    raw_samples = _sequence(value["samples"], length=120, label="radiometry samples")
    samples: list[dict[str, object]] = []
    cells: dict[tuple[int, int], list[int]] = {
        (day, bin_index): [] for day in _DAYS for bin_index in _BINS
    }
    for raw_sample in raw_samples:
        sample = _mapping(
            raw_sample,
            {
                "sample_id",
                "days_between",
                "correlation_bin",
                "selection_round",
                "radiometric_saturation",
            },
            "radiometry sample",
        )
        sample_id = sample["sample_id"]
        day = sample["days_between"]
        bin_index = sample["correlation_bin"]
        selection_round = sample["selection_round"]
        if (
            type(sample_id) is not str
            or not sample_id
            or type(day) is not int
            or type(bin_index) is not int
            or type(selection_round) is not int
            or (day, bin_index) not in cells
        ):
            raise ValueError("radiometry sample identity or stratum is invalid")
        saturation = _mapping(
            sample["radiometric_saturation"], {"lr", "hr"}, "sample saturation"
        )
        normalized = {
            "sample_id": sample_id,
            "days_between": day,
            "correlation_bin": bin_index,
            "selection_round": selection_round,
            "radiometric_saturation": {
                "lr": _saturation(saturation["lr"], "LR saturation"),
                "hr": _saturation(saturation["hr"], "HR saturation"),
            },
        }
        samples.append(normalized)
        cells[(day, bin_index)].append(selection_round)
    sample_ids = [sample["sample_id"] for sample in samples]
    if len(set(sample_ids)) != 120 or any(
        tuple(sorted(rounds)) != _ROUNDS for rounds in cells.values()
    ):
        raise ValueError("radiometry samples must be unique with complete fixed strata")
    lr = _aggregate(samples, "lr")
    hr = _aggregate(samples, "hr")
    supplied_lr = _saturation(value["lr"], "LR aggregate")
    supplied_hr = _saturation(value["hr"], "HR aggregate")
    affected = sum(
        int(
            sample["radiometric_saturation"]["lr"]["clipped_high_count"] > 0
            or sample["radiometric_saturation"]["hr"]["clipped_high_count"] > 0
        )
        for sample in samples
    )
    if (
        supplied_lr != lr
        or supplied_hr != hr
        or not _exact(value["affected_sample_count"], affected)
    ):
        raise ValueError("radiometry aggregates are inconsistent")
    return (
        {
            "policy": {**expected_policy, "bands": list(_BANDS)},
            "affected_sample_count": affected,
            "lr": lr,
            "hr": hr,
        },
        samples,
    )


def _validate_fit(fit: CalibrationFit) -> CalibrationFit:
    if type(fit) is not CalibrationFit:
        raise TypeError("result composer requires an exact CalibrationFit")
    try:
        CalibrationFit.__post_init__(fit)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("calibration fit public contract is invalid") from exc
    return fit


def _validate_revision(revision: Phase2B3BRevision) -> str:
    if type(revision) is not Phase2B3BRevision:
        raise TypeError("result composer requires an exact Phase2B3BRevision")
    if (
        type(revision.branch) is not str
        or not revision.branch
        or type(revision.head_revision) is not str
        or _REVISION.fullmatch(revision.head_revision) is None
        or revision.calculation_revision != PRODUCER_REVISION
        or revision.evidence_publication != PUBLICATION_COMMIT
    ):
        raise ValueError("Phase 2B3-B revision identity is invalid")
    return revision.head_revision


def _audit_map_evidence_sha256(samples: Sequence[object]) -> str:
    evidence = [
        {
            "sample_id": sample["sample_id"],
            "score_sha256": sample["score"]["score_sha256"],
            "risk_sha256": sample["risk"]["risk_sha256"],
        }
        for sample in samples
    ]
    return hashlib.sha256(canonical_json(evidence)).hexdigest()


def build_phase2b3b_result(
    preflight: Mapping[str, object],
    fit: CalibrationFit,
    cache_audit: Mapping[str, object],
    radiometry: Mapping[str, object],
    revision: Phase2B3BRevision,
) -> dict[str, object]:
    """Compose one canonical result after independently verifying every input layer."""

    normalized_audit = _json_object(cache_audit, "cache audit")
    cache_verification = verify_calibration_cache_audit(normalized_audit)
    validated_fit = _validate_fit(fit)
    frozen = _validate_preflight(preflight)
    normalized_radiometry = _json_object(radiometry, "radiometry")
    radiometry_summary, radiometry_samples = _validate_radiometry(normalized_radiometry)
    producer_revision = _validate_revision(revision)

    audit_samples = normalized_audit["samples"]
    audit_sample_ids = tuple(sample["sample_id"] for sample in audit_samples)
    radiometry_sample_ids = tuple(sample["sample_id"] for sample in radiometry_samples)
    if not (
        validated_fit.sample_ids == audit_sample_ids == radiometry_sample_ids
        and validated_fit.calibration_size == normalized_audit["sample_count"] == 120
    ):
        raise ValueError("fit, cache audit, and radiometry ordered samples differ")
    audit_map_evidence_sha256 = _audit_map_evidence_sha256(audit_samples)
    if validated_fit.map_evidence_sha256 != audit_map_evidence_sha256:
        raise ValueError("calibration fit map evidence differs from verified cache audit")

    samples = []
    for audit_sample, radiometry_sample in zip(
        audit_samples, radiometry_samples, strict=True
    ):
        samples.append(
            {
                "sample_id": audit_sample["sample_id"],
                "predictions": [
                    {
                        "seed": prediction["seed"],
                        "cache_key": prediction["cache_key"],
                        "prediction_sha256": prediction["prediction_sha256"],
                    }
                    for prediction in audit_sample["predictions"]
                ],
                "score": {
                    "cache_key": audit_sample["score"]["cache_key"],
                    "score_sha256": audit_sample["score"]["score_sha256"],
                },
                "risk": {"risk_sha256": audit_sample["risk"]["risk_sha256"]},
                "radiometric_saturation": radiometry_sample["radiometric_saturation"],
            }
        )
    cache_audit_sha256 = cache_verification["digests"]["audit_sha256"]
    result = {
        "schema": "trustsr.phase2b3b-calibration.v1",
        "upstream": frozen["upstream"],
        "producer_revision": producer_revision,
        "frozen": {
            "score": frozen["score"],
            "risk": frozen["risk"],
            "input": frozen["input"],
        },
        "target": {
            "alpha": validated_fit.alpha,
            "minimum_coverage": validated_fit.minimum_coverage,
        },
        "threshold": validated_fit.threshold,
        "all_abstain": validated_fit.all_abstain,
        "risk_bound": validated_fit.risk_bound,
        "counts": {
            "calibration": 120,
            "predictions": 600,
            "scores": 120,
            "trusted_pixels": validated_fit.trusted_pixels,
            "total_pixels": validated_fit.total_pixels,
        },
        "coverage": validated_fit.coverage,
        "radiometry": radiometry_summary,
        "samples": samples,
        "cache_audit_sha256": cache_audit_sha256,
        "map_evidence_sha256": validated_fit.map_evidence_sha256,
        "phase_decision": validated_fit.phase_decision,
    }
    canonical_json(result)
    return result
