"""Independent metadata-only verification of Phase 2B3-C results."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    RAW_RADIOMETRIC_MAX,
)
from trustsr.evaluation.phase2b3c_evidence import (
    INPUT_AUDIT_SHA256,
    PHASE2B3B_FILE_SHA256S,
    PHASE2B3B_PUBLICATION_COMMIT,
)
from trustsr.evaluation.phase2b3c_evidence import (
    PRODUCER_REVISION as PHASE2B3B_PRODUCER_REVISION,
)
from trustsr.evaluation.phase2b3c_policy import (
    PHASE2B3C_ALPHA,
    PHASE2B3C_BISECTION_ITERATIONS,
    PHASE2B3C_DELTA,
    PHASE2B3C_EVALUATION_SIZE,
    PHASE2B3C_GRID_SIZE,
    PHASE2B3C_MINIMUM_COVERAGE,
    PHASE2B3C_RISK_UPPER_BOUND,
    PHASE2B3C_THRESHOLD,
)
from trustsr.jsonio import canonical_json

SCHEMA = "trustsr.phase2b3c-evaluation.v1"
VERIFICATION_SCHEMA = "trustsr.phase2b3c-evaluation-metadata-verification.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_ROOT_KEYS = {
    "schema",
    "split",
    "verification_scope",
    "acceptance_authorized",
    "upstream",
    "producer_revision",
    "evaluation",
    "frozen",
    "counts",
    "statistics",
    "radiometry",
    "digests",
    "phase_decision",
    "reasons",
}
_STATISTIC_KEYS = {
    "target",
    "method",
    "evaluation_size",
    "trusted_pixels",
    "total_pixels",
    "coverage",
    "mean_loss",
    "empirical_risk_delta",
    "empirical_risk_violation",
    "risk_ucb",
    "finite_sample_margin",
    "finite_sample_delta",
    "hoeffding_ucb",
    "phase_decision",
    "reasons",
    "strata",
}
_STRATUM_KEYS = {
    "days_between",
    "correlation_bin",
    "roi_count",
    "trusted_pixels",
    "total_pixels",
    "coverage",
    "mean_loss",
    "maximum_loss",
}


@dataclass(frozen=True)
class VerifiedPhase2B3CResult:
    """Non-authorizing identity returned after strict metadata validation."""

    result_sha256: str
    phase_decision: str
    verification_scope: str
    cache_computation_verified: bool
    acceptance_authorized: bool
    evaluation_id: str
    permit_sha256: str
    ledger_event_sha256: str
    input_receipt_sha256: str
    ordered_inputs_sha256: str
    cache_audit_sha256: str
    map_evidence_sha256: str
    producer_revision: str

    def __post_init__(self) -> None:
        if (
            self.verification_scope != "metadata_consistency_only"
            or self.cache_computation_verified is not False
            or self.acceptance_authorized is not False
            or self.phase_decision
            not in {"confirmed", "empirically_met_but_inconclusive", "failed"}
        ):
            raise ValueError("result verification scope or decision is invalid")
        for value in (
            self.result_sha256,
            self.evaluation_id,
            self.permit_sha256,
            self.ledger_event_sha256,
            self.input_receipt_sha256,
            self.ordered_inputs_sha256,
            self.cache_audit_sha256,
            self.map_evidence_sha256,
        ):
            _digest(value, "verified result digest")
        _revision(self.producer_revision, "verified producer revision")


def _object(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} keys are invalid")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _revision(value: object, label: str) -> str:
    if type(value) is not str or _REVISION.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase Git revision")
    return value


def _number(value: object, label: str, *, lower: float, upper: float) -> float:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or not lower <= value <= upper
    ):
        raise ValueError(f"{label} is outside its finite range")
    return value


def _canonical_document(value: object) -> tuple[dict[str, object], bytes]:
    if type(value) is bytes:
        payload = value
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("result must be canonical UTF-8 JSON") from exc
        if type(parsed) is not dict or canonical_json(parsed) != payload:
            raise ValueError("result must be canonical JSON")
        return parsed, payload
    if type(value) is not dict:
        raise TypeError("result must be canonical bytes or an exact JSON object")
    try:
        payload = canonical_json(value)
        parsed = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("result must contain canonical JSON-native values") from exc
    return parsed, payload


def _validate_fixed(value: dict[str, object]) -> None:
    upstream = _object(
        value["upstream"],
        {
            "phase2b3b_publication_commit",
            "phase2b3b_producer_revision",
            "phase2b3b_evidence_sha256s",
            "post_manifest_sha256",
            "input_audit_sha256",
        },
        "result upstream",
    )
    if upstream != {
        "phase2b3b_publication_commit": PHASE2B3B_PUBLICATION_COMMIT,
        "phase2b3b_producer_revision": PHASE2B3B_PRODUCER_REVISION,
        "phase2b3b_evidence_sha256s": dict(PHASE2B3B_FILE_SHA256S),
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": INPUT_AUDIT_SHA256,
    }:
        raise ValueError("result upstream identity is invalid")
    frozen = _object(value["frozen"], {"threshold", "target", "score", "risk", "input"}, "frozen")
    expected = {
        "threshold": PHASE2B3C_THRESHOLD,
        "target": {
            "alpha": PHASE2B3C_ALPHA,
            "minimum_coverage": PHASE2B3C_MINIMUM_COVERAGE,
            "risk_upper_bound": PHASE2B3C_RISK_UPPER_BOUND,
        },
        "score": {
            "name": "ldsr_variance_k5",
            "operator_parameters": {
                "algorithm": "ensemble_variance_score",
                "band_reduction": "mean",
                "correction": 0,
                "seed_count": 5,
                "seed_first": 3407,
                "seed_last": 3411,
            },
            "seeds": [3407, 3408, 3409, 3410, 3411],
        },
        "risk": {"name": "local_l1_risk", "window": 9, "upper_bound": 1.0},
        "input": {
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "crop_policy": CROP_POLICY,
            "bands": ["B04", "B03", "B02", "B08"],
            "scale": 4,
        },
    }
    if frozen != expected:
        raise ValueError("frozen Phase 2B3-C policy is invalid")
    if value["counts"] != {
        "internal_test": 120,
        "strata": 12,
        "predictions": 600,
        "scores": 120,
    }:
        raise ValueError("result counts are invalid")


def _validate_strata(value: object) -> tuple[int, int, tuple[float, ...]]:
    if type(value) is not list or len(value) != 12:
        raise ValueError("statistics strata must contain exactly 12 entries")
    trusted_total = 0
    pixel_total = 0
    means: list[float] = []
    for index, item in enumerate(value):
        stratum = _object(item, _STRATUM_KEYS, "statistics stratum")
        expected_day = (-1, 0, 1)[index // 4]
        expected_bin = index % 4
        trusted = stratum["trusted_pixels"]
        total = stratum["total_pixels"]
        if (
            stratum["days_between"] != expected_day
            or type(stratum["days_between"]) is not int
            or stratum["correlation_bin"] != expected_bin
            or type(stratum["correlation_bin"]) is not int
            or stratum["roi_count"] != 10
            or type(stratum["roi_count"]) is not int
            or type(trusted) is not int
            or type(total) is not int
            or trusted < 0
            or total <= 0
            or trusted > total
        ):
            raise ValueError("statistics strata identity or counts are invalid")
        coverage = _number(stratum["coverage"], "stratum coverage", lower=0.0, upper=1.0)
        mean = _number(stratum["mean_loss"], "stratum mean loss", lower=0.0, upper=1.0)
        maximum = _number(
            stratum["maximum_loss"], "stratum maximum loss", lower=0.0, upper=1.0
        )
        if coverage != trusted / total or mean > maximum:
            raise ValueError("statistics strata aggregates are inconsistent")
        trusted_total += trusted
        pixel_total += total
        means.append(mean)
    return trusted_total, pixel_total, tuple(means)


def _validate_statistics(value: object) -> tuple[str, list[str]]:
    statistics = _object(value, _STATISTIC_KEYS, "statistics")
    if statistics["target"] != {
        "alpha": PHASE2B3C_ALPHA,
        "minimum_coverage": PHASE2B3C_MINIMUM_COVERAGE,
        "risk_upper_bound": PHASE2B3C_RISK_UPPER_BOUND,
    }:
        raise ValueError("statistics target is invalid")
    if statistics["method"] != {
        "name": "one_sided_diversified_grid_kelly",
        "confidence_error": PHASE2B3C_DELTA,
        "grid_size": PHASE2B3C_GRID_SIZE,
        "bet_fractions": [
            index / (PHASE2B3C_GRID_SIZE + 1)
            for index in range(1, PHASE2B3C_GRID_SIZE + 1)
        ],
        "bisection_iterations": PHASE2B3C_BISECTION_ITERATIONS,
    }:
        raise ValueError("statistics method is invalid")
    trusted, total, stratum_means = _validate_strata(statistics["strata"])
    if (
        statistics["evaluation_size"] != PHASE2B3C_EVALUATION_SIZE
        or type(statistics["evaluation_size"]) is not int
        or statistics["trusted_pixels"] != trusted
        or type(statistics["trusted_pixels"]) is not int
        or statistics["total_pixels"] != total
        or type(statistics["total_pixels"]) is not int
    ):
        raise ValueError("statistics aggregate counts differ from strata")
    coverage = _number(statistics["coverage"], "aggregate coverage", lower=0.0, upper=1.0)
    mean_loss = _number(statistics["mean_loss"], "mean loss", lower=0.0, upper=1.0)
    risk_ucb = _number(statistics["risk_ucb"], "risk UCB", lower=0.0, upper=1.0)
    hoeffding = _number(statistics["hoeffding_ucb"], "Hoeffding UCB", lower=0.0, upper=1.0)
    empirical_delta = statistics["empirical_risk_delta"]
    finite_margin = statistics["finite_sample_margin"]
    finite_delta = statistics["finite_sample_delta"]
    signed_values = (empirical_delta, finite_margin, finite_delta)
    if any(
        type(item) is not float or not math.isfinite(item) for item in signed_values
    ):
        raise ValueError("statistics signed deltas must be finite floats")
    empirical_violation = _number(
        statistics["empirical_risk_violation"],
        "empirical risk violation",
        lower=0.0,
        upper=1.0,
    )
    expected_hoeffding = min(
        1.0,
        mean_loss
        + math.sqrt(math.log(1.0 / PHASE2B3C_DELTA) / (2 * PHASE2B3C_EVALUATION_SIZE)),
    )
    if (
        coverage != trusted / total
        or not math.isclose(
            mean_loss, math.fsum(stratum_means) / 12, rel_tol=0.0, abs_tol=2e-15
        )
        or empirical_delta != mean_loss - PHASE2B3C_ALPHA
        or empirical_violation != max(0.0, empirical_delta)
        or finite_margin != risk_ucb - mean_loss
        or finite_delta != risk_ucb - PHASE2B3C_ALPHA
        or hoeffding != expected_hoeffding
    ):
        raise ValueError("statistics aggregate relationships are invalid")
    if coverage < PHASE2B3C_MINIMUM_COVERAGE or mean_loss > PHASE2B3C_ALPHA:
        decision = "failed"
        reasons = [
            reason
            for condition, reason in (
                (coverage < PHASE2B3C_MINIMUM_COVERAGE, "insufficient_coverage"),
                (mean_loss > PHASE2B3C_ALPHA, "empirical_risk_exceeds_target"),
            )
            if condition
        ]
    elif risk_ucb <= PHASE2B3C_ALPHA:
        decision, reasons = "confirmed", []
    else:
        decision = "empirically_met_but_inconclusive"
        reasons = ["finite_sample_upper_bound_exceeds_target"]
    if statistics["phase_decision"] != decision:
        raise ValueError("statistics phase decision is invalid")
    if statistics["reasons"] != reasons:
        raise ValueError("statistics decision reasons are invalid")
    return decision, reasons


def _validate_radiometry(value: object) -> None:
    radiometry = _object(value, {"lr", "hr"}, "radiometry")
    keys = {
        "raw_crop_minimum",
        "raw_crop_maximum",
        "clipped_high_count",
        "clipped_high_by_band",
    }
    for name in ("lr", "hr"):
        item = _object(radiometry[name], keys, f"{name} radiometry")
        by_band = item["clipped_high_by_band"]
        if (
            any(type(item[key]) is not int for key in keys - {"clipped_high_by_band"})
            or item["raw_crop_minimum"] < 0
            or item["raw_crop_maximum"] < item["raw_crop_minimum"]
            or item["raw_crop_maximum"] > RAW_RADIOMETRIC_MAX
            or item["clipped_high_count"] < 0
            or type(by_band) is not list
            or len(by_band) != 4
            or any(type(count) is not int or count < 0 for count in by_band)
            or sum(by_band) != item["clipped_high_count"]
        ):
            raise ValueError("result radiometry is invalid")


def verify_phase2b3c_result(value: object) -> VerifiedPhase2B3CResult:
    """Validate exact result structure without claiming cache recomputation."""

    result, payload = _canonical_document(value)
    _object(result, _ROOT_KEYS, "result")
    if (
        result["schema"] != SCHEMA
        or result["split"] != "internal_test"
        or result["verification_scope"] != "producer_composition_only"
        or result["acceptance_authorized"] is not False
    ):
        raise ValueError("result identity or non-authorizing scope is invalid")
    _validate_fixed(result)
    producer_revision = _revision(result["producer_revision"], "producer revision")
    evaluation = _object(
        result["evaluation"],
        {"evaluation_id", "permit_sha256", "ledger_event_sha256"},
        "evaluation identity",
    )
    digests = _object(
        result["digests"],
        {
            "input_receipt_sha256",
            "ordered_inputs_sha256",
            "ordered_sample_ids_sha256",
            "ordered_membership_sha256",
            "cache_audit_sha256",
            "map_evidence_sha256",
        },
        "result digests",
    )
    for key, item in (*evaluation.items(), *digests.items()):
        _digest(item, key)
    decision, reasons = _validate_statistics(result["statistics"])
    if result["phase_decision"] != decision:
        raise ValueError("result phase decision differs from statistics")
    if result["reasons"] != reasons:
        raise ValueError("result reason strings differ from statistics")
    _validate_radiometry(result["radiometry"])
    return VerifiedPhase2B3CResult(
        result_sha256=hashlib.sha256(payload).hexdigest(),
        phase_decision=decision,
        verification_scope="metadata_consistency_only",
        cache_computation_verified=False,
        acceptance_authorized=False,
        evaluation_id=evaluation["evaluation_id"],
        permit_sha256=evaluation["permit_sha256"],
        ledger_event_sha256=evaluation["ledger_event_sha256"],
        input_receipt_sha256=digests["input_receipt_sha256"],
        ordered_inputs_sha256=digests["ordered_inputs_sha256"],
        cache_audit_sha256=digests["cache_audit_sha256"],
        map_evidence_sha256=digests["map_evidence_sha256"],
        producer_revision=producer_revision,
    )
