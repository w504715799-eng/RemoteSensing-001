"""Pure composition of the one-time Phase 2B3-C evaluation result."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

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
    PHASE2B3C_EVALUATION_SIZE,
    PHASE2B3C_MINIMUM_COVERAGE,
    PHASE2B3C_RISK_UPPER_BOUND,
    PHASE2B3C_THRESHOLD,
)
from trustsr.evaluation.phase2b3c_result_verify import verify_phase2b3c_result
from trustsr.evaluation.phase2b3c_statistics import (
    CONFIRMED,
    EMPIRICAL_RISK_EXCEEDS_TARGET,
    EMPIRICALLY_MET_BUT_INCONCLUSIVE,
    FAILED,
    FINITE_SAMPLE_UPPER_BOUND_EXCEEDS_TARGET,
    INSUFFICIENT_COVERAGE,
    Phase2B3CStatistics,
)
from trustsr.jsonio import canonical_json

SCHEMA = "trustsr.phase2b3c-evaluation.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_BANDS = ("B04", "B03", "B02", "B08")
_DECISIONS = {CONFIRMED, EMPIRICALLY_MET_BUT_INCONCLUSIVE, FAILED}


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _revision(value: object) -> str:
    if type(value) is not str or _REVISION.fullmatch(value) is None:
        raise ValueError("producer revision must be a lowercase Git revision")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _radiometric_asset(value: object, label: str) -> dict[str, object]:
    keys = {
        "raw_crop_minimum",
        "raw_crop_maximum",
        "clipped_high_count",
        "clipped_high_by_band",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} radiometry keys are invalid")
    minimum = _nonnegative_integer(value["raw_crop_minimum"], f"{label} minimum")
    maximum = _nonnegative_integer(value["raw_crop_maximum"], f"{label} maximum")
    clipped = _nonnegative_integer(value["clipped_high_count"], f"{label} clipped count")
    by_band = value["clipped_high_by_band"]
    if (
        type(by_band) is not list
        or len(by_band) != len(_BANDS)
        or any(type(item) is not int or item < 0 for item in by_band)
        or sum(by_band) != clipped
        or minimum > maximum
        or maximum > RAW_RADIOMETRIC_MAX
    ):
        raise ValueError(f"{label} radiometry is invalid")
    return {
        "raw_crop_minimum": minimum,
        "raw_crop_maximum": maximum,
        "clipped_high_count": clipped,
        "clipped_high_by_band": list(by_band),
    }


def _radiometry(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"lr", "hr"}:
        raise ValueError("radiometry must contain exact LR and HR aggregates")
    return {
        "lr": _radiometric_asset(value["lr"], "LR"),
        "hr": _radiometric_asset(value["hr"], "HR"),
    }


def _statistics(value: object) -> tuple[dict[str, object], str, list[str]]:
    if type(value) is not Phase2B3CStatistics:
        raise TypeError("result requires exact Phase2B3CStatistics")
    if (
        type(value.evaluation_size) is not int
        or value.evaluation_size != PHASE2B3C_EVALUATION_SIZE
        or type(value.trusted_pixels) is not int
        or type(value.total_pixels) is not int
        or value.trusted_pixels < 0
        or value.total_pixels <= 0
        or value.trusted_pixels > value.total_pixels
        or type(value.coverage) is not float
        or not math.isfinite(value.coverage)
        or value.coverage != value.trusted_pixels / value.total_pixels
        or type(value.phase_decision) is not str
        or value.phase_decision not in _DECISIONS
        or type(value.reasons) is not tuple
        or len(value.strata) != 12
    ):
        raise ValueError("Phase 2B3-C statistics are internally inconsistent")
    if value.coverage < PHASE2B3C_MINIMUM_COVERAGE or value.mean_loss > PHASE2B3C_ALPHA:
        expected_decision = FAILED
        expected_reasons = tuple(
            reason
            for condition, reason in (
                (value.coverage < PHASE2B3C_MINIMUM_COVERAGE, INSUFFICIENT_COVERAGE),
                (value.mean_loss > PHASE2B3C_ALPHA, EMPIRICAL_RISK_EXCEEDS_TARGET),
            )
            if condition
        )
    elif value.risk_ucb <= PHASE2B3C_ALPHA:
        expected_decision = CONFIRMED
        expected_reasons = ()
    else:
        expected_decision = EMPIRICALLY_MET_BUT_INCONCLUSIVE
        expected_reasons = (FINITE_SAMPLE_UPPER_BOUND_EXCEEDS_TARGET,)
    if value.phase_decision != expected_decision or value.reasons != expected_reasons:
        raise ValueError("Phase 2B3-C decision or reasons are inconsistent")
    document = value.as_dict()
    canonical_json(document)
    return document, value.phase_decision, list(value.reasons)


def build_phase2b3c_result(
    *,
    statistics: Phase2B3CStatistics,
    radiometry: Mapping[str, object],
    evaluation_id: str,
    permit_sha256: str,
    ledger_event_sha256: str,
    input_receipt_sha256: str,
    ordered_inputs_sha256: str,
    ordered_sample_ids_sha256: str,
    ordered_membership_sha256: str,
    cache_audit_sha256: str,
    map_evidence_sha256: str,
    producer_revision: str,
) -> dict[str, object]:
    """Compose the host-free result after upstream computation has been verified."""

    statistic_document, decision, reasons = _statistics(statistics)
    result = {
        "schema": SCHEMA,
        "split": "internal_test",
        "verification_scope": "producer_composition_only",
        "acceptance_authorized": False,
        "upstream": {
            "phase2b3b_publication_commit": PHASE2B3B_PUBLICATION_COMMIT,
            "phase2b3b_producer_revision": PHASE2B3B_PRODUCER_REVISION,
            "phase2b3b_evidence_sha256s": dict(PHASE2B3B_FILE_SHA256S),
            "post_manifest_sha256": POST_MANIFEST_SHA256,
            "input_audit_sha256": INPUT_AUDIT_SHA256,
        },
        "producer_revision": _revision(producer_revision),
        "evaluation": {
            "evaluation_id": _digest(evaluation_id, "evaluation ID"),
            "permit_sha256": _digest(permit_sha256, "permit digest"),
            "ledger_event_sha256": _digest(
                ledger_event_sha256, "ledger event digest"
            ),
        },
        "frozen": {
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
                "bands": list(_BANDS),
                "scale": 4,
            },
        },
        "counts": {
            "internal_test": PHASE2B3C_EVALUATION_SIZE,
            "strata": 12,
            "predictions": PHASE2B3C_EVALUATION_SIZE * 5,
            "scores": PHASE2B3C_EVALUATION_SIZE,
        },
        "statistics": statistic_document,
        "radiometry": _radiometry(radiometry),
        "digests": {
            "input_receipt_sha256": _digest(
                input_receipt_sha256, "input receipt digest"
            ),
            "ordered_inputs_sha256": _digest(
                ordered_inputs_sha256, "ordered inputs digest"
            ),
            "ordered_sample_ids_sha256": _digest(
                ordered_sample_ids_sha256, "ordered sample IDs digest"
            ),
            "ordered_membership_sha256": _digest(
                ordered_membership_sha256, "ordered membership digest"
            ),
            "cache_audit_sha256": _digest(
                cache_audit_sha256, "cache audit digest"
            ),
            "map_evidence_sha256": _digest(
                map_evidence_sha256, "map evidence digest"
            ),
        },
        "phase_decision": decision,
        "reasons": reasons,
    }
    canonical_json(result)
    # The independently implemented verifier rechecks every aggregate
    # relationship before producer output crosses this module boundary.
    verify_phase2b3c_result(result)
    return result
