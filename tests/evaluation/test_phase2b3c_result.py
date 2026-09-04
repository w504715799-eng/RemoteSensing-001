"""Strict Phase 2B3-C evaluation-result composition tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from trustsr.evaluation.phase2b3c_result import build_phase2b3c_result
from trustsr.evaluation.phase2b3c_statistics import (
    CONFIRMED,
    EMPIRICALLY_MET_BUT_INCONCLUSIVE,
    FAILED,
    ROIEvaluation,
    build_phase2b3c_statistics,
)
from trustsr.jsonio import canonical_json


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _statistics(*, loss: float, trusted_pixels: int = 20):
    rois = tuple(
        ROIEvaluation(
            sample_id=f"test-{index:03d}",
            days_between=(-1, 0, 1)[(index % 12) // 4],
            correlation_bin=index % 4,
            selection_round=index // 12 + 1,
            trusted_pixels=trusted_pixels,
            total_pixels=100,
            coverage=trusted_pixels / 100,
            loss=loss,
        )
        for index in range(120)
    )
    return build_phase2b3c_statistics(rois)


def _radiometry() -> dict[str, object]:
    return {
        "lr": {
            "raw_crop_minimum": 0,
            "raw_crop_maximum": 10000,
            "clipped_high_count": 4,
            "clipped_high_by_band": [1, 1, 1, 1],
        },
        "hr": {
            "raw_crop_minimum": 1,
            "raw_crop_maximum": 12000,
            "clipped_high_count": 8,
            "clipped_high_by_band": [2, 2, 2, 2],
        },
    }


def _result(statistics=None) -> dict[str, object]:
    return build_phase2b3c_result(
        statistics=_statistics(loss=0.0) if statistics is None else statistics,
        radiometry=_radiometry(),
        evaluation_id=_sha("evaluation"),
        permit_sha256=_sha("permit"),
        ledger_event_sha256=_sha("ledger"),
        input_receipt_sha256=_sha("input receipt"),
        ordered_inputs_sha256=_sha("inputs"),
        ordered_sample_ids_sha256=_sha("sample ids"),
        ordered_membership_sha256=_sha("membership"),
        cache_audit_sha256=_sha("cache audit"),
        map_evidence_sha256=_sha("maps"),
        producer_revision="1" * 40,
    )


def test_result_contains_frozen_identity_aggregates_and_12_diagnostics() -> None:
    result = _result()

    assert result["schema"] == "trustsr.phase2b3c-evaluation.v1"
    assert result["split"] == "internal_test"
    assert result["phase_decision"] == CONFIRMED
    assert result["reasons"] == []
    assert result["counts"] == {
        "internal_test": 120,
        "strata": 12,
        "predictions": 600,
        "scores": 120,
    }
    assert len(result["statistics"]["strata"]) == 12
    assert result["frozen"]["threshold"] == 7.970395366024563e-06
    assert result["frozen"]["target"]["alpha"] == 0.05
    assert result["frozen"]["target"]["minimum_coverage"] == 0.10
    assert result["verification_scope"] == "producer_composition_only"
    assert result["acceptance_authorized"] is False


@pytest.mark.parametrize(
    ("statistics", "decision", "reasons"),
    (
        (
            _statistics(loss=0.04),
            EMPIRICALLY_MET_BUT_INCONCLUSIVE,
            ["finite_sample_upper_bound_exceeds_target"],
        ),
        (
            _statistics(loss=0.06),
            FAILED,
            ["empirical_risk_exceeds_target"],
        ),
        (
            _statistics(loss=0.0, trusted_pixels=9),
            FAILED,
            ["insufficient_coverage"],
        ),
    ),
)
def test_result_preserves_exact_three_way_decision(
    statistics, decision: str, reasons: list[str]
) -> None:
    result = _result(statistics)

    assert result["phase_decision"] == decision
    assert result["reasons"] == reasons


def test_result_has_no_per_sample_metrics_or_host_metadata() -> None:
    payload = canonical_json(_result()).decode("utf-8")

    assert '"samples"' not in payload
    assert '"sample_id"' not in payload
    for forbidden in ("path", "hostname", "endpoint", "timestamp", "credential", "secret"):
        assert forbidden not in payload.casefold()


def test_result_rejects_noncanonical_identity_and_radiometry() -> None:
    arguments = {
        "statistics": _statistics(loss=0.0),
        "radiometry": _radiometry(),
        "evaluation_id": _sha("evaluation"),
        "permit_sha256": _sha("permit"),
        "ledger_event_sha256": _sha("ledger"),
        "input_receipt_sha256": _sha("input receipt"),
        "ordered_inputs_sha256": _sha("inputs"),
        "ordered_sample_ids_sha256": _sha("sample ids"),
        "ordered_membership_sha256": _sha("membership"),
        "cache_audit_sha256": _sha("cache audit"),
        "map_evidence_sha256": _sha("maps"),
        "producer_revision": "1" * 40,
    }
    with pytest.raises(ValueError, match="digest"):
        build_phase2b3c_result(**{**arguments, "permit_sha256": "wrong"})
    with pytest.raises(ValueError, match="radiometry"):
        build_phase2b3c_result(
            **{
                **arguments,
                "radiometry": {
                    **_radiometry(),
                    "hr": {**_radiometry()["hr"], "clipped_high_count": 7},
                },
            }
        )


def test_result_rejects_forged_statistics_decision() -> None:
    forged = replace(
        _statistics(loss=0.04), phase_decision=CONFIRMED, reasons=()
    )

    with pytest.raises(ValueError, match="decision"):
        _result(forged)
