"""Independent metadata verification for Phase 2B3-C results."""

from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from trustsr.evaluation.phase2b3c_result import build_phase2b3c_result
from trustsr.evaluation.phase2b3c_result_verify import verify_phase2b3c_result
from trustsr.evaluation.phase2b3c_statistics import (
    ROIEvaluation,
    build_phase2b3c_statistics,
)
from trustsr.jsonio import canonical_json


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _result() -> dict[str, object]:
    rois = tuple(
        ROIEvaluation(
            sample_id=f"test-{index:03d}",
            days_between=(-1, 0, 1)[(index % 12) // 4],
            correlation_bin=index % 4,
            selection_round=index // 12 + 1,
            trusted_pixels=20,
            total_pixels=100,
            coverage=0.2,
            loss=0.0,
        )
        for index in range(120)
    )
    return build_phase2b3c_result(
        statistics=build_phase2b3c_statistics(rois),
        radiometry={
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
        },
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


def test_verifies_canonical_result_with_non_authorizing_scope() -> None:
    payload = canonical_json(_result())

    verified = verify_phase2b3c_result(payload)

    assert verified.result_sha256 == hashlib.sha256(payload).hexdigest()
    assert verified.phase_decision == "confirmed"
    assert verified.verification_scope == "metadata_consistency_only"
    assert verified.cache_computation_verified is False
    assert verified.acceptance_authorized is False
    assert verified.evaluation_id == _sha("evaluation")
    assert verified.cache_audit_sha256 == _sha("cache audit")


def test_rejects_noncanonical_bytes_and_extra_or_leaking_fields() -> None:
    with pytest.raises(ValueError, match="canonical"):
        verify_phase2b3c_result(canonical_json(_result()) + b"\n")

    changed = _result()
    changed["samples"] = [{"sample_id": "test-000", "loss": 0.1}]
    with pytest.raises(ValueError, match="keys|forbidden"):
        verify_phase2b3c_result(changed)

    changed = _result()
    changed["runtime_path"] = "/tmp/result"
    with pytest.raises(ValueError, match="keys|forbidden"):
        verify_phase2b3c_result(changed)


def test_rejects_decision_reason_and_aggregate_mutations() -> None:
    changed = _result()
    changed["phase_decision"] = "failed"
    with pytest.raises(ValueError, match="decision"):
        verify_phase2b3c_result(changed)

    changed = _result()
    changed["reasons"] = ["finite_sample_upper_bound_exceeds_target"]
    with pytest.raises(ValueError, match="reason"):
        verify_phase2b3c_result(changed)

    changed = _result()
    changed["statistics"]["trusted_pixels"] += 1
    with pytest.raises(ValueError, match="strata|aggregate"):
        verify_phase2b3c_result(changed)


def test_rejects_stratum_order_and_identity_digest_mutations() -> None:
    changed = _result()
    changed["statistics"]["strata"] = list(
        reversed(changed["statistics"]["strata"])
    )
    with pytest.raises(ValueError, match="strata"):
        verify_phase2b3c_result(changed)

    changed = deepcopy(_result())
    changed["digests"]["map_evidence_sha256"] = "0" * 64
    verified = verify_phase2b3c_result(changed)
    assert verified.map_evidence_sha256 == "0" * 64
