"""Independent final verification for Phase 2B3-B result projections."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest
import test_phase2b3b_result as result_fixtures

from trustsr.evaluation import phase2b3b_result, phase2b3b_result_verify


@dataclass(frozen=True)
class _Case:
    result: dict[str, object]
    audit: dict[str, object]
    preflight: dict[str, object]
    records: tuple[dict[str, object], ...]


def _records(sample_ids: tuple[str, ...]) -> tuple[dict[str, object], ...]:
    records = []
    for membership in result_fixtures._membership(sample_ids):
        records.append(
            {
                "sample_id": membership["sample_id"],
                "selection_sha256": membership["selection_sha256"],
                "spatial_group_id": membership["spatial_group_id"],
                "split": "calibration",
                "days_between": membership["days_between"],
                "correlation_bin": membership["correlation_bin"],
                "selection_round": membership["selection_round"],
                "lr_asset": {"sha256": membership["lr_asset_sha256"]},
                "hr_asset": {"sha256": membership["hr_asset_sha256"]},
            }
        )
    return tuple(records)


def _case(prefix: str = "calibration") -> _Case:
    sample_ids = result_fixtures._sample_ids(prefix)
    preflight = result_fixtures._preflight(sample_ids)
    input_receipt = result_fixtures._input_receipt(sample_ids)
    fit = result_fixtures._fit(sample_ids)
    audit = result_fixtures._audit(sample_ids)
    radiometry = result_fixtures._radiometry(sample_ids)
    result = phase2b3b_result.build_phase2b3b_result(
        preflight,
        input_receipt,
        fit,
        audit,
        radiometry,
        result_fixtures._revision(),
    )
    return _Case(result, audit, preflight, _records(sample_ids))


@pytest.fixture(scope="module")
def valid_case() -> _Case:
    return _case()


def _verify(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: _Case,
    *,
    result: dict[str, object] | None = None,
    audit: dict[str, object] | None = None,
    authority: _Case | None = None,
    events: list[object] | None = None,
) -> object:
    trusted = case if authority is None else authority
    calls = [] if events is None else events

    def load_preflight(*args: object) -> object:
        calls.append(("preflight", args))
        return deepcopy(trusted.preflight)

    def load_records(*args: object) -> object:
        calls.append(("records", args))
        return deepcopy(trusted.records)

    def verify_revision(project_root: Path, revision: str) -> str:
        calls.append(("revision", project_root, revision))
        return revision

    monkeypatch.setattr(phase2b3b_result_verify, "load_phase2b3b_preflight", load_preflight)
    monkeypatch.setattr(phase2b3b_result_verify, "load_calibration_records", load_records)
    monkeypatch.setattr(
        phase2b3b_result_verify,
        "verify_recorded_phase2b3b_revision",
        verify_revision,
    )
    return phase2b3b_result_verify.verify_phase2b3b_result(
        deepcopy(case.result if result is None else result),
        deepcopy(case.audit if audit is None else audit),
        project_root=tmp_path,
        evidence_dir=tmp_path / "evidence",
        storage_root=tmp_path / "storage",
        manifest_path=tmp_path / "manifest.json",
    )


def test_returns_frozen_host_free_receipt_and_calls_all_trusted_gates(
    valid_case: _Case, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[object] = []

    receipt = _verify(monkeypatch, tmp_path, valid_case, events=events)

    assert (
        receipt.schema
        == "trustsr.phase2b3b-calibration-result-metadata-verification.v1"
    )
    assert receipt.verification_scope == "metadata_consistency_only"
    assert receipt.cache_computation_verified is False
    assert receipt.cache_audit_sha256 == valid_case.result["cache_audit_sha256"]
    assert receipt.input_receipt_sha256 == valid_case.result["input_receipt_sha256"]
    assert receipt.ordered_inputs_sha256 == valid_case.result["ordered_inputs_sha256"]
    assert receipt.map_evidence_sha256 == valid_case.result["map_evidence_sha256"]
    assert receipt.producer_revision == "c" * 40
    assert receipt.phase_decision == "freeze_calibration"
    assert len(receipt.result_sha256) == 64
    assert len(receipt.radiometry_aggregate_sha256) == 64
    assert [event[0] for event in events] == ["preflight", "records", "revision"]
    assert events[-1] == ("revision", tmp_path, "c" * 40)
    assert str(tmp_path) not in repr(receipt)
    with pytest.raises(FrozenInstanceError):
        receipt.phase_decision = "changed"  # type: ignore[misc]


def test_receipt_and_entrypoint_express_metadata_only_boundary() -> None:
    receipt_docstring = phase2b3b_result_verify.VerifiedPhase2B3BResult.__doc__
    verify_docstring = phase2b3b_result_verify.verify_phase2b3b_result.__doc__

    assert receipt_docstring is not None
    assert "cache pixels" in receipt_docstring
    assert "score/risk" in receipt_docstring
    assert "cannot authorize acceptance" in receipt_docstring
    assert verify_docstring is not None
    assert "does not prove cache pixels" in verify_docstring


@pytest.mark.parametrize(
    "fault",
    (
        "extra",
        "cache_projection",
        "cache_digest",
        "map_digest",
        "input_digest",
        "missing_membership_digest",
        "radiometry",
        "coverage",
        "risk_bound",
        "decision",
        "bool_count",
        "nan_alpha",
    ),
)
def test_rejects_single_layer_result_attacks(
    valid_case: _Case,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault: str,
) -> None:
    result = deepcopy(valid_case.result)
    if fault == "extra":
        result["extra"] = True
    elif fault == "cache_projection":
        result["samples"][0]["predictions"][0]["cache_key"] = "f" * 64
    elif fault == "cache_digest":
        result["cache_audit_sha256"] = "f" * 64
    elif fault == "map_digest":
        result["map_evidence_sha256"] = "f" * 64
    elif fault == "input_digest":
        result["ordered_inputs_sha256"] = "f" * 64
    elif fault == "missing_membership_digest":
        result["samples"][0]["input"].pop("membership_sha256")
    elif fault == "radiometry":
        result["radiometry"]["lr"]["raw_crop_minimum"] = 101
    elif fault == "coverage":
        result["coverage"] = 0.5
    elif fault == "risk_bound":
        result["risk_bound"] = 0.5
    elif fault == "decision":
        result["phase_decision"] = "stop_insufficient_coverage"
    elif fault == "bool_count":
        result["counts"]["trusted_pixels"] = True
    else:
        result["target"]["alpha"] = float("nan")

    with pytest.raises((TypeError, ValueError)):
        _verify(monkeypatch, tmp_path, valid_case, result=result)


def test_rejects_fully_self_consistent_internal_test_result_against_authority(
    valid_case: _Case, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    forged = _case("internal_test")

    with pytest.raises(ValueError, match="non-calibration|authoritative"):
        _verify(
            monkeypatch,
            tmp_path,
            forged,
            authority=valid_case,
        )


def test_rejects_cache_audit_attack_before_trusting_result_projection(
    valid_case: _Case, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audit = deepcopy(valid_case.audit)
    audit["samples"][0]["predictions"][0]["cache_key"] = "f" * 64

    with pytest.raises(ValueError, match="cache key"):
        _verify(monkeypatch, tmp_path, valid_case, audit=audit)
