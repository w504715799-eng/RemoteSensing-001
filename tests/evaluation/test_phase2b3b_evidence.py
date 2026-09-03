"""Strict, host-free tests for the Phase 2B3-B upstream evidence contract."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from trustsr.evaluation import phase2b3b_evidence

_NAMES = tuple(phase2b3b_evidence.PUBLISHED_EVIDENCE_SHA256S)
_ARTIFACTS = Path(__file__).parents[2] / "artifacts" / "phase2b3a"


@pytest.fixture
def published_evidence_dir(tmp_path: Path) -> Iterator[Path]:
    """Copy only the six public JSON receipts; never read a pixel or cache asset."""

    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    for name in _NAMES:
        (evidence_dir / name).write_bytes((_ARTIFACTS / name).read_bytes())
    yield evidence_dir


def _documents(evidence_dir: Path) -> dict[str, object]:
    return {
        name: json.loads((evidence_dir / name).read_text(encoding="utf-8"))
        for name in _NAMES
    }


def _assert_safe_error(callable_: object, host_path: Path) -> None:
    with pytest.raises(ValueError) as caught:
        callable_()  # type: ignore[operator]
    assert str(host_path) not in str(caught.value)


def test_loads_published_evidence_as_small_immutable_frozen_contract(
    published_evidence_dir: Path,
) -> None:
    """A valid six-file publication exposes the complete fixed B3-B score identity."""

    frozen = phase2b3b_evidence.load_frozen_phase2b3a_evidence(published_evidence_dir)

    assert frozen.score_name == "ldsr_variance_k5"
    assert frozen.operator_parameters == {
        "algorithm": "ensemble_variance_score",
        "band_reduction": "mean",
        "correction": 0,
        "seed_count": 5,
        "seed_first": 3407,
        "seed_last": 3411,
    }
    assert frozen.seeds == (3407, 3408, 3409, 3410, 3411)
    assert frozen.risk_name == "local_l1_risk"
    assert frozen.risk_window == 9
    assert frozen.risk_upper_bound == 1.0
    assert frozen.normalization_policy == "uint16_saturate_10000_divide_10000_v2"
    assert frozen.crop_policy == "center_crop_lr_1_hr_4_v1"
    assert frozen.bands == ("B04", "B03", "B02", "B08")
    assert frozen.scale == 4
    assert frozen.post_manifest_sha256 == (
        "c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a"
    )
    assert frozen.input_audit_sha256 == (
        "fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b"
    )
    assert frozen.producer_revision == "58694420c3c0e11d495953a1963c71b997261601"
    assert dict(frozen.source_digests) == phase2b3b_evidence.PUBLISHED_EVIDENCE_SHA256S
    assert len(frozen.candidate_eligibility_evidence) == 3
    assert frozen.candidate_eligibility_evidence[-1]["name"] == "ldsr_variance_k5"


def test_frozen_evidence_does_not_leak_mutable_nested_structures(
    published_evidence_dir: Path,
) -> None:
    """Consumers cannot amend frozen candidate evidence after validation."""

    frozen = phase2b3b_evidence.load_frozen_phase2b3a_evidence(published_evidence_dir)

    with pytest.raises(TypeError):
        frozen.source_digests["new"] = "x"  # type: ignore[index]
    with pytest.raises(TypeError):
        frozen.candidate_eligibility_evidence[0]["eligible"] = False  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        frozen.candidate_eligibility_evidence[0]["stratum_mean_rho"].append({})  # type: ignore[union-attr]


@pytest.mark.parametrize("name", _NAMES)
def test_rejects_a_single_byte_change_to_each_published_file(
    published_evidence_dir: Path, name: str
) -> None:
    """Any published-byte mutation must invalidate the immutable allowlist digest."""

    path = published_evidence_dir / name
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)

    _assert_safe_error(
        lambda: phase2b3b_evidence.load_frozen_phase2b3a_evidence(published_evidence_dir),
        published_evidence_dir,
    )


@pytest.mark.parametrize("fault", ("missing", "extra", "symlink"))
def test_rejects_missing_extra_or_symlinked_evidence_files(
    published_evidence_dir: Path, fault: str
) -> None:
    """The input is precisely six regular files, with no directory-entry ambiguity."""

    target = published_evidence_dir / _NAMES[0]
    if fault == "missing":
        target.unlink()
    elif fault == "extra":
        (published_evidence_dir / "unapproved.json").write_bytes(b"{}")
    else:
        replacement = published_evidence_dir / "replacement.json"
        replacement.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(replacement.name)

    _assert_safe_error(
        lambda: phase2b3b_evidence.load_frozen_phase2b3a_evidence(published_evidence_dir),
        published_evidence_dir,
    )


def test_rejects_noncanonical_json_before_accepting_its_digest(
    published_evidence_dir: Path,
) -> None:
    """Equivalent JSON bytes are insufficient: published evidence is canonical byte-for-byte."""

    path = published_evidence_dir / _NAMES[0]
    path.write_text(json.dumps(json.loads(path.read_text()), indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical") as caught:
        phase2b3b_evidence.load_frozen_phase2b3a_evidence(published_evidence_dir)
    assert str(published_evidence_dir) not in str(caught.value)


def test_semantic_validator_rejects_false_a1_gate_without_digest_masking(
    published_evidence_dir: Path,
) -> None:
    """A1 acceptance gates remain semantic checks even after byte validation is bypassed in-unit."""

    documents = copy.deepcopy(_documents(published_evidence_dir))
    documents["sen2naipv2-development-smoke-acceptance-v2.json"]["replay_pass"] = False  # type: ignore[index]

    with pytest.raises(ValueError, match="A1"):
        phase2b3b_evidence._validate_semantics(documents)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "receipt_name, receipt_key",
    (
        ("sen2naipv2-development-smoke-acceptance-v2.json", "phase2b3a-a1-result.json"),
        ("sen2naipv2-development-score-acceptance-v1.json", "phase2b3a-a2-cache-audit.json"),
    ),
)
def test_semantic_validator_rejects_wrong_internal_result_or_cache_digest(
    published_evidence_dir: Path, receipt_name: str, receipt_key: str
) -> None:
    """Acceptance receipts bind their result and cache audit rather than merely naming them."""

    documents = copy.deepcopy(_documents(published_evidence_dir))
    documents[receipt_name]["digests"][receipt_key] = "0" * 64  # type: ignore[index]

    with pytest.raises(ValueError, match="digest"):
        phase2b3b_evidence._validate_semantics(documents)  # type: ignore[arg-type]


def test_semantic_validator_rejects_mutated_frozen_cost_rank_without_digest_masking(
    published_evidence_dir: Path,
) -> None:
    """The K5 score is third in the fixed zero-based cost order, not merely any frozen score."""

    documents = copy.deepcopy(_documents(published_evidence_dir))
    for name in (
        "sen2naipv2-development-score-audit-v1.json",
        "sen2naipv2-development-score-acceptance-v1.json",
    ):
        documents[name]["frozen_score"]["cost_rank"] = 0  # type: ignore[index]

    with pytest.raises(ValueError, match="cost"):
        phase2b3b_evidence._validate_semantics(documents)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "result_name, audit_name, counts",
    (
        (
            "sen2naipv2-development-smoke-v2.json",
            "sen2naipv2-development-smoke-cache-audit-v2.json",
            {"sample_count": 5, "prediction_count": 109, "score_count": 21},
        ),
        (
            "sen2naipv2-development-score-audit-v1.json",
            "sen2naipv2-development-score-cache-audit-v1.json",
            {"sample_count": 119, "prediction_count": 839, "score_count": 359},
        ),
    ),
)
def test_semantic_validator_rejects_mutated_fixed_stage_counts(
    published_evidence_dir: Path,
    result_name: str,
    audit_name: str,
    counts: dict[str, int],
) -> None:
    """Both public result and cache-audit receipts carry the fixed A1/A2 stage counts."""

    documents = copy.deepcopy(_documents(published_evidence_dir))
    for name in (result_name, audit_name):
        documents[name].update(counts)  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="count"):
        phase2b3b_evidence._validate_semantics(documents)  # type: ignore[arg-type]


def test_semantic_validator_rejects_non_roi_a2_statistical_unit(
    published_evidence_dir: Path,
) -> None:
    """The A2 selection is frozen at the ROI unit; pixel-level substitution cannot enter B."""

    documents = copy.deepcopy(_documents(published_evidence_dir))
    documents["sen2naipv2-development-score-audit-v1.json"]["statistical_unit"] = "pixel"  # type: ignore[index]

    with pytest.raises(ValueError, match="configuration"):
        phase2b3b_evidence._validate_semantics(documents)  # type: ignore[arg-type]


@pytest.mark.parametrize("fault", ("score_configuration", "candidate_entry", "selected_evidence"))
def test_semantic_validator_rejects_wrong_a2_nested_types_and_selected_candidate(
    published_evidence_dir: Path, fault: str
) -> None:
    """Malformed nested evidence always fails closed with ValueError, never AttributeError."""

    documents = copy.deepcopy(_documents(published_evidence_dir))
    result = documents["sen2naipv2-development-score-audit-v1.json"]
    acceptance = documents["sen2naipv2-development-score-acceptance-v1.json"]
    if fault == "score_configuration":
        result["score_configuration"] = []  # type: ignore[index]
    elif fault == "candidate_entry":
        for frozen in (result["frozen_score"], acceptance["frozen_score"]):  # type: ignore[index]
            frozen["candidate_eligibility_evidence"][0] = []
    else:
        for frozen in (result["frozen_score"], acceptance["frozen_score"]):  # type: ignore[index]
            frozen["selected_candidate_evidence"] = frozen["candidate_eligibility_evidence"][0]

    with pytest.raises(ValueError):
        phase2b3b_evidence._validate_semantics(documents)  # type: ignore[arg-type]


@pytest.mark.parametrize("fault", ("decision", "eligibility", "frozen_mismatch"))
def test_semantic_validator_rejects_a2_freeze_tampering_without_digest_masking(
    published_evidence_dir: Path, fault: str
) -> None:
    """A2 must freeze the fixed, eligible LDSR K5 evidence and match its acceptance receipt."""

    documents = copy.deepcopy(_documents(published_evidence_dir))
    result = documents["sen2naipv2-development-score-audit-v1.json"]
    acceptance = documents["sen2naipv2-development-score-acceptance-v1.json"]
    if fault == "decision":
        result["phase_decision"] = "stop_no_eligible_score"  # type: ignore[index]
    elif fault == "eligibility":
        for frozen in (result["frozen_score"], acceptance["frozen_score"]):  # type: ignore[index]
            frozen["selected_candidate_evidence"]["eligible"] = False
            frozen["candidate_eligibility_evidence"][-1]["eligible"] = False
    else:
        acceptance["frozen_score"]["name"] = "lr_reprojection_l1"  # type: ignore[index]

    with pytest.raises(ValueError, match="A2|frozen|eligibility"):
        phase2b3b_evidence._validate_semantics(documents)  # type: ignore[arg-type]
