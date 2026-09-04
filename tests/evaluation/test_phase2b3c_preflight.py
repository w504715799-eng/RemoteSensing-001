"""Metadata-only tripwires for the Phase 2B3-C preflight."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from trustsr.evaluation.phase2b3c_access import VerifiedAccessPermit
from trustsr.evaluation.phase2b3c_evidence import load_frozen_phase2b3b_evidence
from trustsr.evaluation.phase2b3c_preflight import (
    build_phase2b3c_preflight,
    load_phase2b3c_preflight,
)
from trustsr.evaluation.phase2b3c_revision import VerifiedRevision
from trustsr.jsonio import canonical_json

_PROJECT_ROOT = Path(__file__).parents[2]
_EVIDENCE = _PROJECT_ROOT / "artifacts" / "phase2b3b"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "sample_id": f"internal-test-{index:03d}",
            "selection_sha256": _sha(f"selection:{index}"),
            "spatial_group_id": _sha(f"group:{index}"),
            "split": "internal_test",
            "days_between": (-1, 0, 1)[(index % 12) // 4],
            "correlation_bin": index % 4,
            "selection_round": index // 12 + 1,
            "lr_asset": {
                "relative_path": f"secret/internal_test/{index}/lr.tif",
                "sha256": _sha(f"lr:{index}"),
            },
            "hr_asset": {
                "relative_path": f"secret/internal_test/{index}/hr.tif",
                "sha256": _sha(f"hr:{index}"),
            },
        }
        for index in range(120)
    )


def _revision() -> VerifiedRevision:
    return VerifiedRevision("main", "1" * 40, "1" * 40, _sha("tree"))


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def test_builds_immutable_host_free_preflight_from_metadata_only() -> None:
    evidence = load_frozen_phase2b3b_evidence(_EVIDENCE, _PROJECT_ROOT)

    preflight = build_phase2b3c_preflight(evidence, _records(), _revision())

    assert preflight["schema"] == "trustsr.phase2b3c-preflight.v1"
    assert preflight["authorization_required"] is True
    assert preflight["implementation"] == {
        "revision": "1" * 40,
        "computation_tree_sha256": _sha("tree"),
    }
    assert preflight["evaluation"]["split"] == "internal_test"
    assert preflight["evaluation"]["sample_count"] == 120
    assert len(preflight["evaluation"]["strata"]) == 12
    assert preflight["policy"]["alpha"] == 0.05
    assert preflight["policy"]["minimum_coverage"] == 0.10
    assert preflight["permit"] is None
    payload = json.dumps(_plain(preflight), sort_keys=True)
    assert "secret/" not in payload
    assert '"sample_id"' not in payload
    assert "internal-test-000" not in payload
    with pytest.raises(TypeError):
        preflight["schema"] = "forged"  # type: ignore[index]


def test_rejects_wrong_split_or_balanced_design_before_summary() -> None:
    evidence = load_frozen_phase2b3b_evidence(_EVIDENCE, _PROJECT_ROOT)
    records = list(_records())
    records[0] = {**records[0], "split": "calibration"}
    with pytest.raises(ValueError, match="internal_test"):
        build_phase2b3c_preflight(evidence, records, _revision())


def test_optional_verified_permit_is_cross_bound_without_advancing_ledger() -> None:
    evidence = load_frozen_phase2b3b_evidence(_EVIDENCE, _PROJECT_ROOT)
    initial = build_phase2b3c_preflight(evidence, _records(), _revision())
    permit = VerifiedAccessPermit(
        evaluation_id=_sha("evaluation"),
        permit_sha256=_sha("permit"),
        readiness_sha256=_sha("readiness"),
        implementation_revision="1" * 40,
        computation_tree_sha256=_sha("tree"),
        ordered_membership_sha256=initial["evaluation"][
            "ordered_membership_sha256"
        ],
        environment_sha256=_sha("environment"),
        phase2b3b_acceptance_sha256=(
            "ce7f1a91e9954235e289dfc6eececb67261ff57110caaf15fe6640a0fb3b69ab"
        ),
        _authority=object(),
    )

    permitted = build_phase2b3c_preflight(
        evidence, _records(), _revision(), permit=permit
    )

    assert permitted["permit"] == {
        "evaluation_id": _sha("evaluation"),
        "permit_sha256": _sha("permit"),
        "readiness_sha256": _sha("readiness"),
        "environment_sha256": _sha("environment"),
    }

    records = list(_records())
    records[0] = {**records[0], "selection_round": 2}
    with pytest.raises(ValueError, match="round"):
        build_phase2b3c_preflight(evidence, records, _revision())


def test_path_preflight_invokes_only_metadata_gates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import trustsr.evaluation.phase2b3c_preflight as module

    evidence = load_frozen_phase2b3b_evidence(_EVIDENCE, _PROJECT_ROOT)
    events: list[str] = []

    monkeypatch.setattr(
        module,
        "verify_phase2b3c_implementation_revision",
        lambda root, revision, tree: events.append("revision") or _revision(),
    )
    monkeypatch.setattr(
        module,
        "load_frozen_phase2b3b_evidence",
        lambda evidence_dir, project_root: events.append("evidence") or evidence,
    )
    monkeypatch.setattr(
        module,
        "load_internal_test_records",
        lambda storage_root, manifest: events.append("manifest") or _records(),
    )
    for forbidden in (
        "load_internal_test_pairs",
        "load_complete_cached_internal_test_bundles",
        "LDSRS2X4",
        "cuda",
        "advance_access_ledger",
        "write_phase2b3c_bundle",
    ):
        assert forbidden not in module.__dict__

    preflight = load_phase2b3c_preflight(
        project_root=tmp_path,
        evidence_dir=tmp_path / "evidence",
        storage_root=tmp_path / "storage",
        manifest_path=tmp_path / "manifest.jsonl",
        implementation_revision="1" * 40,
        computation_tree_sha256=_sha("tree"),
    )

    assert events == ["revision", "evidence", "manifest"]
    assert hashlib.sha256(canonical_json(_plain(preflight))).hexdigest()
