"""CPU-only, metadata-only contracts for the Phase 2B3-B preflight."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from trustsr.evaluation import phase2b3b_preflight
from trustsr.evaluation.phase2b3b_evidence import load_frozen_phase2b3a_evidence

_ARTIFACTS = Path(__file__).parents[2] / "artifacts" / "phase2b3a"


def _calibration_records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "sample_id": f"calibration-{day}-{bin_index}-{round_index}",
            "selection_sha256": f"selection-{day}-{bin_index}-{round_index}",
            "spatial_group_id": f"group-{day}-{bin_index}-{round_index}",
            "split": "calibration",
            "days_between": day,
            "correlation_bin": bin_index,
            "selection_round": round_index,
            "lr_asset": {"path": f"secret/lr-{day}-{bin_index}-{round_index}.tif"},
            "hr_asset": {"path": f"secret/hr-{day}-{bin_index}-{round_index}.tif"},
        }
        for day in (-1, 0, 1)
        for bin_index in range(4)
        for round_index in range(1, 11)
    )


@pytest.fixture
def frozen_evidence(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    from trustsr.evaluation.phase2b3b_evidence import PUBLISHED_EVIDENCE_SHA256S

    for name in PUBLISHED_EVIDENCE_SHA256S:
        (evidence_dir / name).write_bytes((_ARTIFACTS / name).read_bytes())
    return load_frozen_phase2b3a_evidence(evidence_dir)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def test_builds_immutable_host_free_preflight_from_frozen_metadata(
    frozen_evidence: object, tmp_path: Path
) -> None:
    """Dropping identity fields or leaking asset paths must break this output contract."""

    result = phase2b3b_preflight.build_phase2b3b_preflight(
        frozen_evidence, _calibration_records()
    )

    assert set(result) == {"schema", "upstream", "calibration", "score", "risk", "input"}
    assert result["schema"] == "trustsr.phase2b3b-preflight.v1"
    assert result["upstream"]["publication_commit"] == (
        "b386d4b38c9f3725107eed178829955d442f5601"
    )
    assert result["upstream"]["producer_revision"] == (
        "58694420c3c0e11d495953a1963c71b997261601"
    )
    assert len(result["upstream"]["evidence_sha256s"]) == 6
    assert result["upstream"]["evidence_sha256s"][
        "sen2naipv2-development-score-acceptance-v1.json"
    ] == "34741fe788cac6e28c6d8b1ce2fd96335b608e1b3e6ffb29e82ac064a2118227"
    assert result["calibration"]["sample_count"] == 120
    assert result["calibration"]["strata"] == tuple(
        {
            "days_between": day,
            "correlation_bin": bin_index,
            "sample_count": 10,
        }
        for day in (-1, 0, 1)
        for bin_index in range(4)
    )
    assert result["score"] == {
        "name": "ldsr_variance_k5",
        "operator_parameters": {
            "algorithm": "ensemble_variance_score",
            "band_reduction": "mean",
            "correction": 0,
            "seed_count": 5,
            "seed_first": 3407,
            "seed_last": 3411,
        },
        "seeds": (3407, 3408, 3409, 3410, 3411),
    }
    assert result["risk"] == {
        "name": "local_l1_risk",
        "window": 9,
        "upper_bound": 1.0,
    }
    assert result["input"] == {
        "normalization_policy": "uint16_saturate_10000_divide_10000_v2",
        "crop_policy": "center_crop_lr_1_hr_4_v1",
        "bands": ("B04", "B03", "B02", "B08"),
        "scale": 4,
    }
    encoded = json.dumps(_plain(result), sort_keys=True)
    assert "secret/" not in encoded
    assert str(tmp_path) not in encoded
    assert "sample_id" not in encoded
    assert "alpha" not in encoded
    assert "coverage" not in encoded

    with pytest.raises(TypeError):
        result["schema"] = "forged"  # type: ignore[index]
    with pytest.raises(TypeError):
        result["risk"]["window"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        result["upstream"]["evidence_sha256s"]["new.json"] = "0" * 64  # type: ignore[index]


@pytest.mark.parametrize("fault", ("count", "stratum", "round"))
def test_rejects_incomplete_or_forged_calibration_metadata(
    frozen_evidence: object, fault: str
) -> None:
    """Skipping a ROI or replacing any fixed calibration cell must fail closed."""

    records = list(_calibration_records())
    if fault == "count":
        records.pop()
    elif fault == "stratum":
        records[0]["days_between"] = 2
    else:
        records[0]["selection_round"] = 2

    with pytest.raises(ValueError, match="120|stratum|round"):
        phase2b3b_preflight.build_phase2b3b_preflight(frozen_evidence, records)


@pytest.mark.parametrize(
    ("fault", "message"),
    (
        ("duplicate_sample_id", "unique sample_id"),
        ("duplicate_selection_sha256", "unique selection_sha256"),
        ("duplicate_spatial_group_id", "unique spatial_group_id"),
        ("missing_sample_id", "sample_id"),
        ("empty_selection_sha256", "selection_sha256"),
        ("non_string_spatial_group_id", "spatial_group_id"),
        ("missing_lr_asset", "lr_asset"),
        ("empty_hr_asset", "hr_asset"),
    ),
)
def test_rejects_forged_calibration_identity_or_asset_metadata(
    frozen_evidence: object, fault: str, message: str
) -> None:
    """Direct callers cannot bypass the frozen subset identity and asset contract."""

    records = [dict(record) for record in _calibration_records()]
    if fault.startswith("duplicate_"):
        field = fault.removeprefix("duplicate_")
        records[1][field] = records[0][field]
    elif fault == "missing_sample_id":
        records[0].pop("sample_id")
    elif fault == "empty_selection_sha256":
        records[0]["selection_sha256"] = ""
    elif fault == "non_string_spatial_group_id":
        records[0]["spatial_group_id"] = 1
    elif fault == "missing_lr_asset":
        records[0].pop("lr_asset")
    else:
        records[0]["hr_asset"] = {}

    with pytest.raises(ValueError, match=message):
        phase2b3b_preflight.build_phase2b3b_preflight(frozen_evidence, records)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"score_name": "three_model_disagreement"}, "frozen"),
        ({"risk_window": 1}, "frozen"),
        ({"source_digests": None}, "frozen"),
        ({"candidate_eligibility_evidence": ()}, "eligibility"),
    ),
)
def test_rejects_forged_phase2b3a_evidence(
    frozen_evidence: object, changes: dict[str, object], message: str
) -> None:
    """Constructing a dataclass directly cannot bypass the published evidence identity."""

    forged = replace(frozen_evidence, **changes)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        phase2b3b_preflight.build_phase2b3b_preflight(forged, _calibration_records())


def test_path_loader_composes_only_verified_evidence_and_calibration_metadata(
    monkeypatch: pytest.MonkeyPatch, frozen_evidence: object, tmp_path: Path
) -> None:
    """Adding a pixel/model loader to preflight must break this narrow orchestration boundary."""

    records = _calibration_records()
    evidence_dir = tmp_path / "six-json-files"
    storage_root = tmp_path / "storage"
    manifest_path = tmp_path / "post.jsonl"
    events: list[tuple[object, ...]] = []

    def load_evidence(path: Path) -> object:
        events.append(("evidence", path))
        return frozen_evidence

    def load_metadata(root: Path, manifest: Path) -> tuple[dict[str, object], ...]:
        events.append(("metadata", root, manifest))
        return records

    monkeypatch.setattr(phase2b3b_preflight, "load_frozen_phase2b3a_evidence", load_evidence)
    monkeypatch.setattr(phase2b3b_preflight, "load_calibration_records", load_metadata)

    result = phase2b3b_preflight.load_phase2b3b_preflight(
        evidence_dir, storage_root, manifest_path
    )

    assert events == [
        ("evidence", evidence_dir),
        ("metadata", storage_root, manifest_path),
    ]
    assert result["calibration"]["sample_count"] == 120


def test_output_is_identical_when_valid_calibration_metadata_order_changes(
    frozen_evidence: object,
) -> None:
    """Input iteration order cannot make a host-dependent or noncanonical receipt."""

    first = phase2b3b_preflight.build_phase2b3b_preflight(
        frozen_evidence, _calibration_records()
    )
    second = phase2b3b_preflight.build_phase2b3b_preflight(
        frozen_evidence, tuple(reversed(_calibration_records()))
    )

    assert _plain(first) == _plain(second)
