"""Synthetic contracts for calibration-only conformal fit orchestration."""

from __future__ import annotations

import hashlib
import math
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from trustsr.artifacts.predictions import tensor_sha256
from trustsr.artifacts.scores import ScoreIdentity
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
)
from trustsr.evaluation.calibration_fit import CalibrationFit, fit_calibration_maps
from trustsr.evaluation.calibration_maps import (
    A2_RESULT_SHA256,
    PUBLICATION_COMMIT,
    CachedCalibrationScore,
    CalibrationMaps,
)
from trustsr.evaluation.calibration_predictions import SEEDS
from trustsr.evaluation.phase2b3b_evidence import INPUT_AUDIT_SHA256, PRODUCER_REVISION
from trustsr.jsonio import canonical_json


def _map(
    index: int,
    *,
    scores: tuple[float, ...] = (0.1,),
    risks: tuple[float, ...] = (0.0,),
) -> CalibrationMaps:
    assert len(scores) == len(risks)
    sample_id = f"calibration-{index:03d}"
    prediction_sha256s = tuple(character * 64 for character in "abcde")
    score = torch.tensor([scores], dtype=torch.float64)
    risk = torch.tensor([risks], dtype=torch.float64)
    identity = ScoreIdentity(
        score_name="ldsr_variance_k5",
        score_schema_version=1,
        sample_id=sample_id,
        input_sha256s=prediction_sha256s,
        operator_parameters={
            "algorithm": "ensemble_variance_score",
            "band_reduction": "mean",
            "correction": 0,
            "seed_first": SEEDS[0],
            "seed_last": SEEDS[-1],
            "seed_count": len(SEEDS),
            "lr_sha256": "f" * 64,
            "source": f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}",
            "post_manifest_sha256": POST_MANIFEST_SHA256,
            "input_audit_sha256": INPUT_AUDIT_SHA256,
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "crop_policy": CROP_POLICY,
            "phase2b3a_publication_commit": PUBLICATION_COMMIT,
            "phase2b3a_a2_result_sha256": A2_RESULT_SHA256,
            "phase2b3a_producer_revision": PRODUCER_REVISION,
        },
    )
    return CalibrationMaps(
        sample_id=sample_id,
        score=CachedCalibrationScore(
            name="ldsr_variance_k5",
            identity=identity,
            score_sha256=tensor_sha256(score),
            tensor=score,
        ),
        score_prediction_sha256s=prediction_sha256s,
        risk_name="local_l1_risk",
        risk_window=9,
        risk_sha256=tensor_sha256(risk),
        risk=risk,
    )


def _maps(**kwargs: object) -> tuple[CalibrationMaps, ...]:
    return tuple(_map(index, **kwargs) for index in range(120))


def test_all_abstain_uses_null_threshold_and_stops() -> None:
    result = fit_calibration_maps(_maps(), alpha=0.001, minimum_coverage=0.0)

    assert result.threshold is None
    assert result.all_abstain is True
    assert result.risk_bound == pytest.approx(1 / 121)
    assert result.trusted_pixels == 0
    assert result.total_pixels == 120
    assert result.coverage == 0.0
    assert result.phase_decision == "stop_insufficient_coverage"
    assert result.as_dict()["threshold"] is None


def test_finite_threshold_freezes_only_when_coverage_meets_explicit_gate() -> None:
    result = fit_calibration_maps(_maps(), alpha=0.02, minimum_coverage=1.0)

    assert result.threshold == pytest.approx(0.1)
    assert result.all_abstain is False
    assert result.risk_bound == pytest.approx(1 / 121)
    assert result.trusted_pixels == result.total_pixels == 120
    assert result.coverage == 1.0
    assert result.phase_decision == "freeze_calibration"


def test_finite_threshold_below_explicit_coverage_gate_stops() -> None:
    result = fit_calibration_maps(
        _maps(scores=(0.1, 0.2), risks=(0.0, 1.0)), alpha=0.02, minimum_coverage=0.75
    )

    assert result.threshold == pytest.approx(0.1)
    assert result.trusted_pixels == 120
    assert result.total_pixels == 240
    assert result.coverage == 0.5
    assert result.phase_decision == "stop_insufficient_coverage"


def test_finite_threshold_rejects_risk_bound_above_alpha() -> None:
    fit = fit_calibration_maps(_maps(), alpha=0.02, minimum_coverage=1.0)

    with pytest.raises(ValueError, match="risk_bound.*alpha"):
        replace(fit, risk_bound=0.03)


@pytest.mark.parametrize("count", [119, 121])
def test_rejects_anything_other_than_exactly_120_maps(count: int) -> None:
    with pytest.raises(ValueError, match="120"):
        fit_calibration_maps(
            tuple(_map(index) for index in range(count)), alpha=0.02, minimum_coverage=0.1
        )


def test_rejects_duplicate_sample_ids_and_forged_map_state() -> None:
    maps = list(_maps())
    maps[-1] = maps[0]
    with pytest.raises(ValueError, match="unique"):
        fit_calibration_maps(maps, alpha=0.02, minimum_coverage=0.1)

    forged = object.__new__(CalibrationMaps)
    valid = _map(119)
    for name in (
        "sample_id",
        "score",
        "score_prediction_sha256s",
        "risk_name",
        "risk_window",
        "risk",
    ):
        object.__setattr__(forged, name, getattr(valid, name))
    object.__setattr__(forged, "risk_sha256", "0" * 64)
    maps = list(_maps())
    maps[-1] = forged
    with pytest.raises(ValueError, match="risk.*digest"):
        fit_calibration_maps(maps, alpha=0.02, minimum_coverage=0.1)


@pytest.mark.parametrize(
    ("alpha", "minimum_coverage"),
    [
        (True, 0.1),
        (float("nan"), 0.1),
        (0.0, 0.1),
        (1.1, 0.1),
        (0.02, True),
        (0.02, float("nan")),
        (0.02, -0.1),
        (0.02, 1.1),
    ],
)
def test_requires_explicit_finite_in_range_scientific_parameters(
    alpha: float, minimum_coverage: float
) -> None:
    with pytest.raises(ValueError, match="alpha|minimum_coverage"):
        fit_calibration_maps(_maps(), alpha=alpha, minimum_coverage=minimum_coverage)


def test_result_is_deterministic_frozen_and_json_safe_with_fresh_dict() -> None:
    maps = _maps(scores=(0.1, 0.2), risks=(0.0, 1.0))
    first = fit_calibration_maps(maps, alpha=0.02, minimum_coverage=0.4)
    second = fit_calibration_maps(list(maps), alpha=0.02, minimum_coverage=0.4)

    assert isinstance(first, CalibrationFit)
    assert first == second
    assert first.as_dict() == second.as_dict()
    payload = first.as_dict()
    assert set(payload) == {
        "alpha",
        "minimum_coverage",
        "threshold",
        "all_abstain",
        "risk_bound",
        "risk_upper_bound",
        "calibration_size",
        "trusted_pixels",
        "total_pixels",
        "coverage",
        "phase_decision",
        "sample_ids",
        "map_evidence_sha256",
    }
    expected_map_evidence_sha256 = hashlib.sha256(
        canonical_json(
            [
                {
                    "sample_id": value.sample_id,
                    "score_sha256": value.score.score_sha256,
                    "risk_sha256": value.risk_sha256,
                }
                for value in maps
            ]
        )
    ).hexdigest()
    assert first.map_evidence_sha256 == expected_map_evidence_sha256
    assert payload["map_evidence_sha256"] == expected_map_evidence_sha256
    assert all(
        type(value) in (str, int, float, bool, type(None), list) for value in payload.values()
    )
    payload["sample_ids"].append("mutated")
    assert len(first.as_dict()["sample_ids"]) == 120
    with pytest.raises(FrozenInstanceError):
        first.threshold = None  # type: ignore[misc]
    assert not math.isinf(first.risk_bound)


def test_rejects_malformed_map_evidence_digest() -> None:
    result = fit_calibration_maps(_maps(), alpha=0.02, minimum_coverage=0.1)

    with pytest.raises(ValueError, match="map_evidence_sha256"):
        replace(result, map_evidence_sha256="A" * 64)
