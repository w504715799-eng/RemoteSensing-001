"""Pure, host-free cache-audit contracts for Phase 2B3-B calibration."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace

import pytest
import torch

from trustsr.artifacts.predictions import PredictionIdentity, tensor_sha256
from trustsr.artifacts.scores import ScoreIdentity
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
)
from trustsr.evaluation import calibration_cache_audit
from trustsr.evaluation.calibration_maps import (
    RISK_NAME,
    RISK_WINDOW,
    SCORE_NAME,
    SCORE_SCHEMA_VERSION,
    CachedCalibrationScore,
    CalibrationMaps,
)
from trustsr.evaluation.calibration_predictions import (
    A2_RESULT_SHA256,
    MODEL_NAME,
    PUBLICATION_COMMIT,
    SEEDS,
    CachedCalibrationPrediction,
    CalibrationPredictionBundle,
    build_cache_provenance,
)
from trustsr.evaluation.phase2b3b_evidence import (
    INPUT_AUDIT_SHA256,
    PRODUCER_REVISION,
)
from trustsr.jsonio import canonical_json

_SOURCE = f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}"


def _score_parameters(lr_sha256: str) -> dict[str, str | int]:
    return {
        "algorithm": "ensemble_variance_score",
        "band_reduction": "mean",
        "correction": 0,
        "seed_first": 3407,
        "seed_last": 3411,
        "seed_count": 5,
        "lr_sha256": lr_sha256,
        "source": _SOURCE,
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": INPUT_AUDIT_SHA256,
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "crop_policy": CROP_POLICY,
        "phase2b3a_publication_commit": PUBLICATION_COMMIT,
        "phase2b3a_a2_result_sha256": A2_RESULT_SHA256,
        "phase2b3a_producer_revision": PRODUCER_REVISION,
    }


def _bundle_and_maps(index: int) -> tuple[CalibrationPredictionBundle, CalibrationMaps]:
    sample_id = f"calibration-{index:03d}"
    lr_sha256 = hashlib.sha256(f"lr:{sample_id}".encode()).hexdigest()
    predictions: list[CachedCalibrationPrediction] = []
    for seed in SEEDS:
        tensor = torch.full(
            (4, 4, 4), seed / 10_000 + index / 1_000_000, dtype=torch.float32
        )
        identity = PredictionIdentity(
            model_provenance=build_cache_provenance(
                {"name": MODEL_NAME, "scale": 4, "seed": seed, "backend": "tiny-cpu-fake"}
            ),
            source=_SOURCE,
            sample_id=sample_id,
            lr_shape=(4, 1, 1),
            lr_dtype="torch.float32",
            lr_sha256=lr_sha256,
        )
        predictions.append(
            CachedCalibrationPrediction(
                model_name=MODEL_NAME,
                seed=seed,
                identity=identity,
                prediction_sha256=tensor_sha256(tensor),
                tensor=tensor,
            )
        )
    bundle = CalibrationPredictionBundle(sample_id=sample_id, items=tuple(predictions))
    prediction_sha256s = tuple(item.prediction_sha256 for item in bundle.items)
    score_tensor = torch.full((4, 4), (index % 10) / 100, dtype=torch.float64)
    score = CachedCalibrationScore(
        name=SCORE_NAME,
        identity=ScoreIdentity(
            score_name=SCORE_NAME,
            score_schema_version=SCORE_SCHEMA_VERSION,
            sample_id=sample_id,
            input_sha256s=prediction_sha256s,
            operator_parameters=_score_parameters(lr_sha256),
        ),
        score_sha256=tensor_sha256(score_tensor),
        tensor=score_tensor,
    )
    risk = torch.full((4, 4), (index % 10) / 20, dtype=torch.float64)
    maps = CalibrationMaps(
        sample_id=sample_id,
        score=score,
        score_prediction_sha256s=prediction_sha256s,
        risk_name=RISK_NAME,
        risk_window=RISK_WINDOW,
        risk_sha256=tensor_sha256(risk),
        risk=risk,
    )
    return bundle, maps


@pytest.fixture
def audit_inputs() -> tuple[
    tuple[CalibrationPredictionBundle, ...], tuple[CalibrationMaps, ...]
]:
    values = tuple(_bundle_and_maps(index) for index in range(120))
    return tuple(value[0] for value in values), tuple(value[1] for value in values)


def test_builds_canonical_host_free_exact_calibration_cache_audit(
    audit_inputs: tuple[
        tuple[CalibrationPredictionBundle, ...], tuple[CalibrationMaps, ...]
    ],
) -> None:
    """Omitting a fixed identity/digest or serializing a tensor/path must break the receipt."""

    bundles, maps = audit_inputs
    first = calibration_cache_audit.build_calibration_cache_audit(bundles, maps)
    second = calibration_cache_audit.build_calibration_cache_audit(bundles, maps)

    assert first["schema"] == "trustsr.phase2b3b-calibration-cache-audit.v1"
    assert first["sample_count"] == 120
    assert first["prediction_count"] == 600
    assert first["score_count"] == 120
    assert len(first["samples"]) == 120
    assert set(first) == {
        "schema",
        "sample_count",
        "prediction_count",
        "score_count",
        "samples",
    }
    assert canonical_json(first) == canonical_json(second)
    assert first is not second
    sample = first["samples"][0]
    assert set(sample) == {"sample_id", "predictions", "score", "risk"}
    assert sample["sample_id"] == "calibration-000"
    assert len(sample["predictions"]) == 5
    assert tuple(entry["seed"] for entry in sample["predictions"]) == SEEDS
    assert all(len(entry["cache_key"]) == 64 for entry in sample["predictions"])
    assert all(len(entry["prediction_sha256"]) == 64 for entry in sample["predictions"])
    assert sample["score"]["name"] == "ldsr_variance_k5"
    assert len(sample["score"]["cache_key"]) == 64
    assert len(sample["score"]["score_sha256"]) == 64
    assert sample["risk"] == {
        "name": "local_l1_risk",
        "window": 9,
        "risk_sha256": maps[0].risk_sha256,
    }
    assert sample["predictions"][0]["identity"]["sample_id"] == "calibration-000"
    assert sample["score"]["identity"]["sample_id"] == "calibration-000"
    assert sample["predictions"][0]["identity"] is not (
        second["samples"][0]["predictions"][0]["identity"]
    )

    encoded = canonical_json(first).decode()
    for forbidden in (
        "path",
        "timestamp",
        "hostname",
        "alpha",
        "coverage",
        "internal_test",
        "hr_sha256",
        "hr_asset",
    ):
        assert forbidden not in encoded
    assert "tensor" not in encoded


def _replace_sequence_item(
    values: Sequence[object], index: int, item: object
) -> tuple[object, ...]:
    result = list(values)
    result[index] = item
    return tuple(result)


@pytest.mark.parametrize("size", (119, 121))
def test_rejects_any_input_count_other_than_exactly_120(
    audit_inputs: tuple[
        tuple[CalibrationPredictionBundle, ...], tuple[CalibrationMaps, ...]
    ],
    size: int,
) -> None:
    """Accepting a partial or appended calibration set would invalidate fixed counts."""

    bundles, maps = audit_inputs
    if size == 119:
        bundle_values, map_values = bundles[:-1], maps[:-1]
    else:
        bundle_values, map_values = (*bundles, bundles[-1]), (*maps, maps[-1])

    with pytest.raises(ValueError, match="120"):
        calibration_cache_audit.build_calibration_cache_audit(bundle_values, map_values)


def test_rejects_duplicate_sample_membership(
    audit_inputs: tuple[
        tuple[CalibrationPredictionBundle, ...], tuple[CalibrationMaps, ...]
    ],
) -> None:
    """Repeating a valid bundle/maps pair cannot stand in for a missing ROI."""

    bundles, maps = audit_inputs
    duplicate_bundles = (*bundles[:-1], bundles[0])
    duplicate_maps = (*maps[:-1], maps[0])

    with pytest.raises(ValueError, match="unique"):
        calibration_cache_audit.build_calibration_cache_audit(
            duplicate_bundles, duplicate_maps
        )


def test_rejects_reordered_or_mismatched_bundle_and_map_sequences(
    audit_inputs: tuple[
        tuple[CalibrationPredictionBundle, ...], tuple[CalibrationMaps, ...]
    ],
) -> None:
    """Pairing maps by set membership instead of input position must fail closed."""

    bundles, maps = audit_inputs
    reordered_maps = (maps[1], maps[0], *maps[2:])

    with pytest.raises(ValueError, match="order|sample"):
        calibration_cache_audit.build_calibration_cache_audit(bundles, reordered_maps)


@pytest.mark.parametrize(
    "fault", ("identity", "prediction_digest", "seed", "score_digest", "risk_digest")
)
def test_reruns_public_value_contracts_against_forged_frozen_state(
    audit_inputs: tuple[
        tuple[CalibrationPredictionBundle, ...], tuple[CalibrationMaps, ...]
    ],
    fault: str,
) -> None:
    """Skipping any nested public contract would expose forged cache evidence."""

    bundles, maps = audit_inputs
    item = bundles[0].items[0]
    if fault == "identity":
        object.__setattr__(item.identity, "sample_id", "forged")
    elif fault == "prediction_digest":
        object.__setattr__(item, "prediction_sha256", "0" * 64)
    elif fault == "seed":
        object.__setattr__(item, "seed", 9999)
    elif fault == "score_digest":
        object.__setattr__(maps[0].score, "score_sha256", "0" * 64)
    else:
        object.__setattr__(maps[0], "risk_sha256", "0" * 64)

    with pytest.raises((TypeError, ValueError), match="identit|digest|seed|K5|risk"):
        calibration_cache_audit.build_calibration_cache_audit(bundles, maps)


def test_rejects_score_lr_identity_not_bound_to_prediction_bundle(
    audit_inputs: tuple[
        tuple[CalibrationPredictionBundle, ...], tuple[CalibrationMaps, ...]
    ],
) -> None:
    """A valid standalone score identity must still bind to this bundle's LR digest."""

    bundles, maps = audit_inputs
    original = maps[0]
    forged_identity = replace(
        original.score.identity,
        operator_parameters={
            **original.score.identity.operator_parameters,
            "lr_sha256": "f" * 64,
        },
    )
    forged_score = CachedCalibrationScore(
        name=original.score.name,
        identity=forged_identity,
        score_sha256=original.score.score_sha256,
        tensor=original.score.tensor,
    )
    forged_maps = CalibrationMaps(
        sample_id=original.sample_id,
        score=forged_score,
        score_prediction_sha256s=original.score_prediction_sha256s,
        risk_name=original.risk_name,
        risk_window=original.risk_window,
        risk_sha256=original.risk_sha256,
        risk=original.risk,
    )

    with pytest.raises(ValueError, match="LR|input"):
        calibration_cache_audit.build_calibration_cache_audit(
            bundles, _replace_sequence_item(maps, 0, forged_maps)
        )
