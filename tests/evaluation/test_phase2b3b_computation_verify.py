"""Independent computation replay for Phase 2B3-B synthetic calibration data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path

import pytest
import test_calibration_cache_replay as replay_fixtures
import test_phase2b3b_result as result_fixtures
import torch

from trustsr.artifacts.predictions import CacheIntegrityError, PredictionCache, tensor_sha256
from trustsr.artifacts.scores import ScoreCache
from trustsr.data.crosssensor_pairs import LoadedCrosssensorPair
from trustsr.evaluation.calibration_cache_audit import build_calibration_cache_audit
from trustsr.evaluation.calibration_cache_replay import replay_calibration_caches
from trustsr.evaluation.calibration_fit import fit_calibration_maps
from trustsr.evaluation.calibration_input_receipt import build_calibration_input_receipt
from trustsr.evaluation.calibration_maps import CachedCalibrationScore
from trustsr.evaluation.calibration_radiometry import build_calibration_radiometry
from trustsr.evaluation.phase2b3b_computation_verify import (
    VerifiedPhase2B3BComputation,
    verify_phase2b3b_computation,
)
from trustsr.evaluation.phase2b3b_result import build_phase2b3b_result
from trustsr.evaluation.phase2b3b_revision import Phase2B3BRevision
from trustsr.jsonio import canonical_json


@dataclass(frozen=True)
class _Case:
    result_bytes: bytes
    audit_bytes: bytes
    preflight: dict[str, object]
    input_receipt: dict[str, object]
    radiometry: dict[str, object]
    revision: Phase2B3BRevision
    pairs: tuple[LoadedCrosssensorPair, ...]
    prediction_cache: PredictionCache
    score_cache: ScoreCache


def _records(pairs: tuple[LoadedCrosssensorPair, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "split": "calibration",
            "sample_id": loaded.metadata.sample_id,
            "selection_sha256": hashlib.sha256(
                f"selection:{loaded.metadata.sample_id}".encode()
            ).hexdigest(),
            "spatial_group_id": loaded.metadata.spatial_group_id,
            "days_between": loaded.metadata.days_between,
            "correlation_bin": loaded.metadata.correlation_bin,
            "selection_round": loaded.metadata.selection_round,
            "lr_asset": {"sha256": loaded.metadata.lr_asset_sha256},
            "hr_asset": {"sha256": loaded.metadata.hr_asset_sha256},
        }
        for loaded in pairs
    )


def _preflight(records: tuple[dict[str, object], ...]) -> dict[str, object]:
    sample_ids = tuple(str(record["sample_id"]) for record in records)
    value = result_fixtures._preflight(sample_ids)
    membership = [
        {
            "sample_id": record["sample_id"],
            "selection_sha256": record["selection_sha256"],
            "spatial_group_id": record["spatial_group_id"],
            "lr_asset_sha256": record["lr_asset"]["sha256"],
            "hr_asset_sha256": record["hr_asset"]["sha256"],
            "days_between": record["days_between"],
            "correlation_bin": record["correlation_bin"],
            "selection_round": record["selection_round"],
        }
        for record in records
    ]
    value["calibration"]["ordered_membership_sha256"] = hashlib.sha256(
        canonical_json(membership)
    ).hexdigest()
    value["calibration"]["input_receipt_sha256s"] = [
        hashlib.sha256(canonical_json(record)).hexdigest() for record in membership
    ]
    return value


@pytest.fixture(scope="module")
def valid_case(tmp_path_factory: pytest.TempPathFactory) -> _Case:
    root = tmp_path_factory.mktemp("phase2b3b-computation")
    audit, pairs, prediction_cache, score_cache = replay_fixtures._prepared(root)
    records = _records(pairs)
    preflight = _preflight(records)
    input_receipt = build_calibration_input_receipt(records, pairs, preflight)
    radiometry = build_calibration_radiometry(pairs)
    revision = result_fixtures._revision()
    replayed = replay_calibration_caches(audit, pairs, prediction_cache, score_cache)
    fit = fit_calibration_maps(replayed.maps, alpha=0.5, minimum_coverage=0.1)
    result = build_phase2b3b_result(
        preflight,
        input_receipt,
        fit,
        audit,
        radiometry,
        revision,
    )
    return _Case(
        result_bytes=canonical_json(result),
        audit_bytes=canonical_json(audit),
        preflight=preflight,
        input_receipt=input_receipt,
        radiometry=radiometry,
        revision=revision,
        pairs=pairs,
        prediction_cache=prediction_cache,
        score_cache=score_cache,
    )


def _verify(
    case: _Case,
    *,
    result_bytes: bytes | None = None,
    audit_bytes: bytes | None = None,
    pairs: tuple[LoadedCrosssensorPair, ...] | None = None,
    score_cache: ScoreCache | None = None,
) -> VerifiedPhase2B3BComputation:
    return verify_phase2b3b_computation(
        case.result_bytes if result_bytes is None else result_bytes,
        case.audit_bytes if audit_bytes is None else audit_bytes,
        preflight=case.preflight,
        input_receipt=case.input_receipt,
        radiometry=case.radiometry,
        revision=case.revision,
        pairs=case.pairs if pairs is None else pairs,
        prediction_cache=case.prediction_cache,
        score_cache=case.score_cache if score_cache is None else score_cache,
    )


def test_recomputes_caches_fit_and_canonical_result_into_a_non_authorizing_receipt(
    valid_case: _Case,
) -> None:
    receipt = _verify(valid_case)

    assert receipt.schema == "trustsr.phase2b3b-calibration-computation-verification.v1"
    assert receipt.verification_scope == "cache_computation_replay"
    assert receipt.cache_computation_verified is True
    assert receipt.prediction_inference_verified is False
    assert receipt.membership_authority_verified is False
    assert receipt.acceptance_authorized is False
    assert receipt.result_sha256 == hashlib.sha256(valid_case.result_bytes).hexdigest()
    assert receipt.cache_audit_sha256 == hashlib.sha256(valid_case.audit_bytes).hexdigest()
    assert str(valid_case.prediction_cache.root) not in repr(receipt)
    assert "internal_test" not in canonical_json(receipt.as_dict()).decode()
    with pytest.raises(FrozenInstanceError):
        receipt.acceptance_authorized = True  # type: ignore[misc]


@pytest.mark.parametrize("fault", ("missing_metadata", "tensor_bytes"))
def test_rejects_missing_or_tampered_prediction_cache_entry(
    valid_case: _Case, fault: str
) -> None:
    audit = json.loads(valid_case.audit_bytes)
    key = audit["samples"][0]["predictions"][0]["cache_key"]
    tensor_path = valid_case.prediction_cache.root / f"{key}.safetensors"
    metadata_path = valid_case.prediction_cache.root / f"{key}.json"
    path = metadata_path if fault == "missing_metadata" else tensor_path
    original = path.read_bytes()
    try:
        if fault == "missing_metadata":
            path.unlink()
        else:
            path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with pytest.raises(CacheIntegrityError):
            _verify(valid_case)
    finally:
        path.write_bytes(original)


def test_rejects_self_consistent_forged_score_cache_and_audit(
    valid_case: _Case, tmp_path: Path
) -> None:
    audit = json.loads(valid_case.audit_bytes)
    replayed = replay_calibration_caches(
        audit,
        valid_case.pairs,
        valid_case.prediction_cache,
        valid_case.score_cache,
    )
    forged_cache = ScoreCache(tmp_path / "forged-scores")
    forged_maps = list(replayed.maps)
    for index, maps in enumerate(forged_maps):
        score = maps.score.tensor
        if index == 0:
            score = torch.zeros_like(score)
        forged_cache.put(maps.score.identity, score)
        if index == 0:
            forged_score = CachedCalibrationScore(
                name=maps.score.name,
                identity=maps.score.identity,
                score_sha256=tensor_sha256(score),
                tensor=score,
            )
            forged_maps[index] = dataclass_replace(maps, score=forged_score)
    forged_audit = build_calibration_cache_audit(replayed.bundles, tuple(forged_maps))
    forged_fit = fit_calibration_maps(
        tuple(forged_maps), alpha=0.5, minimum_coverage=0.1
    )
    forged_result = build_phase2b3b_result(
        valid_case.preflight,
        valid_case.input_receipt,
        forged_fit,
        forged_audit,
        valid_case.radiometry,
        valid_case.revision,
    )

    with pytest.raises(ValueError, match="recomputed ensemble score"):
        _verify(
            valid_case,
            result_bytes=canonical_json(forged_result),
            audit_bytes=canonical_json(forged_audit),
            score_cache=forged_cache,
        )


def test_rejects_changed_loaded_tensor_and_recomputed_risk(valid_case: _Case) -> None:
    changed_pair = dataclass_replace(
        valid_case.pairs[0].pair,
        hr=torch.zeros_like(valid_case.pairs[0].pair.hr),
    )
    changed_loaded = dataclass_replace(valid_case.pairs[0], pair=changed_pair)

    with pytest.raises(ValueError, match="input receipt|risk"):
        _verify(valid_case, pairs=(changed_loaded, *valid_case.pairs[1:]))


@pytest.mark.parametrize("fault", ("result", "alpha", "coverage"))
def test_rejects_tampered_result_calculation_fields(valid_case: _Case, fault: str) -> None:
    result = json.loads(valid_case.result_bytes)
    if fault == "result":
        result["threshold"] = 0.123
    elif fault == "alpha":
        result["target"]["alpha"] = 0.01
    else:
        result["coverage"] = 0.5

    with pytest.raises(ValueError, match="recomputed.*result"):
        _verify(valid_case, result_bytes=canonical_json(result))


@pytest.mark.parametrize("fault", ("audit", "risk", "model_seed"))
def test_rejects_tampered_audit_and_model_seed_identity(valid_case: _Case, fault: str) -> None:
    audit = json.loads(valid_case.audit_bytes)
    prediction = audit["samples"][0]["predictions"][0]
    if fault == "audit":
        prediction["prediction_sha256"] = "f" * 64
    elif fault == "risk":
        audit["samples"][0]["risk"]["risk_sha256"] = "f" * 64
    else:
        prediction["seed"] = 3408

    with pytest.raises(ValueError):
        _verify(valid_case, audit_bytes=canonical_json(audit))


def test_rejects_loaded_sample_order_change(valid_case: _Case) -> None:
    reordered = (valid_case.pairs[1], valid_case.pairs[0], *valid_case.pairs[2:])

    with pytest.raises(ValueError, match="order|membership|identity"):
        _verify(valid_case, pairs=reordered)
