"""Independent cache-computation replay for Phase 2B3-B calibration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from trustsr.artifacts.predictions import PredictionCache, tensor_sha256
from trustsr.artifacts.scores import ScoreCache
from trustsr.data.crosssensor_pairs import LoadedCrosssensorPair
from trustsr.evaluation.calibration_cache_audit import build_calibration_cache_audit
from trustsr.evaluation.calibration_cache_replay import replay_calibration_caches
from trustsr.evaluation.calibration_fit import fit_calibration_maps
from trustsr.evaluation.calibration_input_receipt import (
    build_calibration_input_receipt,
    verify_calibration_input_receipt,
)
from trustsr.evaluation.calibration_radiometry import build_calibration_radiometry
from trustsr.evaluation.phase2b3b_result import build_phase2b3b_result
from trustsr.evaluation.phase2b3b_revision import Phase2B3BRevision
from trustsr.jsonio import canonical_json
from trustsr.risk.local import ensemble_variance_score

SCHEMA = "trustsr.phase2b3b-calibration-computation-verification.v1"
VERIFICATION_SCOPE = "cache_computation_replay"
MAX_DOCUMENT_BYTES = 5 * 1024 * 1024
_RESULT_SCHEMA = "trustsr.phase2b3b-calibration.v1"
_AUDIT_SCHEMA = "trustsr.phase2b3b-calibration-cache-audit.v1"


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class VerifiedPhase2B3BComputation:
    """Host-free computation receipt over caller-supplied loaded inputs.

    ``cache_computation_verified`` covers cache-derived score, risk, fit, audit,
    and result recomputation. This verifier does not rerun LDSR inference, nor
    independently reload manifest membership, Git, runtime, or bundle authority,
    and therefore cannot authorize B acceptance. Direct construction of this
    value object is not proof that the verifier ran.
    """

    schema: str
    verification_scope: str
    cache_computation_verified: bool
    prediction_inference_verified: bool
    membership_authority_verified: bool
    acceptance_authorized: bool
    result_sha256: str
    cache_audit_sha256: str
    map_evidence_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema != SCHEMA
            or self.verification_scope != VERIFICATION_SCOPE
            or self.cache_computation_verified is not True
            or self.prediction_inference_verified is not False
            or self.membership_authority_verified is not False
            or self.acceptance_authorized is not False
        ):
            raise ValueError("computation verification receipt scope is invalid")
        for label, value in (
            ("result", self.result_sha256),
            ("cache audit", self.cache_audit_sha256),
            ("map evidence", self.map_evidence_sha256),
        ):
            _digest(value, f"{label} digest")

    def as_dict(self) -> dict[str, object]:
        """Return a fresh canonical-JSON-native projection."""

        return {
            "schema": self.schema,
            "verification_scope": self.verification_scope,
            "cache_computation_verified": self.cache_computation_verified,
            "prediction_inference_verified": self.prediction_inference_verified,
            "membership_authority_verified": self.membership_authority_verified,
            "acceptance_authorized": self.acceptance_authorized,
            "result_sha256": self.result_sha256,
            "cache_audit_sha256": self.cache_audit_sha256,
            "map_evidence_sha256": self.map_evidence_sha256,
        }


def _canonical_document(payload: object, *, schema: str, label: str) -> dict[str, object]:
    if type(payload) is not bytes:
        raise TypeError(f"committed {label} must be immutable bytes")
    if len(payload) > MAX_DOCUMENT_BYTES:
        raise ValueError(f"committed {label} exceeds the 5 MiB limit")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"committed {label} is not valid UTF-8 JSON") from exc
    if type(value) is not dict or value.get("schema") != schema:
        raise ValueError(f"committed {label} schema is invalid")
    if canonical_json(value) != payload:
        raise ValueError(f"committed {label} is not canonical JSON")
    return value


def _snapshot_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    try:
        snapshot = json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be canonical JSON data") from exc
    if type(snapshot) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return snapshot


def _records_from_input_receipt(input_receipt: dict[str, object]) -> tuple[dict[str, object], ...]:
    verify_calibration_input_receipt(input_receipt)
    samples = input_receipt.get("samples")
    if type(samples) is not list or len(samples) != 120:
        raise ValueError("input receipt must contain exactly 120 samples")
    records: list[dict[str, object]] = []
    for sample in samples:
        if type(sample) is not dict or type(sample.get("membership")) is not dict:
            raise ValueError("input receipt sample membership is invalid")
        membership = sample["membership"]
        records.append(
            {
                "split": "calibration",
                "sample_id": membership["sample_id"],
                "selection_sha256": membership["selection_sha256"],
                "spatial_group_id": membership["spatial_group_id"],
                "days_between": membership["days_between"],
                "correlation_bin": membership["correlation_bin"],
                "selection_round": membership["selection_round"],
                "lr_asset": {"sha256": membership["lr_asset_sha256"]},
                "hr_asset": {"sha256": membership["hr_asset_sha256"]},
            }
        )
    return tuple(records)


def _verify_loaded_inputs(
    input_receipt: dict[str, object],
    preflight: dict[str, object],
    pairs: Sequence[LoadedCrosssensorPair],
) -> dict[str, object]:
    records = _records_from_input_receipt(input_receipt)
    rebuilt = build_calibration_input_receipt(records, pairs, preflight)
    if canonical_json(rebuilt) != canonical_json(input_receipt):
        raise ValueError("loaded calibration tensors differ from the committed input receipt")
    return rebuilt


def _verify_scores(replayed: object) -> None:
    bundles = replayed.bundles
    maps = replayed.maps
    for bundle, calibration_maps in zip(bundles, maps, strict=True):
        samples = torch.stack([item.tensor for item in bundle.items], dim=0)
        recomputed = ensemble_variance_score(samples)
        if (
            tensor_sha256(recomputed) != calibration_maps.score.score_sha256
            or not torch.equal(recomputed, calibration_maps.score.tensor)
        ):
            raise ValueError("recomputed ensemble score differs from the verified score cache")


def _target(result: dict[str, object]) -> tuple[float, float]:
    target = result.get("target")
    if type(target) is not dict or set(target) != {"alpha", "minimum_coverage"}:
        raise ValueError("committed result target is invalid")
    alpha = target["alpha"]
    minimum_coverage = target["minimum_coverage"]
    if type(alpha) is not float or type(minimum_coverage) is not float:
        raise TypeError("committed result target parameters must be exact floats")
    return alpha, minimum_coverage


def verify_phase2b3b_computation(
    committed_result: bytes,
    committed_cache_audit: bytes,
    *,
    preflight: Mapping[str, object],
    input_receipt: Mapping[str, object],
    radiometry: Mapping[str, object],
    revision: Phase2B3BRevision,
    pairs: Sequence[LoadedCrosssensorPair],
    prediction_cache: PredictionCache,
    score_cache: ScoreCache,
) -> VerifiedPhase2B3BComputation:
    """Replay computations over supplied inputs without establishing external authority.

    The caller must independently obtain the preflight, input receipt, loaded pairs,
    revision, and caches from trusted gates. This function recomputes score, risk,
    fit, audit, and result from cached predictions, but does not rerun LDSR inference
    or reload manifest, Git, runtime, or bundle authority. It never authorizes
    acceptance.
    """

    result = _canonical_document(committed_result, schema=_RESULT_SCHEMA, label="result")
    audit = _canonical_document(
        committed_cache_audit, schema=_AUDIT_SCHEMA, label="cache audit"
    )
    preflight_snapshot = _snapshot_mapping(preflight, "preflight")
    input_snapshot = _snapshot_mapping(input_receipt, "input receipt")
    radiometry_snapshot = _snapshot_mapping(radiometry, "radiometry")
    if isinstance(pairs, str | bytes) or not isinstance(pairs, Sequence):
        raise TypeError("calibration pairs must be a stable sequence")
    pair_snapshot = tuple(pairs)
    rebuilt_input = _verify_loaded_inputs(input_snapshot, preflight_snapshot, pair_snapshot)
    rebuilt_radiometry = build_calibration_radiometry(pair_snapshot)
    if canonical_json(rebuilt_radiometry) != canonical_json(radiometry_snapshot):
        raise ValueError("loaded calibration metadata differs from the radiometry receipt")

    replayed = replay_calibration_caches(
        audit, pair_snapshot, prediction_cache, score_cache
    )
    _verify_scores(replayed)
    if canonical_json(build_calibration_cache_audit(replayed.bundles, replayed.maps)) != (
        committed_cache_audit
    ):
        raise ValueError("recomputed calibration cache audit is not byte-identical")
    alpha, minimum_coverage = _target(result)
    fit = fit_calibration_maps(
        replayed.maps,
        alpha=alpha,
        minimum_coverage=minimum_coverage,
    )
    rebuilt_result = build_phase2b3b_result(
        preflight_snapshot,
        rebuilt_input,
        fit,
        audit,
        rebuilt_radiometry,
        revision,
    )
    if canonical_json(rebuilt_result) != committed_result:
        raise ValueError("recomputed Phase 2B3-B result is not byte-identical")

    return VerifiedPhase2B3BComputation(
        schema=SCHEMA,
        verification_scope=VERIFICATION_SCOPE,
        cache_computation_verified=True,
        prediction_inference_verified=False,
        membership_authority_verified=False,
        acceptance_authorized=False,
        result_sha256=hashlib.sha256(committed_result).hexdigest(),
        cache_audit_sha256=hashlib.sha256(committed_cache_audit).hexdigest(),
        map_evidence_sha256=fit.map_evidence_sha256,
    )
