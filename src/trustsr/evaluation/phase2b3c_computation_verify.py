"""Independent cache-derived computation replay for Phase 2B3-C."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from trustsr.artifacts.predictions import tensor_sha256
from trustsr.artifacts.scores import ScoreIdentity
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    LoadedCrosssensorPair,
    RadiometricSaturation,
)
from trustsr.evaluation.internal_test_input_receipt import (
    verify_internal_test_input_receipt,
)
from trustsr.evaluation.internal_test_predictions import (
    SEEDS,
    CachedInternalTestPrediction,
    InternalTestPredictionBundle,
)
from trustsr.evaluation.phase2b3c_evidence import (
    INPUT_AUDIT_SHA256,
    PHASE2B3B_FILE_SHA256S,
    PHASE2B3B_PUBLICATION_COMMIT,
)
from trustsr.evaluation.phase2b3c_evidence import (
    PRODUCER_REVISION as PHASE2B3B_PRODUCER_REVISION,
)
from trustsr.evaluation.phase2b3c_policy import (
    PHASE2B3C_ALPHA,
    PHASE2B3C_BISECTION_ITERATIONS,
    PHASE2B3C_DELTA,
    PHASE2B3C_EVALUATION_SIZE,
    PHASE2B3C_GRID_SIZE,
    PHASE2B3C_MINIMUM_COVERAGE,
    PHASE2B3C_RISK_UPPER_BOUND,
    PHASE2B3C_THRESHOLD,
)
from trustsr.evaluation.phase2b3c_result_verify import verify_phase2b3c_result
from trustsr.evaluation.phase2b3c_runtime import verify_phase2b3c_runtime_manifest
from trustsr.jsonio import canonical_json

SCHEMA = "trustsr.phase2b3c-evaluation-computation-verification.v1"
VERIFICATION_SCOPE = "cache_computation_replay"
MAX_DOCUMENT_BYTES = 5 * 1024 * 1024
_RESULT_SCHEMA = "trustsr.phase2b3c-evaluation.v1"
_AUDIT_SCHEMA = "trustsr.phase2b3c-evaluation-cache-audit.v1"
_RUNTIME_SCHEMA = "trustsr.phase2b3c-evaluation-runtime.v1"
_SOURCE = f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}"
_CONTEXT_KEYS = {
    "experiment_schema",
    "split",
    "post_manifest_sha256",
    "input_audit_sha256",
    "normalization_policy",
    "phase2b3b_publication_commit",
    "phase2b3b_acceptance_sha256",
}


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, init=False)
class VerifiedPhase2B3CComputation:
    """Positive cache replay result with deliberately narrow authority."""

    schema: str
    verification_scope: str
    cache_computation_verified: bool
    prediction_inference_verified: bool
    membership_authority_verified: bool
    acceptance_authorized: bool
    result_sha256: str
    cache_audit_sha256: str
    runtime_sha256: str
    map_evidence_sha256: str
    phase_decision: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("computation receipts are created only by the verifier")

    @classmethod
    def _from_verified(cls, **values: object) -> VerifiedPhase2B3CComputation:
        if set(values) != set(cls.__dataclass_fields__):
            raise TypeError("computation receipt fields are invalid")
        receipt = object.__new__(cls)
        for name in cls.__dataclass_fields__:
            object.__setattr__(receipt, name, values[name])
        receipt.__post_init__()
        return receipt

    def __post_init__(self) -> None:
        if (
            self.schema != SCHEMA
            or self.verification_scope != VERIFICATION_SCOPE
            or self.cache_computation_verified is not True
            or self.prediction_inference_verified is not False
            or self.membership_authority_verified is not False
            or self.acceptance_authorized is not False
            or self.phase_decision
            not in {"confirmed", "empirically_met_but_inconclusive", "failed"}
        ):
            raise ValueError("computation receipt scope is invalid")
        for value in (
            self.result_sha256,
            self.cache_audit_sha256,
            self.runtime_sha256,
            self.map_evidence_sha256,
        ):
            _digest(value, "computation receipt digest")


def _canonical_document(
    value: object, *, schema: str, label: str
) -> tuple[dict[str, object], bytes]:
    if type(value) is not bytes:
        raise TypeError(f"committed {label} must be immutable bytes")
    if len(value) > MAX_DOCUMENT_BYTES:
        raise ValueError(f"committed {label} exceeds the 5 MiB limit")
    try:
        parsed = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"committed {label} is not canonical UTF-8 JSON") from exc
    if (
        type(parsed) is not dict
        or parsed.get("schema") != schema
        or canonical_json(parsed) != value
    ):
        raise ValueError(f"committed {label} schema or canonical bytes are invalid")
    return parsed, value


def _snapshot_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    try:
        parsed = json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be canonical JSON data") from exc
    if type(parsed) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _verify_input_receipt(
    receipt: dict[str, object], pairs: tuple[LoadedCrosssensorPair, ...]
) -> None:
    verified = verify_internal_test_input_receipt(receipt)
    if len(pairs) != PHASE2B3C_EVALUATION_SIZE:
        raise ValueError("computation replay requires exactly 120 pairs")
    samples = receipt["samples"]
    for pair, sample in zip(pairs, samples, strict=True):
        if type(pair) is not LoadedCrosssensorPair or type(sample) is not dict:
            raise TypeError("computation replay input pair or receipt sample is invalid")
        pair.pair.validate()
        membership = sample["membership"]
        lr = sample["lr"]
        hr = sample["hr"]
        metadata = pair.metadata
        expected_metadata = {
            "sample_id": metadata.sample_id,
            "selection_sha256": membership["selection_sha256"],
            "spatial_group_id": metadata.spatial_group_id,
            "lr_asset_sha256": metadata.lr_asset_sha256,
            "hr_asset_sha256": metadata.hr_asset_sha256,
            "days_between": metadata.days_between,
            "correlation_bin": metadata.correlation_bin,
            "selection_round": metadata.selection_round,
        }
        if (
            pair.pair.sample_id != metadata.sample_id
            or pair.pair.source != _SOURCE
            or metadata.split != "internal_test"
            or metadata.manifest_sha256 != POST_MANIFEST_SHA256
            or metadata.normalization_policy != PHASE2B3A_NORMALIZATION_POLICY
            or metadata.crop_policy != CROP_POLICY
            or any(membership[key] != item for key, item in expected_metadata.items())
            or lr["tensor_sha256"] != tensor_sha256(pair.pair.lr)
            or hr["tensor_sha256"] != tensor_sha256(pair.pair.hr)
            or lr["shape"] != list(pair.pair.lr.shape)
            or hr["shape"] != list(pair.pair.hr.shape)
        ):
            raise ValueError("loaded pairs differ from the committed input receipt")
    if hashlib.sha256(canonical_json(receipt)).hexdigest() != verified.source_sha256:
        raise RuntimeError("input receipt digest invariant was violated")


def _score_and_risk(
    pair: LoadedCrosssensorPair, bundle: InternalTestPredictionBundle
) -> tuple[torch.Tensor, torch.Tensor, ScoreIdentity]:
    if type(bundle) is not InternalTestPredictionBundle:
        raise TypeError("computation replay requires exact prediction bundles")
    bundle.__post_init__()
    if bundle.sample_id != pair.pair.sample_id:
        raise ValueError("prediction bundle order differs from loaded pairs")
    for item in bundle.items:
        if type(item) is not CachedInternalTestPrediction:
            raise TypeError("prediction bundle contains an invalid item")
        item.__post_init__()
        if (
            item.identity.sample_id != pair.pair.sample_id
            or item.identity.source != pair.pair.source
            or item.identity.lr_shape != tuple(pair.pair.lr.shape)
            or item.identity.lr_dtype != str(pair.pair.lr.dtype)
            or item.identity.lr_sha256 != tensor_sha256(pair.pair.lr)
        ):
            raise ValueError("prediction identity differs from the loaded LR input")
    stacked = torch.stack([item.tensor for item in bundle.items], dim=0).to(torch.float64)
    score = stacked.var(dim=0, correction=0).mean(dim=0)
    central = bundle.items[0].tensor.to(torch.float64)
    error = (central - pair.pair.hr.to(torch.float64)).abs().mean(dim=0)
    padded = F.pad(error[None, None], (4, 4, 4, 4), mode="reflect")
    risk = F.avg_pool2d(padded, kernel_size=9, stride=1)[0, 0]
    prediction_sha256s = tuple(item.prediction_sha256 for item in bundle.items)
    score_identity = ScoreIdentity(
        score_name="ldsr_variance_k5",
        score_schema_version=1,
        sample_id=pair.pair.sample_id,
        input_sha256s=prediction_sha256s,
        operator_parameters={
            "algorithm": "ensemble_variance_score",
            "band_reduction": "mean",
            "correction": 0,
            "seed_first": 3407,
            "seed_last": 3411,
            "seed_count": 5,
            "lr_sha256": tensor_sha256(pair.pair.lr),
            "source": _SOURCE,
            "post_manifest_sha256": POST_MANIFEST_SHA256,
            "input_audit_sha256": INPUT_AUDIT_SHA256,
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "crop_policy": CROP_POLICY,
            "phase2b3b_publication_commit": PHASE2B3B_PUBLICATION_COMMIT,
            "phase2b3b_acceptance_sha256": PHASE2B3B_FILE_SHA256S[
                "sen2naipv2-calibration-conformal-acceptance-v1.json"
            ],
            "phase2b3b_producer_revision": PHASE2B3B_PRODUCER_REVISION,
        },
    )
    return score.contiguous(), risk.contiguous(), score_identity


def _audit_and_observations(
    pairs: tuple[LoadedCrosssensorPair, ...],
    bundles: tuple[InternalTestPredictionBundle, ...],
) -> tuple[dict[str, object], tuple[dict[str, object], ...], str]:
    samples = []
    observations = []
    map_entries = []
    for pair, bundle in zip(pairs, bundles, strict=True):
        score, risk, score_identity = _score_and_risk(pair, bundle)
        score_digest = tensor_sha256(score)
        risk_digest = tensor_sha256(risk)
        samples.append(
            {
                "sample_id": pair.pair.sample_id,
                "predictions": [
                    {
                        "model_name": item.model_name,
                        "seed": item.seed,
                        "cache_key": item.identity.key,
                        "identity": item.identity.as_dict(),
                        "prediction_sha256": item.prediction_sha256,
                    }
                    for item in bundle.items
                ],
                "score": {
                    "name": "ldsr_variance_k5",
                    "cache_key": score_identity.key,
                    "identity": score_identity.as_dict(),
                    "score_sha256": score_digest,
                },
                "risk": {
                    "name": "local_l1_risk",
                    "window": 9,
                    "risk_sha256": risk_digest,
                },
            }
        )
        trusted = score <= PHASE2B3C_THRESHOLD
        trusted_pixels = int(torch.count_nonzero(trusted).item())
        total_pixels = score.numel()
        observations.append(
            {
                "days_between": pair.metadata.days_between,
                "correlation_bin": pair.metadata.correlation_bin,
                "selection_round": pair.metadata.selection_round,
                "trusted_pixels": trusted_pixels,
                "total_pixels": total_pixels,
                "coverage": trusted_pixels / total_pixels,
                "loss": float(torch.max(risk[trusted]).item()) if trusted_pixels else 0.0,
            }
        )
        map_entries.append(
            {
                "sample_id": pair.pair.sample_id,
                "score_sha256": score_digest,
                "risk_sha256": risk_digest,
            }
        )
    sample_ids = [pair.pair.sample_id for pair in pairs]
    audit = {
        "schema": _AUDIT_SCHEMA,
        "split": "internal_test",
        "ordered_sample_ids_sha256": hashlib.sha256(
            canonical_json(sample_ids)
        ).hexdigest(),
        "sample_count": 120,
        "prediction_count": 600,
        "score_count": 120,
        "samples": samples,
    }
    map_evidence_sha256 = hashlib.sha256(canonical_json(map_entries)).hexdigest()
    return audit, tuple(observations), map_evidence_sha256


def _log_evalue(losses: tuple[float, ...], mean: float) -> float:
    if mean == 1.0:
        if any(loss < 1.0 for loss in losses):
            return math.inf
        components = [120 * math.log1p(-index / 21) for index in range(1, 21)]
    else:
        components = [
            math.fsum(
                math.log1p(-(index / 21) * (loss - mean) / (1.0 - mean))
                for loss in losses
            )
            for index in range(1, 21)
        ]
    maximum = max(components)
    return maximum + math.log(math.fsum(math.exp(item - maximum) for item in components) / 20)


def _risk_ucb(losses: tuple[float, ...]) -> float:
    rejection = math.log(1.0 / PHASE2B3C_DELTA)
    if _log_evalue(losses, 1.0) < rejection:
        return 1.0
    accepted, rejected = 0.0, 1.0
    for _ in range(PHASE2B3C_BISECTION_ITERATIONS):
        midpoint = (accepted + rejected) / 2.0
        if _log_evalue(losses, midpoint) < rejection:
            accepted = midpoint
        else:
            rejected = midpoint
    return rejected


def _statistics(observations: tuple[dict[str, object], ...]) -> dict[str, object]:
    trusted_pixels = sum(item["trusted_pixels"] for item in observations)
    total_pixels = sum(item["total_pixels"] for item in observations)
    losses = tuple(item["loss"] for item in observations)
    coverage = trusted_pixels / total_pixels
    mean_loss = math.fsum(losses) / 120
    risk_ucb = _risk_ucb(losses)
    empirical_delta = mean_loss - PHASE2B3C_ALPHA
    if coverage < PHASE2B3C_MINIMUM_COVERAGE or mean_loss > PHASE2B3C_ALPHA:
        decision = "failed"
        reasons = [
            reason
            for condition, reason in (
                (coverage < PHASE2B3C_MINIMUM_COVERAGE, "insufficient_coverage"),
                (mean_loss > PHASE2B3C_ALPHA, "empirical_risk_exceeds_target"),
            )
            if condition
        ]
    elif risk_ucb <= PHASE2B3C_ALPHA:
        decision, reasons = "confirmed", []
    else:
        decision = "empirically_met_but_inconclusive"
        reasons = ["finite_sample_upper_bound_exceeds_target"]
    strata = []
    for day in (-1, 0, 1):
        for bin_index in range(4):
            members = tuple(
                item
                for item in observations
                if item["days_between"] == day and item["correlation_bin"] == bin_index
            )
            trusted = sum(item["trusted_pixels"] for item in members)
            total = sum(item["total_pixels"] for item in members)
            stratum_losses = tuple(item["loss"] for item in members)
            strata.append(
                {
                    "days_between": day,
                    "correlation_bin": bin_index,
                    "roi_count": 10,
                    "trusted_pixels": trusted,
                    "total_pixels": total,
                    "coverage": trusted / total,
                    "mean_loss": math.fsum(stratum_losses) / 10,
                    "maximum_loss": max(stratum_losses),
                }
            )
    return {
        "target": {
            "alpha": PHASE2B3C_ALPHA,
            "minimum_coverage": PHASE2B3C_MINIMUM_COVERAGE,
            "risk_upper_bound": PHASE2B3C_RISK_UPPER_BOUND,
        },
        "method": {
            "name": "one_sided_diversified_grid_kelly",
            "confidence_error": PHASE2B3C_DELTA,
            "grid_size": PHASE2B3C_GRID_SIZE,
            "bet_fractions": [index / 21 for index in range(1, 21)],
            "bisection_iterations": PHASE2B3C_BISECTION_ITERATIONS,
        },
        "evaluation_size": 120,
        "trusted_pixels": trusted_pixels,
        "total_pixels": total_pixels,
        "coverage": coverage,
        "mean_loss": mean_loss,
        "empirical_risk_delta": empirical_delta,
        "empirical_risk_violation": max(0.0, empirical_delta),
        "risk_ucb": risk_ucb,
        "finite_sample_margin": risk_ucb - mean_loss,
        "finite_sample_delta": risk_ucb - PHASE2B3C_ALPHA,
        "hoeffding_ucb": min(
            1.0,
            mean_loss
            + math.sqrt(math.log(1.0 / PHASE2B3C_DELTA) / (2 * 120)),
        ),
        "phase_decision": decision,
        "reasons": reasons,
        "strata": strata,
    }


def _radiometry(pairs: tuple[LoadedCrosssensorPair, ...]) -> dict[str, object]:
    result = {}
    for kind in ("lr", "hr"):
        values = []
        for pair in pairs:
            saturation = getattr(pair.metadata, f"{kind}_saturation")
            if type(saturation) is not RadiometricSaturation:
                raise ValueError("loaded pair radiometry is invalid")
            saturation.__post_init__()
            values.append(saturation)
        result[kind] = {
            "raw_crop_minimum": min(item.raw_crop_minimum for item in values),
            "raw_crop_maximum": max(item.raw_crop_maximum for item in values),
            "clipped_high_count": sum(item.clipped_high_count for item in values),
            "clipped_high_by_band": [
                sum(item.clipped_high_by_band[index] for item in values)
                for index in range(4)
            ],
        }
    return result


def _frozen() -> dict[str, object]:
    return {
        "threshold": PHASE2B3C_THRESHOLD,
        "target": {
            "alpha": PHASE2B3C_ALPHA,
            "minimum_coverage": PHASE2B3C_MINIMUM_COVERAGE,
            "risk_upper_bound": PHASE2B3C_RISK_UPPER_BOUND,
        },
        "score": {
            "name": "ldsr_variance_k5",
            "operator_parameters": {
                "algorithm": "ensemble_variance_score",
                "band_reduction": "mean",
                "correction": 0,
                "seed_count": 5,
                "seed_first": 3407,
                "seed_last": 3411,
            },
            "seeds": list(SEEDS),
        },
        "risk": {"name": "local_l1_risk", "window": 9, "upper_bound": 1.0},
        "input": {
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "crop_policy": CROP_POLICY,
            "bands": ["B04", "B03", "B02", "B08"],
            "scale": 4,
        },
    }


def _result(
    committed: dict[str, object],
    statistics: dict[str, object],
    radiometry: dict[str, object],
    input_receipt: dict[str, object],
    audit_sha256: str,
    map_evidence_sha256: str,
) -> dict[str, object]:
    return {
        "schema": _RESULT_SCHEMA,
        "split": "internal_test",
        "verification_scope": "producer_composition_only",
        "acceptance_authorized": False,
        "upstream": {
            "phase2b3b_publication_commit": PHASE2B3B_PUBLICATION_COMMIT,
            "phase2b3b_producer_revision": PHASE2B3B_PRODUCER_REVISION,
            "phase2b3b_evidence_sha256s": dict(PHASE2B3B_FILE_SHA256S),
            "post_manifest_sha256": POST_MANIFEST_SHA256,
            "input_audit_sha256": INPUT_AUDIT_SHA256,
        },
        "producer_revision": committed["producer_revision"],
        "evaluation": committed["evaluation"],
        "frozen": _frozen(),
        "counts": {
            "internal_test": 120,
            "strata": 12,
            "predictions": 600,
            "scores": 120,
        },
        "statistics": statistics,
        "radiometry": radiometry,
        "digests": {
            "input_receipt_sha256": hashlib.sha256(
                canonical_json(input_receipt)
            ).hexdigest(),
            "ordered_inputs_sha256": input_receipt["ordered_inputs_sha256"],
            "ordered_sample_ids_sha256": input_receipt[
                "ordered_sample_ids_sha256"
            ],
            "ordered_membership_sha256": input_receipt[
                "ordered_membership_sha256"
            ],
            "cache_audit_sha256": audit_sha256,
            "map_evidence_sha256": map_evidence_sha256,
        },
        "phase_decision": statistics["phase_decision"],
        "reasons": statistics["reasons"],
    }


def _model_inventory(bundle: InternalTestPredictionBundle) -> dict[str, object]:
    provenance = dict(bundle.items[0].identity.model_provenance)
    identity = {
        key: item
        for key, item in provenance.items()
        if key not in _CONTEXT_KEYS and key != "seed"
    }
    inventory = {**identity, "seeds": list(SEEDS)}
    inventory["scientific_identity_sha256"] = hashlib.sha256(
        canonical_json(inventory)
    ).hexdigest()
    return inventory


def _runtime(
    result: dict[str, object],
    result_payload: bytes,
    bundle: InternalTestPredictionBundle,
    dependencies: dict[str, object],
) -> dict[str, object]:
    digests = result["digests"]
    evaluation = result["evaluation"]
    return {
        "schema": _RUNTIME_SCHEMA,
        "phase": "internal_test_evaluation",
        "verification_scope": "metadata_inventory_only",
        "cache_computation_verified": False,
        "dependencies": dependencies,
        "model_inventory": _model_inventory(bundle),
        "inputs": {
            key: digests[key]
            for key in (
                "input_receipt_sha256",
                "ordered_inputs_sha256",
                "ordered_sample_ids_sha256",
                "ordered_membership_sha256",
            )
        },
        "artifacts": {
            "result_sha256": hashlib.sha256(result_payload).hexdigest(),
            "cache_audit_sha256": digests["cache_audit_sha256"],
            "map_evidence_sha256": digests["map_evidence_sha256"],
        },
        "evaluation": evaluation,
        "revision": {
            "producer_revision": result["producer_revision"],
            "phase2b3b_publication_commit": PHASE2B3B_PUBLICATION_COMMIT,
            "phase2b3b_producer_revision": PHASE2B3B_PRODUCER_REVISION,
        },
    }


def verify_phase2b3c_computation(
    committed_result: bytes,
    committed_cache_audit: bytes,
    committed_runtime: bytes,
    *,
    input_receipt: Mapping[str, object],
    pairs: Sequence[LoadedCrosssensorPair],
    bundles: Sequence[InternalTestPredictionBundle],
    dependencies: Mapping[str, object],
) -> VerifiedPhase2B3CComputation:
    """Recompute every cache-derived scientific value without model inference."""

    result, result_payload = _canonical_document(
        committed_result, schema=_RESULT_SCHEMA, label="result"
    )
    _canonical_document(
        committed_cache_audit, schema=_AUDIT_SCHEMA, label="cache audit"
    )
    _canonical_document(committed_runtime, schema=_RUNTIME_SCHEMA, label="runtime")
    verify_phase2b3c_result(result_payload)
    verify_phase2b3c_runtime_manifest(committed_runtime, result=result_payload)
    receipt = _snapshot_mapping(input_receipt, "input receipt")
    dependency_snapshot = _snapshot_mapping(dependencies, "dependencies")
    if isinstance(pairs, str | bytes) or not isinstance(pairs, Sequence):
        raise TypeError("pairs must be a stable sequence")
    if isinstance(bundles, str | bytes) or not isinstance(bundles, Sequence):
        raise TypeError("bundles must be a stable sequence")
    pair_values = tuple(pairs)
    bundle_values = tuple(bundles)
    if len(pair_values) != 120 or len(bundle_values) != 120:
        raise ValueError("computation replay requires exactly 120 pairs and bundles")
    _verify_input_receipt(receipt, pair_values)
    audit, observations, map_evidence_sha256 = _audit_and_observations(
        pair_values, bundle_values
    )
    audit_payload = canonical_json(audit)
    if audit_payload != committed_cache_audit:
        raise ValueError("recomputed cache audit is not byte-identical")
    statistics = _statistics(observations)
    rebuilt_result = _result(
        result,
        statistics,
        _radiometry(pair_values),
        receipt,
        hashlib.sha256(audit_payload).hexdigest(),
        map_evidence_sha256,
    )
    rebuilt_result_payload = canonical_json(rebuilt_result)
    if rebuilt_result_payload != result_payload:
        raise ValueError("recomputed Phase 2B3-C result is not byte-identical")
    rebuilt_runtime = _runtime(
        rebuilt_result,
        rebuilt_result_payload,
        bundle_values[0],
        dependency_snapshot,
    )
    if canonical_json(rebuilt_runtime) != committed_runtime:
        raise ValueError("recomputed Phase 2B3-C runtime projection is not byte-identical")
    return VerifiedPhase2B3CComputation._from_verified(
        schema=SCHEMA,
        verification_scope=VERIFICATION_SCOPE,
        cache_computation_verified=True,
        prediction_inference_verified=False,
        membership_authority_verified=False,
        acceptance_authorized=False,
        result_sha256=hashlib.sha256(result_payload).hexdigest(),
        cache_audit_sha256=hashlib.sha256(audit_payload).hexdigest(),
        runtime_sha256=hashlib.sha256(committed_runtime).hexdigest(),
        map_evidence_sha256=map_evidence_sha256,
        phase_decision=statistics["phase_decision"],
    )
