"""Deterministic four-ROI Phase 2B3-A score audit and cache-only replay."""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from trustsr.artifacts.predictions import (
    PredictionCache,
    PredictionIdentity,
    tensor_sha256,
)
from trustsr.artifacts.scores import (
    ScoreCache,
    ScoreIdentity,
    score_entry_evidence,
)
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    LoadedCrosssensorPair,
)
from trustsr.evaluation.crosssensor_smoke import INPUT_AUDIT_SHA256, snapshot_cache_files
from trustsr.evaluation.development_predictions import (
    A1_SEEDS,
    K5A_SEEDS,
    K5B_SEEDS,
    CachedDevelopmentPrediction,
    DevelopmentPredictionBundle,
)
from trustsr.evaluation.development_predictions import (
    EXPERIMENT_SCHEMA as PREDICTION_EXPERIMENT_SCHEMA,
)
from trustsr.evaluation.score_diagnostics import (
    RoiScoreDiagnostics,
    evaluate_roi_score,
    score_map_spearman,
    top_fraction_jaccard,
)
from trustsr.jsonio import canonical_json
from trustsr.risk.local import ensemble_variance_score, local_l1_risk
from trustsr.risk.proxies import (
    lr_reprojection_l1_score,
    three_model_disagreement_score,
)

A1_RESULT_SCHEMA = "trustsr.phase2b3a-development-smoke.v1"
A1_CACHE_AUDIT_SCHEMA = "trustsr.phase2b3a-development-smoke-cache-audit.v1"
PRIMARY_RISK_WINDOW = 9
SENSITIVITY_RISK_WINDOW = 1
SCORE_SCHEMA_VERSION = 1
SCORE_NAMES = (
    "ldsr_variance_k5a",
    "ldsr_variance_k5b",
    "ldsr_variance_k25",
    "lr_reprojection_l1",
    "three_model_disagreement",
)
_MODEL_SEED_SLOTS = (
    ("bicubic-x4", None),
    ("sen2srlite-x4", None),
    *(("ldsr-s2-x4", seed) for seed in A1_SEEDS),
)
_STABILITY_THRESHOLDS = {
    "k5a_k5b_median_minimum": 0.60,
    "k5a_k5b_worst_minimum": 0.40,
    "k5a_k25_median_minimum": 0.80,
    "k5a_k25_worst_minimum": 0.60,
    "k5a_k25_top10_jaccard_median_minimum": 0.50,
}


@dataclass(frozen=True)
class CachedScoreMap:
    """One verified score map and its cache-bound identity."""

    name: str
    identity: ScoreIdentity
    score_sha256: str
    tensor: torch.Tensor = field(compare=False, repr=False)


def _validate_pair(pair: LoadedCrosssensorPair, *, expected_bin: int | None = None) -> None:
    if not isinstance(pair, LoadedCrosssensorPair):
        raise TypeError("A1 inputs must be LoadedCrosssensorPair values")
    pair.pair.validate()
    metadata = pair.metadata
    if metadata.split != "development":
        raise ValueError("A1 inputs must use the development split")
    if metadata.days_between != -1 or metadata.selection_round != 1:
        raise ValueError("A1 pair is outside the canonical smoke selection")
    if metadata.correlation_bin not in range(4):
        raise ValueError("A1 pair has an invalid correlation bin")
    if expected_bin is not None and metadata.correlation_bin != expected_bin:
        raise ValueError("A1 pairs must use canonical correlation-bin order")
    if metadata.manifest_sha256 != POST_MANIFEST_SHA256:
        raise ValueError("A1 pair has the wrong manifest")
    if metadata.sample_id != pair.pair.sample_id:
        raise ValueError("A1 pair and metadata identities differ")
    if pair.pair.source != f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}":
        raise ValueError("A1 pair has the wrong source identity")
    if metadata.crop_policy != CROP_POLICY or metadata.normalization_policy != NORMALIZATION_POLICY:
        raise ValueError("A1 pair has the wrong input policy")


def _validate_prediction_tensor(
    pair: LoadedCrosssensorPair,
    item: CachedDevelopmentPrediction,
    *,
    expected_name: str,
    expected_seed: int | None,
) -> None:
    if not isinstance(item, CachedDevelopmentPrediction):
        raise TypeError("A1 prediction entries must be cached prediction records")
    if item.model_name != expected_name or item.seed != expected_seed:
        raise ValueError("A1 prediction model/seed order is invalid")
    identity = item.identity
    if not isinstance(identity, PredictionIdentity):
        raise TypeError("A1 prediction identity is invalid")
    if (
        identity.source != pair.pair.source
        or identity.sample_id != pair.pair.sample_id
        or identity.lr_shape != tuple(pair.pair.lr.shape)
        or identity.lr_dtype != str(pair.pair.lr.dtype)
        or identity.lr_sha256 != tensor_sha256(pair.pair.lr)
    ):
        raise ValueError("A1 prediction identity does not match its pair")
    provenance = identity.model_provenance
    if (
        provenance.get("name") != expected_name
        or provenance.get("scale") != 4
        or provenance.get("experiment_schema") != PREDICTION_EXPERIMENT_SCHEMA
        or provenance.get("post_manifest_sha256") != POST_MANIFEST_SHA256
        or provenance.get("input_audit_sha256") != INPUT_AUDIT_SHA256
    ):
        raise ValueError("A1 prediction provenance is invalid")
    if expected_seed is None:
        if "seed" in provenance:
            raise ValueError("A1 deterministic prediction must not have a seed")
    elif provenance.get("seed") != expected_seed:
        raise ValueError("A1 LDSR prediction has the wrong seed provenance")
    tensor = item.tensor
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.dtype != torch.float32
        or tuple(tensor.shape) != tuple(pair.pair.hr.shape)
        or tensor.device.type != "cpu"
        or not tensor.is_contiguous()
        or tensor.requires_grad
        or not torch.isfinite(tensor).all()
        or (tensor < 0).any()
        or (tensor > 1).any()
    ):
        raise ValueError("A1 cached prediction tensor is invalid")
    if item.prediction_sha256 != tensor_sha256(tensor):
        raise ValueError("A1 cached prediction logical SHA-256 is invalid")


def _validate_bundle(pair: LoadedCrosssensorPair, bundle: DevelopmentPredictionBundle) -> None:
    if not isinstance(bundle, DevelopmentPredictionBundle):
        raise TypeError("A1 bundles must be DevelopmentPredictionBundle values")
    if bundle.sample_id != pair.pair.sample_id:
        raise ValueError("A1 bundle membership does not match pair order")
    if tuple(item.seed for item in bundle.ldsr) != A1_SEEDS:
        raise ValueError("A1 bundle must contain the exact ordered 25-seed grid")
    items = (bundle.bicubic, bundle.sen2srlite, *bundle.ldsr)
    for item, (name, seed) in zip(items, _MODEL_SEED_SLOTS, strict=True):
        _validate_prediction_tensor(pair, item, expected_name=name, expected_seed=seed)


def _validate_a1_inputs(
    pairs: Sequence[LoadedCrosssensorPair],
    bundles: Sequence[DevelopmentPredictionBundle],
) -> tuple[tuple[LoadedCrosssensorPair, DevelopmentPredictionBundle], ...]:
    pair_values = tuple(pairs)
    bundle_values = tuple(bundles)
    if len(pair_values) != 4 or len(bundle_values) != 4:
        raise ValueError("A1 requires exactly four canonical pairs and bundles")
    for bin_index, (pair, bundle) in enumerate(zip(pair_values, bundle_values, strict=True)):
        _validate_pair(pair, expected_bin=bin_index)
        _validate_bundle(pair, bundle)
    if len({pair.metadata.sample_id for pair in pair_values}) != 4:
        raise ValueError("A1 requires four distinct sample identities")
    if len({pair.metadata.spatial_group_id for pair in pair_values}) != 4:
        raise ValueError("A1 requires four distinct spatial groups")
    return tuple(zip(pair_values, bundle_values, strict=True))


def _score_identity(
    pair: LoadedCrosssensorPair,
    *,
    name: str,
    inputs: Sequence[CachedDevelopmentPrediction],
    parameters: Mapping[str, str | int | float | bool | None],
) -> ScoreIdentity:
    return ScoreIdentity(
        score_name=name,
        score_schema_version=SCORE_SCHEMA_VERSION,
        sample_id=pair.pair.sample_id,
        input_sha256s=tuple(item.prediction_sha256 for item in inputs),
        operator_parameters={
            **parameters,
            "lr_sha256": tensor_sha256(pair.pair.lr),
        },
    )


def _load_or_compute_score(
    name: str,
    identity: ScoreIdentity,
    cache: ScoreCache,
    compute: Callable[[], torch.Tensor],
) -> CachedScoreMap:
    score = cache.get(identity)
    if score is None:
        produced = compute()
        produced_sha256 = tensor_sha256(produced)
        cache.put(identity, produced)
        score = cache.get(identity)
        if score is None or tensor_sha256(score) != produced_sha256:
            raise RuntimeError("score differs after cache commit")
    score_sha256 = tensor_sha256(score)
    return CachedScoreMap(name, identity, score_sha256, score)


def build_a1_score_maps(
    pair: LoadedCrosssensorPair,
    bundle: DevelopmentPredictionBundle,
    score_cache: ScoreCache,
) -> tuple[CachedScoreMap, ...]:
    """Load or compute the five frozen A1 score maps for one canonical ROI."""

    _validate_pair(pair)
    _validate_bundle(pair, bundle)
    central = bundle.ldsr_for_seed(3407)
    k5a = tuple(bundle.ldsr_for_seed(seed) for seed in K5A_SEEDS)
    k5b = tuple(bundle.ldsr_for_seed(seed) for seed in K5B_SEEDS)
    k25 = tuple(bundle.ldsr_for_seed(seed) for seed in A1_SEEDS)
    specifications: tuple[
        tuple[
            str,
            tuple[CachedDevelopmentPrediction, ...],
            dict[str, str | int | float | bool | None],
            Callable[[], torch.Tensor],
        ],
        ...,
    ] = (
        (
            SCORE_NAMES[0],
            k5a,
            {
                "algorithm": "ensemble_variance_score",
                "band_reduction": "mean",
                "correction": 0,
                "seed_first": K5A_SEEDS[0],
                "seed_last": K5A_SEEDS[-1],
                "seed_count": len(K5A_SEEDS),
            },
            lambda: ensemble_variance_score(torch.stack([item.tensor for item in k5a], dim=0)),
        ),
        (
            SCORE_NAMES[1],
            k5b,
            {
                "algorithm": "ensemble_variance_score",
                "band_reduction": "mean",
                "correction": 0,
                "seed_first": K5B_SEEDS[0],
                "seed_last": K5B_SEEDS[-1],
                "seed_count": len(K5B_SEEDS),
            },
            lambda: ensemble_variance_score(torch.stack([item.tensor for item in k5b], dim=0)),
        ),
        (
            SCORE_NAMES[2],
            k25,
            {
                "algorithm": "ensemble_variance_score",
                "band_reduction": "mean",
                "correction": 0,
                "seed_first": A1_SEEDS[0],
                "seed_last": A1_SEEDS[-1],
                "seed_count": len(A1_SEEDS),
            },
            lambda: ensemble_variance_score(torch.stack([item.tensor for item in k25], dim=0)),
        ),
        (
            SCORE_NAMES[3],
            (central,),
            {
                "algorithm": "lr_reprojection_l1_score",
                "downsample_mode": "area",
                "scale": 4,
                "upsample_mode": "repeat_interleave",
            },
            lambda: lr_reprojection_l1_score(central.tensor, pair.pair.lr, scale=4),
        ),
        (
            SCORE_NAMES[4],
            (bundle.bicubic, bundle.sen2srlite, central),
            {
                "algorithm": "three_model_disagreement_score",
                "band_reduction": "mean",
                "correction": 0,
                "model_order": "bicubic-x4,sen2srlite-x4,ldsr-s2-x4",
            },
            lambda: three_model_disagreement_score(
                (bundle.bicubic.tensor, bundle.sen2srlite.tensor, central.tensor)
            ),
        ),
    )
    return tuple(
        _load_or_compute_score(
            name,
            _score_identity(pair, name=name, inputs=inputs, parameters=parameters),
            score_cache,
            compute,
        )
        for name, inputs, parameters, compute in specifications
    )


def _diagnostic_payload(diagnostic: RoiScoreDiagnostics) -> dict[str, object]:
    payload = asdict(diagnostic)
    payload["coverages"] = list(diagnostic.coverages)
    payload["selective_mean_risks"] = list(diagnostic.selective_mean_risks)
    return payload


def _evaluate_a1_sample(
    pair: LoadedCrosssensorPair,
    bundle: DevelopmentPredictionBundle,
    score_cache: ScoreCache,
) -> dict[str, object]:
    scores = build_a1_score_maps(pair, bundle, score_cache)
    central = bundle.ldsr_for_seed(3407)
    primary = local_l1_risk(central.tensor, pair.pair.hr, window=PRIMARY_RISK_WINDOW)
    sensitivity = local_l1_risk(central.tensor, pair.pair.hr, window=SENSITIVITY_RISK_WINDOW)
    score_records = [
        {
            "name": score.name,
            "cache_key": score.identity.key,
            "score_sha256": score.score_sha256,
            "primary_window_9": _diagnostic_payload(evaluate_roi_score(score.tensor, primary)),
            "sensitivity_window_1": _diagnostic_payload(
                evaluate_roi_score(score.tensor, sensitivity)
            ),
        }
        for score in scores
    ]
    return {
        "sample_id": pair.metadata.sample_id,
        "spatial_group_id": pair.metadata.spatial_group_id,
        "correlation_bin": pair.metadata.correlation_bin,
        "days_between": pair.metadata.days_between,
        "selection_round": pair.metadata.selection_round,
        "lr_tensor_sha256": tensor_sha256(pair.pair.lr),
        "hr_tensor_sha256": tensor_sha256(pair.pair.hr),
        "central_prediction_sha256": central.prediction_sha256,
        "risks": {
            "primary": {
                "name": "local_l1_risk",
                "window": PRIMARY_RISK_WINDOW,
                "risk_sha256": tensor_sha256(primary),
            },
            "sensitivity": {
                "name": "local_l1_risk",
                "window": SENSITIVITY_RISK_WINDOW,
                "risk_sha256": tensor_sha256(sensitivity),
            },
        },
        "stability": {
            "k5a_k5b_spearman": score_map_spearman(scores[0].tensor, scores[1].tensor),
            "k5a_k25_spearman": score_map_spearman(scores[0].tensor, scores[2].tensor),
            "k5a_k25_top10_jaccard": top_fraction_jaccard(
                scores[0].tensor, scores[2].tensor, fraction=0.10
            ),
            "k5a_constant_score": bool(score_records[0]["primary_window_9"]["constant_score"]),
            "k5b_constant_score": bool(score_records[1]["primary_window_9"]["constant_score"]),
            "k25_constant_score": bool(score_records[2]["primary_window_9"]["constant_score"]),
        },
        "scores": score_records,
    }


def _is_k5_statistically_stable(
    k5a_k5b: Sequence[float],
    k5a_k25: Sequence[float],
    jaccards: Sequence[float],
) -> bool:
    return bool(
        statistics.median(k5a_k5b) >= 0.60
        and min(k5a_k5b) >= 0.40
        and statistics.median(k5a_k25) >= 0.80
        and min(k5a_k25) >= 0.60
        and statistics.median(jaccards) >= 0.50
    )


def _a1_result_payload(
    sample_records: Sequence[dict[str, object]],
    *,
    k5_statistically_stable: bool,
) -> dict[str, object]:
    return {
        "schema": A1_RESULT_SCHEMA,
        "dataset_role": "development_engineering_smoke_only",
        "upstream": {
            "post_manifest_sha256": POST_MANIFEST_SHA256,
            "input_audit_sha256": INPUT_AUDIT_SHA256,
        },
        "bands": ["B04", "B03", "B02", "B08"],
        "scale": 4,
        "sample_count": 4,
        "prediction_count": 4 * len(_MODEL_SEED_SLOTS),
        "score_count": 4 * len(SCORE_NAMES),
        "seed_sets": {
            "k5a": list(K5A_SEEDS),
            "k5b": list(K5B_SEEDS),
            "k25": list(A1_SEEDS),
        },
        "stability_thresholds": dict(_STABILITY_THRESHOLDS),
        "k5_statistically_stable": k5_statistically_stable,
        "include_ldsr_variance_k5": k5_statistically_stable,
        "samples": list(sample_records),
    }


def _file_size(root: Path, filename: str) -> int:
    path = root / filename
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("score cache evidence file is invalid")
    return path.stat().st_size


def _score_evidence(
    pair: LoadedCrosssensorPair, score: CachedScoreMap, score_cache: ScoreCache
) -> dict[str, object]:
    evidence = score_entry_evidence(score_cache.root, score.identity)
    loaded = score_cache.get(score.identity)
    if loaded is None or tensor_sha256(loaded) != score.score_sha256:
        raise RuntimeError("score cache evidence differs from evaluated score")
    files = [
        {
            "filename": evidence[kind]["filename"],
            "size_bytes": _file_size(score_cache.root, evidence[kind]["filename"]),
            "sha256": evidence[kind]["sha256"],
        }
        for kind in ("json", "safetensors")
    ]
    return {
        "sample_id": pair.pair.sample_id,
        "correlation_bin": pair.metadata.correlation_bin,
        "name": score.name,
        "cache_key": score.identity.key,
        "identity": score.identity.as_dict(),
        "score_sha256": score.score_sha256,
        "files": files,
    }


def _prediction_evidence(
    pair: LoadedCrosssensorPair, item: CachedDevelopmentPrediction
) -> dict[str, object]:
    return {
        "sample_id": pair.pair.sample_id,
        "correlation_bin": pair.metadata.correlation_bin,
        "model_name": item.model_name,
        "seed": item.seed,
        "cache_key": item.identity.key,
        "identity": item.identity.as_dict(),
        "prediction_sha256": item.prediction_sha256,
    }


def _score_and_prediction_evidence_payload(
    validated: Sequence[tuple[LoadedCrosssensorPair, DevelopmentPredictionBundle]],
    score_cache: ScoreCache,
) -> dict[str, object]:
    prediction_entries: list[dict[str, object]] = []
    score_entries: list[dict[str, object]] = []
    for pair, bundle in validated:
        prediction_entries.extend(
            _prediction_evidence(pair, item)
            for item in (bundle.bicubic, bundle.sen2srlite, *bundle.ldsr)
        )
        score_entries.extend(
            _score_evidence(pair, score, score_cache)
            for score in build_a1_score_maps(pair, bundle, score_cache)
        )
    return {
        "schema": A1_CACHE_AUDIT_SCHEMA,
        "experiment_schema": A1_RESULT_SCHEMA,
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": INPUT_AUDIT_SHA256,
        "sample_count": 4,
        "prediction_count": len(prediction_entries),
        "score_count": len(score_entries),
        "prediction_entries": prediction_entries,
        "score_entries": score_entries,
    }


def evaluate_a1_smoke(
    pairs: Sequence[LoadedCrosssensorPair],
    bundles: Sequence[DevelopmentPredictionBundle],
    score_cache: ScoreCache,
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate the exact four canonical A1 ROIs into host-free JSON payloads."""

    validated = _validate_a1_inputs(pairs, bundles)
    sample_records = tuple(
        _evaluate_a1_sample(pair, bundle, score_cache) for pair, bundle in validated
    )
    k5a_k5b = tuple(float(record["stability"]["k5a_k5b_spearman"]) for record in sample_records)
    k5a_k25 = tuple(float(record["stability"]["k5a_k25_spearman"]) for record in sample_records)
    jaccards = tuple(
        float(record["stability"]["k5a_k25_top10_jaccard"]) for record in sample_records
    )
    result = _a1_result_payload(
        sample_records,
        k5_statistically_stable=_is_k5_statistically_stable(k5a_k5b, k5a_k25, jaccards),
    )
    audit = _score_and_prediction_evidence_payload(validated, score_cache)
    canonical_json(result)
    canonical_json(audit)
    return result, audit


def _require_committed_structure(
    committed_result: Mapping[str, object], committed_audit: Mapping[str, object]
) -> None:
    if committed_result.get("schema") != A1_RESULT_SCHEMA:
        raise ValueError("committed A1 result schema is invalid")
    if committed_audit.get("schema") != A1_CACHE_AUDIT_SCHEMA:
        raise ValueError("committed A1 audit schema is invalid")
    if committed_audit.get("experiment_schema") != A1_RESULT_SCHEMA:
        raise ValueError("committed A1 audit experiment schema is invalid")
    for payload, label in (
        (committed_result, "result"),
        (committed_audit, "audit"),
    ):
        if payload.get("sample_count") != 4:
            raise ValueError(f"committed A1 {label} sample count is invalid")
        if payload.get("prediction_count") != 108:
            raise ValueError(f"committed A1 {label} prediction count is invalid")
        if payload.get("score_count") != 20:
            raise ValueError(f"committed A1 {label} score count is invalid")
    samples = committed_result.get("samples")
    if not isinstance(samples, list) or len(samples) != 4:
        raise ValueError("committed A1 result samples are invalid")
    prediction_entries = committed_audit.get("prediction_entries")
    score_entries = committed_audit.get("score_entries")
    if not isinstance(prediction_entries, list) or len(prediction_entries) != 108:
        raise ValueError("committed A1 audit prediction entries are invalid")
    if not isinstance(score_entries, list) or len(score_entries) != 20:
        raise ValueError("committed A1 audit score entries are invalid")


def _prediction_identity_from_dict(value: object) -> PredictionIdentity:
    if not isinstance(value, dict) or set(value) != {
        "model_provenance",
        "source",
        "sample_id",
        "lr",
    }:
        raise ValueError("committed A1 prediction identity is invalid")
    lr = value["lr"]
    if not isinstance(lr, dict) or set(lr) != {"shape", "dtype", "sha256"}:
        raise ValueError("committed A1 prediction LR identity is invalid")
    try:
        return PredictionIdentity(
            value["model_provenance"],
            value["source"],
            value["sample_id"],
            tuple(lr["shape"]),
            lr["dtype"],
            lr["sha256"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("committed A1 prediction identity is invalid") from exc


def _score_identity_from_dict(value: object) -> ScoreIdentity:
    if not isinstance(value, dict) or set(value) != {
        "score_name",
        "score_schema_version",
        "sample_id",
        "input_sha256s",
        "operator_parameters",
    }:
        raise ValueError("committed A1 score identity is invalid")
    try:
        return ScoreIdentity(
            score_name=value["score_name"],
            score_schema_version=value["score_schema_version"],
            sample_id=value["sample_id"],
            input_sha256s=tuple(value["input_sha256s"]),
            operator_parameters=value["operator_parameters"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("committed A1 score identity is invalid") from exc


def _committed_identities(
    pairs: Sequence[LoadedCrosssensorPair],
    committed_result: Mapping[str, object],
    committed_audit: Mapping[str, object],
) -> tuple[
    tuple[PredictionIdentity, ...],
    tuple[ScoreIdentity, ...],
    list[dict[str, object]],
]:
    _require_committed_structure(committed_result, committed_audit)
    pair_values = tuple(pairs)
    if len(pair_values) != 4:
        raise ValueError("A1 replay requires exactly four canonical pairs")
    samples = committed_result["samples"]
    for bin_index, (pair, sample) in enumerate(zip(pair_values, samples, strict=True)):
        _validate_pair(pair, expected_bin=bin_index)
        if (
            not isinstance(sample, dict)
            or sample.get("sample_id") != pair.pair.sample_id
            or sample.get("correlation_bin") != bin_index
        ):
            raise ValueError("committed A1 result sample order/bin is invalid")
    prediction_entries = committed_audit["prediction_entries"]
    score_entries = committed_audit["score_entries"]
    prediction_identities: list[PredictionIdentity] = []
    score_identities: list[ScoreIdentity] = []
    for pair_index, pair in enumerate(pair_values):
        start = pair_index * len(_MODEL_SEED_SLOTS)
        for entry, (model_name, seed) in zip(
            prediction_entries[start : start + len(_MODEL_SEED_SLOTS)],
            _MODEL_SEED_SLOTS,
            strict=True,
        ):
            if not isinstance(entry, dict):
                raise ValueError("committed A1 prediction audit entry is invalid")
            identity = _prediction_identity_from_dict(entry.get("identity"))
            if (
                entry.get("sample_id") != pair.pair.sample_id
                or entry.get("correlation_bin") != pair.metadata.correlation_bin
                or entry.get("model_name") != model_name
                or entry.get("seed") != seed
                or entry.get("cache_key") != identity.key
            ):
                raise ValueError("committed A1 prediction audit order/seed/key is invalid")
            prediction_identities.append(identity)
        score_start = pair_index * len(SCORE_NAMES)
        for entry, name in zip(
            score_entries[score_start : score_start + len(SCORE_NAMES)],
            SCORE_NAMES,
            strict=True,
        ):
            if not isinstance(entry, dict):
                raise ValueError("committed A1 score audit entry is invalid")
            identity = _score_identity_from_dict(entry.get("identity"))
            if (
                entry.get("sample_id") != pair.pair.sample_id
                or entry.get("correlation_bin") != pair.metadata.correlation_bin
                or entry.get("name") != name
                or identity.score_name != name
                or identity.sample_id != pair.pair.sample_id
                or entry.get("cache_key") != identity.key
            ):
                raise ValueError("committed A1 score audit order/key is invalid")
            score_identities.append(identity)
    return tuple(prediction_identities), tuple(score_identities), prediction_entries


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_score_files(
    cache: ScoreCache, identities: Sequence[ScoreIdentity]
) -> tuple[tuple[object, ...], ...]:
    if len({identity.key for identity in identities}) != len(identities):
        raise ValueError("committed A1 score identities must be unique")
    result: list[tuple[object, ...]] = []
    for identity in identities:
        expected = {
            cache.root / f"{identity.key}.json",
            cache.root / f"{identity.key}.safetensors",
            cache.root / f"{identity.key}.lock",
        }
        named = {path for path in cache.root.iterdir() if path.name.startswith(f"{identity.key}.")}
        if named != expected:
            raise RuntimeError("named A1 score cache entry has unexpected files")
        for path in sorted(expected):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("A1 score cache file is invalid")
            stat = path.stat()
            result.append((path.name, stat.st_size, stat.st_mtime_ns, _stream_sha256(path)))
    return tuple(result)


def _snapshot_all_cache_files(
    prediction_identities: Sequence[PredictionIdentity],
    score_identities: Sequence[ScoreIdentity],
    prediction_cache: PredictionCache,
    score_cache: ScoreCache,
) -> tuple[tuple[object, ...], ...]:
    prediction_snapshot = snapshot_cache_files(prediction_cache.root, prediction_identities)
    return (
        *(("prediction", *item) for item in prediction_snapshot),
        *(("score", *item) for item in _snapshot_score_files(score_cache, score_identities)),
    )


def _bundles_from_prediction_cache(
    pairs: Sequence[LoadedCrosssensorPair],
    identities: Sequence[PredictionIdentity],
    entries: Sequence[dict[str, object]],
    cache: PredictionCache,
) -> tuple[DevelopmentPredictionBundle, ...]:
    bundles: list[DevelopmentPredictionBundle] = []
    slot_count = len(_MODEL_SEED_SLOTS)
    for pair_index, pair in enumerate(pairs):
        records: list[CachedDevelopmentPrediction] = []
        start = pair_index * slot_count
        for identity, entry, (model_name, seed) in zip(
            identities[start : start + slot_count],
            entries[start : start + slot_count],
            _MODEL_SEED_SLOTS,
            strict=True,
        ):
            tensor = cache.get(identity)
            if tensor is None:
                raise RuntimeError("A1 prediction cache entry is missing during replay")
            digest = tensor_sha256(tensor)
            if entry.get("prediction_sha256") != digest:
                raise ValueError("committed A1 prediction logical tensor SHA is invalid")
            record = CachedDevelopmentPrediction(
                model_name=model_name,
                seed=seed,
                identity=identity,
                prediction_sha256=digest,
                tensor=tensor,
            )
            _validate_prediction_tensor(pair, record, expected_name=model_name, expected_seed=seed)
            records.append(record)
        bundles.append(
            DevelopmentPredictionBundle(
                sample_id=pair.pair.sample_id,
                bicubic=records[0],
                sen2srlite=records[1],
                ldsr=tuple(records[2:]),
            )
        )
    return tuple(bundles)


def replay_a1_smoke(
    pairs: Sequence[LoadedCrosssensorPair],
    committed_result: Mapping[str, object],
    committed_audit: Mapping[str, object],
    prediction_cache: PredictionCache,
    score_cache: ScoreCache,
) -> tuple[dict[str, object], dict[str, object]]:
    """Rebuild A1 only from verified caches and prove all cache files stayed stable."""

    pair_values = tuple(pairs)
    prediction_identities, score_identities, entries = _committed_identities(
        pair_values, committed_result, committed_audit
    )
    before = _snapshot_all_cache_files(
        prediction_identities,
        score_identities,
        prediction_cache,
        score_cache,
    )
    bundles = _bundles_from_prediction_cache(
        pair_values, prediction_identities, entries, prediction_cache
    )
    rebuilt_result, rebuilt_audit = evaluate_a1_smoke(pair_values, bundles, score_cache)
    if canonical_json(rebuilt_result) != canonical_json(dict(committed_result)):
        raise ValueError("rebuilt A1 result differs from committed result")
    if canonical_json(rebuilt_audit) != canonical_json(dict(committed_audit)):
        raise ValueError("rebuilt A1 cache audit differs from committed audit")
    after = _snapshot_all_cache_files(
        prediction_identities,
        score_identities,
        prediction_cache,
        score_cache,
    )
    if before != after:
        raise RuntimeError("cache files changed during A1 replay")
    return rebuilt_result, rebuilt_audit
