"""Deterministic Phase 2B3-A score audits and cache-only replay."""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from trustsr.artifacts.predictions import (
    PredictionCache,
    PredictionIdentity,
    build_identity,
    tensor_sha256,
)
from trustsr.artifacts.scores import (
    ScoreCache,
    ScoreIdentity,
    score_entry_evidence,
)
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    RAW_RADIOMETRIC_MAX,
    LoadedCrosssensorPair,
    RadiometricSaturation,
)
from trustsr.evaluation import score_selection
from trustsr.evaluation.crosssensor_smoke import (
    INPUT_AUDIT_SHA256,
    cache_entry_evidence,
    snapshot_cache_files,
)
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
from trustsr.evaluation.score_selection import DevelopmentRoiResult
from trustsr.jsonio import canonical_json
from trustsr.risk.local import ensemble_variance_score, local_l1_risk
from trustsr.risk.proxies import (
    lr_reprojection_l1_score,
    three_model_disagreement_score,
)

A1_RESULT_SCHEMA = "trustsr.phase2b3a-development-smoke.v2"
A1_CACHE_AUDIT_SCHEMA = "trustsr.phase2b3a-development-smoke-cache-audit.v2"
A2_RESULT_SCHEMA = "trustsr.phase2b3a-development-score-audit.v1"
A2_CACHE_AUDIT_SCHEMA = "trustsr.phase2b3a-development-score-cache-audit.v1"
A2_SCORE_NAMES = (
    "lr_reprojection_l1",
    "three_model_disagreement",
    "ldsr_variance_k5",
)
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
_UNSET = object()


def _serialize_radiometric_saturation(
    pair: LoadedCrosssensorPair,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, saturation in (
        ("lr", pair.metadata.lr_saturation),
        ("hr", pair.metadata.hr_saturation),
    ):
        if not isinstance(saturation, RadiometricSaturation):
            raise ValueError("Phase 2B3-A pairs require radiometric saturation records")
        result[name] = {
            "raw_crop_minimum": saturation.raw_crop_minimum,
            "raw_crop_maximum": saturation.raw_crop_maximum,
            "clipped_high_count": saturation.clipped_high_count,
            "clipped_high_by_band": list(saturation.clipped_high_by_band),
        }
    return result


def _build_radiometric_policy(
    sample_records: Sequence[Mapping[str, object]],
    *,
    expected_sample_count: int,
    claimed_policy: object = _UNSET,
) -> dict[str, object]:
    if type(expected_sample_count) is not int or expected_sample_count < 1:
        raise ValueError("radiometric policy sample count is invalid")
    if len(sample_records) != expected_sample_count:
        raise ValueError("radiometric policy sample count is inconsistent")
    lr_total = 0
    hr_total = 0
    affected_samples = 0
    affected_assets = 0
    crop_maxima: list[int] = []
    for sample in sample_records:
        if not isinstance(sample, Mapping):
            raise ValueError("radiometric sample record is invalid")
        saturation = sample.get("radiometric_saturation")
        if not isinstance(saturation, dict) or set(saturation) != {"lr", "hr"}:
            raise ValueError("radiometric saturation record is invalid")
        sample_affected = False
        for asset_name in ("lr", "hr"):
            asset = saturation[asset_name]
            if not isinstance(asset, dict) or set(asset) != {
                "raw_crop_minimum",
                "raw_crop_maximum",
                "clipped_high_count",
                "clipped_high_by_band",
            }:
                raise ValueError("radiometric saturation asset record is invalid")
            minimum = asset["raw_crop_minimum"]
            maximum = asset["raw_crop_maximum"]
            clipped = asset["clipped_high_count"]
            by_band = asset["clipped_high_by_band"]
            if any(type(value) is not int for value in (minimum, maximum, clipped)):
                raise ValueError("radiometric saturation values must be built-in integers")
            if (
                not isinstance(by_band, list)
                or len(by_band) != 4
                or any(type(value) is not int for value in by_band)
            ):
                raise ValueError("radiometric saturation requires four built-in band counts")
            if (
                minimum < 0
                or maximum < 0
                or clipped < 0
                or any(value < 0 for value in by_band)
            ):
                raise ValueError("radiometric saturation values must be non-negative")
            if minimum > maximum:
                raise ValueError("radiometric saturation minimum exceeds maximum")
            if maximum > RAW_RADIOMETRIC_MAX:
                raise ValueError("radiometric saturation maximum exceeds the raw domain")
            if clipped != sum(by_band):
                raise ValueError("radiometric saturation band counts do not match total")
            if (maximum > 10_000) != (clipped > 0):
                raise ValueError(
                    "radiometric saturation maximum and clipped count are inconsistent"
                )
            crop_maxima.append(maximum)
            if clipped > 0:
                affected_assets += 1
                sample_affected = True
            if asset_name == "lr":
                lr_total += clipped
            else:
                hr_total += clipped
        affected_samples += int(sample_affected)
    result = {
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "raw_radiometric_max": RAW_RADIOMETRIC_MAX,
        "saturation_threshold": 10000,
        "bands": ["B04", "B03", "B02", "B08"],
        "sample_count": expected_sample_count,
        "affected_sample_count": affected_samples,
        "affected_asset_count": affected_assets,
        "lr_clipped_high_count": lr_total,
        "hr_clipped_high_count": hr_total,
        "raw_crop_maximum": max(crop_maxima),
    }
    if claimed_policy is not _UNSET:
        integer_keys = {
            "raw_radiometric_max",
            "saturation_threshold",
            "sample_count",
            "affected_sample_count",
            "affected_asset_count",
            "lr_clipped_high_count",
            "hr_clipped_high_count",
            "raw_crop_maximum",
        }
        if (
            not isinstance(claimed_policy, dict)
            or set(claimed_policy) != set(result)
            or any(type(claimed_policy.get(key)) is not int for key in integer_keys)
            or not isinstance(claimed_policy.get("bands"), list)
            or claimed_policy != result
        ):
            raise ValueError("radiometric policy aggregate is invalid")
    return result


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
    if (
        metadata.crop_policy != CROP_POLICY
        or metadata.normalization_policy != PHASE2B3A_NORMALIZATION_POLICY
    ):
        raise ValueError("A1 pair has the wrong input policy")
    if not isinstance(metadata.lr_saturation, RadiometricSaturation) or not isinstance(
        metadata.hr_saturation, RadiometricSaturation
    ):
        raise ValueError("A1 pair requires radiometric saturation records")


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
        or provenance.get("normalization_policy")
        != PHASE2B3A_NORMALIZATION_POLICY
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


def _score_operator_parameters(
    pair: LoadedCrosssensorPair, name: str
) -> dict[str, str | int | float | bool | None]:
    parameters: dict[str, dict[str, str | int | float | bool | None]] = {
        SCORE_NAMES[0]: {
            "algorithm": "ensemble_variance_score",
            "band_reduction": "mean",
            "correction": 0,
            "seed_first": K5A_SEEDS[0],
            "seed_last": K5A_SEEDS[-1],
            "seed_count": len(K5A_SEEDS),
        },
        SCORE_NAMES[1]: {
            "algorithm": "ensemble_variance_score",
            "band_reduction": "mean",
            "correction": 0,
            "seed_first": K5B_SEEDS[0],
            "seed_last": K5B_SEEDS[-1],
            "seed_count": len(K5B_SEEDS),
        },
        SCORE_NAMES[2]: {
            "algorithm": "ensemble_variance_score",
            "band_reduction": "mean",
            "correction": 0,
            "seed_first": A1_SEEDS[0],
            "seed_last": A1_SEEDS[-1],
            "seed_count": len(A1_SEEDS),
        },
        SCORE_NAMES[3]: {
            "algorithm": "lr_reprojection_l1_score",
            "downsample_mode": "area",
            "scale": 4,
            "upsample_mode": "repeat_interleave",
        },
        SCORE_NAMES[4]: {
            "algorithm": "three_model_disagreement_score",
            "band_reduction": "mean",
            "correction": 0,
            "model_order": "bicubic-x4,sen2srlite-x4,ldsr-s2-x4",
        },
    }
    try:
        selected = parameters[name]
    except KeyError as exc:
        raise ValueError("unknown A1 score name") from exc
    return {**selected, "lr_sha256": tensor_sha256(pair.pair.lr)}


def _score_identity_from_hashes(
    pair: LoadedCrosssensorPair, *, name: str, input_sha256s: Sequence[str]
) -> ScoreIdentity:
    return ScoreIdentity(
        score_name=name,
        score_schema_version=SCORE_SCHEMA_VERSION,
        sample_id=pair.pair.sample_id,
        input_sha256s=tuple(input_sha256s),
        operator_parameters=_score_operator_parameters(pair, name),
    )


def _score_input_groups(
    bundle: DevelopmentPredictionBundle,
) -> tuple[tuple[CachedDevelopmentPrediction, ...], ...]:
    central = bundle.ldsr_for_seed(3407)
    return (
        tuple(bundle.ldsr_for_seed(seed) for seed in K5A_SEEDS),
        tuple(bundle.ldsr_for_seed(seed) for seed in K5B_SEEDS),
        tuple(bundle.ldsr_for_seed(seed) for seed in A1_SEEDS),
        (central,),
        (bundle.bicubic, bundle.sen2srlite, central),
    )


def _expected_score_identities(
    pair: LoadedCrosssensorPair, bundle: DevelopmentPredictionBundle
) -> tuple[ScoreIdentity, ...]:
    return tuple(
        _score_identity_from_hashes(
            pair,
            name=name,
            input_sha256s=tuple(item.prediction_sha256 for item in inputs),
        )
        for name, inputs in zip(SCORE_NAMES, _score_input_groups(bundle), strict=True)
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
    computations: tuple[Callable[[], torch.Tensor], ...] = (
        lambda: ensemble_variance_score(torch.stack([item.tensor for item in k5a], dim=0)),
        lambda: ensemble_variance_score(torch.stack([item.tensor for item in k5b], dim=0)),
        lambda: ensemble_variance_score(torch.stack([item.tensor for item in k25], dim=0)),
        lambda: lr_reprojection_l1_score(central.tensor, pair.pair.lr, scale=4),
        lambda: three_model_disagreement_score(
            (bundle.bicubic.tensor, bundle.sen2srlite.tensor, central.tensor)
        ),
    )
    return tuple(
        _load_or_compute_score(name, identity, score_cache, compute)
        for name, identity, compute in zip(
            SCORE_NAMES,
            _expected_score_identities(pair, bundle),
            computations,
            strict=True,
        )
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
    return _evaluate_a1_sample_from_scores(pair, bundle, scores)


def _evaluate_a1_sample_from_scores(
    pair: LoadedCrosssensorPair,
    bundle: DevelopmentPredictionBundle,
    scores: Sequence[CachedScoreMap],
) -> dict[str, object]:
    score_values = tuple(scores)
    if tuple(score.name for score in score_values) != SCORE_NAMES:
        raise ValueError("A1 cached score-map order is invalid")
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
        for score in score_values
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
        "radiometric_saturation": _serialize_radiometric_saturation(pair),
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
            "k5a_k5b_spearman": score_map_spearman(score_values[0].tensor, score_values[1].tensor),
            "k5a_k25_spearman": score_map_spearman(score_values[0].tensor, score_values[2].tensor),
            "k5a_k25_top10_jaccard": top_fraction_jaccard(
                score_values[0].tensor, score_values[2].tensor, fraction=0.10
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
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "radiometric_policy": _build_radiometric_policy(
            sample_records, expected_sample_count=4
        ),
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
    score_groups: Sequence[Sequence[CachedScoreMap]],
    score_cache: ScoreCache,
) -> dict[str, object]:
    prediction_entries: list[dict[str, object]] = []
    score_entries: list[dict[str, object]] = []
    for (pair, bundle), scores in zip(validated, score_groups, strict=True):
        prediction_entries.extend(
            _prediction_evidence(pair, item)
            for item in (bundle.bicubic, bundle.sen2srlite, *bundle.ldsr)
        )
        score_entries.extend(_score_evidence(pair, score, score_cache) for score in scores)
    return {
        "schema": A1_CACHE_AUDIT_SCHEMA,
        "experiment_schema": A1_RESULT_SCHEMA,
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": INPUT_AUDIT_SHA256,
        "sample_count": 4,
        "prediction_count": len(prediction_entries),
        "score_count": len(score_entries),
        "prediction_entries": prediction_entries,
        "score_entries": score_entries,
    }


def _build_a1_payloads_from_scores(
    validated: Sequence[tuple[LoadedCrosssensorPair, DevelopmentPredictionBundle]],
    score_groups: Sequence[Sequence[CachedScoreMap]],
    score_cache: ScoreCache,
) -> tuple[dict[str, object], dict[str, object]]:
    score_values = tuple(tuple(group) for group in score_groups)
    if len(score_values) != len(validated):
        raise ValueError("A1 cached score groups do not match the canonical samples")
    sample_records = tuple(
        _evaluate_a1_sample_from_scores(pair, bundle, scores)
        for (pair, bundle), scores in zip(validated, score_values, strict=True)
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
    audit = _score_and_prediction_evidence_payload(validated, score_values, score_cache)
    canonical_json(result)
    canonical_json(audit)
    return result, audit


def evaluate_a1_smoke(
    pairs: Sequence[LoadedCrosssensorPair],
    bundles: Sequence[DevelopmentPredictionBundle],
    score_cache: ScoreCache,
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate the exact four canonical A1 ROIs into host-free JSON payloads."""

    validated = _validate_a1_inputs(pairs, bundles)
    score_groups = tuple(
        build_a1_score_maps(pair, bundle, score_cache) for pair, bundle in validated
    )
    return _build_a1_payloads_from_scores(validated, score_groups, score_cache)


def _require_committed_structure(
    committed_result: Mapping[str, object], committed_audit: Mapping[str, object]
) -> None:
    if committed_result.get("schema") != A1_RESULT_SCHEMA:
        raise ValueError("committed A1 result schema is invalid")
    if committed_audit.get("schema") != A1_CACHE_AUDIT_SCHEMA:
        raise ValueError("committed A1 audit schema is invalid")
    if committed_audit.get("experiment_schema") != A1_RESULT_SCHEMA:
        raise ValueError("committed A1 audit experiment schema is invalid")
    if (
        committed_result.get("normalization_policy")
        != PHASE2B3A_NORMALIZATION_POLICY
        or committed_audit.get("normalization_policy")
        != PHASE2B3A_NORMALIZATION_POLICY
    ):
        raise ValueError("committed A1 normalization policy is invalid")
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
    _build_radiometric_policy(
        samples,
        expected_sample_count=4,
        claimed_policy=committed_result.get("radiometric_policy"),
    )
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


def _expected_score_identities_from_prediction_entries(
    pair: LoadedCrosssensorPair, entries: Sequence[dict[str, object]]
) -> tuple[ScoreIdentity, ...]:
    if len(entries) != len(_MODEL_SEED_SLOTS):
        raise ValueError("committed A1 prediction evidence count is invalid")
    prediction_sha256s = tuple(entry.get("prediction_sha256") for entry in entries)
    input_groups = (
        prediction_sha256s[2:7],
        prediction_sha256s[7:12],
        prediction_sha256s[2:27],
        (prediction_sha256s[2],),
        (prediction_sha256s[0], prediction_sha256s[1], prediction_sha256s[2]),
    )
    try:
        return tuple(
            _score_identity_from_hashes(pair, name=name, input_sha256s=input_sha256s)
            for name, input_sha256s in zip(SCORE_NAMES, input_groups, strict=True)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("committed A1 prediction logical SHA evidence is invalid") from exc


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
            or sample.get("radiometric_saturation")
            != _serialize_radiometric_saturation(pair)
        ):
            raise ValueError("committed A1 result sample order/bin is invalid")
    prediction_entries = committed_audit["prediction_entries"]
    score_entries = committed_audit["score_entries"]
    prediction_identities: list[PredictionIdentity] = []
    score_identities: list[ScoreIdentity] = []
    for pair_index, pair in enumerate(pair_values):
        start = pair_index * len(_MODEL_SEED_SLOTS)
        pair_prediction_entries = prediction_entries[start : start + len(_MODEL_SEED_SLOTS)]
        for entry, (model_name, seed) in zip(
            pair_prediction_entries,
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
        expected_score_identities = _expected_score_identities_from_prediction_entries(
            pair, pair_prediction_entries
        )
        score_start = pair_index * len(SCORE_NAMES)
        for entry, name, expected_identity in zip(
            score_entries[score_start : score_start + len(SCORE_NAMES)],
            SCORE_NAMES,
            expected_score_identities,
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
                or identity.as_dict() != expected_identity.as_dict()
                or entry.get("cache_key") != identity.key
            ):
                raise ValueError("committed A1 score audit order/key/identity is invalid")
            score_identities.append(expected_identity)
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


def _load_existing_score_groups(
    pairs: Sequence[LoadedCrosssensorPair],
    bundles: Sequence[DevelopmentPredictionBundle],
    identities: Sequence[ScoreIdentity],
    entries: Sequence[dict[str, object]],
    cache: ScoreCache,
) -> tuple[tuple[CachedScoreMap, ...], ...]:
    groups: list[tuple[CachedScoreMap, ...]] = []
    for pair_index, (pair, bundle) in enumerate(zip(pairs, bundles, strict=True)):
        start = pair_index * len(SCORE_NAMES)
        expected_from_verified_predictions = _expected_score_identities(pair, bundle)
        records: list[CachedScoreMap] = []
        for name, identity, expected_identity, entry in zip(
            SCORE_NAMES,
            identities[start : start + len(SCORE_NAMES)],
            expected_from_verified_predictions,
            entries[start : start + len(SCORE_NAMES)],
            strict=True,
        ):
            if identity.as_dict() != expected_identity.as_dict():
                raise ValueError("committed A1 score identity differs from verified predictions")
            score = cache.get(identity)
            if score is None:
                raise RuntimeError("canonical A1 score cache entry is missing during replay")
            score_sha256 = tensor_sha256(score)
            if entry.get("score_sha256") != score_sha256:
                raise ValueError("committed A1 score logical tensor SHA is invalid")
            records.append(CachedScoreMap(name, identity, score_sha256, score))
        groups.append(tuple(records))
    return tuple(groups)


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
    score_entries = committed_audit["score_entries"]
    score_groups = _load_existing_score_groups(
        pair_values,
        bundles,
        score_identities,
        score_entries,
        score_cache,
    )
    validated = _validate_a1_inputs(pair_values, bundles)
    rebuilt_result, rebuilt_audit = _build_a1_payloads_from_scores(
        validated, score_groups, score_cache
    )
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


def _validate_code_revision(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("code_revision must be a 40-character lowercase Git object ID")
    return value


def _a2_candidate_names(include_ldsr_variance_k5: bool) -> tuple[str, ...]:
    if type(include_ldsr_variance_k5) is not bool:
        raise TypeError("include_ldsr_variance_k5 must be a built-in boolean")
    return A2_SCORE_NAMES if include_ldsr_variance_k5 else A2_SCORE_NAMES[:2]


def _a2_prediction_slots(
    include_ldsr_variance_k5: bool,
) -> tuple[tuple[str, int | None], ...]:
    seeds = K5A_SEEDS if include_ldsr_variance_k5 else (3407,)
    return (
        ("bicubic-x4", None),
        ("sen2srlite-x4", None),
        *(("ldsr-s2-x4", seed) for seed in seeds),
    )


def _a2_base_score_configuration(name: str) -> dict[str, object]:
    configurations: dict[str, dict[str, object]] = {
        "lr_reprojection_l1": {
            "algorithm": "lr_reprojection_l1_score",
            "downsample_mode": "area",
            "scale": 4,
            "upsample_mode": "repeat_interleave",
        },
        "three_model_disagreement": {
            "algorithm": "three_model_disagreement_score",
            "band_reduction": "mean",
            "correction": 0,
            "model_order": "bicubic-x4,sen2srlite-x4,ldsr-s2-x4",
        },
        "ldsr_variance_k5": {
            "algorithm": "ensemble_variance_score",
            "band_reduction": "mean",
            "correction": 0,
            "seed_first": K5A_SEEDS[0],
            "seed_last": K5A_SEEDS[-1],
            "seed_count": len(K5A_SEEDS),
        },
    }
    try:
        return dict(configurations[name])
    except KeyError as exc:
        raise ValueError("unknown A2 score name") from exc


def _a2_score_seeds(name: str) -> tuple[int, ...]:
    if name == "ldsr_variance_k5":
        return K5A_SEEDS
    if name in A2_SCORE_NAMES[:2]:
        return (3407,)
    raise ValueError("unknown A2 score name")


def _a2_score_identity(
    pair: LoadedCrosssensorPair,
    *,
    name: str,
    input_sha256s: Sequence[str],
) -> ScoreIdentity:
    return ScoreIdentity(
        score_name=name,
        score_schema_version=SCORE_SCHEMA_VERSION,
        sample_id=pair.pair.sample_id,
        input_sha256s=tuple(input_sha256s),
        operator_parameters={
            **_a2_base_score_configuration(name),
            "lr_sha256": tensor_sha256(pair.pair.lr),
        },
    )


def _a2_score_input_hashes(
    bundle: DevelopmentPredictionBundle, name: str
) -> tuple[str, ...]:
    central = bundle.ldsr_for_seed(3407)
    if name == "lr_reprojection_l1":
        return (central.prediction_sha256,)
    if name == "three_model_disagreement":
        return (
            bundle.bicubic.prediction_sha256,
            bundle.sen2srlite.prediction_sha256,
            central.prediction_sha256,
        )
    if name == "ldsr_variance_k5":
        return tuple(
            bundle.ldsr_for_seed(seed).prediction_sha256 for seed in K5A_SEEDS
        )
    raise ValueError("unknown A2 score name")


def _validate_a2_pair(pair: LoadedCrosssensorPair) -> None:
    if not isinstance(pair, LoadedCrosssensorPair):
        raise TypeError("A2 inputs must be LoadedCrosssensorPair values")
    pair.pair.validate()
    metadata = pair.metadata
    if metadata.split != "development":
        raise ValueError("A2 inputs must use only the development split")
    if metadata.days_between not in (-1, 0, 1):
        raise ValueError("A2 pair has an invalid development stratum")
    if metadata.correlation_bin not in range(4):
        raise ValueError("A2 pair has an invalid development stratum")
    if metadata.selection_round not in range(1, 11):
        raise ValueError("A2 pair has an invalid development stratum")
    if metadata.manifest_sha256 != POST_MANIFEST_SHA256:
        raise ValueError("A2 pair has the wrong manifest")
    if metadata.sample_id != pair.pair.sample_id:
        raise ValueError("A2 pair and metadata identities differ")
    if pair.pair.source != f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}":
        raise ValueError("A2 pair has the wrong source identity")
    if (
        metadata.crop_policy != CROP_POLICY
        or metadata.normalization_policy != PHASE2B3A_NORMALIZATION_POLICY
    ):
        raise ValueError("A2 pair has the wrong input policy")
    if not isinstance(metadata.lr_saturation, RadiometricSaturation) or not isinstance(
        metadata.hr_saturation, RadiometricSaturation
    ):
        raise ValueError("A2 pair requires radiometric saturation records")


def _validate_a2_structure(
    pairs: Sequence[LoadedCrosssensorPair],
) -> tuple[LoadedCrosssensorPair, ...]:
    pair_values = tuple(pairs)
    if len(pair_values) != 120:
        raise ValueError("A2 requires exactly 120 development ROI")
    cells: dict[tuple[int, int], list[int]] = {}
    for pair in pair_values:
        _validate_a2_pair(pair)
        metadata = pair.metadata
        cells.setdefault(
            (metadata.days_between, metadata.correlation_bin), []
        ).append(metadata.selection_round)
    if len({pair.metadata.sample_id for pair in pair_values}) != 120:
        raise ValueError("A2 requires 120 unique sample identities")
    if len({pair.metadata.spatial_group_id for pair in pair_values}) != 120:
        raise ValueError("A2 requires 120 unique spatial groups")
    expected_cells = {(day, bin_index) for day in (-1, 0, 1) for bin_index in range(4)}
    if set(cells) != expected_cells or any(
        sorted(rounds) != list(range(1, 11)) for rounds in cells.values()
    ):
        raise ValueError("A2 development strata must each contain selection rounds 1 through 10")
    return pair_values


def _validate_ordered_development_sample_ids(
    ordered_development_sample_ids: Sequence[str],
    pairs: Sequence[LoadedCrosssensorPair],
) -> tuple[str, ...]:
    if isinstance(ordered_development_sample_ids, str | bytes) or not isinstance(
        ordered_development_sample_ids, Sequence
    ):
        raise TypeError("ordered development sample IDs must be a sequence")
    ordered = tuple(ordered_development_sample_ids)
    if len(ordered) != 120:
        raise ValueError("ordered development sample IDs must contain exactly 120 values")
    if any(type(sample_id) is not str or not sample_id for sample_id in ordered):
        raise ValueError("ordered development sample IDs must be built-in nonempty strings")
    if len(set(ordered)) != 120:
        raise ValueError("ordered development sample IDs must be unique")
    observed = tuple(pair.pair.sample_id for pair in pairs)
    if observed != ordered:
        raise ValueError("A2 pairs do not match the authoritative ordered sample IDs")
    return ordered


def _validate_a2_bundle_tensors(
    pair: LoadedCrosssensorPair,
    bundle: DevelopmentPredictionBundle,
    *,
    include_ldsr_variance_k5: bool,
) -> None:
    if not isinstance(bundle, DevelopmentPredictionBundle):
        raise TypeError("A2 bundles must be DevelopmentPredictionBundle values")
    if bundle.sample_id != pair.pair.sample_id:
        raise ValueError("A2 bundle membership does not match pair order")
    expected_seeds = K5A_SEEDS if include_ldsr_variance_k5 else (3407,)
    if tuple(item.seed for item in bundle.ldsr) != expected_seeds:
        raise ValueError(
            "A2 bundle seed membership conflicts with the accepted A1 variance decision"
        )
    slots = _a2_prediction_slots(include_ldsr_variance_k5)
    values = (bundle.bicubic, bundle.sen2srlite, *bundle.ldsr)
    for item, (name, seed) in zip(values, slots, strict=True):
        _validate_prediction_tensor(pair, item, expected_name=name, expected_seed=seed)


def _build_a2_score_maps(
    pair: LoadedCrosssensorPair,
    bundle: DevelopmentPredictionBundle,
    score_cache: ScoreCache,
    *,
    include_ldsr_variance_k5: bool,
) -> tuple[CachedScoreMap, ...]:
    names = _a2_candidate_names(include_ldsr_variance_k5)
    central = bundle.ldsr_for_seed(3407)
    computations: dict[str, Callable[[], torch.Tensor]] = {
        "lr_reprojection_l1": lambda: lr_reprojection_l1_score(
            central.tensor, pair.pair.lr, scale=4
        ),
        "three_model_disagreement": lambda: three_model_disagreement_score(
            (bundle.bicubic.tensor, bundle.sen2srlite.tensor, central.tensor)
        ),
        "ldsr_variance_k5": lambda: ensemble_variance_score(
            torch.stack(
                [bundle.ldsr_for_seed(seed).tensor for seed in K5A_SEEDS], dim=0
            )
        ),
    }
    return tuple(
        _load_or_compute_score(
            name,
            _a2_score_identity(
                pair,
                name=name,
                input_sha256s=_a2_score_input_hashes(bundle, name),
            ),
            score_cache,
            computations[name],
        )
        for name in names
    )


def _a2_prediction_evidence(
    pair: LoadedCrosssensorPair,
    item: CachedDevelopmentPrediction,
    prediction_cache: PredictionCache,
) -> dict[str, object]:
    evidence = cache_entry_evidence(prediction_cache.root, item.identity)
    if evidence["prediction_sha256"] != item.prediction_sha256:
        raise RuntimeError("A2 prediction cache evidence differs from the prediction bundle")
    return {
        "sample_id": pair.pair.sample_id,
        "model_name": item.model_name,
        "seed": item.seed,
        "cache_key": item.identity.key,
        "identity": item.identity.as_dict(),
        "prediction_sha256": item.prediction_sha256,
        "files": evidence["files"],
    }


def _a2_score_record(
    score: CachedScoreMap,
    primary: RoiScoreDiagnostics,
    sensitivity: RoiScoreDiagnostics,
) -> dict[str, object]:
    return {
        "name": score.name,
        "cache_key": score.identity.key,
        "score_sha256": score.score_sha256,
        "primary_window_9": _diagnostic_payload(primary),
        "sensitivity_window_1": _diagnostic_payload(sensitivity),
    }


def _a2_roi_result(
    pair: LoadedCrosssensorPair, diagnostic: RoiScoreDiagnostics
) -> DevelopmentRoiResult:
    metadata = pair.metadata
    return DevelopmentRoiResult(
        sample_id=metadata.sample_id,
        spatial_group_id=metadata.spatial_group_id,
        split=metadata.split,
        days_between=metadata.days_between,
        correlation_bin=metadata.correlation_bin,
        selection_round=metadata.selection_round,
        rho=diagnostic.rho,
        constant_score=diagnostic.constant_score,
        aurc_gain=diagnostic.aurc_gain,
        high_risk_miss_rate_at_80=diagnostic.high_risk_miss_rate_at_80,
    )


def _descriptive_summary(
    summary: object, results: Sequence[DevelopmentRoiResult]
) -> dict[str, object]:
    payload = asdict(summary)
    payload["failure_reasons"] = list(payload["failure_reasons"])
    payload["mean_rho_ci95"] = list(payload["mean_rho_ci95"])
    payload["mean_aurc_gain_ci95"] = list(payload["mean_aurc_gain_ci95"])
    payload.update(_distribution_evidence(results))
    return payload


def _distribution_evidence(
    results: Sequence[DevelopmentRoiResult],
) -> dict[str, object]:
    rho = np.asarray([item.rho for item in results], dtype=np.float64)
    gains = np.asarray([item.aurc_gain for item in results], dtype=np.float64)
    return {
        "median_rho": float(np.median(rho)),
        "rho_quartiles": [float(value) for value in np.percentile(rho, (25, 75))],
        "median_aurc_gain": float(np.median(gains)),
        "aurc_gain_quartiles": [
            float(value) for value in np.percentile(gains, (25, 75))
        ],
        "stratum_mean_rho": [
            {
                "days_between": day,
                "correlation_bin": bin_index,
                "mean_rho": float(
                    np.mean(
                        [
                            item.rho
                            for item in results
                            if (item.days_between, item.correlation_bin)
                            == (day, bin_index)
                        ]
                    )
                ),
                "mean_aurc_gain": float(
                    np.mean(
                        [
                            item.aurc_gain
                            for item in results
                            if (item.days_between, item.correlation_bin)
                            == (day, bin_index)
                        ]
                    )
                ),
            }
            for day in (-1, 0, 1)
            for bin_index in range(4)
        ],
    }


def _sensitivity_summary(
    results: Sequence[DevelopmentRoiResult], indices: np.ndarray
) -> dict[str, object]:
    rho = np.asarray([item.rho for item in results], dtype=np.float64)
    gains = np.asarray([item.aurc_gain for item in results], dtype=np.float64)
    rho_ci = np.percentile(rho[indices].mean(axis=1), (2.5, 97.5))
    gain_ci = np.percentile(gains[indices].mean(axis=1), (2.5, 97.5))
    return {
        "selection_use": "descriptive_only",
        "nonconstant_count": sum(not item.constant_score for item in results),
        "mean_rho": float(rho.mean()),
        "mean_rho_ci95": [float(value) for value in rho_ci],
        "mean_aurc_gain": float(gains.mean()),
        "mean_aurc_gain_ci95": [float(value) for value in gain_ci],
        **_distribution_evidence(results),
    }


def _a2_result_and_audit(
    sample_records: Sequence[dict[str, object]],
    audit_groups: Sequence[dict[str, object]],
    primary_results: Mapping[str, Sequence[DevelopmentRoiResult]],
    sensitivity_results: Mapping[str, Sequence[DevelopmentRoiResult]],
    *,
    include_ldsr_variance_k5: bool,
    code_revision: str,
) -> tuple[dict[str, object], dict[str, object]]:
    names = _a2_candidate_names(include_ldsr_variance_k5)
    indices = score_selection.build_bootstrap_indices()
    candidate_summaries: list[dict[str, object]] = []
    primary_summary_payloads: list[dict[str, object]] = []
    for name in names:
        primary_summary = score_selection.summarize_candidate(
            name, primary_results[name], bootstrap_indices=indices
        )
        primary_payload = _descriptive_summary(primary_summary, primary_results[name])
        primary_summary_payloads.append(primary_payload)
        candidate_summaries.append(
            {
                "name": name,
                "operator_parameters": _a2_base_score_configuration(name),
                "seeds": list(_a2_score_seeds(name)),
                "primary_window_9": primary_payload,
                "sensitivity_window_1": _sensitivity_summary(
                    sensitivity_results[name], indices
                ),
            }
        )
    try:
        selected = score_selection.freeze_score(primary_results)
    except ValueError as exc:
        if type(exc) is not ValueError or str(exc) != "no development score candidate is eligible":
            raise
        frozen_score = None
        phase_decision = "stop_no_eligible_score"
    else:
        selected_summary = next(
            item for item in primary_summary_payloads if item["name"] == selected.name
        )
        frozen_score = {
            "name": selected.name,
            "operator_parameters": _a2_base_score_configuration(selected.name),
            "seeds": list(_a2_score_seeds(selected.name)),
            "post_manifest_sha256": POST_MANIFEST_SHA256,
            "code_revision": code_revision,
            "cost_rank": selected.cost_rank,
            "statistical_leader": selected.statistical_leader,
            "indistinguishable_candidates": list(
                selected.indistinguishable_candidates
            ),
            "selected_candidate_evidence": selected_summary,
            "candidate_eligibility_evidence": primary_summary_payloads,
        }
        phase_decision = "freeze_score"
    prediction_count = sum(len(group["prediction_entries"]) for group in audit_groups)
    score_count = sum(len(group["score_entries"]) for group in audit_groups)
    score_configuration = {
        name: _a2_base_score_configuration(name) for name in names
    }
    result: dict[str, object] = {
        "schema": A2_RESULT_SCHEMA,
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "radiometric_policy": _build_radiometric_policy(
            sample_records, expected_sample_count=120
        ),
        "dataset_role": "development_score_selection_only",
        "upstream": {
            "post_manifest_sha256": POST_MANIFEST_SHA256,
            "input_audit_sha256": INPUT_AUDIT_SHA256,
        },
        "code_revision": code_revision,
        "bands": ["B04", "B03", "B02", "B08"],
        "scale": 4,
        "sample_count": 120,
        "statistical_unit": "roi",
        "prediction_count": prediction_count,
        "score_count": score_count,
        "include_ldsr_variance_k5": include_ldsr_variance_k5,
        "candidate_names": list(names),
        "score_configuration": score_configuration,
        "risk_configuration": {
            "name": "local_l1_risk",
            "primary_window": PRIMARY_RISK_WINDOW,
            "sensitivity_window": SENSITIVITY_RISK_WINDOW,
        },
        "bootstrap": {
            "algorithm": "numpy.PCG64",
            "seed": score_selection.BOOTSTRAP_SEED,
            "resamples": score_selection.BOOTSTRAP_RESAMPLES,
            "ci_percentiles": [2.5, 97.5],
        },
        "selection_risk": "primary_window_9",
        "candidate_summaries": candidate_summaries,
        "frozen_score": frozen_score,
        "phase_decision": phase_decision,
        "samples": list(sample_records),
    }
    audit: dict[str, object] = {
        "schema": A2_CACHE_AUDIT_SCHEMA,
        "experiment_schema": A2_RESULT_SCHEMA,
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": INPUT_AUDIT_SHA256,
        "code_revision": code_revision,
        "sample_count": 120,
        "prediction_count": prediction_count,
        "score_count": score_count,
        "groups": list(audit_groups),
    }
    canonical_json(result)
    canonical_json(audit)
    return result, audit


def evaluate_a2_development(
    pairs: Sequence[LoadedCrosssensorPair],
    bundles: Iterable[DevelopmentPredictionBundle],
    *,
    prediction_cache: PredictionCache,
    score_cache: ScoreCache,
    include_ldsr_variance_k5: bool,
    code_revision: str,
    ordered_development_sample_ids: Sequence[str],
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate exactly 120 development ROI and freeze from primary R9 evidence."""

    names = _a2_candidate_names(include_ldsr_variance_k5)
    revision = _validate_code_revision(code_revision)
    if not isinstance(prediction_cache, PredictionCache) or not isinstance(
        score_cache, ScoreCache
    ):
        raise TypeError("A2 caches must be PredictionCache and ScoreCache values")
    pair_values = _validate_a2_structure(pairs)
    _validate_ordered_development_sample_ids(
        ordered_development_sample_ids, pair_values
    )
    try:
        bundle_iterator = iter(bundles)
    except TypeError as exc:
        raise TypeError("A2 bundles must be an iterable consumed exactly once") from exc
    sample_records: list[dict[str, object]] = []
    audit_groups: list[dict[str, object]] = []
    primary_results: dict[str, list[DevelopmentRoiResult]] = {
        name: [] for name in names
    }
    sensitivity_results: dict[str, list[DevelopmentRoiResult]] = {
        name: [] for name in names
    }
    for pair in pair_values:
        try:
            bundle = next(bundle_iterator)
        except StopIteration as exc:
            raise ValueError("A2 prediction bundle iterable ended before 120 ROI") from exc
        _validate_a2_bundle_tensors(
            pair, bundle, include_ldsr_variance_k5=include_ldsr_variance_k5
        )
        prediction_items = (bundle.bicubic, bundle.sen2srlite, *bundle.ldsr)
        prediction_entries = [
            _a2_prediction_evidence(pair, item, prediction_cache)
            for item in prediction_items
        ]
        scores = _build_a2_score_maps(
            pair,
            bundle,
            score_cache,
            include_ldsr_variance_k5=include_ldsr_variance_k5,
        )
        central = bundle.ldsr_for_seed(3407)
        primary_risk = local_l1_risk(
            central.tensor, pair.pair.hr, window=PRIMARY_RISK_WINDOW
        )
        sensitivity_risk = local_l1_risk(
            central.tensor, pair.pair.hr, window=SENSITIVITY_RISK_WINDOW
        )
        score_records: list[dict[str, object]] = []
        for score in scores:
            primary = evaluate_roi_score(score.tensor, primary_risk)
            sensitivity = evaluate_roi_score(score.tensor, sensitivity_risk)
            primary_results[score.name].append(_a2_roi_result(pair, primary))
            sensitivity_results[score.name].append(_a2_roi_result(pair, sensitivity))
            score_records.append(_a2_score_record(score, primary, sensitivity))
        sample_records.append(
            {
                "sample_id": pair.metadata.sample_id,
                "spatial_group_id": pair.metadata.spatial_group_id,
                "split": pair.metadata.split,
                "days_between": pair.metadata.days_between,
                "correlation_bin": pair.metadata.correlation_bin,
                "selection_round": pair.metadata.selection_round,
                "lr_tensor_sha256": tensor_sha256(pair.pair.lr),
                "hr_tensor_sha256": tensor_sha256(pair.pair.hr),
                "central_prediction_sha256": central.prediction_sha256,
                "radiometric_saturation": _serialize_radiometric_saturation(pair),
                "risks": {
                    "primary_window_9": tensor_sha256(primary_risk),
                    "sensitivity_window_1": tensor_sha256(sensitivity_risk),
                },
                "scores": score_records,
            }
        )
        audit_groups.append(
            {
                "sample_id": pair.metadata.sample_id,
                "spatial_group_id": pair.metadata.spatial_group_id,
                "days_between": pair.metadata.days_between,
                "correlation_bin": pair.metadata.correlation_bin,
                "selection_round": pair.metadata.selection_round,
                "prediction_entries": prediction_entries,
                "score_entries": [
                    _score_evidence(pair, score, score_cache) for score in scores
                ],
            }
        )
        del (
            prediction_items,
            scores,
            score,
            central,
            primary_risk,
            sensitivity_risk,
            bundle,
        )
    try:
        extra_bundle = next(bundle_iterator)
    except StopIteration:
        pass
    else:
        del extra_bundle
        raise ValueError("A2 prediction bundle iterable contains more than 120 ROI")
    return _a2_result_and_audit(
        sample_records,
        audit_groups,
        primary_results,
        sensitivity_results,
        include_ldsr_variance_k5=include_ldsr_variance_k5,
        code_revision=revision,
    )


def _validate_a2_committed_structure(
    pairs: Sequence[LoadedCrosssensorPair],
    committed_result: Mapping[str, object],
    committed_audit: Mapping[str, object],
) -> tuple[tuple[LoadedCrosssensorPair, ...], bool, str, tuple[str, ...]]:
    if committed_result.get("schema") != A2_RESULT_SCHEMA:
        raise ValueError("committed A2 result schema is invalid")
    if committed_audit.get("schema") != A2_CACHE_AUDIT_SCHEMA:
        raise ValueError("committed A2 audit schema is invalid")
    if committed_audit.get("experiment_schema") != A2_RESULT_SCHEMA:
        raise ValueError("committed A2 audit experiment schema is invalid")
    if (
        committed_result.get("normalization_policy")
        != PHASE2B3A_NORMALIZATION_POLICY
        or committed_audit.get("normalization_policy")
        != PHASE2B3A_NORMALIZATION_POLICY
    ):
        raise ValueError("committed A2 normalization policy is invalid")
    include = committed_result.get("include_ldsr_variance_k5")
    names = _a2_candidate_names(include)  # type: ignore[arg-type]
    revision = _validate_code_revision(committed_result.get("code_revision"))
    if committed_audit.get("code_revision") != revision:
        raise ValueError("committed A2 code revision differs between result and audit")
    expected_upstream = {
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": INPUT_AUDIT_SHA256,
    }
    if committed_result.get("upstream") != expected_upstream or any(
        committed_audit.get(key) != value for key, value in expected_upstream.items()
    ):
        raise ValueError("committed A2 upstream provenance is invalid")
    if committed_result.get("candidate_names") != list(names):
        raise ValueError("committed A2 candidate names are invalid")
    if committed_result.get("score_configuration") != {
        name: _a2_base_score_configuration(name) for name in names
    }:
        raise ValueError("committed A2 score configuration is invalid")
    candidate_summaries = committed_result.get("candidate_summaries")
    if not isinstance(candidate_summaries, list) or len(candidate_summaries) != len(names):
        raise ValueError("committed A2 candidate summaries are incomplete")
    for candidate, name in zip(candidate_summaries, names, strict=True):
        if (
            not isinstance(candidate, dict)
            or candidate.get("name") != name
            or candidate.get("operator_parameters")
            != _a2_base_score_configuration(name)
            or candidate.get("seeds") != list(_a2_score_seeds(name))
            or not isinstance(candidate.get("primary_window_9"), dict)
            or not isinstance(candidate.get("sensitivity_window_1"), dict)
        ):
            raise ValueError("committed A2 candidate summary is invalid")
    frozen = committed_result.get("frozen_score")
    decision = committed_result.get("phase_decision")
    if frozen is None:
        if decision != "stop_no_eligible_score" or any(
            candidate["primary_window_9"].get("eligible") is not False
            for candidate in candidate_summaries
        ):
            raise ValueError("committed A2 no-eligible stop evidence is invalid")
    else:
        if not isinstance(frozen, dict) or decision != "freeze_score":
            raise ValueError("committed A2 frozen score is invalid")
        selected_name = frozen.get("name")
        if (
            type(selected_name) is not str
            or selected_name not in names
            or frozen.get("operator_parameters")
            != _a2_base_score_configuration(selected_name)
            or frozen.get("seeds") != list(_a2_score_seeds(selected_name))
            or frozen.get("post_manifest_sha256") != POST_MANIFEST_SHA256
            or frozen.get("code_revision") != revision
            or not isinstance(frozen.get("selected_candidate_evidence"), dict)
            or not isinstance(frozen.get("candidate_eligibility_evidence"), list)
            or [
                evidence.get("name")
                for evidence in frozen["candidate_eligibility_evidence"]
                if isinstance(evidence, dict)
            ]
            != list(names)
        ):
            raise ValueError("committed A2 frozen score provenance is invalid")
    pair_values = _validate_a2_structure(pairs)
    samples = committed_result.get("samples")
    groups = committed_audit.get("groups")
    if not isinstance(samples, list) or len(samples) != 120:
        raise ValueError("committed A2 result must contain exactly 120 samples")
    _build_radiometric_policy(
        samples,
        expected_sample_count=120,
        claimed_policy=committed_result.get("radiometric_policy"),
    )
    if not isinstance(groups, list) or len(groups) != 120:
        raise ValueError("committed A2 audit must contain exactly 120 groups")
    prediction_count = 120 * len(_a2_prediction_slots(include))
    score_count = 120 * len(names)
    for payload, label in ((committed_result, "result"), (committed_audit, "audit")):
        if payload.get("sample_count") != 120:
            raise ValueError(f"committed A2 {label} sample count is invalid")
        if payload.get("prediction_count") != prediction_count:
            raise ValueError(f"committed A2 {label} prediction count is invalid")
        if payload.get("score_count") != score_count:
            raise ValueError(f"committed A2 {label} score count is invalid")
    for pair, sample, group in zip(pair_values, samples, groups, strict=True):
        expected = (
            pair.metadata.sample_id,
            pair.metadata.spatial_group_id,
            pair.metadata.days_between,
            pair.metadata.correlation_bin,
            pair.metadata.selection_round,
        )
        if not isinstance(sample, dict) or (
            sample.get("sample_id"),
            sample.get("spatial_group_id"),
            sample.get("days_between"),
            sample.get("correlation_bin"),
            sample.get("selection_round"),
        ) != expected:
            raise ValueError("committed A2 result sample order/membership is invalid")
        if sample.get("radiometric_saturation") != _serialize_radiometric_saturation(
            pair
        ):
            raise ValueError("committed A2 sample radiometric saturation is invalid")
        if not isinstance(group, dict) or (
            group.get("sample_id"),
            group.get("spatial_group_id"),
            group.get("days_between"),
            group.get("correlation_bin"),
            group.get("selection_round"),
        ) != expected:
            raise ValueError("committed A2 audit group order/membership is invalid")
        score_records = sample.get("scores")
        if not isinstance(score_records, list) or len(score_records) != len(names):
            raise ValueError("committed A2 sample diagnostics are incomplete")
        diagnostic_keys = {
            "rho",
            "constant_score",
            "coverages",
            "selective_mean_risks",
            "aurc",
            "random_aurc",
            "aurc_gain",
            "high_risk_miss_rate_at_80",
        }
        for score_record, name in zip(score_records, names, strict=True):
            if (
                not isinstance(score_record, dict)
                or score_record.get("name") != name
                or not isinstance(score_record.get("primary_window_9"), dict)
                or set(score_record["primary_window_9"]) != diagnostic_keys
                or not isinstance(score_record.get("sensitivity_window_1"), dict)
                or set(score_record["sensitivity_window_1"]) != diagnostic_keys
                or score_record["primary_window_9"].get("coverages")
                != [index / 10 for index in range(1, 11)]
                or score_record["sensitivity_window_1"].get("coverages")
                != [index / 10 for index in range(1, 11)]
                or len(score_record["primary_window_9"].get("selective_mean_risks", []))
                != 10
                or len(score_record["sensitivity_window_1"].get("selective_mean_risks", []))
                != 10
            ):
                raise ValueError("committed A2 sample diagnostics are incomplete")
    return pair_values, include, revision, names


def _a2_canonical_prediction_identities(
    pairs: Sequence[LoadedCrosssensorPair],
    groups: Sequence[dict[str, object]],
    *,
    include_ldsr_variance_k5: bool,
    names: Sequence[str],
) -> tuple[tuple[PredictionIdentity, ...], ...]:
    slots = _a2_prediction_slots(include_ldsr_variance_k5)
    prediction_groups: list[tuple[PredictionIdentity, ...]] = []
    for pair, group in zip(pairs, groups, strict=True):
        prediction_entries = group.get("prediction_entries")
        score_entries = group.get("score_entries")
        if not isinstance(prediction_entries, list) or len(prediction_entries) != len(slots):
            raise ValueError("committed A2 prediction group is incomplete")
        if not isinstance(score_entries, list) or len(score_entries) != len(names):
            raise ValueError("committed A2 score group is incomplete")
        canonical_predictions: list[PredictionIdentity] = []
        for entry, (model_name, seed) in zip(prediction_entries, slots, strict=True):
            if not isinstance(entry, dict):
                raise ValueError("committed A2 prediction entry is invalid")
            parsed = _prediction_identity_from_dict(entry.get("identity"))
            provenance = dict(parsed.model_provenance)
            if (
                provenance.get("name") != model_name
                or provenance.get("scale") != 4
                or provenance.get("experiment_schema") != PREDICTION_EXPERIMENT_SCHEMA
                or provenance.get("post_manifest_sha256") != POST_MANIFEST_SHA256
                or provenance.get("input_audit_sha256") != INPUT_AUDIT_SHA256
                or provenance.get("normalization_policy")
                != PHASE2B3A_NORMALIZATION_POLICY
                or (seed is None and "seed" in provenance)
                or (seed is not None and provenance.get("seed") != seed)
            ):
                raise ValueError("committed A2 prediction provenance is invalid")
            canonical = build_identity(
                provenance, pair.pair.source, pair.pair.sample_id, pair.pair.lr
            )
            digest = entry.get("prediction_sha256")
            if (
                parsed.as_dict() != canonical.as_dict()
                or entry.get("sample_id") != pair.pair.sample_id
                or entry.get("model_name") != model_name
                or entry.get("seed") != seed
                or entry.get("cache_key") != canonical.key
                or type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("committed A2 prediction identity/key/digest is invalid")
            canonical_predictions.append(canonical)
        prediction_groups.append(tuple(canonical_predictions))
    return tuple(prediction_groups)


def _verify_a2_prediction_hashes(
    pairs: Sequence[LoadedCrosssensorPair],
    groups: Sequence[dict[str, object]],
    identity_groups: Sequence[Sequence[PredictionIdentity]],
    prediction_cache: PredictionCache,
    *,
    include_ldsr_variance_k5: bool,
) -> tuple[tuple[str, ...], ...]:
    verified_groups: list[tuple[str, ...]] = []
    for pair, group, identities in zip(
        pairs, groups, identity_groups, strict=True
    ):
        bundle = _a2_bundle_from_cache(
            pair,
            identities,
            group["prediction_entries"],
            prediction_cache,
            include_ldsr_variance_k5=include_ldsr_variance_k5,
        )
        verified_groups.append(
            tuple(
                item.prediction_sha256
                for item in (bundle.bicubic, bundle.sen2srlite, *bundle.ldsr)
            )
        )
        del bundle
    return tuple(verified_groups)


def _a2_canonical_score_identities(
    pairs: Sequence[LoadedCrosssensorPair],
    groups: Sequence[dict[str, object]],
    verified_prediction_hash_groups: Sequence[Sequence[str]],
    *,
    include_ldsr_variance_k5: bool,
    names: Sequence[str],
) -> tuple[tuple[ScoreIdentity, ...], ...]:
    slots = _a2_prediction_slots(include_ldsr_variance_k5)
    score_groups: list[tuple[ScoreIdentity, ...]] = []
    for pair, group, verified_hashes in zip(
        pairs, groups, verified_prediction_hash_groups, strict=True
    ):
        if len(verified_hashes) != len(slots):
            raise ValueError("verified A2 prediction hash group is incomplete")
        prediction_sha256s = {
            slot: digest for slot, digest in zip(slots, verified_hashes, strict=True)
        }
        expected_inputs = {
            "lr_reprojection_l1": (prediction_sha256s[("ldsr-s2-x4", 3407)],),
            "three_model_disagreement": (
                prediction_sha256s[("bicubic-x4", None)],
                prediction_sha256s[("sen2srlite-x4", None)],
                prediction_sha256s[("ldsr-s2-x4", 3407)],
            ),
            "ldsr_variance_k5": tuple(
                prediction_sha256s[("ldsr-s2-x4", seed)] for seed in K5A_SEEDS
            )
            if include_ldsr_variance_k5
            else (),
        }
        canonical_scores: list[ScoreIdentity] = []
        score_entries = group["score_entries"]
        for entry, name in zip(score_entries, names, strict=True):
            if not isinstance(entry, dict):
                raise ValueError("committed A2 score entry is invalid")
            canonical = _a2_score_identity(
                pair, name=name, input_sha256s=expected_inputs[name]
            )
            parsed = _score_identity_from_dict(entry.get("identity"))
            digest = entry.get("score_sha256")
            if (
                parsed.as_dict() != canonical.as_dict()
                or entry.get("sample_id") != pair.pair.sample_id
                or entry.get("name") != name
                or entry.get("cache_key") != canonical.key
                or type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("committed A2 score identity/key/digest is invalid")
            canonical_scores.append(canonical)
        score_groups.append(tuple(canonical_scores))
    return tuple(score_groups)


def _a2_bundle_from_cache(
    pair: LoadedCrosssensorPair,
    identities: Sequence[PredictionIdentity],
    entries: Sequence[dict[str, object]],
    prediction_cache: PredictionCache,
    *,
    include_ldsr_variance_k5: bool,
) -> DevelopmentPredictionBundle:
    records: list[CachedDevelopmentPrediction] = []
    for identity, entry, (model_name, seed) in zip(
        identities,
        entries,
        _a2_prediction_slots(include_ldsr_variance_k5),
        strict=True,
    ):
        tensor = prediction_cache.get(identity)
        if tensor is None:
            raise RuntimeError("canonical A2 prediction cache entry is missing during replay")
        digest = tensor_sha256(tensor)
        if entry.get("prediction_sha256") != digest:
            raise ValueError("committed A2 prediction logical tensor SHA is invalid")
        record = CachedDevelopmentPrediction(
            model_name=model_name,
            seed=seed,
            identity=identity,
            prediction_sha256=digest,
            tensor=tensor,
        )
        _validate_prediction_tensor(pair, record, expected_name=model_name, expected_seed=seed)
        records.append(record)
    return DevelopmentPredictionBundle(
        sample_id=pair.pair.sample_id,
        bicubic=records[0],
        sen2srlite=records[1],
        ldsr=tuple(records[2:]),
    )


def _a2_scores_from_cache(
    pair: LoadedCrosssensorPair,
    bundle: DevelopmentPredictionBundle,
    identities: Sequence[ScoreIdentity],
    entries: Sequence[dict[str, object]],
    score_cache: ScoreCache,
    names: Sequence[str],
) -> tuple[CachedScoreMap, ...]:
    records: list[CachedScoreMap] = []
    for name, identity, entry in zip(names, identities, entries, strict=True):
        expected = _a2_score_identity(
            pair, name=name, input_sha256s=_a2_score_input_hashes(bundle, name)
        )
        if identity.as_dict() != expected.as_dict():
            raise ValueError("committed A2 score identity differs from verified predictions")
        tensor = score_cache.get(identity)
        if tensor is None:
            raise RuntimeError("canonical A2 score cache entry is missing during replay")
        digest = tensor_sha256(tensor)
        if entry.get("score_sha256") != digest:
            raise ValueError("committed A2 score logical tensor SHA is invalid")
        records.append(CachedScoreMap(name, identity, digest, tensor))
    return tuple(records)


def replay_a2_development(
    pairs: Sequence[LoadedCrosssensorPair],
    committed_result: Mapping[str, object],
    committed_audit: Mapping[str, object],
    prediction_cache: PredictionCache,
    score_cache: ScoreCache,
    *,
    ordered_development_sample_ids: Sequence[str],
) -> tuple[dict[str, object], dict[str, object]]:
    """Rebuild A2 from existing caches without any model, score compute, or write path."""

    pair_values, include, revision, names = _validate_a2_committed_structure(
        pairs, committed_result, committed_audit
    )
    _validate_ordered_development_sample_ids(
        ordered_development_sample_ids, pair_values
    )
    groups = committed_audit["groups"]
    prediction_groups = _a2_canonical_prediction_identities(
        pair_values,
        groups,
        include_ldsr_variance_k5=include,
        names=names,
    )
    prediction_identities = tuple(
        identity for group in prediction_groups for identity in group
    )
    prediction_before = snapshot_cache_files(
        prediction_cache.root, prediction_identities
    )
    verified_prediction_hash_groups = _verify_a2_prediction_hashes(
        pair_values,
        groups,
        prediction_groups,
        prediction_cache,
        include_ldsr_variance_k5=include,
    )
    score_identity_groups = _a2_canonical_score_identities(
        pair_values,
        groups,
        verified_prediction_hash_groups,
        include_ldsr_variance_k5=include,
        names=names,
    )
    score_identities = tuple(
        identity for group in score_identity_groups for identity in group
    )
    score_before = _snapshot_score_files(score_cache, score_identities)
    sample_records: list[dict[str, object]] = []
    rebuilt_groups: list[dict[str, object]] = []
    primary_results: dict[str, list[DevelopmentRoiResult]] = {
        name: [] for name in names
    }
    sensitivity_results: dict[str, list[DevelopmentRoiResult]] = {
        name: [] for name in names
    }
    for pair, group, prediction_ids, score_ids in zip(
        pair_values, groups, prediction_groups, score_identity_groups, strict=True
    ):
        prediction_entries = group["prediction_entries"]
        score_entries = group["score_entries"]
        bundle = _a2_bundle_from_cache(
            pair,
            prediction_ids,
            prediction_entries,
            prediction_cache,
            include_ldsr_variance_k5=include,
        )
        scores = _a2_scores_from_cache(
            pair, bundle, score_ids, score_entries, score_cache, names
        )
        central = bundle.ldsr_for_seed(3407)
        primary_risk = local_l1_risk(
            central.tensor, pair.pair.hr, window=PRIMARY_RISK_WINDOW
        )
        sensitivity_risk = local_l1_risk(
            central.tensor, pair.pair.hr, window=SENSITIVITY_RISK_WINDOW
        )
        rebuilt_score_records: list[dict[str, object]] = []
        for score in scores:
            primary = evaluate_roi_score(score.tensor, primary_risk)
            sensitivity = evaluate_roi_score(score.tensor, sensitivity_risk)
            primary_results[score.name].append(_a2_roi_result(pair, primary))
            sensitivity_results[score.name].append(_a2_roi_result(pair, sensitivity))
            rebuilt_score_records.append(_a2_score_record(score, primary, sensitivity))
        sample_records.append(
            {
                "sample_id": pair.metadata.sample_id,
                "spatial_group_id": pair.metadata.spatial_group_id,
                "split": pair.metadata.split,
                "days_between": pair.metadata.days_between,
                "correlation_bin": pair.metadata.correlation_bin,
                "selection_round": pair.metadata.selection_round,
                "lr_tensor_sha256": tensor_sha256(pair.pair.lr),
                "hr_tensor_sha256": tensor_sha256(pair.pair.hr),
                "central_prediction_sha256": central.prediction_sha256,
                "radiometric_saturation": _serialize_radiometric_saturation(pair),
                "risks": {
                    "primary_window_9": tensor_sha256(primary_risk),
                    "sensitivity_window_1": tensor_sha256(sensitivity_risk),
                },
                "scores": rebuilt_score_records,
            }
        )
        rebuilt_groups.append(
            {
                "sample_id": pair.metadata.sample_id,
                "spatial_group_id": pair.metadata.spatial_group_id,
                "days_between": pair.metadata.days_between,
                "correlation_bin": pair.metadata.correlation_bin,
                "selection_round": pair.metadata.selection_round,
                "prediction_entries": [
                    _a2_prediction_evidence(pair, item, prediction_cache)
                    for item in (bundle.bicubic, bundle.sen2srlite, *bundle.ldsr)
                ],
                "score_entries": [
                    _score_evidence(pair, score, score_cache) for score in scores
                ],
            }
        )
        del bundle, scores, score, central, primary_risk, sensitivity_risk
    rebuilt_result, rebuilt_audit = _a2_result_and_audit(
        sample_records,
        rebuilt_groups,
        primary_results,
        sensitivity_results,
        include_ldsr_variance_k5=include,
        code_revision=revision,
    )
    if canonical_json(rebuilt_result) != canonical_json(dict(committed_result)):
        raise ValueError("rebuilt A2 result differs from committed result")
    if canonical_json(rebuilt_audit) != canonical_json(dict(committed_audit)):
        raise ValueError("rebuilt A2 cache audit differs from committed audit")
    prediction_after = snapshot_cache_files(
        prediction_cache.root, prediction_identities
    )
    score_after = _snapshot_score_files(score_cache, score_identities)
    if prediction_before != prediction_after or score_before != score_after:
        raise RuntimeError("cache files changed during A2 replay")
    return rebuilt_result, rebuilt_audit
