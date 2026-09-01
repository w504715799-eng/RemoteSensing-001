"""Deterministic Phase 2B2-B development smoke evaluation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch

from trustsr.artifacts.predictions import (
    PredictionCache,
    PredictionIdentity,
    build_identity,
    tensor_sha256,
)
from trustsr.contracts import SRPair
from trustsr.data.crosssensor_manifest import SOURCE_OBJECT_SHA256
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    NORMALIZATION_POLICY,
    PHASE2B1B_AUDIT_SHA256,
    POST_MANIFEST_SHA256,
    LoadedCrosssensorPair,
)
from trustsr.data.input_audit import AUDIT_SCHEMA
from trustsr.data.subset_manifest import BASE_MANIFEST_SHA256
from trustsr.evaluation.opensr_metrics import METRIC_KEYS, compute_opensr_metrics
from trustsr.jsonio import canonical_json
from trustsr.models.protocols import JsonScalar, SRModel

EXPERIMENT_SCHEMA = "trustsr.phase2b2b-development-smoke.v1"
CACHE_AUDIT_SCHEMA = "trustsr.phase2b2b-cache-audit.v1"
INPUT_AUDIT_SHA256 = "fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b"
MODEL_NAMES = ("bicubic-x4", "sen2srlite-x4", "ldsr-s2-x4")
_CONTEXT_KEYS = (
    "experiment_schema",
    "post_manifest_sha256",
    "input_audit_sha256",
)


def build_cache_provenance(
    model_provenance: Mapping[str, JsonScalar],
) -> dict[str, JsonScalar]:
    """Bind model provenance to the frozen crosssensor experiment context."""

    if not isinstance(model_provenance, Mapping):
        raise TypeError("model provenance must be a mapping")
    if any(key in model_provenance for key in _CONTEXT_KEYS):
        raise ValueError("model provenance contains a reserved experiment context key")
    result = dict(model_provenance)
    result.update(
        {
            "experiment_schema": EXPERIMENT_SCHEMA,
            "post_manifest_sha256": POST_MANIFEST_SHA256,
            "input_audit_sha256": INPUT_AUDIT_SHA256,
        }
    )
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_files(cache_root: Path, identity: PredictionIdentity) -> tuple[Path, Path]:
    if not isinstance(cache_root, Path) or cache_root.is_symlink() or not cache_root.is_dir():
        raise ValueError("prediction cache root must be an existing non-symlink directory")
    paths = tuple(cache_root / f"{identity.key}{suffix}" for suffix in (".json", ".safetensors"))
    named_paths = {
        path for path in cache_root.iterdir() if path.name.startswith(f"{identity.key}.")
    }
    if named_paths != set(paths):
        raise ValueError("named prediction cache entry must contain exactly two files")
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("named prediction cache file must be a regular file")
    return paths


def cache_entry_evidence(
    cache_root: Path, identity: PredictionIdentity
) -> dict[str, object]:
    """Return host-free evidence for one fully validated prediction cache entry."""

    prediction = PredictionCache(cache_root).get(identity)
    if prediction is None:
        raise ValueError("named prediction cache entry is missing")
    paths = _cache_files(cache_root, identity)
    return {
        "cache_key": identity.key,
        "lr": identity.as_dict()["lr"],
        "prediction_sha256": tensor_sha256(prediction),
        "files": [
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in paths
        ],
    }


def snapshot_cache_files(
    cache_root: Path, identities: Sequence[PredictionIdentity]
) -> tuple[tuple[object, ...], ...]:
    """Snapshot bytes and metadata needed to prove replay does not mutate caches."""

    keys = [identity.key for identity in identities]
    if len(set(keys)) != len(keys):
        raise ValueError("prediction identities must be unique")
    result: list[tuple[object, ...]] = []
    for identity in identities:
        for path in _cache_files(cache_root, identity):
            stat = path.stat()
            result.append(
                (path.name, stat.st_size, stat.st_mtime_ns, _file_sha256(path))
            )
    return tuple(result)


def _validate_pairs(
    loaded_pairs: Sequence[LoadedCrosssensorPair], expected_sample_count: int
) -> tuple[LoadedCrosssensorPair, ...]:
    if type(expected_sample_count) is not int or expected_sample_count not in {1, 4}:
        raise ValueError("expected sample count must be exactly one or four")
    if len(loaded_pairs) != expected_sample_count:
        label = "one" if expected_sample_count == 1 else "four"
        raise ValueError(f"development smoke requires exactly {label} pairs")
    result = tuple(loaded_pairs)
    for bin_index, loaded in enumerate(result):
        if not isinstance(loaded, LoadedCrosssensorPair):
            raise TypeError("development smoke inputs must be loaded crosssensor pairs")
        loaded.pair.validate()
        metadata = loaded.metadata
        if metadata.split != "development":
            raise ValueError("development smoke cannot consume another split")
        if metadata.correlation_bin != bin_index:
            raise ValueError("development smoke pairs must use canonical correlation-bin order")
        if metadata.selection_round != 1 or metadata.days_between != -1:
            raise ValueError("development smoke pair is outside the frozen selection")
        if metadata.manifest_sha256 != POST_MANIFEST_SHA256:
            raise ValueError("development smoke pair has the wrong manifest")
        if metadata.sample_id != loaded.pair.sample_id:
            raise ValueError("development smoke pair and metadata identities differ")
        if loaded.pair.source != f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}":
            raise ValueError("development smoke pair has the wrong source")
        if (
            metadata.crop_policy != CROP_POLICY
            or metadata.normalization_policy != NORMALIZATION_POLICY
        ):
            raise ValueError("development smoke pair has the wrong input policy")
    sample_ids = [loaded.metadata.sample_id for loaded in result]
    groups = [loaded.metadata.spatial_group_id for loaded in result]
    if len(set(sample_ids)) != len(sample_ids) or len(set(groups)) != len(groups):
        raise ValueError("development smoke identities and spatial groups must be unique")
    return result


def _validate_models(
    models: Sequence[SRModel],
) -> tuple[tuple[SRModel, ...], tuple[Mapping[str, JsonScalar], ...]]:
    result = tuple(models)
    if tuple(model.name for model in result) != MODEL_NAMES:
        raise ValueError("development smoke model order must match the frozen three-model order")
    provenances: list[Mapping[str, JsonScalar]] = []
    for model in result:
        if model.scale != 4:
            raise ValueError(f"model {model.name!r} must use scale 4")
        provenance = model.provenance()
        if provenance.get("name") != model.name or provenance.get("scale") != 4:
            raise ValueError(f"model {model.name!r} provenance does not identify the model")
        build_cache_provenance(provenance)
        provenances.append(provenance)
    return result, tuple(provenances)


def _validate_prediction(pair: SRPair, prediction: torch.Tensor) -> torch.Tensor:
    if not isinstance(prediction, torch.Tensor) or prediction.dtype != torch.float32:
        raise ValueError("prediction must be a torch.float32 tensor")
    if tuple(prediction.shape) != tuple(pair.hr.shape):
        raise ValueError("prediction shape must match the frozen HR tensor")
    if prediction.device.type != "cpu" or not prediction.is_contiguous():
        raise ValueError("prediction must be contiguous on CPU")
    if prediction.requires_grad:
        raise ValueError("prediction must be detached")
    if not torch.isfinite(prediction).all() or (prediction < 0).any() or (prediction > 1).any():
        raise ValueError("prediction must be finite and in [0, 1]")
    return prediction


def _finite_metrics(values: Mapping[str, float], sample_id: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in METRIC_KEYS:
        try:
            value = float(values[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"missing finite metric {key!r} for {sample_id!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"metric {key!r} must be finite for {sample_id!r}")
        result[key] = value
    return result


def _sample_records(loaded_pairs: Sequence[LoadedCrosssensorPair]) -> list[dict[str, object]]:
    return [
        {
            "sample_id": loaded.metadata.sample_id,
            "correlation_bin": loaded.metadata.correlation_bin,
            "spatial_group_id": loaded.metadata.spatial_group_id,
            "lr_tensor_sha256": tensor_sha256(loaded.pair.lr),
            "hr_tensor_sha256": tensor_sha256(loaded.pair.hr),
        }
        for loaded in loaded_pairs
    ]


def _upstream() -> dict[str, object]:
    return {
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": INPUT_AUDIT_SHA256,
        "input_audit_schema": AUDIT_SCHEMA,
        "phase2b1b_audit_sha256": PHASE2B1B_AUDIT_SHA256,
        "base_manifest_sha256": BASE_MANIFEST_SHA256,
        "source_object_sha256": SOURCE_OBJECT_SHA256,
    }


def _evaluate_model(
    loaded_pairs: Sequence[LoadedCrosssensorPair],
    model_provenance: Mapping[str, JsonScalar],
    cache_root: Path,
    metric_fn: Callable[[SRPair, torch.Tensor], Mapping[str, float]],
    model: SRModel | None,
) -> tuple[dict[str, object], list[dict[str, object]], list[PredictionIdentity]]:
    cache_provenance = build_cache_provenance(model_provenance)
    cache = PredictionCache(cache_root)
    predictions: list[dict[str, object]] = []
    audit_entries: list[dict[str, object]] = []
    identities: list[PredictionIdentity] = []
    for loaded in loaded_pairs:
        pair = loaded.pair
        identity = build_identity(cache_provenance, pair.source, pair.sample_id, pair.lr)
        identities.append(identity)
        prediction = cache.get(identity)
        if prediction is None:
            if model is None:
                raise ValueError(f"prediction cache is missing for {pair.sample_id!r}")
            generated = _validate_prediction(pair, model.predict(pair.lr))
            cache.put(identity, generated)
            prediction = cache.get(identity)
            if prediction is None:
                raise RuntimeError("prediction cache disappeared after commit")
            if not torch.equal(generated, prediction):
                raise RuntimeError("prediction cache differs after commit")
        prediction = _validate_prediction(pair, prediction)
        metrics = _finite_metrics(metric_fn(pair, prediction), pair.sample_id)
        prediction_sha256 = tensor_sha256(prediction)
        predictions.append(
            {
                "sample_id": pair.sample_id,
                "correlation_bin": loaded.metadata.correlation_bin,
                "cache_key": identity.key,
                "prediction_sha256": prediction_sha256,
                "metrics": metrics,
            }
        )
        evidence = cache_entry_evidence(cache_root, identity)
        if evidence["prediction_sha256"] != prediction_sha256:
            raise RuntimeError("prediction evidence changed after metric computation")
        audit_entries.append(
            {
                "model_name": model_provenance["name"],
                "sample_id": pair.sample_id,
                "correlation_bin": loaded.metadata.correlation_bin,
                **evidence,
            }
        )
    means = {
        key: sum(float(item["metrics"][key]) for item in predictions) / len(predictions)
        for key in METRIC_KEYS
    }
    model_result = {
        "name": model_provenance["name"],
        "model_provenance": dict(model_provenance),
        "cache_provenance": cache_provenance,
        "predictions": predictions,
        "mean_metrics": _finite_metrics(means, "mean"),
    }
    return model_result, audit_entries, identities


def _build_outputs(
    loaded_pairs: Sequence[LoadedCrosssensorPair],
    provenances: Sequence[Mapping[str, JsonScalar]],
    cache_root: Path,
    metric_fn: Callable[[SRPair, torch.Tensor], Mapping[str, float]],
    models: Sequence[SRModel | None],
) -> tuple[dict[str, object], dict[str, object], list[PredictionIdentity]]:
    model_results: list[dict[str, object]] = []
    audit_entries: list[dict[str, object]] = []
    identities: list[PredictionIdentity] = []
    for provenance, model in zip(provenances, models, strict=True):
        model_result, model_audit, model_identities = _evaluate_model(
            loaded_pairs, provenance, cache_root, metric_fn, model
        )
        model_results.append(model_result)
        audit_entries.extend(model_audit)
        identities.extend(model_identities)
    prediction_count = len(loaded_pairs) * len(provenances)
    result: dict[str, object] = {
        "schema": EXPERIMENT_SCHEMA,
        "dataset_role": "development_engineering_smoke_only",
        "upstream": _upstream(),
        "bands": ["B04", "B03", "B02", "B08"],
        "scale": 4,
        "sample_count": len(loaded_pairs),
        "model_count": len(provenances),
        "prediction_count": prediction_count,
        "samples": _sample_records(loaded_pairs),
        "models": model_results,
    }
    audit: dict[str, object] = {
        "schema": CACHE_AUDIT_SCHEMA,
        "experiment_schema": EXPERIMENT_SCHEMA,
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": INPUT_AUDIT_SHA256,
        "prediction_count": prediction_count,
        "entries": audit_entries,
    }
    canonical_json(result)
    canonical_json(audit)
    return result, audit, identities


def evaluate_development_smoke(
    loaded_pairs: Sequence[LoadedCrosssensorPair],
    models: Sequence[SRModel],
    cache_root: Path,
    *,
    metric_fn: Callable[[SRPair, torch.Tensor], Mapping[str, float]] = compute_opensr_metrics,
    expected_sample_count: int = 4,
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate the frozen three-model grid, resuming only valid cache entries."""

    pairs = _validate_pairs(loaded_pairs, expected_sample_count)
    checked_models, provenances = _validate_models(models)
    result, audit, _ = _build_outputs(
        pairs, provenances, cache_root, metric_fn, checked_models
    )
    return result, audit


def _committed_provenances(
    committed_result: Mapping[str, object], committed_audit: Mapping[str, object]
) -> tuple[Mapping[str, JsonScalar], ...]:
    if committed_result.get("schema") != EXPERIMENT_SCHEMA:
        raise ValueError("committed result schema is invalid")
    if committed_result.get("sample_count") != 4 or committed_result.get(
        "prediction_count"
    ) != 12:
        raise ValueError("committed result counts are invalid")
    if committed_audit.get("schema") != CACHE_AUDIT_SCHEMA:
        raise ValueError("committed cache audit schema is invalid")
    entries = committed_audit.get("entries")
    if committed_audit.get("prediction_count") != 12 or not isinstance(entries, list) or len(
        entries
    ) != 12:
        raise ValueError("committed cache audit count is invalid")
    records = committed_result.get("models")
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("committed result models are invalid")
    provenances: list[Mapping[str, JsonScalar]] = []
    for expected_name, record in zip(MODEL_NAMES, records, strict=True):
        if not isinstance(record, dict) or record.get("name") != expected_name:
            raise ValueError("committed result model order is invalid")
        provenance = record.get("model_provenance")
        if not isinstance(provenance, dict):
            raise ValueError("committed result model provenance is invalid")
        if provenance.get("name") != expected_name or provenance.get("scale") != 4:
            raise ValueError("committed result model provenance is invalid")
        if record.get("cache_provenance") != build_cache_provenance(provenance):
            raise ValueError("committed result cache provenance is invalid")
        provenances.append(provenance)
    return tuple(provenances)


def replay_development_smoke(
    loaded_pairs: Sequence[LoadedCrosssensorPair],
    committed_result: Mapping[str, object],
    committed_audit: Mapping[str, object],
    cache_root: Path,
    *,
    metric_fn: Callable[[SRPair, torch.Tensor], Mapping[str, float]] = compute_opensr_metrics,
) -> tuple[dict[str, object], dict[str, object]]:
    """Rebuild deterministic evidence entirely from committed prediction caches."""

    pairs = _validate_pairs(loaded_pairs, 4)
    provenances = _committed_provenances(committed_result, committed_audit)
    identities = [
        build_identity(
            build_cache_provenance(provenance),
            pair.pair.source,
            pair.pair.sample_id,
            pair.pair.lr,
        )
        for provenance in provenances
        for pair in pairs
    ]
    before = snapshot_cache_files(cache_root, identities)
    rebuilt_result, rebuilt_audit, rebuilt_identities = _build_outputs(
        pairs, provenances, cache_root, metric_fn, (None, None, None)
    )
    if [identity.key for identity in rebuilt_identities] != [
        identity.key for identity in identities
    ]:
        raise RuntimeError("cache identities changed during replay")
    if rebuilt_result != dict(committed_result):
        raise ValueError("rebuilt result differs from committed result")
    if rebuilt_audit != dict(committed_audit):
        raise ValueError("rebuilt cache audit differs from committed audit")
    after = snapshot_cache_files(cache_root, identities)
    if after != before:
        raise RuntimeError("prediction cache files changed during replay")
    return rebuilt_result, rebuilt_audit
