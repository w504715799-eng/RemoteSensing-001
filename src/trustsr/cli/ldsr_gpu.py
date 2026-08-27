"""Staged, CUDA-gated LDSR-S2 reproducibility workflow."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from trustsr.artifacts.gpu_run import (
    GPUHardwareSnapshot,
    _atomic_json,
    capture_gpu_hardware,
    collect_gpu_environment,
    stage_artifact_file,
    write_artifact_manifest,
)
from trustsr.artifacts.predictions import (
    PredictionCache,
    PredictionIdentity,
    build_identity,
    canonical_json,
    tensor_sha256,
)
from trustsr.cli.benchmark_baselines import (
    EXPECTED_SPOT_V3_IDENTITIES,
    _production_environment,
    run_benchmark,
)
from trustsr.contracts import SRPair
from trustsr.data.opensr import load_opensr_pairs
from trustsr.evaluation.opensr_metrics import METRIC_KEYS, compute_opensr_metrics
from trustsr.evaluation.repeatability import run_repeatability
from trustsr.models.bicubic import BicubicX4
from trustsr.models.ldsr_s2 import LDSRS2X4
from trustsr.models.sen2srlite import SEN2SRLiteX4

_PHASE_DIRECTORY = Path("phase1b")
_ENVIRONMENT_NAME = "environment.json"
_SINGLE_NAME = "single.json"
_SINGLE_RUNTIME_NAME = "single-runtime.json"
_BENCHMARK_NAME = "spot-v3-three-models.json"
_CACHE_INDEX_NAME = "ldsr-cache-index.json"


def _phase_path(args: argparse.Namespace, filename: str) -> Path:
    return Path(args.artifacts_dir) / _PHASE_DIRECTORY / filename


def _require_expected_spot_v3_identities(pairs: Sequence[SRPair]) -> None:
    identities = [(pair.source, pair.sample_id) for pair in pairs]
    if len(pairs) != len(EXPECTED_SPOT_V3_IDENTITIES) or set(identities) != set(
        EXPECTED_SPOT_V3_IDENTITIES
    ):
        raise ValueError("loaded SPOT v3 pairs do not match the complete expected nine-sample set")
    if len(set(identities)) != len(identities):
        raise ValueError("loaded SPOT v3 pairs are not unique")
    for pair in pairs:
        pair.validate()


def _load_nine_pairs(args: argparse.Namespace) -> list[SRPair]:
    pairs = load_opensr_pairs(
        "spot", Path(args.dataset_cache_dir), "v3", limit=9, expected_count=9
    )
    _require_expected_spot_v3_identities(pairs)
    return pairs


def _finite_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in METRIC_KEYS:
        try:
            value = float(metrics[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"missing or invalid metric: {key}") from exc
        if not math.isfinite(value):
            raise ValueError(f"metric is non-finite: {key}")
        result[key] = value
    return result


def _write_environment(
    args: argparse.Namespace, model: LDSRS2X4, snapshot: GPUHardwareSnapshot
) -> dict[str, object]:
    environment: dict[str, object] = dict(
        collect_gpu_environment(
            hardware_snapshot=snapshot,
            project_root=getattr(args, "project_root", None),
        )
    )
    environment["model_provenance"] = model.provenance()
    _atomic_json(_phase_path(args, _ENVIRONMENT_NAME), environment)
    return environment


def run_preflight(args: argparse.Namespace) -> dict[str, object]:
    """Verify CUDA and LDSR assets, then record the non-deterministic runtime."""
    snapshot = capture_gpu_hardware()
    model = LDSRS2X4.from_pretrained(Path(args.ldsr_model_dir), device="cuda:0")
    return _write_environment(args, model, snapshot)


def run_single(args: argparse.Namespace) -> dict[str, object]:
    """Run the immutable spot-0000 repeatability gate and cache its prediction."""
    capture_gpu_hardware()
    model = LDSRS2X4.from_pretrained(Path(args.ldsr_model_dir), device="cuda:0")
    pairs = _load_nine_pairs(args)
    pair = next(pair for pair in pairs if pair.sample_id == "spot-0000")

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    prediction, repeatability = run_repeatability(model, pair.lr, tolerance=1e-6)
    duration_seconds = time.monotonic() - started
    peak_memory_bytes = int(torch.cuda.max_memory_allocated())

    provenance = model.provenance()
    identity = build_identity(provenance, pair.source, pair.sample_id, pair.lr)
    cache = PredictionCache(Path(args.prediction_cache_dir))
    cache.put(identity, prediction)
    cached = cache.get(identity)
    if cached is None or not torch.equal(cached, prediction):
        raise RuntimeError("prediction cache verification failed")
    result: dict[str, object] = {
        "schema_version": 1,
        "source": pair.source,
        "sample_id": pair.sample_id,
        "lr_sha256": tensor_sha256(pair.lr),
        "model_provenance": provenance,
        "repeatability": repeatability.as_dict(),
        "cache_key": identity.key,
        "metrics": _finite_metrics(compute_opensr_metrics(pair, cached)),
    }
    _atomic_json(_phase_path(args, _SINGLE_NAME), result)
    _atomic_json(
        _phase_path(args, _SINGLE_RUNTIME_NAME),
        {
            "schema_version": 1,
            "duration_seconds": duration_seconds,
            "peak_memory_bytes": peak_memory_bytes,
        },
    )
    return result


def run_three_model_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """Benchmark the fixed CPU baselines and CUDA LDSR model over all SPOT v3 samples."""
    capture_gpu_hardware()
    ldsr = LDSRS2X4.from_pretrained(Path(args.ldsr_model_dir), device="cuda:0")
    pairs = _load_nine_pairs(args)
    sen2srlite = SEN2SRLiteX4.from_pretrained(Path(args.sen2srlite_model_dir), device="cpu")
    environment = _production_environment(project_root=getattr(args, "project_root", None))
    environment["device"] = "cuda:0"
    result = run_benchmark(
        pairs=pairs,
        models=[BicubicX4(), sen2srlite, ldsr],
        cache_root=Path(args.prediction_cache_dir),
        result_path=_phase_path(args, _BENCHMARK_NAME),
        environment=environment,
        expected_model_count=3,
    )
    _write_ldsr_cache_index(args, pairs, ldsr.provenance())
    return result


def _write_ldsr_cache_index(
    args: argparse.Namespace, pairs: Sequence[SRPair], provenance: Mapping[str, object]
) -> Path:
    identities = [
        build_identity(provenance, pair.source, pair.sample_id, pair.lr) for pair in pairs
    ]
    identities.sort(key=lambda identity: (identity.source, identity.sample_id))
    payload = {
        "schema_version": 1,
        "model_provenance": dict(provenance),
        "identities": [
            {
                "source": identity.source,
                "sample_id": identity.sample_id,
                "lr": identity.as_dict()["lr"],
                "cache_key": identity.key,
            }
            for identity in identities
        ],
    }
    path = _phase_path(args, _CACHE_INDEX_NAME)
    _atomic_json(path, payload)
    return path


def _read_json_object(path: Path, description: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def _cache_index_error(reason: str) -> ValueError:
    return ValueError(f"invalid LDSR cache index: {reason}")


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_ldsr_cache_index(
    index: dict[str, object],
    benchmark: dict[str, object],
    single: dict[str, object],
) -> list[PredictionIdentity]:
    if set(index) != {"schema_version", "model_provenance", "identities"}:
        raise _cache_index_error("schema")
    if index["schema_version"] != 1 or type(index["schema_version"]) is not int:
        raise _cache_index_error("schema version")
    provenance = index["model_provenance"]
    if not isinstance(provenance, dict):
        raise _cache_index_error("model provenance")
    try:
        canonical_json(provenance)
    except ValueError as exc:
        raise _cache_index_error("model provenance") from exc
    if provenance.get("name") != "ldsr-s2-x4" or provenance.get("scale") != 4:
        raise _cache_index_error("model provenance")
    entries = index["identities"]
    if not isinstance(entries, list) or len(entries) != len(EXPECTED_SPOT_V3_IDENTITIES):
        raise _cache_index_error("must contain exactly nine identities")

    identities: list[PredictionIdentity] = []
    entry_pairs: list[tuple[str, str]] = []
    keys: list[str] = []
    entry_lrs: dict[tuple[str, str], dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "source",
            "sample_id",
            "lr",
            "cache_key",
        }:
            raise _cache_index_error("identity schema")
        source, sample_id, lr, key = (
            entry["source"],
            entry["sample_id"],
            entry["lr"],
            entry["cache_key"],
        )
        if type(source) is not str or type(sample_id) is not str or not _is_digest(key):
            raise _cache_index_error("identity data")
        if not isinstance(lr, dict) or set(lr) != {"shape", "dtype", "sha256"}:
            raise _cache_index_error("LR identity schema")
        shape, dtype, lr_sha256 = lr["shape"], lr["dtype"], lr["sha256"]
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
            or shape[0] != 4
            or dtype != "torch.float32"
            or not _is_digest(lr_sha256)
        ):
            raise _cache_index_error("LR identity data")
        try:
            identity = PredictionIdentity(
                provenance,
                source,
                sample_id,
                tuple(shape),
                dtype,
                lr_sha256,
            )
        except (TypeError, ValueError) as exc:
            raise _cache_index_error("identity data") from exc
        if identity.key != key:
            raise _cache_index_error("cache key is not bound to its identity")
        identities.append(identity)
        pair = (source, sample_id)
        entry_pairs.append(pair)
        keys.append(key)
        entry_lrs[pair] = lr

    if (
        len(set(entry_pairs)) != len(entry_pairs)
        or set(entry_pairs) != set(EXPECTED_SPOT_V3_IDENTITIES)
        or len(set(keys)) != len(keys)
    ):
        raise _cache_index_error("identities must be the distinct exact SPOT v3 set")
    if entry_pairs != sorted(entry_pairs):
        raise _cache_index_error("identities must use deterministic order")

    try:
        benchmark_run = benchmark["run"]
        benchmark_models = benchmark["models"]
        if not isinstance(benchmark_run, dict) or not isinstance(benchmark_models, dict):
            raise TypeError
        ldsr_result = benchmark_models["ldsr-s2-x4"]
        if not isinstance(ldsr_result, dict) or ldsr_result["provenance"] != provenance:
            raise TypeError
        run_samples = benchmark_run["samples"]
        model_samples = ldsr_result["samples"]
        if (
            benchmark_run["sample_count"] != 9
            or not isinstance(run_samples, list)
            or not isinstance(model_samples, list)
            or len(run_samples) != 9
            or len(model_samples) != 9
        ):
            raise TypeError
        benchmark_lrs = {
            (item["source"], item["sample_id"]): item["lr"]
            for item in run_samples
            if isinstance(item, dict)
        }
        benchmark_pairs = [
            (item["source"], item["sample_id"])
            for item in model_samples
            if isinstance(item, dict)
        ]
    except (KeyError, TypeError) as exc:
        raise _cache_index_error("benchmark binding") from exc
    if (
        len(benchmark_lrs) != 9
        or set(benchmark_lrs) != set(EXPECTED_SPOT_V3_IDENTITIES)
        or any(benchmark_lrs[pair] != entry_lrs[pair] for pair in entry_pairs)
        or len(benchmark_pairs) != 9
        or len(set(benchmark_pairs)) != 9
        or set(benchmark_pairs) != set(EXPECTED_SPOT_V3_IDENTITIES)
    ):
        raise _cache_index_error("benchmark binding")

    first = identities[0]
    if (
        single.get("source") != first.source
        or single.get("sample_id") != first.sample_id
        or single.get("lr_sha256") != first.lr_sha256
        or single.get("model_provenance") != provenance
        or single.get("cache_key") != first.key
    ):
        raise _cache_index_error("single-result binding")
    return identities


def _named_cache_artifacts(
    args: argparse.Namespace,
    index: dict[str, object],
    benchmark: dict[str, object],
    single: dict[str, object],
) -> list[Path]:
    cache_root = Path(args.prediction_cache_dir)
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise ValueError("prediction cache root must be an existing non-symlink directory")
    identities = _validate_ldsr_cache_index(index, benchmark, single)
    cache = PredictionCache(cache_root)
    named: list[Path] = []
    for identity in identities:
        if cache.get(identity) is None:
            raise FileNotFoundError(f"missing named prediction cache artifact: {identity.key}")
        for suffix in (".json", ".safetensors"):
            candidate = cache_root / f"{identity.key}{suffix}"
            if not candidate.is_file() or candidate.is_symlink():
                raise FileNotFoundError(
                    f"missing named prediction cache artifact: {candidate.name}"
                )
            named.append(candidate)
    return named


def run_manifest(args: argparse.Namespace) -> dict[str, object]:
    """Write the allowlisted manifest for current Phase 1B outputs only."""
    capture_gpu_hardware()
    root = Path(args.artifacts_dir)
    required = [
        _phase_path(args, _ENVIRONMENT_NAME),
        _phase_path(args, _SINGLE_NAME),
        _phase_path(args, _SINGLE_RUNTIME_NAME),
        _phase_path(args, _BENCHMARK_NAME),
        _phase_path(args, _CACHE_INDEX_NAME),
    ]
    for path in required:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing allowlisted Phase 1B output: {path.name}")
    single = _read_json_object(required[1], "single result")
    benchmark = _read_json_object(required[3], "benchmark result")
    index = _read_json_object(required[4], "LDSR cache index")
    named_cache_artifacts = _named_cache_artifacts(args, index, benchmark, single)
    staged_cache_paths = [
        stage_artifact_file(root, source, Path("phase1b") / "cache" / source.name)
        for source in named_cache_artifacts
    ]
    paths = required + staged_cache_paths
    relative_paths: list[Path] = []
    for path in paths:
        try:
            relative_paths.append(path.relative_to(root))
        except ValueError as exc:
            raise ValueError("allowlisted artifact escapes the artifact root") from exc
    manifest_path = write_artifact_manifest(root, relative_paths)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--dataset-cache-dir", type=Path, default=Path("data/cache/opensr-test"))
    parser.add_argument("--ldsr-model-dir", type=Path, default=Path("models/LDSR-S2"))
    parser.add_argument(
        "--sen2srlite-model-dir", type=Path, default=Path("models/SEN2SRLite_RGBN")
    )
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--prediction-cache-dir", type=Path, default=Path("artifacts/cache/predictions")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run staged LDSR-S2 CUDA reproducibility checks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in (
        ("preflight", run_preflight),
        ("single", run_single),
        ("benchmark", run_three_model_benchmark),
        ("manifest", run_manifest),
    ):
        child = subparsers.add_parser(name)
        _add_paths(child)
        child.set_defaults(handler=handler)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = args.handler(args)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
