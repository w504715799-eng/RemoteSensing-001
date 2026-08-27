"""Staged, CUDA-gated LDSR-S2 reproducibility workflow."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from trustsr.artifacts.gpu_run import _atomic_json, collect_gpu_environment, write_artifact_manifest
from trustsr.artifacts.predictions import PredictionCache, build_identity, tensor_sha256
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


def _active_compute_pids() -> set[int]:
    """Return active CUDA compute PIDs using a fixed, non-shell command."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
    except OSError as exc:
        raise RuntimeError("cannot inspect active CUDA compute processes") from exc
    if completed.returncode != 0:
        raise RuntimeError("cannot inspect active CUDA compute processes")
    pids: set[int] = set()
    for line in completed.stdout.splitlines():
        value = line.strip()
        if not value or value.lower() in {"no running processes found", "none"}:
            continue
        try:
            pid = int(value)
        except ValueError as exc:
            raise RuntimeError("nvidia-smi returned an invalid compute PID") from exc
        if pid <= 0:
            raise RuntimeError("nvidia-smi returned an invalid compute PID")
        pids.add(pid)
    return pids


def _require_cuda_idle() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the LDSR-S2 GPU workflow")
    pids = _active_compute_pids()
    if pids:
        raise RuntimeError("another active CUDA compute process prevents this staged run")


def _require_only_current_compute_process() -> None:
    pids = _active_compute_pids()
    unexpected = pids - {os.getpid()}
    if unexpected:
        raise RuntimeError("another active CUDA compute process appeared during model construction")


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


def _write_environment(args: argparse.Namespace, model: LDSRS2X4) -> dict[str, object]:
    environment: dict[str, object] = dict(collect_gpu_environment())
    environment["model_provenance"] = model.provenance()
    _atomic_json(_phase_path(args, _ENVIRONMENT_NAME), environment)
    return environment


def run_preflight(args: argparse.Namespace) -> dict[str, object]:
    """Verify CUDA and LDSR assets, then record the non-deterministic runtime."""
    _require_cuda_idle()
    model = LDSRS2X4.from_pretrained(Path(args.ldsr_model_dir), device="cuda:0")
    _require_only_current_compute_process()
    return _write_environment(args, model)


def run_single(args: argparse.Namespace) -> dict[str, object]:
    """Run the immutable spot-0000 repeatability gate and cache its prediction."""
    _require_cuda_idle()
    model = LDSRS2X4.from_pretrained(Path(args.ldsr_model_dir), device="cuda:0")
    _require_only_current_compute_process()
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
        "metrics": _finite_metrics(compute_opensr_metrics(pair, prediction)),
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
    _require_cuda_idle()
    ldsr = LDSRS2X4.from_pretrained(Path(args.ldsr_model_dir), device="cuda:0")
    _require_only_current_compute_process()
    pairs = _load_nine_pairs(args)
    sen2srlite = SEN2SRLiteX4.from_pretrained(Path(args.sen2srlite_model_dir), device="cpu")
    environment = _production_environment()
    environment["device"] = "cuda:0"
    return run_benchmark(
        pairs=pairs,
        models=[BicubicX4(), sen2srlite, ldsr],
        cache_root=Path(args.prediction_cache_dir),
        result_path=_phase_path(args, _BENCHMARK_NAME),
        environment=environment,
        expected_model_count=3,
    )


def _cache_keys_from_json(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for name, item in value.items():
            if name == "cache_key" and isinstance(item, str):
                keys.add(item)
            elif name == "cache_keys" and isinstance(item, list):
                keys.update(entry for entry in item if isinstance(entry, str))
            keys.update(_cache_keys_from_json(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_cache_keys_from_json(item))
    return keys


def _named_cache_artifacts(args: argparse.Namespace, result_paths: Sequence[Path]) -> list[Path]:
    cache_root = Path(args.prediction_cache_dir)
    named: list[Path] = []
    for result_path in result_paths:
        try:
            payload: Any = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read deterministic result: {result_path.name}") from exc
        for key in _cache_keys_from_json(payload):
            if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
                raise ValueError("result named an invalid prediction cache key")
            for suffix in (".json", ".safetensors"):
                candidate = cache_root / f"{key}{suffix}"
                if not candidate.is_file() or candidate.is_symlink():
                    raise FileNotFoundError(
                        f"missing named prediction cache artifact: {candidate.name}"
                    )
                named.append(candidate)
    return named


def run_manifest(args: argparse.Namespace) -> dict[str, object]:
    """Write the allowlisted manifest for current Phase 1B outputs only."""
    _require_cuda_idle()
    root = Path(args.artifacts_dir)
    required = [
        _phase_path(args, _ENVIRONMENT_NAME),
        _phase_path(args, _SINGLE_NAME),
        _phase_path(args, _SINGLE_RUNTIME_NAME),
        _phase_path(args, _BENCHMARK_NAME),
    ]
    for path in required:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"missing allowlisted Phase 1B output: {path.name}")
    paths = required + _named_cache_artifacts(args, [required[1], required[3]])
    relative_paths: list[Path] = []
    for path in paths:
        try:
            relative_paths.append(path.relative_to(root))
        except ValueError as exc:
            raise ValueError("allowlisted artifact escapes the artifact root") from exc
    manifest_path = write_artifact_manifest(root, relative_paths)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _add_paths(parser: argparse.ArgumentParser) -> None:
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
