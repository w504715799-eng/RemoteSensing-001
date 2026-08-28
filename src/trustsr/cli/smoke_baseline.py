"""Run the reproducible Phase-0 CPU bicubic smoke test."""

import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import opensr_test
import torch

from trustsr.data.opensr import load_opensr_pairs
from trustsr.evaluation.opensr_metrics import METRIC_KEYS, compute_opensr_metrics
from trustsr.models.bicubic import BicubicX4


def _git_commit() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else "unavailable"


def _finite_metrics(metrics: dict[str, float], sample_id: str) -> dict[str, float]:
    """Normalize and reject non-finite metrics before serializing a result."""
    normalized: dict[str, float] = {}
    for key in METRIC_KEYS:
        value = float(metrics[key])
        if not math.isfinite(value):
            raise ValueError(f"non-finite metric {key!r} for sample {sample_id!r}")
        normalized[key] = value
    return normalized


def run(
    dataset: str,
    version: str,
    limit: int,
    cache_dir: Path,
    output: Path,
) -> dict[str, object]:
    """Evaluate a bounded OpenSR SPOT sample set and write provenance JSON."""
    pairs = load_opensr_pairs(dataset, cache_dir, version, limit)
    if not pairs:
        raise ValueError("smoke run produced no samples")

    model = BicubicX4()
    samples: list[dict[str, object]] = []
    for pair in pairs:
        sr = model.predict(pair.lr)
        metrics = _finite_metrics(compute_opensr_metrics(pair, sr), pair.sample_id)
        samples.append({"sample_id": pair.sample_id, "metrics": metrics})

    means = {
        key: sum(float(item["metrics"][key]) for item in samples) / len(samples)
        for key in METRIC_KEYS
    }
    mean_metrics = _finite_metrics(means, "mean")
    manifest = "\n".join(f"{pair.source}:{pair.sample_id}" for pair in pairs)
    result: dict[str, object] = {
        "run": {
            "dataset": dataset,
            "dataset_version": version,
            "dataset_role": "development_smoke_only",
            "sample_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
            "model": model.name,
            "scale": 4,
            "bands": ["B04", "B03", "B02", "B08"],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "opensr_test": opensr_test.__version__,
            "device": "cpu",
            "git_commit": _git_commit(),
        },
        "samples": samples,
        "mean_metrics": mean_metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase-0 bicubic smoke test")
    parser.add_argument("--dataset", default="spot", choices=["spot"])
    parser.add_argument("--version", default="v3", choices=["v3"])
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/opensr-test"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/phase0/bicubic-spot-v3.json")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args.dataset, args.version, args.limit, args.cache_dir, args.output)
    print(json.dumps(result["mean_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
