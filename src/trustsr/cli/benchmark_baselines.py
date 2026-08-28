"""Run the deterministic Phase-1 CPU baseline benchmark on SPOT v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import opensr_test
import torch

from trustsr.artifacts.gpu_run import resolve_project_root
from trustsr.artifacts.predictions import PredictionCache, build_identity, tensor_sha256
from trustsr.contracts import SRPair
from trustsr.data.opensr import load_opensr_pairs
from trustsr.evaluation.opensr_metrics import METRIC_KEYS, compute_opensr_metrics
from trustsr.models.bicubic import BicubicX4
from trustsr.models.protocols import SRModel
from trustsr.models.sen2srlite import SEN2SRLiteX4

EXPECTED_SPOT_V3_IDENTITIES = tuple(
    ("opensr-test/spot/v3", f"spot-{index:04d}") for index in range(9)
)
_RUN_ENVIRONMENT_KEYS = ("git_commit", "python", "torch", "opensr_test", "device")


def _git_commit(project_root: Path | str | None = None) -> str:
    reviewed_root = resolve_project_root(project_root)
    process = subprocess.run(
        ["git", "-C", str(reviewed_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else "unavailable"


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": tensor_sha256(tensor),
    }


def _sample_manifest(pairs: Sequence[SRPair]) -> list[dict[str, Any]]:
    return [
        {
            "source": pair.source,
            "sample_id": pair.sample_id,
            "lr": _tensor_metadata(pair.lr),
            "hr": _tensor_metadata(pair.hr),
        }
        for pair in pairs
    ]


def _manifest_hash(manifest: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_nine_unique_pairs(pairs: Sequence[SRPair]) -> None:
    if len(pairs) != 9:
        raise ValueError("benchmark requires exactly nine SPOT samples")
    identities = [(pair.source, pair.sample_id) for pair in pairs]
    if len(set(identities)) != len(identities):
        raise ValueError("benchmark samples must have unique source/sample_id identities")
    for pair in pairs:
        pair.validate()


def _validate_prediction(pair: SRPair, prediction: torch.Tensor) -> torch.Tensor:
    expected_shape = (4, pair.lr.shape[1] * 4, pair.lr.shape[2] * 4)
    if not isinstance(prediction, torch.Tensor):
        raise ValueError("prediction must be a torch.Tensor")
    if prediction.dtype != torch.float32:
        raise ValueError("prediction must be torch.float32")
    if (
        prediction.device.type != "cpu"
        or not prediction.is_contiguous()
        or prediction.requires_grad
    ):
        raise ValueError("prediction must be detached, contiguous, and on CPU")
    if tuple(prediction.shape) != expected_shape:
        raise ValueError(f"prediction must have shape {expected_shape}")
    if not torch.isfinite(prediction).all() or (prediction < 0).any() or (prediction > 1).any():
        raise ValueError("prediction must be finite and in [0, 1]")
    return prediction


def _finite_metrics(metrics: Mapping[str, float], sample_id: str) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key in METRIC_KEYS:
        value = float(metrics[key])
        if not math.isfinite(value):
            raise ValueError(f"non-finite metric {key!r} for sample {sample_id!r}")
        normalized[key] = value
    return normalized


def _atomic_write_json(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False, mode="wb"
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_benchmark(
    *,
    pairs: Sequence[SRPair],
    models: Sequence[SRModel],
    cache_root: Path | str,
    result_path: Path | str,
    environment: Mapping[str, str],
    expected_model_count: int = 2,
) -> dict[str, object]:
    """Evaluate models over one frozen nine-sample manifest and write JSON."""
    _require_nine_unique_pairs(pairs)
    if type(expected_model_count) is not int or expected_model_count <= 0:
        raise ValueError("expected_model_count must be an exact positive int")
    if len(models) != expected_model_count:
        if expected_model_count == 2:
            raise ValueError("benchmark requires exactly two models")
        raise ValueError(f"benchmark requires exactly {expected_model_count} models")
    if environment.get("dataset") != "spot" or environment.get("dataset_version") != "v3":
        raise ValueError("benchmark requires the SPOT v3 development dataset")

    manifest = _sample_manifest(pairs)
    manifest_sha256 = _manifest_hash(manifest)
    names = [model.name for model in models]
    if len(set(names)) != len(names):
        raise ValueError("benchmark model names must be unique")
    cache = PredictionCache(cache_root)
    models_result: dict[str, object] = {}
    for model in models:
        if model.scale != 4:
            raise ValueError(f"model {model.name!r} must have scale 4")
        provenance = model.provenance()
        if provenance.get("name") != model.name or provenance.get("scale") != model.scale:
            raise ValueError(f"model {model.name!r} provenance does not identify the model")
        samples: list[dict[str, object]] = []
        for pair in pairs:
            identity = build_identity(provenance, pair.source, pair.sample_id, pair.lr)
            prediction = cache.get(identity)
            if prediction is None:
                prediction = _validate_prediction(pair, model.predict(pair.lr))
                cache.put(identity, prediction)
            else:
                prediction = _validate_prediction(pair, prediction)
            metrics = _finite_metrics(compute_opensr_metrics(pair, prediction), pair.sample_id)
            samples.append(
                {"source": pair.source, "sample_id": pair.sample_id, "metrics": metrics}
            )
        mean_metrics = _finite_metrics(
            {
                key: sum(float(item["metrics"][key]) for item in samples) / len(samples)
                for key in METRIC_KEYS
            },
            "mean",
        )
        models_result[model.name] = {
            "provenance": provenance,
            "sample_manifest_sha256": manifest_sha256,
            "samples": samples,
            "mean_metrics": mean_metrics,
        }

    run: dict[str, object] = {
        "dataset": environment["dataset"],
        "dataset_version": environment["dataset_version"],
        "dataset_role": "development_reproducibility_check_only",
        "sample_count": len(pairs),
        "sample_manifest_sha256": manifest_sha256,
        "samples": manifest,
        "bands": ["B04", "B03", "B02", "B08"],
        "scale": 4,
    }
    for key in _RUN_ENVIRONMENT_KEYS:
        run[key] = environment[key]
    result: dict[str, object] = {"run": run, "models": models_result}
    _atomic_write_json(Path(result_path), result)
    return result


def _production_environment(*, project_root: Path | str | None = None) -> dict[str, str]:
    return {
        "dataset": "spot",
        "dataset_version": "v3",
        "git_commit": _git_commit(project_root),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "opensr_test": opensr_test.__version__,
        "device": "cpu",
    }


def _require_expected_spot_v3_identities(pairs: Sequence[SRPair]) -> None:
    actual = {(pair.source, pair.sample_id) for pair in pairs}
    expected = set(EXPECTED_SPOT_V3_IDENTITIES)
    if actual != expected or len(pairs) != len(EXPECTED_SPOT_V3_IDENTITIES):
        raise ValueError(
            "loaded SPOT v3 pairs do not match the complete expected nine-sample identity set"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase-1 SPOT v3 CPU baseline benchmark")
    parser.add_argument("--dataset-cache-dir", type=Path, default=Path("data/cache/opensr-test"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/SEN2SRLite_RGBN"))
    parser.add_argument(
        "--prediction-cache-dir", type=Path, default=Path("artifacts/cache/predictions")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/phase1/spot-v3-baselines.json")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pairs = load_opensr_pairs(
        "spot", args.dataset_cache_dir, "v3", limit=9, expected_count=9
    )
    _require_expected_spot_v3_identities(pairs)
    result = run_benchmark(
        pairs=pairs,
        models=[BicubicX4(), SEN2SRLiteX4.from_pretrained(args.model_dir, device="cpu")],
        cache_root=args.prediction_cache_dir,
        result_path=args.output,
        environment=_production_environment(),
    )
    means = {name: value["mean_metrics"] for name, value in result["models"].items()}
    print(json.dumps(means, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
