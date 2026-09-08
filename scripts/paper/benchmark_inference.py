"""Bounded first-paper GPU/CPU timing; only three authenticated development ROIs."""

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path
from statistics import fmean

import torch

from scripts.paper.evaluate_neighborhood import RESULT_SHA, checked_bytes
from trustsr.artifacts.predictions import tensor_sha256
from trustsr.data.crosssensor_pairs import (
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    load_crosssensor_pair,
)
from trustsr.jsonio import atomic_write_bytes, canonical_json
from trustsr.models.bicubic import BicubicX4
from trustsr.models.ldsr_assets import CHECKPOINT_NAME
from trustsr.models.ldsr_s2 import LDSRS2X4
from trustsr.models.sen2srlite import MODEL_ASSET_SHA256, SEN2SRLiteX4


def select_measurement_records(records, published):
    """Bind complete development membership before choosing the first three IDs."""
    development = [r for r in records if r["split"] == "development"]
    expected = [r["sample_id"] for r in published["samples"]]
    actual = [r["sample_id"] for r in development]
    if (len(expected) != 120 or len(set(expected)) != 120 or len(actual) != 120
            or len(set(actual)) != 120 or set(actual) != set(expected)):
        raise ValueError("exact published development membership required")
    return sorted(development, key=lambda r: r["sample_id"])[:3]


def validate_output(output, storage, ldsr_directory, sen2sr_directory):
    """Do not overwrite or create outputs inside data, models, or this repository."""
    resolved = output.resolve()
    protected = [storage, ldsr_directory, sen2sr_directory, Path(__file__).resolve().parents[2]]
    if output.exists() or output.is_symlink():
        raise ValueError("new output directory required")
    if any(resolved == root.resolve() or root.resolve() in resolved.parents for root in protected):
        raise ValueError("output must be outside data, model and repository trees")


def measure_prediction(model, lr, *, cuda_device=None):
    """Time one complete prediction, including transfer to/from the adapter device."""
    if (not isinstance(lr, torch.Tensor) or lr.dtype != torch.float32
            or tuple(lr.shape) != (4, 128, 128) or not torch.isfinite(lr).all()
            or (lr < 0).any() or (lr > 1).any()):
        raise ValueError("invalid benchmark LR")
    input_sha = tensor_sha256(lr)
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
    started = time.perf_counter()
    prediction = model.predict(lr)
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
    seconds = time.perf_counter() - started
    if (not isinstance(prediction, torch.Tensor) or prediction.dtype != torch.float32
            or tuple(prediction.shape) != (4, 512, 512) or not torch.isfinite(prediction).all()
            or (prediction < 0).any() or (prediction > 1).any()):
        raise ValueError("invalid benchmark prediction")
    return {
        "seconds": seconds, "input_sha256": input_sha,
        "output_sha256": tensor_sha256(prediction),
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(cuda_device)) if cuda_device is not None else None
        ),
        "cuda_peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(cuda_device)) if cuda_device is not None else None
        ),
    }


def run(storage, ldsr_directory, sen2sr_directory, output):
    """Hardware-gated run; no prediction cache reads/writes and no Spain access."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real LDSR timing; no inputs were loaded")
    validate_output(output, storage, ldsr_directory, sen2sr_directory)
    if storage.resolve(strict=True) != storage.absolute() or not storage.is_dir():
        raise ValueError("storage must be a regular non-symlink directory")
    # Existing adapter factories verify contents. The preflight prevents download fallback.
    for root, names in ((ldsr_directory, [CHECKPOINT_NAME]),
                        (sen2sr_directory, MODEL_ASSET_SHA256)):
        for name in names:
            path = root / name
            if (not path.is_file() or path.is_symlink()
                    or path.resolve() != path.absolute()):
                raise ValueError("all model assets must already exist as regular files")
    for package, expected in (("opensr-model", "1.1.1"), ("sen2sr", "0.8.5"),
                              ("mlstac", "0.4.9")):
        if version(package) != expected:
            raise ValueError("model dependency version differs from frozen A")
    if str(torch.__version__) != "2.12.1+cu130":
        raise ValueError("PyTorch differs from frozen A; do not silently change execution identity")
    torch.set_num_threads(1)
    started = time.perf_counter()
    repository = Path(__file__).resolve().parents[2]
    published = json.loads(checked_bytes(
        repository / "artifacts/phase2b3a/sen2naipv2-development-score-audit-v1.json", RESULT_SHA,
    ))
    manifest = storage / "trustsr/phase2b1b/selections" / POST_MANIFEST_SHA256 / "samples.jsonl"
    records = [
        json.loads(line) for line in checked_bytes(manifest, POST_MANIFEST_SHA256).splitlines()
    ]
    selected = select_measurement_records(records, published)
    reference = {r["sample_id"]: r for r in published["samples"]}
    pairs = []
    for record in selected:
        for kind in ("lr", "hr"):
            asset = record[f"{kind}_asset"]
            relative = f"subset-v1/development/{record['sample_id']}/{kind}.tif"
            if asset["relative_path"] != relative:
                raise ValueError("asset outside exact development path")
            checked_bytes(storage / "trustsr/phase2b1b" / relative,
                          asset["sha256"], asset["size_bytes"])
        pair = load_crosssensor_pair(
            storage, record, manifest_sha256=POST_MANIFEST_SHA256,
            normalization_policy=PHASE2B3A_NORMALIZATION_POLICY,
        ).pair
        if any(
            tensor_sha256(getattr(pair, kind))
            != reference[pair.sample_id][f"{kind}_tensor_sha256"]
            for kind in ("lr", "hr")
        ):
            raise ValueError("normalized development pair differs from published A")
        pairs.append(pair)
    input_seconds = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "trustsr.first-paper-inference-timing.v1",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "development_result_sha256": RESULT_SHA,
        "manifest_sha256": POST_MANIFEST_SHA256,
        "sample_ids": [pair.sample_id for pair in pairs],
        "input_seconds": input_seconds, "cpu_threads": 1,
        "torch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "models": {}, "status": "running",
    }
    atomic_write_bytes(output / "runtime.json", canonical_json(result))
    factories = [
        ("bicubic", BicubicX4, None, (None,)),
        ("sen2srlite", lambda: SEN2SRLiteX4.from_pretrained(sen2sr_directory, device="cpu"),
         None, (None,)),
        ("ldsr", lambda: LDSRS2X4.from_pretrained(ldsr_directory, device="cuda:0"),
         "cuda:0", tuple(range(3407, 3412))),
    ]
    for name, factory, device, seeds in factories:
        load_start = time.perf_counter()
        model = factory()
        if device is not None:
            torch.cuda.synchronize(device)
        group = {"load_seconds": time.perf_counter() - load_start,
                 "provenance": model.provenance(), "cold_start": None, "measurements": []}
        result["models"][name] = group
        if name != "bicubic":
            group["cold_start"] = measure_prediction(model, pairs[0].lr, cuda_device=device)
            atomic_write_bytes(output / "runtime.json", canonical_json(result))
        for pair in pairs:
            for seed in seeds:
                active_model = model.for_seed(seed) if seed is not None else model
                row = measure_prediction(active_model, pair.lr, cuda_device=device)
                group["measurements"].append({"sample_id": pair.sample_id, "seed": seed, **row})
                atomic_write_bytes(output / "runtime.json", canonical_json(result))
                print(f"MEASURED {name} {pair.sample_id} seed={seed}", flush=True)
        group["mean_seconds"] = fmean(row["seconds"] for row in group["measurements"])
    result["status"] = "complete"
    result["scope"] = "inference-only; no external experiment, scoring or prediction storage"
    atomic_write_bytes(output / "runtime.json", canonical_json(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--ldsr-model-dir", type=Path, required=True)
    parser.add_argument("--sen2sr-model-dir", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    run(args.storage_root, args.ldsr_model_dir, args.sen2sr_model_dir, args.output_directory)


if __name__ == "__main__":
    main()
