"""Development-only cache reuse for the preregistered neighborhood adaptation."""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from statistics import fmean

import torch
from safetensors.torch import load

from trustsr.artifacts.predictions import tensor_sha256
from trustsr.data.crosssensor_pairs import (
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    load_crosssensor_pair,
)
from trustsr.evaluation.score_diagnostics import evaluate_roi_score
from trustsr.jsonio import canonical_json
from trustsr.risk.local import ensemble_variance_score, local_l1_risk
from trustsr.risk.neighborhood import neighborhood_variance_score

RESULT_SHA = "5bb0e5138d6ed1df6c65744556be02ccd48b77d3288df39630d16fbd9cd2dce9"
AUDIT_SHA = "d61c36e2180a2dc3468d4d9aba083ac0925d163ac2bb910e0227138e9fa249f1"


def checked_bytes(path, digest, size=None):
    if path.is_symlink() or path.resolve(strict=True) != path.absolute() or not path.is_file():
        raise ValueError("input must be a regular non-symlink path")
    raw = path.read_bytes()
    if (size is not None and len(raw) != size) or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("input size or SHA-256 mismatch")
    return raw


def select_inputs(audit, result, records):
    expected = {s["sample_id"] for s in result["samples"]}
    development = [r for r in records if r["split"] == "development"]
    if len(expected) != 120 or len(development) != 120:
        raise ValueError("exact 120 development ROIs required")
    if {r["sample_id"] for r in development} != expected:
        raise ValueError("development membership mismatch")
    groups = {g["sample_id"]: g for g in audit["groups"]}
    if len(audit["groups"]) != 120 or set(groups) != expected:
        raise ValueError("published audit membership mismatch")
    selected = []
    for record in sorted(development, key=lambda r: r["sample_id"]):
        entries = sorted(
            (
                e
                for e in groups[record["sample_id"]]["prediction_entries"]
                if e["model_name"] == "ldsr-s2-x4"
            ),
            key=lambda e: e["seed"],
        )
        if [e["seed"] for e in entries] != list(range(3407, 3412)):
            raise ValueError("exact five seeds required")
        if any(
            e["sample_id"] != record["sample_id"]
            or e["identity"]["sample_id"] != record["sample_id"]
            for e in entries
        ):
            raise ValueError("prediction membership mismatch")
        selected.append((record, entries))
    return selected


def choose_window(means):
    if set(means) != {3, 9} or not all(math.isfinite(v) for v in means.values()):
        raise ValueError("two finite candidate means required")
    return 3 if means[3] <= means[9] + 1e-12 else 9


def run(storage_root, output_directory):
    if not storage_root.is_dir() or storage_root.resolve() != storage_root.absolute():
        raise ValueError("storage root must be an existing non-symlink path")
    if output_directory.exists() or output_directory.is_symlink():
        raise ValueError("new output directory required")
    if (storage_root / "trustsr") in output_directory.resolve().parents:
        raise ValueError("output must be outside historical experiment trees")
    torch.set_num_threads(1)
    started = time.perf_counter()
    repository = Path(__file__).resolve().parents[2]
    result = json.loads(
        checked_bytes(
            repository / "artifacts/phase2b3a/sen2naipv2-development-score-audit-v1.json",
            RESULT_SHA,
        )
    )
    audit = json.loads(
        checked_bytes(
            repository / "artifacts/phase2b3a/sen2naipv2-development-score-cache-audit-v1.json",
            AUDIT_SHA,
        )
    )
    manifest = (
        storage_root / "trustsr/phase2b1b/selections" / POST_MANIFEST_SHA256 / "samples.jsonl"
    )
    records = [
        json.loads(line) for line in checked_bytes(manifest, POST_MANIFEST_SHA256).splitlines()
    ]
    selected = select_inputs(audit, result, records)
    cache = storage_root / "trustsr/phase2b3a/predictions" / POST_MANIFEST_SHA256
    # Complete file identity check before any diagnostic or candidate result is observed.
    for record, entries in selected:
        for kind in ("lr", "hr"):
            asset = record[f"{kind}_asset"]
            relative = f"subset-v1/development/{record['sample_id']}/{kind}.tif"
            if asset["relative_path"] != relative:
                raise ValueError("asset outside exact development path")
            checked_bytes(
                storage_root / "trustsr/phase2b1b" / relative, asset["sha256"], asset["size_bytes"]
            )
        for entry in entries:
            if hashlib.sha256(canonical_json(entry["identity"])).hexdigest() != entry["cache_key"]:
                raise ValueError("invalid cache identity")
            if {f["filename"] for f in entry["files"]} != {
                entry["cache_key"] + s for s in (".json", ".safetensors")
            }:
                raise ValueError("unexpected cache filename")
            for file in entry["files"]:
                checked_bytes(cache / file["filename"], file["sha256"], file["size_bytes"])
    print("INPUTS_VERIFIED development=120 predictions=600 assets=240", flush=True)
    published = {s["sample_id"]: s for s in result["samples"]}
    rows, timings = [], []
    for index, (record, entries) in enumerate(selected, 1):
        pair = load_crosssensor_pair(
            storage_root,
            record,
            manifest_sha256=POST_MANIFEST_SHA256,
            normalization_policy=PHASE2B3A_NORMALIZATION_POLICY,
        ).pair
        old = published[pair.sample_id]
        if (
            tensor_sha256(pair.lr) != old["lr_tensor_sha256"]
            or tensor_sha256(pair.hr) != old["hr_tensor_sha256"]
        ):
            raise ValueError("normalized development tensor mismatch")
        predictions = []
        for entry in entries:
            file = next(f for f in entry["files"] if f["filename"].endswith(".safetensors"))
            tensors = load(
                checked_bytes(cache / file["filename"], file["sha256"], file["size_bytes"])
            )
            prediction = tensors["prediction"]
            if tensor_sha256(prediction) != entry["prediction_sha256"]:
                raise ValueError("prediction tensor mismatch")
            predictions.append(prediction)
        samples = torch.stack(predictions)
        risk_maps = {w: local_l1_risk(predictions[0], pair.hr, window=w) for w in (1, 9)}
        baseline = next(s for s in old["scores"] if s["name"] == "ldsr_variance_k5")
        for window, key in [(1, "sensitivity_window_1"), (9, "primary_window_9")]:
            if tensor_sha256(risk_maps[window]) != old["risks"][key]:
                raise ValueError("risk map mismatch")
            reproduced = evaluate_roi_score(
                ensemble_variance_score(samples), risk_maps[window]
            )
            if abs(reproduced.aurc - baseline[key]["aurc"]) > 1e-12:
                raise ValueError("baseline AURC compatibility failure")
        for candidate in (3, 9):
            start = time.perf_counter()
            score = neighborhood_variance_score(samples, window=candidate)
            timings.append(dict(candidate=candidate, seconds=time.perf_counter() - start))
            for window in (1, 9):
                stats = evaluate_roi_score(score, risk_maps[window])
                rows.append(
                    dict(
                        sample_id=pair.sample_id,
                        candidate=candidate,
                        risk_window=window,
                        aurc=stats.aurc,
                        rho=stats.rho,
                        random_aurc=stats.random_aurc,
                        selective_mean_risks=list(stats.selective_mean_risks),
                    )
                )
        if index % 10 == 0:
            print(f"COMPLETED_ROIS {index}/120", flush=True)
    summaries = [
        dict(
            candidate=c,
            risk_window=w,
            mean_aurc=fmean(
                r["aurc"] for r in rows if r["candidate"] == c and r["risk_window"] == w
            ),
            mean_rho=fmean(r["rho"] for r in rows if r["candidate"] == c and r["risk_window"] == w),
        )
        for c in (3, 9)
        for w in (1, 9)
    ]
    science = dict(
        schema="trustsr.first-paper-neighborhood-development.v1",
        dataset_role="development",
        sample_count=120,
        prediction_count=600,
        inference_count=0,
        upstream_result_sha256=RESULT_SHA,
        upstream_audit_sha256=AUDIT_SHA,
        selected_window=choose_window(
            {r["candidate"]: r["mean_aurc"] for r in summaries if r["risk_window"] == 9}
        ),
        gaussian_sigma=1,
        gaussian_radius=3,
        baseline_reproduction_tolerance=1e-12,
        summaries=summaries,
        samples=rows,
    )
    runtime = dict(
        elapsed_seconds=time.perf_counter() - started,
        score_timings=timings,
        torch_version=torch.__version__,
        cpu_threads=1,
        gpu_inference_count=0,
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    for filename, payload in [("science.json", science), ("runtime.json", runtime)]:
        with (output_directory / filename).open("xb") as handle:
            handle.write(canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "selected_window": science["selected_window"],
                "summaries": summaries,
                "elapsed_seconds": runtime["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    run(args.storage_root, args.output_directory)
