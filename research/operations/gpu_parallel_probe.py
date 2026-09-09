"""Bounded synthetic-only GPU parity and throughput probe; no study data access."""

import argparse
import hashlib
import json
import multiprocessing
import os
import time
import traceback
from pathlib import Path

SEEDS = tuple(range(3407, 3412))
TIMEOUT_SECONDS = 1800


def synthetic_input(index):
    import numpy as np
    import torch

    # Integer arithmetic makes these distinct RGBN inputs reproducible without RNG.
    values = np.arange(4 * 128 * 128, dtype=np.uint32)
    values = (values * (37 + 16 * index) + 101 * index) % 4096
    return torch.from_numpy((values.astype(np.float32) / 4095).reshape(4, 128, 128))


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def worker(model_dir, device, indices, connection, start):
    try:
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"
        import torch

        from trustsr.models.ldsr_s2 import LDSRS2X4

        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        model = LDSRS2X4.from_pretrained(model_dir, device=device)
        inputs = {index: synthetic_input(index) for index in indices}
        model.for_seed(SEEDS[0]).predict(synthetic_input(0))
        torch.cuda.synchronize(device)
        connection.send({"status": "ready", "pid": os.getpid(), "provenance": model.provenance()})
        if not start.wait(TIMEOUT_SECONDS):
            raise TimeoutError("parent did not release timing barrier")
        began = time.perf_counter()
        rows = []
        for index, lr in inputs.items():
            predictions = []
            for seed in SEEDS:
                prediction = model.for_seed(seed).predict(lr)
                predictions.append(
                    {
                        "seed": seed,
                        "sha256": tensor_hash(prediction),
                        "shape": list(prediction.shape),
                        "dtype": str(prediction.dtype),
                    }
                )
                del prediction
            rows.append(
                {"input_index": index, "input_sha256": tensor_hash(lr), "predictions": predictions}
            )
        torch.cuda.synchronize(device)
        connection.send(
            {
                "status": "done",
                "rows": rows,
                "inference_and_hash_seconds": time.perf_counter() - began,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            }
        )
    except BaseException:
        connection.send({"status": "error", "traceback": traceback.format_exc()})
    finally:
        connection.close()


def receive(connection, deadline):
    if not connection.poll(max(0, deadline - time.monotonic())):
        raise TimeoutError("synthetic probe exceeded bounded mode deadline")
    message = connection.recv()
    if message["status"] == "error":
        raise RuntimeError(message["traceback"])
    return message


def run_mode(model_dir, device, workers):
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    processes, connections, ready = [], [], []
    began = time.perf_counter()
    deadline = time.monotonic() + TIMEOUT_SECONDS
    try:
        # Sequential preloads bound transient initialization pressure. Both parallel
        # models remain alive behind the same event before measurement starts.
        for indices in [(0, 1)] if workers == 1 else [(0,), (1,)]:
            parent, child = context.Pipe(duplex=False)
            process = context.Process(
                target=worker, args=(str(model_dir), device, indices, child, start)
            )
            processes.append(process)
            connections.append(parent)
            process.start()
            child.close()
            message = receive(parent, deadline)
            if message["status"] != "ready":
                raise RuntimeError("invalid worker readiness message")
            ready.append(message)
        setup_seconds = time.perf_counter() - began
        measured = time.perf_counter()
        start.set()
        results = [receive(connection, deadline) for connection in connections]
        wall_seconds = time.perf_counter() - measured
        if any(result["status"] != "done" for result in results):
            raise RuntimeError("invalid worker completion message")
        return {
            "workers": workers,
            "setup_and_warmup_seconds": setup_seconds,
            "steady_wall_seconds": wall_seconds,
            "ready": ready,
            "results": results,
        }
    finally:
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        for connection in connections:
            connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if not args.model_dir.is_dir():
        parser.error("--model-dir must be an existing local model directory")
    if args.output.exists():
        parser.error("--output must be a new JSON file")
    report = {
        "schema_version": 1,
        "evidence_kind": "synthetic_operational_probe",
        "scientific_claim": False,
        "seeds": SEEDS,
        "device": args.device,
        "input_shape": [4, 128, 128],
        "input_dtype": "float32",
        "timing_scope": (
            "post-warmup prediction, CPU transfer, hashing and result IPC; "
            "excludes model initialization"
        ),
        "accepted": False,
    }
    try:
        report["serial"] = run_mode(args.model_dir, args.device, 1)
        report["parallel"] = run_mode(args.model_dir, args.device, 2)

        def rows(mode):
            return sorted(
                (row for result in mode["results"] for row in result["rows"]),
                key=lambda row: row["input_index"],
            )

        report["exact_parity"] = rows(report["serial"]) == rows(report["parallel"])
        report["speedup"] = (
            report["serial"]["steady_wall_seconds"] / report["parallel"]["steady_wall_seconds"]
        )
        report["accepted"] = report["exact_parity"] and report["speedup"] > 1
        report["decision"] = (
            "synthetic parity and speedup observed"
            if report["accepted"]
            else "reject parallel: parity mismatch or no observed speedup"
        )
    except Exception:
        report["decision"] = "reject parallel: probe failed"
        report["error"] = traceback.format_exc()
    # Exclusive creation prevents overwriting prior evidence. No tensors persist.
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted": report["accepted"],
                "decision": report["decision"],
            }
        )
    )
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
