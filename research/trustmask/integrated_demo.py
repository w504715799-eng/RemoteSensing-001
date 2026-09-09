"""Runnable synthetic-only CPU integration; no data files, network or model access."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.trustmask.pipeline import ROLES, Features, Observation, run_study


def synthetic_partitions():
    random = np.random.default_rng(3407)
    partitions = {}
    for role, count in zip(ROLES, (8, 32, 8, 32, 16), strict=True):
        records = []
        for i in range(count):
            signal = random.uniform(0, 1, (8, 8))
            residual = 0.01 + 0.4 * signal
            variance = 0.01 + 0.2 * random.uniform(0, 1, (8, 8))
            feature = Features(residual, variance, 0.5 * variance, 0.1 * signal)
            # Deliberately artificial reference-risk map; not a measured R9 outcome.
            risk = np.where(signal < 0.6, 0.005, 0.2).astype(np.float64)
            records.append(Observation(f"{role}-{i}", f"{role}-group-{i}", feature, risk))
        partitions[role] = records
    return partitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional new synthetic report file")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise ValueError("output already exists")
    result = run_study(synthetic_partitions())
    result["evidence_kind"] = "synthetic_not_research_evidence"
    result["generator_seed"] = 3407
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "evidence_kind": result["evidence_kind"],
                "method_count": result["evaluation"]["method_count"],
                "evaluation": result["evaluation"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
