from pathlib import Path

import numpy as np
import opensr_test
import torch

from trustsr.contracts import SRPair

L2A_RGBN_INDICES = (3, 2, 1, 7)
REFLECTANCE_SCALE = 10_000.0


def _to_reflectance(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array).copy()).to(torch.float32).div_(
        REFLECTANCE_SCALE
    ).clamp_(0.0, 1.0)


def load_opensr_pairs(
    dataset_name: str,
    cache_dir: Path,
    version: str = "v3",
    limit: int = 2,
) -> list[SRPair]:
    if dataset_name != "spot":
        raise ValueError("Phase 0 allows the SPOT development dataset only")
    if limit < 1:
        raise ValueError("limit must be positive")

    cache_dir.mkdir(parents=True, exist_ok=True)
    raw = opensr_test.load(dataset_name, model_dir=str(cache_dir), version=version)
    count = min(limit, len(raw["L2A"]))
    pairs: list[SRPair] = []
    for index in range(count):
        pair = SRPair(
            sample_id=f"{dataset_name}-{index:04d}",
            source=f"opensr-test/{dataset_name}/{version}",
            lr=_to_reflectance(
                np.take(raw["L2A"][index], L2A_RGBN_INDICES, axis=0)
            ),
            hr=_to_reflectance(raw["HRharm"][index]),
            scale=4,
        )
        pair.validate()
        pairs.append(pair)
    return pairs
