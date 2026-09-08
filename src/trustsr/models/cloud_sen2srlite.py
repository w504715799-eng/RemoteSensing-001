"""Cloud CPU compatibility policy; use only in a single-worker process.

PyTorch thread settings are process-global: callers must not run concurrent
inference while this adapter is active. This does not alter the frozen adapter.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch

from trustsr.models.protocols import JsonScalar
from trustsr.models.sen2srlite import SEN2SRLiteX4, verify_model_assets


@contextmanager
def _historical_cpu_threads():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(96)
        yield
    finally:
        torch.set_num_threads(previous)


class CloudSEN2SRLiteX4(SEN2SRLiteX4):
    """Pin the CPU profile verified against three historical development ROIs."""

    @classmethod
    def from_pretrained(cls, cache_dir: Path | str, device: str = "cpu"):
        if device != "cpu":
            raise ValueError("cloud compatibility policy requires CPU")
        verify_model_assets(cache_dir)
        with _historical_cpu_threads():
            return super().from_pretrained(cache_dir, device=device)

    def predict(self, lr: torch.Tensor) -> torch.Tensor:
        if self.device != "cpu":
            raise ValueError("cloud compatibility policy requires CPU")
        with _historical_cpu_threads():
            return super().predict(lr)

    def provenance(self) -> dict[str, JsonScalar]:
        result = super().provenance()
        result.update({
            "cpu_execution_policy": "cloud-sen2srlite-96-v1",
            "cpu_intraop_threads": 96,
            "cpu_interop_threads": torch.get_num_interop_threads(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "mkldnn_deterministic": torch.backends.mkldnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        })
        return result
