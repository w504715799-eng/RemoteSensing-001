"""Deterministic, CUDA-only adapter for the verified LDSR-S2 backend."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trustsr.models.ldsr_assets import (
    CHECKPOINT_NAME,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    CHECKPOINT_URL,
    CONFIG_SHA256,
)
from trustsr.models.ldsr_backend import build_verified_backend
from trustsr.models.protocols import JsonScalar

OPENSR_MODEL_VERSION = "1.1.1"


def _validate_configuration(
    *,
    seed: int,
    sampling_steps: int,
    sampling_eta: float,
    sampling_temperature: float,
    histogram_matching: bool,
) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(sampling_steps) is not int or sampling_steps <= 0:
        raise ValueError("sampling_steps must be a positive integer")
    if type(sampling_eta) is not float or not math.isfinite(sampling_eta) or sampling_eta < 0:
        raise ValueError("sampling_eta must be a finite non-negative float")
    if (
        type(sampling_temperature) is not float
        or not math.isfinite(sampling_temperature)
        or sampling_temperature < 0
    ):
        raise ValueError("sampling_temperature must be a finite non-negative float")
    if type(histogram_matching) is not bool:
        raise ValueError("histogram_matching must be a boolean")


@contextmanager
def _isolated_randomness(device: str | torch.device, seed: int) -> Iterator[None]:
    """Seed a backend call without changing caller-visible random state."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    torch_device = torch.device(device)
    cuda_devices = [torch_device.index or 0] if torch_device.type == "cuda" else []
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch_device.type == "cuda" else None
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch_device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic


class LDSRS2X4:
    name = "ldsr-s2-x4"
    scale = 4

    def __init__(
        self,
        backend: Callable[..., torch.Tensor] | Any,
        *,
        device: str = "cuda:0",
        seed: int = 3407,
        sampling_steps: int = 100,
        sampling_eta: float = 0.95,
        sampling_temperature: float = 1.0,
        histogram_matching: bool = True,
    ) -> None:
        _validate_configuration(
            seed=seed,
            sampling_steps=sampling_steps,
            sampling_eta=sampling_eta,
            sampling_temperature=sampling_temperature,
            histogram_matching=histogram_matching,
        )
        self._backend = backend
        self.device = device
        self._torch_device = torch.device(device)
        self.seed = seed
        self.sampling_steps = sampling_steps
        self.sampling_eta = sampling_eta
        self.sampling_temperature = sampling_temperature
        self.histogram_matching = histogram_matching

    @classmethod
    def from_pretrained(cls, model_dir: Path | str, *, device: str = "cuda:0") -> LDSRS2X4:
        torch_device = torch.device(device)
        device_index = torch_device.index or 0
        if (
            torch_device.type != "cuda"
            or not torch.cuda.is_available()
            or device_index >= torch.cuda.device_count()
        ):
            raise ValueError(f"requested CUDA device is unavailable: {device}")
        return cls(build_verified_backend(model_dir, device=device), device=device)

    def provenance(self) -> dict[str, JsonScalar]:
        return {
            "name": self.name,
            "scale": self.scale,
            "implementation_schema_version": 1,
            "opensr_model_version": OPENSR_MODEL_VERSION,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "checkpoint_name": CHECKPOINT_NAME,
            "checkpoint_url": CHECKPOINT_URL,
            "checkpoint_size": CHECKPOINT_SIZE,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "config_sha256": CONFIG_SHA256,
            "device": "cuda" if self._torch_device.type == "cuda" else str(self._torch_device),
            "seed": self.seed,
            "sampling_steps": self.sampling_steps,
            "sampling_eta": self.sampling_eta,
            "sampling_temperature": self.sampling_temperature,
            "histogram_matching": self.histogram_matching,
            "output_policy": "clip_to_[0,1]",
        }

    def predict(self, lr: torch.Tensor) -> torch.Tensor:
        if not isinstance(lr, torch.Tensor) or lr.dtype != torch.float32:
            raise ValueError("expected float32 tensor")
        if tuple(lr.shape) != (4, 128, 128):
            raise ValueError("expected RGBN tensor with shape (4, 128, 128)")
        if not torch.isfinite(lr).all():
            raise ValueError("input must contain only finite values")
        if (lr < 0).any() or (lr > 1).any():
            raise ValueError("input values must be in [0, 1]")

        with torch.inference_mode(), _isolated_randomness(self._torch_device, self.seed):
            output = self._backend(
                lr.unsqueeze(0).to(self._torch_device),
                sampling_steps=self.sampling_steps,
                sampling_eta=self.sampling_eta,
                sampling_temperature=self.sampling_temperature,
                histogram_matching=self.histogram_matching,
                save_iterations=False,
                verbose=False,
            )
            if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, 4, 512, 512):
                raise ValueError("backend must return tensor with shape (1, 4, 512, 512)")
            if not torch.isfinite(output).all():
                raise ValueError("backend output must contain only finite values")
            return (
                output.squeeze(0)
                .to(dtype=torch.float32, device="cpu")
                .clamp_(0, 1)
                .detach()
                .contiguous()
            )


__all__ = ["LDSRS2X4"]
