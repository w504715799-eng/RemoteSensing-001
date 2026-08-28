"""Verified adapter for the SEN2SRLite RGBN x4 model."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlstac
import torch

from trustsr.models.protocols import JsonScalar

MODEL_MANIFEST_URL = (
    "https://huggingface.co/tacofoundation/sen2sr/resolve/main/"
    "SEN2SRLite/NonReference_RGBN_x4/mlm.json"
)
MODEL_ID = "SEN2SRLite_NonReference_RGBN_x4"
MODEL_ASSET_SHA256: dict[str, str] = {
    "example_data.safetensor": "c895c7da8a8d48882b73a2a1955e4260714b97540eea290229a284d73f129985",
    "hard_constraint.safetensor": (
        "fbad981519066387c413ead1d6af7ef3e0d2947c34147ba90163fc79ae539239"
    ),
    "load.py": "4b6c836b1f73078c62c84d4374b2d8daee5345f6239f64e0b6be29432383bac6",
    "mlm.json": "59caa5c6af96a6fbebdbd771d93c91cc2d3a770302cd2f262b5409e77a40e3f7",
    "model.safetensor": "479aa796d5068d0b1206118ccbca27bd3223df0214db1a9b31a1e18349ed1c7e",
}


def verify_model_assets(
    root: Path | str, expected: dict[str, str] = MODEL_ASSET_SHA256
) -> None:
    """Verify each named model asset without traversing outside ``root``."""
    root = Path(root).resolve()
    for name, digest in expected.items():
        path = (root / name).resolve()
        if path.parent != root:
            raise ValueError(f"asset path escapes model root: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"missing model asset: {name}")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        actual = hasher.hexdigest()
        if actual != digest:
            raise ValueError(
                f"model asset hash mismatch for {name}: expected {digest}, got {actual}"
            )


def _model_root(downloaded: Any, cache_dir: Path) -> Path:
    """Extract a downloaded model directory while constraining it to cache_dir."""
    candidate = getattr(downloaded, "source", None) or getattr(downloaded, "file", None)
    path = Path(candidate) if candidate is not None else cache_dir
    if path.is_file():
        path = path.parent
    path = path.resolve()
    cache_dir = cache_dir.resolve()
    if path != cache_dir and cache_dir not in path.parents:
        raise ValueError("downloaded model root is outside cache directory")
    return path


def download_verified_model(cache_dir: Path | str) -> Any:
    """Download, verify, and only then dynamically load the model manifest."""
    cache_dir = Path(cache_dir)
    if all((cache_dir / name).is_file() for name in MODEL_ASSET_SHA256):
        verify_model_assets(cache_dir)
        return mlstac.load(cache_dir / "mlm.json")
    downloaded = mlstac.download(MODEL_MANIFEST_URL, cache_dir)
    root = _model_root(downloaded, cache_dir)
    verify_model_assets(root)
    return mlstac.load(root / "mlm.json")


class SEN2SRLiteX4:
    name = "sen2srlite-x4"
    scale = 4

    def __init__(self, backend: Callable[[torch.Tensor], torch.Tensor] | Any, device: str = "cpu"):
        self._backend = backend
        self.device = device

    @classmethod
    def from_pretrained(cls, cache_dir: Path | str, device: str = "cpu") -> SEN2SRLiteX4:
        if device != "cpu" and not (device.startswith("cuda") and torch.cuda.is_available()):
            raise ValueError(f"requested device is unavailable: {device}")
        loader = download_verified_model(cache_dir)
        backend = loader.compiled_model(device=device)
        return cls(backend, device=device)

    def provenance(self) -> dict[str, JsonScalar]:
        result: dict[str, JsonScalar] = {
            "name": self.name,
            "scale": self.scale,
            "model_id": MODEL_ID,
            "manifest_url": MODEL_MANIFEST_URL,
            "mlstac_version": version("mlstac"),
            "sen2sr_version": version("sen2sr"),
            "device": self.device,
            "output_policy": "clip_to_[0,1]",
            "torch_version": torch.__version__,
            "implementation_schema_version": 1,
        }
        result.update(
            {f"asset_sha256:{name}": digest for name, digest in MODEL_ASSET_SHA256.items()}
        )
        return result

    @torch.inference_mode()
    def predict(self, lr: torch.Tensor) -> torch.Tensor:
        if not isinstance(lr, torch.Tensor) or lr.dtype != torch.float32:
            raise ValueError("expected float32 tensor")
        if lr.ndim != 3 or tuple(lr.shape) != (4, 128, 128):
            raise ValueError("expected RGBN tensor with shape (4, 128, 128)")
        if not torch.isfinite(lr).all():
            raise ValueError("input must contain only finite values")
        if (lr < 0).any() or (lr > 1).any():
            raise ValueError("input values must be in [0, 1]")
        output = self._backend(lr.unsqueeze(0).to(self.device))
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
