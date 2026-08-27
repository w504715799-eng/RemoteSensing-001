"""Safely construct the upstream LDSR-S2 backend from verified assets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from trustsr.models.ldsr_assets import (
    AssetIntegrityError,
    VerifiedAsset,
    download_verified_checkpoint,
    verify_packaged_config,
)


class BackendLoadError(RuntimeError):
    """Raised when a verified checkpoint cannot be safely loaded into the backend."""


def load_verified_state_dict(
    checkpoint: VerifiedAsset,
    *,
    map_location: str | torch.device,
    torch_load: Callable[..., Any] = torch.load,
) -> dict[str, torch.Tensor]:
    """Load and validate a checkpoint using PyTorch's weights-only mode."""

    loaded = torch_load(
        checkpoint.path,
        map_location=torch.device(map_location),
        weights_only=True,
    )
    if not isinstance(loaded, Mapping):
        raise BackendLoadError("checkpoint must be a mapping")
    state_dict = loaded.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise BackendLoadError("checkpoint state_dict must be a mapping")

    filtered: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise BackendLoadError("checkpoint state_dict keys must be strings")
        if not isinstance(value, torch.Tensor):
            raise BackendLoadError("checkpoint state_dict values must be tensors")
        if "loss" not in key:
            filtered[key] = value
    return filtered


def build_verified_backend(
    model_dir: Path | str,
    *,
    device: str,
    package_module: ModuleType | None = None,
    omega_conf: Any | None = None,
) -> Any:
    """Verify assets, construct the upstream model, and load its state strictly."""

    if package_module is None:
        import opensr_model as package_module
    if omega_conf is None:
        from omegaconf import OmegaConf as omega_conf

    checkpoint = download_verified_checkpoint(model_dir)
    package_file = getattr(package_module, "__file__", None)
    if package_file is None:
        raise BackendLoadError("upstream package has no __file__")
    config_asset = verify_packaged_config(Path(package_file).parent)
    config = omega_conf.load(config_asset.path)
    backend = package_module.SRLatentDiffusion(config, device=device)

    filtered = load_verified_state_dict(checkpoint, map_location=device, torch_load=torch.load)
    try:
        backend.model.load_state_dict(filtered, strict=True)
    except Exception as exc:
        raise BackendLoadError("strict state load failed") from exc

    if getattr(backend, "training", None) is not False:
        raise BackendLoadError("backend must be in evaluation mode")
    if getattr(backend.model, "training", None) is not False:
        raise BackendLoadError("backend model must be in evaluation mode")
    return backend


__all__ = [
    "AssetIntegrityError",
    "BackendLoadError",
    "build_verified_backend",
    "load_verified_state_dict",
]
