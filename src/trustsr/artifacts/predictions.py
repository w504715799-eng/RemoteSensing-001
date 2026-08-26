"""Identity-bound prediction cache using safetensors and JSON metadata."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from safetensors.torch import load_file, save_file

SCHEMA_VERSION = 1


class CacheIntegrityError(RuntimeError):
    """A cache entry exists but is corrupt or does not match its identity."""


def canonical_json(value: Any) -> bytes:
    """Return canonical UTF-8 JSON (the representation used for identities)."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON") from exc


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash the logical tensor's contiguous CPU byte representation."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    data = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class PredictionIdentity:
    model_provenance: Mapping[str, Any]
    source: str
    sample_id: str
    lr_shape: tuple[int, ...]
    lr_dtype: str
    lr_sha256: str

    def __post_init__(self) -> None:
        # Keep the public constructor subject to the same scalar/copy invariant
        # as build_identity; callers cannot smuggle in mutable nested mappings.
        frozen_provenance = MappingProxyType(_validate_provenance(self.model_provenance))
        object.__setattr__(self, "model_provenance", frozen_provenance)
        if not isinstance(self.source, str) or not isinstance(self.sample_id, str):
            raise TypeError("source and sample_id must be strings")
        object.__setattr__(self, "lr_shape", tuple(int(x) for x in self.lr_shape))

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_provenance": dict(self.model_provenance),
            "source": self.source,
            "sample_id": self.sample_id,
            "lr": {
                "shape": list(self.lr_shape),
                "dtype": self.lr_dtype,
                "sha256": self.lr_sha256,
            },
        }

    @property
    def key(self) -> str:
        return hashlib.sha256(canonical_json(self.as_dict())).hexdigest()

    @property
    def cache_key(self) -> str:
        return self.key


def _validate_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise TypeError("model provenance must be a mapping")
    result = dict(provenance)
    # Canonical serialization also rejects non-scalars/NaN, while this makes the
    # scalar-only contract explicit and prevents surprising custom JSON objects.
    for key, value in result.items():
        if not isinstance(key, str) or (
            not isinstance(value, (str, int, float, bool)) and value is not None
        ):
            raise TypeError("model provenance must contain JSON scalar values")
    canonical_json(result)
    return result


def build_identity(
    model_provenance: Mapping[str, Any], source: str, sample_id: str, lr: torch.Tensor
) -> PredictionIdentity:
    """Build the stable identity used as the cache key."""
    if not isinstance(source, str) or not isinstance(sample_id, str):
        raise TypeError("source and sample_id must be strings")
    if not isinstance(lr, torch.Tensor) or lr.ndim != 3:
        raise ValueError("lr must be a 3-D tensor")
    return PredictionIdentity(
        MappingProxyType(_validate_provenance(model_provenance)),
        source,
        sample_id,
        tuple(int(x) for x in lr.shape),
        str(lr.dtype),
        tensor_sha256(lr),
    )


# Descriptive aliases kept intentionally small for callers that prefer verbs.
make_identity = build_identity


def prediction_cache_key(identity: PredictionIdentity) -> str:
    return identity.key


class PredictionCache:
    """A cache whose entries are committed by their JSON sidecar."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: str) -> tuple[Path, Path]:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("invalid cache key")
        return self.root / f"{key}.safetensors", self.root / f"{key}.json"

    @staticmethod
    def _validate_prediction(prediction: torch.Tensor, lr_shape: tuple[int, ...]) -> torch.Tensor:
        if not isinstance(prediction, torch.Tensor) or prediction.dtype != torch.float32:
            raise ValueError("prediction must be torch.float32")
        if len(lr_shape) != 3 or lr_shape[0] != 4:
            raise ValueError("lr must have four channels")
        expected = (4, lr_shape[1] * 4, lr_shape[2] * 4)
        if tuple(prediction.shape) != expected:
            raise ValueError(f"prediction must have shape {expected}")
        if not torch.isfinite(prediction).all() or (prediction < 0).any() or (prediction > 1).any():
            raise ValueError("prediction must be finite and in [0, 1]")
        return prediction.detach().to(device="cpu").contiguous()

    def put(self, identity: PredictionIdentity, prediction: torch.Tensor) -> str:
        prediction = self._validate_prediction(prediction, identity.lr_shape)
        tensor_path, metadata_path = self._paths(identity.key)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "cache_key": identity.key,
            "identity": identity.as_dict(),
            "prediction": {
                "shape": list(prediction.shape),
                "dtype": str(prediction.dtype),
                "sha256": tensor_sha256(prediction),
            },
            "tensor_filename": tensor_path.name,
        }
        tensor_tmp = metadata_tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.root, prefix=f".{identity.key}.", suffix=".tmp", delete=False
            ) as f:
                tensor_tmp = Path(f.name)
            with tempfile.NamedTemporaryFile(
                dir=self.root,
                prefix=f".{identity.key}.",
                suffix=".tmp",
                delete=False,
                mode="wb",
            ) as f:
                metadata_tmp = Path(f.name)
            save_file({"prediction": prediction}, str(tensor_tmp))
            with tensor_tmp.open("rb") as stream:
                os.fsync(stream.fileno())
            with metadata_tmp.open("wb") as stream:
                stream.write(canonical_json(metadata))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tensor_tmp, tensor_path)
            tensor_tmp = None
            self._fsync_directory()
            os.replace(metadata_tmp, metadata_path)
            metadata_tmp = None
            self._fsync_directory()
        finally:
            for path in (tensor_tmp, metadata_tmp):
                if path is not None:
                    path.unlink(missing_ok=True)
        return identity.key

    def _fsync_directory(self) -> None:
        """Persist directory entries when the platform supports it."""
        try:
            fd = os.open(self.root, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    store = put
    save = put

    def get(self, identity: PredictionIdentity) -> torch.Tensor | None:
        tensor_path, metadata_path = self._paths(identity.key)
        if not metadata_path.exists():
            return None
        if not tensor_path.exists():
            raise CacheIntegrityError("cache metadata exists without tensor")
        try:
            metadata = json.loads(metadata_path.read_bytes().decode("utf-8"))
            expected = {
                "schema_version": SCHEMA_VERSION,
                "cache_key": identity.key,
                "identity": identity.as_dict(),
                "tensor_filename": tensor_path.name,
            }
            if any(metadata.get(k) != v for k, v in expected.items()):
                raise CacheIntegrityError("cache metadata mismatch")
            prediction_meta = metadata["prediction"]
            tensors = load_file(str(tensor_path), device="cpu")
            prediction = tensors["prediction"]
            actual_prediction_meta = {
                "shape": list(prediction.shape),
                "dtype": str(prediction.dtype),
                "sha256": tensor_sha256(prediction),
            }
            if prediction_meta != actual_prediction_meta:
                raise CacheIntegrityError("prediction metadata mismatch")
            return self._validate_prediction(prediction, identity.lr_shape)
        except CacheIntegrityError:
            raise
        except Exception as exc:
            raise CacheIntegrityError("invalid prediction cache entry") from exc

    load = get
