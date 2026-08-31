"""Deterministic Phase 2B2-B development smoke evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from trustsr.artifacts.predictions import (
    PredictionCache,
    PredictionIdentity,
    tensor_sha256,
)
from trustsr.data.crosssensor_pairs import POST_MANIFEST_SHA256
from trustsr.models.protocols import JsonScalar

EXPERIMENT_SCHEMA = "trustsr.phase2b2b-development-smoke.v1"
CACHE_AUDIT_SCHEMA = "trustsr.phase2b2b-cache-audit.v1"
INPUT_AUDIT_SHA256 = "fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b"
MODEL_NAMES = ("bicubic-x4", "sen2srlite-x4", "ldsr-s2-x4")
_CONTEXT_KEYS = (
    "experiment_schema",
    "post_manifest_sha256",
    "input_audit_sha256",
)


def build_cache_provenance(
    model_provenance: Mapping[str, JsonScalar],
) -> dict[str, JsonScalar]:
    """Bind model provenance to the frozen crosssensor experiment context."""

    if not isinstance(model_provenance, Mapping):
        raise TypeError("model provenance must be a mapping")
    if any(key in model_provenance for key in _CONTEXT_KEYS):
        raise ValueError("model provenance contains a reserved experiment context key")
    result = dict(model_provenance)
    result.update(
        {
            "experiment_schema": EXPERIMENT_SCHEMA,
            "post_manifest_sha256": POST_MANIFEST_SHA256,
            "input_audit_sha256": INPUT_AUDIT_SHA256,
        }
    )
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_files(cache_root: Path, identity: PredictionIdentity) -> tuple[Path, Path]:
    if not isinstance(cache_root, Path) or cache_root.is_symlink() or not cache_root.is_dir():
        raise ValueError("prediction cache root must be an existing non-symlink directory")
    paths = tuple(cache_root / f"{identity.key}{suffix}" for suffix in (".json", ".safetensors"))
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("named prediction cache file must be a regular file")
    return paths


def cache_entry_evidence(
    cache_root: Path, identity: PredictionIdentity
) -> dict[str, object]:
    """Return host-free evidence for one fully validated prediction cache entry."""

    prediction = PredictionCache(cache_root).get(identity)
    if prediction is None:
        raise ValueError("named prediction cache entry is missing")
    paths = _cache_files(cache_root, identity)
    return {
        "cache_key": identity.key,
        "lr": identity.as_dict()["lr"],
        "prediction_sha256": tensor_sha256(prediction),
        "files": [
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in paths
        ],
    }


def snapshot_cache_files(
    cache_root: Path, identities: Sequence[PredictionIdentity]
) -> tuple[tuple[object, ...], ...]:
    """Snapshot bytes and metadata needed to prove replay does not mutate caches."""

    keys = [identity.key for identity in identities]
    if len(set(keys)) != len(keys):
        raise ValueError("prediction identities must be unique")
    result: list[tuple[object, ...]] = []
    for identity in identities:
        for path in _cache_files(cache_root, identity):
            stat = path.stat()
            result.append(
                (path.name, stat.st_size, stat.st_mtime_ns, _file_sha256(path))
            )
    return tuple(result)
