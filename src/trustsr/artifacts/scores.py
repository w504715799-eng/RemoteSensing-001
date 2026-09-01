"""Identity-bound cache for finite non-negative score maps."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch
from safetensors.torch import load_file, save_file

from trustsr.artifacts.predictions import CacheIntegrityError, tensor_sha256
from trustsr.jsonio import canonical_json

SCORE_CACHE_SCHEMA_VERSION = 1
type JsonScalar = str | int | float | bool | None


def _is_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_scalar_mapping(value: Mapping[str, JsonScalar]) -> dict[str, JsonScalar]:
    if not isinstance(value, Mapping):
        raise TypeError("operator_parameters must be a mapping")
    parameters = dict(value)
    for key, parameter in parameters.items():
        if type(key) is not str or not isinstance(parameter, (str, int, float, bool, type(None))):
            raise TypeError("operator_parameters must contain JSON scalar values")
    try:
        canonical_json(parameters)
    except ValueError as exc:
        raise ValueError("operator_parameters must be canonical JSON scalars") from exc
    return parameters


def _validate_score_identity_fields(identity: ScoreIdentity) -> None:
    if type(identity.score_name) is not str or type(identity.sample_id) is not str:
        raise TypeError("score_name and sample_id must be strings")
    if not identity.score_name or not identity.sample_id:
        raise ValueError("score_name and sample_id must not be empty")
    if type(identity.score_schema_version) is not int:
        raise TypeError("score_schema_version must be an integer")
    if identity.score_schema_version <= 0:
        raise ValueError("score_schema_version must be positive")
    if type(identity.input_sha256s) is not tuple:
        raise TypeError("input_sha256s must be an immutable tuple")
    if not identity.input_sha256s:
        raise ValueError("input_sha256s must not be empty")
    if not all(_is_digest(digest) for digest in identity.input_sha256s):
        raise ValueError("input_sha256s must contain lowercase SHA-256 digests")


@dataclass(frozen=True)
class ScoreIdentity:
    """All inputs that determine a score-map value and cache location."""

    score_name: str
    score_schema_version: int
    sample_id: str
    input_sha256s: tuple[str, ...]
    operator_parameters: Mapping[str, JsonScalar]

    def __post_init__(self) -> None:
        parameters = _validated_scalar_mapping(self.operator_parameters)
        object.__setattr__(self, "operator_parameters", MappingProxyType(parameters))
        _validate_score_identity_fields(self)

    def as_dict(self) -> dict[str, object]:
        return {
            "score_name": self.score_name,
            "score_schema_version": self.score_schema_version,
            "sample_id": self.sample_id,
            "input_sha256s": list(self.input_sha256s),
            "operator_parameters": dict(self.operator_parameters),
        }

    @property
    def key(self) -> str:
        return hashlib.sha256(canonical_json(self.as_dict())).hexdigest()


def _validate_score(score: torch.Tensor) -> torch.Tensor:
    if not isinstance(score, torch.Tensor) or score.dtype != torch.float64:
        raise ValueError("score must be torch.float64")
    if score.ndim != 2 or any(dimension <= 0 for dimension in score.shape):
        raise ValueError("score must be a non-empty 2-D tensor")
    if not torch.isfinite(score).all() or (score < 0).any():
        raise ValueError("score must be finite and non-negative")
    return score.detach().to(device="cpu").contiguous()


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ScoreCache:
    """A score cache committed atomically by its JSON sidecar."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, identity: ScoreIdentity) -> tuple[Path, Path]:
        return self.root / f"{identity.key}.safetensors", self.root / f"{identity.key}.json"

    def put(self, identity: ScoreIdentity, score: torch.Tensor) -> str:
        validated = _validate_score(score)
        with self._identity_lock(identity):
            existing = self._verified_load(identity)
            if existing is not None:
                return identity.key
            return self._atomic_commit(identity, validated)

    @contextmanager
    def _identity_lock(self, identity: ScoreIdentity):
        import fcntl

        lock_path = self.root / f"{identity.key}.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise CacheIntegrityError("cannot safely open score cache lock") from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise CacheIntegrityError("score cache lock must be a regular non-symlink file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            except OSError as exc:
                raise CacheIntegrityError("cannot acquire score cache lock") from exc
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _atomic_commit(self, identity: ScoreIdentity, score: torch.Tensor) -> str:
        tensor_path, metadata_path = self._paths(identity)
        metadata = {
            "schema_version": SCORE_CACHE_SCHEMA_VERSION,
            "cache_key": identity.key,
            "identity": identity.as_dict(),
            "tensor_filename": tensor_path.name,
            "score": {
                "shape": list(score.shape),
                "dtype": str(score.dtype),
                "sha256": tensor_sha256(score),
            },
        }
        tensor_temporary: Path | None = None
        metadata_temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.root, prefix=f".{identity.key}.", suffix=".tmp", delete=False
            ) as stream:
                tensor_temporary = Path(stream.name)
            save_file({"score": score}, str(tensor_temporary))
            with tensor_temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            with tempfile.NamedTemporaryFile(
                dir=self.root,
                prefix=f".{identity.key}.",
                suffix=".tmp",
                delete=False,
                mode="wb",
            ) as stream:
                metadata_temporary = Path(stream.name)
                stream.write(canonical_json(metadata))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tensor_temporary, tensor_path)
            tensor_temporary = None
            self._fsync_directory()
            os.replace(metadata_temporary, metadata_path)
            metadata_temporary = None
            self._fsync_directory()
        finally:
            for temporary in (tensor_temporary, metadata_temporary):
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return identity.key

    def get(self, identity: ScoreIdentity) -> torch.Tensor | None:
        return self._verified_load(identity)

    def _verified_load(self, identity: ScoreIdentity) -> torch.Tensor | None:
        tensor_path, metadata_path = self._paths(identity)
        tensor_present = tensor_path.exists() or tensor_path.is_symlink()
        metadata_present = metadata_path.exists() or metadata_path.is_symlink()
        if not tensor_present and not metadata_present:
            return None
        if tensor_present != metadata_present:
            present = "tensor" if tensor_present else "metadata"
            missing = "metadata" if tensor_present else "tensor"
            raise CacheIntegrityError(f"cache {present} exists without {missing}")
        if (
            tensor_path.is_symlink()
            or metadata_path.is_symlink()
            or not tensor_path.is_file()
            or not metadata_path.is_file()
        ):
            raise CacheIntegrityError("cache entry must contain regular non-symlink files")
        try:
            raw_metadata = metadata_path.read_bytes()
            metadata = json.loads(raw_metadata.decode("utf-8"))
            if canonical_json(metadata) != raw_metadata:
                raise CacheIntegrityError("invalid non-canonical score cache metadata")
            expected = {
                "schema_version": SCORE_CACHE_SCHEMA_VERSION,
                "cache_key": identity.key,
                "identity": identity.as_dict(),
                "tensor_filename": tensor_path.name,
            }
            if not isinstance(metadata, dict) or set(metadata) != {*expected, "score"}:
                raise CacheIntegrityError("score cache metadata mismatch")
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise CacheIntegrityError("score cache metadata mismatch")
            tensors = load_file(str(tensor_path), device="cpu")
            if set(tensors) != {"score"}:
                raise CacheIntegrityError("score cache has unexpected tensor keys")
            score = _validate_score(tensors["score"])
            actual_score = {
                "shape": list(score.shape),
                "dtype": str(score.dtype),
                "sha256": tensor_sha256(score),
            }
            if metadata["score"] != actual_score:
                raise CacheIntegrityError("score cache tensor metadata mismatch")
            return score
        except CacheIntegrityError:
            raise
        except Exception as exc:
            raise CacheIntegrityError("invalid score cache entry") from exc

    def _fsync_directory(self) -> None:
        try:
            descriptor = os.open(self.root, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def score_entry_evidence(root: Path | str, identity: ScoreIdentity) -> dict[str, dict[str, str]]:
    """Return exact-file SHA-256 evidence after verifying the cache entry."""
    cache = ScoreCache(root)
    if cache.get(identity) is None:
        raise CacheIntegrityError("score cache entry is missing")
    tensor_path, metadata_path = cache._paths(identity)
    return {
        "json": {"filename": metadata_path.name, "sha256": _stream_sha256(metadata_path)},
        "safetensors": {"filename": tensor_path.name, "sha256": _stream_sha256(tensor_path)},
    }
