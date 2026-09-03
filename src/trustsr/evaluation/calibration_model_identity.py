"""Frozen scientific LDSR model identity for Phase 2B3-B calibration."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from trustsr.jsonio import canonical_json
from trustsr.models.ldsr_assets import (
    CHECKPOINT_NAME,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    CHECKPOINT_URL,
    CONFIG_SHA256,
)
from trustsr.models.versions import OPENSR_MODEL_VERSION

_SEEDS = (3407, 3408, 3409, 3410, 3411)
_INPUT_KEYS = {
    "name",
    "scale",
    "implementation_schema_version",
    "opensr_model_version",
    "torch_version",
    "cuda_runtime",
    "checkpoint_name",
    "checkpoint_url",
    "checkpoint_size",
    "checkpoint_sha256",
    "config_sha256",
    "device",
    "seed",
    "sampling_steps",
    "sampling_eta",
    "sampling_temperature",
    "histogram_matching",
    "output_policy",
}
_CACHED_KEYS = _INPUT_KEYS - {"checkpoint_url", "device"}
_VERSION = re.compile(r"^[0-9][0-9A-Za-z.+_-]*$")
_FORBIDDEN_VERSION_MARKERS = ("internal_test", "token", "secret", "host")
_FIXED_IDENTITY_FIELDS = {
    "name": "ldsr-s2-x4",
    "scale": 4,
    "implementation_schema_version": 1,
    "opensr_model_version": OPENSR_MODEL_VERSION,
    "checkpoint_name": CHECKPOINT_NAME,
    "checkpoint_size": CHECKPOINT_SIZE,
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "config_sha256": CONFIG_SHA256,
    "sampling_steps": 100,
    "sampling_eta": 0.95,
    "sampling_temperature": 1.0,
    "histogram_matching": True,
    "output_policy": "clip_to_[0,1]",
}


def _validated_version(value: object, key: str) -> str:
    if (
        type(value) is not str
        or _VERSION.fullmatch(value) is None
        or any(marker in value.casefold() for marker in _FORBIDDEN_VERSION_MARKERS)
    ):
        raise ValueError(f"LDSR provenance {key} must be a host-free version string")
    return value


@dataclass(frozen=True)
class CalibrationModelIdentity:
    """Host-free scientific identity distilled from one real LDSR seed view."""

    name: str
    scale: int
    implementation_schema_version: int
    opensr_model_version: str
    torch_version: str
    cuda_runtime: str
    checkpoint_name: str
    checkpoint_size: int
    checkpoint_sha256: str
    config_sha256: str
    seed: int
    sampling_steps: int
    sampling_eta: float
    sampling_temperature: float
    histogram_matching: bool
    output_policy: str

    def __post_init__(self) -> None:
        for key, expected in _FIXED_IDENTITY_FIELDS.items():
            value = getattr(self, key)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f"LDSR identity {key} differs from the frozen contract")
        if type(self.seed) is not int or self.seed not in _SEEDS:
            raise ValueError("LDSR identity seed is outside the frozen K5 set")
        _validated_version(self.torch_version, "torch_version")
        _validated_version(self.cuda_runtime, "cuda_runtime")

    def as_dict(self) -> dict[str, object]:
        """Return a fresh canonical-JSON-native scientific identity."""

        return {
            "name": self.name,
            "scale": self.scale,
            "implementation_schema_version": self.implementation_schema_version,
            "opensr_model_version": self.opensr_model_version,
            "torch_version": self.torch_version,
            "cuda_runtime": self.cuda_runtime,
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_size": self.checkpoint_size,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "seed": self.seed,
            "sampling_steps": self.sampling_steps,
            "sampling_eta": self.sampling_eta,
            "sampling_temperature": self.sampling_temperature,
            "histogram_matching": self.histogram_matching,
            "output_policy": self.output_policy,
        }


def _require_exact(provenance: Mapping[str, object], key: str, expected: object) -> None:
    value = provenance[key]
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"LDSR provenance {key} differs from the frozen identity")


def _require_version(provenance: Mapping[str, object], key: str) -> str:
    return _validated_version(provenance[key], key)


def validate_calibration_model_identity(
    provenance: Mapping[str, object],
) -> CalibrationModelIdentity:
    """Validate complete LDSR provenance and return its host-free scientific identity."""

    if not isinstance(provenance, Mapping):
        raise TypeError("LDSR provenance must be a mapping")
    if set(provenance) != _INPUT_KEYS:
        raise ValueError("LDSR provenance keys do not match the frozen schema")

    expected = {
        **_FIXED_IDENTITY_FIELDS,
        "checkpoint_url": CHECKPOINT_URL,
        "device": "cuda",
    }
    for key, value in expected.items():
        _require_exact(provenance, key, value)

    seed = provenance["seed"]
    if type(seed) is not int or seed not in _SEEDS:
        raise ValueError("LDSR provenance seed is outside the frozen K5 set")
    torch_version = _require_version(provenance, "torch_version")
    cuda_runtime = _require_version(provenance, "cuda_runtime")

    identity = CalibrationModelIdentity(
        name="ldsr-s2-x4",
        scale=4,
        implementation_schema_version=1,
        opensr_model_version=OPENSR_MODEL_VERSION,
        torch_version=torch_version,
        cuda_runtime=cuda_runtime,
        checkpoint_name=CHECKPOINT_NAME,
        checkpoint_size=CHECKPOINT_SIZE,
        checkpoint_sha256=CHECKPOINT_SHA256,
        config_sha256=CONFIG_SHA256,
        seed=seed,
        sampling_steps=100,
        sampling_eta=0.95,
        sampling_temperature=1.0,
        histogram_matching=True,
        output_policy="clip_to_[0,1]",
    )
    canonical_json(identity.as_dict())
    return identity


def validate_cached_calibration_model_identity(
    provenance: Mapping[str, object],
) -> CalibrationModelIdentity:
    """Revalidate the exact host-free identity permitted in calibration caches."""

    if not isinstance(provenance, Mapping):
        raise TypeError("cached LDSR identity must be a mapping")
    if set(provenance) != _CACHED_KEYS:
        raise ValueError("cached LDSR identity keys do not match the frozen schema")
    try:
        identity = CalibrationModelIdentity(**dict(provenance))
    except (TypeError, ValueError) as exc:
        raise ValueError("cached LDSR identity differs from the frozen contract") from exc
    canonical_json(identity.as_dict())
    return identity
