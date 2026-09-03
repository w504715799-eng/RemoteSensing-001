"""Strict scientific provenance tests for the Phase 2B3-B LDSR model."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from trustsr.evaluation.calibration_model_identity import (
    CalibrationModelIdentity,
    validate_calibration_model_identity,
)
from trustsr.jsonio import canonical_json
from trustsr.models.ldsr_assets import (
    CHECKPOINT_NAME,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    CHECKPOINT_URL,
    CONFIG_SHA256,
)
from trustsr.models.versions import OPENSR_MODEL_VERSION


def _provenance(*, seed: int = 3407) -> dict[str, object]:
    return {
        "name": "ldsr-s2-x4",
        "scale": 4,
        "implementation_schema_version": 1,
        "opensr_model_version": OPENSR_MODEL_VERSION,
        "torch_version": "2.7.1+cu128",
        "cuda_runtime": "12.8",
        "checkpoint_name": CHECKPOINT_NAME,
        "checkpoint_url": CHECKPOINT_URL,
        "checkpoint_size": CHECKPOINT_SIZE,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "device": "cuda",
        "seed": seed,
        "sampling_steps": 100,
        "sampling_eta": 0.95,
        "sampling_temperature": 1.0,
        "histogram_matching": True,
        "output_policy": "clip_to_[0,1]",
    }


def test_normalizes_complete_real_provenance_to_frozen_host_free_identity() -> None:
    identity = validate_calibration_model_identity(_provenance())

    assert isinstance(identity, CalibrationModelIdentity)
    assert identity.as_dict() == {
        "name": "ldsr-s2-x4",
        "scale": 4,
        "implementation_schema_version": 1,
        "opensr_model_version": OPENSR_MODEL_VERSION,
        "torch_version": "2.7.1+cu128",
        "cuda_runtime": "12.8",
        "checkpoint_name": CHECKPOINT_NAME,
        "checkpoint_size": CHECKPOINT_SIZE,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "seed": 3407,
        "sampling_steps": 100,
        "sampling_eta": 0.95,
        "sampling_temperature": 1.0,
        "histogram_matching": True,
        "output_policy": "clip_to_[0,1]",
    }
    encoded = canonical_json(identity.as_dict()).decode("utf-8")
    for forbidden in ("checkpoint_url", "device", "backend", "cuda:0", "/home/"):
        assert forbidden not in encoded
    with pytest.raises(FrozenInstanceError):
        identity.seed = 3408  # type: ignore[misc]


def test_supports_exactly_the_frozen_k5_seed_set() -> None:
    assert tuple(
        validate_calibration_model_identity(_provenance(seed=seed)).seed
        for seed in (3407, 3408, 3409, 3410, 3411)
    ) == (3407, 3408, 3409, 3410, 3411)


def test_as_dict_returns_fresh_json_native_values() -> None:
    identity = validate_calibration_model_identity(_provenance())
    first = identity.as_dict()
    second = identity.as_dict()

    assert first == second
    assert first is not second
    first["seed"] = 3411
    assert second["seed"] == 3407
    assert canonical_json(second)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("name", "other-model"),
        ("scale", True),
        ("implementation_schema_version", True),
        ("opensr_model_version", "0.0.0"),
        ("torch_version", ""),
        ("torch_version", 271),
        ("torch_version", "/home/operator/torch-build"),
        ("cuda_runtime", ""),
        ("cuda_runtime", None),
        ("cuda_runtime", "12.8\nremote-host"),
        ("checkpoint_name", "other.ckpt"),
        ("checkpoint_url", "https://example.invalid/model.ckpt"),
        ("checkpoint_size", True),
        ("checkpoint_size", CHECKPOINT_SIZE + 1),
        ("checkpoint_sha256", "0" * 64),
        ("config_sha256", "0" * 64),
        ("device", "cuda:0"),
        ("seed", True),
        ("seed", 3412),
        ("sampling_steps", True),
        ("sampling_steps", 99),
        ("sampling_eta", 1),
        ("sampling_eta", float("nan")),
        ("sampling_temperature", 1),
        ("sampling_temperature", float("nan")),
        ("histogram_matching", 1),
        ("histogram_matching", False),
        ("output_policy", "unbounded"),
    ),
)
def test_rejects_wrong_or_weakly_typed_scientific_identity(
    field: str, value: object
) -> None:
    provenance = _provenance()
    provenance[field] = value

    with pytest.raises((TypeError, ValueError)):
        validate_calibration_model_identity(provenance)


@pytest.mark.parametrize("fault", ("missing", "extra", "backend"))
def test_rejects_non_exact_provenance_schema(fault: str) -> None:
    provenance = _provenance()
    if fault == "missing":
        provenance.pop("config_sha256")
    elif fault == "extra":
        provenance["host_path"] = "/tmp/model"
    else:
        provenance["backend"] = "tiny-cpu-fake"

    with pytest.raises(ValueError, match="keys"):
        validate_calibration_model_identity(provenance)


def test_rejects_non_mapping_input() -> None:
    with pytest.raises(TypeError, match="mapping"):
        validate_calibration_model_identity([])  # type: ignore[arg-type]


def test_frozen_identity_public_constructor_cannot_bypass_the_contract() -> None:
    identity = validate_calibration_model_identity(_provenance())

    with pytest.raises(ValueError):
        replace(identity, checkpoint_sha256="0" * 64)
    with pytest.raises(ValueError):
        replace(identity, torch_version="/home/operator/private-build")
