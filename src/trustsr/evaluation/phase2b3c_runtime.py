"""Host-free, non-circular runtime inventory for Phase 2B3-C."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from trustsr.evaluation.internal_test_predictions import (
    validate_cached_internal_test_prediction_provenance,
)
from trustsr.evaluation.phase2b3c_evidence import (
    PHASE2B3B_PUBLICATION_COMMIT,
)
from trustsr.evaluation.phase2b3c_evidence import (
    PRODUCER_REVISION as PHASE2B3B_PRODUCER_REVISION,
)
from trustsr.evaluation.phase2b3c_result_verify import verify_phase2b3c_result
from trustsr.jsonio import canonical_json
from trustsr.models.ldsr_assets import (
    CHECKPOINT_NAME,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    CONFIG_SHA256,
)
from trustsr.models.versions import OPENSR_MODEL_VERSION

SCHEMA = "trustsr.phase2b3c-evaluation-runtime.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_PYTHON_VERSION = re.compile(r"[0-9]+\.[0-9]+")
_VERSION = re.compile(r"[0-9][0-9A-Za-z.+_-]*")
_PACKAGES = ("numpy", "opensr-model", "rasterio", "torch", "trustsr")
_FORBIDDEN_VERSION_MARKERS = ("internal_test", "token", "secret", "host", "/", "\\")
_CONTEXT_KEYS = {
    "experiment_schema",
    "split",
    "post_manifest_sha256",
    "input_audit_sha256",
    "normalization_policy",
    "phase2b3b_publication_commit",
    "phase2b3b_acceptance_sha256",
}
_ROOT_KEYS = {
    "schema",
    "phase",
    "verification_scope",
    "cache_computation_verified",
    "dependencies",
    "model_inventory",
    "inputs",
    "artifacts",
    "evaluation",
    "revision",
}


@dataclass(frozen=True)
class VerifiedPhase2B3CRuntime:
    """Metadata-only runtime identity; never an acceptance credential."""

    runtime_sha256: str
    result_sha256: str
    cache_audit_sha256: str
    map_evidence_sha256: str
    input_receipt_sha256: str
    model_identity_sha256: str
    producer_revision: str
    verification_scope: str
    cache_computation_verified: bool

    def __post_init__(self) -> None:
        if (
            self.verification_scope != "metadata_inventory_only"
            or self.cache_computation_verified is not False
        ):
            raise ValueError("runtime verification scope is invalid")
        for value in (
            self.runtime_sha256,
            self.result_sha256,
            self.cache_audit_sha256,
            self.map_evidence_sha256,
            self.input_receipt_sha256,
            self.model_identity_sha256,
        ):
            _digest(value, "runtime identity digest")
        _revision(self.producer_revision, "runtime producer revision")


def _object(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} keys are invalid")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _revision(value: object, label: str) -> str:
    if type(value) is not str or _REVISION.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase Git revision")
    return value


def _version(value: object, label: str) -> str:
    if (
        type(value) is not str
        or _VERSION.fullmatch(value) is None
        or any(marker in value.casefold() for marker in _FORBIDDEN_VERSION_MARKERS)
    ):
        raise ValueError(f"{label} must be a host-free version")
    return value


def _canonical_document(value: object, label: str) -> tuple[dict[str, object], bytes]:
    if type(value) is bytes:
        payload = value
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label} must be canonical UTF-8 JSON") from exc
        if type(parsed) is not dict or canonical_json(parsed) != payload:
            raise ValueError(f"{label} must be canonical JSON")
        return parsed, payload
    if type(value) is not dict:
        raise TypeError(f"{label} must be canonical bytes or an exact JSON object")
    try:
        payload = canonical_json(value)
        parsed = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain JSON-native values") from exc
    return parsed, payload


def _dependencies(value: object) -> dict[str, object]:
    dependencies = _object(
        value, {"python", "uv_lock_sha256", "packages"}, "runtime dependencies"
    )
    python = dependencies["python"]
    if type(python) is not str or _PYTHON_VERSION.fullmatch(python) is None:
        raise ValueError("runtime dependencies python must be a major.minor version")
    lock_digest = _digest(dependencies["uv_lock_sha256"], "uv.lock digest")
    packages = _object(dependencies["packages"], set(_PACKAGES), "runtime packages")
    normalized_packages = {
        name: _version(packages[name], f"runtime package {name}") for name in _PACKAGES
    }
    return {
        "python": python,
        "uv_lock_sha256": lock_digest,
        "packages": normalized_packages,
    }


def _model_inventory_from_provenance(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("model provenance must be a mapping")
    provenance = validate_cached_internal_test_prediction_provenance(value, seed=3407)
    identity = {
        key: provenance[key]
        for key in provenance
        if key not in _CONTEXT_KEYS and key != "seed"
    }
    inventory = {**identity, "seeds": [3407, 3408, 3409, 3410, 3411]}
    inventory["scientific_identity_sha256"] = hashlib.sha256(
        canonical_json(inventory)
    ).hexdigest()
    return inventory


def _validate_model_inventory(value: object) -> dict[str, object]:
    seedless_keys = {
        "name",
        "scale",
        "implementation_schema_version",
        "opensr_model_version",
        "torch_version",
        "cuda_runtime",
        "checkpoint_name",
        "checkpoint_size",
        "checkpoint_sha256",
        "config_sha256",
        "sampling_steps",
        "sampling_eta",
        "sampling_temperature",
        "histogram_matching",
        "output_policy",
    }
    inventory = _object(
        value, seedless_keys | {"seeds", "scientific_identity_sha256"}, "model inventory"
    )
    expected = {
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
        "seeds": [3407, 3408, 3409, 3410, 3411],
    }
    if any(
        type(inventory.get(key)) is not type(item) or inventory.get(key) != item
        for key, item in expected.items()
    ):
        raise ValueError("model inventory differs from the frozen identity")
    _version(inventory["torch_version"], "model torch version")
    _version(inventory["cuda_runtime"], "model CUDA runtime")
    projected = {key: inventory[key] for key in inventory if key != "scientific_identity_sha256"}
    if inventory["scientific_identity_sha256"] != hashlib.sha256(
        canonical_json(projected)
    ).hexdigest():
        raise ValueError("model scientific identity digest is invalid")
    return inventory


def _expected_runtime(
    result: object, model_inventory: dict[str, object], dependencies: dict[str, object]
) -> dict[str, object]:
    result_value, result_payload = _canonical_document(result, "runtime result")
    verified = verify_phase2b3c_result(result_payload)
    digests = result_value["digests"]
    return {
        "schema": SCHEMA,
        "phase": "internal_test_evaluation",
        "verification_scope": "metadata_inventory_only",
        "cache_computation_verified": False,
        "dependencies": dependencies,
        "model_inventory": model_inventory,
        "inputs": {
            "input_receipt_sha256": verified.input_receipt_sha256,
            "ordered_inputs_sha256": verified.ordered_inputs_sha256,
            "ordered_sample_ids_sha256": digests["ordered_sample_ids_sha256"],
            "ordered_membership_sha256": digests["ordered_membership_sha256"],
        },
        "artifacts": {
            "result_sha256": hashlib.sha256(result_payload).hexdigest(),
            "cache_audit_sha256": verified.cache_audit_sha256,
            "map_evidence_sha256": verified.map_evidence_sha256,
        },
        "evaluation": {
            "evaluation_id": verified.evaluation_id,
            "permit_sha256": verified.permit_sha256,
            "ledger_event_sha256": verified.ledger_event_sha256,
        },
        "revision": {
            "producer_revision": verified.producer_revision,
            "phase2b3b_publication_commit": PHASE2B3B_PUBLICATION_COMMIT,
            "phase2b3b_producer_revision": PHASE2B3B_PRODUCER_REVISION,
        },
    }


def build_phase2b3c_runtime_manifest(
    *,
    result: object,
    model_provenance: Mapping[str, object],
    dependencies: Mapping[str, object],
) -> dict[str, object]:
    """Build a deterministic inventory with no path, host, or later-artifact identity."""

    runtime = _expected_runtime(
        result,
        _model_inventory_from_provenance(model_provenance),
        _dependencies(dependencies),
    )
    canonical_json(runtime)
    return runtime


def verify_phase2b3c_runtime_manifest(
    runtime: object, *, result: object
) -> VerifiedPhase2B3CRuntime:
    """Verify runtime inventory and its result cross-binding without computation claims."""

    value, payload = _canonical_document(runtime, "runtime manifest")
    _object(value, _ROOT_KEYS, "runtime manifest")
    dependencies = _dependencies(value["dependencies"])
    model_inventory = _validate_model_inventory(value["model_inventory"])
    expected = _expected_runtime(result, model_inventory, dependencies)
    if value != expected:
        raise ValueError("runtime manifest differs from its result or frozen inventory")
    artifacts = value["artifacts"]
    inputs = value["inputs"]
    revision = value["revision"]
    return VerifiedPhase2B3CRuntime(
        runtime_sha256=hashlib.sha256(payload).hexdigest(),
        result_sha256=artifacts["result_sha256"],
        cache_audit_sha256=artifacts["cache_audit_sha256"],
        map_evidence_sha256=artifacts["map_evidence_sha256"],
        input_receipt_sha256=inputs["input_receipt_sha256"],
        model_identity_sha256=model_inventory["scientific_identity_sha256"],
        producer_revision=revision["producer_revision"],
        verification_scope="metadata_inventory_only",
        cache_computation_verified=False,
    )
