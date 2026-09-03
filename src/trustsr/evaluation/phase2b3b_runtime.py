"""Host-free, non-circular runtime metadata manifests for Phase 2B3-B."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from trustsr.evaluation.calibration_cache_verify import verify_calibration_cache_audit
from trustsr.evaluation.calibration_input_receipt import VerifiedCalibrationInputReceipt
from trustsr.evaluation.calibration_model_identity import (
    CalibrationModelIdentity,
    validate_cached_calibration_model_identity,
)
from trustsr.evaluation.calibration_predictions import (
    validate_cached_calibration_prediction_provenance,
)
from trustsr.evaluation.phase2b3b_evidence import (
    INPUT_AUDIT_SHA256,
    POST_MANIFEST_SHA256,
    PRODUCER_REVISION,
    PUBLICATION_COMMIT,
)
from trustsr.evaluation.phase2b3b_result_verify import VerifiedPhase2B3BResult
from trustsr.evaluation.phase2b3b_revision import verify_recorded_phase2b3b_revision
from trustsr.jsonio import canonical_json

SCHEMA = "trustsr.phase2b3b-calibration-runtime.v1"
_RESULT_SCHEMA = "trustsr.phase2b3b-calibration.v1"
_AUDIT_SCHEMA = "trustsr.phase2b3b-calibration-cache-audit.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_PYTHON_VERSION = re.compile(r"[0-9]+\.[0-9]+")
_VERSION = re.compile(r"[0-9][0-9A-Za-z.+_-]*")
_FORBIDDEN_VERSION_MARKERS = ("internal_test", "token", "secret", "host")
_PACKAGES = ("numpy", "opensr-model", "rasterio", "torch", "trustsr")
_SEEDS = (3407, 3408, 3409, 3410, 3411)
_RUNTIME_KEYS = {
    "schema",
    "phase",
    "verification_scope",
    "cache_computation_verified",
    "dependencies",
    "model_inventory",
    "inputs",
    "artifacts",
    "revision",
}
_DEPENDENCIES_KEYS = {"python", "uv_lock_sha256", "packages"}
_INPUT_KEYS = {
    "post_manifest_sha256",
    "input_audit_sha256",
    "normalization_policy",
    "crop_policy",
    "bands",
    "scale",
    "ordered_sample_ids_sha256",
    "ordered_membership_sha256",
    "input_receipt_sha256",
    "ordered_inputs_sha256",
}
_ARTIFACT_KEYS = {
    "result_sha256",
    "cache_audit_sha256",
    "map_evidence_sha256",
    "cache_audit_identity_digests",
}
_REVISION_KEYS = {
    "producer_revision",
    "phase2b3a_calculation_revision",
    "phase2b3a_publication_commit",
}
_RESULT_KEYS = {
    "schema",
    "split",
    "upstream",
    "producer_revision",
    "frozen",
    "target",
    "threshold",
    "all_abstain",
    "risk_bound",
    "counts",
    "coverage",
    "radiometry",
    "samples",
    "input_receipt_sha256",
    "ordered_inputs_sha256",
    "cache_audit_sha256",
    "map_evidence_sha256",
    "phase_decision",
}
_UPSTREAM_KEYS = {
    "post_manifest_sha256",
    "input_audit_sha256",
    "phase2b3a_publication_commit",
    "phase2b3a_calculation_revision",
    "evidence_sha256s",
    "ordered_sample_ids_sha256",
    "ordered_membership_sha256",
}
_FROZEN_KEYS = {"score", "risk", "input"}
_FROZEN_INPUT_KEYS = {"normalization_policy", "crop_policy", "bands", "scale"}
_AUDIT_IDENTITY_DIGEST_KEYS = {
    "prediction_identities_sha256",
    "score_identities_sha256",
    "risk_receipts_sha256",
}


@dataclass(frozen=True)
class VerifiedPhase2B3BRuntime:
    """Metadata-only runtime receipt; it cannot authorize acceptance."""

    schema: str
    verification_scope: str
    cache_computation_verified: bool
    runtime_sha256: str
    result_sha256: str
    cache_audit_sha256: str
    input_receipt_sha256: str
    ordered_inputs_sha256: str
    map_evidence_sha256: str
    producer_revision: str
    model_identity_sha256: str

    def __post_init__(self) -> None:
        if (
            self.schema != SCHEMA
            or self.verification_scope != "metadata_inventory_only"
            or self.cache_computation_verified is not False
        ):
            raise ValueError("runtime receipt scope is invalid")
        for value in (
            self.runtime_sha256,
            self.result_sha256,
            self.cache_audit_sha256,
            self.input_receipt_sha256,
            self.ordered_inputs_sha256,
            self.map_evidence_sha256,
            self.model_identity_sha256,
        ):
            _digest(value, "runtime receipt digest")
        _revision(self.producer_revision, "runtime receipt producer revision")


def _exact_mapping(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} must be an exact JSON object")
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


def _canonical_document(
    value: object, *, schema: str, label: str
) -> tuple[dict[str, object], bytes]:
    if type(value) is bytes:
        payload = value
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label} must be canonical UTF-8 JSON") from exc
        if type(parsed) is not dict or canonical_json(parsed) != payload:
            raise ValueError(f"{label} must be canonical JSON")
    elif type(value) is dict:
        try:
            payload = canonical_json(value)
            parsed = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be canonical JSON data") from exc
    else:
        raise TypeError(f"{label} must be canonical bytes or an exact JSON object")
    if parsed.get("schema") != schema:
        raise ValueError(f"{label} schema is invalid")
    return parsed, payload


def _reject_leaks(value: object, *, key: str = "runtime") -> None:
    if type(value) is dict:
        for nested_key, nested in value.items():
            if type(nested_key) is not str:
                raise TypeError("runtime keys must be built-in strings")
            lowered = nested_key.casefold()
            if any(
                marker in lowered
                for marker in (
                    "path",
                    "endpoint",
                    "timestamp",
                    "hostname",
                    "credential",
                    "password",
                    "secret",
                    "token",
                    "uuid",
                    "device",
                )
            ):
                raise ValueError("runtime contains forbidden host metadata")
            _reject_leaks(nested, key=nested_key)
        return
    if type(value) is list:
        for nested in value:
            _reject_leaks(nested, key=key)
        return
    if type(value) is str:
        lowered = value.casefold()
        if "development" in lowered or "internal_test" in lowered:
            raise ValueError("runtime contains a forbidden split")
        if (
            value.startswith(("/", "\\\\"))
            or "://" in value
            or re.match(r"^[A-Za-z]:[\\/]", value) is not None
            or any(marker in lowered for marker in ("token=", "secret=", "host="))
        ):
            raise ValueError("runtime contains forbidden host metadata")
        return
    if type(value) not in (int, float, bool, type(None)):
        raise TypeError(f"runtime {key} must contain exact JSON values")


def _validate_dependencies(value: object) -> dict[str, object]:
    dependencies = _exact_mapping(value, _DEPENDENCIES_KEYS, "runtime dependencies")
    python = _exact_mapping(dependencies["python"], {"major_minor"}, "runtime Python")
    major_minor = python["major_minor"]
    if type(major_minor) is not str or _PYTHON_VERSION.fullmatch(major_minor) is None:
        raise ValueError("runtime Python major/minor is invalid")
    packages = _exact_mapping(
        dependencies["packages"], set(_PACKAGES), "runtime package inventory"
    )
    return {
        "python": {"major_minor": major_minor},
        "uv_lock_sha256": _digest(dependencies["uv_lock_sha256"], "runtime uv.lock digest"),
        "packages": {
            name: _version(packages[name], f"runtime package {name}")
            for name in _PACKAGES
        },
    }


def _capture_dependencies(project_root: Path) -> dict[str, object]:
    if not isinstance(project_root, Path):
        raise TypeError("project_root must be a pathlib.Path")
    lock_path = project_root / "uv.lock"
    try:
        if lock_path.is_symlink() or not lock_path.is_file():
            raise ValueError("clean checkout must contain a regular uv.lock")
        lock_bytes = lock_path.read_bytes()
    except OSError as exc:
        raise ValueError("clean checkout must contain a readable uv.lock") from exc
    return {
        "python": {"major_minor": f"{sys.version_info.major}.{sys.version_info.minor}"},
        "uv_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "packages": {name: importlib.metadata.version(name) for name in _PACKAGES},
    }


def _validated_result_receipt(
    value: object, *, result_sha256: str, audit_sha256: str
) -> VerifiedPhase2B3BResult:
    if type(value) is not VerifiedPhase2B3BResult:
        raise TypeError("runtime requires an exact verified result receipt")
    if (
        value.schema != "trustsr.phase2b3b-calibration-result-metadata-verification.v1"
        or value.verification_scope != "metadata_consistency_only"
        or value.cache_computation_verified is not False
        or value.result_sha256 != result_sha256
        or value.cache_audit_sha256 != audit_sha256
    ):
        raise ValueError("result receipt differs from canonical runtime inputs")
    for digest in (
        value.ordered_sample_ids_sha256,
        value.ordered_membership_sha256,
        value.input_receipt_sha256,
        value.ordered_inputs_sha256,
        value.map_evidence_sha256,
        value.radiometry_aggregate_sha256,
    ):
        _digest(digest, "verified result receipt digest")
    _revision(value.producer_revision, "verified result producer revision")
    if type(value.phase_decision) is not str or not value.phase_decision:
        raise ValueError("verified result phase decision is invalid")
    return value


def _validated_input_receipt(value: object) -> VerifiedCalibrationInputReceipt:
    if type(value) is not VerifiedCalibrationInputReceipt:
        raise TypeError("runtime requires an exact verified input receipt")
    for digest in (
        value.source_sha256,
        value.ordered_inputs_sha256,
        value.ordered_sample_ids_sha256,
        value.ordered_membership_sha256,
    ):
        _digest(digest, "verified input receipt digest")
    if type(value.sample_count) is not int or value.sample_count != 120:
        raise ValueError("verified input receipt count is invalid")
    return value


def _result_projection(
    result: dict[str, object],
    receipt: VerifiedPhase2B3BResult,
    input_receipt: VerifiedCalibrationInputReceipt,
) -> tuple[dict[str, object], dict[str, object]]:
    value = _exact_mapping(result, _RESULT_KEYS, "runtime result")
    if value["schema"] != _RESULT_SCHEMA or value["split"] != "calibration":
        raise ValueError("runtime result schema or split is invalid")
    upstream = _exact_mapping(value["upstream"], _UPSTREAM_KEYS, "runtime result upstream")
    frozen = _exact_mapping(value["frozen"], _FROZEN_KEYS, "runtime result frozen")
    frozen_input = _exact_mapping(
        frozen["input"], _FROZEN_INPUT_KEYS, "runtime result frozen input"
    )
    bands = frozen_input["bands"]
    if type(bands) is not list or bands != ["B04", "B03", "B02", "B08"]:
        raise ValueError("runtime result bands are invalid")
    if (
        frozen_input["normalization_policy"] != "uint16_saturate_10000_divide_10000_v2"
        or frozen_input["crop_policy"] != "center_crop_lr_1_hr_4_v1"
        or type(frozen_input["scale"]) is not int
        or frozen_input["scale"] != 4
    ):
        raise ValueError("runtime result frozen input is invalid")
    producer = _revision(value["producer_revision"], "runtime result producer revision")
    if (
        producer != receipt.producer_revision
        or _digest(value["cache_audit_sha256"], "runtime result audit digest")
        != receipt.cache_audit_sha256
        or _digest(value["input_receipt_sha256"], "runtime result input digest")
        != input_receipt.source_sha256
        or _digest(value["ordered_inputs_sha256"], "runtime result ordered inputs digest")
        != input_receipt.ordered_inputs_sha256
        or _digest(value["map_evidence_sha256"], "runtime result map digest")
        != receipt.map_evidence_sha256
        or _digest(upstream["ordered_sample_ids_sha256"], "runtime result order digest")
        != input_receipt.ordered_sample_ids_sha256
        or _digest(upstream["ordered_membership_sha256"], "runtime result membership digest")
        != input_receipt.ordered_membership_sha256
    ):
        raise ValueError("runtime result projections differ from verified receipts")
    post_manifest_sha256 = _digest(upstream["post_manifest_sha256"], "runtime manifest")
    input_audit_sha256 = _digest(upstream["input_audit_sha256"], "runtime input audit")
    calculation_revision = _revision(
        upstream["phase2b3a_calculation_revision"], "runtime calculation revision"
    )
    publication_commit = _revision(
        upstream["phase2b3a_publication_commit"], "runtime publication revision"
    )
    if (
        post_manifest_sha256 != POST_MANIFEST_SHA256
        or input_audit_sha256 != INPUT_AUDIT_SHA256
        or calculation_revision != PRODUCER_REVISION
        or publication_commit != PUBLICATION_COMMIT
    ):
        raise ValueError("runtime result upstream differs from frozen Phase2B3A evidence")
    inputs = {
        "post_manifest_sha256": post_manifest_sha256,
        "input_audit_sha256": input_audit_sha256,
        "normalization_policy": frozen_input["normalization_policy"],
        "crop_policy": frozen_input["crop_policy"],
        "bands": list(bands),
        "scale": 4,
        "ordered_sample_ids_sha256": input_receipt.ordered_sample_ids_sha256,
        "ordered_membership_sha256": input_receipt.ordered_membership_sha256,
        "input_receipt_sha256": input_receipt.source_sha256,
        "ordered_inputs_sha256": input_receipt.ordered_inputs_sha256,
    }
    revision = {
        "producer_revision": producer,
        "phase2b3a_calculation_revision": calculation_revision,
        "phase2b3a_publication_commit": publication_commit,
    }
    return inputs, revision


def _model_inventory(audit: dict[str, object]) -> dict[str, object]:
    samples = audit.get("samples")
    if type(samples) is not list or len(samples) != 120:
        raise ValueError("runtime audit samples are invalid")
    identity_projection: dict[str, object] | None = None
    observed_seeds: list[int] = []
    for sample in samples:
        if type(sample) is not dict or type(sample.get("predictions")) is not list:
            raise ValueError("runtime audit prediction entries are invalid")
        predictions = sample["predictions"]
        if len(predictions) != len(_SEEDS):
            raise ValueError("runtime audit must contain the fixed K5 predictions")
        for prediction, seed in zip(predictions, _SEEDS, strict=True):
            if type(prediction) is not dict or prediction.get("seed") != seed:
                raise ValueError("runtime audit seed order is invalid")
            raw_identity = prediction.get("identity")
            if (
                type(raw_identity) is not dict
                or type(raw_identity.get("model_provenance")) is not dict
            ):
                raise ValueError("runtime audit provenance is invalid")
            provenance = validate_cached_calibration_prediction_provenance(
                raw_identity["model_provenance"], seed=seed
            )
            identity = validate_cached_calibration_model_identity(
                {
                    key: provenance[key]
                    for key in CalibrationModelIdentity.__dataclass_fields__
                }
            )
            if identity.seed != seed:
                raise ValueError("runtime audit provenance seed is invalid")
            projection = identity.as_dict()
            projection.pop("seed")
            if identity_projection is None:
                identity_projection = projection
            elif projection != identity_projection:
                raise ValueError("runtime audit has mixed model scientific identities")
            observed_seeds.append(seed)
    if identity_projection is None or tuple(observed_seeds[: len(_SEEDS)]) != _SEEDS:
        raise ValueError("runtime audit has no fixed model identity")
    if any(
        observed_seeds[index : index + len(_SEEDS)] != list(_SEEDS)
        for index in range(0, 600, 5)
    ):
        raise ValueError("runtime audit has inconsistent K5 seed slots")
    return {"identity": identity_projection, "seeds": list(_SEEDS)}


def _expected_runtime(
    result: object,
    cache_audit: object,
    *,
    result_verification: object,
    input_verification: object,
    project_root: Path,
) -> dict[str, object]:
    result_value, result_payload = _canonical_document(
        result, schema=_RESULT_SCHEMA, label="runtime result"
    )
    audit_value, _ = _canonical_document(
        cache_audit, schema=_AUDIT_SCHEMA, label="runtime cache audit"
    )
    audit_verification = verify_calibration_cache_audit(audit_value)
    audit_sha256 = audit_verification["digests"]["audit_sha256"]
    result_sha256 = hashlib.sha256(result_payload).hexdigest()
    receipt = _validated_result_receipt(
        result_verification, result_sha256=result_sha256, audit_sha256=audit_sha256
    )
    input_receipt = _validated_input_receipt(input_verification)
    inputs, revision = _result_projection(result_value, receipt, input_receipt)
    if verify_recorded_phase2b3b_revision(project_root, revision["producer_revision"]) != revision[
        "producer_revision"
    ]:
        raise ValueError("runtime recorded revision gate is inconsistent")
    audit_digests = audit_verification["digests"]
    identity_digests = {
        key: _digest(audit_digests[key], f"runtime audit {key}")
        for key in _AUDIT_IDENTITY_DIGEST_KEYS
    }
    model_inventory = _model_inventory(audit_value)
    runtime = {
        "schema": SCHEMA,
        "phase": "calibration",
        "verification_scope": "metadata_inventory_only",
        "cache_computation_verified": False,
        "dependencies": _validate_dependencies(_capture_dependencies(project_root)),
        "model_inventory": model_inventory,
        "inputs": inputs,
        "artifacts": {
            "result_sha256": result_sha256,
            "cache_audit_sha256": audit_sha256,
            "map_evidence_sha256": receipt.map_evidence_sha256,
            "cache_audit_identity_digests": identity_digests,
        },
        "revision": revision,
    }
    _validate_runtime_shape(runtime)
    _reject_leaks(runtime)
    return json.loads(canonical_json(runtime))


def _validate_runtime_shape(value: object) -> dict[str, object]:
    runtime = _exact_mapping(value, _RUNTIME_KEYS, "runtime manifest")
    if (
        runtime["schema"] != SCHEMA
        or runtime["phase"] != "calibration"
        or runtime["verification_scope"] != "metadata_inventory_only"
        or runtime["cache_computation_verified"] is not False
    ):
        raise ValueError("runtime manifest scope is invalid")
    _validate_dependencies(runtime["dependencies"])
    inventory = _exact_mapping(runtime["model_inventory"], {"identity", "seeds"}, "model inventory")
    identity = inventory["identity"]
    if type(identity) is not dict or "seed" in identity:
        raise ValueError("runtime model identity projection is invalid")
    try:
        CalibrationModelIdentity(seed=_SEEDS[0], **identity)
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime model identity is invalid") from exc
    if inventory["seeds"] != list(_SEEDS) or type(inventory["seeds"]) is not list:
        raise ValueError("runtime model seed inventory is invalid")
    inputs = _exact_mapping(runtime["inputs"], _INPUT_KEYS, "runtime inputs")
    for key in (
        "post_manifest_sha256",
        "input_audit_sha256",
        "ordered_sample_ids_sha256",
        "ordered_membership_sha256",
        "input_receipt_sha256",
        "ordered_inputs_sha256",
    ):
        _digest(inputs[key], f"runtime input {key}")
    artifacts = _exact_mapping(runtime["artifacts"], _ARTIFACT_KEYS, "runtime artifacts")
    for key in ("result_sha256", "cache_audit_sha256", "map_evidence_sha256"):
        _digest(artifacts[key], f"runtime artifact {key}")
    digests = _exact_mapping(
        artifacts["cache_audit_identity_digests"],
        _AUDIT_IDENTITY_DIGEST_KEYS,
        "runtime audit identity digests",
    )
    for key in _AUDIT_IDENTITY_DIGEST_KEYS:
        _digest(digests[key], f"runtime audit {key}")
    revision = _exact_mapping(runtime["revision"], _REVISION_KEYS, "runtime revision")
    for key in _REVISION_KEYS:
        _revision(revision[key], f"runtime {key}")
    return runtime


def build_phase2b3b_runtime_manifest(
    result: object,
    cache_audit: object,
    *,
    result_verification: VerifiedPhase2B3BResult,
    input_verification: VerifiedCalibrationInputReceipt,
    project_root: Path,
) -> dict[str, object]:
    """Compose a runtime inventory from existing verified result/input/cache evidence."""

    return _expected_runtime(
        result,
        cache_audit,
        result_verification=result_verification,
        input_verification=input_verification,
        project_root=project_root,
    )


def verify_phase2b3b_runtime_manifest(
    runtime: object,
    result: object,
    cache_audit: object,
    *,
    result_verification: VerifiedPhase2B3BResult,
    input_verification: VerifiedCalibrationInputReceipt,
    project_root: Path,
) -> VerifiedPhase2B3BRuntime:
    """Verify one runtime manifest against independently verified upstream projections."""

    value, payload = _canonical_document(runtime, schema=SCHEMA, label="runtime manifest")
    _validate_runtime_shape(value)
    _reject_leaks(value)
    expected = _expected_runtime(
        result,
        cache_audit,
        result_verification=result_verification,
        input_verification=input_verification,
        project_root=project_root,
    )
    if value != expected:
        raise ValueError("runtime manifest differs from verified upstream projections")
    return VerifiedPhase2B3BRuntime(
        schema=SCHEMA,
        verification_scope="metadata_inventory_only",
        cache_computation_verified=False,
        runtime_sha256=hashlib.sha256(payload).hexdigest(),
        result_sha256=expected["artifacts"]["result_sha256"],
        cache_audit_sha256=expected["artifacts"]["cache_audit_sha256"],
        input_receipt_sha256=expected["inputs"]["input_receipt_sha256"],
        ordered_inputs_sha256=expected["inputs"]["ordered_inputs_sha256"],
        map_evidence_sha256=expected["artifacts"]["map_evidence_sha256"],
        producer_revision=expected["revision"]["producer_revision"],
        model_identity_sha256=hashlib.sha256(
            canonical_json(expected["model_inventory"]["identity"])
        ).hexdigest(),
    )
