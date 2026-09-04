"""Host-free Phase 2B3-C runtime inventory contracts."""

from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from trustsr.evaluation.internal_test_predictions import build_cache_provenance
from trustsr.evaluation.phase2b3c_result import build_phase2b3c_result
from trustsr.evaluation.phase2b3c_runtime import (
    build_phase2b3c_runtime_manifest,
    verify_phase2b3c_runtime_manifest,
)
from trustsr.evaluation.phase2b3c_statistics import (
    ROIEvaluation,
    build_phase2b3c_statistics,
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


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _result() -> dict[str, object]:
    rois = tuple(
        ROIEvaluation(
            f"test-{index:03d}",
            (-1, 0, 1)[(index % 12) // 4],
            index % 4,
            index // 12 + 1,
            20,
            100,
            0.2,
            0.0,
        )
        for index in range(120)
    )
    asset = {
        "raw_crop_minimum": 0,
        "raw_crop_maximum": 10000,
        "clipped_high_count": 0,
        "clipped_high_by_band": [0, 0, 0, 0],
    }
    return build_phase2b3c_result(
        statistics=build_phase2b3c_statistics(rois),
        radiometry={"lr": asset, "hr": asset},
        evaluation_id=_sha("evaluation"),
        permit_sha256=_sha("permit"),
        ledger_event_sha256=_sha("ledger"),
        input_receipt_sha256=_sha("receipt"),
        ordered_inputs_sha256=_sha("inputs"),
        ordered_sample_ids_sha256=_sha("ids"),
        ordered_membership_sha256=_sha("membership"),
        cache_audit_sha256=_sha("audit"),
        map_evidence_sha256=_sha("maps"),
        producer_revision="1" * 40,
    )


def _model_provenance() -> dict[str, object]:
    return build_cache_provenance(
        {
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
            "seed": 3407,
            "sampling_steps": 100,
            "sampling_eta": 0.95,
            "sampling_temperature": 1.0,
            "histogram_matching": True,
            "output_policy": "clip_to_[0,1]",
        }
    )


def _dependencies() -> dict[str, object]:
    return {
        "python": "3.12",
        "uv_lock_sha256": _sha("uv.lock"),
        "packages": {
            "numpy": "2.2.6",
            "opensr-model": "0.3.1",
            "rasterio": "1.4.3",
            "torch": "2.7.1+cu128",
            "trustsr": "0.1.0",
        },
    }


def test_builds_and_verifies_host_free_non_authorizing_runtime() -> None:
    result = _result()
    runtime = build_phase2b3c_runtime_manifest(
        result=result,
        model_provenance=_model_provenance(),
        dependencies=_dependencies(),
    )

    verified = verify_phase2b3c_runtime_manifest(runtime, result=result)

    assert runtime["schema"] == "trustsr.phase2b3c-evaluation-runtime.v1"
    assert runtime["verification_scope"] == "metadata_inventory_only"
    assert runtime["cache_computation_verified"] is False
    assert verified.result_sha256 == hashlib.sha256(canonical_json(result)).hexdigest()
    assert verified.runtime_sha256 == hashlib.sha256(canonical_json(runtime)).hexdigest()
    assert runtime["model_inventory"]["seeds"] == [3407, 3408, 3409, 3410, 3411]
    payload = canonical_json(runtime).decode().casefold()
    for marker in ("/tmp", "hostname", "endpoint", "timestamp", "credential"):
        assert marker not in payload
    for forbidden in ("replay_sha256", "bundle_sha256", "acceptance_sha256"):
        assert forbidden not in payload


def test_runtime_rejects_host_metadata_and_model_identity_mutation() -> None:
    result = _result()
    with pytest.raises(ValueError, match="dependencies|host"):
        build_phase2b3c_runtime_manifest(
            result=result,
            model_provenance=_model_provenance(),
            dependencies={**_dependencies(), "project_path": "/tmp/repo"},
        )

    runtime = build_phase2b3c_runtime_manifest(
        result=result,
        model_provenance=_model_provenance(),
        dependencies=_dependencies(),
    )
    changed = deepcopy(runtime)
    changed["model_inventory"]["sampling_steps"] = 99
    with pytest.raises(ValueError, match="model"):
        verify_phase2b3c_runtime_manifest(changed, result=result)


def test_runtime_cross_binds_result_and_rejects_noncanonical_bytes() -> None:
    result = _result()
    runtime = build_phase2b3c_runtime_manifest(
        result=result,
        model_provenance=_model_provenance(),
        dependencies=_dependencies(),
    )
    changed_result = deepcopy(result)
    changed_result["radiometry"]["lr"]["raw_crop_maximum"] = 9999
    with pytest.raises(ValueError, match="result"):
        verify_phase2b3c_runtime_manifest(runtime, result=changed_result)
    with pytest.raises(ValueError, match="canonical"):
        verify_phase2b3c_runtime_manifest(
            canonical_json(runtime) + b"\n", result=result
        )
