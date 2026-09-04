"""Independent downstream computation replay for Phase 2B3-C."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from trustsr.artifacts.predictions import build_identity, tensor_sha256
from trustsr.artifacts.scores import ScoreCache
from trustsr.contracts import SRPair
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
    RadiometricSaturation,
)
from trustsr.evaluation.internal_test_cache_audit import build_internal_test_cache_audit
from trustsr.evaluation.internal_test_input_receipt import build_internal_test_input_receipt
from trustsr.evaluation.internal_test_maps import load_or_compute_internal_test_maps
from trustsr.evaluation.internal_test_metrics import build_internal_test_metrics
from trustsr.evaluation.internal_test_predictions import (
    MODEL_NAME,
    SEEDS,
    CachedInternalTestPrediction,
    InternalTestPredictionBundle,
    build_cache_provenance,
)
from trustsr.evaluation.phase2b3c_computation_verify import (
    verify_phase2b3c_computation,
)
from trustsr.evaluation.phase2b3c_policy import PHASE2B3C_THRESHOLD
from trustsr.evaluation.phase2b3c_result import build_phase2b3c_result
from trustsr.evaluation.phase2b3c_runtime import build_phase2b3c_runtime_manifest
from trustsr.evaluation.phase2b3c_statistics import build_phase2b3c_statistics
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


def _record(index: int) -> dict[str, object]:
    return {
        "sample_id": f"test-{index:03d}",
        "selection_sha256": _sha(f"selection:{index}"),
        "spatial_group_id": _sha(f"group:{index}"),
        "split": "internal_test",
        "days_between": (-1, 0, 1)[(index % 12) // 4],
        "correlation_bin": index % 4,
        "selection_round": index // 12 + 1,
        "lr_asset": {"sha256": _sha(f"lr asset:{index}")},
        "hr_asset": {"sha256": _sha(f"hr asset:{index}")},
    }


def _pair(record: dict[str, object]) -> LoadedCrosssensorPair:
    sample_id = str(record["sample_id"])
    return LoadedCrosssensorPair(
        pair=SRPair(
            sample_id=sample_id,
            source=f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}",
            lr=torch.full((4, 3, 3), 0.25, dtype=torch.float32),
            hr=torch.full((4, 12, 12), 0.5, dtype=torch.float32),
            scale=4,
        ),
        metadata=CrosssensorPairMetadata(
            manifest_sha256=POST_MANIFEST_SHA256,
            sample_id=sample_id,
            split="internal_test",
            spatial_group_id=str(record["spatial_group_id"]),
            days_between=int(record["days_between"]),
            correlation_bin=int(record["correlation_bin"]),
            selection_round=int(record["selection_round"]),
            lr_asset_sha256=str(record["lr_asset"]["sha256"]),
            hr_asset_sha256=str(record["hr_asset"]["sha256"]),
            lr_crop_transform=(10.0, 0.0, 10.0, 0.0, -10.0, -10.0),
            hr_crop_transform=(2.5, 0.0, 10.0, 0.0, -2.5, -10.0),
            crop_bounds=(10.0, -30.0, 40.0, -10.0),
            crop_policy=CROP_POLICY,
            normalization_policy=PHASE2B3A_NORMALIZATION_POLICY,
            lr_saturation=RadiometricSaturation(2500, 2500, 0, (0, 0, 0, 0)),
            hr_saturation=RadiometricSaturation(5000, 5000, 0, (0, 0, 0, 0)),
        ),
    )


def _provenance(seed: int) -> dict[str, object]:
    return build_cache_provenance(
        {
            "name": MODEL_NAME,
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
    )


def _bundle(pair: LoadedCrosssensorPair) -> InternalTestPredictionBundle:
    values = (0.499, 0.5, 0.501, 0.5, 0.5)
    items = []
    for seed, value in zip(SEEDS, values, strict=True):
        tensor = torch.full((4, 12, 12), value, dtype=torch.float32)
        identity = build_identity(
            _provenance(seed),
            pair.pair.source,
            pair.pair.sample_id,
            pair.pair.lr,
        )
        items.append(
            CachedInternalTestPrediction(
                MODEL_NAME, seed, identity, tensor_sha256(tensor), tensor
            )
        )
    return InternalTestPredictionBundle(pair.pair.sample_id, tuple(items))


def _dependencies() -> dict[str, object]:
    return {
        "python": "3.12",
        "uv_lock_sha256": _sha("lock"),
        "packages": {
            "numpy": "2.2.6",
            "opensr-model": "0.3.1",
            "rasterio": "1.4.3",
            "torch": "2.7.1+cu128",
            "trustsr": "0.1.0",
        },
    }


def _candidate(tmp_path: Path) -> dict[str, object]:
    records = tuple(_record(index) for index in range(120))
    pairs = tuple(_pair(record) for record in records)
    bundles = tuple(_bundle(pair) for pair in pairs)
    maps = tuple(
        load_or_compute_internal_test_maps(pair, bundle, ScoreCache(tmp_path))
        for pair, bundle in zip(pairs, bundles, strict=True)
    )
    metrics = build_internal_test_metrics(
        pairs, maps, threshold=PHASE2B3C_THRESHOLD
    )
    audit = build_internal_test_cache_audit(bundles, maps)
    input_receipt = build_internal_test_input_receipt(records, pairs)
    verified_input_sha256 = hashlib.sha256(canonical_json(input_receipt)).hexdigest()
    result = build_phase2b3c_result(
        statistics=build_phase2b3c_statistics(metrics.rois),
        radiometry={
            "lr": {
                "raw_crop_minimum": 2500,
                "raw_crop_maximum": 2500,
                "clipped_high_count": 0,
                "clipped_high_by_band": [0, 0, 0, 0],
            },
            "hr": {
                "raw_crop_minimum": 5000,
                "raw_crop_maximum": 5000,
                "clipped_high_count": 0,
                "clipped_high_by_band": [0, 0, 0, 0],
            },
        },
        evaluation_id=_sha("evaluation"),
        permit_sha256=_sha("permit"),
        ledger_event_sha256=_sha("ledger"),
        input_receipt_sha256=verified_input_sha256,
        ordered_inputs_sha256=input_receipt["ordered_inputs_sha256"],
        ordered_sample_ids_sha256=input_receipt["ordered_sample_ids_sha256"],
        ordered_membership_sha256=input_receipt["ordered_membership_sha256"],
        cache_audit_sha256=hashlib.sha256(canonical_json(audit)).hexdigest(),
        map_evidence_sha256=metrics.map_evidence_sha256,
        producer_revision="1" * 40,
    )
    runtime = build_phase2b3c_runtime_manifest(
        result=result,
        model_provenance=bundles[0].items[0].identity.model_provenance,
        dependencies=_dependencies(),
    )
    return {
        "pairs": pairs,
        "bundles": bundles,
        "input_receipt": input_receipt,
        "dependencies": _dependencies(),
        "result": canonical_json(result),
        "audit": canonical_json(audit),
        "runtime": canonical_json(runtime),
    }


def test_independently_replays_all_downstream_computation_without_inference(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path)

    verified = verify_phase2b3c_computation(
        candidate["result"],
        candidate["audit"],
        candidate["runtime"],
        input_receipt=candidate["input_receipt"],
        pairs=candidate["pairs"],
        bundles=candidate["bundles"],
        dependencies=candidate["dependencies"],
    )

    assert verified.cache_computation_verified is True
    assert verified.prediction_inference_verified is False
    assert verified.membership_authority_verified is False
    assert verified.acceptance_authorized is False
    assert verified.phase_decision == "confirmed"


def test_verifier_does_not_import_producer_computation_helpers() -> None:
    import trustsr.evaluation.phase2b3c_computation_verify as module

    for forbidden in (
        "build_internal_test_cache_audit",
        "load_or_compute_internal_test_maps",
        "build_internal_test_metrics",
        "build_phase2b3c_statistics",
        "build_phase2b3c_result",
        "build_phase2b3c_runtime_manifest",
        "ensemble_variance_score",
        "local_l1_risk",
    ):
        assert forbidden not in module.__dict__


def test_rejects_result_audit_runtime_and_cache_tensor_mutations(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    changed_result = deepcopy(__import__("json").loads(candidate["result"]))
    changed_result["radiometry"]["lr"]["raw_crop_minimum"] = 2499
    with pytest.raises(ValueError, match="result|radiometry"):
        verify_phase2b3c_computation(
            canonical_json(changed_result),
            candidate["audit"],
            candidate["runtime"],
            input_receipt=candidate["input_receipt"],
            pairs=candidate["pairs"],
            bundles=candidate["bundles"],
            dependencies=candidate["dependencies"],
        )

    changed_audit = deepcopy(__import__("json").loads(candidate["audit"]))
    changed_audit["samples"][0]["risk"]["risk_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="audit"):
        verify_phase2b3c_computation(
            candidate["result"],
            canonical_json(changed_audit),
            candidate["runtime"],
            input_receipt=candidate["input_receipt"],
            pairs=candidate["pairs"],
            bundles=candidate["bundles"],
            dependencies=candidate["dependencies"],
        )

    changed_runtime = deepcopy(__import__("json").loads(candidate["runtime"]))
    changed_runtime["dependencies"]["packages"]["torch"] = "2.8.0+cu129"
    with pytest.raises(ValueError, match="runtime"):
        verify_phase2b3c_computation(
            candidate["result"],
            candidate["audit"],
            canonical_json(changed_runtime),
            input_receipt=candidate["input_receipt"],
            pairs=candidate["pairs"],
            bundles=candidate["bundles"],
            dependencies=candidate["dependencies"],
        )

    bundles = list(candidate["bundles"])
    first = bundles[0]
    tensor = torch.full((4, 12, 12), 0.498, dtype=torch.float32)
    changed_item = replace(
        first.items[0], tensor=tensor, prediction_sha256=tensor_sha256(tensor)
    )
    bundles[0] = InternalTestPredictionBundle(
        first.sample_id, (changed_item, *first.items[1:])
    )
    with pytest.raises(ValueError, match="audit|result|score"):
        verify_phase2b3c_computation(
            candidate["result"],
            candidate["audit"],
            candidate["runtime"],
            input_receipt=candidate["input_receipt"],
            pairs=candidate["pairs"],
            bundles=tuple(bundles),
            dependencies=candidate["dependencies"],
        )
