"""Pure contracts for the final Phase 2B3-B calibration result composer."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace

import pytest

from trustsr.artifacts.predictions import PredictionIdentity
from trustsr.artifacts.scores import ScoreIdentity
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
)
from trustsr.evaluation import phase2b3b_result
from trustsr.evaluation.calibration_fit import (
    FREEZE_CALIBRATION,
    STOP_INSUFFICIENT_COVERAGE,
    CalibrationFit,
)
from trustsr.evaluation.calibration_predictions import (
    A2_RESULT_SHA256,
    MODEL_NAME,
    PUBLICATION_COMMIT,
    SEEDS,
    build_cache_provenance,
)
from trustsr.evaluation.phase2b3b_evidence import (
    INPUT_AUDIT_SHA256,
    PRODUCER_REVISION,
    PUBLISHED_EVIDENCE_SHA256S,
)
from trustsr.evaluation.phase2b3b_revision import Phase2B3BRevision
from trustsr.jsonio import canonical_json
from trustsr.models.ldsr_assets import (
    CHECKPOINT_NAME,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    CHECKPOINT_URL,
    CONFIG_SHA256,
)
from trustsr.models.versions import OPENSR_MODEL_VERSION

_SOURCE = f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _sample_ids(prefix: str = "calibration") -> tuple[str, ...]:
    return tuple(f"{prefix}-{index:03d}" for index in range(120))


def _ordered_sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json(list(sample_ids))).hexdigest()


def _membership(sample_ids: Sequence[str]) -> list[dict[str, object]]:
    return [
        {
            "sample_id": sample_id,
            "selection_sha256": _sha(f"selection:{sample_id}"),
            "spatial_group_id": f"group:{sample_id}",
            "lr_asset_sha256": _sha(f"lr-asset:{sample_id}"),
            "hr_asset_sha256": _sha(f"hr-asset:{sample_id}"),
            "days_between": (-1, 0, 1)[index // 40],
            "correlation_bin": (index % 40) // 10,
            "selection_round": index % 10 + 1,
        }
        for index, sample_id in enumerate(sample_ids)
    ]


def _preflight(sample_ids: Sequence[str] | None = None) -> dict[str, object]:
    ordered_ids = _sample_ids() if sample_ids is None else tuple(sample_ids)
    membership = _membership(ordered_ids)
    return {
        "schema": "trustsr.phase2b3b-preflight.v1",
        "upstream": {
            "publication_commit": PUBLICATION_COMMIT,
            "producer_revision": PRODUCER_REVISION,
            "post_manifest_sha256": POST_MANIFEST_SHA256,
            "input_audit_sha256": INPUT_AUDIT_SHA256,
            "evidence_sha256s": dict(PUBLISHED_EVIDENCE_SHA256S),
        },
        "calibration": {
            "split": "calibration",
            "sample_count": 120,
            "ordered_sample_ids_sha256": _ordered_sample_ids_sha256(ordered_ids),
            "ordered_membership_sha256": hashlib.sha256(canonical_json(membership)).hexdigest(),
            "input_receipt_sha256s": [
                hashlib.sha256(canonical_json(record)).hexdigest() for record in membership
            ],
            "strata": [
                {
                    "days_between": day,
                    "correlation_bin": bin_index,
                    "sample_count": 10,
                }
                for day in (-1, 0, 1)
                for bin_index in range(4)
            ],
        },
        "score": {
            "name": "ldsr_variance_k5",
            "operator_parameters": {
                "algorithm": "ensemble_variance_score",
                "band_reduction": "mean",
                "correction": 0,
                "seed_count": 5,
                "seed_first": 3407,
                "seed_last": 3411,
            },
            "seeds": list(SEEDS),
        },
        "risk": {"name": "local_l1_risk", "window": 9, "upper_bound": 1.0},
        "input": {
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "crop_policy": CROP_POLICY,
            "bands": ["B04", "B03", "B02", "B08"],
            "scale": 4,
        },
    }


def _refresh_input_receipt(receipt: dict[str, object]) -> None:
    samples = receipt["samples"]
    membership = [sample["membership"] for sample in samples]
    sample_ids = [record["sample_id"] for record in membership]
    receipt["ordered_sample_ids_sha256"] = _ordered_sample_ids_sha256(sample_ids)
    receipt["ordered_membership_sha256"] = hashlib.sha256(
        canonical_json(membership)
    ).hexdigest()
    receipt["input_receipt_sha256s"] = [
        hashlib.sha256(canonical_json(record)).hexdigest() for record in membership
    ]
    receipt["ordered_inputs_sha256"] = hashlib.sha256(canonical_json(samples)).hexdigest()


def _input_receipt(sample_ids: Sequence[str]) -> dict[str, object]:
    membership = _membership(sample_ids)
    samples = [
        {
            "membership": record,
            "lr": {
                "asset_sha256": record["lr_asset_sha256"],
                "tensor_sha256": _sha(f"lr:{record['sample_id']}"),
                "shape": [4, 1, 1],
                "dtype": "torch.float32",
            },
            "hr": {
                "asset_sha256": record["hr_asset_sha256"],
                "tensor_sha256": _sha(f"hr:{record['sample_id']}"),
                "shape": [4, 4, 4],
                "dtype": "torch.float32",
            },
        }
        for record in membership
    ]
    receipt: dict[str, object] = {
        "schema": "trustsr.phase2b3b-calibration-input-receipt.v1",
        "split": "calibration",
        "sample_count": 120,
        "input": {
            "manifest_sha256": POST_MANIFEST_SHA256,
            "source": _SOURCE,
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "crop_policy": CROP_POLICY,
            "bands": ["B04", "B03", "B02", "B08"],
            "scale": 4,
        },
        "ordered_sample_ids_sha256": "",
        "ordered_membership_sha256": "",
        "input_receipt_sha256s": [],
        "samples": samples,
        "ordered_inputs_sha256": "",
    }
    _refresh_input_receipt(receipt)
    return receipt


def _model_provenance(seed: int) -> dict[str, str | int]:
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


def _score_parameters(lr_sha256: str) -> dict[str, str | int]:
    return {
        "algorithm": "ensemble_variance_score",
        "band_reduction": "mean",
        "correction": 0,
        "seed_first": SEEDS[0],
        "seed_last": SEEDS[-1],
        "seed_count": len(SEEDS),
        "lr_sha256": lr_sha256,
        "source": _SOURCE,
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": INPUT_AUDIT_SHA256,
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "crop_policy": CROP_POLICY,
        "phase2b3a_publication_commit": PUBLICATION_COMMIT,
        "phase2b3a_a2_result_sha256": A2_RESULT_SHA256,
        "phase2b3a_producer_revision": PRODUCER_REVISION,
    }


def _audit(
    sample_ids: Sequence[str],
    *,
    lr_digest_prefix: str = "lr",
    lr_shape: tuple[int, int, int] = (4, 1, 1),
) -> dict[str, object]:
    samples: list[dict[str, object]] = []
    for sample_id in sample_ids:
        lr_sha256 = _sha(f"{lr_digest_prefix}:{sample_id}")
        predictions: list[dict[str, object]] = []
        prediction_sha256s: list[str] = []
        for seed in SEEDS:
            identity = PredictionIdentity(
                model_provenance=_model_provenance(seed),
                source=_SOURCE,
                sample_id=sample_id,
                lr_shape=lr_shape,
                lr_dtype="torch.float32",
                lr_sha256=lr_sha256,
            )
            prediction_sha256 = _sha(f"prediction:{sample_id}:{seed}")
            prediction_sha256s.append(prediction_sha256)
            predictions.append(
                {
                    "model_name": MODEL_NAME,
                    "seed": seed,
                    "cache_key": identity.key,
                    "identity": identity.as_dict(),
                    "prediction_sha256": prediction_sha256,
                }
            )
        score_identity = ScoreIdentity(
            score_name="ldsr_variance_k5",
            score_schema_version=1,
            sample_id=sample_id,
            input_sha256s=tuple(prediction_sha256s),
            operator_parameters=_score_parameters(lr_sha256),
        )
        samples.append(
            {
                "sample_id": sample_id,
                "predictions": predictions,
                "score": {
                    "name": "ldsr_variance_k5",
                    "cache_key": score_identity.key,
                    "identity": score_identity.as_dict(),
                    "score_sha256": _sha(f"score:{sample_id}"),
                },
                "risk": {
                    "name": "local_l1_risk",
                    "window": 9,
                    "risk_sha256": _sha(f"risk:{sample_id}"),
                },
            }
        )
    return {
        "schema": "trustsr.phase2b3b-calibration-cache-audit.v1",
        "split": "calibration",
        "ordered_sample_ids_sha256": _ordered_sample_ids_sha256(sample_ids),
        "sample_count": 120,
        "prediction_count": 600,
        "score_count": 120,
        "samples": samples,
    }


def _saturation() -> dict[str, object]:
    return {
        "raw_crop_minimum": 100,
        "raw_crop_maximum": 9000,
        "clipped_high_count": 0,
        "clipped_high_by_band": [0, 0, 0, 0],
    }


def _radiometry(sample_ids: Sequence[str]) -> dict[str, object]:
    samples = [
        {
            "sample_id": sample_id,
            "days_between": (-1, 0, 1)[index // 40],
            "correlation_bin": (index % 40) // 10,
            "selection_round": index % 10 + 1,
            "radiometric_saturation": {"lr": _saturation(), "hr": _saturation()},
        }
        for index, sample_id in enumerate(sample_ids)
    ]
    return {
        "schema": "trustsr.phase2b3b-calibration-radiometry.v1",
        "split": "calibration",
        "ordered_sample_ids_sha256": _ordered_sample_ids_sha256(sample_ids),
        "policy": {
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "raw_radiometric_max": 32767,
            "saturation_threshold": 10000,
            "saturation_operation": "minimum(raw,10000)",
            "saturation_scope": "aligned_crop_only",
            "reflectance_divisor": 10000.0,
            "crop_policy": CROP_POLICY,
            "bands": ["B04", "B03", "B02", "B08"],
        },
        "sample_count": 120,
        "affected_sample_count": 0,
        "lr": _saturation(),
        "hr": _saturation(),
        "samples": samples,
    }


def _fit(sample_ids: tuple[str, ...], *, all_abstain: bool = False) -> CalibrationFit:
    map_evidence_sha256 = hashlib.sha256(
        canonical_json(
            [
                {
                    "sample_id": sample_id,
                    "score_sha256": _sha(f"score:{sample_id}"),
                    "risk_sha256": _sha(f"risk:{sample_id}"),
                }
                for sample_id in sample_ids
            ]
        )
    ).hexdigest()
    return CalibrationFit(
        alpha=0.02,
        minimum_coverage=0.10,
        threshold=None if all_abstain else 0.1,
        all_abstain=all_abstain,
        risk_bound=1 / 121,
        risk_upper_bound=1.0,
        calibration_size=120,
        trusted_pixels=0 if all_abstain else 120,
        total_pixels=120,
        coverage=0.0 if all_abstain else 1.0,
        phase_decision=(STOP_INSUFFICIENT_COVERAGE if all_abstain else FREEZE_CALIBRATION),
        sample_ids=sample_ids,
        map_evidence_sha256=map_evidence_sha256,
    )


def _revision() -> Phase2B3BRevision:
    return Phase2B3BRevision(
        branch="main",
        head_revision="c" * 40,
        calculation_revision=PRODUCER_REVISION,
        evidence_publication=PUBLICATION_COMMIT,
    )


def test_composes_minimal_canonical_result_with_cross_layer_sample_binding() -> None:
    sample_ids = _sample_ids()
    inputs = (
        _preflight(),
        _input_receipt(sample_ids),
        _fit(sample_ids),
        _audit(sample_ids),
        _radiometry(sample_ids),
    )

    first = phase2b3b_result.build_phase2b3b_result(*inputs, _revision())
    second = phase2b3b_result.build_phase2b3b_result(*inputs, _revision())

    assert set(first) == {
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
    assert first["schema"] == "trustsr.phase2b3b-calibration.v1"
    assert first["split"] == "calibration"
    assert first["producer_revision"] == "c" * 40
    assert first["input_receipt_sha256"] == hashlib.sha256(
        canonical_json(inputs[1])
    ).hexdigest()
    assert first["ordered_inputs_sha256"] == inputs[1]["ordered_inputs_sha256"]
    assert first["cache_audit_sha256"] == hashlib.sha256(canonical_json(inputs[3])).hexdigest()
    assert first["map_evidence_sha256"] == inputs[2].map_evidence_sha256
    assert first["upstream"]["phase2b3a_publication_commit"] == PUBLICATION_COMMIT
    assert first["upstream"]["ordered_sample_ids_sha256"] == (
        _ordered_sample_ids_sha256(sample_ids)
    )
    assert (
        first["upstream"]["ordered_membership_sha256"]
        == _preflight()["calibration"]["ordered_membership_sha256"]
    )
    assert first["frozen"]["risk"] == {
        "name": "local_l1_risk",
        "window": 9,
        "upper_bound": 1.0,
    }
    assert first["target"] == {"alpha": 0.02, "minimum_coverage": 0.10}
    assert first["threshold"] == 0.1
    assert first["phase_decision"] == "freeze_calibration"
    assert first["counts"] == {
        "calibration": 120,
        "predictions": 600,
        "scores": 120,
        "trusted_pixels": 120,
        "total_pixels": 120,
    }
    assert len(first["samples"]) == 120
    assert first["samples"][0]["sample_id"] == "calibration-000"
    assert set(first["samples"][0]) == {
        "sample_id",
        "split",
        "predictions",
        "score",
        "risk",
        "radiometric_saturation",
        "input",
    }
    assert first["samples"][0]["split"] == "calibration"
    assert set(first["samples"][0]["predictions"][0]) == {
        "seed",
        "cache_key",
        "prediction_sha256",
    }
    assert first["samples"][0]["input"] == {
        "membership_sha256": inputs[1]["input_receipt_sha256s"][0],
        "lr": {
            "asset_sha256": _sha("lr-asset:calibration-000"),
            "tensor_sha256": _sha("lr:calibration-000"),
            "shape": [4, 1, 1],
            "dtype": "torch.float32",
        },
        "hr": {
            "asset_sha256": _sha("hr-asset:calibration-000"),
            "tensor_sha256": _sha("hr:calibration-000"),
            "shape": [4, 4, 4],
            "dtype": "torch.float32",
        },
    }
    assert "identity" not in canonical_json(first).decode()
    assert canonical_json(first) == canonical_json(second)
    assert first is not second
    first["samples"][0]["radiometric_saturation"]["lr"]["clipped_high_by_band"][0] = 99
    assert second["samples"][0]["radiometric_saturation"]["lr"]["clipped_high_by_band"][0] == 0


def test_all_abstain_result_keeps_null_threshold_and_single_stop_decision() -> None:
    sample_ids = _sample_ids()

    result = phase2b3b_result.build_phase2b3b_result(
        _preflight(),
        _input_receipt(sample_ids),
        _fit(sample_ids, all_abstain=True),
        _audit(sample_ids),
        _radiometry(sample_ids),
        _revision(),
    )

    assert result["threshold"] is None
    assert result["all_abstain"] is True
    assert result["coverage"] == 0.0
    assert result["phase_decision"] == "stop_insufficient_coverage"
    assert canonical_json(result).decode().count("phase_decision") == 1


def test_accepts_cache_audit_builder_tuple_arrays_before_independent_verification() -> None:
    sample_ids = _sample_ids()
    audit = _audit(sample_ids)
    for sample in audit["samples"]:
        sample["predictions"] = tuple(sample["predictions"])
    audit["samples"] = tuple(audit["samples"])

    result = phase2b3b_result.build_phase2b3b_result(
        _preflight(),
        _input_receipt(sample_ids),
        _fit(sample_ids),
        audit,
        _radiometry(sample_ids),
        _revision(),
    )

    assert result["counts"]["predictions"] == 600


@pytest.mark.parametrize(
    "fault",
    (
        "extra",
        "schema",
        "upstream",
        "count",
        "stratum",
        "score",
        "risk",
        "input",
        "membership",
        "preflight_split",
        "ordered_ids",
        "receipt_count",
        "receipt_digest",
    ),
)
def test_rejects_forged_preflight_schema_and_frozen_identity(fault: str) -> None:
    sample_ids = _sample_ids()
    preflight = _preflight()
    if fault == "extra":
        preflight["extra"] = True
    elif fault == "schema":
        preflight["schema"] = "wrong"
    elif fault == "upstream":
        preflight["upstream"]["publication_commit"] = "0" * 40
    elif fault == "count":
        preflight["calibration"]["sample_count"] = 119
    elif fault == "stratum":
        preflight["calibration"]["strata"][0]["sample_count"] = 9
    elif fault == "score":
        preflight["score"]["name"] = "three_model_disagreement"
    elif fault == "risk":
        preflight["risk"]["window"] = 1
    elif fault == "membership":
        preflight["calibration"]["ordered_membership_sha256"] = "invalid"
    elif fault == "preflight_split":
        preflight["calibration"]["split"] = "internal_test"
    elif fault == "ordered_ids":
        preflight["calibration"]["ordered_sample_ids_sha256"] = "0" * 64
    elif fault == "receipt_count":
        preflight["calibration"]["input_receipt_sha256s"].pop()
    elif fault == "receipt_digest":
        preflight["calibration"]["input_receipt_sha256s"][0] = "invalid"
    else:
        preflight["input"]["scale"] = 2

    with pytest.raises(ValueError):
        phase2b3b_result.build_phase2b3b_result(
            preflight,
            _input_receipt(sample_ids),
            _fit(sample_ids),
            _audit(sample_ids),
            _radiometry(sample_ids),
            _revision(),
        )


@pytest.mark.parametrize(
    "fault", ("schema", "policy", "aggregate", "bool_aggregate", "sample", "extra")
)
def test_rejects_forged_radiometry_and_cross_layer_order(fault: str) -> None:
    sample_ids = _sample_ids()
    radiometry = _radiometry(sample_ids)
    if fault == "schema":
        radiometry["schema"] = "wrong"
    elif fault == "policy":
        radiometry["policy"]["saturation_threshold"] = 9999
    elif fault == "aggregate":
        radiometry["lr"]["raw_crop_minimum"] = 101
    elif fault == "bool_aggregate":
        radiometry["lr"]["clipped_high_count"] = False
    elif fault == "sample":
        radiometry = _radiometry(tuple(reversed(sample_ids)))
    else:
        radiometry["extra"] = True

    with pytest.raises(ValueError):
        phase2b3b_result.build_phase2b3b_result(
            _preflight(),
            _input_receipt(sample_ids),
            _fit(sample_ids),
            _audit(sample_ids),
            radiometry,
            _revision(),
        )


@pytest.mark.parametrize("fault", ("calculation", "publication", "head", "head_type"))
def test_rejects_forged_revision_identity(fault: str) -> None:
    sample_ids = _sample_ids()
    changes: dict[str, object]
    if fault == "calculation":
        changes = {"calculation_revision": "0" * 40}
    elif fault == "publication":
        changes = {"evidence_publication": "0" * 40}
    elif fault == "head":
        changes = {"head_revision": "C" * 40}
    else:
        changes = {"head_revision": None}

    with pytest.raises(ValueError):
        phase2b3b_result.build_phase2b3b_result(
            _preflight(),
            _input_receipt(sample_ids),
            _fit(sample_ids),
            _audit(sample_ids),
            _radiometry(sample_ids),
            replace(_revision(), **changes),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("layer", ("input", "fit", "audit", "radiometry"))
def test_rejects_any_ordered_sample_mismatch_across_layers(layer: str) -> None:
    sample_ids = _sample_ids()
    alternate = _sample_ids("other")
    fit = _fit(alternate if layer == "fit" else sample_ids)
    audit = _audit(alternate if layer == "audit" else sample_ids)
    radiometry = _radiometry(alternate if layer == "radiometry" else sample_ids)
    input_receipt = _input_receipt(alternate if layer == "input" else sample_ids)

    with pytest.raises(ValueError, match="ordered samples"):
        phase2b3b_result.build_phase2b3b_result(
            _preflight(), input_receipt, fit, audit, radiometry, _revision()
        )


def test_rejects_fully_self_consistent_arbitrary_membership() -> None:
    """A self-consistent internal-test-shaped stack is not frozen calibration membership."""

    arbitrary_ids = _sample_ids("internal_test")

    with pytest.raises(ValueError, match="authoritative|membership"):
        phase2b3b_result.build_phase2b3b_result(
            _preflight(),
            _input_receipt(arbitrary_ids),
            _fit(arbitrary_ids),
            _audit(arbitrary_ids),
            _radiometry(arbitrary_ids),
            _revision(),
        )


def test_runs_independent_input_and_cache_verifiers_before_fit_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_ids = _sample_ids()
    fit = _fit(sample_ids)
    object.__setattr__(fit, "coverage", 0.5)
    audit = _audit(sample_ids)
    audit["extra"] = True
    events: list[str] = []
    real_input_verify = phase2b3b_result.verify_calibration_input_receipt
    real_cache_verify = phase2b3b_result.verify_calibration_cache_audit

    def verify_input(value: object) -> object:
        events.append("input")
        return real_input_verify(value)

    def verify_cache(value: object) -> object:
        events.append("cache")
        return real_cache_verify(value)

    monkeypatch.setattr(
        phase2b3b_result, "verify_calibration_input_receipt", verify_input
    )
    monkeypatch.setattr(
        phase2b3b_result, "verify_calibration_cache_audit", verify_cache
    )

    with pytest.raises(ValueError):
        phase2b3b_result.build_phase2b3b_result(
            _preflight(),
            _input_receipt(sample_ids),
            fit,
            audit,
            _radiometry(sample_ids),
            _revision(),
        )
    assert events == ["input", "cache"]


def test_rejects_forged_fit_after_a_valid_independent_cache_verification() -> None:
    sample_ids = _sample_ids()
    fit = _fit(sample_ids)
    object.__setattr__(fit, "coverage", 0.5)

    with pytest.raises(ValueError, match="fit public contract"):
        phase2b3b_result.build_phase2b3b_result(
            _preflight(),
            _input_receipt(sample_ids),
            fit,
            _audit(sample_ids),
            _radiometry(sample_ids),
            _revision(),
        )


@pytest.mark.parametrize("map_kind", ("score", "risk"))
def test_rejects_audit_map_digest_change_with_identical_sample_ids(
    map_kind: str,
) -> None:
    sample_ids = _sample_ids()
    audit = _audit(sample_ids)
    digest_key = f"{map_kind}_sha256"
    audit["samples"][0][map_kind][digest_key] = "f" * 64

    with pytest.raises(ValueError, match="map evidence"):
        phase2b3b_result.build_phase2b3b_result(
            _preflight(),
            _input_receipt(sample_ids),
            _fit(sample_ids),
            audit,
            _radiometry(sample_ids),
            _revision(),
        )


def test_rejects_forged_fit_map_evidence_digest() -> None:
    sample_ids = _sample_ids()
    fit = replace(_fit(sample_ids), map_evidence_sha256="f" * 64)

    with pytest.raises(ValueError, match="map evidence"):
        phase2b3b_result.build_phase2b3b_result(
            _preflight(),
            _input_receipt(sample_ids),
            fit,
            _audit(sample_ids),
            _radiometry(sample_ids),
            _revision(),
        )


@pytest.mark.parametrize("fault", ("extra", "aggregate", "dtype"))
def test_rejects_internally_invalid_input_receipt(fault: str) -> None:
    sample_ids = _sample_ids()
    receipt = _input_receipt(sample_ids)
    if fault == "extra":
        receipt["extra"] = True
    elif fault == "aggregate":
        receipt["ordered_inputs_sha256"] = "f" * 64
    else:
        receipt["samples"][0]["lr"]["dtype"] = "torch.float64"

    with pytest.raises(ValueError):
        phase2b3b_result.build_phase2b3b_result(
            _preflight(),
            receipt,
            _fit(sample_ids),
            _audit(sample_ids),
            _radiometry(sample_ids),
            _revision(),
        )


def test_rejects_internally_valid_receipt_per_record_membership_attack() -> None:
    sample_ids = _sample_ids()
    receipt = _input_receipt(sample_ids)
    receipt["samples"][0]["membership"]["selection_sha256"] = _sha("replacement")
    _refresh_input_receipt(receipt)

    with pytest.raises(ValueError, match="authoritative calibration membership"):
        phase2b3b_result.build_phase2b3b_result(
            _preflight(),
            receipt,
            _fit(sample_ids),
            _audit(sample_ids),
            _radiometry(sample_ids),
            _revision(),
        )


def test_rejects_internally_valid_receipt_strata_attack() -> None:
    sample_ids = _sample_ids()
    receipt = _input_receipt(sample_ids)
    first = receipt["samples"][0]["membership"]
    second = receipt["samples"][40]["membership"]
    for field in ("days_between", "correlation_bin", "selection_round"):
        first[field], second[field] = second[field], first[field]
    _refresh_input_receipt(receipt)
    preflight = _preflight()
    preflight["calibration"]["ordered_membership_sha256"] = receipt[
        "ordered_membership_sha256"
    ]
    preflight["calibration"]["input_receipt_sha256s"] = receipt[
        "input_receipt_sha256s"
    ]

    with pytest.raises(ValueError, match="radiometry strata"):
        phase2b3b_result.build_phase2b3b_result(
            preflight,
            receipt,
            _fit(sample_ids),
            _audit(sample_ids),
            _radiometry(sample_ids),
            _revision(),
        )


@pytest.mark.parametrize("fault", ("sha256", "shape"))
def test_rejects_internally_valid_audit_lr_replacement(fault: str) -> None:
    sample_ids = _sample_ids()
    kwargs = (
        {"lr_digest_prefix": "replacement-lr"}
        if fault == "sha256"
        else {"lr_shape": (4, 2, 2)}
    )

    with pytest.raises(ValueError, match="LR identity"):
        phase2b3b_result.build_phase2b3b_result(
            _preflight(),
            _input_receipt(sample_ids),
            _fit(sample_ids),
            _audit(sample_ids, **kwargs),
            _radiometry(sample_ids),
            _revision(),
        )


def test_result_leaks_no_runtime_tensor_path_host_or_unapproved_default() -> None:
    sample_ids = _sample_ids()
    result = phase2b3b_result.build_phase2b3b_result(
        _preflight(),
        _input_receipt(sample_ids),
        _fit(sample_ids),
        _audit(sample_ids),
        _radiometry(sample_ids),
        _revision(),
    )
    encoded = canonical_json(result).decode()

    for forbidden in ("path", "timestamp", "hostname", "branch", "internal_test"):
        assert forbidden not in encoded
