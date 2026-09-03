"""Independent verification of parsed Phase 2B3-B cache audits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import pytest

from trustsr.artifacts.predictions import PredictionIdentity
from trustsr.artifacts.scores import ScoreIdentity
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
)
from trustsr.evaluation import calibration_cache_verify
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

_SOURCE = f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


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
        "seed_first": 3407,
        "seed_last": 3411,
        "seed_count": 5,
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


def _sample(index: int) -> dict[str, object]:
    sample_id = f"calibration-{index:03d}"
    lr_sha256 = _sha(f"lr:{sample_id}")
    predictions: list[dict[str, object]] = []
    prediction_sha256s: list[str] = []
    for seed in SEEDS:
        identity = PredictionIdentity(
            model_provenance=_model_provenance(seed),
            source=_SOURCE,
            sample_id=sample_id,
            lr_shape=(4, 1, 1),
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
    return {
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


@pytest.fixture
def parsed_audit() -> dict[str, object]:
    samples = [_sample(index) for index in range(120)]
    return {
        "schema": "trustsr.phase2b3b-calibration-cache-audit.v1",
        "split": "calibration",
        "ordered_sample_ids_sha256": hashlib.sha256(
            canonical_json([sample["sample_id"] for sample in samples])
        ).hexdigest(),
        "sample_count": 120,
        "prediction_count": 600,
        "score_count": 120,
        "samples": samples,
    }


def test_independently_verifies_audit_and_returns_immutable_host_free_receipt(
    parsed_audit: dict[str, object],
) -> None:
    """Trusting supplied cache keys or returning host data must break this receipt contract."""

    first = calibration_cache_verify.verify_calibration_cache_audit(parsed_audit)
    reparsed = json.loads(canonical_json(parsed_audit))
    second = calibration_cache_verify.verify_calibration_cache_audit(reparsed)

    assert first["schema"] == "trustsr.phase2b3b-calibration-cache-verification.v1"
    assert first["split"] == "calibration"
    assert first["ordered_sample_ids_sha256"] == parsed_audit["ordered_sample_ids_sha256"]
    assert first["counts"] == {
        "samples": 120,
        "predictions": 600,
        "scores": 120,
    }
    assert set(first["digests"]) == {
        "audit_sha256",
        "prediction_identities_sha256",
        "score_identities_sha256",
        "risk_receipts_sha256",
    }
    assert (
        first["digests"]["audit_sha256"] == hashlib.sha256(canonical_json(parsed_audit)).hexdigest()
    )
    assert first == second
    assert all(len(value) == 64 for value in first["digests"].values())
    encoded = repr(first)
    for forbidden in ("path", "time", "host", "alpha", "coverage", "internal_test"):
        assert forbidden not in encoded

    with pytest.raises(TypeError):
        first["schema"] = "forged"  # type: ignore[index]
    with pytest.raises(TypeError):
        first["digests"]["audit_sha256"] = "0" * 64  # type: ignore[index]


def test_rejects_one_sample_with_another_legal_runtime_identity(
    parsed_audit: dict[str, object],
) -> None:
    sample = _assert_mapping(_assert_list(parsed_audit["samples"])[1])
    predictions = _assert_list(sample["predictions"])
    for prediction in predictions:
        entry = _assert_mapping(prediction)
        original = _assert_mapping(entry["identity"])
        provenance = dict(_assert_mapping(original["model_provenance"]))
        provenance["torch_version"] = "2.12.1+cu130"
        provenance["cuda_runtime"] = "13.0"
        lr = _assert_mapping(original["lr"])
        identity = PredictionIdentity(
            model_provenance=provenance,
            source=original["source"],
            sample_id=original["sample_id"],
            lr_shape=tuple(_assert_list(lr["shape"])),
            lr_dtype=lr["dtype"],
            lr_sha256=lr["sha256"],
        )
        entry["identity"] = identity.as_dict()
        entry["cache_key"] = identity.key

    with pytest.raises(ValueError, match="model scientific identit"):
        calibration_cache_verify.verify_calibration_cache_audit(parsed_audit)


def _assert_mapping(value: object) -> Mapping[str, object]:
    assert isinstance(value, Mapping)
    return value


def _assert_list(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


@pytest.mark.parametrize(
    "fault",
    (
        "top_extra",
        "top_missing",
        "sample_extra",
        "sample_missing",
        "prediction_extra",
        "prediction_missing",
        "prediction_identity_extra",
        "prediction_lr_missing",
        "score_extra",
        "score_missing",
        "score_identity_extra",
        "risk_extra",
        "risk_missing",
        "sample_count",
        "prediction_count",
        "score_count",
        "audit_split",
        "ordered_sample_digest",
        "seed",
        "seed_order",
        "model",
        "checkpoint",
        "config",
        "implementation",
        "runtime",
        "context",
        "source",
        "sample",
        "prediction_cache_key",
        "prediction_digest",
        "score_name",
        "score_input_binding",
        "score_cache_key",
        "score_digest",
        "risk_name",
        "risk_window",
        "risk_digest",
        "duplicate_sample",
    ),
)
def test_rejects_hostile_schema_identity_and_digest_mutations(
    parsed_audit: dict[str, object], fault: str
) -> None:
    """Each mutation represents one trust decision the verifier must make independently."""

    samples = _assert_list(parsed_audit["samples"])
    sample = _assert_mapping(samples[0])
    predictions = _assert_list(sample["predictions"])
    prediction = _assert_mapping(predictions[0])
    prediction_identity = _assert_mapping(prediction["identity"])
    prediction_lr = _assert_mapping(prediction_identity["lr"])
    score = _assert_mapping(sample["score"])
    score_identity = _assert_mapping(score["identity"])
    risk = _assert_mapping(sample["risk"])

    if fault == "top_extra":
        parsed_audit["extra"] = True
    elif fault == "top_missing":
        parsed_audit.pop("schema")
    elif fault == "sample_extra":
        sample["extra"] = True
    elif fault == "sample_missing":
        sample.pop("risk")
    elif fault == "prediction_extra":
        prediction["extra"] = True
    elif fault == "prediction_missing":
        prediction.pop("identity")
    elif fault == "prediction_identity_extra":
        prediction_identity["extra"] = True
    elif fault == "prediction_lr_missing":
        prediction_lr.pop("dtype")
    elif fault == "score_extra":
        score["extra"] = True
    elif fault == "score_missing":
        score.pop("identity")
    elif fault == "score_identity_extra":
        score_identity["extra"] = True
    elif fault == "risk_extra":
        risk["extra"] = True
    elif fault == "risk_missing":
        risk.pop("window")
    elif fault in {"sample_count", "prediction_count", "score_count"}:
        parsed_audit[fault] = True
    elif fault == "audit_split":
        parsed_audit["split"] = "internal_test"
    elif fault == "ordered_sample_digest":
        parsed_audit["ordered_sample_ids_sha256"] = "0" * 64
    elif fault == "seed":
        prediction["seed"] = 9999
    elif fault == "seed_order":
        predictions[0], predictions[1] = predictions[1], predictions[0]
    elif fault == "model":
        prediction["model_name"] = "other-model"
    elif fault == "checkpoint":
        _assert_mapping(prediction_identity["model_provenance"])["checkpoint_sha256"] = "0" * 64
    elif fault == "config":
        _assert_mapping(prediction_identity["model_provenance"])["config_sha256"] = "0" * 64
    elif fault == "implementation":
        _assert_mapping(prediction_identity["model_provenance"])[
            "implementation_schema_version"
        ] = 2
    elif fault == "runtime":
        _assert_mapping(prediction_identity["model_provenance"])["cuda_runtime"] = "forged"
    elif fault == "context":
        _assert_mapping(prediction_identity["model_provenance"])["post_manifest_sha256"] = "0" * 64
    elif fault == "source":
        prediction_identity["source"] = "wrong-source"
    elif fault == "sample":
        prediction_identity["sample_id"] = "other-sample"
    elif fault == "prediction_cache_key":
        prediction["cache_key"] = "0" * 64
    elif fault == "prediction_digest":
        prediction["prediction_sha256"] = "not-a-digest"
    elif fault == "score_name":
        score["name"] = "three_model_disagreement"
    elif fault == "score_input_binding":
        _assert_list(score_identity["input_sha256s"])[0] = "f" * 64
    elif fault == "score_cache_key":
        score["cache_key"] = "0" * 64
    elif fault == "score_digest":
        score["score_sha256"] = "not-a-digest"
    elif fault == "risk_name":
        risk["name"] = "other-risk"
    elif fault == "risk_window":
        risk["window"] = 1
    elif fault == "risk_digest":
        risk["risk_sha256"] = "not-a-digest"
    else:
        samples[-1] = json.loads(json.dumps(samples[0]))

    with pytest.raises(ValueError):
        calibration_cache_verify.verify_calibration_cache_audit(parsed_audit)
