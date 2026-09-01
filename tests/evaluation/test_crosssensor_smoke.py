"""Deterministic Phase 2B2-B development smoke evaluation."""

from __future__ import annotations

import hashlib
import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from trustsr.artifacts.predictions import PredictionCache, build_identity, tensor_sha256
from trustsr.contracts import SRPair
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
)
from trustsr.evaluation import crosssensor_smoke
from trustsr.evaluation.crosssensor_smoke import (
    EXPERIMENT_SCHEMA,
    INPUT_AUDIT_SHA256,
    build_cache_provenance,
    cache_entry_evidence,
    evaluate_development_smoke,
    replay_development_smoke,
    snapshot_cache_files,
)
from trustsr.evaluation.opensr_metrics import METRIC_KEYS


def _lr(value: float = 0.25) -> torch.Tensor:
    return torch.full((4, 2, 3), value, dtype=torch.float32)


def _prediction(value: float = 0.5) -> torch.Tensor:
    return torch.full((4, 8, 12), value, dtype=torch.float32)


def _stored_identity(tmp_path: Path):
    provenance = build_cache_provenance({"name": "bicubic-x4", "scale": 4})
    identity = build_identity(provenance, "source", "sample-0", _lr())
    cache = PredictionCache(tmp_path)
    cache.put(identity, _prediction())
    return identity


def test_build_cache_provenance_binds_experiment_and_upstream() -> None:
    provenance = build_cache_provenance(
        {"name": "bicubic-x4", "scale": 4, "implementation_schema_version": 1}
    )

    assert provenance == {
        "name": "bicubic-x4",
        "scale": 4,
        "implementation_schema_version": 1,
        "experiment_schema": "trustsr.phase2b2b-development-smoke.v1",
        "post_manifest_sha256": POST_MANIFEST_SHA256,
        "input_audit_sha256": (
            "fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b"
        ),
    }


def test_context_change_changes_prediction_identity(monkeypatch) -> None:
    model = {"name": "bicubic-x4", "scale": 4}
    first = build_identity(build_cache_provenance(model), "source", "sample-0", _lr())
    monkeypatch.setattr(crosssensor_smoke, "INPUT_AUDIT_SHA256", "e" * 64)
    second = build_identity(build_cache_provenance(model), "source", "sample-0", _lr())

    assert first.key != second.key


def test_cache_provenance_rejects_reserved_context_collision() -> None:
    for key, value in (
        ("experiment_schema", EXPERIMENT_SCHEMA),
        ("post_manifest_sha256", POST_MANIFEST_SHA256),
        ("input_audit_sha256", INPUT_AUDIT_SHA256),
    ):
        try:
            build_cache_provenance({"name": "bicubic-x4", "scale": 4, key: value})
        except ValueError as exc:
            assert "reserved" in str(exc)
        else:
            raise AssertionError(f"reserved key was accepted: {key}")


def test_cache_entry_evidence_hashes_both_named_files(tmp_path: Path) -> None:
    identity = _stored_identity(tmp_path)

    evidence = cache_entry_evidence(tmp_path, identity)

    assert evidence["cache_key"] == identity.key
    assert evidence["lr"] == identity.as_dict()["lr"]
    assert evidence["prediction_sha256"] == hashlib.sha256(
        _prediction().numpy().tobytes()
    ).hexdigest()
    assert [item["filename"] for item in evidence["files"]] == [
        f"{identity.key}.json",
        f"{identity.key}.safetensors",
    ]
    for item in evidence["files"]:
        path = tmp_path / item["filename"]
        assert item["size_bytes"] == path.stat().st_size
        assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_snapshot_cache_files_observes_size_mtime_and_digest_change(tmp_path: Path) -> None:
    identity = _stored_identity(tmp_path)
    before = snapshot_cache_files(tmp_path, [identity])
    tensor_path = tmp_path / f"{identity.key}.safetensors"
    original = tensor_path.read_bytes()
    tensor_path.write_bytes(original + b"damage")
    stat = tensor_path.stat()
    os.utime(tensor_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    after = snapshot_cache_files(tmp_path, [identity])

    assert before != after


def test_named_cache_entry_rejects_extra_suffix(tmp_path: Path) -> None:
    identity = _stored_identity(tmp_path)
    (tmp_path / f"{identity.key}.unexpected").write_bytes(b"extra")

    with pytest.raises(ValueError, match="exactly.*two"):
        snapshot_cache_files(tmp_path, [identity])


def _loaded_pairs() -> tuple[LoadedCrosssensorPair, ...]:
    result = []
    for bin_index in range(4):
        sample_id = f"development-{bin_index}"
        value = (bin_index + 1) / 10
        result.append(
            LoadedCrosssensorPair(
                pair=SRPair(
                    source=f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}",
                    sample_id=sample_id,
                    lr=torch.full((4, 2, 3), value, dtype=torch.float32),
                    hr=torch.full((4, 8, 12), value, dtype=torch.float32),
                    scale=4,
                ),
                metadata=CrosssensorPairMetadata(
                    manifest_sha256=POST_MANIFEST_SHA256,
                    sample_id=sample_id,
                    split="development",
                    spatial_group_id=f"group-{bin_index}",
                    days_between=-1,
                    correlation_bin=bin_index,
                    selection_round=1,
                    lr_asset_sha256=f"{bin_index + 1:x}" * 64,
                    hr_asset_sha256=f"{bin_index + 5:x}" * 64,
                    lr_crop_transform=(10.0, 0.0, 10.0, 0.0, -10.0, -10.0),
                    hr_crop_transform=(2.5, 0.0, 10.0, 0.0, -2.5, -10.0),
                    crop_bounds=(10.0, -30.0, 40.0, -10.0),
                    crop_policy=CROP_POLICY,
                    normalization_policy=NORMALIZATION_POLICY,
                ),
            )
        )
    return tuple(result)


class FakeModel:
    scale = 4

    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value
        self.calls = 0
        self.provenance_calls = 0
        self.last_prediction: torch.Tensor | None = None

    def provenance(self):
        self.provenance_calls += 1
        return {"name": self.name, "scale": self.scale, "value": self.value}

    def predict(self, lr: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.last_prediction = torch.full(
            (4, lr.shape[1] * 4, lr.shape[2] * 4), self.value, dtype=torch.float32
        )
        return self.last_prediction


def _models() -> list[FakeModel]:
    return [
        FakeModel("bicubic-x4", 0.1),
        FakeModel("sen2srlite-x4", 0.2),
        FakeModel("ldsr-s2-x4", 0.3),
    ]


def _metrics(_pair: SRPair, prediction: torch.Tensor) -> dict[str, float]:
    base = float(prediction.mean())
    return {key: base + index for index, key in enumerate(METRIC_KEYS)}


def test_model_grid_cold_run_writes_12_caches_and_warm_run_is_identical(
    tmp_path: Path,
) -> None:
    cold_models = _models()
    cold_result, cold_audit = evaluate_development_smoke(
        _loaded_pairs(), cold_models, tmp_path, metric_fn=_metrics
    )
    warm_models = _models()
    warm_result, warm_audit = evaluate_development_smoke(
        _loaded_pairs(), warm_models, tmp_path, metric_fn=_metrics
    )

    assert [model.calls for model in cold_models] == [4, 4, 4]
    assert [model.calls for model in warm_models] == [0, 0, 0]
    assert [model.provenance_calls for model in cold_models] == [1, 1, 1]
    assert [model.provenance_calls for model in warm_models] == [1, 1, 1]
    assert cold_result == warm_result
    assert cold_audit == warm_audit
    assert cold_result["sample_count"] == 4
    assert cold_result["prediction_count"] == 12
    assert cold_audit["prediction_count"] == 12
    assert len(cold_audit["entries"]) == 12
    assert len(tuple(tmp_path.glob("*.json"))) == 12
    assert len(tuple(tmp_path.glob("*.safetensors"))) == 12
    for model_index, model_record in enumerate(cold_result["models"], start=1):
        assert len(model_record["predictions"]) == 4
        assert len(model_record["mean_metrics"]) == 7
        assert model_record["mean_metrics"]["reflectance"] == pytest.approx(
            model_index / 10
        )


def test_model_grid_metrics_use_reloaded_cache_tensor(tmp_path: Path) -> None:
    models = _models()

    def metrics(pair: SRPair, prediction: torch.Tensor) -> dict[str, float]:
        model = models[int(round(float(prediction.mean()) * 10)) - 1]
        assert model.last_prediction is not None
        assert prediction.data_ptr() != model.last_prediction.data_ptr()
        return _metrics(pair, prediction)

    evaluate_development_smoke(_loaded_pairs(), models, tmp_path, metric_fn=metrics)


def test_model_grid_rejects_cache_value_changed_during_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_get = PredictionCache.get

    def changed_get(cache: PredictionCache, identity):
        prediction = original_get(cache, identity)
        if prediction is None:
            return None
        return (prediction + 0.01).contiguous()

    monkeypatch.setattr(PredictionCache, "get", changed_get)

    with pytest.raises(RuntimeError, match="differs after commit"):
        evaluate_development_smoke(_loaded_pairs(), _models(), tmp_path, metric_fn=_metrics)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("pair-count", "exactly four"),
        ("wrong-split", "development"),
        ("model-order", "model order"),
        ("model-scale", "scale 4"),
        ("nonfinite-metric", "finite"),
    ],
)
def test_model_grid_rejects_contract_damage(
    tmp_path: Path, damage: str, message: str
) -> None:
    pairs = list(_loaded_pairs())
    models = _models()
    metric_fn = _metrics
    if damage == "pair-count":
        pairs.pop()
    elif damage == "wrong-split":
        pairs[0] = LoadedCrosssensorPair(
            pair=pairs[0].pair,
            metadata=CrosssensorPairMetadata(
                **{**pairs[0].metadata.__dict__, "split": "calibration"}
            ),
        )
    elif damage == "model-order":
        models.reverse()
    elif damage == "model-scale":
        models[0].scale = 2
    else:
        def nonfinite_metrics(pair: SRPair, prediction: torch.Tensor) -> dict[str, float]:
            return {**_metrics(pair, prediction), "spectral": float("nan")}

        metric_fn = nonfinite_metrics

    with pytest.raises(ValueError, match=message):
        evaluate_development_smoke(pairs, models, tmp_path, metric_fn=metric_fn)


@pytest.mark.parametrize(
    "damage",
    ["dtype", "shape", "device", "nan", "range", "requires-grad", "noncontiguous"],
)
def test_model_grid_rejects_invalid_generated_prediction(
    tmp_path: Path, damage: str
) -> None:
    models = _models()
    prediction = torch.zeros((4, 8, 12), dtype=torch.float32)
    if damage == "dtype":
        prediction = prediction.to(torch.float64)
    elif damage == "shape":
        prediction = torch.zeros((4, 7, 12), dtype=torch.float32)
    elif damage == "device":
        prediction = torch.zeros((4, 8, 12), dtype=torch.float32, device="meta")
    elif damage == "nan":
        prediction[0, 0, 0] = float("nan")
    elif damage == "range":
        prediction[0, 0, 0] = 2.0
    elif damage == "requires-grad":
        prediction.requires_grad_(True)
    else:
        prediction = torch.zeros((4, 12, 8), dtype=torch.float32).transpose(1, 2)

    def invalid_prediction(_lr: torch.Tensor) -> torch.Tensor:
        return prediction

    models[0].predict = invalid_prediction

    with pytest.raises(ValueError, match="prediction"):
        evaluate_development_smoke(_loaded_pairs(), models, tmp_path, metric_fn=_metrics)


def test_replay_rebuilds_identical_outputs_without_models(tmp_path: Path) -> None:
    result, audit = evaluate_development_smoke(
        _loaded_pairs(), _models(), tmp_path, metric_fn=_metrics
    )

    rebuilt_result, rebuilt_audit = replay_development_smoke(
        _loaded_pairs(), result, audit, tmp_path, metric_fn=_metrics
    )

    assert rebuilt_result == result
    assert rebuilt_audit == audit


@pytest.mark.parametrize(
    "target",
    [
        "result-schema",
        "result-model-order",
        "result-provenance",
        "audit-count",
        "audit-extra-entry",
        "audit-entry-order",
        "audit-cache-key",
        "audit-lr-digest",
        "audit-prediction-digest",
        "result-prediction-digest",
        "cache-bytes",
    ],
)
def test_replay_rejects_changed_evidence(tmp_path: Path, target: str) -> None:
    result, audit = evaluate_development_smoke(
        _loaded_pairs(), _models(), tmp_path, metric_fn=_metrics
    )
    damaged_result = deepcopy(result)
    damaged_audit = deepcopy(audit)
    if target == "result-schema":
        damaged_result["schema"] = "changed"
    elif target == "result-model-order":
        damaged_result["models"].reverse()
    elif target == "result-provenance":
        damaged_result["models"][0]["model_provenance"]["implementation"] = "changed"
    elif target == "audit-count":
        damaged_audit["prediction_count"] = 11
    elif target == "audit-extra-entry":
        damaged_audit["entries"].append(deepcopy(damaged_audit["entries"][0]))
    elif target == "audit-entry-order":
        damaged_audit["entries"].reverse()
    elif target == "audit-cache-key":
        damaged_audit["entries"][0]["cache_key"] = "0" * 64
    elif target == "audit-lr-digest":
        damaged_audit["entries"][0]["lr"]["sha256"] = "0" * 64
    elif target == "audit-prediction-digest":
        damaged_audit["entries"][0]["prediction_sha256"] = "0" * 64
    elif target == "result-prediction-digest":
        damaged_result["models"][0]["predictions"][0]["prediction_sha256"] = "0" * 64
    else:
        key = damaged_audit["entries"][0]["cache_key"]
        path = tmp_path / f"{key}.safetensors"
        path.write_bytes(path.read_bytes() + b"damage")

    with pytest.raises((ValueError, RuntimeError), match="cache|committed|result|audit"):
        replay_development_smoke(
            _loaded_pairs(), damaged_result, damaged_audit, tmp_path, metric_fn=_metrics
        )


def test_replay_detects_cache_mtime_change_during_metrics(tmp_path: Path) -> None:
    result, audit = evaluate_development_smoke(
        _loaded_pairs(), _models(), tmp_path, metric_fn=_metrics
    )
    first_key = audit["entries"][0]["cache_key"]
    first_path = tmp_path / f"{first_key}.json"
    changed = False

    def changing_metrics(pair: SRPair, prediction: torch.Tensor) -> dict[str, float]:
        nonlocal changed
        if not changed:
            stat = first_path.stat()
            os.utime(first_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            changed = True
        return _metrics(pair, prediction)

    with pytest.raises(RuntimeError, match="changed during replay"):
        replay_development_smoke(
            _loaded_pairs(), result, audit, tmp_path, metric_fn=changing_metrics
        )


def test_result_prediction_digest_matches_cached_tensor(tmp_path: Path) -> None:
    result, _ = evaluate_development_smoke(
        _loaded_pairs(), _models(), tmp_path, metric_fn=_metrics
    )

    assert result["models"][0]["predictions"][0]["prediction_sha256"] == tensor_sha256(
        torch.full((4, 8, 12), 0.1, dtype=torch.float32)
    )
