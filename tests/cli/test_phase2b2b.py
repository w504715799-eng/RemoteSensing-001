"""Staged Phase 2B2-B development smoke orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from trustsr.cli import phase2b2b
from trustsr.contracts import SRPair
from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
    CrosssensorPairMetadata,
    LoadedCrosssensorPair,
)
from trustsr.evaluation.crosssensor_smoke import INPUT_AUDIT_SHA256
from trustsr.jsonio import canonical_json


def _arguments(tmp_path: Path, stage: str) -> list[str]:
    return [
        stage,
        "--storage-root",
        str(tmp_path),
        "--selection-manifest",
        str(tmp_path / "samples.jsonl"),
        "--selection-manifest-sha256",
        POST_MANIFEST_SHA256,
        "--input-audit",
        str(tmp_path / "phase2b2a-input-audit.json"),
        "--input-audit-sha256",
        INPUT_AUDIT_SHA256,
        "--sen2srlite-model-dir",
        str(tmp_path / "models" / "sen2srlite"),
        "--ldsr-model-dir",
        str(tmp_path / "models" / "ldsr"),
        "--confirm-cloud-storage",
    ]


def _args(tmp_path: Path, stage: str = "smoke") -> argparse.Namespace:
    return phase2b2b.build_parser().parse_args(_arguments(tmp_path, stage))


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "sample_id": f"sample-{split}-{bin_index}",
            "split": split,
            "spatial_group_id": f"group-{split}-{bin_index}",
            "days_between": -1,
            "correlation_bin": bin_index,
            "selection_round": 1,
        }
        for split in ("calibration", "development", "internal_test")
        for bin_index in range(4)
    )


def _loaded(record: dict[str, object]) -> LoadedCrosssensorPair:
    sample_id = str(record["sample_id"])
    bin_index = int(record["correlation_bin"])
    return LoadedCrosssensorPair(
        pair=SRPair(
            source=f"sen2naipv2-crosssensor/{POST_MANIFEST_SHA256}",
            sample_id=sample_id,
            lr=torch.full((4, 2, 3), 0.25, dtype=torch.float32),
            hr=torch.full((4, 8, 12), 0.5, dtype=torch.float32),
            scale=4,
        ),
        metadata=CrosssensorPairMetadata(
            manifest_sha256=POST_MANIFEST_SHA256,
            sample_id=sample_id,
            split="development",
            spatial_group_id=str(record["spatial_group_id"]),
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


class FakeModel:
    scale = 4

    def __init__(self, name: str):
        self.name = name

    def provenance(self):
        return {"name": self.name, "scale": 4}

    def predict(self, lr: torch.Tensor) -> torch.Tensor:
        return torch.zeros((4, lr.shape[1] * 4, lr.shape[2] * 4), dtype=torch.float32)


def _models() -> list[FakeModel]:
    return [FakeModel(name) for name in phase2b2b.MODEL_NAMES]


def test_parser_has_only_staged_commands_and_no_scientific_overrides(tmp_path: Path) -> None:
    parser = phase2b2b.build_parser()
    subparser_actions = [
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ]

    assert set(subparser_actions[0].choices) == {"preflight", "single", "smoke", "replay"}
    for forbidden in ("--split", "--sample-id", "--limit", "--seed", "--metric"):
        with pytest.raises(SystemExit):
            parser.parse_args([*_arguments(tmp_path, "smoke"), forbidden, "x"])


def test_load_development_pairs_filters_before_loading_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records()
    loaded_ids: list[str] = []
    monkeypatch.setattr(phase2b2b, "_validate_upstream", lambda args: (tmp_path, records))

    def load_pair(root: Path, record: dict[str, object], *, manifest_sha256: str):
        assert root == tmp_path
        assert manifest_sha256 == POST_MANIFEST_SHA256
        loaded_ids.append(str(record["sample_id"]))
        return _loaded(record)

    monkeypatch.setattr(phase2b2b, "load_crosssensor_pair", load_pair)

    pairs = phase2b2b._load_development_pairs(_args(tmp_path))

    assert [pair.metadata.correlation_bin for pair in pairs] == [0, 1, 2, 3]
    assert loaded_ids == [f"sample-development-{index}" for index in range(4)]


def test_invalid_upstream_stops_before_phase_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject(_args):
        raise ValueError("input audit SHA-256 does not match")

    monkeypatch.setattr(phase2b2b, "_validate_upstream", reject)

    with pytest.raises(ValueError, match="input audit"):
        phase2b2b.run_smoke(_args(tmp_path))

    assert not (tmp_path / "trustsr" / "phase2b2b").exists()


def test_validate_upstream_rejects_alternate_symlink_to_frozen_input_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = (
        tmp_path
        / "trustsr/phase2b2a/input-audits"
        / POST_MANIFEST_SHA256
        / "phase2b2a-input-audit.json"
    )
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"audit")
    alternate = tmp_path / "alternate-input-audit.json"
    alternate.symlink_to(expected)
    args = _args(tmp_path)
    args.input_audit = alternate
    monkeypatch.setattr(
        phase2b2b,
        "require_cloud_confirmation",
        lambda root, confirmed: root,
    )
    monkeypatch.setattr(
        phase2b2b,
        "load_crosssensor_records",
        lambda root, path, *, expected_sha256: _records(),
    )
    monkeypatch.setattr(
        phase2b2b,
        "load_input_audit",
        lambda path, *, expected_sha256: pytest.fail("symlink reached audit loader"),
    )

    with pytest.raises(ValueError, match="digest-addressed path"):
        phase2b2b._validate_upstream(args)


def test_compute_stage_checks_gpu_before_loading_pair_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def validate(_args):
        events.append("upstream")
        return tmp_path, _records()

    def reject_gpu():
        events.append("gpu")
        raise RuntimeError("GPU gate")

    def load_pair(*args, **kwargs):
        events.append("pixels")
        return _loaded(args[1])

    monkeypatch.setattr(phase2b2b, "_validate_upstream", validate)
    monkeypatch.setattr(phase2b2b, "capture_gpu_hardware", reject_gpu)
    monkeypatch.setattr(phase2b2b, "load_crosssensor_pair", load_pair)

    with pytest.raises(RuntimeError, match="GPU gate"):
        phase2b2b.run_smoke(_args(tmp_path))

    assert events == ["upstream", "gpu"]


def test_compute_stage_rejects_symlinked_phase_output_before_gpu_or_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "trustsr").mkdir()
    (tmp_path / "trustsr" / "phase2b2b").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(
        phase2b2b,
        "_validate_upstream",
        lambda args: (tmp_path, _records()),
    )
    monkeypatch.setattr(
        phase2b2b,
        "capture_gpu_hardware",
        lambda: pytest.fail("symlinked path reached GPU gate"),
    )
    monkeypatch.setattr(
        phase2b2b,
        "load_crosssensor_pair",
        lambda *args, **kwargs: pytest.fail("symlinked path loaded pixels"),
    )

    with pytest.raises(ValueError, match="must not contain symlinks"):
        phase2b2b.run_smoke(_args(tmp_path))

    assert tuple(outside.iterdir()) == ()


def test_preflight_constructs_models_without_loading_pair_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = _models()
    monkeypatch.setattr(phase2b2b, "_validate_upstream", lambda args: (tmp_path, _records()))
    monkeypatch.setattr(phase2b2b, "_model_factory", lambda args: models)
    monkeypatch.setattr(
        phase2b2b,
        "_load_development_pairs",
        lambda args, limit=None: pytest.fail("preflight loaded pair pixels"),
    )
    monkeypatch.setattr(phase2b2b, "capture_gpu_hardware", lambda: object())
    monkeypatch.setattr(
        phase2b2b,
        "collect_gpu_environment",
        lambda **kwargs: {"schema_version": 1, "runtime": {}},
    )

    result = phase2b2b.run_preflight(_args(tmp_path, "preflight"))

    assert result["stage"] == "preflight"
    runtime = json.loads(
        (
            tmp_path
            / "trustsr/phase2b2b/results"
            / POST_MANIFEST_SHA256
            / "preflight-runtime.json"
        ).read_text()
    )
    assert [item["name"] for item in runtime["models"]] == list(phase2b2b.MODEL_NAMES)


@pytest.mark.parametrize(
    ("handler_name", "sample_count", "result_name", "runtime_name"),
    [
        ("run_single", 1, "single-result.json", "single-runtime.json"),
        ("run_smoke", 4, "development-three-model-smoke.json", "smoke-runtime.json"),
    ],
)
def test_compute_stages_commit_deterministic_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    handler_name: str,
    sample_count: int,
    result_name: str,
    runtime_name: str,
) -> None:
    pairs = tuple(_loaded(record) for record in _records() if record["split"] == "development")
    monkeypatch.setattr(phase2b2b, "_validate_upstream", lambda args: (tmp_path, _records()))
    monkeypatch.setattr(
        phase2b2b,
        "_load_development_pairs",
        lambda args, limit=None: pairs[:limit] if limit is not None else pairs,
    )
    monkeypatch.setattr(phase2b2b, "_model_factory", lambda args: _models())
    monkeypatch.setattr(phase2b2b, "capture_gpu_hardware", lambda: object())
    calls: list[int] = []

    def evaluate(loaded_pairs, models, cache_root, *, expected_sample_count=4):
        calls.append(expected_sample_count)
        return (
            {"schema": "result", "sample_count": len(loaded_pairs)},
            {"schema": "audit", "prediction_count": len(loaded_pairs) * len(models)},
        )

    monkeypatch.setattr(phase2b2b, "evaluate_development_smoke", evaluate)
    monkeypatch.setattr(phase2b2b, "_runtime_measurement", lambda: {"peak_memory_bytes": 7})

    result = getattr(phase2b2b, handler_name)(_args(tmp_path, handler_name.removeprefix("run_")))

    assert calls == [sample_count]
    result_path = (
        tmp_path / "trustsr/phase2b2b/results" / POST_MANIFEST_SHA256 / result_name
    )
    assert result_path.read_bytes() == canonical_json(
        {"schema": "result", "sample_count": sample_count}
    )
    assert result["sample_count"] == sample_count
    runtime = json.loads(result_path.with_name(runtime_name).read_text())
    assert runtime["duration_seconds"] >= 0.0
    assert runtime["peak_memory_bytes"] == 7


def test_replay_reads_committed_bytes_without_model_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = {"schema": "result", "sample_count": 4}
    audit = {"schema": "audit", "prediction_count": 12}
    result_dir = tmp_path / "trustsr/phase2b2b/results" / POST_MANIFEST_SHA256
    result_dir.mkdir(parents=True)
    (result_dir / "development-three-model-smoke.json").write_bytes(canonical_json(result))
    (result_dir / "development-cache-audit.json").write_bytes(canonical_json(audit))
    pairs = tuple(_loaded(record) for record in _records() if record["split"] == "development")
    monkeypatch.setattr(phase2b2b, "_validate_upstream", lambda args: (tmp_path, _records()))
    monkeypatch.setattr(
        phase2b2b, "_load_development_pairs", lambda args, limit=None: pairs
    )
    monkeypatch.setattr(
        phase2b2b,
        "_model_factory",
        lambda args: pytest.fail("replay constructed models"),
        raising=False,
    )
    monkeypatch.setattr(
        phase2b2b,
        "replay_development_smoke",
        lambda loaded_pairs, committed_result, committed_audit, cache_root: (
            dict(committed_result),
            dict(committed_audit),
        ),
    )

    replayed = phase2b2b.run_replay(_args(tmp_path, "replay"))

    assert replayed == {"stage": "replay", "prediction_count": 12, "byte_identical": True}
