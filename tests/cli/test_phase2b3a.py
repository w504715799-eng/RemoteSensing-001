"""Strict staged orchestration for the Phase 2B3-A score audit."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import pytest
import torch

from trustsr.artifacts.predictions import PredictionCache, build_identity
from trustsr.cli import phase2b3a
from trustsr.data.crosssensor_pairs import POST_MANIFEST_SHA256
from trustsr.evaluation.development_predictions import build_cache_provenance


def test_parser_exposes_only_frozen_stages_and_no_arbitrary_selection() -> None:
    parser = phase2b3a.build_parser()
    subparsers = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]

    assert tuple(subparsers[0].choices) == phase2b3a.STAGES
    help_text = parser.format_help()
    assert "preflight" in help_text
    assert "development-replay" in help_text
    assert "--limit" not in help_text
    assert "--sample-id" not in help_text


def test_phase_directories_are_exactly_digest_addressed(tmp_path: Path) -> None:
    phase = tmp_path / "trustsr" / "phase2b3a"

    assert phase2b3a._phase_root(tmp_path) == phase
    assert phase2b3a._prediction_cache_directory(tmp_path) == (
        phase / "predictions" / POST_MANIFEST_SHA256
    )
    assert phase2b3a._score_cache_directory(tmp_path) == (
        phase / "scores" / POST_MANIFEST_SHA256
    )
    assert phase2b3a._result_directory(tmp_path) == (
        phase / "results" / POST_MANIFEST_SHA256
    )


@pytest.mark.parametrize(
    ("peak", "total", "free", "projected", "expected"),
    [
        (80, 100, 10 * 1024**3, 4_800.0, True),
        (81, 100, 10 * 1024**3, 4_800.0, False),
        (80, 100, 10 * 1024**3 - 1, 4_800.0, False),
        (80, 100, 10 * 1024**3, 4_800.0001, False),
    ],
)
def test_resource_gate_uses_all_three_inclusive_frozen_bounds(
    peak: int, total: int, free: int, projected: float, expected: bool
) -> None:
    assert phase2b3a._resource_gate_pass(
        single_peak_memory_bytes=peak,
        gpu_total_memory_bytes=total,
        persistent_free_bytes=free,
        projected_a2_uncached_seconds=projected,
    ) is expected


def test_projection_uses_median_duration_and_exact_missing_seed_count() -> None:
    projection = phase2b3a._project_a2_uncached_seconds(
        [3.0, 1.0, 2.0, 100.0], missing_seed_predictions=17
    )
    assert projection == 42.5


@pytest.mark.parametrize("value", [-1.0, math.inf, math.nan])
def test_projection_rejects_invalid_uncached_duration(value: float) -> None:
    with pytest.raises(ValueError, match="projection"):
        phase2b3a._project_a2_uncached_seconds(
            [value], missing_seed_predictions=1
        )


def test_missing_seed_count_verifies_existing_cache_bytes_before_counting(
    tmp_path: Path,
) -> None:
    cache = PredictionCache(tmp_path)
    lr = torch.full((4, 2, 2), 0.25, dtype=torch.float32)
    provenance = build_cache_provenance(
        {"name": "ldsr-s2-x4", "scale": 4, "seed": 3407}
    )
    identity = build_identity(provenance, "source", "sample-1", lr)
    cache.put(identity, torch.full((4, 8, 8), 0.5, dtype=torch.float32))
    assert phase2b3a._existing_a2_seed_count(
        tmp_path, ("sample-1",), (3407,)
    ) == 1
    tensor_path = tmp_path / f"{identity.key}.safetensors"
    payload = bytearray(tensor_path.read_bytes())
    payload[-1] ^= 1
    tensor_path.write_bytes(payload)

    with pytest.raises(RuntimeError):
        phase2b3a._existing_a2_seed_count(tmp_path, ("sample-1",), (3407,))


def test_authoritative_development_ids_preserve_selected_manifest_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = tuple({"sample_id": f"sample-{index}"} for index in range(120))
    selected = tuple(records[index] for index in range(119, -1, -1))
    monkeypatch.setattr(phase2b3a, "select_development_records", lambda values: selected)

    assert phase2b3a._ordered_development_sample_ids(records) == tuple(
        f"sample-{index}" for index in range(119, -1, -1)
    )


@pytest.mark.parametrize(("phase", "count"), [("a1", 4), ("a2", 120)])
def test_frozen_pair_loaders_touch_only_the_exact_development_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    count: int,
) -> None:
    selected = tuple(
        {"sample_id": f"development-{index}", "split": "development"}
        for index in range(count)
    )
    context = phase2b3a._StageContext(
        root=tmp_path,
        records=selected,
        project_root=tmp_path,
        code_revision="a" * 40,
        persistent_free_bytes=10 * 1024**3,
        hardware=None,
    )
    loaded: list[str] = []
    monkeypatch.setattr(
        phase2b3a,
        "select_development_smoke_records",
        lambda records: selected,
    )
    monkeypatch.setattr(
        phase2b3a,
        "select_development_records",
        lambda records: selected,
    )

    def load_pair(root, record, *, manifest_sha256):
        assert root == tmp_path
        assert manifest_sha256 == POST_MANIFEST_SHA256
        loaded.append(record["split"])
        return record

    monkeypatch.setattr(phase2b3a, "load_crosssensor_pair", load_pair)
    args = argparse.Namespace(selection_manifest_sha256=POST_MANIFEST_SHA256)

    if phase == "a1":
        phase2b3a._load_a1_pairs(context, args)
    else:
        phase2b3a._load_a2_pairs(context, args)

    assert loaded == ["development"] * count


def test_a2_bundle_generator_is_lazy_one_shot_and_forwards_exact_seed_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, tuple[int, ...]]] = []

    def build(pair, models, seeds, cache):
        events.append((pair, seeds))
        return f"bundle-{pair}"

    monkeypatch.setattr(phase2b3a, "_bundle_for_pair", build)
    bundles = phase2b3a._one_shot_bundles(
        ("roi-1", "roi-2"), (object(),) * 3, (3407,), object()
    )
    assert events == []
    assert next(bundles) == "bundle-roi-1"
    assert events == [("roi-1", (3407,))]
    assert next(bundles) == "bundle-roi-2"
    with pytest.raises(StopIteration):
        next(bundles)


@pytest.mark.parametrize(
    ("stdout", "symbolic_ref", "status", "message"),
    [
        ("a" * 40, "", "", "detached"),
        ("a" * 40, "main", " M pyproject.toml", "clean"),
        ("a" * 40, "main", "", "mismatch"),
    ],
)
def test_reviewed_checkout_rejects_detached_dirty_or_mismatched_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    symbolic_ref: str,
    status: str,
    message: str,
) -> None:
    for relative in ("uv.lock", "pyproject.toml"):
        (tmp_path / relative).write_text("fixture", encoding="utf-8")
    (tmp_path / "src" / "trustsr").mkdir(parents=True)
    values = {
        ("rev-parse", "HEAD"): stdout,
        ("symbolic-ref", "--short", "HEAD"): symbolic_ref,
        ("status", "--porcelain"): status,
    }

    def run(argv, **kwargs):
        command = tuple(argv[3:])
        return subprocess.CompletedProcess(argv, 0, values[command] + "\n", "")

    monkeypatch.setattr(phase2b3a.subprocess, "run", run)
    expected = "b" * 40 if message == "mismatch" else stdout

    with pytest.raises(ValueError, match=message):
        phase2b3a._validate_reviewed_checkout(tmp_path, expected_revision=expected)


def test_execution_checkout_must_be_the_reviewed_project_root(tmp_path: Path) -> None:
    reviewed = tmp_path / "reviewed"
    executing = tmp_path / "other"
    reviewed.mkdir()
    executing.mkdir()

    with pytest.raises(ValueError, match="mismatch"):
        phase2b3a._require_execution_from_reviewed_root(reviewed, executing)


def test_replay_handlers_never_call_a_raising_model_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []

    def raising(_args):
        called.append("model")
        raise AssertionError("replay constructed models")

    monkeypatch.setattr(phase2b3a, "_model_factory", raising)
    monkeypatch.setattr(phase2b3a, "_run_a1_replay", lambda args: {"stage": "replay"})
    monkeypatch.setattr(
        phase2b3a,
        "_run_a2_replay",
        lambda args: {"stage": "development-replay"},
    )

    assert phase2b3a.run_replay(argparse.Namespace()) == {"stage": "replay"}
    assert phase2b3a.run_development_replay(argparse.Namespace()) == {
        "stage": "development-replay"
    }
    assert called == []


def test_preflight_validates_everything_before_constructing_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        phase2b3a,
        "_preflight_context",
        lambda args, *, capture_hardware: events.append("preflight") or object(),
    )
    monkeypatch.setattr(
        phase2b3a,
        "_model_factory",
        lambda args: events.append("models") or (),
    )
    monkeypatch.setattr(
        phase2b3a,
        "_write_preflight_runtime",
        lambda context, models: {"stage": "preflight", "model_count": len(models)},
    )

    assert phase2b3a.run_preflight(argparse.Namespace()) == {
        "stage": "preflight",
        "model_count": 0,
    }
    assert events == ["preflight", "models"]


def test_main_keeps_noise_out_of_canonical_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def noisy(_args):
        print("model noise")
        return {"stage": "smoke", "sample_count": 4}

    class Parser:
        @staticmethod
        def parse_args(_argv):
            return argparse.Namespace(handler=noisy)

    monkeypatch.setattr(phase2b3a, "build_parser", Parser)

    assert phase2b3a.main([]) == 0
    captured = capsys.readouterr()
    assert captured.out == '{"sample_count":4,"stage":"smoke"}\n'
    assert captured.err == "model noise\n"


def test_a2_bundle_manifest_atomically_replaces_completed_a1_manifest(
    tmp_path: Path,
) -> None:
    directory = phase2b3a._result_directory(tmp_path)
    directory.mkdir(parents=True)
    for name in (
        "phase2b3a-a1-result.json",
        "phase2b3a-a1-cache-audit.json",
        "phase2b3a-a1-runtime.json",
        "phase2b3a-a1-replay.json",
        "phase2b3a-a2-result.json",
        "phase2b3a-a2-cache-audit.json",
        "phase2b3a-a2-runtime.json",
        "phase2b3a-a2-replay.json",
    ):
        (directory / name).write_bytes(b"{}")

    phase2b3a._write_bundle_manifest(tmp_path, "a1")
    phase2b3a._write_bundle_manifest(tmp_path, "a2")

    manifest = json.loads(
        (directory / "phase2b3a-bundle-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["phase"] == "a2"
    assert all("-a2-" in item["basename"] for item in manifest["files"])


def test_bundle_manifest_rejects_any_evidence_file_above_five_mib(
    tmp_path: Path,
) -> None:
    directory = phase2b3a._result_directory(tmp_path)
    directory.mkdir(parents=True)
    for name in (
        "phase2b3a-a1-result.json",
        "phase2b3a-a1-cache-audit.json",
        "phase2b3a-a1-runtime.json",
        "phase2b3a-a1-replay.json",
    ):
        (directory / name).write_bytes(b"{}")
    (directory / "phase2b3a-a1-cache-audit.json").write_bytes(
        b"x" * (5 * 1024**2 + 1)
    )

    with pytest.raises(ValueError, match="5 MiB"):
        phase2b3a._write_bundle_manifest(tmp_path, "a1")
