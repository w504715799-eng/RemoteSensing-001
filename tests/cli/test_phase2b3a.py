"""Strict staged orchestration for the Phase 2B3-A score audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from trustsr.artifacts.predictions import PredictionCache, build_identity
from trustsr.cli import phase2b3a
from trustsr.data.crosssensor_pairs import (
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
)
from trustsr.evaluation.development_predictions import build_cache_provenance


def test_parser_exposes_only_frozen_stages_and_no_arbitrary_selection() -> None:
    parser = phase2b3a.build_parser()
    subparsers = [
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ]

    assert tuple(subparsers[0].choices) == phase2b3a.STAGES
    help_text = parser.format_help()
    assert "preflight" in help_text
    assert "development-replay" in help_text
    assert "--limit" not in help_text
    assert "--sample-id" not in help_text

    common = [
        "preflight",
        "--storage-root",
        ".",
        "--selection-manifest",
        "selection.json",
        "--selection-manifest-sha256",
        "a" * 64,
        "--input-audit",
        "audit.json",
        "--input-audit-sha256",
        "b" * 64,
        "--sen2srlite-model-dir",
        "sen2",
        "--ldsr-model-dir",
        "ldsr",
        "--project-root",
        ".",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(common)
    parsed = parser.parse_args(
        [
            *common,
            "--reviewed-commit",
            "c" * 40,
        ]
    )
    assert parsed.reviewed_commit == "c" * 40


@pytest.mark.parametrize("revision", ["A" * 40, "a" * 39, "g" * 40])
def test_parser_rejects_noncanonical_reviewed_commit(revision: str) -> None:
    parser = phase2b3a.build_parser()
    argv = [
        "preflight",
        "--storage-root",
        ".",
        "--selection-manifest",
        "selection.json",
        "--selection-manifest-sha256",
        "a" * 64,
        "--input-audit",
        "audit.json",
        "--input-audit-sha256",
        "b" * 64,
        "--sen2srlite-model-dir",
        "sen2",
        "--ldsr-model-dir",
        "ldsr",
        "--project-root",
        ".",
        "--reviewed-commit",
        revision,
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


@pytest.mark.parametrize("stage", ["replay", "development-replay"])
def test_replay_parser_rejects_model_arguments(stage: str) -> None:
    parser = phase2b3a.build_parser()
    common = [
        stage,
        "--storage-root",
        ".",
        "--selection-manifest",
        "selection.json",
        "--selection-manifest-sha256",
        "a" * 64,
        "--input-audit",
        "audit.json",
        "--input-audit-sha256",
        "b" * 64,
        "--project-root",
        ".",
        "--reviewed-commit",
        "c" * 40,
    ]
    assert not hasattr(parser.parse_args(common), "sen2srlite_model_dir")
    with pytest.raises(SystemExit):
        parser.parse_args([*common, "--sen2srlite-model-dir", "model"])


def test_phase_directories_are_exactly_digest_addressed(tmp_path: Path) -> None:
    phase = tmp_path / "trustsr" / "phase2b3a"

    assert phase2b3a._phase_root(tmp_path) == phase
    assert phase2b3a._prediction_cache_directory(tmp_path) == (
        phase / "predictions" / POST_MANIFEST_SHA256
    )
    assert phase2b3a._score_cache_directory(tmp_path) == (phase / "scores" / POST_MANIFEST_SHA256)
    assert phase2b3a._result_directory(tmp_path) == (phase / "results" / POST_MANIFEST_SHA256)


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
    assert (
        phase2b3a._resource_gate_pass(
            single_peak_memory_bytes=peak,
            gpu_total_memory_bytes=total,
            persistent_free_bytes=free,
            projected_a2_uncached_seconds=projected,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("peak", "total", "free", "projected"),
    [
        (-1, 100, 10 * 1024**3, 1.0),
        (1, -1, 10 * 1024**3, 1.0),
        (1, 100, -1, 1.0),
        (1, 100, 10 * 1024**3, -1.0),
        (1, 100, 10 * 1024**3, math.nan),
        (1, 100, 10 * 1024**3, math.inf),
    ],
)
def test_resource_gate_rejects_invalid_measurements(
    peak: int, total: int, free: int, projected: float
) -> None:
    with pytest.raises(ValueError, match="resource"):
        phase2b3a._resource_gate_pass(
            single_peak_memory_bytes=peak,
            gpu_total_memory_bytes=total,
            persistent_free_bytes=free,
            projected_a2_uncached_seconds=projected,
        )


def _write_valid_a1_acceptance(
    root: Path, producer_commit: str, *, runtime_damage: str | None = None
) -> str:
    radiometric_policy = {
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "raw_radiometric_max": 32767,
        "saturation_threshold": 10000,
        "bands": ["B04", "B03", "B02", "B08"],
        "sample_count": 4,
        "affected_sample_count": 0,
        "affected_asset_count": 0,
        "lr_clipped_high_count": 0,
        "hr_clipped_high_count": 0,
        "raw_crop_maximum": 5500,
    }
    runtime = {
        "schema": "trustsr.phase2b3a-a1-runtime.v2",
        "git_commit": producer_commit,
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "radiometric_policy": radiometric_policy,
        "single_repeatability_pass": True,
        "single_peak_memory_bytes": 80,
        "gpu_total_memory_bytes": 100,
        "persistent_free_bytes": 10 * 1024**3,
        "a1_uncached_ldsr_prediction_seconds": [1.0],
        "a1_median_uncached_ldsr_prediction_seconds": 1.0,
        "missing_a2_seed_predictions": 1,
        "projected_a2_uncached_seconds": 1.0,
        "resource_gate_pass": True,
    }
    if runtime_damage == "normalization-policy":
        runtime["normalization_policy"] = "uint16_divide_10000_no_clip_v1"
    elif runtime_damage == "radiometric-policy":
        runtime["radiometric_policy"] = {
            **radiometric_policy,
            "affected_asset_count": 1,
        }
    elif runtime_damage == "boolean-radiometric-policy":
        runtime["radiometric_policy"] = {
            **radiometric_policy,
            "affected_sample_count": False,
        }
    runtime_bytes, runtime_sha256 = phase2b3a._write_runtime(
        root, phase2b3a._A1_RUNTIME, runtime
    )
    result = {
        "schema": "trustsr.phase2b3a-development-smoke.v2",
        "sample_count": 4,
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "radiometric_policy": radiometric_policy,
        "include_ldsr_variance_k5": True,
        "k5_statistically_stable": True,
        "runtime_manifest_sha256": runtime_sha256,
    }
    audit = {
        "schema": "trustsr.phase2b3a-development-smoke-cache-audit.v2",
        "sample_count": 4,
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
    }
    result_bytes, audit_bytes = phase2b3a._commit_pair(
        root,
        (phase2b3a._A1_RESULT, result),
        (phase2b3a._A1_AUDIT, audit),
    )
    replay = phase2b3a._receipt(
        "a1", result_bytes, audit_bytes, hashlib.sha256(runtime_bytes).hexdigest()
    )
    directory = phase2b3a._result_directory(root)
    phase2b3a._commit_identical_or_new(
        directory / phase2b3a._A1_REPLAY,
        phase2b3a.canonical_json(replay),
        root,
    )
    return hashlib.sha256(phase2b3a.canonical_json(replay)).hexdigest()


def test_a1_acceptance_allows_verified_ancestor_producer_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    producer_commit = "a" * 40
    current_commit = "b" * 40
    replay_sha256 = _write_valid_a1_acceptance(tmp_path, producer_commit)
    calls: list[list[str]] = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(phase2b3a.subprocess, "run", run)
    context = phase2b3a._StageContext(
        root=tmp_path,
        records=(),
        project_root=tmp_path,
        code_revision=current_commit,
        persistent_free_bytes=10 * 1024**3,
        hardware=None,
    )

    assert phase2b3a._verify_a1_cloud_acceptance(context) == (
        True,
        replay_sha256,
        producer_commit,
    )
    assert calls == [
        [
            "git",
            "-C",
            str(tmp_path),
            "merge-base",
            "--is-ancestor",
            producer_commit,
            current_commit,
        ]
    ]


def test_a1_acceptance_rejects_nonancestor_producer_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    producer_commit = "a" * 40
    current_commit = "b" * 40
    _write_valid_a1_acceptance(tmp_path, producer_commit)
    monkeypatch.setattr(
        phase2b3a.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, "", ""),
    )
    context = phase2b3a._StageContext(
        root=tmp_path,
        records=(),
        project_root=tmp_path,
        code_revision=current_commit,
        persistent_free_bytes=10 * 1024**3,
        hardware=None,
    )

    with pytest.raises(ValueError, match="not an ancestor"):
        phase2b3a._verify_a1_cloud_acceptance(context)


@pytest.mark.parametrize(
    "damage",
    ["normalization-policy", "radiometric-policy", "boolean-radiometric-policy"],
)
def test_a1_acceptance_rejects_runtime_policy_that_differs_from_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    _write_valid_a1_acceptance(tmp_path, "a" * 40, runtime_damage=damage)
    monkeypatch.setattr(
        phase2b3a.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, "", ""),
    )
    context = phase2b3a._StageContext(
        root=tmp_path,
        records=(),
        project_root=tmp_path,
        code_revision="b" * 40,
        persistent_free_bytes=10 * 1024**3,
        hardware=None,
    )

    with pytest.raises(ValueError, match="radiometric|normalization|policy"):
        phase2b3a._verify_a1_cloud_acceptance(context)


def test_git_ancestor_rejects_noncanonical_producer_without_running_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        phase2b3a.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("git was run")),
    )

    with pytest.raises(ValueError, match="canonical producer"):
        phase2b3a._require_git_ancestor(tmp_path, "A" * 40, "b" * 40)


def test_git_ancestor_inspection_error_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        phase2b3a.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 128, "", "bad git"),
    )

    with pytest.raises(ValueError, match="Git"):
        phase2b3a._require_git_ancestor(tmp_path, "a" * 40, "b" * 40)


def test_projection_uses_median_duration_and_exact_missing_seed_count() -> None:
    projection = phase2b3a._project_a2_uncached_seconds(
        [3.0, 1.0, 2.0, 100.0], missing_seed_predictions=17
    )
    assert projection == 42.5


@pytest.mark.parametrize("value", [-1.0, math.inf, math.nan])
def test_projection_rejects_invalid_uncached_duration(value: float) -> None:
    with pytest.raises(ValueError, match="projection"):
        phase2b3a._project_a2_uncached_seconds([value], missing_seed_predictions=1)


def test_missing_seed_count_verifies_existing_cache_bytes_before_counting(
    tmp_path: Path,
) -> None:
    cache = PredictionCache(tmp_path)
    lr = torch.full((4, 2, 2), 0.25, dtype=torch.float32)
    provenance = build_cache_provenance({"name": "ldsr-s2-x4", "scale": 4, "seed": 3407})
    identity = build_identity(provenance, phase2b3a._SOURCE, "sample-1", lr)
    cache.put(identity, torch.full((4, 8, 8), 0.5, dtype=torch.float32))
    assert (
        phase2b3a._existing_a2_seed_count(
            tmp_path,
            ({"sample_id": "sample-1"},),
            (3407,),
            {3407: provenance},
            lambda record: lr,
        )
        == 1
    )
    tensor_path = tmp_path / f"{identity.key}.safetensors"
    payload = bytearray(tensor_path.read_bytes())
    payload[-1] ^= 1
    tensor_path.write_bytes(payload)

    with pytest.raises(RuntimeError):
        phase2b3a._existing_a2_seed_count(
            tmp_path,
            ({"sample_id": "sample-1"},),
            (3407,),
            {3407: provenance},
            lambda record: lr,
        )


def test_stale_complete_cache_entry_does_not_reduce_missing_count(tmp_path: Path) -> None:
    cache = PredictionCache(tmp_path)
    lr = torch.full((4, 2, 2), 0.25, dtype=torch.float32)
    current = build_cache_provenance(
        {"name": "ldsr-s2-x4", "scale": 4, "seed": 3407, "checkpoint_sha256": "a" * 64}
    )
    stale = build_cache_provenance(
        {"name": "ldsr-s2-x4", "scale": 4, "seed": 3407, "checkpoint_sha256": "b" * 64}
    )
    cache.put(
        build_identity(stale, phase2b3a._SOURCE, "sample-1", lr),
        torch.full((4, 8, 8), 0.5, dtype=torch.float32),
    )
    loads: list[str] = []
    assert (
        phase2b3a._existing_a2_seed_count(
            tmp_path,
            ({"sample_id": "sample-1"},),
            (3407,),
            {3407: current},
            lambda record: loads.append(record["sample_id"]) or lr,
        )
        == 0
    )
    assert loads == []


@pytest.mark.parametrize("suffix", [".json", ".safetensors"])
def test_projection_cache_rejects_orphan_or_symlink(tmp_path: Path, suffix: str) -> None:
    (tmp_path / f"{'a' * 64}{suffix}").write_bytes(b"{}")
    with pytest.raises(RuntimeError, match="orphan"):
        phase2b3a._existing_a2_seed_count(
            tmp_path,
            ({"sample_id": "sample-1"},),
            (3407,),
            {3407: {}},
            lambda record: None,
        )


def test_projection_cache_rejects_symlink_entry(tmp_path: Path) -> None:
    (tmp_path / f"{'a' * 64}.json").symlink_to(tmp_path / "missing")
    (tmp_path / f"{'a' * 64}.safetensors").write_bytes(b"bytes")
    with pytest.raises(RuntimeError, match="invalid"):
        phase2b3a._existing_a2_seed_count(
            tmp_path,
            ({"sample_id": "sample-1"},),
            (3407,),
            {3407: {}},
            lambda record: None,
        )


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
        {"sample_id": f"development-{index}", "split": "development"} for index in range(count)
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

    def load_pair(root, record, *, manifest_sha256, normalization_policy):
        assert root == tmp_path
        assert manifest_sha256 == POST_MANIFEST_SHA256
        assert normalization_policy == PHASE2B3A_NORMALIZATION_POLICY
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
    bundles = phase2b3a._one_shot_bundles(("roi-1", "roi-2"), (object(),) * 3, (3407,), object())
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


def test_git_inspection_nonzero_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        phase2b3a.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 128, "", "bad git"),
    )
    with pytest.raises(ValueError, match="Git"):
        phase2b3a._run_git(tmp_path, "rev-parse", "HEAD")


def test_stage_preflight_passes_required_reviewed_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str | None] = []
    monkeypatch.setattr(phase2b3a, "_validate_upstream", lambda args: (Path("/x"), ()))
    monkeypatch.setattr(phase2b3a, "_require_safe_output_roots", lambda root: None)
    monkeypatch.setattr(
        phase2b3a,
        "_validate_reviewed_checkout",
        lambda root, *, expected_revision=None: (
            seen.append(expected_revision) or (Path("/reviewed"), expected_revision)
        ),
    )
    monkeypatch.setattr(phase2b3a, "resolve_project_root", lambda path: Path("/reviewed"))
    monkeypatch.setattr(phase2b3a, "_require_execution_from_reviewed_root", lambda *a: None)
    monkeypatch.setattr(phase2b3a, "_require_safe_model_directory", lambda path: path)
    monkeypatch.setattr(phase2b3a, "_validate_base_runtime", lambda: None)
    monkeypatch.setattr(phase2b3a, "capture_gpu_hardware", lambda: object())
    monkeypatch.setattr(
        phase2b3a.shutil, "disk_usage", lambda root: type("D", (), {"free": 10 * 1024**3})()
    )
    args = argparse.Namespace(
        project_root=Path("/reviewed"),
        reviewed_commit="d" * 40,
        sen2srlite_model_dir=Path("/sen2"),
        ldsr_model_dir=Path("/ldsr"),
    )
    monkeypatch.setattr(phase2b3a, "_model_factory", lambda args: ())
    revisions: list[str] = []
    monkeypatch.setattr(
        phase2b3a,
        "_write_preflight_runtime",
        lambda context, models: (
            revisions.append(context.code_revision) or {"stage": "preflight", "model_count": 0}
        ),
    )
    assert phase2b3a.run_preflight(args) == {"stage": "preflight", "model_count": 0}
    assert revisions == ["d" * 40]
    assert seen == ["d" * 40]


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
    assert phase2b3a.run_development_replay(argparse.Namespace()) == {"stage": "development-replay"}
    assert called == []


@pytest.mark.parametrize("phase", ["a1", "a2"])
def test_real_replay_implementation_never_constructs_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    context = phase2b3a._StageContext(
        root=tmp_path,
        records=(),
        project_root=tmp_path,
        code_revision="a" * 40,
        persistent_free_bytes=10 * 1024**3,
        hardware=None,
    )
    monkeypatch.setattr(phase2b3a, "_preflight_context", lambda args, *, capture_hardware: context)
    monkeypatch.setattr(
        phase2b3a,
        "_model_factory",
        lambda args: (_ for _ in ()).throw(AssertionError("model factory called")),
    )
    radiometric_policy = {
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "sample_count": 4 if phase == "a1" else 120,
    }
    runtime_name = phase2b3a._A1_RUNTIME if phase == "a1" else phase2b3a._A2_RUNTIME
    runtime = (
        {
            "schema": "trustsr.phase2b3a-a1-runtime.v2",
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "radiometric_policy": radiometric_policy,
        }
        if phase == "a1"
        else {
            "schema": "trustsr.phase2b3a-a2-runtime.v1",
            "git_commit": "a" * 40,
            "a1_producer_commit": "9" * 40,
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "radiometric_policy": radiometric_policy,
        }
    )
    _, runtime_sha = phase2b3a._write_runtime(tmp_path, runtime_name, runtime)
    scientific = {
        "schema": (
            "trustsr.phase2b3a-development-smoke.v2"
            if phase == "a1"
            else "trustsr.phase2b3a-development-score-audit.v1"
        ),
        "scientific": True,
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
        "radiometric_policy": radiometric_policy,
    }
    result = {**scientific, "runtime_manifest_sha256": runtime_sha}
    audit = {
        "schema": (
            "trustsr.phase2b3a-development-smoke-cache-audit.v2"
            if phase == "a1"
            else "trustsr.phase2b3a-development-score-cache-audit.v1"
        ),
        "audit": True,
        "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
    }
    if phase == "a1":
        phase2b3a._commit_pair(
            tmp_path, (phase2b3a._A1_RESULT, result), (phase2b3a._A1_AUDIT, audit)
        )
        monkeypatch.setattr(phase2b3a, "_load_a1_pairs", lambda context, args: ())
        monkeypatch.setattr(
            phase2b3a, "replay_a1_smoke", lambda *args: (scientific, audit)
        )
        outcome = phase2b3a.run_replay(argparse.Namespace())
        assert outcome == {"stage": "replay", "byte_identical": True}
    else:
        ancestry: list[tuple[Path, object, str]] = []
        monkeypatch.setattr(
            phase2b3a,
            "_require_git_ancestor",
            lambda root, ancestor, descendant: (
                ancestry.append((root, ancestor, descendant)) or ancestor
            ),
        )
        phase2b3a._commit_pair(
            tmp_path, (phase2b3a._A2_RESULT, result), (phase2b3a._A2_AUDIT, audit)
        )
        monkeypatch.setattr(phase2b3a, "_ordered_development_sample_ids", lambda records: ())
        monkeypatch.setattr(phase2b3a, "_load_a2_pairs", lambda context, args: ())
        monkeypatch.setattr(
            phase2b3a,
            "replay_a2_development",
            lambda *args, **kwargs: (scientific, audit),
        )
        outcome = phase2b3a.run_development_replay(argparse.Namespace())
        assert outcome == {"stage": "development-replay", "byte_identical": True}
        assert ancestry == [(tmp_path, "9" * 40, "a" * 40)]


def test_a2_replay_rejects_runtime_normalization_policy_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = phase2b3a._StageContext(
        root=tmp_path,
        records=(),
        project_root=tmp_path,
        code_revision="a" * 40,
        persistent_free_bytes=10 * 1024**3,
        hardware=None,
    )
    policy = {"normalization_policy": PHASE2B3A_NORMALIZATION_POLICY, "sample_count": 120}
    runtime = {
        "schema": "trustsr.phase2b3a-a2-runtime.v1",
        "git_commit": "a" * 40,
        "a1_producer_commit": "9" * 40,
        "normalization_policy": "uint16_divide_10000_no_clip_v1",
        "radiometric_policy": policy,
    }
    _, runtime_sha = phase2b3a._write_runtime(tmp_path, phase2b3a._A2_RUNTIME, runtime)
    phase2b3a._commit_pair(
        tmp_path,
        (
            phase2b3a._A2_RESULT,
            {
                "schema": "trustsr.phase2b3a-development-score-audit.v1",
                "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
                "radiometric_policy": policy,
                "runtime_manifest_sha256": runtime_sha,
            },
        ),
        (
            phase2b3a._A2_AUDIT,
            {
                "schema": "trustsr.phase2b3a-development-score-cache-audit.v1",
                "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            },
        ),
    )
    monkeypatch.setattr(phase2b3a, "_preflight_context", lambda *args, **kwargs: context)
    monkeypatch.setattr(phase2b3a, "_ordered_development_sample_ids", lambda records: ())
    monkeypatch.setattr(phase2b3a, "_load_a2_pairs", lambda context, args: ())
    monkeypatch.setattr(phase2b3a, "_require_git_ancestor", lambda *args: "9" * 40)

    with pytest.raises(ValueError, match="normalization|policy"):
        phase2b3a.run_development_replay(argparse.Namespace())


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


@pytest.mark.parametrize(
    ("stage", "include_k5"),
    [
        ("single", None),
        ("smoke", None),
        ("development", False),
        ("development", True),
    ],
    ids=("single", "smoke", "development-single-seed", "development-k5-enabled"),
)
def test_compute_stage_actual_handlers_preserve_frozen_membership_and_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    include_k5: bool | None,
) -> None:
    records = tuple({"sample_id": f"sample-{index}"} for index in range(120))
    hardware = SimpleNamespace(memory_total_mib=1024)
    context = phase2b3a._StageContext(
        root=tmp_path,
        records=records,
        project_root=tmp_path,
        code_revision="a" * 40,
        persistent_free_bytes=10 * 1024**3,
        hardware=hardware,
    )
    monkeypatch.setattr(phase2b3a, "_preflight_context", lambda args, *, capture_hardware: context)
    monkeypatch.setattr(phase2b3a, "_validate_models", lambda models: None)
    monkeypatch.setattr(phase2b3a, "_write_runtime", lambda *args: (b"{}", "b" * 64))
    monkeypatch.setattr(phase2b3a, "_commit_pair", lambda *args: (b"{}", b"{}"))
    args = argparse.Namespace(selection_manifest_sha256=POST_MANIFEST_SHA256)

    if stage == "single":
        lr = torch.zeros((4, 2, 2), dtype=torch.float32)
        pair = SimpleNamespace(pair=SimpleNamespace(lr=lr))
        seeded = SimpleNamespace(predict=lambda tensor: torch.zeros((4, 8, 8), dtype=torch.float32))
        ldsr = SimpleNamespace(for_seed=lambda seed: seeded)
        monkeypatch.setattr(
            phase2b3a, "_load_a1_pairs", lambda context, args, single=False: (pair,)
        )
        monkeypatch.setattr(phase2b3a, "_model_factory", lambda args: (object(), object(), ldsr))
        monkeypatch.setattr(phase2b3a, "_runtime_measurement", lambda: 1)
        outcome = phase2b3a.run_single(args)
        assert outcome == {"stage": "single", "repeatability_pass": True}
    elif stage == "smoke":
        radiometric_policy = {
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "raw_radiometric_max": 32767,
            "saturation_threshold": 10000,
            "bands": ["B04", "B03", "B02", "B08"],
            "sample_count": 4,
            "affected_sample_count": 0,
            "affected_asset_count": 0,
            "lr_clipped_high_count": 0,
            "hr_clipped_high_count": 0,
            "raw_crop_maximum": 5500,
        }
        pairs = tuple(
            SimpleNamespace(
                pair=SimpleNamespace(sample_id=f"sample-{index}", lr=torch.zeros((4, 1, 1)))
            )
            for index in range(4)
        )
        monkeypatch.setattr(phase2b3a, "_load_a1_pairs", lambda context, args: pairs)
        monkeypatch.setattr(
            phase2b3a,
            "_model_factory",
            lambda args: (
                object(),
                object(),
                SimpleNamespace(
                    for_seed=lambda seed: SimpleNamespace(
                        provenance=lambda: {"name": "ldsr-s2-x4", "seed": seed}
                    )
                ),
            ),
        )
        built: list[tuple[str, tuple[int, ...]]] = []

        def bundle_for_pair(pair, models, seeds, cache):
            del models, cache
            sample_id = pair.pair.sample_id
            built.append((sample_id, seeds))
            return sample_id, seeds

        def consume_a1(passed_pairs, bundles, score_cache):
            del score_cache
            assert passed_pairs is pairs
            assert bundles == tuple((pair.pair.sample_id, phase2b3a.A1_SEEDS) for pair in pairs)
            assert built == [
                (pair.pair.sample_id, phase2b3a.A1_SEEDS) for pair in pairs
            ]
            return {
                "include_ldsr_variance_k5": False,
                "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
                "radiometric_policy": radiometric_policy,
            }, {}

        monkeypatch.setattr(phase2b3a, "_bundle_for_pair", bundle_for_pair)
        monkeypatch.setattr(
            phase2b3a,
            "evaluate_a1_smoke",
            consume_a1,
        )
        monkeypatch.setattr(
            phase2b3a,
            "_read_canonical",
            lambda path: (
                b"{}",
                {
                    "schema": "trustsr.phase2b3a-single-runtime.v1",
                    "git_commit": "a" * 40,
                    "single_repeatability_pass": True,
                    "uncached_ldsr_prediction_seconds": [1.0],
                    "single_peak_memory_bytes": 1,
                    "gpu_total_memory_bytes": 100,
                },
            ),
        )
        monkeypatch.setattr(phase2b3a, "select_development_records", lambda records: records)
        projection_lr = torch.full((4, 1, 1), 0.25, dtype=torch.float32)
        projection_provenance = build_cache_provenance(
            {"name": "ldsr-s2-x4", "seed": 3407}
        )
        projection_identity = build_identity(
            projection_provenance,
            phase2b3a._SOURCE,
            "sample-4",
            projection_lr,
        )
        PredictionCache(phase2b3a._prediction_cache_directory(tmp_path)).put(
            projection_identity, torch.ones((4, 4, 4), dtype=torch.float32)
        )
        projection_loads: list[str] = []

        def load_projection_pair(root, record, *, manifest_sha256, normalization_policy):
            assert root == tmp_path
            assert manifest_sha256 == POST_MANIFEST_SHA256
            assert normalization_policy == PHASE2B3A_NORMALIZATION_POLICY
            projection_loads.append(record["sample_id"])
            return SimpleNamespace(pair=SimpleNamespace(lr=projection_lr))

        monkeypatch.setattr(phase2b3a, "load_crosssensor_pair", load_projection_pair)
        runtime_payloads: list[dict[str, object]] = []
        monkeypatch.setattr(
            phase2b3a,
            "_write_runtime",
            lambda root, name, payload: (
                runtime_payloads.append(payload) or (b"{}", "b" * 64)
            ),
        )
        outcome = phase2b3a.run_smoke(args)
        assert outcome["stage"] == "smoke" and outcome["sample_count"] == 4
        assert runtime_payloads[0]["schema"] == "trustsr.phase2b3a-a1-runtime.v2"
        assert runtime_payloads[0]["normalization_policy"] == PHASE2B3A_NORMALIZATION_POLICY
        assert runtime_payloads[0]["radiometric_policy"] == radiometric_policy
        assert projection_loads == ["sample-4"]
    else:
        radiometric_policy = {
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "raw_radiometric_max": 32767,
            "saturation_threshold": 10000,
            "bands": ["B04", "B03", "B02", "B08"],
            "sample_count": 120,
            "affected_sample_count": 1,
            "affected_asset_count": 2,
            "lr_clipped_high_count": 8,
            "hr_clipped_high_count": 117,
            "raw_crop_maximum": 11968,
        }
        pairs = tuple(
            SimpleNamespace(pair=SimpleNamespace(sample_id=f"sample-{index}"))
            for index in range(120)
        )
        monkeypatch.setattr(
            phase2b3a,
            "_verify_a1_cloud_acceptance",
            lambda context: (include_k5, "c" * 64, "d" * 40),
        )
        monkeypatch.setattr(phase2b3a, "select_development_records", lambda records: records)
        monkeypatch.setattr(phase2b3a, "_load_a2_pairs", lambda context, args: pairs)
        monkeypatch.setattr(
            phase2b3a, "_model_factory", lambda args: (object(), object(), object())
        )
        built = []

        def bundle_for_pair(pair, models, seeds, cache):
            del models, cache
            sample_id = pair.pair.sample_id
            built.append((sample_id, seeds))
            return sample_id, seeds

        def consume_a2(
            passed_pairs,
            bundles,
            *,
            prediction_cache,
            score_cache,
            include_ldsr_variance_k5,
            code_revision,
            ordered_development_sample_ids,
        ):
            del prediction_cache, score_cache
            assert passed_pairs is pairs
            assert iter(bundles) is bundles
            expected_seeds = phase2b3a.K5A_SEEDS if include_k5 else (3407,)
            expected = [(f"sample-{index}", expected_seeds) for index in range(120)]
            assert list(bundles) == expected
            assert list(bundles) == []
            assert built == expected
            assert include_ldsr_variance_k5 is include_k5
            assert code_revision == "a" * 40
            assert ordered_development_sample_ids == tuple(
                f"sample-{index}" for index in range(120)
            )
            return {
                "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
                "radiometric_policy": radiometric_policy,
            }, {}

        monkeypatch.setattr(phase2b3a, "_bundle_for_pair", bundle_for_pair)
        monkeypatch.setattr(phase2b3a, "evaluate_a2_development", consume_a2)
        runtime_payloads: list[dict[str, object]] = []
        monkeypatch.setattr(
            phase2b3a,
            "_write_runtime",
            lambda root, name, payload: (
                runtime_payloads.append(payload) or (b"{}", "b" * 64)
            ),
        )
        outcome = phase2b3a.run_development(args)
        assert outcome["stage"] == "development" and outcome["sample_count"] == 120
        assert runtime_payloads[0]["git_commit"] == "a" * 40
        assert runtime_payloads[0]["a1_producer_commit"] == "d" * 40
        assert runtime_payloads[0]["schema"] == "trustsr.phase2b3a-a2-runtime.v1"
        assert runtime_payloads[0]["normalization_policy"] == PHASE2B3A_NORMALIZATION_POLICY
        assert runtime_payloads[0]["radiometric_policy"] == radiometric_policy


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

    phase2b3a._commit_pair(tmp_path, (phase2b3a._A1_RESULT, {}), (phase2b3a._A1_AUDIT, {}))
    phase2b3a._commit_pair(tmp_path, (phase2b3a._A2_RESULT, {}), (phase2b3a._A2_AUDIT, {}))

    phase2b3a._write_bundle_manifest(tmp_path, "a1")
    a1_manifest = json.loads(
        (directory / "phase2b3a-bundle-manifest.json").read_text(encoding="utf-8")
    )
    assert a1_manifest["schema"] == "trustsr.phase2b3a-bundle-manifest.v2"
    phase2b3a._write_bundle_manifest(tmp_path, "a2")

    manifest = json.loads(
        (directory / "phase2b3a-bundle-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == "trustsr.phase2b3a-bundle-manifest.v1"
    assert manifest["phase"] == "a2"
    assert all("-a2-" in item["basename"] for item in manifest["files"])


def test_a1_replay_receipt_is_v2_while_a2_remains_v1() -> None:
    a1 = phase2b3a._receipt("a1", b"result", b"audit", "a" * 64)
    a2 = phase2b3a._receipt("a2", b"result", b"audit", "a" * 64)

    assert a1["schema"] == "trustsr.phase2b3a-a1-replay.v2"
    assert a2["schema"] == "trustsr.phase2b3a-a2-replay.v1"


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
    (directory / "phase2b3a-a1-cache-audit.json").write_bytes(b"x" * (5 * 1024**2 + 1))
    marker = phase2b3a._pair_marker(
        phase2b3a._A1_RESULT,
        (directory / phase2b3a._A1_RESULT).read_bytes(),
        phase2b3a._A1_AUDIT,
        (directory / phase2b3a._A1_AUDIT).read_bytes(),
    )
    (directory / phase2b3a._A1_PAIR_COMMIT).write_bytes(marker)

    with pytest.raises(ValueError, match="5 MiB"):
        phase2b3a._write_bundle_manifest(tmp_path, "a1")


def test_scientific_pair_is_committed_only_after_both_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = phase2b3a.atomic_write_bytes
    writes = 0

    def fail_second(path: Path, payload: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("injected second write failure")
        original(path, payload)

    monkeypatch.setattr(phase2b3a, "atomic_write_bytes", fail_second)
    with pytest.raises(OSError, match="injected"):
        phase2b3a._commit_pair(
            tmp_path,
            (phase2b3a._A1_RESULT, {"value": 1}),
            (phase2b3a._A1_AUDIT, {"value": 2}),
        )
    directory = phase2b3a._result_directory(tmp_path)
    assert not (directory / phase2b3a._A1_PAIR_COMMIT).exists()
    with pytest.raises(ValueError, match="committed|pair"):
        phase2b3a._read_committed_pair(tmp_path, phase2b3a._A1_RESULT, phase2b3a._A1_AUDIT)


def test_partial_same_byte_pair_can_complete_but_collision_fails_closed(
    tmp_path: Path,
) -> None:
    directory = phase2b3a._result_directory(tmp_path)
    directory.mkdir(parents=True)
    result_bytes = phase2b3a.canonical_json({"value": 1})
    (directory / phase2b3a._A1_RESULT).write_bytes(result_bytes)
    phase2b3a._commit_pair(
        tmp_path,
        (phase2b3a._A1_RESULT, {"value": 1}),
        (phase2b3a._A1_AUDIT, {"value": 2}),
    )
    assert (
        phase2b3a._read_committed_pair(tmp_path, phase2b3a._A1_RESULT, phase2b3a._A1_AUDIT)[0][0]
        == result_bytes
    )
    with pytest.raises(ValueError, match="different bytes"):
        phase2b3a._commit_pair(
            tmp_path,
            (phase2b3a._A1_RESULT, {"value": 9}),
            (phase2b3a._A1_AUDIT, {"value": 2}),
        )


def test_concurrent_scientific_pair_writers_commit_one_identical_pair(tmp_path: Path) -> None:
    script = (
        "import sys; from pathlib import Path; from trustsr.cli.phase2b3a import "
        "_commit_pair,_A1_RESULT,_A1_AUDIT; "
        "_commit_pair(Path(sys.argv[1]), (_A1_RESULT, {'value': 1}), (_A1_AUDIT, {'value': 2}))"
    )
    processes = [subprocess.Popen([sys.executable, "-c", script, str(tmp_path)]) for _ in range(4)]
    assert [process.wait() for process in processes] == [0, 0, 0, 0]
    (result, _), (audit, _) = phase2b3a._read_committed_pair(
        tmp_path, phase2b3a._A1_RESULT, phase2b3a._A1_AUDIT
    )
    assert result == phase2b3a.canonical_json({"value": 1})
    assert audit == phase2b3a.canonical_json({"value": 2})
