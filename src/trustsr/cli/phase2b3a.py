"""Run the six fail-closed Phase 2B3-A cloud stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from trustsr.artifacts.gpu_run import (
    GPUHardwareSnapshot,
    capture_gpu_hardware,
    collect_gpu_environment,
    resolve_project_root,
)
from trustsr.artifacts.predictions import (
    PredictionCache,
    PredictionIdentity,
    tensor_sha256,
)
from trustsr.artifacts.scores import ScoreCache
from trustsr.cli.phase2b2b import (
    _require_safe_derived_path,
    _require_safe_model_directory,
    _validate_upstream,
)
from trustsr.data.crosssensor_pairs import (
    POST_MANIFEST_SHA256,
    LoadedCrosssensorPair,
    load_crosssensor_pair,
    select_development_records,
    select_development_smoke_records,
)
from trustsr.evaluation.crosssensor_smoke import INPUT_AUDIT_SHA256
from trustsr.evaluation.development_predictions import (
    A1_SEEDS,
    K5A_SEEDS,
    DevelopmentPredictionBundle,
    load_or_generate_prediction_bundle,
)
from trustsr.evaluation.development_score_audit import (
    evaluate_a1_smoke,
    evaluate_a2_development,
    replay_a1_smoke,
    replay_a2_development,
)
from trustsr.jsonio import atomic_write_bytes, canonical_json
from trustsr.models.protocols import SRModel

STAGES = (
    "preflight",
    "single",
    "smoke",
    "replay",
    "development",
    "development-replay",
)

_A1_RESULT = "phase2b3a-a1-result.json"
_A1_AUDIT = "phase2b3a-a1-cache-audit.json"
_A1_RUNTIME = "phase2b3a-a1-runtime.json"
_A1_REPLAY = "phase2b3a-a1-replay.json"
_A2_RESULT = "phase2b3a-a2-result.json"
_A2_AUDIT = "phase2b3a-a2-cache-audit.json"
_A2_RUNTIME = "phase2b3a-a2-runtime.json"
_A2_REPLAY = "phase2b3a-a2-replay.json"
_BUNDLE_MANIFEST = "phase2b3a-bundle-manifest.json"
_SINGLE_RUNTIME = "phase2b3a-single-runtime.json"
_PREFLIGHT_RUNTIME = "phase2b3a-preflight-runtime.json"
_MINIMUM_FREE_BYTES = 10 * 1024**3
_MAX_BUNDLE_FILE_BYTES = 5 * 1024**2
_MODEL_NAMES = ("bicubic-x4", "sen2srlite-x4", "ldsr-s2-x4")
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class _StageContext:
    root: Path
    records: tuple[dict[str, object], ...]
    project_root: Path
    code_revision: str
    persistent_free_bytes: int
    hardware: GPUHardwareSnapshot | None


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest-sha256", required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--input-audit-sha256", required=True)
    parser.add_argument("--sen2srlite-model-dir", type=Path, required=True)
    parser.add_argument("--ldsr-model-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--confirm-cloud-storage", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    handlers = (
        run_preflight,
        run_single,
        run_smoke,
        run_replay,
        run_development,
        run_development_replay,
    )
    for name, handler in zip(STAGES, handlers, strict=True):
        child = subparsers.add_parser(name)
        _add_arguments(child)
        child.set_defaults(handler=handler)
    return parser


def _phase_root(root: Path) -> Path:
    return root / "trustsr" / "phase2b3a"


def _prediction_cache_directory(root: Path) -> Path:
    return _phase_root(root) / "predictions" / POST_MANIFEST_SHA256


def _score_cache_directory(root: Path) -> Path:
    return _phase_root(root) / "scores" / POST_MANIFEST_SHA256


def _result_directory(root: Path) -> Path:
    return _phase_root(root) / "results" / POST_MANIFEST_SHA256


def _run_git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("reviewed Git checkout cannot be inspected") from exc
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _validate_reviewed_checkout(
    project_root: Path, *, expected_revision: str | None = None
) -> tuple[Path, str]:
    root = resolve_project_root(project_root)
    revision = _run_git(root, "rev-parse", "HEAD")
    branch = _run_git(root, "symbolic-ref", "--short", "HEAD")
    status = _run_git(root, "status", "--porcelain")
    if not branch:
        raise ValueError("reviewed Git checkout must not be detached")
    if status:
        raise ValueError("reviewed Git checkout must be clean")
    if len(revision) != 40 or any(character not in _HEX for character in revision):
        raise ValueError("reviewed Git checkout has an invalid commit")
    if expected_revision is not None and revision != expected_revision:
        raise ValueError("reviewed Git checkout commit mismatch")
    return root, revision


def _require_execution_from_reviewed_root(
    reviewed_root: Path, execution_root: Path
) -> None:
    if reviewed_root.resolve(strict=True) != execution_root.resolve(strict=True):
        raise ValueError("executing checkout and reviewed project root mismatch")


def _require_safe_output_roots(root: Path) -> None:
    for directory in (
        _prediction_cache_directory(root),
        _score_cache_directory(root),
        _result_directory(root),
    ):
        _require_safe_derived_path(root, directory)
        if directory.exists() and not directory.is_dir():
            raise ValueError("Phase 2B3-A output directory must be a directory")


def _validate_base_runtime() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Phase 2B3-A requires the reviewed Python 3.12 base runtime")
    if not isinstance(torch.__version__, str) or not torch.__version__:
        raise RuntimeError("Phase 2B3-A requires the reviewed base PyTorch package")


def _preflight_context(
    args: argparse.Namespace, *, capture_hardware: bool
) -> _StageContext:
    root, records = _validate_upstream(args)
    _require_safe_output_roots(root)
    project_root, revision = _validate_reviewed_checkout(Path(args.project_root))
    execution_root = resolve_project_root(Path(__file__).resolve().parents[3])
    _require_execution_from_reviewed_root(project_root, execution_root)
    _require_safe_model_directory(Path(args.sen2srlite_model_dir))
    _require_safe_model_directory(Path(args.ldsr_model_dir))
    _validate_base_runtime()
    persistent_free_bytes = shutil.disk_usage(root).free
    if persistent_free_bytes < _MINIMUM_FREE_BYTES:
        raise RuntimeError("Phase 2B3-A requires at least 10 GiB persistent free space")
    hardware = capture_gpu_hardware() if capture_hardware else None
    return _StageContext(
        root=root,
        records=records,
        project_root=project_root,
        code_revision=revision,
        persistent_free_bytes=persistent_free_bytes,
        hardware=hardware,
    )


def _model_factory(args: argparse.Namespace) -> tuple[SRModel, SRModel, Any]:
    sen2srlite_directory = _require_safe_model_directory(
        Path(args.sen2srlite_model_dir)
    )
    ldsr_directory = _require_safe_model_directory(Path(args.ldsr_model_dir))
    from trustsr.models.bicubic import BicubicX4
    from trustsr.models.ldsr_s2 import LDSRS2X4
    from trustsr.models.sen2srlite import SEN2SRLiteX4

    return (
        BicubicX4(),
        SEN2SRLiteX4.from_pretrained(sen2srlite_directory, device="cpu"),
        LDSRS2X4.from_pretrained(ldsr_directory, device="cuda:0"),
    )


def _validate_models(models: Sequence[SRModel]) -> None:
    if tuple(getattr(model, "name", None) for model in models) != _MODEL_NAMES:
        raise ValueError("models must use the frozen Phase 2B3-A order")
    for model in models:
        provenance = model.provenance()
        if (
            getattr(model, "scale", None) != 4
            or provenance.get("name") != model.name
            or provenance.get("scale") != 4
        ):
            raise ValueError("model provenance does not match the frozen adapter")


def _load_pairs(
    context: _StageContext, records: Sequence[Mapping[str, object]], args: argparse.Namespace
) -> tuple[LoadedCrosssensorPair, ...]:
    return tuple(
        load_crosssensor_pair(
            context.root,
            record,
            manifest_sha256=args.selection_manifest_sha256,
        )
        for record in records
    )


def _load_a1_pairs(
    context: _StageContext, args: argparse.Namespace, *, single: bool = False
) -> tuple[LoadedCrosssensorPair, ...]:
    selected = select_development_smoke_records(context.records)
    return _load_pairs(context, selected[:1] if single else selected, args)


def _load_a2_pairs(
    context: _StageContext, args: argparse.Namespace
) -> tuple[LoadedCrosssensorPair, ...]:
    return _load_pairs(context, select_development_records(context.records), args)


def _ordered_development_sample_ids(
    records: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    selected = select_development_records(records)
    result = tuple(record.get("sample_id") for record in selected)
    if any(type(sample_id) is not str or not sample_id for sample_id in result):
        raise ValueError("authoritative development sample IDs are invalid")
    return result  # type: ignore[return-value]


class _TimedSeedView:
    def __init__(self, model: SRModel, durations: list[float]) -> None:
        self._model = model
        self._durations = durations
        self.name = model.name
        self.scale = model.scale

    def provenance(self) -> Mapping[str, object]:
        return self._model.provenance()

    def predict(self, lr: torch.Tensor) -> torch.Tensor:
        started = time.monotonic()
        result = self._model.predict(lr)
        self._durations.append(time.monotonic() - started)
        return result


class _TimedLDSR:
    name = "ldsr-s2-x4"
    scale = 4

    def __init__(self, model: Any, durations: list[float]) -> None:
        self._model = model
        self._durations = durations

    def provenance(self) -> Mapping[str, object]:
        return self._model.provenance()

    def for_seed(self, seed: int) -> _TimedSeedView:
        return _TimedSeedView(self._model.for_seed(seed), self._durations)


def _bundle_for_pair(
    pair: LoadedCrosssensorPair,
    models: Sequence[Any],
    seeds: tuple[int, ...],
    cache: PredictionCache,
) -> DevelopmentPredictionBundle:
    return load_or_generate_prediction_bundle(
        pair,
        bicubic=models[0],
        sen2srlite=models[1],
        ldsr=models[2],
        ldsr_seeds=seeds,
        cache=cache,
    )


def _one_shot_bundles(
    pairs: Sequence[LoadedCrosssensorPair],
    models: Sequence[Any],
    seeds: tuple[int, ...],
    cache: PredictionCache,
) -> Iterable[DevelopmentPredictionBundle]:
    for pair in pairs:
        bundle = _bundle_for_pair(pair, models, seeds, cache)
        yield bundle
        del bundle


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_canonical(path: Path) -> tuple[bytes, dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required evidence file is invalid: {path.name}")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"required evidence cannot be read: {path.name}") from exc
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise ValueError(f"required evidence is not canonical JSON: {path.name}")
    return raw, value


def _preflight_output(path: Path, payload: bytes, root: Path) -> bool:
    _require_safe_derived_path(root, path)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError("existing deterministic output must be a regular file")
        if path.read_bytes() != payload:
            raise ValueError("existing deterministic output has different bytes")
        return True
    return False


def _commit_identical_or_new(path: Path, payload: bytes, root: Path) -> bool:
    reused = _preflight_output(path, payload, root)
    if not reused:
        atomic_write_bytes(path, payload)
    return reused


def _commit_pair(
    root: Path,
    first: tuple[str, Mapping[str, object]],
    second: tuple[str, Mapping[str, object]],
) -> tuple[bytes, bytes]:
    directory = _result_directory(root)
    first_bytes, second_bytes = canonical_json(dict(first[1])), canonical_json(dict(second[1]))
    _preflight_output(directory / first[0], first_bytes, root)
    _preflight_output(directory / second[0], second_bytes, root)
    _commit_identical_or_new(directory / first[0], first_bytes, root)
    _commit_identical_or_new(directory / second[0], second_bytes, root)
    return first_bytes, second_bytes


def _write_runtime(root: Path, name: str, value: Mapping[str, object]) -> tuple[bytes, str]:
    payload = canonical_json(dict(value))
    _commit_identical_or_new(_result_directory(root) / name, payload, root)
    return payload, _sha256(payload)


def _runtime_measurement() -> int:
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0


def _project_a2_uncached_seconds(
    uncached_ldsr_prediction_seconds: Sequence[float], *, missing_seed_predictions: int
) -> float:
    durations = tuple(float(value) for value in uncached_ldsr_prediction_seconds)
    if (
        not durations
        or any(not math.isfinite(value) or value < 0.0 for value in durations)
        or type(missing_seed_predictions) is not int
        or missing_seed_predictions < 0
    ):
        raise ValueError("A2 projection inputs are invalid")
    return float(statistics.median(durations) * missing_seed_predictions)


def _resource_gate_pass(
    *,
    single_peak_memory_bytes: int,
    gpu_total_memory_bytes: int,
    persistent_free_bytes: int,
    projected_a2_uncached_seconds: float,
) -> bool:
    if any(
        type(value) is not int or value < 0
        for value in (
            single_peak_memory_bytes,
            gpu_total_memory_bytes,
            persistent_free_bytes,
        )
    ) or gpu_total_memory_bytes == 0:
        raise ValueError("resource measurements must be non-negative integers")
    return bool(
        single_peak_memory_bytes <= int(0.80 * gpu_total_memory_bytes)
        and persistent_free_bytes >= _MINIMUM_FREE_BYTES
        and 1.5 * float(projected_a2_uncached_seconds) <= 2 * 60 * 60
    )


def _existing_a2_seed_count(
    cache_root: Path, sample_ids: Sequence[str], seeds: Sequence[int]
) -> int:
    if not cache_root.exists():
        return 0
    wanted = set(sample_ids)
    expected = {(sample_id, seed) for sample_id in wanted for seed in seeds}
    found: set[tuple[str, int]] = set()
    cache = PredictionCache(cache_root)
    for metadata_path in cache_root.glob("*.json"):
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise RuntimeError("prediction cache contains an invalid metadata file")
        raw = metadata_path.read_bytes()
        try:
            metadata = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("prediction cache metadata is invalid") from exc
        if canonical_json(metadata) != raw:
            raise RuntimeError("prediction cache metadata is not canonical")
        identity = metadata.get("identity", {})
        provenance = identity.get("model_provenance", {}) if isinstance(identity, dict) else {}
        sample_id = identity.get("sample_id") if isinstance(identity, dict) else None
        seed = provenance.get("seed") if isinstance(provenance, dict) else None
        if sample_id not in wanted or seed not in seeds or provenance.get("name") != "ldsr-s2-x4":
            continue
        lr = identity.get("lr")
        if (
            set(identity) != {"model_provenance", "source", "sample_id", "lr"}
            or not isinstance(lr, dict)
            or set(lr) != {"shape", "dtype", "sha256"}
            or provenance.get("experiment_schema")
            != "trustsr.phase2b3a-predictions.v1"
            or provenance.get("post_manifest_sha256") != POST_MANIFEST_SHA256
            or provenance.get("input_audit_sha256") != INPUT_AUDIT_SHA256
        ):
            raise RuntimeError("prediction cache identity is invalid for A2 projection")
        try:
            parsed_identity = PredictionIdentity(
                provenance,
                identity["source"],
                identity["sample_id"],
                tuple(lr["shape"]),
                lr["dtype"],
                lr["sha256"],
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("prediction cache identity is invalid for A2 projection") from exc
        if parsed_identity.key != metadata_path.stem or cache.get(parsed_identity) is None:
            raise RuntimeError("prediction cache entry failed A2 projection verification")
        tensor_name = metadata.get("tensor_filename")
        tensor_path = cache_root / tensor_name if isinstance(tensor_name, str) else None
        if (
            tensor_path is None
            or tensor_path.is_symlink()
            or not tensor_path.is_file()
            or metadata_path.stem != tensor_path.stem
        ):
            raise RuntimeError("prediction cache entry is incomplete")
        slot = (sample_id, seed)
        if slot in found:
            raise RuntimeError("prediction cache contains duplicate A2 seed evidence")
        found.add(slot)
    if not found <= expected:
        raise RuntimeError("prediction cache contains unexpected A2 seed evidence")
    return len(found)


def _write_preflight_runtime(
    context: _StageContext, models: Sequence[SRModel]
) -> dict[str, object]:
    _validate_models(models)
    assert context.hardware is not None
    environment = dict(
        collect_gpu_environment(
            hardware_snapshot=context.hardware,
            project_root=context.project_root,
        )
    )
    environment["models"] = [model.provenance() for model in models]
    environment["persistent_free_bytes"] = context.persistent_free_bytes
    atomic_write_bytes(
        _result_directory(context.root) / _PREFLIGHT_RUNTIME,
        canonical_json(environment),
    )
    return {"stage": "preflight", "model_count": 3}


def run_preflight(args: argparse.Namespace) -> dict[str, object]:
    context = _preflight_context(args, capture_hardware=True)
    models = _model_factory(args)
    return _write_preflight_runtime(context, models)


def run_single(args: argparse.Namespace) -> dict[str, object]:
    context = _preflight_context(args, capture_hardware=True)
    pairs = _load_a1_pairs(context, args, single=True)
    models = _model_factory(args)
    _validate_models(models)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    logical_sha256s: list[str] = []
    durations: list[float] = []
    for _ in range(2):
        started = time.monotonic()
        prediction = models[2].for_seed(3407).predict(pairs[0].pair.lr)
        durations.append(time.monotonic() - started)
        logical_sha256s.append(tensor_sha256(prediction))
        del prediction
    if logical_sha256s[0] != logical_sha256s[1]:
        raise RuntimeError("single direct LDSR predictions are not repeatable")
    assert context.hardware is not None
    runtime = {
        "schema": "trustsr.phase2b3a-single-runtime.v1",
        "git_commit": context.code_revision,
        "single_repeatability_pass": True,
        "single_prediction_sha256": logical_sha256s[0],
        "uncached_ldsr_prediction_seconds": durations,
        "single_peak_memory_bytes": _runtime_measurement(),
        "gpu_total_memory_bytes": context.hardware.memory_total_mib * 1024**2,
        "persistent_free_bytes": context.persistent_free_bytes,
    }
    _write_runtime(context.root, _SINGLE_RUNTIME, runtime)
    return {"stage": "single", "repeatability_pass": True}


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    context = _preflight_context(args, capture_hardware=True)
    pairs = _load_a1_pairs(context, args)
    models = _model_factory(args)
    _validate_models(models)
    durations: list[float] = []
    timed_models = (models[0], models[1], _TimedLDSR(models[2], durations))
    prediction_cache = PredictionCache(_prediction_cache_directory(context.root))
    score_cache = ScoreCache(_score_cache_directory(context.root))
    bundles = tuple(
        _one_shot_bundles(pairs, timed_models, A1_SEEDS, prediction_cache)
    )
    result, audit = evaluate_a1_smoke(pairs, bundles, score_cache)
    _, single = _read_canonical(_result_directory(context.root) / _SINGLE_RUNTIME)
    if (
        single.get("schema") != "trustsr.phase2b3a-single-runtime.v1"
        or single.get("git_commit") != context.code_revision
        or single.get("single_repeatability_pass") is not True
    ):
        raise ValueError("single-stage runtime evidence is invalid")
    include = result.get("include_ldsr_variance_k5")
    if type(include) is not bool:
        raise ValueError("A1 result has an invalid K=5 decision")
    target_seeds = K5A_SEEDS if include else (3407,)
    ordered_ids = _ordered_development_sample_ids(context.records)
    existing = _existing_a2_seed_count(
        _prediction_cache_directory(context.root), ordered_ids, target_seeds
    )
    missing = len(ordered_ids) * len(target_seeds) - existing
    if missing < 0:
        raise RuntimeError("A2 cache contains more seed evidence than expected")
    if not durations:
        durations = [float(value) for value in single["uncached_ldsr_prediction_seconds"]]
    projection = _project_a2_uncached_seconds(
        durations, missing_seed_predictions=missing
    )
    runtime = {
        "schema": "trustsr.phase2b3a-a1-runtime.v1",
        "git_commit": context.code_revision,
        "single_repeatability_pass": True,
        "single_peak_memory_bytes": single["single_peak_memory_bytes"],
        "gpu_total_memory_bytes": single["gpu_total_memory_bytes"],
        "persistent_free_bytes": context.persistent_free_bytes,
        "a1_uncached_ldsr_prediction_seconds": durations,
        "a1_median_uncached_ldsr_prediction_seconds": float(statistics.median(durations)),
        "missing_a2_seed_predictions": missing,
        "projected_a2_uncached_seconds": projection,
    }
    runtime["resource_gate_pass"] = _resource_gate_pass(
        single_peak_memory_bytes=int(runtime["single_peak_memory_bytes"]),
        gpu_total_memory_bytes=int(runtime["gpu_total_memory_bytes"]),
        persistent_free_bytes=context.persistent_free_bytes,
        projected_a2_uncached_seconds=projection,
    )
    _, runtime_sha256 = _write_runtime(context.root, _A1_RUNTIME, runtime)
    result = {**result, "runtime_manifest_sha256": runtime_sha256}
    result_bytes, audit_bytes = _commit_pair(
        context.root, (_A1_RESULT, result), (_A1_AUDIT, audit)
    )
    return {
        "stage": "smoke",
        "sample_count": 4,
        "result_sha256": _sha256(result_bytes),
        "cache_audit_sha256": _sha256(audit_bytes),
    }


def _verify_runtime_reference(
    result: Mapping[str, object], runtime_bytes: bytes
) -> str:
    runtime_sha256 = _sha256(runtime_bytes)
    if result.get("runtime_manifest_sha256") != runtime_sha256:
        raise ValueError("scientific result runtime-manifest SHA-256 is invalid")
    return runtime_sha256


def _receipt(
    phase: str,
    result_bytes: bytes,
    audit_bytes: bytes,
    runtime_sha256: str,
) -> dict[str, object]:
    return {
        "schema": f"trustsr.phase2b3a-{phase}-replay.v1",
        "byte_identical": True,
        "result_sha256": _sha256(result_bytes),
        "cache_audit_sha256": _sha256(audit_bytes),
        "runtime_manifest_sha256": runtime_sha256,
    }


def _write_bundle_manifest(root: Path, phase: str) -> None:
    names = (
        (_A1_RESULT, _A1_AUDIT, _A1_RUNTIME, _A1_REPLAY)
        if phase == "a1"
        else (_A2_RESULT, _A2_AUDIT, _A2_RUNTIME, _A2_REPLAY)
    )
    directory = _result_directory(root)
    files = []
    for name in sorted(names):
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("bundle evidence is missing or invalid")
        payload = path.read_bytes()
        if len(payload) > _MAX_BUNDLE_FILE_BYTES:
            raise ValueError("bundle evidence file exceeds the 5 MiB limit")
        files.append(
            {"basename": name, "size_bytes": len(payload), "sha256": _sha256(payload)}
        )
    manifest = {
        "schema": "trustsr.phase2b3a-bundle-manifest.v1",
        "phase": phase,
        "files": files,
    }
    manifest_path = directory / _BUNDLE_MANIFEST
    _require_safe_derived_path(root, manifest_path)
    if manifest_path.is_symlink() or (
        manifest_path.exists() and not manifest_path.is_file()
    ):
        raise ValueError("bundle manifest output must be a regular file")
    atomic_write_bytes(manifest_path, canonical_json(manifest))


def _run_a1_replay(args: argparse.Namespace) -> dict[str, object]:
    context = _preflight_context(args, capture_hardware=False)
    pairs = _load_a1_pairs(context, args)
    directory = _result_directory(context.root)
    result_bytes, result = _read_canonical(directory / _A1_RESULT)
    audit_bytes, audit = _read_canonical(directory / _A1_AUDIT)
    runtime_bytes, _ = _read_canonical(directory / _A1_RUNTIME)
    runtime_sha256 = _verify_runtime_reference(result, runtime_bytes)
    scientific_result = dict(result)
    scientific_result.pop("runtime_manifest_sha256")
    rebuilt_result, rebuilt_audit = replay_a1_smoke(
        pairs,
        scientific_result,
        audit,
        PredictionCache(_prediction_cache_directory(context.root)),
        ScoreCache(_score_cache_directory(context.root)),
    )
    rebuilt_result["runtime_manifest_sha256"] = runtime_sha256
    if (
        canonical_json(rebuilt_result) != result_bytes
        or canonical_json(rebuilt_audit) != audit_bytes
    ):
        raise RuntimeError("A1 replay is not byte-identical to committed evidence")
    receipt = _receipt("a1", result_bytes, audit_bytes, runtime_sha256)
    _commit_identical_or_new(directory / _A1_REPLAY, canonical_json(receipt), context.root)
    _write_bundle_manifest(context.root, "a1")
    return {"stage": "replay", "byte_identical": True}


def run_replay(args: argparse.Namespace) -> dict[str, object]:
    return _run_a1_replay(args)


def _verify_a1_cloud_acceptance(context: _StageContext) -> tuple[bool, str]:
    directory = _result_directory(context.root)
    result_bytes, result = _read_canonical(directory / _A1_RESULT)
    audit_bytes, audit = _read_canonical(directory / _A1_AUDIT)
    runtime_bytes, runtime = _read_canonical(directory / _A1_RUNTIME)
    _, replay = _read_canonical(directory / _A1_REPLAY)
    runtime_sha256 = _verify_runtime_reference(result, runtime_bytes)
    peak = runtime.get("single_peak_memory_bytes")
    total = runtime.get("gpu_total_memory_bytes")
    free = runtime.get("persistent_free_bytes")
    projection = runtime.get("projected_a2_uncached_seconds")
    median = runtime.get("a1_median_uncached_ldsr_prediction_seconds")
    missing = runtime.get("missing_a2_seed_predictions")
    if (
        any(type(value) is not int for value in (peak, total, free, missing))
        or not isinstance(projection, int | float)
        or not isinstance(median, int | float)
        or float(projection) != float(median) * missing
    ):
        raise ValueError("A1 runtime resource projection is invalid")
    calculated_resource_gate = _resource_gate_pass(
        single_peak_memory_bytes=peak,
        gpu_total_memory_bytes=total,
        persistent_free_bytes=free,
        projected_a2_uncached_seconds=float(projection),
    )
    if (
        result.get("sample_count") != 4
        or audit.get("sample_count") != 4
        or replay.get("byte_identical") is not True
        or replay.get("result_sha256") != _sha256(result_bytes)
        or replay.get("cache_audit_sha256") != _sha256(audit_bytes)
        or replay.get("runtime_manifest_sha256") != runtime_sha256
        or runtime.get("git_commit") != context.code_revision
        or runtime.get("single_repeatability_pass") is not True
        or runtime.get("resource_gate_pass") is not calculated_resource_gate
        or calculated_resource_gate is not True
    ):
        raise ValueError("A1 acceptance or resource evidence is invalid")
    include = result.get("include_ldsr_variance_k5")
    stable = result.get("k5_statistically_stable")
    if type(include) is not bool or type(stable) is not bool or include is not stable:
        raise ValueError("A1 acceptance contains an invalid K=5 decision")
    return include, _sha256(canonical_json(replay))


def run_development(args: argparse.Namespace) -> dict[str, object]:
    context = _preflight_context(args, capture_hardware=True)
    include, a1_replay_sha256 = _verify_a1_cloud_acceptance(context)
    ordered_ids = _ordered_development_sample_ids(context.records)
    pairs = _load_a2_pairs(context, args)
    models = _model_factory(args)
    _validate_models(models)
    prediction_cache = PredictionCache(_prediction_cache_directory(context.root))
    score_cache = ScoreCache(_score_cache_directory(context.root))
    seeds = K5A_SEEDS if include else (3407,)
    bundles = _one_shot_bundles(pairs, models, seeds, prediction_cache)
    result, audit = evaluate_a2_development(
        pairs,
        bundles,
        prediction_cache=prediction_cache,
        score_cache=score_cache,
        include_ldsr_variance_k5=include,
        code_revision=context.code_revision,
        ordered_development_sample_ids=ordered_ids,
    )
    runtime = {
        "schema": "trustsr.phase2b3a-a2-runtime.v1",
        "git_commit": context.code_revision,
        "a1_acceptance_pass": True,
        "a1_replay_sha256": a1_replay_sha256,
        "sample_count": 120,
    }
    _, runtime_sha256 = _write_runtime(context.root, _A2_RUNTIME, runtime)
    result = {**result, "runtime_manifest_sha256": runtime_sha256}
    result_bytes, audit_bytes = _commit_pair(
        context.root, (_A2_RESULT, result), (_A2_AUDIT, audit)
    )
    return {
        "stage": "development",
        "sample_count": 120,
        "result_sha256": _sha256(result_bytes),
        "cache_audit_sha256": _sha256(audit_bytes),
    }


def _run_a2_replay(args: argparse.Namespace) -> dict[str, object]:
    context = _preflight_context(args, capture_hardware=False)
    ordered_ids = _ordered_development_sample_ids(context.records)
    pairs = _load_a2_pairs(context, args)
    directory = _result_directory(context.root)
    result_bytes, result = _read_canonical(directory / _A2_RESULT)
    audit_bytes, audit = _read_canonical(directory / _A2_AUDIT)
    runtime_bytes, runtime = _read_canonical(directory / _A2_RUNTIME)
    runtime_sha256 = _verify_runtime_reference(result, runtime_bytes)
    if runtime.get("git_commit") != context.code_revision:
        raise ValueError("A2 runtime commit does not match the reviewed checkout")
    scientific_result = dict(result)
    scientific_result.pop("runtime_manifest_sha256")
    rebuilt_result, rebuilt_audit = replay_a2_development(
        pairs,
        scientific_result,
        audit,
        PredictionCache(_prediction_cache_directory(context.root)),
        ScoreCache(_score_cache_directory(context.root)),
        ordered_development_sample_ids=ordered_ids,
    )
    rebuilt_result["runtime_manifest_sha256"] = runtime_sha256
    if (
        canonical_json(rebuilt_result) != result_bytes
        or canonical_json(rebuilt_audit) != audit_bytes
    ):
        raise RuntimeError("A2 replay is not byte-identical to committed evidence")
    receipt = _receipt("a2", result_bytes, audit_bytes, runtime_sha256)
    _commit_identical_or_new(directory / _A2_REPLAY, canonical_json(receipt), context.root)
    _write_bundle_manifest(context.root, "a2")
    return {"stage": "development-replay", "byte_identical": True}


def run_development_replay(args: argparse.Namespace) -> dict[str, object]:
    return _run_a2_replay(args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with redirect_stdout(sys.stderr):
        result = args.handler(args)
    if not isinstance(result, Mapping):
        raise TypeError("Phase 2B3-A handler must return a mapping")
    print(canonical_json(dict(result)).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
