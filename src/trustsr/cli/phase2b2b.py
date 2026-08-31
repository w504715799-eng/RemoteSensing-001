"""Run staged Phase 2B2-B development-only three-model smoke checks."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from trustsr.artifacts.gpu_run import (
    _atomic_json,
    capture_gpu_hardware,
    collect_gpu_environment,
)
from trustsr.data.crosssensor_pairs import (
    POST_MANIFEST_SHA256,
    LoadedCrosssensorPair,
    load_crosssensor_pair,
    load_crosssensor_records,
    select_development_smoke_records,
)
from trustsr.data.crosssensor_source import require_cloud_confirmation
from trustsr.data.input_audit import load_input_audit
from trustsr.evaluation.crosssensor_smoke import (
    INPUT_AUDIT_SHA256,
    MODEL_NAMES,
    evaluate_development_smoke,
    replay_development_smoke,
)
from trustsr.jsonio import atomic_write_bytes, canonical_json
from trustsr.models.protocols import SRModel

_RESULT_NAME = "development-three-model-smoke.json"
_AUDIT_NAME = "development-cache-audit.json"
_SINGLE_NAME = "single-result.json"


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest-sha256", required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--input-audit-sha256", required=True)
    parser.add_argument("--sen2srlite-model-dir", type=Path, required=True)
    parser.add_argument("--ldsr-model-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--confirm-cloud-storage", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name, handler in (
        ("preflight", run_preflight),
        ("single", run_single),
        ("smoke", run_smoke),
        ("replay", run_replay),
    ):
        child = subparsers.add_parser(name)
        _add_arguments(child)
        child.set_defaults(handler=handler)
    return parser


def _validate_upstream(
    args: argparse.Namespace,
) -> tuple[Path, tuple[dict[str, object], ...]]:
    root = require_cloud_confirmation(
        Path(args.storage_root), bool(args.confirm_cloud_storage)
    )
    if args.selection_manifest_sha256 != POST_MANIFEST_SHA256:
        raise ValueError("expected the frozen post-manifest SHA-256")
    if args.input_audit_sha256 != INPUT_AUDIT_SHA256:
        raise ValueError("expected the frozen Phase 2B2-A input audit SHA-256")
    records = load_crosssensor_records(
        root,
        Path(args.selection_manifest),
        expected_sha256=args.selection_manifest_sha256,
    )
    expected_audit = (
        root
        / "trustsr"
        / "phase2b2a"
        / "input-audits"
        / POST_MANIFEST_SHA256
        / "phase2b2a-input-audit.json"
    )
    actual_audit = Path(args.input_audit)
    if actual_audit != expected_audit:
        raise ValueError("input audit must use the frozen digest-addressed path")
    current = root
    for component in actual_audit.relative_to(root).parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("input audit digest-addressed path must not contain symlinks")
    load_input_audit(actual_audit, expected_sha256=args.input_audit_sha256)
    return root, records


def _load_development_pairs(
    args: argparse.Namespace, *, limit: int | None = None
) -> tuple[LoadedCrosssensorPair, ...]:
    root, records = _validate_upstream(args)
    selected = select_development_smoke_records(records)
    if limit is not None:
        if limit != 1:
            raise ValueError("only the canonical single-sample gate may limit inputs")
        selected = selected[:1]
    return tuple(
        load_crosssensor_pair(
            root,
            record,
            manifest_sha256=args.selection_manifest_sha256,
        )
        for record in selected
    )


def _phase_root(root: Path) -> Path:
    return root / "trustsr" / "phase2b2b"


def _result_directory(root: Path) -> Path:
    return _phase_root(root) / "results" / POST_MANIFEST_SHA256


def _cache_directory(root: Path) -> Path:
    return _phase_root(root) / "predictions" / POST_MANIFEST_SHA256


def _require_safe_derived_path(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Phase 2B2-B output path escapes storage root") from exc
    current = root
    for component in path.relative_to(root).parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("Phase 2B2-B output path must not contain symlinks")
        if current.exists() and not current.is_dir() and current != path:
            raise ValueError("Phase 2B2-B output parent must be a directory")


def _require_safe_output_directories(root: Path) -> None:
    for directory in (_result_directory(root), _cache_directory(root)):
        _require_safe_derived_path(root, directory)
        if directory.exists() and not directory.is_dir():
            raise ValueError("Phase 2B2-B output directory must be a directory")


def _commit_identical_or_new(path: Path, payload: bytes, root: Path) -> bool:
    _require_safe_derived_path(root, path)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError("existing deterministic output must be a regular file")
        if path.read_bytes() != payload:
            raise ValueError("existing deterministic output has different bytes")
        return True
    atomic_write_bytes(path, payload)
    return False


def _model_factory(args: argparse.Namespace) -> tuple[SRModel, ...]:
    from trustsr.models.bicubic import BicubicX4
    from trustsr.models.ldsr_s2 import LDSRS2X4
    from trustsr.models.sen2srlite import SEN2SRLiteX4

    return (
        BicubicX4(),
        SEN2SRLiteX4.from_pretrained(Path(args.sen2srlite_model_dir), device="cpu"),
        LDSRS2X4.from_pretrained(Path(args.ldsr_model_dir), device="cuda:0"),
    )


def _model_records(models: Sequence[SRModel]) -> list[dict[str, object]]:
    if tuple(model.name for model in models) != MODEL_NAMES:
        raise ValueError("models must use the frozen Phase 2B2-B order")
    result = []
    for model in models:
        provenance = model.provenance()
        if model.scale != 4 or provenance.get("name") != model.name or provenance.get(
            "scale"
        ) != 4:
            raise ValueError("model provenance does not match the frozen adapter")
        result.append({"name": model.name, "provenance": provenance})
    return result


def _runtime_measurement() -> dict[str, object]:
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    return {"peak_memory_bytes": peak}


def run_preflight(args: argparse.Namespace) -> dict[str, object]:
    root, _ = _validate_upstream(args)
    _require_safe_output_directories(root)
    snapshot = capture_gpu_hardware()
    models = _model_factory(args)
    model_records = _model_records(models)
    environment = dict(
        collect_gpu_environment(
            hardware_snapshot=snapshot,
            project_root=getattr(args, "project_root", None),
        )
    )
    environment["models"] = model_records
    path = _result_directory(root) / "preflight-runtime.json"
    _require_safe_derived_path(root, path)
    _atomic_json(path, environment)
    return {"stage": "preflight", "model_count": 3}


def _run_compute(
    args: argparse.Namespace, *, expected_sample_count: int
) -> tuple[dict[str, object], dict[str, object], dict[str, object], Path]:
    root, _ = _validate_upstream(args)
    _require_safe_output_directories(root)
    capture_gpu_hardware()
    pairs = _load_development_pairs(
        args, limit=1 if expected_sample_count == 1 else None
    )
    models = _model_factory(args)
    _model_records(models)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result, audit = evaluate_development_smoke(
        pairs,
        models,
        _cache_directory(root),
        expected_sample_count=expected_sample_count,
    )
    runtime = _runtime_measurement()
    runtime["duration_seconds"] = time.monotonic() - started
    return result, audit, runtime, root


def run_single(args: argparse.Namespace) -> dict[str, object]:
    result, _, runtime, root = _run_compute(args, expected_sample_count=1)
    _commit_identical_or_new(
        _result_directory(root) / _SINGLE_NAME,
        canonical_json(result),
        root,
    )
    _atomic_json(_result_directory(root) / "single-runtime.json", runtime)
    return result


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    result, audit, runtime, root = _run_compute(args, expected_sample_count=4)
    result_payload = canonical_json(result)
    audit_payload = canonical_json(audit)
    result_path = _result_directory(root) / _RESULT_NAME
    audit_path = _result_directory(root) / _AUDIT_NAME
    _require_safe_derived_path(root, result_path)
    _require_safe_derived_path(root, audit_path)
    _commit_identical_or_new(result_path, result_payload, root)
    _commit_identical_or_new(audit_path, audit_payload, root)
    _atomic_json(_result_directory(root) / "smoke-runtime.json", runtime)
    return result


def _read_canonical_object(path: Path) -> tuple[bytes, dict[str, object]]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read committed deterministic result: {path.name}") from exc
    if not isinstance(payload, dict) or canonical_json(payload) != raw:
        raise ValueError(f"committed deterministic result is not canonical: {path.name}")
    return raw, payload


def run_replay(args: argparse.Namespace) -> dict[str, object]:
    root, _ = _validate_upstream(args)
    _require_safe_output_directories(root)
    pairs = _load_development_pairs(args)
    result_bytes, result = _read_canonical_object(_result_directory(root) / _RESULT_NAME)
    audit_bytes, audit = _read_canonical_object(_result_directory(root) / _AUDIT_NAME)
    rebuilt_result, rebuilt_audit = replay_development_smoke(
        pairs,
        result,
        audit,
        _cache_directory(root),
    )
    if canonical_json(rebuilt_result) != result_bytes or canonical_json(
        rebuilt_audit
    ) != audit_bytes:
        raise RuntimeError("replay is not byte-identical to committed evidence")
    return {"stage": "replay", "prediction_count": 12, "byte_identical": True}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    if not isinstance(result, Mapping):
        raise TypeError("Phase 2B2-B handler must return a mapping")
    print(canonical_json(dict(result)).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
