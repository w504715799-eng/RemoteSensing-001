"""Formal calibration and inference-free replay orchestration for Phase 2B3-B."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import stat
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from trustsr.artifacts.predictions import PredictionCache, tensor_sha256
from trustsr.artifacts.scores import ScoreCache
from trustsr.data.calibration_pairs import load_calibration_pairs
from trustsr.data.calibration_subset import load_calibration_records
from trustsr.data.crosssensor_pairs import LoadedCrosssensorPair
from trustsr.data.crosssensor_source import require_cloud_confirmation
from trustsr.evaluation.calibration_cache_audit import build_calibration_cache_audit
from trustsr.evaluation.calibration_cache_replay import replay_calibration_caches
from trustsr.evaluation.calibration_fit import fit_calibration_maps
from trustsr.evaluation.calibration_input_receipt import build_calibration_input_receipt
from trustsr.evaluation.calibration_maps import load_or_compute_calibration_maps
from trustsr.evaluation.calibration_predictions import load_or_generate_calibration_bundle
from trustsr.evaluation.calibration_radiometry import build_calibration_radiometry
from trustsr.evaluation.calibration_replay_receipt import build_calibration_replay_receipt
from trustsr.evaluation.phase2b3b_bundle import (
    BundleWriteReceipt,
    read_phase2b3b_bundle,
    write_phase2b3b_bundle,
)
from trustsr.evaluation.phase2b3b_evidence import POST_MANIFEST_SHA256
from trustsr.evaluation.phase2b3b_policy import (
    APPROVED_ALPHA,
    APPROVED_MINIMUM_COVERAGE,
)
from trustsr.evaluation.phase2b3b_preflight import load_phase2b3b_preflight
from trustsr.evaluation.phase2b3b_result import build_phase2b3b_result
from trustsr.evaluation.phase2b3b_revision import (
    Phase2B3BRevision,
    verify_phase2b3b_revision,
    verify_recorded_phase2b3b_revision,
)
from trustsr.evaluation.phase2b3b_runtime import build_phase2b3b_runtime_manifest
from trustsr.jsonio import canonical_json
from trustsr.risk.local import ensemble_variance_score

_MINIMUM_FREE_BYTES = 10 * 1024**3
_RESULT_NAME = "phase2b3b-calibration-result.json"
_AUDIT_NAME = "phase2b3b-calibration-cache-audit.json"
_RUNTIME_NAME = "phase2b3b-calibration-runtime.json"
_REPLAY_NAME = "phase2b3b-calibration-replay.json"


def _json_snapshot(value: Mapping[str, object]) -> dict[str, object]:
    snapshot = json.loads(canonical_json(value))
    if type(snapshot) is not dict:
        raise ValueError("formal workflow document must be a JSON object")
    return snapshot


@dataclass(frozen=True)
class Phase2B3BStoragePaths:
    """Fixed external paths derived from one canonical persistent root."""

    root: Path
    prediction_cache_dir: Path
    score_cache_dir: Path
    bundle_dir: Path
    lock_path: Path


@dataclass(frozen=True)
class Phase2B3BWorkflowReceipt:
    """Host-free identity emitted after a complete formal workflow stage."""

    stage: str
    sample_count: int
    byte_identical: bool
    manifest_sha256: str
    result_sha256: str
    cache_audit_sha256: str
    runtime_manifest_sha256: str
    replay_sha256: str
    phase_decision: str

    def __post_init__(self) -> None:
        if self.stage not in {"calibration", "calibration-replay"}:
            raise ValueError("formal workflow stage is invalid")
        if self.sample_count != 120 or self.byte_identical is not True:
            raise ValueError("formal workflow receipt is incomplete")
        for digest in (
            self.manifest_sha256,
            self.result_sha256,
            self.cache_audit_sha256,
            self.runtime_manifest_sha256,
            self.replay_sha256,
        ):
            if type(digest) is not str or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("formal workflow receipt digest is invalid")
        if self.phase_decision not in {
            "freeze_calibration",
            "stop_insufficient_coverage",
        }:
            raise ValueError("formal workflow phase decision is invalid")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "trustsr.phase2b3b-formal-workflow-cli.v1",
            "stage": self.stage,
            "sample_count": self.sample_count,
            "byte_identical": self.byte_identical,
            "manifest_sha256": self.manifest_sha256,
            "result_sha256": self.result_sha256,
            "cache_audit_sha256": self.cache_audit_sha256,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "replay_sha256": self.replay_sha256,
            "phase_decision": self.phase_decision,
        }


def _safe_derived_path(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Phase 2B3-B path escapes the persistent storage root") from exc
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("Phase 2B3-B path must not contain symlink components")
        if current.exists() and current != path and not current.is_dir():
            raise ValueError("Phase 2B3-B path parent must be a directory")


def phase2b3b_storage_paths(storage_root: Path) -> Phase2B3BStoragePaths:
    """Derive the immutable Phase 2B3-B cache, bundle, and lock paths."""

    if not isinstance(storage_root, Path) or not storage_root.is_absolute():
        raise ValueError("storage root must be an absolute path")
    phase_root = storage_root / "trustsr" / "phase2b3b"
    return Phase2B3BStoragePaths(
        root=storage_root,
        prediction_cache_dir=phase_root / "predictions" / POST_MANIFEST_SHA256,
        score_cache_dir=phase_root / "scores" / POST_MANIFEST_SHA256,
        bundle_dir=phase_root / "bundles" / POST_MANIFEST_SHA256,
        lock_path=phase_root / ".formal.lock",
    )


def validate_phase2b3b_storage(
    storage_root: Path, confirmed_persistent_storage: bool
) -> Phase2B3BStoragePaths:
    """Validate the external root and capacity without creating stage output."""

    root = require_cloud_confirmation(storage_root, confirmed_persistent_storage)
    paths = phase2b3b_storage_paths(root)
    for path in (
        paths.prediction_cache_dir,
        paths.score_cache_dir,
        paths.bundle_dir,
        paths.lock_path,
    ):
        _safe_derived_path(root, path)
    if shutil.disk_usage(root).free <= _MINIMUM_FREE_BYTES:
        raise ValueError("Phase 2B3-B requires more than 10 GiB persistent free space")
    return paths


def _ensure_output_parents(paths: Phase2B3BStoragePaths) -> None:
    for directory in (
        paths.prediction_cache_dir,
        paths.score_cache_dir,
        paths.bundle_dir.parent,
    ):
        _safe_derived_path(paths.root, directory)
        if directory.exists() and not directory.is_dir():
            raise ValueError("Phase 2B3-B output directory is invalid")
        directory.mkdir(parents=True, exist_ok=True)
        _safe_derived_path(paths.root, directory)


@contextmanager
def _formal_lock(paths: Phase2B3BStoragePaths) -> Iterable[None]:
    _safe_derived_path(paths.root, paths.lock_path)
    paths.lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(paths.lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Phase 2B3-B formal lock must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another Phase 2B3-B formal stage holds the lock") from exc
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _load_ldsr(model_dir: Path) -> Any:
    from trustsr.cli.phase2b2b import _require_safe_model_directory
    from trustsr.models.ldsr_s2 import LDSRS2X4

    return LDSRS2X4.from_pretrained(
        _require_safe_model_directory(model_dir), device="cuda:0"
    )


def _load_authoritative_inputs(
    *,
    project_root: Path,
    evidence_dir: Path,
    storage_root: Path,
    manifest_path: Path,
) -> tuple[
    Phase2B3BRevision,
    Mapping[str, object],
    Sequence[Mapping[str, object]],
    tuple[LoadedCrosssensorPair, ...],
]:
    revision = verify_phase2b3b_revision(project_root)
    preflight = load_phase2b3b_preflight(evidence_dir, storage_root, manifest_path)
    records = load_calibration_records(storage_root, manifest_path)
    pairs = load_calibration_pairs(storage_root, records)
    return revision, preflight, records, pairs


def _rebuild_from_caches(
    *,
    audit: Mapping[str, object],
    pairs: Sequence[LoadedCrosssensorPair],
    prediction_cache: PredictionCache,
    score_cache: ScoreCache,
    preflight: Mapping[str, object],
    input_receipt: Mapping[str, object],
    radiometry: Mapping[str, object],
    revision: Phase2B3BRevision,
) -> tuple[dict[str, object], dict[str, object]]:
    replayed = replay_calibration_caches(audit, pairs, prediction_cache, score_cache)
    for bundle, maps in zip(replayed.bundles, replayed.maps, strict=True):
        samples = torch.stack([item.tensor for item in bundle.items], dim=0)
        recomputed_score = ensemble_variance_score(samples)
        if (
            tensor_sha256(recomputed_score) != maps.score.score_sha256
            or not torch.equal(recomputed_score, maps.score.tensor)
        ):
            raise ValueError("recomputed ensemble score differs from the verified score cache")
    rebuilt_audit = _json_snapshot(
        build_calibration_cache_audit(replayed.bundles, replayed.maps)
    )
    fit = fit_calibration_maps(
        replayed.maps,
        alpha=APPROVED_ALPHA,
        minimum_coverage=APPROVED_MINIMUM_COVERAGE,
    )
    rebuilt_result = build_phase2b3b_result(
        preflight,
        input_receipt,
        fit,
        rebuilt_audit,
        radiometry,
        revision,
    )
    return rebuilt_result, rebuilt_audit


def _workflow_receipt(
    stage: str,
    write_receipt: BundleWriteReceipt,
    result: Mapping[str, object],
) -> Phase2B3BWorkflowReceipt:
    if type(write_receipt) is not BundleWriteReceipt:
        raise TypeError("formal workflow requires an atomic bundle write receipt")
    digests = dict(write_receipt.file_sha256s)
    return Phase2B3BWorkflowReceipt(
        stage=stage,
        sample_count=120,
        byte_identical=True,
        manifest_sha256=write_receipt.manifest_sha256,
        result_sha256=digests[_RESULT_NAME],
        cache_audit_sha256=digests[_AUDIT_NAME],
        runtime_manifest_sha256=digests[_RUNTIME_NAME],
        replay_sha256=digests[_REPLAY_NAME],
        phase_decision=str(result["phase_decision"]),
    )


def run_formal_calibration(
    *,
    project_root: Path,
    evidence_dir: Path,
    storage_root: Path,
    manifest_path: Path,
    ldsr_model_dir: Path,
    confirmed_persistent_storage: bool,
) -> Phase2B3BWorkflowReceipt:
    """Run the sole formal K5 calibration and immediate inference-free replay."""

    paths = validate_phase2b3b_storage(storage_root, confirmed_persistent_storage)
    with _formal_lock(paths):
        _ensure_output_parents(paths)
        revision, preflight, records, pairs = _load_authoritative_inputs(
            project_root=project_root,
            evidence_dir=evidence_dir,
            storage_root=paths.root,
            manifest_path=manifest_path,
        )
        input_receipt = build_calibration_input_receipt(records, pairs, preflight)
        radiometry = build_calibration_radiometry(pairs)
        ldsr = _load_ldsr(ldsr_model_dir)
        prediction_cache = PredictionCache(paths.prediction_cache_dir)
        score_cache = ScoreCache(paths.score_cache_dir)
        bundles = tuple(
            load_or_generate_calibration_bundle(pair, ldsr=ldsr, cache=prediction_cache)
            for pair in pairs
        )
        maps = tuple(
            load_or_compute_calibration_maps(pair, bundle, score_cache)
            for pair, bundle in zip(pairs, bundles, strict=True)
        )
        audit = _json_snapshot(build_calibration_cache_audit(bundles, maps))
        fit = fit_calibration_maps(
            maps,
            alpha=APPROVED_ALPHA,
            minimum_coverage=APPROVED_MINIMUM_COVERAGE,
        )
        result = build_phase2b3b_result(
            preflight,
            input_receipt,
            fit,
            audit,
            radiometry,
            revision,
        )
        runtime = build_phase2b3b_runtime_manifest(
            result,
            audit,
            input_receipt,
            project_root=project_root,
            evidence_dir=evidence_dir,
            storage_root=paths.root,
            manifest_path=manifest_path,
        )
        rebuilt_result, rebuilt_audit = _rebuild_from_caches(
            audit=audit,
            pairs=pairs,
            prediction_cache=prediction_cache,
            score_cache=score_cache,
            preflight=preflight,
            input_receipt=input_receipt,
            radiometry=radiometry,
            revision=revision,
        )
        replay = build_calibration_replay_receipt(
            canonical_json(result),
            canonical_json(audit),
            canonical_json(runtime),
            rebuilt_result,
            rebuilt_audit,
        )
        written = write_phase2b3b_bundle(
            paths.bundle_dir,
            result=result,
            cache_audit=audit,
            runtime=runtime,
            replay=replay,
        )
        return _workflow_receipt("calibration", written, result)


def run_formal_calibration_replay(
    *,
    project_root: Path,
    evidence_dir: Path,
    storage_root: Path,
    manifest_path: Path,
    confirmed_persistent_storage: bool,
) -> Phase2B3BWorkflowReceipt:
    """Rebuild committed calibration evidence from caches without constructing LDSR."""

    paths = validate_phase2b3b_storage(storage_root, confirmed_persistent_storage)
    with _formal_lock(paths):
        _ensure_output_parents(paths)
        current_revision, preflight, records, pairs = _load_authoritative_inputs(
            project_root=project_root,
            evidence_dir=evidence_dir,
            storage_root=paths.root,
            manifest_path=manifest_path,
        )
        loaded = read_phase2b3b_bundle(paths.bundle_dir)
        documents = loaded.documents()
        payloads = dict(loaded.payloads)
        result = documents[_RESULT_NAME]
        audit = documents[_AUDIT_NAME]
        runtime = documents[_RUNTIME_NAME]
        producer_revision = result.get("producer_revision")
        if type(producer_revision) is not str:
            raise ValueError("committed result producer revision is invalid")
        verify_recorded_phase2b3b_revision(project_root, producer_revision)
        recorded_revision = replace(current_revision, head_revision=producer_revision)
        input_receipt = build_calibration_input_receipt(records, pairs, preflight)
        radiometry = build_calibration_radiometry(pairs)
        rebuilt_result, rebuilt_audit = _rebuild_from_caches(
            audit=audit,
            pairs=pairs,
            prediction_cache=PredictionCache(paths.prediction_cache_dir),
            score_cache=ScoreCache(paths.score_cache_dir),
            preflight=preflight,
            input_receipt=input_receipt,
            radiometry=radiometry,
            revision=recorded_revision,
        )
        replay = build_calibration_replay_receipt(
            payloads[_RESULT_NAME],
            payloads[_AUDIT_NAME],
            payloads[_RUNTIME_NAME],
            rebuilt_result,
            rebuilt_audit,
        )
        if canonical_json(replay) != payloads[_REPLAY_NAME]:
            raise ValueError("rebuilt replay receipt is not byte-identical")
        written = write_phase2b3b_bundle(
            paths.bundle_dir,
            result=rebuilt_result,
            cache_audit=rebuilt_audit,
            runtime=runtime,
            replay=replay,
        )
        return _workflow_receipt("calibration-replay", written, rebuilt_result)
