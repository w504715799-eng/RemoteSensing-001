"""Formal one-time evaluation and inference-free replay for Phase 2B3-C."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trustsr.artifacts.predictions import PredictionCache
from trustsr.artifacts.scores import ScoreCache
from trustsr.data.crosssensor_pairs import POST_MANIFEST_SHA256, LoadedCrosssensorPair
from trustsr.data.crosssensor_source import require_cloud_confirmation
from trustsr.data.internal_test_pairs import load_internal_test_pairs
from trustsr.data.internal_test_subset import load_internal_test_records
from trustsr.evaluation.internal_test_cache_audit import build_internal_test_cache_audit
from trustsr.evaluation.internal_test_input_receipt import (
    build_internal_test_input_receipt,
    verify_internal_test_input_receipt,
)
from trustsr.evaluation.internal_test_maps import load_or_compute_internal_test_maps
from trustsr.evaluation.internal_test_metrics import build_internal_test_metrics
from trustsr.evaluation.internal_test_predictions import (
    InternalTestPredictionBundle,
    load_or_generate_internal_test_bundle,
    probe_cached_internal_test_bundles,
)
from trustsr.evaluation.phase2b3c_access import (
    AccessLedgerSnapshot,
    VerifiedAccessPermit,
    advance_access_ledger,
    build_phase2b3c_readiness,
    load_access_ledger,
    verify_phase2b3c_access_permit,
)
from trustsr.evaluation.phase2b3c_bundle import (
    BUNDLE_DOCUMENT_SCHEMAS,
    BundleWriteReceipt,
    build_phase2b3c_ledger_snapshot,
    read_phase2b3c_bundle,
    verify_phase2b3c_bundle_documents,
    write_phase2b3c_bundle,
)
from trustsr.evaluation.phase2b3c_computation_verify import (
    verify_phase2b3c_computation,
)
from trustsr.evaluation.phase2b3c_evidence import load_frozen_phase2b3b_evidence
from trustsr.evaluation.phase2b3c_policy import PHASE2B3C_THRESHOLD
from trustsr.evaluation.phase2b3c_preflight import build_phase2b3c_preflight
from trustsr.evaluation.phase2b3c_replay_receipt import (
    build_phase2b3c_replay_receipt,
)
from trustsr.evaluation.phase2b3c_result import build_phase2b3c_result
from trustsr.evaluation.phase2b3c_revision import (
    VerifiedRevision,
    phase2b3c_computation_tree_sha256,
    verify_phase2b3c_implementation_revision,
)
from trustsr.evaluation.phase2b3c_runtime import build_phase2b3c_runtime_manifest
from trustsr.evaluation.phase2b3c_statistics import build_phase2b3c_statistics
from trustsr.jsonio import canonical_json

_MINIMUM_FREE_BYTES = 10 * 1024**3
_RESULT_NAME = "phase2b3c-evaluation-result.json"
_AUDIT_NAME = "phase2b3c-evaluation-cache-audit.json"
_RUNTIME_NAME = "phase2b3c-evaluation-runtime.json"
_REPLAY_NAME = "phase2b3c-evaluation-replay.json"
_LEDGER_NAME = "phase2b3c-access-ledger-snapshot.json"
_PACKAGES = ("numpy", "opensr-model", "rasterio", "torch", "trustsr")


@dataclass(frozen=True)
class Phase2B3CStoragePaths:
    """Fixed external paths derived from one confirmed persistent root."""

    root: Path
    prediction_cache_dir: Path
    score_cache_dir: Path
    bundle_dir: Path
    lock_path: Path


@dataclass(frozen=True)
class Phase2B3CAuthority:
    """Pre-pixel identities established from Git, metadata, and reviewed permit."""

    revision: VerifiedRevision
    records: tuple[Mapping[str, object], ...]
    permit: VerifiedAccessPermit
    dependencies: dict[str, object]


@dataclass(frozen=True)
class Phase2B3CWorkflowReceipt:
    """Host-free identity emitted after a complete formal workflow stage."""

    stage: str
    sample_count: int
    byte_identical: bool
    manifest_sha256: str
    result_sha256: str
    cache_audit_sha256: str
    runtime_sha256: str
    replay_sha256: str
    ledger_snapshot_sha256: str
    phase_decision: str

    def __post_init__(self) -> None:
        if self.stage not in {"evaluate", "evaluation-replay"}:
            raise ValueError("Phase 2B3-C workflow stage is invalid")
        if self.sample_count != 120 or self.byte_identical is not True:
            raise ValueError("Phase 2B3-C workflow receipt is incomplete")
        for value in (
            self.manifest_sha256,
            self.result_sha256,
            self.cache_audit_sha256,
            self.runtime_sha256,
            self.replay_sha256,
            self.ledger_snapshot_sha256,
        ):
            if type(value) is not str or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("Phase 2B3-C workflow receipt digest is invalid")
        if self.phase_decision not in {
            "confirmed",
            "empirically_met_but_inconclusive",
            "failed",
        }:
            raise ValueError("Phase 2B3-C workflow decision is invalid")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "trustsr.phase2b3c-formal-workflow-cli.v1",
            "stage": self.stage,
            "sample_count": self.sample_count,
            "byte_identical": self.byte_identical,
            "manifest_sha256": self.manifest_sha256,
            "result_sha256": self.result_sha256,
            "cache_audit_sha256": self.cache_audit_sha256,
            "runtime_sha256": self.runtime_sha256,
            "replay_sha256": self.replay_sha256,
            "ledger_snapshot_sha256": self.ledger_snapshot_sha256,
            "phase_decision": self.phase_decision,
        }


def _json_native(value: object) -> object:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("workflow document mapping keys must be strings")
        return {key: _json_native(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_json_native(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise ValueError("workflow document contains a non-JSON value")


def _json_snapshot(value: Mapping[str, object]) -> dict[str, object]:
    result = json.loads(canonical_json(_json_native(value)))
    if type(result) is not dict:
        raise ValueError("workflow document must be a JSON object")
    return result


def phase2b3c_storage_paths(storage_root: Path) -> Phase2B3CStoragePaths:
    """Derive immutable cache, bundle, and lock paths without touching them."""

    if not isinstance(storage_root, Path) or not storage_root.is_absolute():
        raise ValueError("storage root must be an absolute path")
    phase_root = storage_root / "trustsr" / "phase2b3c"
    return Phase2B3CStoragePaths(
        root=storage_root,
        prediction_cache_dir=phase_root / "predictions" / POST_MANIFEST_SHA256,
        score_cache_dir=phase_root / "scores" / POST_MANIFEST_SHA256,
        bundle_dir=phase_root / "bundles" / POST_MANIFEST_SHA256,
        lock_path=phase_root / ".formal.lock",
    )


def _safe_derived_path(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Phase 2B3-C path escapes persistent storage") from exc
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("Phase 2B3-C path contains a symlink")
        if current.exists() and current != path and not current.is_dir():
            raise ValueError("Phase 2B3-C path parent is not a directory")


def validate_phase2b3c_storage(
    storage_root: Path, confirmed_persistent_storage: bool
) -> Phase2B3CStoragePaths:
    """Validate the root and capacity without touching protected cache entries."""

    root = require_cloud_confirmation(storage_root, confirmed_persistent_storage)
    paths = phase2b3c_storage_paths(root)
    for path in (
        paths.prediction_cache_dir,
        paths.score_cache_dir,
        paths.bundle_dir,
        paths.lock_path,
    ):
        _safe_derived_path(root, path)
    if shutil.disk_usage(root).free <= _MINIMUM_FREE_BYTES:
        raise ValueError("Phase 2B3-C requires more than 10 GiB persistent free space")
    return paths


def _ensure_output_parents(paths: Phase2B3CStoragePaths) -> None:
    for directory in (
        paths.prediction_cache_dir,
        paths.score_cache_dir,
        paths.bundle_dir.parent,
    ):
        _safe_derived_path(paths.root, directory)
        directory.mkdir(parents=True, exist_ok=True)
        _safe_derived_path(paths.root, directory)


@contextmanager
def phase2b3c_formal_lock(paths: Phase2B3CStoragePaths) -> Iterable[None]:
    """Serialize all evaluation and replay stages for the fixed storage root."""

    _safe_derived_path(paths.root, paths.lock_path)
    paths.lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        paths.lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Phase 2B3-C formal lock must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another Phase 2B3-C stage holds the formal lock") from exc
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _capture_dependencies(project_root: Path) -> dict[str, object]:
    lock = project_root / "uv.lock"
    payload = lock.read_bytes()
    packages = {}
    for name in _PACKAGES:
        packages[name] = (
            "0.1.0" if name == "trustsr" else importlib.metadata.version(name)
        )
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "uv_lock_sha256": hashlib.sha256(payload).hexdigest(),
        "packages": packages,
    }


def _permit_candidate(path: Path) -> dict[str, object]:
    if not isinstance(path, Path):
        raise TypeError("access permit path must be a Path")
    try:
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("access permit candidate must be a regular non-symlink file")
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError("access permit candidate is unreadable") from exc
    if len(payload) > 1024**2:
        raise ValueError("access permit candidate exceeds the size limit")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("access permit candidate is not canonical JSON") from exc
    if type(value) is not dict or canonical_json(value) != payload:
        raise ValueError("access permit candidate is not canonical JSON")
    return value


def _prepare_authority(
    *,
    project_root: Path,
    evidence_dir: Path,
    storage_root: Path,
    manifest_path: Path,
    access_permit_path: Path,
) -> Phase2B3CAuthority:
    candidate = _permit_candidate(access_permit_path)
    implementation = candidate.get("implementation")
    if type(implementation) is not dict:
        raise ValueError("access permit implementation identity is missing")
    revision = verify_phase2b3c_implementation_revision(
        project_root,
        implementation.get("revision"),
        implementation.get("computation_tree_sha256"),
    )
    evidence = load_frozen_phase2b3b_evidence(evidence_dir, project_root)
    records = load_internal_test_records(storage_root, manifest_path)
    preflight = build_phase2b3c_preflight(evidence, records, revision)
    dependencies = _capture_dependencies(project_root)
    readiness = build_phase2b3c_readiness(
        implementation_revision=revision.implementation_revision,
        computation_tree_sha256=revision.computation_tree_sha256,
        phase2b3b_result_sha256=evidence.result_sha256,
        phase2b3b_cache_audit_sha256=evidence.cache_audit_sha256,
        phase2b3b_acceptance_sha256=evidence.acceptance_sha256,
        ordered_membership_sha256=preflight["evaluation"][
            "ordered_membership_sha256"
        ],
        environment_sha256=hashlib.sha256(canonical_json(dependencies)).hexdigest(),
    )
    permit = verify_phase2b3c_access_permit(
        access_permit_path, project_root, readiness
    )
    build_phase2b3c_preflight(evidence, records, revision, permit=permit)
    return Phase2B3CAuthority(
        revision=revision,
        records=tuple(records),
        permit=permit,
        dependencies=dependencies,
    )


def _current_revision(project_root: Path) -> VerifiedRevision:
    completed = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "--verify", "HEAD^{commit}"],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
        stdin=subprocess.DEVNULL,
        timeout=10.0,
    )
    revision = completed.stdout.rstrip("\n")
    tree = phase2b3c_computation_tree_sha256(project_root, revision)
    return verify_phase2b3c_implementation_revision(project_root, revision, tree)


def run_metadata_preflight(
    *,
    project_root: Path,
    evidence_dir: Path,
    storage_root: Path,
    manifest_path: Path,
    confirmed_persistent_storage: bool,
    access_permit_path: Path | None = None,
) -> dict[str, object]:
    """Produce readiness metadata without permit, ledger, pixels, caches, or models."""

    paths = validate_phase2b3c_storage(storage_root, confirmed_persistent_storage)
    with phase2b3c_formal_lock(paths):
        revision = _current_revision(project_root)
        evidence = load_frozen_phase2b3b_evidence(evidence_dir, project_root)
        records = load_internal_test_records(paths.root, manifest_path)
        preflight = build_phase2b3c_preflight(evidence, records, revision)
        dependencies = _capture_dependencies(project_root)
        readiness = build_phase2b3c_readiness(
            implementation_revision=revision.implementation_revision,
            computation_tree_sha256=revision.computation_tree_sha256,
            phase2b3b_result_sha256=evidence.result_sha256,
            phase2b3b_cache_audit_sha256=evidence.cache_audit_sha256,
            phase2b3b_acceptance_sha256=evidence.acceptance_sha256,
            ordered_membership_sha256=preflight["evaluation"][
                "ordered_membership_sha256"
            ],
            environment_sha256=hashlib.sha256(
                canonical_json(dependencies)
            ).hexdigest(),
        )
        if access_permit_path is not None:
            permit = verify_phase2b3c_access_permit(
                access_permit_path, project_root, readiness
            )
            preflight = build_phase2b3c_preflight(
                evidence, records, revision, permit=permit
            )
    return {
        "schema": "trustsr.phase2b3c-cli-preflight.v1",
        "preflight": _json_snapshot(preflight),
        "readiness": readiness,
    }


def _open_authorized_access(
    paths: Phase2B3CStoragePaths, permit: VerifiedAccessPermit
) -> AccessLedgerSnapshot:
    try:
        current = load_access_ledger(paths.root, permit)
    except ValueError:
        current = advance_access_ledger(paths.root, permit, "reserved")
    if current.state == "reserved":
        current = advance_access_ledger(paths.root, permit, "pixels_opened")
    if current.state in {"invalidated", "accepted"}:
        raise ValueError("Phase 2B3-C access ledger is terminal")
    if current.state not in {"pixels_opened", "caches_complete", "bundle_complete"}:
        raise ValueError("Phase 2B3-C access ledger state is invalid")
    return current


def _load_ldsr(model_dir: Path) -> Any:
    from trustsr.cli.phase2b2b import _require_safe_model_directory
    from trustsr.models.ldsr_s2 import LDSRS2X4

    return LDSRS2X4.from_pretrained(
        _require_safe_model_directory(model_dir), device="cuda:0"
    )


def _radiometry(pairs: tuple[LoadedCrosssensorPair, ...]) -> dict[str, object]:
    result = {}
    for kind in ("lr", "hr"):
        values = [getattr(pair.metadata, f"{kind}_saturation") for pair in pairs]
        result[kind] = {
            "raw_crop_minimum": min(item.raw_crop_minimum for item in values),
            "raw_crop_maximum": max(item.raw_crop_maximum for item in values),
            "clipped_high_count": sum(item.clipped_high_count for item in values),
            "clipped_high_by_band": [
                sum(item.clipped_high_by_band[index] for item in values)
                for index in range(4)
            ],
        }
    return result


def _documents_from_caches(
    *,
    authority: Phase2B3CAuthority,
    pairs: tuple[LoadedCrosssensorPair, ...],
    bundles: tuple[InternalTestPredictionBundle, ...],
    score_cache: ScoreCache,
    input_receipt: dict[str, object],
    cache_snapshot: AccessLedgerSnapshot,
) -> dict[str, dict[str, object]]:
    maps = tuple(
        load_or_compute_internal_test_maps(pair, bundle, score_cache)
        for pair, bundle in zip(pairs, bundles, strict=True)
    )
    metrics = build_internal_test_metrics(
        pairs, maps, threshold=PHASE2B3C_THRESHOLD
    )
    statistics = build_phase2b3c_statistics(metrics.rois)
    audit = _json_snapshot(build_internal_test_cache_audit(bundles, maps))
    verified_input = verify_internal_test_input_receipt(input_receipt)
    result = build_phase2b3c_result(
        statistics=statistics,
        radiometry=_radiometry(pairs),
        evaluation_id=authority.permit.evaluation_id,
        permit_sha256=authority.permit.permit_sha256,
        ledger_event_sha256=cache_snapshot.event_sha256,
        input_receipt_sha256=verified_input.source_sha256,
        ordered_inputs_sha256=verified_input.ordered_inputs_sha256,
        ordered_sample_ids_sha256=verified_input.ordered_sample_ids_sha256,
        ordered_membership_sha256=verified_input.ordered_membership_sha256,
        cache_audit_sha256=hashlib.sha256(canonical_json(audit)).hexdigest(),
        map_evidence_sha256=metrics.map_evidence_sha256,
        producer_revision=authority.revision.implementation_revision,
    )
    runtime = build_phase2b3c_runtime_manifest(
        result=result,
        model_provenance=bundles[0].items[0].identity.model_provenance,
        dependencies=authority.dependencies,
    )
    result_payload = canonical_json(result)
    audit_payload = canonical_json(audit)
    runtime_payload = canonical_json(runtime)
    verify_phase2b3c_computation(
        result_payload,
        audit_payload,
        runtime_payload,
        input_receipt=input_receipt,
        pairs=pairs,
        bundles=bundles,
        dependencies=authority.dependencies,
    )
    replay = build_phase2b3c_replay_receipt(
        result_payload,
        audit_payload,
        runtime_payload,
        rebuilt_result=result_payload,
        rebuilt_cache_audit=audit_payload,
    )
    if cache_snapshot.previous_event_sha256 is None:
        raise ValueError("caches_complete ledger event lacks its predecessor")
    ledger = build_phase2b3c_ledger_snapshot(
        evaluation_id=authority.permit.evaluation_id,
        permit_sha256=authority.permit.permit_sha256,
        state=cache_snapshot.state,
        sequence=cache_snapshot.sequence,
        event_sha256=cache_snapshot.event_sha256,
        previous_event_sha256=cache_snapshot.previous_event_sha256,
    )
    documents = dict(
        zip(
            BUNDLE_DOCUMENT_SCHEMAS,
            (result, audit, runtime, replay, ledger),
            strict=True,
        )
    )
    verify_phase2b3c_bundle_documents(documents)
    return documents


def _workflow_receipt(
    stage: str,
    written: BundleWriteReceipt,
    result: Mapping[str, object],
) -> Phase2B3CWorkflowReceipt:
    digests = dict(written.file_sha256s)
    return Phase2B3CWorkflowReceipt(
        stage=stage,
        sample_count=120,
        byte_identical=True,
        manifest_sha256=written.manifest_sha256,
        result_sha256=digests[_RESULT_NAME],
        cache_audit_sha256=digests[_AUDIT_NAME],
        runtime_sha256=digests[_RUNTIME_NAME],
        replay_sha256=digests[_REPLAY_NAME],
        ledger_snapshot_sha256=digests[_LEDGER_NAME],
        phase_decision=str(result["phase_decision"]),
    )


def run_formal_evaluation(
    *,
    project_root: Path,
    evidence_dir: Path,
    storage_root: Path,
    manifest_path: Path,
    access_permit_path: Path,
    ldsr_model_dir: Path | None,
    confirmed_persistent_storage: bool,
) -> Phase2B3CWorkflowReceipt:
    """Run the sole permitted evaluation and immediate inference-free reconstruction."""

    paths = validate_phase2b3c_storage(storage_root, confirmed_persistent_storage)
    with phase2b3c_formal_lock(paths):
        authority = _prepare_authority(
            project_root=project_root,
            evidence_dir=evidence_dir,
            storage_root=paths.root,
            manifest_path=manifest_path,
            access_permit_path=access_permit_path,
        )
        access = _open_authorized_access(paths, authority.permit)
        if access.state == "bundle_complete":
            raise ValueError("evaluation bundle already exists; use evaluation-replay")
        pairs = load_internal_test_pairs(
            paths.root, authority.records, access.access_guard()
        )
        input_receipt = build_internal_test_input_receipt(authority.records, pairs)
        _ensure_output_parents(paths)
        prediction_cache = PredictionCache(paths.prediction_cache_dir)
        score_cache = ScoreCache(paths.score_cache_dir)
        probe = probe_cached_internal_test_bundles(pairs, cache=prediction_cache)
        if probe.bundles is None:
            if ldsr_model_dir is None:
                raise RuntimeError(
                    f"verified K5 cache has {probe.present_count}/600 entries and "
                    f"{probe.missing_count} missing; GPU authorization is required before "
                    "supplying --ldsr-model-dir"
                )
            ldsr = _load_ldsr(ldsr_model_dir)
            tuple(
                load_or_generate_internal_test_bundle(
                    pair, ldsr=ldsr, cache=prediction_cache
                )
                for pair in pairs
            )
            probe = probe_cached_internal_test_bundles(pairs, cache=prediction_cache)
            if probe.bundles is None:
                raise RuntimeError("prediction cache remains incomplete after authorized inference")
        bundles = probe.bundles
        cache_snapshot = (
            access
            if access.state == "caches_complete"
            else advance_access_ledger(paths.root, authority.permit, "caches_complete")
        )
        documents = _documents_from_caches(
            authority=authority,
            pairs=pairs,
            bundles=bundles,
            score_cache=score_cache,
            input_receipt=input_receipt,
            cache_snapshot=cache_snapshot,
        )
        written = write_phase2b3c_bundle(
            paths.bundle_dir,
            result=documents[_RESULT_NAME],
            cache_audit=documents[_AUDIT_NAME],
            runtime=documents[_RUNTIME_NAME],
            replay=documents[_REPLAY_NAME],
            ledger_snapshot=documents[_LEDGER_NAME],
        )
        advance_access_ledger(paths.root, authority.permit, "bundle_complete")
        return _workflow_receipt("evaluate", written, documents[_RESULT_NAME])


def run_formal_evaluation_replay(
    *,
    project_root: Path,
    evidence_dir: Path,
    storage_root: Path,
    manifest_path: Path,
    access_permit_path: Path,
    confirmed_persistent_storage: bool,
) -> Phase2B3CWorkflowReceipt:
    """Replay the existing bundle from verified caches without importing LDSR."""

    paths = validate_phase2b3c_storage(storage_root, confirmed_persistent_storage)
    with phase2b3c_formal_lock(paths):
        authority = _prepare_authority(
            project_root=project_root,
            evidence_dir=evidence_dir,
            storage_root=paths.root,
            manifest_path=manifest_path,
            access_permit_path=access_permit_path,
        )
        access = _open_authorized_access(paths, authority.permit)
        if access.state != "bundle_complete":
            raise ValueError("evaluation-replay requires the bundle_complete ledger state")
        pairs = load_internal_test_pairs(
            paths.root, authority.records, access.access_guard()
        )
        input_receipt = build_internal_test_input_receipt(authority.records, pairs)
        probe = probe_cached_internal_test_bundles(
            pairs, cache=PredictionCache(paths.prediction_cache_dir)
        )
        if probe.bundles is None:
            raise RuntimeError(
                f"evaluation-replay cache has {probe.present_count}/600 entries and "
                f"{probe.missing_count} missing; inference is forbidden"
            )
        loaded = read_phase2b3c_bundle(paths.bundle_dir)
        documents = loaded.documents()
        verify_phase2b3c_bundle_documents(documents)
        verify_phase2b3c_computation(
            dict(loaded.payloads)[_RESULT_NAME],
            dict(loaded.payloads)[_AUDIT_NAME],
            dict(loaded.payloads)[_RUNTIME_NAME],
            input_receipt=input_receipt,
            pairs=pairs,
            bundles=probe.bundles,
            dependencies=authority.dependencies,
        )
        written = write_phase2b3c_bundle(
            paths.bundle_dir,
            result=documents[_RESULT_NAME],
            cache_audit=documents[_AUDIT_NAME],
            runtime=documents[_RUNTIME_NAME],
            replay=documents[_REPLAY_NAME],
            ledger_snapshot=documents[_LEDGER_NAME],
        )
        return _workflow_receipt(
            "evaluation-replay", written, documents[_RESULT_NAME]
        )
