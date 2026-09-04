"""Metadata-only authority verification for Phase 2B3-C candidate bundles."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import stat
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from trustsr.data.internal_test_subset import load_internal_test_records
from trustsr.evaluation.phase2b3c_access import (
    AccessLedgerSnapshot,
    VerifiedAccessPermit,
    build_phase2b3c_readiness,
    load_access_ledger,
    verify_phase2b3c_access_permit,
)
from trustsr.evaluation.phase2b3c_bundle import (
    BUNDLE_DOCUMENT_SCHEMAS,
    read_phase2b3c_bundle,
    verify_phase2b3c_bundle_documents,
)
from trustsr.evaluation.phase2b3c_evidence import load_frozen_phase2b3b_evidence
from trustsr.evaluation.phase2b3c_preflight import build_phase2b3c_preflight
from trustsr.evaluation.phase2b3c_result_verify import verify_phase2b3c_result
from trustsr.evaluation.phase2b3c_revision import (
    verify_phase2b3c_implementation_revision,
)
from trustsr.evaluation.phase2b3c_runtime import verify_phase2b3c_runtime_manifest
from trustsr.jsonio import canonical_json

SCHEMA = "trustsr.phase2b3c-candidate-bundle-metadata-verification.v1"
VERIFICATION_SCOPE = "metadata_consistency_only"
_RESULT_NAME = "phase2b3c-evaluation-result.json"
_AUDIT_NAME = "phase2b3c-evaluation-cache-audit.json"
_RUNTIME_NAME = "phase2b3c-evaluation-runtime.json"
_REPLAY_NAME = "phase2b3c-evaluation-replay.json"
_LEDGER_NAME = "phase2b3c-access-ledger-snapshot.json"
_PACKAGES = ("numpy", "opensr-model", "rasterio", "torch", "trustsr")
_BUNDLE_VERIFIER_AUTHORITY = object()
_METADATA_AUTHORITY = object()
_MAX_PERMIT_BYTES = 1024**2


def _stat_identity(item: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _digest(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _revision(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase Git revision")
    return value


def _object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an exact JSON object")
    return value


def _capture_dependencies(project_root: Path) -> dict[str, object]:
    lock_payload = (project_root / "uv.lock").read_bytes()
    packages = {
        name: "0.1.0" if name == "trustsr" else importlib.metadata.version(name)
        for name in _PACKAGES
    }
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "uv_lock_sha256": hashlib.sha256(lock_payload).hexdigest(),
        "packages": packages,
    }


def _permit_implementation(path: Path) -> tuple[str, str]:
    if not isinstance(path, Path):
        raise TypeError("access permit candidate path must be a Path")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("access permit candidate is unreadable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= _MAX_PERMIT_BYTES:
            raise ValueError("access permit candidate exceeds the size limit")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise ValueError("access permit candidate ended before its descriptor size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("access permit candidate exceeds its descriptor size")
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise ValueError("access permit candidate changed while being read")
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("access permit candidate is not canonical JSON") from exc
    if type(value) is not dict or canonical_json(value) != payload:
        raise ValueError("access permit candidate is not canonical JSON")
    implementation = _object(value.get("implementation"), "permit implementation")
    return (
        _revision(implementation.get("revision"), "permit implementation revision"),
        _digest(
            implementation.get("computation_tree_sha256"),
            "permit computation tree",
        ),
    )


@dataclass(frozen=True, init=False)
class VerifiedPhase2B3CMetadataAuthority:
    """Opaque permit, ledger, revision, environment, and membership authority."""

    permit: VerifiedAccessPermit
    ledger: AccessLedgerSnapshot
    revision: str
    head_revision: str
    dependencies: dict[str, object]
    records: tuple[Mapping[str, object], ...]
    ordered_sample_ids_sha256: str
    ordered_membership_sha256: str
    _authority: object = field(repr=False, compare=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("metadata authority is created only by the verifier")

    @classmethod
    def _from_verified(cls, **values: object) -> VerifiedPhase2B3CMetadataAuthority:
        public_fields = set(cls.__dataclass_fields__) - {"_authority"}
        if set(values) != public_fields:
            raise TypeError("metadata authority fields are invalid")
        authority = object.__new__(cls)
        for name in public_fields:
            object.__setattr__(authority, name, values[name])
        object.__setattr__(authority, "_authority", _METADATA_AUTHORITY)
        authority.__post_init__()
        return authority

    def __post_init__(self) -> None:
        if (
            getattr(self, "_authority", None) is not _METADATA_AUTHORITY
            or type(self.permit) is not VerifiedAccessPermit
            or type(self.ledger) is not AccessLedgerSnapshot
            or type(self.dependencies) is not dict
            or type(self.records) is not tuple
            or len(self.records) != 120
        ):
            raise ValueError("metadata authority was not issued by the verifier")
        _revision(self.revision, "metadata authority implementation revision")
        _revision(self.head_revision, "metadata authority head revision")
        _digest(self.ordered_sample_ids_sha256, "metadata authority sample IDs")
        _digest(self.ordered_membership_sha256, "metadata authority membership")
        canonical_json(self.dependencies)


@dataclass(frozen=True, init=False)
class VerifiedPhase2B3CBundle:
    """Metadata receipt that cannot prove cache pixels or authorize acceptance."""

    schema: str
    verification_scope: str
    cache_computation_verified: bool
    acceptance_authorized: bool
    manifest_sha256: str
    result_sha256: str
    cache_audit_sha256: str
    runtime_sha256: str
    replay_sha256: str
    ledger_snapshot_sha256: str
    evaluation_id: str
    permit_sha256: str
    producer_revision: str
    ordered_sample_ids_sha256: str
    ordered_membership_sha256: str
    input_receipt_sha256: str
    ordered_inputs_sha256: str
    map_evidence_sha256: str
    model_identity_sha256: str
    bundle_complete_event_sha256: str
    phase_decision: str
    _authority: object = field(repr=False, compare=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("bundle verification receipts are created only by the verifier")

    @classmethod
    def _from_verified(cls, **values: object) -> VerifiedPhase2B3CBundle:
        public_fields = set(cls.__dataclass_fields__) - {"_authority"}
        if set(values) != public_fields:
            raise TypeError("bundle verification receipt fields are invalid")
        receipt = object.__new__(cls)
        for name in public_fields:
            object.__setattr__(receipt, name, values[name])
        object.__setattr__(receipt, "_authority", _BUNDLE_VERIFIER_AUTHORITY)
        receipt.__post_init__()
        return receipt

    def __post_init__(self) -> None:
        if (
            getattr(self, "_authority", None) is not _BUNDLE_VERIFIER_AUTHORITY
            or self.schema != SCHEMA
            or self.verification_scope != VERIFICATION_SCOPE
            or self.cache_computation_verified is not False
            or self.acceptance_authorized is not False
            or self.phase_decision
            not in {"confirmed", "empirically_met_but_inconclusive", "failed"}
        ):
            raise ValueError("bundle verification receipt scope is invalid")
        for name in (
            "manifest_sha256",
            "result_sha256",
            "cache_audit_sha256",
            "runtime_sha256",
            "replay_sha256",
            "ledger_snapshot_sha256",
            "evaluation_id",
            "permit_sha256",
            "ordered_sample_ids_sha256",
            "ordered_membership_sha256",
            "input_receipt_sha256",
            "ordered_inputs_sha256",
            "map_evidence_sha256",
            "model_identity_sha256",
            "bundle_complete_event_sha256",
        ):
            _digest(getattr(self, name), f"bundle receipt {name}")
        _revision(self.producer_revision, "bundle receipt producer revision")


def load_phase2b3c_metadata_authority(
    *,
    project_root: Path,
    evidence_dir: Path,
    storage_root: Path,
    manifest_path: Path,
    access_permit_path: Path,
) -> VerifiedPhase2B3CMetadataAuthority:
    implementation_revision, tree_sha256 = _permit_implementation(access_permit_path)
    revision = verify_phase2b3c_implementation_revision(
        project_root, implementation_revision, tree_sha256
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
    permitted = build_phase2b3c_preflight(evidence, records, revision, permit=permit)
    ledger = load_access_ledger(storage_root, permit)
    return VerifiedPhase2B3CMetadataAuthority._from_verified(
        permit=permit,
        ledger=ledger,
        revision=revision.implementation_revision,
        head_revision=revision.head_revision,
        dependencies=dependencies,
        records=tuple(records),
        ordered_sample_ids_sha256=permitted["evaluation"][
            "ordered_sample_ids_sha256"
        ],
        ordered_membership_sha256=permitted["evaluation"][
            "ordered_membership_sha256"
        ],
    )


def verify_phase2b3c_bundle(
    bundle_dir: Path,
    *,
    project_root: Path,
    evidence_dir: Path,
    storage_root: Path,
    manifest_path: Path,
    access_permit_path: Path,
    authority: VerifiedPhase2B3CMetadataAuthority | None = None,
) -> VerifiedPhase2B3CBundle:
    """Cross-bind all bundle metadata without reading pixels, caches, or a model."""

    loaded = read_phase2b3c_bundle(bundle_dir)
    payloads = dict(loaded.payloads)
    if set(payloads) != set(BUNDLE_DOCUMENT_SCHEMAS):
        raise ValueError("candidate bundle does not contain the exact document set")
    documents = loaded.documents()
    semantic = verify_phase2b3c_bundle_documents(documents)
    result = verify_phase2b3c_result(payloads[_RESULT_NAME])
    runtime = verify_phase2b3c_runtime_manifest(
        payloads[_RUNTIME_NAME], result=payloads[_RESULT_NAME]
    )
    if authority is None:
        authority = load_phase2b3c_metadata_authority(
            project_root=project_root,
            evidence_dir=evidence_dir,
            storage_root=storage_root,
            manifest_path=manifest_path,
            access_permit_path=access_permit_path,
        )
    elif (
        type(authority) is not VerifiedPhase2B3CMetadataAuthority
        or getattr(authority, "_authority", None) is not _METADATA_AUTHORITY
    ):
        raise ValueError("bundle verifier requires authentic metadata authority")
    authority.__post_init__()
    permit = authority.permit
    ledger = authority.ledger
    result_document = documents[_RESULT_NAME]
    result_digests = _object(result_document.get("digests"), "result digests")
    runtime_document = documents[_RUNTIME_NAME]
    if runtime_document.get("dependencies") != authority.dependencies:
        raise ValueError("runtime dependencies differ from the verified environment")
    if (
        semantic.result_sha256 != result.result_sha256
        or semantic.runtime_sha256 != runtime.runtime_sha256
        or semantic.cache_audit_sha256 != result.cache_audit_sha256
        or semantic.phase_decision != result.phase_decision
        or runtime.result_sha256 != result.result_sha256
        or runtime.cache_audit_sha256 != result.cache_audit_sha256
        or runtime.input_receipt_sha256 != result.input_receipt_sha256
        or runtime.map_evidence_sha256 != result.map_evidence_sha256
    ):
        raise ValueError("bundle verifier receipts are not cross-bound")
    if (
        result.evaluation_id != permit.evaluation_id
        or result.permit_sha256 != permit.permit_sha256
        or result.producer_revision != authority.revision
        or result_digests.get("ordered_sample_ids_sha256")
        != authority.ordered_sample_ids_sha256
        or result_digests.get("ordered_membership_sha256")
        != authority.ordered_membership_sha256
    ):
        raise ValueError("bundle result differs from permit or membership authority")
    if (
        ledger.state != "bundle_complete"
        or ledger.sequence != 3
        or ledger.evaluation_id != permit.evaluation_id
        or ledger.permit_sha256 != permit.permit_sha256
        or ledger.previous_event_sha256 != result.ledger_event_sha256
    ):
        raise ValueError("bundle ledger authority is not the exact bundle_complete state")
    return VerifiedPhase2B3CBundle._from_verified(
        schema=SCHEMA,
        verification_scope=VERIFICATION_SCOPE,
        cache_computation_verified=False,
        acceptance_authorized=False,
        manifest_sha256=loaded.manifest_sha256,
        result_sha256=result.result_sha256,
        cache_audit_sha256=semantic.cache_audit_sha256,
        runtime_sha256=semantic.runtime_sha256,
        replay_sha256=semantic.replay_sha256,
        ledger_snapshot_sha256=semantic.ledger_snapshot_sha256,
        evaluation_id=result.evaluation_id,
        permit_sha256=result.permit_sha256,
        producer_revision=result.producer_revision,
        ordered_sample_ids_sha256=result_digests["ordered_sample_ids_sha256"],
        ordered_membership_sha256=result_digests["ordered_membership_sha256"],
        input_receipt_sha256=result.input_receipt_sha256,
        ordered_inputs_sha256=result.ordered_inputs_sha256,
        map_evidence_sha256=result.map_evidence_sha256,
        model_identity_sha256=runtime.model_identity_sha256,
        bundle_complete_event_sha256=ledger.event_sha256,
        phase_decision=result.phase_decision,
    )
