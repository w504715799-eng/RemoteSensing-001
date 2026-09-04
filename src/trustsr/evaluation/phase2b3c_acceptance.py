"""Independent Phase 2B3-C acceptance and Git-safe transaction publication."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import shutil
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from trustsr.artifacts.predictions import PredictionCache
from trustsr.data.internal_test_pairs import load_internal_test_pairs
from trustsr.evaluation.internal_test_input_receipt import (
    build_internal_test_input_receipt,
)
from trustsr.evaluation.internal_test_predictions import (
    InternalTestPredictionCacheProbe,
    probe_cached_internal_test_bundles,
)
from trustsr.evaluation.phase2b3c_access import advance_access_ledger
from trustsr.evaluation.phase2b3c_bundle import (
    LoadedPhase2B3CBundle,
    read_phase2b3c_bundle,
)
from trustsr.evaluation.phase2b3c_bundle_verify import (
    VerifiedPhase2B3CBundle,
    load_phase2b3c_metadata_authority,
    verify_phase2b3c_bundle,
)
from trustsr.evaluation.phase2b3c_computation_verify import (
    VerifiedPhase2B3CComputation,
    verify_phase2b3c_computation,
)
from trustsr.evaluation.phase2b3c_policy import (
    PHASE2B3C_ALPHA,
    PHASE2B3C_MINIMUM_COVERAGE,
    PHASE2B3C_RISK_UPPER_BOUND,
)
from trustsr.evaluation.phase2b3c_replay_receipt import (
    verify_phase2b3c_replay_receipt,
)
from trustsr.evaluation.phase2b3c_result_verify import verify_phase2b3c_result
from trustsr.evaluation.phase2b3c_workflow import (
    Phase2B3CStoragePaths,
    phase2b3c_formal_lock,
    phase2b3c_storage_paths,
    validate_phase2b3c_storage,
)
from trustsr.jsonio import canonical_json

ACCEPTANCE_SCHEMA = "trustsr.phase2b3c-evaluation-acceptance.v1"
VERIFICATION_SCOPE = "independent_internal_test_acceptance"
RESULT_PUBLICATION_NAME = "sen2naipv2-internal-test-evaluation-v1.json"
AUDIT_PUBLICATION_NAME = (
    "sen2naipv2-internal-test-evaluation-cache-audit-v1.json"
)
ACCEPTANCE_PUBLICATION_NAME = (
    "sen2naipv2-internal-test-evaluation-acceptance-v1.json"
)
_PERMIT_PUBLICATION_NAME = "sen2naipv2-internal-test-access-authorization-v1.json"
_PUBLICATION_NAMES = (
    RESULT_PUBLICATION_NAME,
    AUDIT_PUBLICATION_NAME,
    ACCEPTANCE_PUBLICATION_NAME,
)
_RESULT_NAME = "phase2b3c-evaluation-result.json"
_AUDIT_NAME = "phase2b3c-evaluation-cache-audit.json"
_RUNTIME_NAME = "phase2b3c-evaluation-runtime.json"
_REPLAY_NAME = "phase2b3c-evaluation-replay.json"
_LEDGER_NAME = "phase2b3c-access-ledger-snapshot.json"
_DECISIONS = {"confirmed", "empirically_met_but_inconclusive", "failed"}
_CHECKS = {
    "bundle_integrity_pass": True,
    "metadata_authority_pass": True,
    "cache_computation_replay_pass": True,
    "byte_identical_replay_pass": True,
    "copied_bundle_pass": True,
    "one_time_access_pass": True,
}
_ACCEPTANCE_AUTHORITY = object()


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise TypeError("acceptance digests require immutable bytes")
    return hashlib.sha256(payload).hexdigest()


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


def _validate_metadata_receipt(
    loaded: LoadedPhase2B3CBundle, receipt: VerifiedPhase2B3CBundle
) -> None:
    if type(receipt) is not VerifiedPhase2B3CBundle:
        raise TypeError("acceptance requires the exact metadata verification receipt")
    receipt.__post_init__()
    payloads = dict(loaded.payloads)
    documents = loaded.documents()
    result = _object(documents[_RESULT_NAME], "evaluation result")
    evaluation = _object(result.get("evaluation"), "result evaluation")
    digests = _object(result.get("digests"), "result digests")
    expected = {
        "manifest_sha256": loaded.manifest_sha256,
        "result_sha256": _sha256(payloads[_RESULT_NAME]),
        "cache_audit_sha256": _sha256(payloads[_AUDIT_NAME]),
        "runtime_sha256": _sha256(payloads[_RUNTIME_NAME]),
        "replay_sha256": _sha256(payloads[_REPLAY_NAME]),
        "ledger_snapshot_sha256": _sha256(payloads[_LEDGER_NAME]),
        "evaluation_id": evaluation.get("evaluation_id"),
        "permit_sha256": evaluation.get("permit_sha256"),
        "producer_revision": result.get("producer_revision"),
        "ordered_sample_ids_sha256": digests.get("ordered_sample_ids_sha256"),
        "ordered_membership_sha256": digests.get("ordered_membership_sha256"),
        "input_receipt_sha256": digests.get("input_receipt_sha256"),
        "ordered_inputs_sha256": digests.get("ordered_inputs_sha256"),
        "map_evidence_sha256": digests.get("map_evidence_sha256"),
        "phase_decision": result.get("phase_decision"),
    }
    if (
        receipt.schema
        != "trustsr.phase2b3c-candidate-bundle-metadata-verification.v1"
        or receipt.verification_scope != "metadata_consistency_only"
        or receipt.cache_computation_verified is not False
        or receipt.acceptance_authorized is not False
        or any(getattr(receipt, key) != value for key, value in expected.items())
    ):
        raise ValueError("metadata verification receipt differs from bundle bytes")


def _validate_computation_receipt(
    receipt: VerifiedPhase2B3CComputation, metadata: VerifiedPhase2B3CBundle
) -> None:
    if type(receipt) is not VerifiedPhase2B3CComputation:
        raise TypeError("acceptance requires the exact computation verification receipt")
    receipt.__post_init__()
    if (
        receipt.result_sha256 != metadata.result_sha256
        or receipt.cache_audit_sha256 != metadata.cache_audit_sha256
        or receipt.runtime_sha256 != metadata.runtime_sha256
        or receipt.map_evidence_sha256 != metadata.map_evidence_sha256
        or receipt.phase_decision != metadata.phase_decision
    ):
        raise ValueError("computation verification receipt differs from bundle bytes")


def _validate_byte_identical_replay(
    loaded: LoadedPhase2B3CBundle, metadata: VerifiedPhase2B3CBundle
) -> None:
    payloads = dict(loaded.payloads)
    verified = verify_phase2b3c_replay_receipt(
        payloads[_REPLAY_NAME],
        committed_result=payloads[_RESULT_NAME],
        committed_cache_audit=payloads[_AUDIT_NAME],
        committed_runtime=payloads[_RUNTIME_NAME],
    )
    if (
        verified.replay_sha256 != metadata.replay_sha256
        or verified.result_sha256 != metadata.result_sha256
        or verified.cache_audit_sha256 != metadata.cache_audit_sha256
        or verified.runtime_sha256 != metadata.runtime_sha256
        or verified.map_evidence_sha256 != metadata.map_evidence_sha256
    ):
        raise ValueError("byte-identical replay differs from verified bundle metadata")


@dataclass(frozen=True, init=False)
class VerifiedPhase2B3CAcceptance:
    """Opaque acceptance capability issued only after both independent verifiers."""

    payload: bytes
    phase_decision: str
    verifier_revision: str
    bundle_complete_event_sha256: str
    _authority: object = field(repr=False, compare=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("acceptance receipts are created only by both verifiers")

    @classmethod
    def _from_verified(
        cls, document: Mapping[str, object]
    ) -> VerifiedPhase2B3CAcceptance:
        if type(document) is not dict:
            raise TypeError("verified acceptance must be an exact JSON object")
        payload = canonical_json(document)
        decision = document.get("phase_decision")
        if decision not in _DECISIONS:
            raise ValueError("verified acceptance decision is invalid")
        receipt = object.__new__(cls)
        object.__setattr__(receipt, "payload", payload)
        object.__setattr__(receipt, "phase_decision", decision)
        implementation = _object(
            document.get("implementation"), "acceptance implementation"
        )
        one_time = _object(
            document.get("one_time_access"), "acceptance one-time access"
        )
        object.__setattr__(
            receipt,
            "verifier_revision",
            _revision(implementation.get("verifier_revision"), "verifier revision"),
        )
        object.__setattr__(
            receipt,
            "bundle_complete_event_sha256",
            _digest(
                one_time.get("bundle_complete_event_sha256"),
                "bundle_complete event",
            ),
        )
        object.__setattr__(receipt, "_authority", _ACCEPTANCE_AUTHORITY)
        receipt.__post_init__()
        return receipt

    def __post_init__(self) -> None:
        if (
            getattr(self, "_authority", None) is not _ACCEPTANCE_AUTHORITY
            or type(self.payload) is not bytes
            or self.phase_decision not in _DECISIONS
        ):
            raise ValueError("verified acceptance authority or payload identity is invalid")
        try:
            document = json.loads(self.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("verified acceptance payload is not JSON") from exc
        if (
            type(document) is not dict
            or canonical_json(document) != self.payload
            or document.get("phase_decision") != self.phase_decision
        ):
            raise ValueError("verified acceptance payload is invalid")
        implementation = _object(
            document.get("implementation"), "verified acceptance implementation"
        )
        one_time = _object(
            document.get("one_time_access"), "verified acceptance one-time access"
        )
        if (
            implementation.get("verifier_revision") != self.verifier_revision
            or one_time.get("bundle_complete_event_sha256")
            != self.bundle_complete_event_sha256
        ):
            raise ValueError("verified acceptance receipt bindings are invalid")

    def as_dict(self) -> dict[str, object]:
        self.__post_init__()
        value = json.loads(self.payload)
        if type(value) is not dict:  # pragma: no cover - post-init invariant
            raise RuntimeError("verified acceptance payload is not an object")
        return value


def build_phase2b3c_acceptance(
    loaded_bundle: LoadedPhase2B3CBundle,
    metadata_verification: VerifiedPhase2B3CBundle,
    computation_verification: VerifiedPhase2B3CComputation,
    *,
    verifier_revision: str,
) -> VerifiedPhase2B3CAcceptance:
    """Authorize exactly the observed three-way decision, never an alternative."""

    if type(loaded_bundle) is not LoadedPhase2B3CBundle:
        raise TypeError("acceptance requires an exact loaded bundle snapshot")
    loaded_bundle.__post_init__()
    _validate_metadata_receipt(loaded_bundle, metadata_verification)
    _validate_computation_receipt(computation_verification, metadata_verification)
    _validate_byte_identical_replay(loaded_bundle, metadata_verification)
    decision = metadata_verification.phase_decision
    document = {
        "schema": ACCEPTANCE_SCHEMA,
        "verification_scope": VERIFICATION_SCOPE,
        "acceptance_authorized": True,
        "checks": dict(_CHECKS),
        "target": {
            "alpha": PHASE2B3C_ALPHA,
            "minimum_coverage": PHASE2B3C_MINIMUM_COVERAGE,
            "risk_upper_bound": PHASE2B3C_RISK_UPPER_BOUND,
        },
        "phase_decision": decision,
        "digests": {
            "bundle_manifest_sha256": loaded_bundle.manifest_sha256,
            "result_sha256": metadata_verification.result_sha256,
            "cache_audit_sha256": metadata_verification.cache_audit_sha256,
            "runtime_sha256": metadata_verification.runtime_sha256,
            "replay_sha256": metadata_verification.replay_sha256,
            "ledger_snapshot_sha256": metadata_verification.ledger_snapshot_sha256,
            "access_permit_sha256": metadata_verification.permit_sha256,
        },
        "implementation": {
            "producer_revision": _revision(
                metadata_verification.producer_revision, "producer revision"
            ),
            "verifier_revision": _revision(verifier_revision, "verifier revision"),
        },
        "one_time_access": {
            "evaluation_id": metadata_verification.evaluation_id,
            "bundle_complete_event_sha256": (
                metadata_verification.bundle_complete_event_sha256
            ),
            "terminal_state": "accepted",
        },
    }
    return VerifiedPhase2B3CAcceptance._from_verified(document)


@dataclass(frozen=True)
class Phase2B3CPublicationReceipt:
    publication_sha256: str
    result_sha256: str
    cache_audit_sha256: str
    acceptance_sha256: str
    phase_decision: str
    reused: bool

    def __post_init__(self) -> None:
        for value in (
            self.publication_sha256,
            self.result_sha256,
            self.cache_audit_sha256,
            self.acceptance_sha256,
        ):
            _digest(value, "publication digest")
        if self.phase_decision not in _DECISIONS or type(self.reused) is not bool:
            raise ValueError("publication receipt decision or reuse state is invalid")


def _canonical_project_root(project_root: Path) -> Path:
    if not isinstance(project_root, Path) or not project_root.is_absolute():
        raise ValueError("publication project root must be an absolute directory")
    try:
        if (
            project_root.is_symlink()
            or project_root.resolve(strict=True) != project_root.absolute()
        ):
            raise ValueError("publication project root must be canonical")
    except OSError as exc:
        raise ValueError("publication project root must be canonical") from exc
    return project_root


def _publication_payloads(
    loaded: LoadedPhase2B3CBundle, acceptance: VerifiedPhase2B3CAcceptance
) -> dict[str, bytes]:
    if type(loaded) is not LoadedPhase2B3CBundle:
        raise TypeError("publication requires an exact loaded bundle")
    loaded.__post_init__()
    if type(acceptance) is not VerifiedPhase2B3CAcceptance:
        raise TypeError("publication requires a verified acceptance capability")
    acceptance.__post_init__()
    value = acceptance.as_dict()
    digests = _object(value.get("digests"), "acceptance digests")
    target = _object(value.get("target"), "acceptance target")
    implementation = _object(
        value.get("implementation"), "acceptance implementation"
    )
    one_time = _object(value.get("one_time_access"), "acceptance one-time access")
    payloads = dict(loaded.payloads)
    result = _object(loaded.documents()[_RESULT_NAME], "publication result")
    verified_result = verify_phase2b3c_result(payloads[_RESULT_NAME])
    evaluation = _object(result.get("evaluation"), "publication evaluation")
    frozen = _object(result.get("frozen"), "publication frozen configuration")
    result_target = _object(frozen.get("target"), "publication result target")
    replay = verify_phase2b3c_replay_receipt(
        payloads[_REPLAY_NAME],
        committed_result=payloads[_RESULT_NAME],
        committed_cache_audit=payloads[_AUDIT_NAME],
        committed_runtime=payloads[_RUNTIME_NAME],
    )
    expected = {
        "bundle_manifest_sha256": loaded.manifest_sha256,
        "result_sha256": _sha256(payloads[_RESULT_NAME]),
        "cache_audit_sha256": _sha256(payloads[_AUDIT_NAME]),
        "runtime_sha256": _sha256(payloads[_RUNTIME_NAME]),
        "replay_sha256": _sha256(payloads[_REPLAY_NAME]),
        "ledger_snapshot_sha256": _sha256(payloads[_LEDGER_NAME]),
        "access_permit_sha256": evaluation.get("permit_sha256"),
    }
    if (
        set(value)
        != {
            "schema",
            "verification_scope",
            "acceptance_authorized",
            "checks",
            "target",
            "phase_decision",
            "digests",
            "implementation",
            "one_time_access",
        }
        or value.get("schema") != ACCEPTANCE_SCHEMA
        or value.get("verification_scope") != VERIFICATION_SCOPE
        or value.get("acceptance_authorized") is not True
        or value.get("checks") != _CHECKS
        or value.get("phase_decision") != acceptance.phase_decision
        or value.get("phase_decision") != result.get("phase_decision")
        or target
        != {
            "alpha": PHASE2B3C_ALPHA,
            "minimum_coverage": PHASE2B3C_MINIMUM_COVERAGE,
            "risk_upper_bound": PHASE2B3C_RISK_UPPER_BOUND,
        }
        or target != result_target
        or digests != expected
        or set(implementation) != {"producer_revision", "verifier_revision"}
        or implementation.get("producer_revision")
        != verified_result.producer_revision
        or _revision(
            implementation.get("verifier_revision"), "publication verifier revision"
        )
        != acceptance.verifier_revision
        or one_time
        != {
            "evaluation_id": evaluation.get("evaluation_id"),
            "bundle_complete_event_sha256": acceptance.bundle_complete_event_sha256,
            "terminal_state": "accepted",
        }
        or replay.replay_sha256 != expected["replay_sha256"]
        or replay.result_sha256 != expected["result_sha256"]
        or replay.cache_audit_sha256 != expected["cache_audit_sha256"]
        or replay.runtime_sha256 != expected["runtime_sha256"]
        or replay.map_evidence_sha256 != verified_result.map_evidence_sha256
    ):
        raise ValueError("publication acceptance authority differs from bundle bytes")
    return {
        RESULT_PUBLICATION_NAME: payloads[_RESULT_NAME],
        AUDIT_PUBLICATION_NAME: payloads[_AUDIT_NAME],
        ACCEPTANCE_PUBLICATION_NAME: acceptance.payload,
    }


def _publication_digest(payloads: Mapping[str, bytes]) -> str:
    return _sha256(
        canonical_json(
            [
                {"basename": name, "sha256": _sha256(payloads[name])}
                for name in _PUBLICATION_NAMES
            ]
        )
    )


def _publication_receipt(
    payloads: Mapping[str, bytes], decision: str, *, reused: bool
) -> Phase2B3CPublicationReceipt:
    return Phase2B3CPublicationReceipt(
        publication_sha256=_publication_digest(payloads),
        result_sha256=_sha256(payloads[RESULT_PUBLICATION_NAME]),
        cache_audit_sha256=_sha256(payloads[AUDIT_PUBLICATION_NAME]),
        acceptance_sha256=_sha256(payloads[ACCEPTANCE_PUBLICATION_NAME]),
        phase_decision=decision,
        reused=reused,
    )


def _write_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_stable_file_descriptor(
    descriptor: int, *, label: str, maximum_bytes: int
) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= maximum_bytes:
        raise ValueError(f"{label} is not a size-bounded regular file")
    chunks: list[bytes] = []
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(descriptor, min(before.st_size - offset, 64 * 1024), offset)
        if not chunk:
            raise ValueError(f"{label} ended before its descriptor size")
        chunks.append(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, before.st_size):
        raise ValueError(f"{label} exceeds its descriptor size")
    after = os.fstat(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"{label} changed while being read")
    return b"".join(chunks)


def _open_regular_at(directory_descriptor: int, name: str, label: str) -> int:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened safely") from exc
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"{label} must be a regular non-symlink file")
    return descriptor


def _validate_publication_directory(
    target_descriptor: int,
    permit_descriptor: int,
    payloads: Mapping[str, bytes],
    permit_sha256: str,
) -> bool:
    names = {entry.name for entry in os.scandir(target_descriptor)}
    allowed = {_PERMIT_PUBLICATION_NAME, *_PUBLICATION_NAMES}
    if not names <= allowed or _PERMIT_PUBLICATION_NAME not in names:
        raise ValueError("Phase 2B3-C publication contains an unexpected entry")
    current_permit = os.stat(
        _PERMIT_PUBLICATION_NAME,
        dir_fd=target_descriptor,
        follow_symlinks=False,
    )
    locked_permit = os.fstat(permit_descriptor)
    if (
        not stat.S_ISREG(current_permit.st_mode)
        or (current_permit.st_dev, current_permit.st_ino)
        != (locked_permit.st_dev, locked_permit.st_ino)
        or _sha256(
            _read_stable_file_descriptor(
                permit_descriptor,
                label="Phase 2B3-C publication permit",
                maximum_bytes=1024**2,
            )
        )
        != permit_sha256
    ):
        raise ValueError("Phase 2B3-C publication permit differs from acceptance")
    present = names & set(_PUBLICATION_NAMES)
    if not present:
        return False
    if present != set(_PUBLICATION_NAMES):
        raise ValueError("existing Phase 2B3-C publication is partial")
    for name in _PUBLICATION_NAMES:
        descriptor = _open_regular_at(
            target_descriptor, name, "Phase 2B3-C publication evidence"
        )
        try:
            observed = _read_stable_file_descriptor(
                descriptor,
                label="Phase 2B3-C publication evidence",
                maximum_bytes=5 * 1024**2,
            )
        finally:
            os.close(descriptor)
        if observed != payloads[name]:
            raise ValueError("existing Phase 2B3-C publication has different bytes")
    return True


def publish_phase2b3c_evidence(
    project_root: Path,
    loaded_bundle: LoadedPhase2B3CBundle,
    acceptance: VerifiedPhase2B3CAcceptance,
) -> Phase2B3CPublicationReceipt:
    """Commit three files all-or-clean under locks, preserving the permit.

    The permit already occupies the final directory, so POSIX cannot provide one
    directory-rename visibility point for the three result names.  The formal
    workflow lock plus the permit lock serialize cooperating readers/writers;
    failures roll back only destination inodes created by this invocation.
    """

    root = _canonical_project_root(project_root)
    artifacts = root / "artifacts"
    target = artifacts / "phase2b3c"
    payloads = _publication_payloads(loaded_bundle, acceptance)
    permit_sha256 = acceptance.as_dict()["digests"]["access_permit_sha256"]
    _digest(permit_sha256, "publication permit")
    if artifacts.is_symlink() or not artifacts.is_dir():
        raise ValueError("repository artifacts directory is invalid")
    if (
        not target.exists()
        or target.is_symlink()
        or not target.is_dir()
        or target.resolve(strict=True) != target.absolute()
    ):
        raise ValueError("Phase 2B3-C publication requires the preexisting permit directory")
    target_descriptor = os.open(
        target,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        descriptor = _open_regular_at(
            target_descriptor,
            _PERMIT_PUBLICATION_NAME,
            "Phase 2B3-C publication permit",
        )
    except BaseException:
        os.close(target_descriptor)
        raise
    staging = artifacts / f".phase2b3c-publication.{secrets.token_hex(16)}"
    created: list[tuple[str, int, int]] = []
    staging_descriptor: int | None = None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Phase 2B3-C publication permit is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if _validate_publication_directory(
            target_descriptor, descriptor, payloads, permit_sha256
        ):
            return _publication_receipt(
                payloads, acceptance.phase_decision, reused=True
            )
        staging.mkdir(mode=0o700)
        for name in _PUBLICATION_NAMES:
            _write_file(staging / name, payloads[name])
        _fsync_directory(staging)
        staging_descriptor = os.open(
            staging,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        for name in _PUBLICATION_NAMES:
            os.link(
                name,
                name,
                src_dir_fd=staging_descriptor,
                dst_dir_fd=target_descriptor,
                follow_symlinks=False,
            )
            metadata = os.stat(
                name, dir_fd=target_descriptor, follow_symlinks=False
            )
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("published evidence destination is not a regular file")
            created.append((name, metadata.st_dev, metadata.st_ino))
        _fsync_directory(target)
        if not _validate_publication_directory(
            target_descriptor, descriptor, payloads, permit_sha256
        ):
            raise RuntimeError("published evidence set is unexpectedly absent")
        return _publication_receipt(payloads, acceptance.phase_decision, reused=False)
    except BaseException:
        for name, device, inode in reversed(created):
            try:
                metadata = os.stat(
                    name, dir_fd=target_descriptor, follow_symlinks=False
                )
                if (
                    stat.S_ISREG(metadata.st_mode)
                    and metadata.st_dev == device
                    and metadata.st_ino == inode
                ):
                    os.unlink(name, dir_fd=target_descriptor)
            except OSError:
                pass
        if created:
            _fsync_directory(target)
        raise
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        if staging.exists():
            shutil.rmtree(staging)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
            os.close(target_descriptor)


@dataclass(frozen=True)
class IndependentPhase2B3CVerification:
    publication: Phase2B3CPublicationReceipt
    accepted_ledger_event_sha256: str

    def as_dict(self) -> dict[str, object]:
        self.publication.__post_init__()
        _digest(self.accepted_ledger_event_sha256, "accepted ledger event")
        return {
            "schema": "trustsr.phase2b3c-independent-verification-cli.v1",
            "verification_scope": VERIFICATION_SCOPE,
            "cache_computation_verified": True,
            "prediction_inference_verified": False,
            "acceptance_authorized": True,
            "ledger_state": "accepted",
            "publication_sha256": self.publication.publication_sha256,
            "acceptance_sha256": self.publication.acceptance_sha256,
            "result_sha256": self.publication.result_sha256,
            "cache_audit_sha256": self.publication.cache_audit_sha256,
            "phase_decision": self.publication.phase_decision,
        }


def _require_copied_bundle(bundle_dir: Path, producer_bundle_dir: Path) -> None:
    if not isinstance(bundle_dir, Path) or not bundle_dir.is_absolute():
        raise ValueError("independent verification requires an absolute copied bundle")
    try:
        if bundle_dir.is_symlink() or bundle_dir.resolve(strict=True) != bundle_dir.absolute():
            raise ValueError("independent verification requires a canonical copied bundle")
    except OSError as exc:
        raise ValueError("independent verification requires a canonical copied bundle") from exc
    producer = producer_bundle_dir.absolute()
    if bundle_dir == producer or producer in bundle_dir.parents:
        raise ValueError("independent verification requires a separate copied bundle")


def _require_cache_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir() or path.resolve(strict=True) != path.absolute():
        raise ValueError("independent verification cache directory is missing or unsafe")


def _run_locked_independent_verification(
    *,
    paths: Phase2B3CStoragePaths,
    bundle_dir: Path,
    project_root: Path,
    evidence_dir: Path,
    manifest_path: Path,
    access_permit_path: Path,
) -> IndependentPhase2B3CVerification:
    authority = load_phase2b3c_metadata_authority(
        project_root=project_root,
        evidence_dir=evidence_dir,
        storage_root=paths.root,
        manifest_path=manifest_path,
        access_permit_path=access_permit_path,
    )
    metadata = verify_phase2b3c_bundle(
        bundle_dir,
        project_root=project_root,
        evidence_dir=evidence_dir,
        storage_root=paths.root,
        manifest_path=manifest_path,
        access_permit_path=access_permit_path,
        authority=authority,
    )
    permit = authority.permit
    ledger = authority.ledger
    if (
        ledger.state != "bundle_complete"
        or ledger.event_sha256 != metadata.bundle_complete_event_sha256
    ):
        raise ValueError("independent verifier ledger changed after metadata verification")
    pairs = load_internal_test_pairs(
        paths.root, authority.records, ledger.access_guard()
    )
    input_receipt = build_internal_test_input_receipt(authority.records, pairs)
    cache_paths = phase2b3c_storage_paths(paths.root)
    _require_cache_directory(cache_paths.prediction_cache_dir)
    probe = probe_cached_internal_test_bundles(
        pairs, cache=PredictionCache(cache_paths.prediction_cache_dir)
    )
    if (
        type(probe) is not InternalTestPredictionCacheProbe
        or probe.bundles is None
        or probe.present_count != 600
        or probe.missing_count != 0
        or len(probe.bundles) != 120
    ):
        raise ValueError("independent verification requires the complete exact K5 cache")
    loaded = read_phase2b3c_bundle(bundle_dir)
    payloads = dict(loaded.payloads)
    computation = verify_phase2b3c_computation(
        payloads[_RESULT_NAME],
        payloads[_AUDIT_NAME],
        payloads[_RUNTIME_NAME],
        input_receipt=input_receipt,
        pairs=pairs,
        bundles=probe.bundles,
        dependencies=authority.dependencies,
    )
    acceptance = build_phase2b3c_acceptance(
        loaded,
        metadata,
        computation,
        verifier_revision=authority.head_revision,
    )
    publication = publish_phase2b3c_evidence(project_root, loaded, acceptance)
    accepted = advance_access_ledger(paths.root, permit, "accepted")
    if accepted.state != "accepted" or accepted.previous_event_sha256 != ledger.event_sha256:
        raise RuntimeError("accepted ledger transition is not bound to bundle_complete")
    return IndependentPhase2B3CVerification(
        publication=publication,
        accepted_ledger_event_sha256=accepted.event_sha256,
    )


def run_independent_phase2b3c_verification(
    *,
    bundle_dir: Path,
    project_root: Path,
    evidence_dir: Path,
    storage_root: Path,
    manifest_path: Path,
    access_permit_path: Path,
    confirmed_persistent_storage: bool,
) -> IndependentPhase2B3CVerification:
    """Replay verified caches, publish acceptance, then seal the one-time ledger."""

    paths = validate_phase2b3c_storage(storage_root, confirmed_persistent_storage)
    with phase2b3c_formal_lock(paths):
        _require_copied_bundle(bundle_dir, paths.bundle_dir)
        return _run_locked_independent_verification(
            paths=paths,
            bundle_dir=bundle_dir,
            project_root=project_root,
            evidence_dir=evidence_dir,
            manifest_path=manifest_path,
            access_permit_path=access_permit_path,
        )
