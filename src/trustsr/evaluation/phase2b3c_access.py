"""Readiness, reviewed permit, and immutable Phase 2B3-C access ledger."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from trustsr.data.crosssensor_pairs import (
    CROP_POLICY,
    PHASE2B3A_NORMALIZATION_POLICY,
    POST_MANIFEST_SHA256,
)
from trustsr.data.internal_test_pairs import (
    PixelsOpenedAccessGuard,
    _issue_pixels_opened_access_guard,
)
from trustsr.evaluation.phase2b3c_evidence import (
    INPUT_AUDIT_SHA256,
    PHASE2B3B_FILE_SHA256S,
    PHASE2B3B_PUBLICATION_COMMIT,
)
from trustsr.evaluation.phase2b3c_policy import (
    PHASE2B3C_ALPHA,
    PHASE2B3C_BISECTION_ITERATIONS,
    PHASE2B3C_DELTA,
    PHASE2B3C_EVALUATION_SIZE,
    PHASE2B3C_GRID_SIZE,
    PHASE2B3C_MINIMUM_COVERAGE,
    PHASE2B3C_RISK_UPPER_BOUND,
    PHASE2B3C_THRESHOLD,
)
from trustsr.jsonio import canonical_json

ACCESS_PERMIT_FILENAME = "sen2naipv2-internal-test-access-authorization-v1.json"
AUTHORIZATION_STATEMENT = (
    "Explicit user authorization for the one-time Phase 2B3-C internal_test "
    "evaluation was recorded before any protected data access."
)
READINESS_SCHEMA = "trustsr.phase2b3c-readiness.v1"
PERMIT_SCHEMA = "trustsr.phase2b3c-internal-test-access-authorization.v1"
LEDGER_EVENT_SCHEMA = "trustsr.phase2b3c-access-ledger-event.v1"
_REVISION = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_EVENT_NAME = re.compile(
    r"(?P<sequence>[0-9]{3})-(?P<state>reserved|pixels_opened|caches_complete|"
    r"bundle_complete|accepted|invalidated)\.json"
)
_NORMAL_STATES = (
    "reserved",
    "pixels_opened",
    "caches_complete",
    "bundle_complete",
    "accepted",
)
_TERMINAL_STATES = {"accepted", "invalidated"}
_MAX_DOCUMENT_BYTES = 1024**2
_PERMIT_AUTHORITY = object()
_LEDGER_AUTHORITY = object()


@dataclass(frozen=True)
class VerifiedAccessPermit:
    """Exact reviewed authorization identity; this module cannot create one."""

    evaluation_id: str
    permit_sha256: str
    readiness_sha256: str
    implementation_revision: str
    computation_tree_sha256: str
    ordered_membership_sha256: str
    environment_sha256: str
    phase2b3b_acceptance_sha256: str
    _authority: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class AccessLedgerSnapshot:
    """Validated latest append-only ledger event."""

    state: str
    sequence: int
    evaluation_id: str
    event_sha256: str
    previous_event_sha256: str | None
    permit_sha256: str
    _authority: object = field(repr=False, compare=False)

    def access_guard(self) -> PixelsOpenedAccessGuard:
        """Issue the pixel capability only for a consumed, non-invalidated evaluation."""

        if self._authority is not _LEDGER_AUTHORITY:
            raise ValueError("access ledger snapshot was not issued by the ledger verifier")
        if self.state == "invalidated":
            raise ValueError("an invalidated evaluation cannot receive an access guard")
        if self.state == "reserved":
            raise ValueError("pixels_opened must be recorded before issuing an access guard")
        return _issue_pixels_opened_access_guard(self.evaluation_id, self.event_sha256)


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _revision(value: object, label: str) -> str:
    if type(value) is not str or _REVISION.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical lowercase implementation revision")
    return value


def _policy() -> dict[str, object]:
    return {
        "alpha": PHASE2B3C_ALPHA,
        "minimum_coverage": PHASE2B3C_MINIMUM_COVERAGE,
        "delta": PHASE2B3C_DELTA,
        "bet_grid_size": PHASE2B3C_GRID_SIZE,
        "bisection_iterations": PHASE2B3C_BISECTION_ITERATIONS,
        "threshold": PHASE2B3C_THRESHOLD,
        "score": {
            "name": "ldsr_variance_k5",
            "operator_parameters": {
                "algorithm": "ensemble_variance_score",
                "band_reduction": "mean",
                "correction": 0,
                "seed_count": 5,
                "seed_first": 3407,
                "seed_last": 3411,
            },
            "seeds": [3407, 3408, 3409, 3410, 3411],
        },
        "risk": {
            "name": "local_l1_risk",
            "window": 9,
            "upper_bound": PHASE2B3C_RISK_UPPER_BOUND,
        },
    }


def build_phase2b3c_readiness(
    *,
    implementation_revision: str,
    computation_tree_sha256: str,
    phase2b3b_result_sha256: str,
    phase2b3b_cache_audit_sha256: str,
    phase2b3b_acceptance_sha256: str,
    ordered_membership_sha256: str,
    environment_sha256: str,
) -> dict[str, object]:
    """Build deterministic metadata only; this is explicitly not an access credential."""

    implementation_revision = _revision(
        implementation_revision, "implementation revision"
    )
    computation_tree_sha256 = _digest(
        computation_tree_sha256, "computation tree digest"
    )
    ordered_membership_sha256 = _digest(
        ordered_membership_sha256, "ordered membership digest"
    )
    environment_sha256 = _digest(environment_sha256, "environment digest")
    expected_b = {
        "result_sha256": PHASE2B3B_FILE_SHA256S[
            "sen2naipv2-calibration-conformal-v1.json"
        ],
        "cache_audit_sha256": PHASE2B3B_FILE_SHA256S[
            "sen2naipv2-calibration-conformal-cache-audit-v1.json"
        ],
        "acceptance_sha256": PHASE2B3B_FILE_SHA256S[
            "sen2naipv2-calibration-conformal-acceptance-v1.json"
        ],
    }
    supplied_b = {
        "result_sha256": _digest(phase2b3b_result_sha256, "Phase 2B3-B result"),
        "cache_audit_sha256": _digest(
            phase2b3b_cache_audit_sha256, "Phase 2B3-B cache audit"
        ),
        "acceptance_sha256": _digest(
            phase2b3b_acceptance_sha256, "Phase 2B3-B acceptance"
        ),
    }
    if supplied_b != expected_b:
        raise ValueError("Phase 2B3-B publication digests do not match the frozen evidence")
    identity: dict[str, object] = {
        "schema": READINESS_SCHEMA,
        "authorization_required": True,
        "one_time_evaluation": True,
        "implementation": {
            "revision": implementation_revision,
            "computation_tree_sha256": computation_tree_sha256,
        },
        "phase2b3b": {
            "publication_commit": PHASE2B3B_PUBLICATION_COMMIT,
            **expected_b,
        },
        "input": {
            "post_manifest_sha256": POST_MANIFEST_SHA256,
            "input_audit_sha256": INPUT_AUDIT_SHA256,
            "ordered_membership_sha256": ordered_membership_sha256,
            "sample_count": PHASE2B3C_EVALUATION_SIZE,
            "normalization_policy": PHASE2B3A_NORMALIZATION_POLICY,
            "crop_policy": CROP_POLICY,
        },
        "policy": _policy(),
        "environment_sha256": environment_sha256,
    }
    evaluation_id = _sha256(identity)
    return {**identity, "evaluation_id": evaluation_id}


def _canonical_root(path: Path, label: str) -> Path:
    if not isinstance(path, Path):
        raise ValueError(f"{label} must be an existing canonical non-symlink directory")
    try:
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"{label} must be an existing canonical non-symlink directory")
        resolved = path.resolve(strict=True)
        if resolved != path.absolute():
            raise ValueError(f"{label} must be an existing canonical non-symlink directory")
    except OSError as exc:
        raise ValueError(
            f"{label} must be an existing canonical non-symlink directory"
        ) from exc
    return resolved


def _read_canonical_file(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened without following symlinks") from exc
    try:
        return _read_canonical_descriptor(descriptor, label)
    finally:
        os.close(descriptor)


def _read_canonical_descriptor(
    descriptor: int, label: str
) -> tuple[dict[str, object], bytes]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= _MAX_DOCUMENT_BYTES:
        raise ValueError(f"{label} exceeds the 1 MiB limit")
    chunks: list[bytes] = []
    remaining = before.st_size
    while remaining:
        chunk = os.read(descriptor, min(remaining, 64 * 1024))
        if not chunk:
            raise ValueError(f"{label} ended before its descriptor size")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
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
    raw = b"".join(chunks)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if type(value) is not dict or canonical_json(value) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return value, raw


def _validated_readiness(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("access permit requires an exact readiness document")
    expected_keys = {
        "schema",
        "authorization_required",
        "one_time_evaluation",
        "implementation",
        "phase2b3b",
        "input",
        "policy",
        "environment_sha256",
        "evaluation_id",
    }
    if set(value) != expected_keys:
        raise ValueError("readiness document keys are invalid")
    implementation = value.get("implementation")
    phase2b3b = value.get("phase2b3b")
    input_identity = value.get("input")
    if not all(isinstance(item, Mapping) for item in (implementation, phase2b3b, input_identity)):
        raise ValueError("readiness identity sections are invalid")
    try:
        rebuilt = build_phase2b3c_readiness(
            implementation_revision=implementation["revision"],  # type: ignore[index]
            computation_tree_sha256=implementation[  # type: ignore[index]
                "computation_tree_sha256"
            ],
            phase2b3b_result_sha256=phase2b3b["result_sha256"],  # type: ignore[index]
            phase2b3b_cache_audit_sha256=phase2b3b[  # type: ignore[index]
                "cache_audit_sha256"
            ],
            phase2b3b_acceptance_sha256=phase2b3b[  # type: ignore[index]
                "acceptance_sha256"
            ],
            ordered_membership_sha256=input_identity[  # type: ignore[index]
                "ordered_membership_sha256"
            ],
            environment_sha256=value["environment_sha256"],
        )
    except KeyError as exc:
        raise ValueError("readiness identity fields are incomplete") from exc
    if rebuilt != value:
        raise ValueError("readiness document does not match the frozen Phase 2B3-C identity")
    return value


def verify_phase2b3c_access_permit(
    permit_path: Path, project_root: Path, readiness: Mapping[str, object]
) -> VerifiedAccessPermit:
    """Verify a separately reviewed permit at its one allowlisted repository path."""

    root = _canonical_root(project_root, "project root")
    expected_path = root / "artifacts" / "phase2b3c" / ACCESS_PERMIT_FILENAME
    if not isinstance(permit_path, Path) or permit_path.absolute() != expected_path:
        raise ValueError("access permit must use the exact repository location")
    try:
        if permit_path.resolve(strict=True) != expected_path:
            raise ValueError("access permit path must not contain a symlink")
    except OSError as exc:
        raise ValueError("access permit path is unreadable") from exc
    document, raw = _read_canonical_file(permit_path, "access permit")
    if document.get("authorization_statement") != AUTHORIZATION_STATEMENT:
        raise ValueError("access permit authorization statement is invalid")
    if document.get("one_time_evaluation") is not True:
        raise ValueError("access permit must authorize a one-time evaluation")
    readiness_value = _validated_readiness(readiness)
    expected = {
        "schema": PERMIT_SCHEMA,
        "evaluation_id": readiness_value.get("evaluation_id"),
        "readiness_sha256": _sha256(readiness_value),
        "implementation": readiness_value.get("implementation"),
        "phase2b3b": readiness_value.get("phase2b3b"),
        "input": readiness_value.get("input"),
        "policy": readiness_value.get("policy"),
        "environment_sha256": readiness_value.get("environment_sha256"),
        "one_time_evaluation": True,
        "authorization_statement": AUTHORIZATION_STATEMENT,
    }
    if document != expected:
        raise ValueError("access permit does not exactly match the readiness request")
    evaluation_id = _digest(document["evaluation_id"], "evaluation ID")
    implementation = document["implementation"]
    phase2b3b = document["phase2b3b"]
    input_identity = document["input"]
    if not all(isinstance(item, Mapping) for item in (implementation, phase2b3b, input_identity)):
        raise ValueError("access permit identity sections are invalid")
    return VerifiedAccessPermit(
        evaluation_id=evaluation_id,
        permit_sha256=hashlib.sha256(raw).hexdigest(),
        readiness_sha256=_digest(document["readiness_sha256"], "readiness"),
        implementation_revision=_revision(
            implementation["revision"], "implementation revision"  # type: ignore[index]
        ),
        computation_tree_sha256=_digest(
            implementation["computation_tree_sha256"],  # type: ignore[index]
            "computation tree digest",
        ),
        ordered_membership_sha256=_digest(
            input_identity["ordered_membership_sha256"],  # type: ignore[index]
            "ordered membership digest",
        ),
        environment_sha256=_digest(document["environment_sha256"], "environment"),
        phase2b3b_acceptance_sha256=_digest(
            phase2b3b["acceptance_sha256"], "Phase 2B3-B acceptance"  # type: ignore[index]
        ),
        _authority=_PERMIT_AUTHORITY,
    )


def _storage_identity(root: Path) -> str:
    return _sha256({"canonical_storage_root": str(root)})


def _binding(root: Path, permit: VerifiedAccessPermit) -> dict[str, object]:
    if (
        type(permit) is not VerifiedAccessPermit
        or permit._authority is not _PERMIT_AUTHORITY
    ):
        raise ValueError("ledger requires an exact verified access permit")
    return {
        "evaluation_id": permit.evaluation_id,
        "permit_sha256": permit.permit_sha256,
        "readiness_sha256": permit.readiness_sha256,
        "implementation_revision": permit.implementation_revision,
        "computation_tree_sha256": permit.computation_tree_sha256,
        "ordered_membership_sha256": permit.ordered_membership_sha256,
        "environment_sha256": permit.environment_sha256,
        "phase2b3b_acceptance_sha256": permit.phase2b3b_acceptance_sha256,
        "storage_root_sha256": _storage_identity(root),
    }


def _ledger_parent(root: Path) -> Path:
    parent = root / "trustsr" / "phase2b3c" / "access-ledger"
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        root_descriptor = os.open(root, flags)
        current = root_descriptor
        for component in ("trustsr", "phase2b3c", "access-ledger"):
            try:
                os.mkdir(component, mode=0o700, dir_fd=current)
            except FileExistsError:
                pass
            child = os.open(component, flags, dir_fd=current)
            if current != root_descriptor:
                os.close(current)
            current = child
        os.close(current)
        os.close(root_descriptor)
    except OSError as exc:
        if "current" in locals() and current != root_descriptor:
            os.close(current)
        if "root_descriptor" in locals():
            os.close(root_descriptor)
        raise ValueError("access ledger directory cannot be created safely") from exc
    if parent.resolve(strict=True) != parent.absolute():  # postcondition, not authority
        raise ValueError("access ledger directory must not contain symlinks")
    return parent


@contextmanager
def phase2b3c_access_lock(
    storage_root: Path, evaluation_id: str
) -> Iterator[None]:
    """Hold the per-evaluation nonblocking process lock."""

    root = _canonical_root(storage_root, "storage root")
    evaluation_id = _digest(evaluation_id, "evaluation ID")
    parent = _ledger_parent(root)
    lock_name = f".{evaluation_id}.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor = os.open(parent, _directory_open_flags())
    try:
        descriptor = os.open(lock_name, flags, 0o600, dir_fd=parent_descriptor)
    except OSError as exc:
        os.close(parent_descriptor)
        raise ValueError("unable to open the access ledger lock") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("access ledger lock must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Phase 2B3-C access ledger lock is already held") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
            os.close(parent_descriptor)


def _ledger_directory(root: Path, evaluation_id: str, *, create: bool) -> Path:
    parent = _ledger_parent(root)
    directory = parent / evaluation_id
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = os.open(parent, flags)
    try:
        if create:
            try:
                os.mkdir(evaluation_id, mode=0o700, dir_fd=parent_descriptor)
            except FileExistsError:
                pass
        try:
            child = os.open(evaluation_id, flags, dir_fd=parent_descriptor)
        except OSError as exc:
            raise ValueError("access ledger is missing or contains a symlink") from exc
        os.close(child)
    finally:
        os.close(parent_descriptor)
    return directory


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _event_document(
    *,
    state: str,
    sequence: int,
    previous: str | None,
    binding: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": LEDGER_EVENT_SCHEMA,
        "sequence": sequence,
        "state": state,
        "previous_event_sha256": previous,
        "binding": dict(binding),
    }


def _publish_no_replace(directory: Path, name: str, payload: bytes) -> None:
    directory_fd = os.open(directory, _directory_open_flags())
    temporary_name = f".event-{secrets.token_hex(16)}.partial"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        try:
            descriptor = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
        except OSError as exc:
            raise ValueError("unable to stage ledger event") from exc
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise ValueError("ledger event already exists and cannot be replaced") from exc
        except OSError as exc:
            raise ValueError("unable to publish ledger event without replacement") from exc
        os.fsync(directory_fd)
    finally:
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)


def _valid_transition(previous: str | None, requested: str) -> bool:
    if previous is None:
        return requested == "reserved"
    if requested == "invalidated":
        return previous in {"pixels_opened", "caches_complete", "bundle_complete"}
    try:
        return _NORMAL_STATES.index(requested) == _NORMAL_STATES.index(previous) + 1
    except ValueError:
        return False


def _load_event_chain(
    directory_fd: int,
    permit: VerifiedAccessPermit,
    expected_binding: Mapping[str, object],
    *,
    absent_ok: bool,
) -> AccessLedgerSnapshot | None:
    event_paths: list[tuple[int, str, str]] = []
    with os.scandir(directory_fd) as entries:
        for entry in entries:
            match = _EVENT_NAME.fullmatch(entry.name)
            metadata = os.stat(entry.name, dir_fd=directory_fd, follow_symlinks=False)
            if match is None or not stat.S_ISREG(metadata.st_mode):
                raise ValueError("access ledger contains an unexpected entry")
            event_paths.append((int(match["sequence"]), match["state"], entry.name))
    event_paths.sort(key=lambda item: item[0])
    if not event_paths:
        if absent_ok:
            return None
        raise ValueError("access ledger contains no event")
    if [item[0] for item in event_paths] != list(range(len(event_paths))):
        raise ValueError("access ledger event sequence must be contiguous")
    previous_digest: str | None = None
    previous_state: str | None = None
    snapshot: AccessLedgerSnapshot | None = None
    for sequence, filename_state, name in event_paths:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=directory_fd)
        except OSError as exc:
            raise ValueError("access ledger event cannot be opened safely") from exc
        try:
            document, raw = _read_canonical_descriptor(descriptor, "access ledger event")
        finally:
            os.close(descriptor)
        expected_keys = {
            "schema",
            "sequence",
            "state",
            "previous_event_sha256",
            "binding",
        }
        if set(document) != expected_keys or document.get("schema") != LEDGER_EVENT_SCHEMA:
            raise ValueError("access ledger event schema is invalid")
        state = document.get("state")
        if (
            type(state) is not str
            or state != filename_state
            or document.get("sequence") != sequence
            or type(document.get("sequence")) is not int
        ):
            raise ValueError("access ledger event filename or sequence is invalid")
        if document.get("binding") != expected_binding:
            raise ValueError("access ledger binding does not match permit or storage root")
        if document.get("previous_event_sha256") != previous_digest:
            raise ValueError("access ledger predecessor digest is invalid")
        if not _valid_transition(previous_state, state):
            raise ValueError("access ledger transition history is invalid")
        digest = hashlib.sha256(raw).hexdigest()
        snapshot = AccessLedgerSnapshot(
            state=state,
            sequence=sequence,
            evaluation_id=permit.evaluation_id,
            event_sha256=digest,
            previous_event_sha256=previous_digest,
            permit_sha256=permit.permit_sha256,
            _authority=_LEDGER_AUTHORITY,
        )
        previous_digest = digest
        previous_state = state
    return snapshot


def _load_unlocked(
    root: Path, permit: VerifiedAccessPermit, *, absent_ok: bool
) -> AccessLedgerSnapshot | None:
    parent = _ledger_parent(root)
    directory = parent / permit.evaluation_id
    if not directory.exists():
        if absent_ok:
            return None
        raise ValueError("access ledger does not exist")
    directory = _ledger_directory(root, permit.evaluation_id, create=False)
    expected_binding = _binding(root, permit)
    directory_fd = os.open(
        directory,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        return _load_event_chain(
            directory_fd,
            permit,
            expected_binding,
            absent_ok=absent_ok,
        )
    finally:
        os.close(directory_fd)


def load_access_ledger(
    storage_root: Path, permit: VerifiedAccessPermit
) -> AccessLedgerSnapshot:
    """Load and authenticate the complete immutable event chain."""

    root = _canonical_root(storage_root, "storage root")
    with phase2b3c_access_lock(root, permit.evaluation_id):
        snapshot = _load_unlocked(root, permit, absent_ok=False)
    if snapshot is None:  # pragma: no cover - absent_ok=False invariant
        raise RuntimeError("access ledger invariant was violated")
    return snapshot


def advance_access_ledger(
    storage_root: Path, permit: VerifiedAccessPermit, state: str
) -> AccessLedgerSnapshot:
    """Append exactly one legal transition, or return an exact idempotent resume."""

    root = _canonical_root(storage_root, "storage root")
    if (
        type(permit) is not VerifiedAccessPermit
        or permit._authority is not _PERMIT_AUTHORITY
    ):
        raise ValueError("ledger requires an exact verified access permit")
    if type(state) is not str or state not in {*_NORMAL_STATES, "invalidated"}:
        raise ValueError("requested access ledger state is invalid")
    with phase2b3c_access_lock(root, permit.evaluation_id):
        current = _load_unlocked(root, permit, absent_ok=True)
        if current is not None and current.state == state:
            return current
        if current is not None and current.state in _TERMINAL_STATES:
            raise ValueError("access ledger is terminal")
        previous_state = None if current is None else current.state
        if not _valid_transition(previous_state, state):
            raise ValueError("requested access ledger transition is not allowed")
        sequence = 0 if current is None else current.sequence + 1
        previous_digest = None if current is None else current.event_sha256
        directory = _ledger_directory(root, permit.evaluation_id, create=True)
        document = _event_document(
            state=state,
            sequence=sequence,
            previous=previous_digest,
            binding=_binding(root, permit),
        )
        name = f"{sequence:03d}-{state}.json"
        _publish_no_replace(directory, name, canonical_json(document))
        created = _load_unlocked(root, permit, absent_ok=False)
        if created is None or created.state != state:  # pragma: no cover - fail closed
            raise RuntimeError("published access ledger event could not be revalidated")
        return created
