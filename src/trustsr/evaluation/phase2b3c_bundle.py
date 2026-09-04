"""Atomic canonical bundle I/O for Phase 2B3-C evaluation evidence."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from trustsr.evaluation.phase2b3c_replay_receipt import (
    verify_phase2b3c_replay_receipt,
)
from trustsr.evaluation.phase2b3c_result_verify import verify_phase2b3c_result
from trustsr.evaluation.phase2b3c_runtime import verify_phase2b3c_runtime_manifest
from trustsr.jsonio import atomic_write_bytes, canonical_json

BUNDLE_MANIFEST_BASENAME = "phase2b3c-bundle-manifest.json"
BUNDLE_DOCUMENT_SCHEMAS = {
    "phase2b3c-evaluation-result.json": "trustsr.phase2b3c-evaluation.v1",
    "phase2b3c-evaluation-cache-audit.json": (
        "trustsr.phase2b3c-evaluation-cache-audit.v1"
    ),
    "phase2b3c-evaluation-runtime.json": "trustsr.phase2b3c-evaluation-runtime.v1",
    "phase2b3c-evaluation-replay.json": "trustsr.phase2b3c-evaluation-replay.v1",
    "phase2b3c-access-ledger-snapshot.json": (
        "trustsr.phase2b3c-access-ledger-snapshot.v1"
    ),
}

_MAX_FILE_BYTES = 5 * 1024 * 1024
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_MANIFEST_SCHEMA = "trustsr.phase2b3c-bundle-manifest.v1"
_RENAME_NOREPLACE = 1
_LIBC = ctypes.CDLL(None, use_errno=True)
_LEDGER_SNAPSHOT_SCHEMA = "trustsr.phase2b3c-access-ledger-snapshot.v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _document_payload(name: str, value: object) -> bytes:
    if type(value) is not dict:
        raise TypeError(f"bundle document must be an exact JSON object: {name}")
    if value.get("schema") != BUNDLE_DOCUMENT_SCHEMAS[name]:
        raise ValueError(f"bundle document schema is invalid: {name}")
    payload = canonical_json(value)
    if len(payload) > _MAX_FILE_BYTES:
        raise ValueError(f"bundle document exceeds the 5 MiB limit: {name}")
    return payload


def _validated_payloads(documents: Mapping[str, object]) -> dict[str, bytes]:
    if not isinstance(documents, Mapping) or set(documents) != set(BUNDLE_DOCUMENT_SCHEMAS):
        raise ValueError("bundle documents must match the exact file allowlist")
    return {name: _document_payload(name, documents[name]) for name in BUNDLE_DOCUMENT_SCHEMAS}


def _manifest_from_payloads(payloads: Mapping[str, bytes]) -> dict[str, object]:
    if type(payloads) is not dict or tuple(payloads) != tuple(BUNDLE_DOCUMENT_SCHEMAS):
        raise ValueError("bundle payloads must use the exact canonical order")
    if any(type(payload) is not bytes for payload in payloads.values()):
        raise TypeError("bundle payloads must be immutable bytes")
    return {
        "schema": _MANIFEST_SCHEMA,
        "phase": "internal_test_evaluation",
        "files": [
            {
                "basename": name,
                "size_bytes": len(payloads[name]),
                "sha256": _sha256(payloads[name]),
            }
            for name in sorted(payloads)
        ],
    }


def build_phase2b3c_bundle_manifest(
    documents: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact five-document manifest without reading or writing paths."""

    return _manifest_from_payloads(_validated_payloads(documents))


def _required_digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def build_phase2b3c_ledger_snapshot(
    *,
    evaluation_id: str,
    permit_sha256: str,
    state: str,
    sequence: int,
    event_sha256: str,
    previous_event_sha256: str,
) -> dict[str, object]:
    """Project the immutable pre-publication ledger state without host metadata."""

    if state != "caches_complete" or type(state) is not str or sequence != 2:
        raise ValueError("ledger snapshot must be the exact caches_complete event")
    return {
        "schema": _LEDGER_SNAPSHOT_SCHEMA,
        "evaluation_id": _required_digest(evaluation_id, "evaluation ID"),
        "permit_sha256": _required_digest(permit_sha256, "permit digest"),
        "state": "caches_complete",
        "sequence": 2,
        "event_sha256": _required_digest(event_sha256, "ledger event digest"),
        "previous_event_sha256": _required_digest(
            previous_event_sha256, "previous ledger event digest"
        ),
    }


def _verified_ledger_snapshot(
    value: object, *, evaluation_id: str, permit_sha256: str, event_sha256: str
) -> dict[str, object]:
    keys = {
        "schema",
        "evaluation_id",
        "permit_sha256",
        "state",
        "sequence",
        "event_sha256",
        "previous_event_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise ValueError("ledger snapshot keys are invalid")
    expected = build_phase2b3c_ledger_snapshot(
        evaluation_id=evaluation_id,
        permit_sha256=permit_sha256,
        state=value["state"],
        sequence=value["sequence"],
        event_sha256=event_sha256,
        previous_event_sha256=value["previous_event_sha256"],
    )
    if value != expected:
        raise ValueError("ledger snapshot differs from the result evaluation identity")
    return value


def _verified_cache_audit(
    value: object, *, ordered_sample_ids_sha256: str, map_evidence_sha256: str
) -> bytes:
    keys = {
        "schema",
        "split",
        "ordered_sample_ids_sha256",
        "sample_count",
        "prediction_count",
        "score_count",
        "samples",
    }
    if type(value) is not dict or set(value) != keys:
        raise ValueError("cache audit keys are invalid")
    if (
        value["schema"] != BUNDLE_DOCUMENT_SCHEMAS["phase2b3c-evaluation-cache-audit.json"]
        or value["split"] != "internal_test"
        or value["sample_count"] != 120
        or type(value["sample_count"]) is not int
        or value["prediction_count"] != 600
        or type(value["prediction_count"]) is not int
        or value["score_count"] != 120
        or type(value["score_count"]) is not int
        or type(value["samples"]) is not list
        or len(value["samples"]) != 120
    ):
        raise ValueError("cache audit identity or counts are invalid")
    sample_ids: list[str] = []
    map_entries: list[dict[str, object]] = []
    for sample in value["samples"]:
        if type(sample) is not dict or set(sample) != {
            "sample_id",
            "predictions",
            "score",
            "risk",
        }:
            raise ValueError("cache audit sample keys are invalid")
        sample_id = sample["sample_id"]
        predictions = sample["predictions"]
        score = sample["score"]
        risk = sample["risk"]
        if (
            type(sample_id) is not str
            or not sample_id
            or type(predictions) is not list
            or len(predictions) != 5
            or type(score) is not dict
            or set(score)
            != {"name", "cache_key", "identity", "score_sha256"}
            or type(risk) is not dict
            or set(risk) != {"name", "window", "risk_sha256"}
        ):
            raise ValueError("cache audit sample identity is invalid")
        for prediction, seed in zip(predictions, range(3407, 3412), strict=True):
            if (
                type(prediction) is not dict
                or set(prediction)
                != {
                    "model_name",
                    "seed",
                    "cache_key",
                    "identity",
                    "prediction_sha256",
                }
                or prediction["model_name"] != "ldsr-s2-x4"
                or prediction["seed"] != seed
                or type(prediction["seed"]) is not int
                or type(prediction["identity"]) is not dict
            ):
                raise ValueError("cache audit prediction identity is invalid")
            _required_digest(prediction["cache_key"], "prediction cache key")
            _required_digest(prediction["prediction_sha256"], "prediction digest")
        if score["name"] != "ldsr_variance_k5" or type(score["identity"]) is not dict:
            raise ValueError("cache audit score identity is invalid")
        _required_digest(score["cache_key"], "score cache key")
        score_digest = _required_digest(score["score_sha256"], "score digest")
        if risk["name"] != "local_l1_risk" or risk["window"] != 9:
            raise ValueError("cache audit risk identity is invalid")
        risk_digest = _required_digest(risk["risk_sha256"], "risk digest")
        sample_ids.append(sample_id)
        map_entries.append(
            {
                "sample_id": sample_id,
                "score_sha256": score_digest,
                "risk_sha256": risk_digest,
            }
        )
    if len(set(sample_ids)) != 120:
        raise ValueError("cache audit sample IDs must be unique")
    observed_ids_digest = hashlib.sha256(canonical_json(sample_ids)).hexdigest()
    if (
        value["ordered_sample_ids_sha256"] != observed_ids_digest
        or observed_ids_digest != ordered_sample_ids_sha256
    ):
        raise ValueError("cache audit ordered sample identity is invalid")
    if hashlib.sha256(canonical_json(map_entries)).hexdigest() != map_evidence_sha256:
        raise ValueError("cache audit map evidence digest is invalid")
    return canonical_json(value)


@dataclass(frozen=True)
class VerifiedPhase2B3CBundleDocuments:
    """Semantic identity of the five acyclic predecessor documents."""

    evaluation_id: str
    phase_decision: str
    result_sha256: str
    cache_audit_sha256: str
    runtime_sha256: str
    replay_sha256: str
    ledger_snapshot_sha256: str
    cache_computation_verified: bool
    acceptance_authorized: bool

    def __post_init__(self) -> None:
        for value in (
            self.evaluation_id,
            self.result_sha256,
            self.cache_audit_sha256,
            self.runtime_sha256,
            self.replay_sha256,
            self.ledger_snapshot_sha256,
        ):
            _required_digest(value, "verified bundle digest")
        if (
            self.phase_decision
            not in {"confirmed", "empirically_met_but_inconclusive", "failed"}
            or self.cache_computation_verified is not False
            or self.acceptance_authorized is not False
        ):
            raise ValueError("verified bundle scope or decision is invalid")


def verify_phase2b3c_bundle_documents(
    documents: Mapping[str, object],
) -> VerifiedPhase2B3CBundleDocuments:
    """Verify result → runtime → replay plus the audit and ledger cross-bindings."""

    payloads = _validated_payloads(documents)
    values = {
        name: json.loads(payload.decode("utf-8")) for name, payload in payloads.items()
    }
    result_name = "phase2b3c-evaluation-result.json"
    audit_name = "phase2b3c-evaluation-cache-audit.json"
    runtime_name = "phase2b3c-evaluation-runtime.json"
    replay_name = "phase2b3c-evaluation-replay.json"
    ledger_name = "phase2b3c-access-ledger-snapshot.json"
    result = verify_phase2b3c_result(payloads[result_name])
    result_value = values[result_name]
    audit_payload = _verified_cache_audit(
        values[audit_name],
        ordered_sample_ids_sha256=result_value["digests"][
            "ordered_sample_ids_sha256"
        ],
        map_evidence_sha256=result.map_evidence_sha256,
    )
    audit_sha256 = hashlib.sha256(audit_payload).hexdigest()
    if audit_sha256 != result.cache_audit_sha256:
        raise ValueError("cache audit digest differs from the result")
    runtime = verify_phase2b3c_runtime_manifest(
        payloads[runtime_name], result=payloads[result_name]
    )
    replay = verify_phase2b3c_replay_receipt(
        payloads[replay_name],
        committed_result=payloads[result_name],
        committed_cache_audit=payloads[audit_name],
        committed_runtime=payloads[runtime_name],
    )
    _verified_ledger_snapshot(
        values[ledger_name],
        evaluation_id=result.evaluation_id,
        permit_sha256=result.permit_sha256,
        event_sha256=result.ledger_event_sha256,
    )
    return VerifiedPhase2B3CBundleDocuments(
        evaluation_id=result.evaluation_id,
        phase_decision=result.phase_decision,
        result_sha256=result.result_sha256,
        cache_audit_sha256=audit_sha256,
        runtime_sha256=runtime.runtime_sha256,
        replay_sha256=replay.replay_sha256,
        ledger_snapshot_sha256=hashlib.sha256(payloads[ledger_name]).hexdigest(),
        cache_computation_verified=False,
        acceptance_authorized=False,
    )


def _canonical_object(payload: bytes, name: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"bundle file is not valid JSON: {name}") from exc
    if type(value) is not dict or canonical_json(value) != payload:
        raise ValueError(f"bundle file is not canonical JSON: {name}")
    return value


def _validate_snapshot(payloads: tuple[tuple[str, bytes], ...], manifest_payload: bytes) -> None:
    if type(payloads) is not tuple or tuple(name for name, _ in payloads) != tuple(
        BUNDLE_DOCUMENT_SCHEMAS
    ):
        raise ValueError("loaded bundle payload order is invalid")
    if any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not bytes
        for item in payloads
    ):
        raise TypeError("loaded bundle payloads must be immutable named bytes")
    if type(manifest_payload) is not bytes:
        raise TypeError("loaded bundle manifest must be immutable bytes")

    manifest = _canonical_object(manifest_payload, BUNDLE_MANIFEST_BASENAME)
    if set(manifest) != {"schema", "phase", "files"} or (
        manifest["schema"] != _MANIFEST_SCHEMA
        or manifest["phase"] != "internal_test_evaluation"
    ):
        raise ValueError("bundle manifest schema is invalid")
    entries = manifest["files"]
    if type(entries) is not list or len(entries) != len(BUNDLE_DOCUMENT_SCHEMAS):
        raise ValueError("bundle manifest file entries are invalid")

    payload_by_name = dict(payloads)
    for entry, expected_name in zip(entries, sorted(BUNDLE_DOCUMENT_SCHEMAS), strict=True):
        if type(entry) is not dict or set(entry) != {
            "basename",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("bundle manifest file entry schema is invalid")
        size = entry["size_bytes"]
        digest = entry["sha256"]
        if (
            entry["basename"] != expected_name
            or type(size) is not int
            or size < 0
            or size > _MAX_FILE_BYTES
            or type(digest) is not str
            or _DIGEST_PATTERN.fullmatch(digest) is None
        ):
            raise ValueError("bundle manifest file entry identity is invalid")
        payload = payload_by_name[expected_name]
        if len(payload) != size or _sha256(payload) != digest:
            raise ValueError("bundle file size or digest differs from the manifest")
        value = _canonical_object(payload, expected_name)
        if value.get("schema") != BUNDLE_DOCUMENT_SCHEMAS[expected_name]:
            raise ValueError(f"bundle document schema is invalid: {expected_name}")


@dataclass(frozen=True)
class LoadedPhase2B3CBundle:
    """Immutable, internally descriptor-consistent bytes from one bundle read."""

    payloads: tuple[tuple[str, bytes], ...] = field(repr=False)
    manifest_payload: bytes = field(repr=False)
    manifest_sha256: str

    def __post_init__(self) -> None:
        _validate_snapshot(self.payloads, self.manifest_payload)
        if (
            type(self.manifest_sha256) is not str
            or _DIGEST_PATTERN.fullmatch(self.manifest_sha256) is None
            or self.manifest_sha256 != _sha256(self.manifest_payload)
        ):
            raise ValueError("loaded bundle manifest identity is invalid")

    def documents(self) -> dict[str, dict[str, object]]:
        """Return freshly parsed documents so callers cannot mutate verified state."""

        return {name: json.loads(payload.decode("utf-8")) for name, payload in self.payloads}


@dataclass(frozen=True, init=False)
class BundleWriteReceipt:
    """Host-free identity returned only after descriptor-based publication checks."""

    manifest_sha256: str
    file_sha256s: tuple[tuple[str, str], ...]

    def __init__(self) -> None:
        raise TypeError("bundle write receipts are created only after publication checks")

    @classmethod
    def _from_loaded(cls, bundle: LoadedPhase2B3CBundle) -> BundleWriteReceipt:
        if type(bundle) is not LoadedPhase2B3CBundle:
            raise TypeError("bundle receipt requires an exact loaded bundle")
        bundle.__post_init__()
        value = object.__new__(cls)
        object.__setattr__(value, "manifest_sha256", bundle.manifest_sha256)
        object.__setattr__(
            value,
            "file_sha256s",
            tuple((name, _sha256(payload)) for name, payload in bundle.payloads),
        )
        value.__post_init__()
        return value

    def __post_init__(self) -> None:
        if (
            type(self.manifest_sha256) is not str
            or _DIGEST_PATTERN.fullmatch(self.manifest_sha256) is None
            or type(self.file_sha256s) is not tuple
            or tuple(name for name, _ in self.file_sha256s) != tuple(BUNDLE_DOCUMENT_SCHEMAS)
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[1]) is not str
                or _DIGEST_PATTERN.fullmatch(item[1]) is None
                for item in self.file_sha256s
            )
        ):
            raise ValueError("bundle write receipt identity is invalid")


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_canonical_directory(path: Path, label: str) -> int:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError(f"{label} must be an absolute canonical directory")
    try:
        if path.resolve(strict=True) != path.absolute():
            raise ValueError(f"{label} must be an absolute canonical directory")
    except OSError as exc:
        raise ValueError(f"{label} must be an absolute canonical directory") from exc

    descriptor = os.open(path.anchor, _directory_flags())
    try:
        for component in path.parts[1:]:
            if component in {"", ".", ".."}:
                raise ValueError(f"{label} must be an absolute canonical directory")
            next_descriptor = os.open(component, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except (OSError, ValueError) as exc:
        os.close(descriptor)
        raise ValueError(f"{label} must be an absolute canonical directory") from exc
    return descriptor


def _open_bundle_parent(bundle_dir: Path) -> tuple[int, str]:
    if (
        not isinstance(bundle_dir, Path)
        or not bundle_dir.is_absolute()
        or bundle_dir.name in {"", ".", ".."}
    ):
        raise ValueError("bundle directory must be an absolute canonical child path")
    return _open_canonical_directory(bundle_dir.parent, "bundle parent"), bundle_dir.name


def _read_regular_at(directory_fd: int, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise ValueError(f"bundle file is missing or unsafe: {name}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_FILE_BYTES:
            raise ValueError(f"bundle file is not a bounded regular file: {name}")
        chunks: list[bytes] = []
        remaining = _MAX_FILE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) != metadata.st_size or len(payload) > _MAX_FILE_BYTES:
            raise ValueError(f"bundle file size changed or exceeds the limit: {name}")
        return payload
    finally:
        os.close(descriptor)


def _read_bundle_at(directory_fd: int) -> LoadedPhase2B3CBundle:
    expected_names = {BUNDLE_MANIFEST_BASENAME, *BUNDLE_DOCUMENT_SCHEMAS}
    if set(os.listdir(directory_fd)) != expected_names:
        raise ValueError("bundle directory must contain the exact file allowlist")
    manifest_payload = _read_regular_at(directory_fd, BUNDLE_MANIFEST_BASENAME)
    payloads = tuple(
        (name, _read_regular_at(directory_fd, name)) for name in BUNDLE_DOCUMENT_SCHEMAS
    )
    if set(os.listdir(directory_fd)) != expected_names:
        raise ValueError("bundle directory changed during verification")
    return LoadedPhase2B3CBundle(
        payloads=payloads,
        manifest_payload=manifest_payload,
        manifest_sha256=_sha256(manifest_payload),
    )


def _read_child_bundle_at(parent_fd: int, name: str) -> LoadedPhase2B3CBundle:
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError("bundle directory cannot be opened safely") from exc
    try:
        return _read_bundle_at(descriptor)
    finally:
        os.close(descriptor)


def _read_child_if_present(parent_fd: int, name: str) -> LoadedPhase2B3CBundle | None:
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError("bundle directory cannot be opened safely") from exc
    try:
        return _read_bundle_at(descriptor)
    finally:
        os.close(descriptor)


def read_phase2b3c_bundle(bundle_dir: Path) -> LoadedPhase2B3CBundle:
    """Read exactly five allowlisted files through no-follow descriptors."""

    descriptor = _open_canonical_directory(bundle_dir, "bundle directory")
    try:
        return _read_bundle_at(descriptor)
    finally:
        os.close(descriptor)


def _receipt(bundle: LoadedPhase2B3CBundle) -> BundleWriteReceipt:
    return BundleWriteReceipt._from_loaded(bundle)


def _new_staging_at(parent_fd: int, target_name: str) -> tuple[str, int]:
    for _ in range(128):
        name = f".{target_name}.{secrets.token_hex(12)}.tmp"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        try:
            return name, os.open(name, _directory_flags(), dir_fd=parent_fd)
        except OSError:
            os.rmdir(name, dir_fd=parent_fd)
            raise
    raise FileExistsError("unable to allocate a unique bundle staging directory")


def _write_regular_at(directory_fd: int, name: str, payload: bytes) -> None:
    descriptor_path = Path(f"/proc/self/fd/{directory_fd}")
    atomic_write_bytes(descriptor_path / name, payload)


def _remove_staging_at(parent_fd: int, staging_fd: int, staging_name: str) -> None:
    for name in os.listdir(staging_fd):
        os.unlink(name, dir_fd=staging_fd)
    os.rmdir(staging_name, dir_fd=parent_fd)


def _rename_noreplace_at(parent_fd: int, source: str, target: str) -> None:
    renameat2 = getattr(_LIBC, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is required for fail-closed publication")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            parent_fd,
            os.fsencode(source),
            parent_fd,
            os.fsencode(target),
            _RENAME_NOREPLACE,
        )
        != 0
    ):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), target)


def _matches_expected(
    bundle: LoadedPhase2B3CBundle,
    payloads: Mapping[str, bytes],
    manifest_payload: bytes,
) -> bool:
    return dict(bundle.payloads) == payloads and bundle.manifest_payload == manifest_payload


def write_phase2b3c_bundle(
    bundle_dir: Path,
    *,
    result: Mapping[str, object],
    cache_audit: Mapping[str, object],
    runtime: Mapping[str, object],
    replay: Mapping[str, object],
    ledger_snapshot: Mapping[str, object],
) -> BundleWriteReceipt:
    """Atomically publish one exact bundle, or reuse byte-identical evidence."""

    documents = dict(
        zip(
            BUNDLE_DOCUMENT_SCHEMAS,
            (result, cache_audit, runtime, replay, ledger_snapshot),
            strict=True,
        )
    )
    payloads = _validated_payloads(documents)
    manifest_payload = canonical_json(_manifest_from_payloads(payloads))
    parent_fd, target_name = _open_bundle_parent(bundle_dir)
    try:
        existing = _read_child_if_present(parent_fd, target_name)
        if existing is not None:
            if not _matches_expected(existing, payloads, manifest_payload):
                raise ValueError("existing bundle has different bytes")
            return _receipt(existing)

        staging_name, staging_fd = _new_staging_at(parent_fd, target_name)
        published = False
        try:
            try:
                for name, payload in payloads.items():
                    _write_regular_at(staging_fd, name, payload)
                _write_regular_at(staging_fd, BUNDLE_MANIFEST_BASENAME, manifest_payload)
                os.fsync(staging_fd)
                staged = _read_bundle_at(staging_fd)
                if not _matches_expected(staged, payloads, manifest_payload):
                    raise ValueError("staged bundle has different bytes")
                try:
                    _rename_noreplace_at(parent_fd, staging_name, target_name)
                except FileExistsError:
                    existing = _read_child_bundle_at(parent_fd, target_name)
                else:
                    published = True
                    os.fsync(parent_fd)
                    existing = _read_child_bundle_at(parent_fd, target_name)
            finally:
                if not published:
                    _remove_staging_at(parent_fd, staging_fd, staging_name)
        finally:
            os.close(staging_fd)

        if not _matches_expected(existing, payloads, manifest_payload):
            raise ValueError("existing bundle has different bytes")
        return _receipt(existing)
    finally:
        os.close(parent_fd)
