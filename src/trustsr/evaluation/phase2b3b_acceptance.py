"""Independent acceptance and atomic Git-safe publication for Phase 2B3-B."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from trustsr.artifacts.predictions import PredictionCache
from trustsr.artifacts.scores import ScoreCache
from trustsr.data.calibration_pairs import load_calibration_pairs
from trustsr.data.calibration_subset import load_calibration_records
from trustsr.evaluation.calibration_input_receipt import build_calibration_input_receipt
from trustsr.evaluation.calibration_radiometry import build_calibration_radiometry
from trustsr.evaluation.phase2b3b_bundle import (
    LoadedPhase2B3BBundle,
    read_phase2b3b_bundle,
)
from trustsr.evaluation.phase2b3b_bundle_verify import (
    VerifiedPhase2B3BBundle,
    verify_phase2b3b_bundle,
)
from trustsr.evaluation.phase2b3b_computation_verify import (
    VerifiedPhase2B3BComputation,
    verify_phase2b3b_computation,
)
from trustsr.evaluation.phase2b3b_policy import require_approved_operating_point
from trustsr.evaluation.phase2b3b_preflight import load_phase2b3b_preflight
from trustsr.evaluation.phase2b3b_revision import (
    verify_phase2b3b_revision,
    verify_recorded_phase2b3b_revision,
)
from trustsr.evaluation.phase2b3b_workflow import (
    Phase2B3BStoragePaths,
    phase2b3b_formal_lock,
    phase2b3b_storage_paths,
    validate_phase2b3b_storage,
)
from trustsr.jsonio import canonical_json

ACCEPTANCE_SCHEMA = "trustsr.phase2b3b-calibration-acceptance.v1"
VERIFICATION_SCOPE = "independent_calibration_acceptance"
RESULT_PUBLICATION_NAME = "sen2naipv2-calibration-conformal-v1.json"
AUDIT_PUBLICATION_NAME = "sen2naipv2-calibration-conformal-cache-audit-v1.json"
ACCEPTANCE_PUBLICATION_NAME = "sen2naipv2-calibration-conformal-acceptance-v1.json"
_RESULT_NAME = "phase2b3b-calibration-result.json"
_AUDIT_NAME = "phase2b3b-calibration-cache-audit.json"
_RUNTIME_NAME = "phase2b3b-calibration-runtime.json"
_REPLAY_NAME = "phase2b3b-calibration-replay.json"
_PUBLICATION_NAMES = (
    RESULT_PUBLICATION_NAME,
    AUDIT_PUBLICATION_NAME,
    ACCEPTANCE_PUBLICATION_NAME,
)
_ACCEPTANCE_KEYS = {
    "schema",
    "verification_scope",
    "acceptance_authorized",
    "checks",
    "target",
    "phase_decision",
    "digests",
    "frozen_calibration",
}
_ACCEPTANCE_CHECKS = {
    "bundle_integrity_pass": True,
    "metadata_authority_pass": True,
    "cache_computation_replay_pass": True,
    "byte_identical_replay_pass": True,
    "calibration_only_pass": True,
}
_ACCEPTANCE_DIGEST_KEYS = {
    "bundle_manifest_sha256",
    "result_sha256",
    "cache_audit_sha256",
    "runtime_manifest_sha256",
    "replay_sha256",
}
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_LIBC = ctypes.CDLL(None, use_errno=True)


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise TypeError("acceptance digests require immutable bytes")
    return hashlib.sha256(payload).hexdigest()


def _is_digest(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an exact JSON object")
    return value


def _validate_metadata_receipt(
    loaded: LoadedPhase2B3BBundle,
    receipt: VerifiedPhase2B3BBundle,
    result: dict[str, object],
    payloads: dict[str, bytes],
) -> None:
    if type(receipt) is not VerifiedPhase2B3BBundle:
        raise TypeError("acceptance requires the exact metadata verification receipt")
    receipt.__post_init__()
    expected = {
        "manifest_sha256": loaded.manifest_sha256,
        "result_sha256": _sha256(payloads[_RESULT_NAME]),
        "cache_audit_sha256": _sha256(payloads[_AUDIT_NAME]),
        "runtime_manifest_sha256": _sha256(payloads[_RUNTIME_NAME]),
        "replay_sha256": _sha256(payloads[_REPLAY_NAME]),
        "producer_revision": result.get("producer_revision"),
        "ordered_sample_ids_sha256": _mapping(
            result.get("upstream"), "result upstream"
        ).get("ordered_sample_ids_sha256"),
        "ordered_membership_sha256": _mapping(
            result.get("upstream"), "result upstream"
        ).get("ordered_membership_sha256"),
        "input_receipt_sha256": result.get("input_receipt_sha256"),
        "ordered_inputs_sha256": result.get("ordered_inputs_sha256"),
        "map_evidence_sha256": result.get("map_evidence_sha256"),
        "phase_decision": result.get("phase_decision"),
    }
    if (
        receipt.schema
        != "trustsr.phase2b3b-candidate-bundle-metadata-verification.v1"
        or receipt.verification_scope != "metadata_consistency_only"
        or receipt.cache_computation_verified is not False
        or any(getattr(receipt, key) != value for key, value in expected.items())
    ):
        raise ValueError("metadata verification receipt differs from bundle bytes")
    for key, value in expected.items():
        if key not in {"producer_revision", "phase_decision"} and not _is_digest(value):
            raise ValueError("metadata verification receipt contains an invalid digest")


def _validate_computation_receipt(
    receipt: VerifiedPhase2B3BComputation,
    metadata: VerifiedPhase2B3BBundle,
) -> None:
    if type(receipt) is not VerifiedPhase2B3BComputation:
        raise TypeError("acceptance requires the exact computation verification receipt")
    receipt.__post_init__()
    if (
        receipt.result_sha256 != metadata.result_sha256
        or receipt.cache_audit_sha256 != metadata.cache_audit_sha256
        or receipt.map_evidence_sha256 != metadata.map_evidence_sha256
    ):
        raise ValueError("computation verification receipt differs from bundle bytes")


def _validate_replay(
    replay: dict[str, object], metadata: VerifiedPhase2B3BBundle
) -> None:
    if set(replay) != {
        "schema",
        "byte_identical",
        "result_sha256",
        "cache_audit_sha256",
        "runtime_manifest_sha256",
    }:
        raise ValueError("replay receipt keys are invalid")
    if (
        replay["schema"] != "trustsr.phase2b3b-calibration-replay.v1"
        or replay["byte_identical"] is not True
        or replay["result_sha256"] != metadata.result_sha256
        or replay["cache_audit_sha256"] != metadata.cache_audit_sha256
        or replay["runtime_manifest_sha256"] != metadata.runtime_manifest_sha256
    ):
        raise ValueError("replay receipt is not byte-identical to the verified bundle")


def _frozen_payload(
    result: dict[str, object],
    runtime: dict[str, object],
    result_sha256: str,
) -> dict[str, object]:
    upstream = _mapping(result.get("upstream"), "result upstream")
    frozen = _mapping(result.get("frozen"), "result frozen configuration")
    counts = _mapping(result.get("counts"), "result counts")
    model_inventory = _mapping(runtime.get("model_inventory"), "runtime model inventory")
    if set(model_inventory) != {"identity", "seeds"}:
        raise ValueError("runtime model inventory keys are invalid")
    return {
        "upstream": upstream,
        "score": _mapping(frozen.get("score"), "frozen score"),
        "risk": _mapping(frozen.get("risk"), "frozen risk"),
        "target": _mapping(result.get("target"), "result target"),
        "threshold": result.get("threshold"),
        "risk_bound": result.get("risk_bound"),
        "counts": counts,
        "coverage": result.get("coverage"),
        "input": {
            **_mapping(frozen.get("input"), "frozen input"),
            "post_manifest_sha256": upstream.get("post_manifest_sha256"),
            "input_audit_sha256": upstream.get("input_audit_sha256"),
            "ordered_sample_ids_sha256": upstream.get("ordered_sample_ids_sha256"),
            "ordered_membership_sha256": upstream.get("ordered_membership_sha256"),
            "input_receipt_sha256": result.get("input_receipt_sha256"),
            "ordered_inputs_sha256": result.get("ordered_inputs_sha256"),
        },
        "model_inventory": model_inventory,
        "producer_revision": result.get("producer_revision"),
        "result_sha256": result_sha256,
    }


def build_phase2b3b_acceptance(
    loaded_bundle: LoadedPhase2B3BBundle,
    metadata_verification: VerifiedPhase2B3BBundle,
    computation_verification: VerifiedPhase2B3BComputation,
) -> VerifiedPhase2B3BAcceptance:
    """Authorize the observed B decision only after both independent verifiers pass."""

    if type(loaded_bundle) is not LoadedPhase2B3BBundle:
        raise TypeError("acceptance requires an exact loaded bundle snapshot")
    loaded_bundle.__post_init__()
    documents = loaded_bundle.documents()
    payloads = dict(loaded_bundle.payloads)
    result = _mapping(documents[_RESULT_NAME], "calibration result")
    runtime = _mapping(documents[_RUNTIME_NAME], "runtime manifest")
    replay = _mapping(documents[_REPLAY_NAME], "replay receipt")
    _validate_metadata_receipt(
        loaded_bundle, metadata_verification, result, payloads
    )
    _validate_computation_receipt(computation_verification, metadata_verification)
    _validate_replay(replay, metadata_verification)
    target = _mapping(result.get("target"), "result target")
    if set(target) != {"alpha", "minimum_coverage"}:
        raise ValueError("result target keys are invalid")
    alpha, minimum_coverage = require_approved_operating_point(
        target["alpha"], target["minimum_coverage"]
    )
    decision = result.get("phase_decision")
    if decision not in {"freeze_calibration", "stop_insufficient_coverage"}:
        raise ValueError("result phase decision is invalid")
    if decision != metadata_verification.phase_decision:
        raise ValueError("verified phase decision differs from the bundle")
    frozen_calibration = (
        _frozen_payload(result, runtime, metadata_verification.result_sha256)
        if decision == "freeze_calibration"
        else None
    )
    acceptance = {
        "schema": ACCEPTANCE_SCHEMA,
        "verification_scope": VERIFICATION_SCOPE,
        "acceptance_authorized": True,
        "checks": {
            "bundle_integrity_pass": True,
            "metadata_authority_pass": True,
            "cache_computation_replay_pass": True,
            "byte_identical_replay_pass": True,
            "calibration_only_pass": True,
        },
        "target": {"alpha": alpha, "minimum_coverage": minimum_coverage},
        "phase_decision": decision,
        "digests": {
            "bundle_manifest_sha256": loaded_bundle.manifest_sha256,
            "result_sha256": metadata_verification.result_sha256,
            "cache_audit_sha256": metadata_verification.cache_audit_sha256,
            "runtime_manifest_sha256": metadata_verification.runtime_manifest_sha256,
            "replay_sha256": metadata_verification.replay_sha256,
        },
        "frozen_calibration": frozen_calibration,
    }
    return VerifiedPhase2B3BAcceptance._from_verified(acceptance)


@dataclass(frozen=True, init=False)
class VerifiedPhase2B3BAcceptance:
    """Opaque acceptance capability created only after both verifiers pass."""

    payload: bytes
    phase_decision: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("acceptance receipts are created only by both verifiers")

    @classmethod
    def _from_verified(
        cls, document: Mapping[str, object]
    ) -> VerifiedPhase2B3BAcceptance:
        if type(document) is not dict:
            raise TypeError("verified acceptance must be an exact JSON object")
        payload = canonical_json(document)
        decision = document.get("phase_decision")
        if decision not in {"freeze_calibration", "stop_insufficient_coverage"}:
            raise ValueError("verified acceptance phase decision is invalid")
        receipt = object.__new__(cls)
        object.__setattr__(receipt, "payload", payload)
        object.__setattr__(receipt, "phase_decision", decision)
        receipt.__post_init__()
        return receipt

    def __post_init__(self) -> None:
        if type(self.payload) is not bytes:
            raise TypeError("verified acceptance payload must be immutable bytes")
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

    def as_dict(self) -> dict[str, object]:
        """Return a fresh JSON-native projection of the verified document."""

        self.__post_init__()
        document = json.loads(self.payload.decode("utf-8"))
        if type(document) is not dict:
            raise AssertionError("verified acceptance must decode to an object")
        return document


@dataclass(frozen=True)
class Phase2B3BPublicationReceipt:
    """Host-free digest receipt for the three-file Git-safe publication."""

    publication_sha256: str
    result_sha256: str
    cache_audit_sha256: str
    acceptance_sha256: str
    phase_decision: str
    reused: bool

    def __post_init__(self) -> None:
        if any(
            not _is_digest(value)
            for value in (
                self.publication_sha256,
                self.result_sha256,
                self.cache_audit_sha256,
                self.acceptance_sha256,
            )
        ):
            raise ValueError("publication receipt digest is invalid")
        if self.phase_decision not in {
            "freeze_calibration",
            "stop_insufficient_coverage",
        } or type(self.reused) is not bool:
            raise ValueError("publication receipt decision or reuse state is invalid")


def _canonical_project_root(project_root: Path) -> Path:
    if not isinstance(project_root, Path) or not project_root.is_absolute():
        raise ValueError("publication project root must be an absolute canonical directory")
    try:
        if (
            project_root.is_symlink()
            or project_root.resolve(strict=True) != project_root.absolute()
        ):
            raise ValueError("publication project root must be an absolute canonical directory")
    except OSError as exc:
        raise ValueError(
            "publication project root must be an absolute canonical directory"
        ) from exc
    return project_root


def _publication_payloads(
    loaded_bundle: LoadedPhase2B3BBundle,
    acceptance: VerifiedPhase2B3BAcceptance,
) -> dict[str, bytes]:
    if type(loaded_bundle) is not LoadedPhase2B3BBundle:
        raise TypeError("publication requires an exact loaded bundle")
    loaded_bundle.__post_init__()
    if type(acceptance) is not VerifiedPhase2B3BAcceptance:
        raise TypeError("publication requires a verified acceptance capability")
    acceptance.__post_init__()
    acceptance_document = acceptance.as_dict()
    _validate_acceptance_for_publication(loaded_bundle, acceptance_document)
    payloads = dict(loaded_bundle.payloads)
    result = payloads[_RESULT_NAME]
    audit = payloads[_AUDIT_NAME]
    return {
        RESULT_PUBLICATION_NAME: result,
        AUDIT_PUBLICATION_NAME: audit,
        ACCEPTANCE_PUBLICATION_NAME: acceptance.payload,
    }


def _validate_acceptance_for_publication(
    loaded_bundle: LoadedPhase2B3BBundle,
    acceptance: dict[str, object],
) -> None:
    if set(acceptance) != _ACCEPTANCE_KEYS:
        raise ValueError("publication acceptance keys are invalid")
    if (
        acceptance["schema"] != ACCEPTANCE_SCHEMA
        or acceptance["verification_scope"] != VERIFICATION_SCOPE
        or acceptance["acceptance_authorized"] is not True
    ):
        raise ValueError("publication acceptance authority is invalid")
    checks = _mapping(acceptance["checks"], "acceptance checks")
    if checks != _ACCEPTANCE_CHECKS or canonical_json(checks) != canonical_json(
        _ACCEPTANCE_CHECKS
    ):
        raise ValueError("publication acceptance checks are invalid")
    target = _mapping(acceptance["target"], "acceptance target")
    if set(target) != {"alpha", "minimum_coverage"}:
        raise ValueError("publication acceptance target keys are invalid")
    require_approved_operating_point(target["alpha"], target["minimum_coverage"])

    documents = loaded_bundle.documents()
    payloads = dict(loaded_bundle.payloads)
    result = _mapping(documents[_RESULT_NAME], "publication result")
    runtime = _mapping(documents[_RUNTIME_NAME], "publication runtime")
    replay = _mapping(documents[_REPLAY_NAME], "publication replay")
    decision = acceptance["phase_decision"]
    if (
        decision not in {"freeze_calibration", "stop_insufficient_coverage"}
        or decision != result.get("phase_decision")
    ):
        raise ValueError("publication acceptance decision differs from the bundle")
    result_target = _mapping(result.get("target"), "publication result target")
    if canonical_json(result_target) != canonical_json(target):
        raise ValueError("publication acceptance target differs from the bundle")

    digests = _mapping(acceptance["digests"], "acceptance digests")
    if set(digests) != _ACCEPTANCE_DIGEST_KEYS:
        raise ValueError("publication acceptance digest keys are invalid")
    expected_digests = {
        "bundle_manifest_sha256": loaded_bundle.manifest_sha256,
        "result_sha256": _sha256(payloads[_RESULT_NAME]),
        "cache_audit_sha256": _sha256(payloads[_AUDIT_NAME]),
        "runtime_manifest_sha256": _sha256(payloads[_RUNTIME_NAME]),
        "replay_sha256": _sha256(payloads[_REPLAY_NAME]),
    }
    if canonical_json(digests) != canonical_json(expected_digests):
        raise ValueError("publication acceptance digests differ from the bundle")

    if set(replay) != {
        "schema",
        "byte_identical",
        "result_sha256",
        "cache_audit_sha256",
        "runtime_manifest_sha256",
    } or (
        replay["schema"] != "trustsr.phase2b3b-calibration-replay.v1"
        or replay["byte_identical"] is not True
        or replay["result_sha256"] != expected_digests["result_sha256"]
        or replay["cache_audit_sha256"] != expected_digests["cache_audit_sha256"]
        or replay["runtime_manifest_sha256"]
        != expected_digests["runtime_manifest_sha256"]
    ):
        raise ValueError("publication acceptance replay differs from the bundle")

    expected_frozen = (
        _frozen_payload(result, runtime, expected_digests["result_sha256"])
        if decision == "freeze_calibration"
        else None
    )
    if canonical_json(acceptance["frozen_calibration"]) != canonical_json(
        expected_frozen
    ):
        raise ValueError("publication acceptance payload differs from the bundle")


def _publication_digest(payloads: Mapping[str, bytes]) -> str:
    entries = [
        {"basename": name, "sha256": _sha256(payloads[name])}
        for name in _PUBLICATION_NAMES
    ]
    return _sha256(canonical_json(entries))


def _publication_receipt(
    payloads: Mapping[str, bytes], *, phase_decision: str, reused: bool
) -> Phase2B3BPublicationReceipt:
    return Phase2B3BPublicationReceipt(
        publication_sha256=_publication_digest(payloads),
        result_sha256=_sha256(payloads[RESULT_PUBLICATION_NAME]),
        cache_audit_sha256=_sha256(payloads[AUDIT_PUBLICATION_NAME]),
        acceptance_sha256=_sha256(payloads[ACCEPTANCE_PUBLICATION_NAME]),
        phase_decision=phase_decision,
        reused=reused,
    )


def _validate_existing_publication(
    target: Path, payloads: Mapping[str, bytes]
) -> None:
    if target.is_symlink() or not target.is_dir():
        raise ValueError("existing Phase 2B3-B publication must be a regular directory")
    if target.resolve(strict=True) != target.absolute():
        raise ValueError("existing Phase 2B3-B publication path is not canonical")
    if {entry.name for entry in target.iterdir()} != set(_PUBLICATION_NAMES):
        raise ValueError("existing Phase 2B3-B publication has missing or extra files")
    for name in _PUBLICATION_NAMES:
        path = target / name
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payloads[name]:
            raise ValueError("existing Phase 2B3-B publication has different bytes")


def _write_publication_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
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


def _rename_noreplace(source: Path, target: Path) -> None:
    renameat2 = getattr(_LIBC, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable", target)
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(target),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), target)


def publish_phase2b3b_evidence(
    project_root: Path,
    loaded_bundle: LoadedPhase2B3BBundle,
    acceptance: VerifiedPhase2B3BAcceptance,
) -> Phase2B3BPublicationReceipt:
    """Publish the exact three Git-safe files as one directory transaction."""

    root = _canonical_project_root(project_root)
    artifacts = root / "artifacts"
    if artifacts.is_symlink() or not artifacts.is_dir():
        raise ValueError("repository artifacts directory is invalid")
    if artifacts.resolve(strict=True) != artifacts.absolute():
        raise ValueError("repository artifacts directory is not canonical")
    target = artifacts / "phase2b3b"
    payloads = _publication_payloads(loaded_bundle, acceptance)
    decision = acceptance.phase_decision
    if target.exists() or target.is_symlink():
        _validate_existing_publication(target, payloads)
        return _publication_receipt(payloads, phase_decision=decision, reused=True)

    staging = Path(tempfile.mkdtemp(prefix=".phase2b3b.", dir=artifacts))
    published = False
    try:
        for name in _PUBLICATION_NAMES:
            _write_publication_file(staging / name, payloads[name])
        _fsync_directory(staging)
        try:
            _rename_noreplace(staging, target)
        except OSError as exc:
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            _validate_existing_publication(target, payloads)
            return _publication_receipt(payloads, phase_decision=decision, reused=True)
        published = True
        _fsync_directory(artifacts)
        _validate_existing_publication(target, payloads)
        return _publication_receipt(payloads, phase_decision=decision, reused=False)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


@dataclass(frozen=True)
class IndependentPhase2B3BVerification:
    """Final host-free receipt after acceptance and Git-safe publication."""

    publication: Phase2B3BPublicationReceipt

    def as_dict(self) -> dict[str, object]:
        if type(self.publication) is not Phase2B3BPublicationReceipt:
            raise TypeError("independent verification requires a publication receipt")
        self.publication.__post_init__()
        return {
            "schema": "trustsr.phase2b3b-independent-verification-cli.v1",
            "verification_scope": VERIFICATION_SCOPE,
            "cache_computation_verified": True,
            "prediction_inference_verified": False,
            "acceptance_authorized": True,
            "publication_sha256": self.publication.publication_sha256,
            "acceptance_sha256": self.publication.acceptance_sha256,
            "result_sha256": self.publication.result_sha256,
            "cache_audit_sha256": self.publication.cache_audit_sha256,
            "phase_decision": self.publication.phase_decision,
        }


def _require_existing_cache_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir() or path.resolve(strict=True) != path.absolute():
        raise ValueError("independent verification cache directory is missing or unsafe")


def _require_copied_bundle(bundle_dir: Path, producer_bundle_dir: Path) -> None:
    if not isinstance(bundle_dir, Path) or not bundle_dir.is_absolute():
        raise ValueError("independent verification requires an absolute copied bundle")
    try:
        if bundle_dir.is_symlink() or bundle_dir.resolve(strict=True) != bundle_dir.absolute():
            raise ValueError("independent verification requires a canonical copied bundle")
    except OSError as exc:
        raise ValueError(
            "independent verification requires a canonical copied bundle"
        ) from exc
    producer = producer_bundle_dir.absolute()
    if bundle_dir == producer or producer in bundle_dir.parents:
        raise ValueError("independent verification requires a separate copied bundle")


def _run_locked_independent_verification(
    *,
    paths: Phase2B3BStoragePaths,
    bundle_dir: Path,
    project_root: Path,
    evidence_dir: Path,
    manifest_path: Path,
) -> IndependentPhase2B3BVerification:
    current_revision = verify_phase2b3b_revision(project_root)
    metadata = verify_phase2b3b_bundle(
        bundle_dir,
        project_root=project_root,
        evidence_dir=evidence_dir,
        storage_root=paths.root,
        manifest_path=manifest_path,
    )
    loaded = read_phase2b3b_bundle(bundle_dir)
    payloads = dict(loaded.payloads)
    preflight = load_phase2b3b_preflight(evidence_dir, paths.root, manifest_path)
    records = load_calibration_records(paths.root, manifest_path)
    pairs = load_calibration_pairs(paths.root, records)
    input_receipt = build_calibration_input_receipt(records, pairs, preflight)
    radiometry = build_calibration_radiometry(pairs)
    verify_recorded_phase2b3b_revision(project_root, metadata.producer_revision)
    recorded_revision = replace(
        current_revision, head_revision=metadata.producer_revision
    )
    cache_paths = phase2b3b_storage_paths(paths.root)
    _require_existing_cache_directory(cache_paths.prediction_cache_dir)
    _require_existing_cache_directory(cache_paths.score_cache_dir)
    computation = verify_phase2b3b_computation(
        payloads[_RESULT_NAME],
        payloads[_AUDIT_NAME],
        preflight=preflight,
        input_receipt=input_receipt,
        radiometry=radiometry,
        revision=recorded_revision,
        pairs=pairs,
        prediction_cache=PredictionCache(cache_paths.prediction_cache_dir),
        score_cache=ScoreCache(cache_paths.score_cache_dir),
    )
    acceptance = build_phase2b3b_acceptance(loaded, metadata, computation)
    publication = publish_phase2b3b_evidence(project_root, loaded, acceptance)
    return IndependentPhase2B3BVerification(publication=publication)


def run_independent_phase2b3b_verification(
    *,
    bundle_dir: Path,
    project_root: Path,
    evidence_dir: Path,
    storage_root: Path,
    manifest_path: Path,
    confirmed_persistent_storage: bool,
) -> IndependentPhase2B3BVerification:
    """Independently verify computations and publish the observed B decision."""

    paths = validate_phase2b3b_storage(storage_root, confirmed_persistent_storage)
    with phase2b3b_formal_lock(paths):
        _require_copied_bundle(bundle_dir, paths.bundle_dir)
        return _run_locked_independent_verification(
            paths=paths,
            bundle_dir=bundle_dir,
            project_root=project_root,
            evidence_dir=evidence_dir,
            manifest_path=manifest_path,
        )
