"""Frozen Phase 2B3-B publication boundary consumed by Phase 2B3-C."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from trustsr.jsonio import canonical_json

PHASE2B3B_PUBLICATION_COMMIT = "f8f49a820d22b7dea2e003736ee465f9d7788f7d"
POST_MANIFEST_SHA256 = "c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a"
INPUT_AUDIT_SHA256 = "fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b"
PRODUCER_REVISION = "2d17c141174c1062f8a056b491326487202a6deb"
THRESHOLD = 7.970395366024563e-06
ALPHA = 0.05
MINIMUM_COVERAGE = 0.10
SEEDS = (3407, 3408, 3409, 3410, 3411)

_RESULT_NAME = "sen2naipv2-calibration-conformal-v1.json"
_AUDIT_NAME = "sen2naipv2-calibration-conformal-cache-audit-v1.json"
_ACCEPTANCE_NAME = "sen2naipv2-calibration-conformal-acceptance-v1.json"
PHASE2B3B_FILE_SHA256S = MappingProxyType(
    {
        _RESULT_NAME: "5fe01daaaa27443d5289cc366eacbe1fb628492c25bab16fadd164d3a8eb3174",
        _AUDIT_NAME: "40aff8aaccb27719d958d42026d68d8efc0c7f79479e3bd99777053ee1d6399f",
        _ACCEPTANCE_NAME: "ce7f1a91e9954235e289dfc6eececb67261ff57110caaf15fe6640a0fb3b69ab",
    }
)
_SCHEMAS = {
    _RESULT_NAME: "trustsr.phase2b3b-calibration.v1",
    _AUDIT_NAME: "trustsr.phase2b3b-calibration-cache-audit.v1",
    _ACCEPTANCE_NAME: "trustsr.phase2b3b-calibration-acceptance.v1",
}
_MAX_FILE_BYTES = 5 * 1024**2
_FROZEN_CALIBRATION_SHA256 = (
    "96657600fb0fce2de6d2f07376dcd5fbb26a722a9124d8751ffe3857e6e1074f"
)
_GIT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class FrozenPhase2B3BEvidence:
    """Immutable, minimal Phase 2B3-B calibration identity."""

    result_sha256: str
    cache_audit_sha256: str
    acceptance_sha256: str
    threshold: float
    alpha: float
    minimum_coverage: float
    seeds: tuple[int, ...]
    post_manifest_sha256: str
    input_audit_sha256: str
    producer_revision: str
    publication_commit: str
    _frozen_calibration_json: bytes

    def as_dict(self) -> dict[str, object]:
        """Return a fresh mutable copy of the validated frozen calibration."""

        value = json.loads(self._frozen_calibration_json)
        if type(value) is not dict:  # pragma: no cover - construction invariant
            raise RuntimeError("frozen calibration invariant was violated")
        return value


def _validate_directory(evidence_dir: Path) -> None:
    if not isinstance(evidence_dir, Path) or evidence_dir.is_symlink() or not evidence_dir.is_dir():
        raise ValueError("evidence directory must be an existing non-symlink directory")
    try:
        if evidence_dir.resolve(strict=True) != evidence_dir.absolute():
            raise ValueError("evidence directory must not contain symlink components")
        names = {entry.name for entry in evidence_dir.iterdir()}
    except OSError as exc:
        raise ValueError("evidence directory is unreadable") from exc
    if names != set(PHASE2B3B_FILE_SHA256S):
        raise ValueError("evidence directory must contain the exact three allowlisted files")


def _read_document(path: Path, name: str) -> dict[str, object]:
    try:
        mode = path.lstat().st_mode
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"evidence file is unreadable: {name}") from exc
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise ValueError(f"evidence file must be a regular non-symlink file: {name}")
    if len(raw) > _MAX_FILE_BYTES:
        raise ValueError(f"evidence file exceeds the 5 MiB limit: {name}")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PHASE2B3B_FILE_SHA256S[name]:
        raise ValueError(f"evidence digest mismatch: {name}")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"evidence file is not valid JSON: {name}") from exc
    if type(value) is not dict or canonical_json(value) != raw:
        raise ValueError(f"evidence file is not canonical JSON: {name}")
    if value.get("schema") != _SCHEMAS[name]:
        raise ValueError(f"evidence schema is invalid: {name}")
    return value


def _validate_frozen_calibration(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("frozen calibration is invalid")
    target = value.get("target")
    score = value.get("score")
    input_identity = value.get("input")
    risk = value.get("risk")
    counts = value.get("counts")
    if value.get("threshold") != THRESHOLD:
        raise ValueError("frozen threshold does not match the approved calibration")
    if target != {"alpha": ALPHA, "minimum_coverage": MINIMUM_COVERAGE}:
        raise ValueError("frozen target does not match the approved calibration")
    if (
        type(score) is not dict
        or score.get("name") != "ldsr_variance_k5"
        or score.get("seeds") != list(SEEDS)
        or score.get("operator_parameters")
        != {
            "algorithm": "ensemble_variance_score",
            "band_reduction": "mean",
            "correction": 0,
            "seed_count": 5,
            "seed_first": 3407,
            "seed_last": 3411,
        }
    ):
        raise ValueError("frozen score identity is invalid")
    if (
        type(input_identity) is not dict
        or input_identity.get("post_manifest_sha256") != POST_MANIFEST_SHA256
        or input_identity.get("input_audit_sha256") != INPUT_AUDIT_SHA256
        or input_identity.get("bands") != ["B04", "B03", "B02", "B08"]
        or input_identity.get("scale") != 4
    ):
        raise ValueError("frozen input identity is invalid")
    if risk != {"name": "local_l1_risk", "upper_bound": 1.0, "window": 9}:
        raise ValueError("frozen risk identity is invalid")
    if (
        type(counts) is not dict
        or counts.get("calibration") != 120
        or counts.get("predictions") != 600
        or counts.get("scores") != 120
    ):
        raise ValueError("frozen calibration counts are invalid")
    if value.get("producer_revision") != PRODUCER_REVISION:
        raise ValueError("frozen producer revision is invalid")
    if hashlib.sha256(canonical_json(value)).hexdigest() != _FROZEN_CALIBRATION_SHA256:
        raise ValueError("full frozen calibration payload is invalid")
    return value


def _validate_publication(project_root: Path) -> None:
    try:
        if project_root.is_symlink() or not project_root.is_dir():
            raise ValueError("project root must be an existing canonical directory")
        root = project_root.resolve(strict=True)
        if root != project_root.absolute():
            raise ValueError("project root must be an existing canonical directory")
        top_level = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            shell=False,
            stdin=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT_SECONDS,
        ).stdout.rstrip("\n")
        if top_level != str(root):
            raise ValueError("project root must identify the Git checkout top level")
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                PHASE2B3B_PUBLICATION_COMMIT,
                "HEAD",
            ],
            check=True,
            capture_output=True,
            shell=False,
            stdin=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
        for name, expected_digest in PHASE2B3B_FILE_SHA256S.items():
            published = subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "show",
                    f"{PHASE2B3B_PUBLICATION_COMMIT}:artifacts/phase2b3b/{name}",
                ],
                check=True,
                capture_output=True,
                shell=False,
                stdin=subprocess.DEVNULL,
                timeout=_GIT_TIMEOUT_SECONDS,
            ).stdout
            if hashlib.sha256(published).hexdigest() != expected_digest:
                raise ValueError("published evidence digest is absent from the frozen commit")
    except ValueError:
        raise
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(
            "Phase 2B3-B publication commit is not an ancestor of this checkout"
        ) from exc


def load_frozen_phase2b3b_evidence(
    evidence_dir: Path, project_root: Path
) -> FrozenPhase2B3BEvidence:
    """Load only the exact accepted Phase 2B3-B publication receipts.

    ``project_root`` is intentionally part of this API so later callers cannot
    silently detach evidence validation from checkout validation. Task 2 only
    consumes Git-safe published artifacts; revision enforcement is performed by
    :mod:`phase2b3c_revision` before any protected-data command can run.
    """

    _validate_directory(evidence_dir)
    documents = {
        name: _read_document(evidence_dir / name, name) for name in PHASE2B3B_FILE_SHA256S
    }
    result = documents[_RESULT_NAME]
    audit = documents[_AUDIT_NAME]
    acceptance = documents[_ACCEPTANCE_NAME]
    frozen = _validate_frozen_calibration(acceptance.get("frozen_calibration"))

    if (
        acceptance.get("acceptance_authorized") is not True
        or acceptance.get("phase_decision") != "freeze_calibration"
        or acceptance.get("target") != frozen["target"]
        or acceptance.get("digests", {}).get("result_sha256")
        != PHASE2B3B_FILE_SHA256S[_RESULT_NAME]
        or acceptance.get("digests", {}).get("cache_audit_sha256")
        != PHASE2B3B_FILE_SHA256S[_AUDIT_NAME]
    ):
        raise ValueError("Phase 2B3-B acceptance receipt is invalid")
    if (
        result.get("split") != "calibration"
        or result.get("target") != frozen["target"]
        or result.get("threshold") != frozen["threshold"]
        or result.get("producer_revision") != PRODUCER_REVISION
    ):
        raise ValueError("Phase 2B3-B result identity is invalid")
    if (
        audit.get("split") != "calibration"
        or audit.get("sample_count") != 120
        or audit.get("prediction_count") != 600
        or audit.get("score_count") != 120
    ):
        raise ValueError("Phase 2B3-B cache audit identity is invalid")
    _validate_publication(project_root)

    return FrozenPhase2B3BEvidence(
        result_sha256=PHASE2B3B_FILE_SHA256S[_RESULT_NAME],
        cache_audit_sha256=PHASE2B3B_FILE_SHA256S[_AUDIT_NAME],
        acceptance_sha256=PHASE2B3B_FILE_SHA256S[_ACCEPTANCE_NAME],
        threshold=THRESHOLD,
        alpha=ALPHA,
        minimum_coverage=MINIMUM_COVERAGE,
        seeds=SEEDS,
        post_manifest_sha256=POST_MANIFEST_SHA256,
        input_audit_sha256=INPUT_AUDIT_SHA256,
        producer_revision=PRODUCER_REVISION,
        publication_commit=PHASE2B3B_PUBLICATION_COMMIT,
        _frozen_calibration_json=canonical_json(frozen),
    )
