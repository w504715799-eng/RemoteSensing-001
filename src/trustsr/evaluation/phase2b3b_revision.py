"""Strict local Git revision gate for Phase 2B3-B preflight consumers."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from trustsr.evaluation.phase2b3b_evidence import (
    PRODUCER_REVISION as PHASE2B3A_CALCULATION_REVISION,
)
from trustsr.evaluation.phase2b3b_evidence import (
    PUBLICATION_COMMIT as PHASE2B3A_EVIDENCE_PUBLICATION,
)

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_GIT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class Phase2B3BRevision:
    """Host-free identity proven by the local checkout gate."""

    branch: str
    head_revision: str
    calculation_revision: str
    evidence_publication: str


def _canonical_project_root(project_root: Path) -> Path:
    if not isinstance(project_root, Path):
        raise ValueError("project root must be an existing canonical non-symlink directory")
    try:
        if project_root.is_symlink() or not project_root.is_dir():
            raise ValueError("project root must be an existing canonical non-symlink directory")
        resolved = project_root.resolve(strict=True)
        if resolved != project_root.absolute():
            raise ValueError("project root must be an existing canonical non-symlink directory")
    except OSError as exc:
        raise ValueError(
            "project root must be an existing canonical non-symlink directory"
        ) from exc
    return resolved


def _run_git(project_root: Path, *arguments: str, failure: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            stdin=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("unable to inspect local Git checkout") from exc
    if completed.returncode != 0 or not isinstance(completed.stdout, str):
        raise ValueError(failure)
    return completed.stdout


def _single_line(output: str, failure: str) -> str:
    if not output.endswith("\n") or output.count("\n") != 1:
        raise ValueError(failure)
    return output[:-1]


def _canonical_revision(output: str, failure: str) -> str:
    revision = _single_line(output, failure)
    if _REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError(failure)
    return revision


def _require_frozen_revision(revision: str, label: str) -> None:
    if _REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError(f"frozen {label} is not a canonical revision")


def verify_phase2b3b_revision(project_root: Path) -> Phase2B3BRevision:
    """Prove a clean attached checkout descends from both frozen Phase 2B3-A commits."""

    root = _canonical_project_root(project_root)
    _require_frozen_revision(PHASE2B3A_CALCULATION_REVISION, "calculation revision")
    _require_frozen_revision(PHASE2B3A_EVIDENCE_PUBLICATION, "evidence publication")

    discovered_root = _single_line(
        _run_git(
            root,
            "rev-parse",
            "--show-toplevel",
            failure="project root is not a Git checkout",
        ),
        "project root is not a Git checkout",
    )
    if discovered_root != str(root):
        raise ValueError("project root must identify the Git checkout top level")

    branch_ref = _single_line(
        _run_git(
            root,
            "symbolic-ref",
            "--quiet",
            "HEAD",
            failure="Git checkout must have an attached branch",
        ),
        "Git checkout must have an attached branch",
    )
    if not branch_ref.startswith("refs/heads/") or branch_ref == "refs/heads/":
        raise ValueError("Git checkout must have an attached branch")
    branch = branch_ref.removeprefix("refs/heads/")

    head_revision = _canonical_revision(
        _run_git(
            root,
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
            failure="Git HEAD is not an exact canonical commit",
        ),
        "Git HEAD is not an exact canonical commit",
    )
    branch_revision = _canonical_revision(
        _run_git(
            root,
            "rev-parse",
            "--verify",
            f"{branch_ref}^{{commit}}",
            failure="attached branch does not identify an exact commit",
        ),
        "attached branch does not identify an exact commit",
    )
    if branch_revision != head_revision:
        raise ValueError("attached branch and HEAD revisions do not match")

    status = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        failure="unable to verify clean Git checkout",
    )
    if status:
        raise ValueError("Git checkout must be clean")

    _run_git(
        root,
        "merge-base",
        "--is-ancestor",
        PHASE2B3A_CALCULATION_REVISION,
        head_revision,
        failure="Git HEAD must descend from the frozen calculation revision",
    )
    _run_git(
        root,
        "merge-base",
        "--is-ancestor",
        PHASE2B3A_EVIDENCE_PUBLICATION,
        head_revision,
        failure="Git HEAD must descend from the frozen evidence publication",
    )

    return Phase2B3BRevision(
        branch=branch,
        head_revision=head_revision,
        calculation_revision=PHASE2B3A_CALCULATION_REVISION,
        evidence_publication=PHASE2B3A_EVIDENCE_PUBLICATION,
    )


def verify_recorded_phase2b3b_revision(
    project_root: Path, recorded_revision: str
) -> str:
    """Revalidate a recorded producer commit inside a trusted local checkout.

    The producer commit may precede the verifier checkout, but it must exist as
    the exact recorded object, descend from both frozen Phase 2B3-A commits,
    and be an ancestor of the clean attached verifier HEAD.
    """

    if (
        type(recorded_revision) is not str
        or _REVISION_PATTERN.fullmatch(recorded_revision) is None
    ):
        raise ValueError("recorded producer revision is not canonical")

    checkout = verify_phase2b3b_revision(project_root)
    resolved_revision = _canonical_revision(
        _run_git(
            project_root,
            "rev-parse",
            "--verify",
            f"{recorded_revision}^{{commit}}",
            failure="recorded producer revision does not exist",
        ),
        "recorded producer revision does not exist",
    )
    if resolved_revision != recorded_revision:
        raise ValueError("recorded producer revision is not exact")

    _run_git(
        project_root,
        "merge-base",
        "--is-ancestor",
        PHASE2B3A_CALCULATION_REVISION,
        recorded_revision,
        failure="recorded producer revision must descend from the frozen calculation revision",
    )
    _run_git(
        project_root,
        "merge-base",
        "--is-ancestor",
        PHASE2B3A_EVIDENCE_PUBLICATION,
        recorded_revision,
        failure="recorded producer revision must descend from the frozen evidence publication",
    )
    _run_git(
        project_root,
        "merge-base",
        "--is-ancestor",
        recorded_revision,
        checkout.head_revision,
        failure="recorded producer revision must be an ancestor of Git HEAD",
    )
    return resolved_revision
