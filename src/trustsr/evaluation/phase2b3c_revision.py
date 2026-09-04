"""Git computation-tree identity for a permit-bound Phase 2B3-C revision."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from trustsr.evaluation.phase2b3c_evidence import PHASE2B3B_PUBLICATION_COMMIT
from trustsr.jsonio import canonical_json

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_TIMEOUT_SECONDS = 10.0
_COMPUTATION_PATHS = ("pyproject.toml", "uv.lock", "src/trustsr")
_PERMIT_PATH = (
    "artifacts/phase2b3c/sen2naipv2-internal-test-access-authorization-v1.json"
)


@dataclass(frozen=True)
class VerifiedRevision:
    """Local checkout identity bound to an immutable computation tree."""

    branch: str
    head_revision: str
    implementation_revision: str
    computation_tree_sha256: str


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


def _run_git(root: Path, *arguments: str, failure: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            stdin=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("unable to inspect local Git checkout") from exc
    if completed.returncode != 0:
        raise ValueError(failure)
    return completed.stdout


def _single_line(output: str, failure: str) -> str:
    if not output.endswith("\n") or output.count("\n") != 1:
        raise ValueError(failure)
    return output[:-1]


def _require_revision(revision: object, label: str) -> str:
    if type(revision) is not str or _REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError(f"{label} is not a canonical revision")
    return revision


def _resolve_revision(root: Path, revision: str, label: str) -> str:
    resolved = _single_line(
        _run_git(
            root,
            "rev-parse",
            "--verify",
            f"{revision}^{{commit}}",
            failure=f"{label} does not exist",
        ),
        f"{label} does not exist",
    )
    if resolved != revision:
        raise ValueError(f"{label} is not exact")
    return resolved


def _require_top_level(root: Path) -> None:
    discovered = _single_line(
        _run_git(
            root,
            "rev-parse",
            "--show-toplevel",
            failure="project root is not a Git checkout",
        ),
        "project root is not a Git checkout",
    )
    if discovered != str(root):
        raise ValueError("project root must identify the Git checkout top level")


def phase2b3c_computation_tree_sha256(project_root: Path, revision: str) -> str:
    """Hash Git object identities for all Phase 2B3-C computation inputs."""

    root = _canonical_project_root(project_root)
    _require_top_level(root)
    revision = _require_revision(revision, "implementation revision")
    _resolve_revision(root, revision, "implementation revision")
    output = _run_git(
        root,
        "ls-tree",
        "-r",
        "--full-tree",
        revision,
        "--",
        *_COMPUTATION_PATHS,
        failure="unable to inspect the computation tree",
    )
    entries: list[dict[str, str]] = []
    observed: set[str] = set()
    for line in output.splitlines():
        try:
            metadata, path = line.split("\t", 1)
            mode, object_type, object_id = metadata.split(" ", 2)
        except ValueError as exc:
            raise ValueError("computation tree listing is invalid") from exc
        if (
            path in observed
            or object_type != "blob"
            or _REVISION_PATTERN.fullmatch(object_id) is None
            or mode not in {"100644", "100755"}
        ):
            raise ValueError("computation tree listing is invalid")
        observed.add(path)
        entries.append(
            {"mode": mode, "object": object_id, "path": path, "type": object_type}
        )
    if "pyproject.toml" not in observed or "uv.lock" not in observed:
        raise ValueError("computation tree is missing required files")
    if not any(path.startswith("src/trustsr/") for path in observed):
        raise ValueError("computation tree is missing trustsr sources")
    entries.sort(key=lambda entry: entry["path"])
    return hashlib.sha256(canonical_json(entries)).hexdigest()


def verify_phase2b3c_implementation_revision(
    project_root: Path,
    implementation_revision: str,
    computation_tree_sha256: str,
) -> VerifiedRevision:
    """Verify the clean attached checkout is the frozen implementation or permit-only."""

    root = _canonical_project_root(project_root)
    _require_top_level(root)
    implementation_revision = _require_revision(
        implementation_revision, "implementation revision"
    )
    if (
        type(computation_tree_sha256) is not str
        or _SHA256_PATTERN.fullmatch(computation_tree_sha256) is None
    ):
        raise ValueError("computation tree digest is not canonical")
    _resolve_revision(root, implementation_revision, "implementation revision")
    publication = _require_revision(PHASE2B3B_PUBLICATION_COMMIT, "publication revision")
    _resolve_revision(root, publication, "publication revision")

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
    head = _single_line(
        _run_git(
            root,
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
            failure="Git HEAD is not an exact commit",
        ),
        "Git HEAD is not an exact commit",
    )
    _require_revision(head, "Git HEAD revision")

    dirty = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *_COMPUTATION_PATHS,
        failure="unable to verify clean computation tree",
    )
    if dirty:
        raise ValueError("computation tree must be clean")
    _run_git(
        root,
        "merge-base",
        "--is-ancestor",
        publication,
        implementation_revision,
        failure="implementation revision must descend from the Phase 2B3-B publication",
    )
    _run_git(
        root,
        "merge-base",
        "--is-ancestor",
        implementation_revision,
        head,
        failure="Git HEAD must descend from the implementation revision",
    )

    implementation_digest = phase2b3c_computation_tree_sha256(root, implementation_revision)
    head_digest = phase2b3c_computation_tree_sha256(root, head)
    if (
        implementation_digest != computation_tree_sha256
        or head_digest != computation_tree_sha256
    ):
        raise ValueError("computation tree does not match the frozen digest")

    changed = _run_git(
        root,
        "diff",
        "--name-only",
        implementation_revision,
        head,
        failure="unable to inspect post-implementation changes",
    ).splitlines()
    if any(path != _PERMIT_PATH for path in changed):
        raise ValueError("only the exact internal-test access permit may follow implementation")

    return VerifiedRevision(
        branch=branch_ref.removeprefix("refs/heads/"),
        head_revision=head,
        implementation_revision=implementation_revision,
        computation_tree_sha256=computation_tree_sha256,
    )
