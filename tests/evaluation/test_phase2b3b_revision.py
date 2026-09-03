"""Local-only tests for the Phase 2B3-B Git revision gate."""

from __future__ import annotations

import importlib
import subprocess
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from types import ModuleType

import pytest


@dataclass(frozen=True)
class _Repository:
    root: Path
    calculation: str
    publication: str
    head: str


def _module() -> ModuleType:
    return importlib.import_module("trustsr.evaluation.phase2b3b_revision")


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    return completed.stdout.strip()


def _commit(root: Path, message: str) -> str:
    tracked = root / "tracked.txt"
    prior = tracked.read_text(encoding="utf-8") if tracked.exists() else ""
    tracked.write_text(f"{prior}{message}\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-q", "-m", message)
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path: Path) -> _Repository:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "-q", "-b", "phase2b3b-test")
    _git(root, "config", "user.name", "Phase2B3B Test")
    _git(root, "config", "user.email", "phase2b3b@example.invalid")
    _commit(root, "root")
    calculation = _commit(root, "calculation")
    publication = _commit(root, "publication")
    head = _commit(root, "head")
    return _Repository(root=root, calculation=calculation, publication=publication, head=head)


def _pin_required_ancestors(monkeypatch: pytest.MonkeyPatch, repository: _Repository) -> ModuleType:
    module = _module()
    monkeypatch.setattr(module, "PHASE2B3A_CALCULATION_REVISION", repository.calculation)
    monkeypatch.setattr(module, "PHASE2B3A_EVIDENCE_PUBLICATION", repository.publication)
    return module


def test_accepts_clean_attached_checkout_and_returns_host_free_frozen_identity(
    repository: _Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _pin_required_ancestors(monkeypatch, repository)

    result = module.verify_phase2b3b_revision(repository.root)

    assert result.branch == "phase2b3b-test"
    assert result.head_revision == repository.head
    assert result.calculation_revision == repository.calculation
    assert result.evidence_publication == repository.publication
    assert str(repository.root) not in repr(result)
    with pytest.raises(FrozenInstanceError):
        result.head_revision = "0" * 40


def test_frozen_required_revisions_are_exact_canonical_commits() -> None:
    module = _module()

    assert module.PHASE2B3A_CALCULATION_REVISION == (
        "58694420c3c0e11d495953a1963c71b997261601"
    )
    assert module.PHASE2B3A_EVIDENCE_PUBLICATION == (
        "b386d4b38c9f3725107eed178829955d442f5601"
    )


def test_rejects_checkout_whose_head_does_not_descend_from_required_revisions(
    repository: _Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _pin_required_ancestors(monkeypatch, repository)
    root_revision = _git(repository.root, "rev-list", "--max-parents=0", "HEAD")
    _git(repository.root, "switch", "-q", "-c", "unrelated", root_revision)
    _commit(repository.root, "unrelated")

    with pytest.raises(ValueError, match="calculation revision"):
        module.verify_phase2b3b_revision(repository.root)


def test_rejects_detached_checkout(
    repository: _Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _pin_required_ancestors(monkeypatch, repository)
    _git(repository.root, "switch", "-q", "--detach", repository.head)

    with pytest.raises(ValueError, match="attached branch"):
        module.verify_phase2b3b_revision(repository.root)


@pytest.mark.parametrize("dirty_kind", ("tracked", "untracked"))
def test_rejects_dirty_checkout(
    repository: _Repository,
    monkeypatch: pytest.MonkeyPatch,
    dirty_kind: str,
) -> None:
    module = _pin_required_ancestors(monkeypatch, repository)
    if dirty_kind == "tracked":
        (repository.root / "tracked.txt").write_text("changed\n", encoding="utf-8")
    else:
        (repository.root / "untracked.txt").write_text("new\n", encoding="utf-8")

    with pytest.raises(ValueError, match="clean"):
        module.verify_phase2b3b_revision(repository.root)


@pytest.mark.parametrize("invalid_root_kind", ("symlink", "subdirectory", "not_git"))
def test_rejects_noncanonical_or_non_root_directory(
    repository: _Repository,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    invalid_root_kind: str,
) -> None:
    module = _pin_required_ancestors(monkeypatch, repository)
    if invalid_root_kind == "symlink":
        candidate = tmp_path / "repository-link"
        candidate.symlink_to(repository.root, target_is_directory=True)
    elif invalid_root_kind == "subdirectory":
        candidate = repository.root / "nested"
        candidate.mkdir()
    else:
        candidate = tmp_path / "not-git"
        candidate.mkdir()

    with pytest.raises(ValueError, match="project root") as caught:
        module.verify_phase2b3b_revision(candidate)
    assert str(candidate) not in str(caught.value)


def test_subprocess_exception_fails_closed_without_host_path(
    repository: _Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _pin_required_ancestors(monkeypatch, repository)

    def fail_run(*args: object, **kwargs: object) -> object:
        raise OSError(f"cannot execute in {repository.root}")

    monkeypatch.setattr(module.subprocess, "run", fail_run)

    with pytest.raises(ValueError, match="inspect local Git checkout") as caught:
        module.verify_phase2b3b_revision(repository.root)
    assert str(repository.root) not in str(caught.value)


def test_uses_explicit_local_git_argv_without_shell_or_network_command(
    repository: _Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _pin_required_ancestors(monkeypatch, repository)
    real_run = subprocess.run
    calls: list[tuple[list[str], dict[str, object]]] = []

    def record_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        return real_run(argv, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", record_run)

    module.verify_phase2b3b_revision(repository.root)

    assert calls
    assert all(argv[0] == "git" and argv[1] == "-C" for argv, _ in calls)
    assert all(kwargs.get("shell") is False for _, kwargs in calls)
    assert not {"fetch", "pull", "push", "ls-remote"}.intersection(
        argument for argv, _ in calls for argument in argv
    )


def test_revalidates_recorded_producer_revision_in_trusted_checkout(
    repository: _Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _pin_required_ancestors(monkeypatch, repository)

    result = module.verify_recorded_phase2b3b_revision(
        repository.root, repository.publication
    )

    assert result == repository.publication


@pytest.mark.parametrize(
    "recorded_revision",
    (
        "c" * 40,
        "HEAD",
        "A" * 40,
        "0" * 39,
    ),
)
def test_rejects_fabricated_or_noncanonical_recorded_revision(
    repository: _Repository,
    monkeypatch: pytest.MonkeyPatch,
    recorded_revision: str,
) -> None:
    module = _pin_required_ancestors(monkeypatch, repository)

    with pytest.raises(ValueError, match="recorded producer revision"):
        module.verify_recorded_phase2b3b_revision(repository.root, recorded_revision)


def test_rejects_recorded_revision_that_does_not_descend_from_frozen_commits(
    repository: _Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _pin_required_ancestors(monkeypatch, repository)
    root_revision = _git(repository.root, "rev-list", "--max-parents=0", "HEAD")
    _git(repository.root, "switch", "-q", "-c", "unrelated-producer", root_revision)
    unrelated = _commit(repository.root, "unrelated producer")
    _git(repository.root, "switch", "-q", "phase2b3b-test")

    with pytest.raises(ValueError, match="calculation revision"):
        module.verify_recorded_phase2b3b_revision(repository.root, unrelated)


def test_rejects_recorded_revision_that_is_not_ancestor_of_verifier_head(
    repository: _Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _pin_required_ancestors(monkeypatch, repository)
    future = _commit(repository.root, "future producer")
    _git(repository.root, "switch", "-q", "-c", "verifier", repository.head)

    with pytest.raises(ValueError, match="ancestor of Git HEAD"):
        module.verify_recorded_phase2b3b_revision(repository.root, future)
