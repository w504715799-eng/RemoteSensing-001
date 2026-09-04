"""Git computation-tree identity for a permit-bound Phase 2B3-C revision."""

from __future__ import annotations

import importlib
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


def _module():
    return importlib.import_module("trustsr.evaluation.phase2b3c_revision")


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    ).stdout.strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "repository"
    (root / "src" / "trustsr").mkdir(parents=True)
    (root / "artifacts" / "phase2b3c").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / "src" / "trustsr" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.name", "Phase2B3C Test")
    _git(root, "config", "user.email", "phase2b3c@example.invalid")
    publication = _commit(root, "publication")
    (root / "docs").mkdir()
    (root / "docs" / "design.md").write_text("design\n", encoding="utf-8")
    implementation = _commit(root, "implementation")
    return root, publication, implementation


def test_computation_tree_is_stable_across_noncomputation_commits(
    repository: tuple[Path, str, str]
) -> None:
    module = _module()
    root, _, implementation = repository
    before = module.phase2b3c_computation_tree_sha256(root, implementation)
    (root / "docs" / "notes.md").write_text("notes\n", encoding="utf-8")
    later = _commit(root, "docs")

    assert module.phase2b3c_computation_tree_sha256(root, later) == before


def test_verifies_clean_attached_implementation_and_future_permit_only_commit(
    repository: tuple[Path, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    root, publication, implementation = repository
    monkeypatch.setattr(module, "PHASE2B3B_PUBLICATION_COMMIT", publication)
    digest = module.phase2b3c_computation_tree_sha256(root, implementation)

    first = module.verify_phase2b3c_implementation_revision(root, implementation, digest)
    permit = (
        root
        / "artifacts"
        / "phase2b3c"
        / "sen2naipv2-internal-test-access-authorization-v1.json"
    )
    permit.write_text("{}\n", encoding="utf-8")
    head = _commit(root, "permit")
    second = module.verify_phase2b3c_implementation_revision(root, implementation, digest)

    assert first.head_revision == implementation
    assert second.head_revision == head
    assert second.implementation_revision == implementation
    assert second.computation_tree_sha256 == digest
    with pytest.raises(FrozenInstanceError):
        second.head_revision = "0" * 40


def test_rejects_computation_change_after_implementation(
    repository: tuple[Path, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    root, publication, implementation = repository
    monkeypatch.setattr(module, "PHASE2B3B_PUBLICATION_COMMIT", publication)
    digest = module.phase2b3c_computation_tree_sha256(root, implementation)
    (root / "src" / "trustsr" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    _commit(root, "changed computation")

    with pytest.raises(ValueError, match="computation tree"):
        module.verify_phase2b3c_implementation_revision(root, implementation, digest)


def test_rejects_dirty_computation_tree(
    repository: tuple[Path, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    root, publication, implementation = repository
    monkeypatch.setattr(module, "PHASE2B3B_PUBLICATION_COMMIT", publication)
    digest = module.phase2b3c_computation_tree_sha256(root, implementation)
    (root / "src" / "trustsr" / "module.py").write_text("DIRTY = True\n", encoding="utf-8")

    with pytest.raises(ValueError, match="clean"):
        module.verify_phase2b3c_implementation_revision(root, implementation, digest)


@pytest.mark.parametrize("revision", ("HEAD", "A" * 40, "0" * 39, "f" * 40))
def test_rejects_noncanonical_or_missing_implementation_revision(
    repository: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
    revision: str,
) -> None:
    module = _module()
    root, publication, _ = repository
    monkeypatch.setattr(module, "PHASE2B3B_PUBLICATION_COMMIT", publication)

    with pytest.raises(ValueError, match="implementation revision"):
        module.verify_phase2b3c_implementation_revision(root, revision, "0" * 64)


def test_rejects_symlink_or_subdirectory_project_root(
    repository: tuple[Path, str, str], tmp_path: Path
) -> None:
    module = _module()
    root, _, implementation = repository
    digest = module.phase2b3c_computation_tree_sha256(root, implementation)
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)

    for candidate in (link, root / "src"):
        with pytest.raises(ValueError, match="project root"):
            module.verify_phase2b3c_implementation_revision(candidate, implementation, digest)
