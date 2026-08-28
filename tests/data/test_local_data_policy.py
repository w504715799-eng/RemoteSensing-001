import subprocess
from pathlib import Path

from trustsr.data.local_policy import tracked_data_policy_violations


def _git(repo_root: Path, *args: str) -> None:
    subprocess.run(
        ("git", "-C", str(repo_root), *args),
        check=True,
        capture_output=True,
        text=True,
    )


def _temporary_repository(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repository"
    repo_root.mkdir()
    _git(repo_root, "init")
    return repo_root


def _is_ignored(repo_root: Path, relative_path: str) -> bool:
    result = subprocess.run(
        ("git", "-C", str(repo_root), "check-ignore", "-q", "--no-index", "--", relative_path),
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def test_ignore_rules_keep_pinned_metadata_addable() -> None:
    """Catch ignore rules that expose generated artifacts or hide pinned metadata."""
    repo_root = Path(__file__).parents[2]

    assert not _is_ignored(repo_root, "artifacts/datasets/sen2naipv2-source-v1.json")
    assert _is_ignored(repo_root, "artifacts/datasets/sample.taco")
    assert _is_ignored(repo_root, "artifacts/datasets/cache/sample.json")
    assert _is_ignored(repo_root, "artifacts/datasets/pixels/sample.json")
    assert _is_ignored(repo_root, "artifacts/phase0/result.json")
    assert _is_ignored(repo_root, "artifacts/cache/predictions/result.json")
    assert _is_ignored(repo_root, "artifacts/phase1/result.json")
    assert _is_ignored(repo_root, "artifacts/remote-phase1b/result.json")


def test_policy_rejects_a_tracked_taco_file(tmp_path: Path) -> None:
    """Catch a policy regression that permits tracked local TACO pixel data."""
    repo_root = _temporary_repository(tmp_path)
    taco = repo_root / "artifacts/datasets/sample.taco"
    taco.parent.mkdir(parents=True)
    taco.write_bytes(b"not-real-pixels")
    _git(repo_root, "add", "-f", str(taco.relative_to(repo_root)))

    violations = tracked_data_policy_violations(repo_root)

    assert any("sample.taco" in violation for violation in violations)


def test_policy_rejects_disallowed_and_oversized_tracked_dataset_files(tmp_path: Path) -> None:
    """Catch a policy regression that allows tracked dataset payloads or large metadata."""
    repo_root = _temporary_repository(tmp_path)
    dataset_root = repo_root / "artifacts/datasets"
    dataset_root.mkdir(parents=True)
    (dataset_root / "payload.bin").write_bytes(b"not-real-pixels")
    (dataset_root / "oversized.json").write_bytes(b"0" * 1_048_577)
    _git(repo_root, "add", "-f", "artifacts/datasets/payload.bin")
    _git(repo_root, "add", "-f", "artifacts/datasets/oversized.json")

    violations = tracked_data_policy_violations(repo_root)

    assert any("payload.bin" in violation for violation in violations)
    assert any(
        "oversized.json" in violation and "1_048_576" in violation for violation in violations
    )


def test_policy_rejects_oversized_staged_metadata_after_worktree_shrink(tmp_path: Path) -> None:
    """Catch checks that let a small working copy hide an oversized indexed blob."""
    repo_root = _temporary_repository(tmp_path)
    metadata = repo_root / "artifacts/datasets/indexed-oversized.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(b"0" * 1_048_577)
    _git(repo_root, "add", "artifacts/datasets/indexed-oversized.json")
    metadata.write_bytes(b"{}")

    violations = tracked_data_policy_violations(repo_root)

    assert any("indexed-oversized.json" in violation for violation in violations)


def test_policy_rejects_oversized_worktree_metadata_after_small_stage(tmp_path: Path) -> None:
    """Catch checks that let a small staged blob hide an oversized working copy."""
    repo_root = _temporary_repository(tmp_path)
    metadata = repo_root / "artifacts/datasets/worktree-oversized.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(b"{}")
    _git(repo_root, "add", "artifacts/datasets/worktree-oversized.json")
    metadata.write_bytes(b"0" * 1_048_577)

    violations = tracked_data_policy_violations(repo_root)

    assert any("worktree-oversized.json" in violation for violation in violations)


def test_policy_accepts_this_repositorys_tracked_metadata() -> None:
    """Catch policy regressions that reject this repository's tracked metadata only."""
    repo_root = Path(__file__).parents[2]

    assert tracked_data_policy_violations(repo_root) == ()
