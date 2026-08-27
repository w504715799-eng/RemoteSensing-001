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


def test_policy_accepts_this_repositorys_tracked_metadata() -> None:
    """Catch policy regressions that reject this repository's tracked metadata only."""
    repo_root = Path(__file__).parents[2]

    assert tracked_data_policy_violations(repo_root) == ()
