"""Repository policy that prevents tracked local SEN2NAIPv2 pixel data."""

import os
import subprocess
from pathlib import Path

_DATASET_DIRECTORY = Path("artifacts/datasets")
_ALLOWED_DATASET_SUFFIXES = frozenset({".json", ".md"})
_MAX_DATASET_FILE_BYTES = 1_048_576


def tracked_data_policy_violations(repo_root: Path) -> tuple[str, ...]:
    """Return sorted policy violations for Git-tracked files in ``repo_root`` only."""
    resolved_root = repo_root.resolve()
    result = subprocess.run(
        ("git", "-C", str(resolved_root), "ls-files", "-z"),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        details = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git ls-files failed for {resolved_root}: {details}")

    violations: list[str] = []
    for encoded_path in result.stdout.split(b"\0"):
        if not encoded_path:
            continue
        relative_path = Path(os.fsdecode(encoded_path))
        resolved_path = (resolved_root / relative_path).resolve()
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError:
            violations.append(f"tracked path resolves outside repository: {relative_path}")
            continue

        if str(relative_path).endswith(".taco"):
            violations.append(f"tracked TACO data file: {relative_path}")

        if relative_path.parts[:2] != _DATASET_DIRECTORY.parts:
            continue
        if relative_path.suffix not in _ALLOWED_DATASET_SUFFIXES:
            violations.append(
                "tracked artifacts/datasets file must use .json or .md: " f"{relative_path}"
            )
            continue
        try:
            size_bytes = resolved_path.stat().st_size
        except OSError as error:
            violations.append(f"cannot inspect tracked dataset file: {relative_path} ({error})")
            continue
        if size_bytes > _MAX_DATASET_FILE_BYTES:
            violations.append(
                "tracked artifacts/datasets file exceeds 1_048_576 bytes: "
                f"{relative_path} ({size_bytes} bytes)"
            )

    return tuple(sorted(violations))
