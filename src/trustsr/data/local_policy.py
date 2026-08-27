"""Repository policy that prevents tracked local SEN2NAIPv2 pixel data."""

import os
import subprocess
from pathlib import Path

_DATASET_DIRECTORY = Path("artifacts/datasets")
_ALLOWED_DATASET_SUFFIXES = frozenset({".json", ".md"})
_MAX_DATASET_FILE_BYTES = 1_048_576


def _git_output(repo_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ("git", "-C", str(repo_root), *args),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        details = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed for {repo_root}: {details}")
    return result.stdout


def _indexed_blob_sizes(repo_root: Path) -> tuple[tuple[Path, int], ...]:
    records = _git_output(repo_root, "ls-files", "--stage", "-z").split(b"\0")
    blob_sizes: dict[bytes, int] = {}
    entries: list[tuple[Path, int]] = []
    for record in records:
        if not record:
            continue
        metadata, separator, encoded_path = record.partition(b"\t")
        fields = metadata.split()
        if not separator or not encoded_path or len(fields) != 3:
            raise RuntimeError("git ls-files --stage returned a malformed index entry")
        mode, object_id, stage = fields
        if not mode.isdigit() or not object_id or stage != b"0":
            raise RuntimeError("git ls-files --stage returned an unsupported index entry")
        if object_id not in blob_sizes:
            object_type = _git_output(repo_root, "cat-file", "-t", object_id.decode()).strip()
            if object_type != b"blob":
                raise RuntimeError("git index entry does not reference a blob")
            size_output = _git_output(repo_root, "cat-file", "-s", object_id.decode()).strip()
            try:
                blob_sizes[object_id] = int(size_output)
            except ValueError as error:
                raise RuntimeError("git cat-file returned a malformed blob size") from error
        entries.append((Path(os.fsdecode(encoded_path)), blob_sizes[object_id]))
    return tuple(entries)


def tracked_data_policy_violations(repo_root: Path) -> tuple[str, ...]:
    """Return sorted policy violations for Git-tracked files in ``repo_root`` only."""
    resolved_root = repo_root.resolve()

    violations: list[str] = []
    for relative_path, indexed_size in _indexed_blob_sizes(resolved_root):
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
        observed_sizes = [indexed_size]
        if resolved_path.exists():
            try:
                observed_sizes.append(resolved_path.stat().st_size)
            except OSError as error:
                violations.append(f"cannot inspect tracked dataset file: {relative_path} ({error})")
                continue
        if max(observed_sizes) > _MAX_DATASET_FILE_BYTES:
            violations.append(
                "tracked artifacts/datasets file exceeds 1_048_576 bytes: "
                f"{relative_path} ({max(observed_sizes)} bytes)"
            )

    return tuple(sorted(violations))
