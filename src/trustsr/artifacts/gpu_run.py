"""Safe, portable manifests for the staged LDSR-S2 GPU workflow."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from trustsr.artifacts.predictions import canonical_json
from trustsr.models.ldsr_s2 import OPENSR_MODEL_VERSION
from trustsr.models.protocols import JsonScalar

_GPU_COMMAND = [
    "nvidia-smi",
    "--query-gpu=name,uuid,driver_version,memory.total,memory.free,compute_cap",
    "--format=csv,noheader,nounits",
]
_NVCC_COMMAND = ["nvcc", "--version"]
_CONDA_COMMAND = ["conda", "--version"]
_UV_COMMAND = ["uv", "--version"]
_GIT_COMMAND = ["git", "rev-parse", "HEAD"]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _read_cgroup_limit(name: str) -> str:
    try:
        return (Path("/sys/fs/cgroup") / name).read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"


def _run_text(
    command_runner: Callable[..., subprocess.CompletedProcess[str]], argv: list[str]
) -> str:
    try:
        result = command_runner(argv, check=False, capture_output=True, text=True, shell=False)
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    if result.returncode != 0:
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def _parse_gpu(text: str) -> dict[str, JsonScalar]:
    unavailable: dict[str, JsonScalar] = {
        "name": "unavailable",
        "uuid": "unavailable",
        "driver_version": "unavailable",
        "memory_total_mib": "unavailable",
        "memory_free_mib": "unavailable",
        "compute_capability": "unavailable",
    }
    if text == "unavailable":
        return unavailable
    fields = [part.strip() for part in text.splitlines()[0].split(",")]
    if len(fields) != 6:
        return unavailable
    try:
        total, free = int(fields[3]), int(fields[4])
    except ValueError:
        return unavailable
    return {
        "name": fields[0],
        "uuid": fields[1],
        "driver_version": fields[2],
        "memory_total_mib": total,
        "memory_free_mib": free,
        "compute_capability": fields[5],
    }


def _parse_nvcc_version(text: str) -> str:
    if text == "unavailable":
        return text
    marker = "release "
    if marker not in text:
        return "unavailable"
    return text.split(marker, 1)[1].split(",", 1)[0].strip() or "unavailable"


def _command_version(text: str) -> str:
    if text == "unavailable":
        return text
    parts = text.split()
    return parts[-1] if parts else "unavailable"


def _repository_lock() -> Path:
    for directory in Path(__file__).resolve().parents:
        candidate = directory / "uv.lock"
        if candidate.is_file():
            return candidate
    return Path("uv.lock")


def _lock_sha256() -> str:
    try:
        return hashlib.sha256(_repository_lock().read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


def collect_gpu_environment(
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, JsonScalar]:
    """Collect only an allowlisted, scalar GPU runtime record.

    Commands are fixed argv arrays so caller input can never become shell input.
    """
    gpu = _parse_gpu(_run_text(command_runner, _GPU_COMMAND))
    runtime: dict[str, JsonScalar] = {
        "python": platform.python_version(),
        "conda": _command_version(_run_text(command_runner, _CONDA_COMMAND)),
        "uv": _command_version(_run_text(command_runner, _UV_COMMAND)),
        "torch": _package_version("torch"),
        "cuda_toolkit": _parse_nvcc_version(_run_text(command_runner, _NVCC_COMMAND)),
        "cuda_runtime": torch.version.cuda or "unavailable",
        "opensr_model": _package_version("opensr-model"),
        "opensr_test": _package_version("opensr-test"),
    }
    return {
        "schema_version": 1,
        "run_started_utc": _utc_now(),
        "git_commit": _run_text(command_runner, _GIT_COMMAND),
        "runtime": runtime,
        "gpu": gpu,
        "limits": {
            "cpu": _read_cgroup_limit("cpu.max"),
            "memory": _read_cgroup_limit("memory.max"),
        },
        "determinism": {
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        },
        "dependency_lock_sha256": _lock_sha256(),
        "model_provenance": {
            "name": "ldsr-s2-x4",
            "scale": 4,
            "opensr_model_version": OPENSR_MODEL_VERSION,
        },
    }


def _safe_relative(root: Path, relative: Path) -> tuple[Path, str]:
    if root.is_symlink():
        raise ValueError("artifact manifest root must not be a symlink")
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact path must be a confined relative path")
    if not relative.parts or relative == Path("."):
        raise ValueError("artifact path must name a file")
    resolved_root = root.resolve(strict=False)
    candidate = root / relative
    # A symlink at any component is forbidden, including a link that happens to
    # resolve beneath root.  This eliminates time-of-check path redirection.
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("artifact path must not traverse a symlink")
    try:
        candidate.resolve(strict=False).relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("artifact path escapes manifest root") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"artifact is missing or not a regular file: {relative}")
    return candidate, relative.as_posix()


def _safe_output_path(root: Path, relative: Path) -> Path:
    """Constrain a not-yet-created output path without following symlinks."""
    if root.is_symlink():
        raise ValueError("artifact manifest root must not be a symlink")
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("artifact output path must be a confined relative path")
    resolved_root = root.resolve(strict=False)
    candidate = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("artifact output path must not traverse a symlink")
    try:
        candidate.resolve(strict=False).relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("artifact output path escapes manifest root") from exc
    return candidate


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False, mode="wb"
        ) as stream:
            temporary = Path(stream.name)
            stream.write(canonical_json(value))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        try:
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def stage_artifact_file(root: Path, source: Path, relative_destination: Path) -> Path:
    """Atomically copy one verified regular file into a confined artifact tree."""
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise ValueError("staged artifact source must be a regular non-symlink file")
    destination = _safe_output_path(Path(root), Path(relative_destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with source.open("rb") as input_stream, tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
            mode="wb",
        ) as output_stream:
            temporary = Path(output_stream.name)
            shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary, destination)
        temporary = None
        try:
            descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def write_artifact_manifest(root: Path, relative_paths: Sequence[Path]) -> Path:
    """Hash named regular files below ``root`` into a canonical manifest."""
    root = Path(root)
    entries: list[dict[str, JsonScalar]] = []
    seen: set[str] = set()
    for relative in relative_paths:
        file_path, portable = _safe_relative(root, Path(relative))
        if portable in seen:
            raise ValueError("artifact manifest paths must be unique")
        seen.add(portable)
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        entries.append({"path": portable, "size": file_path.stat().st_size, "sha256": digest})
    entries.sort(key=lambda entry: str(entry["path"]))
    manifest_path = _safe_output_path(root, Path("phase1b/artifact-manifest.json"))
    _atomic_json(manifest_path, {"schema_version": 1, "files": entries})
    return manifest_path


def verify_artifact_manifest(root: Path, manifest_path: Path) -> None:
    """Verify a strict manifest and every recorded file digest."""
    root = Path(root)
    try:
        manifest_file, _ = _safe_relative(root, Path(manifest_path).relative_to(root))
    except ValueError:
        raise ValueError("manifest path must be confined below root") from None
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("artifact manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "files"}:
        raise ValueError("artifact manifest has an invalid schema")
    if manifest["schema_version"] != 1 or not isinstance(manifest["files"], list):
        raise ValueError("artifact manifest has an invalid schema")
    paths: list[str] = []
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            raise ValueError("artifact manifest entry has an invalid schema")
        path, size, digest = entry["path"], entry["size"], entry["sha256"]
        if (
            not isinstance(path, str)
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("artifact manifest entry has an invalid digest or size")
        file_path, portable = _safe_relative(root, Path(path))
        actual_digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if file_path.stat().st_size != size or actual_digest != digest:
            raise ValueError(f"artifact manifest mismatch: {portable}")
        paths.append(portable)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("artifact manifest paths must be sorted and unique")
