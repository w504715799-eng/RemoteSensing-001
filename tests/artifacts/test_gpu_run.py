import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trustsr.artifacts.gpu_run import (
    GPUHardwareSnapshot,
    capture_gpu_hardware,
    collect_gpu_environment,
    verify_artifact_manifest,
    write_artifact_manifest,
)

REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("name", "compute_capability"),
    [
        ("NVIDIA GeForce RTX 3090", "8.6"),
        ("NVIDIA GeForce RTX 4090", "8.9"),
        ("NVIDIA A100", "8.0"),
        ("NVIDIA future accelerator", "10.2"),
    ],
)
def test_hardware_snapshot_accepts_single_gpu_with_supported_capability_before_model_construction(
    name: str, compute_capability: str
) -> None:
    """A product-name gate would wrongly reject compatible rental substitutions."""

    def runner(argv, **_kwargs):
        output = (
            f"{name}, GPU-verified, 555.1, 24564, 24058, {compute_capability}\n"
            if "--query-compute-apps=pid" not in argv
            else ""
        )
        return subprocess.CompletedProcess(argv, 0, output, "")

    snapshot = capture_gpu_hardware(
        command_runner=runner,
        cuda_available=lambda: True,
        current_pid=os.getpid(),
    )

    assert snapshot.name == name
    assert snapshot.memory_total_mib == 24564
    assert snapshot.memory_free_mib == 24058
    assert snapshot.compute_capability == compute_capability


def test_collect_gpu_environment_binds_reviewed_root_snapshot_and_active_prefix_uv(
    monkeypatch, tmp_path
):
    calls = []
    lock_digest = hashlib.sha256((REPOSITORY / "uv.lock").read_bytes()).hexdigest()
    sensitive_working_directory = tmp_path / "sensitive-working-directory"
    sensitive_working_directory.mkdir()
    conflicting_bin = tmp_path / "conflicting-bin"
    conflicting_bin.mkdir()
    (conflicting_bin / "uv").write_text("#!/bin/sh\necho 'uv 99.0.0'\n")
    (conflicting_bin / "uv").chmod(0o755)
    monkeypatch.setenv("PATH", f"{conflicting_bin}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.chdir(sensitive_working_directory)
    active_uv = Path(sys.executable).absolute().parent / "uv"

    def runner(argv, **kwargs):
        assert Path.cwd() == sensitive_working_directory
        calls.append((argv, kwargs))
        outputs = {
            (
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,memory.free,compute_cap",
                "--format=csv,noheader,nounits",
            ): "NVIDIA A100, GPU-uuid, 555.1, 24576, 20000, 8.6\n",
            (
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ): "",
            ("nvcc", "--version"): "Cuda compilation tools, release 12.4, V12.4.1\n",
            ("conda", "--version"): "conda 24.7.1\n",
            (str(active_uv), "--version"): "uv 0.12.5\n",
            ("git", "-C", str(REPOSITORY.resolve()), "rev-parse", "HEAD"): "reviewed-commit\n",
        }
        return subprocess.CompletedProcess(argv, 0, outputs[tuple(argv)], "")

    monkeypatch.setattr("trustsr.artifacts.gpu_run._read_cgroup_limit", lambda name: "max")
    monkeypatch.setattr(
        "trustsr.artifacts.gpu_run._package_version", lambda name: f"{name}-version"
    )
    monkeypatch.setattr("trustsr.artifacts.gpu_run._utc_now", lambda: "2026-08-27T00:00:00Z")

    snapshot = capture_gpu_hardware(command_runner=runner, cuda_available=lambda: True)
    result = collect_gpu_environment(
        hardware_snapshot=snapshot,
        project_root=REPOSITORY,
        command_runner=runner,
    )

    assert result["schema_version"] == 1
    assert result["run_started_utc"] == "2026-08-27T00:00:00Z"
    assert result["git_commit"] == "reviewed-commit"
    assert result["gpu"] == {
        "name": "NVIDIA A100",
        "uuid": "GPU-uuid",
        "driver_version": "555.1",
        "memory_total_mib": 24576,
        "memory_free_mib": 20000,
        "compute_capability": "8.6",
    }
    assert result["limits"] == {"cpu": "max", "memory": "max"}
    assert result["runtime"]["python"]
    assert result["runtime"]["uv"] == "0.12.5"
    assert result["runtime"]["cuda_toolkit"] == "12.4"
    assert result["runtime"]["opensr_model"] == "opensr-model-version"
    assert result["dependency_lock_sha256"] == lock_digest
    assert result["model_provenance"]["name"] == "ldsr-s2-x4"
    assert all(isinstance(argv, list) for argv, _ in calls)
    assert all(kwargs["shell"] is False for _, kwargs in calls)
    text = json.dumps(result)
    assert str(sensitive_working_directory) not in text
    for forbidden in ("ssh", "password", "private", "github", "hostname", "port"):
        assert forbidden not in text.lower()


def test_collect_gpu_environment_records_versions_not_trailing_build_metadata(monkeypatch):
    """The runtime manifest must retain each tool's version token."""
    active_uv = Path(sys.executable).absolute().parent / "uv"
    outputs = {
        ("conda", "--version"): "conda 26.5.3\n",
        (str(active_uv), "--version"): "uv 0.12.5 (x86_64-unknown-linux-gnu)\n",
        ("nvcc", "--version"): "Cuda compilation tools, release 12.4, V12.4.1\n",
        ("git", "-C", str(REPOSITORY.resolve()), "rev-parse", "HEAD"): "reviewed-commit\n",
    }

    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, outputs[tuple(argv)], "")

    monkeypatch.setattr("trustsr.artifacts.gpu_run._read_cgroup_limit", lambda name: "max")
    snapshot = GPUHardwareSnapshot(
        name="NVIDIA A100",
        uuid="GPU-uuid",
        driver_version="555.1",
        memory_total_mib=24576,
        memory_free_mib=20000,
        compute_capability="8.6",
        compute_pids=(),
    )

    runtime = collect_gpu_environment(
        hardware_snapshot=snapshot,
        project_root=REPOSITORY,
        command_runner=runner,
    )["runtime"]

    assert runtime["uv"] == "0.12.5"
    assert runtime["conda"] == "26.5.3"


@pytest.mark.parametrize(
    ("conda_output", "uv_output", "runtime_key"),
    [
        ("conda release-26\n", "uv 0.12.5\n", "conda"),
        ("conda 26.5.3\n", "uv release-0.12\n", "uv"),
    ],
)
def test_collect_gpu_environment_marks_malformed_tool_versions_unavailable(
    monkeypatch, conda_output: str, uv_output: str, runtime_key: str
) -> None:
    """A non-version token must not enter the runtime manifest."""
    active_uv = Path(sys.executable).absolute().parent / "uv"
    outputs = {
        ("conda", "--version"): conda_output,
        (str(active_uv), "--version"): uv_output,
        ("nvcc", "--version"): "Cuda compilation tools, release 12.4, V12.4.1\n",
        ("git", "-C", str(REPOSITORY.resolve()), "rev-parse", "HEAD"): "reviewed-commit\n",
    }

    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 0, outputs[tuple(argv)], "")

    monkeypatch.setattr("trustsr.artifacts.gpu_run._read_cgroup_limit", lambda name: "max")
    snapshot = GPUHardwareSnapshot(
        name="NVIDIA A100",
        uuid="GPU-uuid",
        driver_version="555.1",
        memory_total_mib=24576,
        memory_free_mib=20000,
        compute_capability="8.6",
        compute_pids=(),
    )

    runtime = collect_gpu_environment(
        hardware_snapshot=snapshot,
        project_root=REPOSITORY,
        command_runner=runner,
    )["runtime"]

    assert runtime[runtime_key] == "unavailable"


@pytest.mark.parametrize(
    ("gpu_output", "process_output", "cuda_available", "message"),
    [
        (
            "\n".join(
                [
                    "NVIDIA A100, GPU-one, 555.1, 24576, 20000, 8.6",
                    "NVIDIA A100, GPU-two, 555.1, 24576, 20000, 8.6",
                ]
            ),
            "",
            True,
            "exactly one GPU",
        ),
        (
            "NVIDIA A100, GPU-one, 555.1, 24576, 18431, 8.6",
            "",
            True,
            "18 GiB",
        ),
        (
            "NVIDIA A100, GPU-one, 555.1, 24576, 20000, 8.6",
            str(os.getpid() + 1),
            True,
            "foreign",
        ),
        (
            "NVIDIA A100, GPU-one, 555.1, 24576, 20000, 8.6",
            "",
            False,
            "CUDA",
        ),
    ],
)
def test_hardware_snapshot_fails_closed_on_an_unacceptable_initial_state(
    gpu_output, process_output, cuda_available, message
):
    def runner(argv, **_kwargs):
        output = process_output if "--query-compute-apps=pid" in argv else gpu_output
        return subprocess.CompletedProcess(argv, 0, output, "")

    with pytest.raises(RuntimeError, match=message):
        capture_gpu_hardware(
            command_runner=runner,
            cuda_available=lambda: cuda_available,
            current_pid=os.getpid(),
        )


@pytest.mark.parametrize(
    "compute_capability",
    ["7.5", "7.9", "", "8", "8.", ".0", "8.0.1", "NaN", "infinity", "-8.0", "8.-1"],
)
def test_hardware_snapshot_rejects_unsupported_or_malformed_compute_capability(
    compute_capability: str,
) -> None:
    """Accepting a non-8.x-capable GPU or non-integer version is unsafe."""

    def runner(argv, **_kwargs):
        output = (
            f"NVIDIA T4, GPU-one, 555.1, 24576, 20000, {compute_capability}"
            if "--query-compute-apps=pid" not in argv
            else ""
        )
        return subprocess.CompletedProcess(argv, 0, output, "")

    with pytest.raises(RuntimeError, match="compute capability"):
        capture_gpu_hardware(command_runner=runner, cuda_available=lambda: True)


@pytest.mark.parametrize(
    "total, free",
    [("0", "0"), ("24576", "-1"), ("24576", "24577"), ("unknown", "20000")],
)
def test_hardware_snapshot_rejects_invalid_memory_values_before_model_construction(
    total: str, free: str
) -> None:
    """Invalid memory accounting must not make a capability-compatible GPU runnable."""

    def runner(argv, **_kwargs):
        output = (
            f"NVIDIA A100, GPU-one, 555.1, {total}, {free}, 8.0"
            if "--query-compute-apps=pid" not in argv
            else ""
        )
        return subprocess.CompletedProcess(argv, 0, output, "")

    with pytest.raises(RuntimeError, match="GPU memory"):
        capture_gpu_hardware(command_runner=runner, cuda_available=lambda: True)


def test_hardware_snapshot_is_frozen() -> None:
    snapshot = GPUHardwareSnapshot(
        name="NVIDIA A100",
        uuid="GPU-one",
        driver_version="555.1",
        memory_total_mib=24576,
        memory_free_mib=20000,
        compute_capability="8.6",
        compute_pids=(),
    )

    with pytest.raises(AttributeError):
        snapshot.memory_free_mib = 1


def test_artifact_manifest_is_sorted_confined_and_verifies_contents(tmp_path: Path):
    (tmp_path / "phase1b").mkdir()
    (tmp_path / "phase1b" / "z.json").write_text("z")
    (tmp_path / "phase1b" / "a.json").write_text("alpha")

    path = write_artifact_manifest(
        tmp_path, [Path("phase1b/z.json"), Path("phase1b/a.json")]
    )

    manifest = json.loads(path.read_text())
    assert manifest["schema_version"] == 1
    assert [entry["path"] for entry in manifest["files"]] == [
        "phase1b/a.json",
        "phase1b/z.json",
    ]
    assert manifest["files"][0]["size"] == 5
    assert manifest["files"][0]["sha256"] == hashlib.sha256(b"alpha").hexdigest()
    verify_artifact_manifest(tmp_path, path)


@pytest.mark.parametrize(
    "relative", [Path("/tmp/outside"), Path("../outside"), Path("phase1b/../x")]
)
def test_artifact_manifest_rejects_nonconfined_paths(tmp_path: Path, relative: Path):
    with pytest.raises(ValueError):
        write_artifact_manifest(tmp_path, [relative])


def test_artifact_manifest_rejects_symlink_and_modified_or_malformed_entries(tmp_path: Path):
    target = tmp_path.parent / "outside-target.json"
    target.write_text("content")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        write_artifact_manifest(tmp_path, [Path("link.json")])

    tracked = tmp_path / "tracked.json"
    tracked.write_text("content")

    manifest_path = write_artifact_manifest(tmp_path, [Path("tracked.json")])
    tracked.write_text("changed")
    with pytest.raises(ValueError, match="mismatch"):
        verify_artifact_manifest(tmp_path, manifest_path)
    payload = json.loads(manifest_path.read_text())
    payload["files"][0]["sha256"] = "not-a-digest"
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        verify_artifact_manifest(tmp_path, manifest_path)


def test_artifact_manifest_rejects_a_missing_named_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="missing"):
        write_artifact_manifest(tmp_path, [Path("missing.json")])


def test_artifact_manifest_rejects_a_symlinked_root(tmp_path: Path):
    real_root = tmp_path / "real"
    real_root.mkdir()
    (real_root / "output.json").write_text("content")
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="root"):
        write_artifact_manifest(linked_root, [Path("output.json")])


def test_artifact_manifest_rejects_symlinked_output_directory(tmp_path: Path):
    (tmp_path / "output.json").write_text("content")
    outside = tmp_path.parent / "outside-phase"
    outside.mkdir(exist_ok=True)
    (tmp_path / "phase1b").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        write_artifact_manifest(tmp_path, [Path("output.json")])
