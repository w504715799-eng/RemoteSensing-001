"""Contract tests for the Phase 1B operator shell entry points."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPTS = REPOSITORY / "scripts" / "phase1b"


def _make_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _environment(tmp_path: Path, *, available_kib: int = 30_000_000) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir(parents=True)
    log = tmp_path / "commands.log"
    _make_executable(
        fake_bin / "df",
        "#!/usr/bin/env bash\nprintf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'\n"
        f"printf '/dev/fake 30000000 1 {available_kib} 1%% /fake\\n'\n",
    )
    _make_executable(
        fake_bin / "realpath",
        "#!/usr/bin/env bash\nvalue=\"${!#}\"\n"
        "if [[ -n \"${FAKE_REMOTE_ROOT:-}\" && \"$value\" == \"$FAKE_REMOTE_ROOT\" ]]; then\n"
        "  printf '/root/rivermind-data/test-root\\n'\n"
        "else\n"
        "  /usr/bin/realpath -e -- \"$value\"\n"
        "fi\n",
    )
    _make_executable(
        fake_bin / "conda",
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'conda %s\\n' \"$*\" >> \"$COMMAND_LOG\"\n"
        "prefix=\"\"\nfor ((index = 1; index <= $#; index++)); do\n"
        "  if [[ \"${!index}\" == --prefix ]]; then\n"
        "    next=$((index + 1)); prefix=\"${!next}\"\n  fi\ndone\n"
        "mkdir -p \"$prefix/bin\"\n"
        "printf '#!/usr/bin/env bash\\nif [[ \"${1:-}\" == --version ]]; then echo \"Python 3.12.4\"; exit 0; fi\\n"
        "printf \"python %%s\\\\n\" \"$*\" >> \"$COMMAND_LOG\"\\n' > \"$prefix/bin/python\"\n"
        "chmod +x \"$prefix/bin/python\"\n"
        "printf '#!/usr/bin/env bash\\nif [[ \"${1:-}\" == --version ]]; then echo \"uv 0.12.5\"; exit 0; fi\\n"
        "printf \"uv %%s env=%%s\\\\n\" \"$*\" \"${UV_PROJECT_ENVIRONMENT:-}\" >> \"$COMMAND_LOG\"\\n' > \"$prefix/bin/uv\"\n"
        "chmod +x \"$prefix/bin/uv\"\n",
    )
    _make_executable(
        fake_bin / "rsync",
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'rsync %s\\n' \"$*\" >> \"$COMMAND_LOG\"\n"
        "if [[ -n \"${RSYNC_ARGUMENT_DIR:-}\" ]]; then\n"
        "  mkdir -p \"$RSYNC_ARGUMENT_DIR\"; call=$(find \"$RSYNC_ARGUMENT_DIR\" -type f | wc -l)\n"
        "  printf '%s\\0' \"$@\" > \"$RSYNC_ARGUMENT_DIR/$call\"\nfi\n"
        "args=(\"$@\"); source=\"${args[$(( $# - 2 ))]}\"; destination=\"${args[$(( $# - 1 ))]}\"\n"
        "source=\"${source#*:}\"\n"
        "if [[ -n \"${REMOTE_FIXTURE_ROOT:-}\" ]]; then source=\"${REMOTE_FIXTURE_ROOT}${source#/root/rivermind-data/phase1b}\"; fi\n"
        "mkdir -p \"$destination\"\ncp \"$source\" \"$destination\"\n",
    )
    _make_executable(
        fake_bin / "ssh",
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'ssh %s\\n' \"$*\" >> \"$COMMAND_LOG\"\n"
        "if [[ -n \"${SSH_ARGUMENTS_FILE:-}\" ]]; then printf '%s\\0' \"$@\" > \"$SSH_ARGUMENTS_FILE\"; fi\n"
        "printf '%s\\n' \"${SSH_REALPATH_RESULT:-/root/rivermind-data/phase1b}\"\n",
    )
    _make_executable(
        fake_bin / "uv",
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'local-uv %s\\n' \"$*\" >> \"$COMMAND_LOG\"\n"
        "if [[ \"${1:-}\" == run ]]; then shift; fi\n"
        "if [[ \"${1:-}\" == --directory ]]; then shift 2; fi\n"
        "if [[ \"${1:-}\" == python ]]; then shift; fi\nexec \"$REAL_PYTHON\" \"$@\"\n",
    )
    environment = {
        **os.environ,
        "COMMAND_LOG": str(log),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "REAL_PYTHON": sys.executable,
        "FAKE_REMOTE_ROOT": str(tmp_path / "root" / "rivermind-data" / "phase1b"),
    }
    return environment, log


def _run(script: str, *arguments: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    environment, _ = _environment(tmp_path)
    return subprocess.run(
        ["bash", str(SCRIPTS / script), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _write_manifest(root: Path, paths: dict[str, str]) -> None:
    entries = []
    for relative, content in sorted(paths.items()):
        file_path = root / "artifacts" / relative
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        entries.append(
            {
                "path": relative,
                "size": len(content.encode("utf-8")),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )
    manifest = root / "artifacts" / "phase1b" / "artifact-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"schema_version": 1, "files": entries}), encoding="utf-8")


def _nul_arguments(path: Path) -> list[str]:
    return [item.decode("utf-8") for item in path.read_bytes().split(b"\0") if item]


def test_scripts_use_strict_mode_and_contain_no_prohibited_remote_controls() -> None:
    prohibited = (
        "StrictHostKeyChecking=no",
        "shutdown",
        "poweroff",
        "rm -rf",
        "git reset",
        "github token",
        "password",
        "username",
    )
    for name in ("bootstrap_remote.sh", "run_remote.sh", "pull_artifacts.sh"):
        contents = (SCRIPTS / name).read_text(encoding="utf-8")
        assert "set -euo pipefail" in contents
        assert all(value.lower() not in contents.lower() for value in prohibited)


@pytest.mark.parametrize(
    ("script", "arguments"),
    [
        ("bootstrap_remote.sh", ("", "/repo")),
        ("bootstrap_remote.sh", ("/root/rivermind-data/run", "/")),
        ("run_remote.sh", ("/root", "preflight")),
        ("run_remote.sh", ("/root/rivermind-data/run*", "preflight")),
        ("pull_artifacts.sh", ("alias", "/root/rivermind-data/run\nnext", "/tmp/out")),
        ("pull_artifacts.sh", ("raw/host", "/root/rivermind-data/run", "/tmp/out")),
    ],
)
def test_scripts_reject_invalid_or_unconfined_arguments(
    script: str, arguments: tuple[str, ...], tmp_path: Path
) -> None:
    completed = _run(script, *arguments, tmp_path=tmp_path)

    assert completed.returncode != 0
    assert "invalid" in completed.stderr.lower() or "under" in completed.stderr.lower()


@pytest.mark.parametrize("invalid_path", ("", "/", "/root", "~unsafe", "path*", "path\nnext"))
@pytest.mark.parametrize("slot", ("bootstrap_root", "repo_dir", "pull_remote_root", "pull_local_output"))
def test_scripts_reject_every_required_invalid_path_class(
    invalid_path: str, slot: str, tmp_path: Path
) -> None:
    remote_root = tmp_path / "root" / "rivermind-data" / "phase1b"
    remote_root.mkdir(parents=True)
    repo_dir = tmp_path / "repository"
    repo_dir.mkdir()
    (repo_dir / "uv.lock").write_text("locked", encoding="utf-8")
    cases = {
        "bootstrap_root": ("bootstrap_remote.sh", [str(remote_root), str(repo_dir)], 0),
        "repo_dir": ("bootstrap_remote.sh", [str(remote_root), str(repo_dir)], 1),
        "pull_local_output": (
            "pull_artifacts.sh",
            ["phase1b-gpu", "/root/rivermind-data/phase1b", str(tmp_path / "out")],
            2,
        ),
        "pull_remote_root": (
            "pull_artifacts.sh",
            ["phase1b-gpu", "/root/rivermind-data/phase1b", str(tmp_path / "out")],
            1,
        ),
    }
    script, arguments, index = cases[slot]
    arguments[index] = invalid_path

    completed = _run(script, *arguments, tmp_path=tmp_path)

    assert completed.returncode != 0
    assert "invalid" in completed.stderr.lower() or "under" in completed.stderr.lower()


def test_stage_runner_rejects_every_required_invalid_remote_path_class(tmp_path: Path) -> None:
    for index, invalid_path in enumerate(("", "/", "/root", "~unsafe", "path*", "path\nnext")):
        completed = _run("run_remote.sh", invalid_path, "preflight", tmp_path=tmp_path / str(index))
        assert completed.returncode != 0


def test_stage_runner_rejects_an_unknown_stage(tmp_path: Path) -> None:
    remote_root = tmp_path / "root" / "rivermind-data" / "phase1b"
    remote_root.mkdir(parents=True)
    completed = _run("run_remote.sh", str(remote_root), "unknown", tmp_path=tmp_path)

    assert completed.returncode != 0
    assert "stage must be" in completed.stderr


@pytest.mark.parametrize(
    ("script", "arguments"),
    [
        ("bootstrap_remote.sh", ("/root/rivermind-data/run",)),
        ("run_remote.sh", ("/root/rivermind-data/run", "preflight", "extra")),
        ("pull_artifacts.sh", ("phase1b-gpu", "/root/rivermind-data/run")),
    ],
)
def test_scripts_require_their_exact_argument_count(
    script: str, arguments: tuple[str, ...], tmp_path: Path
) -> None:
    completed = _run(script, *arguments, tmp_path=tmp_path)

    assert completed.returncode != 0
    assert "argument count" in completed.stderr


def test_every_shell_entry_point_parses_with_bash_n() -> None:
    for name in ("bootstrap_remote.sh", "run_remote.sh", "pull_artifacts.sh"):
        completed = subprocess.run(
            ["bash", "-n", str(SCRIPTS / name)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


def test_bootstrap_creates_only_a_prefix_and_runs_frozen_gpu_sync(tmp_path: Path) -> None:
    remote_root = tmp_path / "root" / "rivermind-data" / "phase1b"
    repo_dir = tmp_path / "repository"
    remote_root.mkdir(parents=True)
    repo_dir.mkdir()
    (repo_dir / "uv.lock").write_text("locked", encoding="utf-8")
    environment, log = _environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(SCRIPTS / "bootstrap_remote.sh"), str(remote_root), str(repo_dir)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    commands = log.read_text(encoding="utf-8")
    assert f"conda create --yes --prefix {remote_root}/conda-env python=3.12 pip" in commands
    assert "python -m pip install uv==0.12.5" in commands
    assert f"uv sync --directory {repo_dir} --frozen --no-dev --extra gpu env={remote_root}/conda-env" in commands
    assert (remote_root / "conda-env" / ".trustsr-uv-lock.sha256").read_text().strip() == hashlib.sha256(
        b"locked"
    ).hexdigest()


def test_bootstrap_refuses_low_disk_space_before_any_write(tmp_path: Path) -> None:
    remote_root = tmp_path / "root" / "rivermind-data" / "phase1b"
    repo_dir = tmp_path / "repository"
    remote_root.mkdir(parents=True)
    repo_dir.mkdir()
    (repo_dir / "uv.lock").write_text("locked", encoding="utf-8")
    environment, log = _environment(tmp_path, available_kib=15 * 1024 * 1024 - 1)

    completed = subprocess.run(
        ["bash", str(SCRIPTS / "bootstrap_remote.sh"), str(remote_root), str(repo_dir)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert "15 gib" in completed.stderr.lower()
    assert not log.exists()
    assert not (remote_root / "conda-env").exists()


@pytest.mark.parametrize("mismatch", ("python", "uv", "lock_digest"))
def test_bootstrap_refuses_to_mutate_each_incompatible_existing_prefix(
    mismatch: str, tmp_path: Path
) -> None:
    remote_root = tmp_path / "root" / "rivermind-data" / "phase1b"
    repo_dir = tmp_path / "repository"
    (remote_root / "conda-env" / "bin").mkdir(parents=True)
    repo_dir.mkdir()
    (repo_dir / "uv.lock").write_text("locked", encoding="utf-8")
    python_version = "Python 3.11.9" if mismatch == "python" else "Python 3.12.4"
    uv_version = "uv 0.12.4" if mismatch == "uv" else "uv 0.12.5"
    _make_executable(
        remote_root / "conda-env" / "bin" / "python",
        f"#!/usr/bin/env bash\necho '{python_version}'\n",
    )
    _make_executable(
        remote_root / "conda-env" / "bin" / "uv",
        f"#!/usr/bin/env bash\necho '{uv_version}'\n",
    )
    digest = hashlib.sha256(b"locked").hexdigest()
    if mismatch == "lock_digest":
        digest = "0" * 64
    (remote_root / "conda-env" / ".trustsr-uv-lock.sha256").write_text(f"{digest}\n")
    environment, log = _environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(SCRIPTS / "bootstrap_remote.sh"), str(remote_root), str(repo_dir)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert "remove" in completed.stderr.lower() or "recreate" in completed.stderr.lower()
    assert not log.exists()


def test_stage_runner_exports_fixed_paths_and_invokes_exactly_one_matching_command(tmp_path: Path) -> None:
    remote_root = tmp_path / "root" / "rivermind-data" / "phase1b"
    (remote_root / "repo").mkdir(parents=True)
    executable = remote_root / "conda-env" / "bin" / "trustsr-ldsr-gpu"
    executable.parent.mkdir(parents=True)
    _make_executable(
        executable,
        "#!/usr/bin/env bash\nprintf 'stage %s data=%s sen2=%s ldsr=%s artifacts=%s cwd=%s\\n' \"$*\" \\\n"
        "  \"$TRUSTSR_DATA_CACHE_DIR\" \"$TRUSTSR_SEN2SR_MODEL_DIR\" \"$TRUSTSR_LDSR_MODEL_DIR\" \\\n"
        "  \"$TRUSTSR_ARTIFACT_ROOT\" \"$PWD\" >> \"$COMMAND_LOG\"\n",
    )
    environment, log = _environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(SCRIPTS / "run_remote.sh"), str(remote_root), "benchmark"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(f"stage benchmark --project-root {remote_root}/repo")
    assert f"data={remote_root}/data/opensr" in lines[0]
    assert f"sen2={remote_root}/models/sen2srlite" in lines[0]
    assert f"ldsr={remote_root}/models/ldsr-s2" in lines[0]
    assert f"artifacts={remote_root}/artifacts" in lines[0]
    assert f"cwd={remote_root}/repo" in lines[0]


@pytest.mark.parametrize("stage", ("preflight", "single", "benchmark", "manifest"))
def test_stage_runner_maps_each_stage_to_the_exact_cli_argument_array(
    stage: str, tmp_path: Path
) -> None:
    remote_root = tmp_path / "root" / "rivermind-data" / "phase1b"
    (remote_root / "repo").mkdir(parents=True)
    executable = remote_root / "conda-env" / "bin" / "trustsr-ldsr-gpu"
    executable.parent.mkdir(parents=True)
    _make_executable(
        executable,
        "#!/usr/bin/env bash\nprintf '%s\\0' \"$@\" > \"$STAGE_ARGUMENTS_FILE\"\n",
    )
    stage_arguments = tmp_path / "stage-arguments"
    environment, _ = _environment(tmp_path)
    environment["STAGE_ARGUMENTS_FILE"] = str(stage_arguments)

    completed = subprocess.run(
        ["bash", str(SCRIPTS / "run_remote.sh"), str(remote_root), stage],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert _nul_arguments(stage_arguments) == [
        stage,
        "--project-root",
        f"{remote_root}/repo",
        "--dataset-cache-dir",
        f"{remote_root}/data/opensr",
        "--ldsr-model-dir",
        f"{remote_root}/models/ldsr-s2",
        "--sen2srlite-model-dir",
        f"{remote_root}/models/sen2srlite",
        "--artifacts-dir",
        f"{remote_root}/artifacts",
        "--prediction-cache-dir",
        f"{remote_root}/artifacts/cache/predictions",
    ]


@pytest.mark.parametrize("escaping_directory", ("data", "models", "artifacts"))
def test_stage_runner_rejects_descendant_symlink_escape_before_cli_or_outside_write(
    tmp_path: Path, escaping_directory: str
) -> None:
    remote_root = tmp_path / "root" / "rivermind-data" / "phase1b"
    (remote_root / "repo").mkdir(parents=True)
    executable = remote_root / "conda-env" / "bin" / "trustsr-ldsr-gpu"
    executable.parent.mkdir(parents=True)
    _make_executable(
        executable,
        "#!/usr/bin/env bash\nprintf 'called' > \"$CLI_CALLED_FILE\"\n",
    )
    outside = tmp_path / f"outside-{escaping_directory}"
    outside.mkdir()
    (remote_root / escaping_directory).symlink_to(outside, target_is_directory=True)
    cli_called = tmp_path / "cli-called"
    environment, _ = _environment(tmp_path)
    environment["CLI_CALLED_FILE"] = str(cli_called)

    completed = subprocess.run(
        ["bash", str(SCRIPTS / "run_remote.sh"), str(remote_root), "preflight"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert "symlink" in completed.stderr.lower() or "escape" in completed.stderr.lower()
    assert not cli_called.exists()
    assert list(outside.iterdir()) == []


def test_puller_fetches_manifest_then_only_allowlisted_files_and_verifies_digests(tmp_path: Path) -> None:
    remote_root = tmp_path / "root" / "rivermind-data" / "phase1b"
    _write_manifest(
        remote_root,
        {
            "phase1b/environment.json": "environment",
            "phase1b/cache/prediction.json": "prediction",
        },
    )
    local_output = tmp_path / "local-artifacts"
    environment, log = _environment(tmp_path)
    environment["REMOTE_FIXTURE_ROOT"] = str(remote_root)

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "pull_artifacts.sh"),
            "phase1b-gpu",
            "/root/rivermind-data/phase1b",
            str(local_output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    calls = [line for line in log.read_text(encoding="utf-8").splitlines() if line.startswith("rsync ")]
    assert len(calls) == 3
    assert "--protect-args" in calls[0]
    remote_artifacts = "phase1b-gpu:/root/rivermind-data/phase1b/artifacts"
    assert calls[0].endswith(f"{remote_artifacts}/phase1b/artifact-manifest.json {local_output}/phase1b")
    assert calls[1].endswith(f"{remote_artifacts}/phase1b/cache/prediction.json {local_output}/phase1b/cache")
    assert calls[2].endswith(f"{remote_artifacts}/phase1b/environment.json {local_output}/phase1b")
    assert (local_output / "phase1b" / "cache" / "prediction.json").read_text() == "prediction"
    assert any(line.startswith("local-uv run --directory") for line in log.read_text().splitlines())


def test_puller_uses_exact_ssh_and_manifest_allowlisted_rsync_argument_arrays(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote-fixture"
    _write_manifest(
        remote_root,
        {
            "phase1b/cache/prediction.json": "prediction",
            "phase1b/environment.json": "environment",
        },
    )
    local_output = tmp_path / "local-artifacts"
    environment, _ = _environment(tmp_path)
    environment["REMOTE_FIXTURE_ROOT"] = str(remote_root)
    ssh_arguments = tmp_path / "ssh-arguments"
    rsync_arguments = tmp_path / "rsync-arguments"
    environment["SSH_ARGUMENTS_FILE"] = str(ssh_arguments)
    environment["RSYNC_ARGUMENT_DIR"] = str(rsync_arguments)

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "pull_artifacts.sh"),
            "phase1b-gpu",
            "/root/rivermind-data/phase1b",
            str(local_output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert _nul_arguments(ssh_arguments) == [
        "--",
        "phase1b-gpu",
        "realpath -e -- /root/rivermind-data/phase1b",
    ]
    arguments = [_nul_arguments(rsync_arguments / str(index)) for index in range(3)]
    source_root = "phase1b-gpu:/root/rivermind-data/phase1b/artifacts"
    assert arguments == [
        [
            "--archive",
            "--protect-args",
            "--",
            f"{source_root}/phase1b/artifact-manifest.json",
            f"{local_output}/phase1b",
        ],
        [
            "--archive",
            "--protect-args",
            "--",
            f"{source_root}/phase1b/cache/prediction.json",
            f"{local_output}/phase1b/cache",
        ],
        [
            "--archive",
            "--protect-args",
            "--",
            f"{source_root}/phase1b/environment.json",
            f"{local_output}/phase1b",
        ],
    ]


def test_puller_rejects_escaping_manifest_path_before_transferring_it(tmp_path: Path) -> None:
    remote_root = tmp_path / "root" / "rivermind-data" / "phase1b"
    manifest = remote_root / "artifacts" / "phase1b" / "artifact-manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {
                        "path": "../outside.json",
                        "size": 0,
                        "sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    environment, log = _environment(tmp_path)
    environment["REMOTE_FIXTURE_ROOT"] = str(remote_root)

    completed = subprocess.run(
        ["bash", str(SCRIPTS / "pull_artifacts.sh"), "phase1b-gpu", "/root/rivermind-data/phase1b", str(tmp_path / "out")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    rsync_calls = [line for line in log.read_text().splitlines() if line.startswith("rsync ")]
    assert len(rsync_calls) == 1
    assert "artifact-manifest.json" in rsync_calls[0]


def test_puller_rejects_newline_manifest_path_before_transferring_it(tmp_path: Path) -> None:
    remote_root = tmp_path / "root" / "rivermind-data" / "phase1b"
    manifest = remote_root / "artifacts" / "phase1b" / "artifact-manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {
                        "path": "phase1b/valid\ninvalid.json",
                        "size": 0,
                        "sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    environment, log = _environment(tmp_path)
    environment["REMOTE_FIXTURE_ROOT"] = str(remote_root)

    completed = subprocess.run(
        ["bash", str(SCRIPTS / "pull_artifacts.sh"), "phase1b-gpu", "/root/rivermind-data/phase1b", str(tmp_path / "out")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert len([line for line in log.read_text().splitlines() if line.startswith("rsync ")]) == 1


def test_puller_accepts_a_confined_remote_path_that_is_not_local(tmp_path: Path) -> None:
    remote_fixture = tmp_path / "remote-fixture"
    manifest = remote_fixture / "artifacts" / "phase1b" / "artifact-manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"schema_version": 1, "files": []}),
        encoding="utf-8",
    )
    environment, log = _environment(tmp_path)
    fake_bin = Path(environment["PATH"].split(os.pathsep, maxsplit=1)[0])
    _make_executable(fake_bin / "realpath", "#!/usr/bin/env bash\nexit 1\n")
    _make_executable(
        fake_bin / "rsync",
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'rsync %s\\n' \"$*\" >> \"$COMMAND_LOG\"\n"
        "destination=\"${!#}\"\nmkdir -p \"$destination\"\n"
        "cp \"$REMOTE_FIXTURE_ROOT/artifacts/phase1b/artifact-manifest.json\" \"$destination\"\n",
    )
    environment["REMOTE_FIXTURE_ROOT"] = str(remote_fixture)

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "pull_artifacts.sh"),
            "phase1b-gpu",
            "/root/rivermind-data/phase1b",
            str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert len([line for line in log.read_text().splitlines() if line.startswith("rsync ")]) == 1


def test_puller_rejects_a_remote_root_that_resolves_outside_the_data_disk(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote-fixture"
    _write_manifest(remote_root, {})
    environment, log = _environment(tmp_path)
    environment["REMOTE_FIXTURE_ROOT"] = str(remote_root)
    environment["SSH_REALPATH_RESULT"] = "/outside-data-disk"

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "pull_artifacts.sh"),
            "phase1b-gpu",
            "/root/rivermind-data/phase1b-link",
            str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert not [line for line in log.read_text().splitlines() if line.startswith("rsync ")]


def test_puller_rejects_nul_manifest_path_before_any_artifact_transfer(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote-fixture"
    manifest = remote_root / "artifacts" / "phase1b" / "artifact-manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {
                        "path": "phase1b/nul\0path.json",
                        "size": 0,
                        "sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    environment, log = _environment(tmp_path)
    environment["REMOTE_FIXTURE_ROOT"] = str(remote_root)

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "pull_artifacts.sh"),
            "phase1b-gpu",
            "/root/rivermind-data/phase1b",
            str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert len([line for line in log.read_text().splitlines() if line.startswith("rsync ")]) == 1
