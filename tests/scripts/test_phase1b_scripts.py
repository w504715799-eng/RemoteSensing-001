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
        "  printf '%s\\n' \"${FAKE_CANONICAL_ROOT:-/root/rivermind-fs/test-root}\"\n"
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
        "expected=(create --yes --override-channels --channel conda-forge --prefix \"$prefix\" python=3.12 pip)\n"
        "if (( $# != ${#expected[@]} )); then\n"
        "  printf 'unexpected conda create argv: %s\\n' \"$*\" >&2; exit 99\n"
        "fi\n"
        "for ((index = 1; index <= $#; index++)); do\n"
        "  [[ \"${!index}\" == \"${expected[$((index - 1))]}\" ]] || {\n"
        "    printf 'unexpected conda create argv: %s\\n' \"$*\" >&2; exit 99\n"
        "  }\n"
        "done\n"
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
        "if [[ -n \"${REMOTE_FIXTURE_ROOT:-}\" ]]; then\n"
        "  source=\"${source#/root/rivermind-fs/phase1b}\"\n"
        "  source=\"${REMOTE_FIXTURE_ROOT}${source}\"\n"
        "fi\n"
        "mkdir -p \"$destination\"\ncp \"$source\" \"$destination\"\n",
    )
    _make_executable(
        fake_bin / "ssh",
        "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'ssh %s\\n' \"$*\" >> \"$COMMAND_LOG\"\n"
        "if [[ -n \"${SSH_ARGUMENTS_FILE:-}\" ]]; then printf '%s\\0' \"$@\" > \"$SSH_ARGUMENTS_FILE\"; fi\n"
        "printf '%s\\n' \"${SSH_REALPATH_RESULT:-/root/rivermind-fs/phase1b}\"\n",
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
        "FAKE_REMOTE_ROOT": str(tmp_path / "root" / "rivermind-fs" / "phase1b"),
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
        ("bootstrap_remote.sh", ("/root/rivermind-data/run", "/repo")),
        ("run_remote.sh", ("/root", "preflight")),
        ("run_remote.sh", ("/root/rivermind-fs/run*", "preflight")),
        ("pull_artifacts.sh", ("alias", "/root/rivermind-fs/run\nnext", "/tmp/out")),
        ("pull_artifacts.sh", ("raw/host", "/root/rivermind-fs/run", "/tmp/out")),
    ],
)
def test_scripts_reject_invalid_or_unconfined_arguments(
    script: str, arguments: tuple[str, ...], tmp_path: Path
) -> None:
    completed = _run(script, *arguments, tmp_path=tmp_path)

    assert completed.returncode != 0
    assert "invalid" in completed.stderr.lower() or "under" in completed.stderr.lower()


@pytest.mark.parametrize("script", ("bootstrap_remote.sh", "run_remote.sh", "pull_artifacts.sh"))
def test_scripts_reject_the_obsolete_data_disk_root_before_side_effects(
    script: str, tmp_path: Path
) -> None:
    old_root = tmp_path / "root" / "rivermind-data" / "phase1b"
    old_root.mkdir(parents=True)
    environment, log = _environment(tmp_path)
    environment["FAKE_REMOTE_ROOT"] = str(old_root)
    environment["FAKE_CANONICAL_ROOT"] = "/root/rivermind-data/test-root"

    if script == "bootstrap_remote.sh":
        repository = tmp_path / "repository"
        repository.mkdir()
        (repository / "uv.lock").write_text("locked", encoding="utf-8")
        arguments = [str(old_root), str(repository)]
    elif script == "run_remote.sh":
        (old_root / "repo").mkdir()
        executable = old_root / "conda-env" / "bin" / "trustsr-ldsr-gpu"
        executable.parent.mkdir(parents=True)
        _make_executable(executable, "#!/usr/bin/env bash\nprintf 'called' >> \"$COMMAND_LOG\"\n")
        arguments = [str(old_root), "preflight"]
    else:
        _write_manifest(old_root, {})
        environment["REMOTE_FIXTURE_ROOT"] = str(old_root)
        environment["SSH_REALPATH_RESULT"] = "/root/rivermind-data/phase1b"
        arguments = ["phase1b-gpu", "/root/rivermind-data/phase1b", str(tmp_path / "out")]

    completed = subprocess.run(
        ["bash", str(SCRIPTS / script), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert not log.exists()


@pytest.mark.parametrize("invalid_path", ("", "/", "/root", "~unsafe", "path*", "path\nnext"))
@pytest.mark.parametrize("slot", ("bootstrap_root", "repo_dir", "pull_remote_root", "pull_local_output"))
def test_scripts_reject_every_required_invalid_path_class(
    invalid_path: str, slot: str, tmp_path: Path
) -> None:
    remote_root = tmp_path / "root" / "rivermind-fs" / "phase1b"
    remote_root.mkdir(parents=True)
    repo_dir = tmp_path / "repository"
    repo_dir.mkdir()
    (repo_dir / "uv.lock").write_text("locked", encoding="utf-8")
    cases = {
        "bootstrap_root": ("bootstrap_remote.sh", [str(remote_root), str(repo_dir)], 0),
        "repo_dir": ("bootstrap_remote.sh", [str(remote_root), str(repo_dir)], 1),
        "pull_local_output": (
            "pull_artifacts.sh",
            ["phase1b-gpu", "/root/rivermind-fs/phase1b", str(tmp_path / "out")],
            2,
        ),
        "pull_remote_root": (
            "pull_artifacts.sh",
            ["phase1b-gpu", "/root/rivermind-fs/phase1b", str(tmp_path / "out")],
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
    remote_root = tmp_path / "root" / "rivermind-fs" / "phase1b"
    remote_root.mkdir(parents=True)
    completed = _run("run_remote.sh", str(remote_root), "unknown", tmp_path=tmp_path)

    assert completed.returncode != 0
    assert "stage must be" in completed.stderr


@pytest.mark.parametrize(
    ("script", "arguments"),
    [
        ("bootstrap_remote.sh", ("/root/rivermind-fs/run",)),
        ("run_remote.sh", ("/root/rivermind-fs/run", "preflight", "extra")),
        ("pull_artifacts.sh", ("phase1b-gpu", "/root/rivermind-fs/run")),
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


def test_bootstrap_contract_reuses_the_fixed_base_and_preserves_its_cuda_stack() -> None:
    """Removing a preflight, dry-run, or preservation check would break this contract."""
    contents = (SCRIPTS / "bootstrap_remote.sh").read_text(encoding="utf-8")

    assert "/opt/conda/bin/python" in contents
    assert "--version" in contents
    assert "conda create" not in contents
    assert "conda-env" not in contents
    assert "uv sync" not in contents
    assert "torch.cuda.is_available()" in contents
    assert "torch.version.cuda" in contents
    assert "--dry-run" in contents
    assert "--report" in contents
    assert "--upgrade-strategy only-if-needed" in contents
    assert "opensr-model" in contents
    assert '"uv==0.12.5"' in contents
    assert "pip check" in contents
    assert ".trustsr-bootstrap-provenance.json" in contents


def test_bootstrap_fails_closed_before_conda_prefix_creation_when_fixed_base_is_absent(
    tmp_path: Path,
) -> None:
    """A missing cloud-image interpreter must not fall back to creating a prefix."""
    remote_root = tmp_path / "root" / "rivermind-fs" / "phase1b"
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

    assert completed.returncode != 0
    assert "/opt/conda/bin/python" in completed.stderr
    assert not log.exists()
    assert not (remote_root / "conda-env").exists()


def test_bootstrap_refuses_low_disk_space_before_any_write(tmp_path: Path) -> None:
    remote_root = tmp_path / "root" / "rivermind-fs" / "phase1b"
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


def test_bootstrap_ignores_an_existing_partial_conda_prefix(tmp_path: Path) -> None:
    remote_root = tmp_path / "root" / "rivermind-fs" / "phase1b"
    repo_dir = tmp_path / "repository"
    (remote_root / "conda-env" / "bin").mkdir(parents=True)
    repo_dir.mkdir()
    (repo_dir / "uv.lock").write_text("locked", encoding="utf-8")
    sentinel = remote_root / "conda-env" / "preserve-me"
    sentinel.write_text("partial environment", encoding="utf-8")
    environment, log = _environment(tmp_path)

    completed = subprocess.run(
        ["bash", str(SCRIPTS / "bootstrap_remote.sh"), str(remote_root), str(repo_dir)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert "/opt/conda/bin/python" in completed.stderr
    assert not log.exists()
    assert sentinel.read_text(encoding="utf-8") == "partial environment"


def test_stage_runner_contract_uses_the_fixed_base_module_cli_without_a_prefix() -> None:
    """A prefix command or a console-script launcher would use the wrong runtime."""
    contents = (SCRIPTS / "run_remote.sh").read_text(encoding="utf-8")

    assert "/opt/conda/bin/python" in contents
    assert "-m trustsr.cli.ldsr_gpu" in contents
    assert "conda-env" not in contents
    assert "trustsr-ldsr-gpu" not in contents


@pytest.mark.parametrize("escaping_directory", ("data", "models", "artifacts"))
def test_stage_runner_rejects_descendant_symlink_escape_before_cli_or_outside_write(
    tmp_path: Path, escaping_directory: str
) -> None:
    remote_root = tmp_path / "root" / "rivermind-fs" / "phase1b"
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


@pytest.mark.parametrize(
    "escaping_relative", (Path("artifacts/phase1b"), Path("artifacts/phase1b/cache"))
)
def test_stage_runner_rejects_fixed_output_tree_symlink_before_cli_or_any_output_write(
    tmp_path: Path, escaping_relative: Path
) -> None:
    remote_root = tmp_path / "root" / "rivermind-fs" / "phase1b"
    (remote_root / "repo").mkdir(parents=True)
    executable = remote_root / "conda-env" / "bin" / "trustsr-ldsr-gpu"
    executable.parent.mkdir(parents=True)
    _make_executable(
        executable,
        "#!/usr/bin/env bash\n"
        "printf 'called' > \"$CLI_CALLED_FILE\"\n"
        "mkdir -p \"$TRUSTSR_ARTIFACT_ROOT/phase1b/cache\"\n"
        "printf 'environment' > \"$TRUSTSR_ARTIFACT_ROOT/phase1b/environment.json\"\n"
        "printf 'cache' > \"$TRUSTSR_ARTIFACT_ROOT/phase1b/cache/output.json\"\n",
    )
    outside = tmp_path / f"outside-{'-'.join(escaping_relative.parts)}"
    outside.mkdir()
    escaping_path = remote_root / escaping_relative
    escaping_path.parent.mkdir(parents=True, exist_ok=True)
    escaping_path.symlink_to(outside, target_is_directory=True)
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
    assert not (remote_root / "artifacts/phase1b/environment.json").is_file()


def test_puller_fetches_manifest_then_only_allowlisted_files_and_verifies_digests(tmp_path: Path) -> None:
    remote_root = tmp_path / "root" / "rivermind-fs" / "phase1b"
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
            "/root/rivermind-fs/phase1b",
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
    remote_artifacts = "phase1b-gpu:/root/rivermind-fs/phase1b/artifacts"
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
            "/root/rivermind-fs/phase1b",
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
        "realpath -e -- /root/rivermind-fs/phase1b",
    ]
    arguments = [_nul_arguments(rsync_arguments / str(index)) for index in range(3)]
    source_root = "phase1b-gpu:/root/rivermind-fs/phase1b/artifacts"
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
    remote_root = tmp_path / "root" / "rivermind-fs" / "phase1b"
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
        ["bash", str(SCRIPTS / "pull_artifacts.sh"), "phase1b-gpu", "/root/rivermind-fs/phase1b", str(tmp_path / "out")],
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
    remote_root = tmp_path / "root" / "rivermind-fs" / "phase1b"
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
        ["bash", str(SCRIPTS / "pull_artifacts.sh"), "phase1b-gpu", "/root/rivermind-fs/phase1b", str(tmp_path / "out")],
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
            "/root/rivermind-fs/phase1b",
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
            "/root/rivermind-fs/phase1b-link",
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
            "/root/rivermind-fs/phase1b",
            str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert len([line for line in log.read_text().splitlines() if line.startswith("rsync ")]) == 1
