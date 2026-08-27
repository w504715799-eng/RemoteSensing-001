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


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    log = tmp_path / "commands.log"
    _make_executable(
        fake_bin / "df",
        "#!/usr/bin/env bash\nprintf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'\n"
        "printf '/dev/fake 30000000 1 30000000 1%% /fake\\n'\n",
    )
    _make_executable(
        fake_bin / "realpath",
        "#!/usr/bin/env bash\nvalue=\"${!#}\"\n"
        "case \"$value\" in /root|/root/*) printf '%s\\n' \"$value\" ;; *) "
        "printf '/root/rivermind-data/test-root\\n' ;; esac\n",
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
        "args=(\"$@\"); source=\"${args[$(( $# - 2 ))]}\"; destination=\"${args[$(( $# - 1 ))]}\"\n"
        "source=\"${source#*:}\"\n"
        "if [[ -n \"${REMOTE_FIXTURE_ROOT:-}\" ]]; then source=\"${REMOTE_FIXTURE_ROOT}${source#/root/rivermind-data/phase1b}\"; fi\n"
        "mkdir -p \"$destination\"\ncp \"$source\" \"$destination\"\n",
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


def test_bootstrap_refuses_to_mutate_an_incompatible_existing_prefix(tmp_path: Path) -> None:
    remote_root = tmp_path / "root" / "rivermind-data" / "phase1b"
    repo_dir = tmp_path / "repository"
    (remote_root / "conda-env" / "bin").mkdir(parents=True)
    repo_dir.mkdir()
    (repo_dir / "uv.lock").write_text("locked", encoding="utf-8")
    _make_executable(remote_root / "conda-env" / "bin" / "python", "#!/usr/bin/env bash\necho 'Python 3.11.9'\n")
    _make_executable(remote_root / "conda-env" / "bin" / "uv", "#!/usr/bin/env bash\necho 'uv 0.12.5'\n")
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
    executable = remote_root / "conda-env" / "bin" / "trustsr-ldsr-gpu"
    executable.parent.mkdir(parents=True)
    _make_executable(
        executable,
        "#!/usr/bin/env bash\nprintf 'stage %s data=%s sen2=%s ldsr=%s artifacts=%s\\n' \"$*\" \\\n"
        "  \"$TRUSTSR_DATA_CACHE_DIR\" \"$TRUSTSR_SEN2SR_MODEL_DIR\" \"$TRUSTSR_LDSR_MODEL_DIR\" \\\n"
        "  \"$TRUSTSR_ARTIFACT_ROOT\" >> \"$COMMAND_LOG\"\n",
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
    assert lines[0].startswith("stage benchmark --dataset-cache-dir")
    assert f"data={remote_root}/data/opensr" in lines[0]
    assert f"sen2={remote_root}/models/sen2srlite" in lines[0]
    assert f"ldsr={remote_root}/models/ldsr-s2" in lines[0]
    assert f"artifacts={remote_root}/artifacts" in lines[0]


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
