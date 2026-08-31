"""Executable contracts for the Phase 2B1B cloud runner."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
RUNNER = REPOSITORY / "scripts" / "phase2b1b" / "run_cloud.sh"


def _make_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _repository(tmp_path: Path, *, name: str = "repository") -> Path:
    repository = tmp_path / name
    (repository / "src" / "trustsr").mkdir(parents=True)
    (repository / "pyproject.toml").write_text(
        "[project]\nname = 'fixture'\n", encoding="utf-8"
    )
    return repository


def _environment(
    tmp_path: Path,
    *,
    mount_root: Path,
    available_kib: int = 10_000_000,
    available_inodes: int = 10_000,
) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir(exist_ok=True)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    conda_called = tmp_path / "conda-called"
    _make_executable(
        fake_bin / "mountpoint",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "[[ \"$*\" == \"-q -- $FAKE_MOUNT_ROOT\" ]]\n",
    )
    _make_executable(
        fake_bin / "df",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "case \"$1\" in\n"
        "  -Pk)\n"
        "    printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'\n"
        "    printf '/dev/fake 20000000 1 %s 1%% /persistent\\n' \"$FAKE_AVAILABLE_KIB\"\n"
        "    ;;\n"
        "  -Pi)\n"
        "    printf 'Filesystem Inodes IUsed IFree IUse%% Mounted on\\n'\n"
        "    printf '/dev/fake 20000 1 %s 1%% /persistent\\n' \"$FAKE_AVAILABLE_INODES\"\n"
        "    ;;\n"
        "  *) exit 98 ;;\n"
        "esac\n",
    )
    _make_executable(
        fake_bin / "conda",
        "#!/usr/bin/env bash\n"
        "touch \"$CONDA_CALLED\"\n"
        "exit 99\n",
    )
    return (
        {
            **os.environ,
            "CONDA_CALLED": str(conda_called),
            "FAKE_AVAILABLE_KIB": str(available_kib),
            "FAKE_AVAILABLE_INODES": str(available_inodes),
            "FAKE_MOUNT_ROOT": str(mount_root),
            "HOME": str(home),
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        },
        conda_called,
    )


def _fake_base_python(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    calls = tmp_path / "python-calls.jsonl"
    launcher = tmp_path / "base-python"
    _make_executable(
        launcher,
        f"#!{sys.executable}\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "arguments = sys.argv[1:]\n"
        "with Path(os.environ['FAKE_PYTHON_CALLS']).open('a', encoding='utf-8') as stream:\n"
        "    stream.write(json.dumps(arguments) + '\\n')\n"
        "if arguments[:2] == ['-m', 'trustsr.cli.phase2b1b']:\n"
        "    print(json.dumps({'stage': arguments[2]}, separators=(',', ':')))\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(f'unexpected fake Python argv: {arguments!r}')\n",
    )
    return launcher, {"FAKE_PYTHON_CALLS": str(calls)}, calls


def _invoke(
    tmp_path: Path,
    *,
    storage_root: str | Path,
    repository: Path,
    stage: str,
    stage_arguments: list[str],
    available_kib: int = 10_000_000,
    available_inodes: int = 10_000,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]], Path]:
    environment, conda_called = _environment(
        tmp_path,
        mount_root=Path(storage_root) if isinstance(storage_root, Path) else Path("/invalid"),
        available_kib=available_kib,
        available_inodes=available_inodes,
    )
    base_python, python_environment, calls_path = _fake_base_python(tmp_path)
    environment.update(python_environment)
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; shift; run_main "$@"',
            "phase2b1b-script-test",
            str(RUNNER),
            str(base_python),
            str(storage_root),
            str(repository),
            stage,
            *stage_arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    calls = (
        [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]
        if calls_path.exists()
        else []
    )
    return completed, calls, conda_called


@pytest.mark.parametrize("stage", ["select", "extract", "audit"])
def test_runner_uses_base_python_and_forwards_only_phase2b1b_stages(
    tmp_path: Path, stage: str
) -> None:
    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    repository = _repository(tmp_path)

    completed, calls, conda_called = _invoke(
        tmp_path,
        storage_root=storage_root,
        repository=repository,
        stage=stage,
        stage_arguments=["--confirm-cloud-storage"],
    )

    assert completed.returncode == 0, completed.stderr
    assert calls == [
        [
            "-m",
            "trustsr.cli.phase2b1b",
            stage,
            "--storage-root",
            str(storage_root),
            "--confirm-cloud-storage",
        ]
    ]
    assert not conda_called.exists()


@pytest.mark.parametrize("bad_root", ["/", "/root", "relative", "/tmp/*"])
def test_runner_rejects_unsafe_storage_before_python(
    tmp_path: Path, bad_root: str
) -> None:
    repository = _repository(tmp_path)

    completed, calls, _ = _invoke(
        tmp_path,
        storage_root=bad_root,
        repository=repository,
        stage="select",
        stage_arguments=["--confirm-cloud-storage"],
    )

    assert completed.returncode == 2
    assert calls == []


def test_runner_requires_more_than_five_gib_and_explicit_confirmation(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    repository = _repository(tmp_path)

    low_space, calls, _ = _invoke(
        tmp_path,
        storage_root=storage_root,
        repository=repository,
        stage="select",
        stage_arguments=["--confirm-cloud-storage"],
        available_kib=5 * 1024 * 1024,
    )
    unconfirmed, unconfirmed_calls, _ = _invoke(
        tmp_path,
        storage_root=storage_root,
        repository=repository,
        stage="select",
        stage_arguments=["--source"],
    )

    assert low_space.returncode == 2
    assert "more than 5 GiB" in low_space.stderr
    assert calls == []
    assert unconfirmed.returncode == 2
    assert "confirm-cloud-storage" in unconfirmed.stderr
    assert unconfirmed_calls == []


def test_runner_defers_inode_capacity_to_the_restart_aware_extract_stage(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    repository = _repository(tmp_path)

    completed, calls, _ = _invoke(
        tmp_path,
        storage_root=storage_root,
        repository=repository,
        stage="extract",
        stage_arguments=["--confirm-cloud-storage"],
        available_inodes=0,
    )

    assert completed.returncode == 0, completed.stderr
    assert calls == [
        [
            "-m",
            "trustsr.cli.phase2b1b",
            "extract",
            "--storage-root",
            str(storage_root),
            "--confirm-cloud-storage",
        ]
    ]


def test_runner_appends_each_canonical_result_to_the_explicit_stage_log(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    repository = _repository(tmp_path)

    for _ in range(2):
        completed, _, _ = _invoke(
            tmp_path,
            storage_root=storage_root,
            repository=repository,
            stage="select",
            stage_arguments=["--confirm-cloud-storage"],
        )
        assert completed.returncode == 0, completed.stderr

    log = storage_root / "trustsr" / "phase2b1b" / "logs" / "select.jsonl"
    assert log.read_text(encoding="utf-8").splitlines() == [
        '{"stage":"select"}',
        '{"stage":"select"}',
    ]


def test_runner_rejects_symlink_components_and_storage_override(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "persistent"
    real_root.mkdir()
    symlink_root = tmp_path / "linked-persistent"
    symlink_root.symlink_to(real_root, target_is_directory=True)
    repository = _repository(tmp_path)

    symlinked, symlink_calls, _ = _invoke(
        tmp_path,
        storage_root=symlink_root,
        repository=repository,
        stage="select",
        stage_arguments=["--confirm-cloud-storage"],
    )
    overridden, override_calls, _ = _invoke(
        tmp_path,
        storage_root=real_root,
        repository=repository,
        stage="select",
        stage_arguments=["--confirm-cloud-storage", "--storage-root", "/tmp/other"],
    )

    assert symlinked.returncode == 2
    assert symlink_calls == []
    assert overridden.returncode == 2
    assert override_calls == []


def test_runner_rejects_unknown_stage_and_repository_with_colon(tmp_path: Path) -> None:
    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    repository = _repository(tmp_path)
    colon_repository = _repository(tmp_path, name="repository:bad")

    unknown, unknown_calls, _ = _invoke(
        tmp_path,
        storage_root=storage_root,
        repository=repository,
        stage="download",
        stage_arguments=["--confirm-cloud-storage"],
    )
    colon, colon_calls, _ = _invoke(
        tmp_path,
        storage_root=storage_root,
        repository=colon_repository,
        stage="select",
        stage_arguments=["--confirm-cloud-storage"],
    )

    assert unknown.returncode == 2
    assert unknown_calls == []
    assert colon.returncode == 2
    assert colon_calls == []
