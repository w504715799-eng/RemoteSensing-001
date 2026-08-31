"""Executable contracts for the Phase 2B2-B base-environment cloud runner."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from trustsr.data.crosssensor_pairs import POST_MANIFEST_SHA256
from trustsr.evaluation.crosssensor_smoke import INPUT_AUDIT_SHA256

REPOSITORY = Path(__file__).resolve().parents[2]
RUNNER = REPOSITORY / "scripts" / "phase2b2b" / "run_cloud.sh"


def _make_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _repository(tmp_path: Path, name: str = "repository") -> Path:
    repository = tmp_path / name
    (repository / "src" / "trustsr").mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    return repository


def _environment(
    tmp_path: Path,
    *,
    mount_root: Path,
    available_kib: int = 10 * 1024 * 1024,
    available_inodes: int = 2048,
    mounted: bool = True,
) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir(exist_ok=True)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    prohibited = tmp_path / "prohibited-command"
    mount_exit = 0 if mounted else 1
    _make_executable(
        fake_bin / "mountpoint",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "[[ \"$*\" == \"-q -- $FAKE_MOUNT_ROOT\" ]]\n"
        f"exit {mount_exit}\n",
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
        "    printf '/dev/fake 5000 1 %s 1%% /persistent\\n' \"$FAKE_AVAILABLE_INODES\"\n"
        "    ;;\n"
        "  *) exit 98 ;;\n"
        "esac\n",
    )
    for name in ("conda", "pip", "wget", "curl"):
        _make_executable(
            fake_bin / name,
            "#!/usr/bin/env bash\n"
            "touch \"$PROHIBITED_COMMAND\"\n"
            "exit 99\n",
        )
    return (
        {
            **os.environ,
            "FAKE_AVAILABLE_INODES": str(available_inodes),
            "FAKE_AVAILABLE_KIB": str(available_kib),
            "FAKE_MOUNT_ROOT": str(mount_root),
            "HOME": str(home),
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "PROHIBITED_COMMAND": str(prohibited),
        },
        prohibited,
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
        "if arguments[:2] == ['-m', 'trustsr.cli.phase2b2b']:\n"
        "    print(json.dumps({'stage': arguments[2]}, separators=(',', ':')))\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(f'unexpected fake Python argv: {arguments!r}')\n",
    )
    return launcher, {"FAKE_PYTHON_CALLS": str(calls)}, calls


def _default_arguments(storage_root: Path) -> list[str]:
    return [
        "--selection-manifest",
        str(storage_root / "samples.jsonl"),
        "--selection-manifest-sha256",
        POST_MANIFEST_SHA256,
        "--input-audit",
        str(storage_root / "phase2b2a-input-audit.json"),
        "--input-audit-sha256",
        INPUT_AUDIT_SHA256,
        "--sen2srlite-model-dir",
        str(storage_root / "models" / "sen2srlite"),
        "--ldsr-model-dir",
        str(storage_root / "models" / "ldsr"),
        "--confirm-cloud-storage",
    ]


def _invoke(
    tmp_path: Path,
    *,
    stage: str = "smoke",
    storage_root: str | Path | None = None,
    repository: Path | None = None,
    stage_arguments: list[str] | None = None,
    available_kib: int = 10 * 1024 * 1024,
    available_inodes: int = 2048,
    mounted: bool = True,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]], Path]:
    root = storage_root if storage_root is not None else tmp_path / "persistent"
    if isinstance(root, Path):
        root.mkdir(exist_ok=True)
    repo = repository if repository is not None else _repository(tmp_path)
    environment, prohibited = _environment(
        tmp_path,
        mount_root=root if isinstance(root, Path) else Path("/invalid"),
        available_kib=available_kib,
        available_inodes=available_inodes,
        mounted=mounted,
    )
    base_python, python_environment, calls_path = _fake_base_python(tmp_path)
    environment.update(python_environment)
    arguments = (
        stage_arguments
        if stage_arguments is not None
        else _default_arguments(root if isinstance(root, Path) else Path("/invalid"))
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; shift; run_main "$@"',
            "phase2b2b-script-test",
            str(RUNNER),
            str(base_python),
            str(root),
            str(repo),
            stage,
            *arguments,
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
    return completed, calls, prohibited


@pytest.mark.parametrize("stage", ["preflight", "single", "smoke", "replay"])
def test_runner_uses_base_python_without_install_or_download_commands(
    tmp_path: Path, stage: str
) -> None:
    storage_root = tmp_path / "persistent"
    completed, calls, prohibited = _invoke(
        tmp_path, stage=stage, storage_root=storage_root
    )

    assert completed.returncode == 0, completed.stderr
    assert calls == [
        [
            "-m",
            "trustsr.cli.phase2b2b",
            stage,
            "--storage-root",
            str(storage_root),
            *_default_arguments(storage_root),
        ]
    ]
    assert not prohibited.exists()
    assert json.loads(completed.stdout) == {"stage": stage}
    log = storage_root / "trustsr" / "phase2b2b" / "logs" / f"{stage}.jsonl"
    assert log.read_text(encoding="utf-8") == completed.stdout


@pytest.mark.parametrize("bad_stage", ["benchmark", "all", "SMOKE", "smoke;id"])
def test_runner_rejects_unknown_stage_before_python(tmp_path: Path, bad_stage: str) -> None:
    completed, calls, _ = _invoke(tmp_path, stage=bad_stage)

    assert completed.returncode == 2
    assert calls == []


@pytest.mark.parametrize("bad_root", ["/", "/root", "relative", "/tmp/*"])
def test_runner_rejects_unsafe_storage_before_python(tmp_path: Path, bad_root: str) -> None:
    completed, calls, _ = _invoke(tmp_path, storage_root=bad_root)

    assert completed.returncode == 2
    assert calls == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mounted": False}, "mountpoint"),
        ({"available_kib": 8 * 1024 * 1024}, "more than 8 GiB"),
        ({"available_inodes": 1024}, "more than 1024 free inodes"),
    ],
)
def test_runner_requires_mount_capacity_and_inodes(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    completed, calls, _ = _invoke(tmp_path, **kwargs)

    assert completed.returncode == 2
    assert message in completed.stderr
    assert calls == []


@pytest.mark.parametrize(
    "arguments",
    [
        ["--selection-manifest", "/persistent/samples.jsonl"],
        ["--storage-root", "/different", "--confirm-cloud-storage"],
        ["--st", "/different", "--confirm-cloud-storage"],
    ],
)
def test_runner_requires_confirmation_and_rejects_storage_override(
    tmp_path: Path, arguments: list[str]
) -> None:
    completed, calls, _ = _invoke(tmp_path, stage_arguments=arguments)

    assert completed.returncode == 2
    assert calls == []


def test_runner_rejects_symlink_or_colon_repository(tmp_path: Path) -> None:
    real = _repository(tmp_path, "real-repository")
    link = tmp_path / "repository-link"
    link.symlink_to(real, target_is_directory=True)
    linked, calls, _ = _invoke(tmp_path, repository=link)
    assert linked.returncode == 2
    assert calls == []

    colon = _repository(tmp_path, "repository:colon")
    colon_result, calls, _ = _invoke(tmp_path, repository=colon)
    assert colon_result.returncode == 2
    assert calls == []
