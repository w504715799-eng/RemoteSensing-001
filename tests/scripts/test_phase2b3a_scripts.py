"""Executable security contracts for the Phase 2B3-A cloud boundary."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
RUNNER = REPOSITORY / "scripts" / "phase2b3a" / "run_cloud.sh"
PULLER = REPOSITORY / "scripts" / "phase2b3a" / "pull_results.sh"
REVISION = "a" * 40
POST_MANIFEST_SHA256 = "c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a"
PHASE_FILES = {
    phase: tuple(
        sorted(
            (
                f"phase2b3a-{phase}-result.json",
                f"phase2b3a-{phase}-cache-audit.json",
                f"phase2b3a-{phase}-runtime.json",
                f"phase2b3a-{phase}-replay.json",
            )
        )
    )
    for phase in ("a1", "a2")
}


def _make_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "reviewed-repository"
    (repository / "src" / "trustsr").mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return repository


def _runner_environment(
    tmp_path: Path,
    storage_root: Path,
    *,
    mounted: bool = True,
    available_kib: int = 10 * 1024 * 1024,
    available_inodes: int = 2048,
    git_head: str = REVISION,
    git_branch: str = "reviewed",
    git_status: str = "",
    git_exit: int = 0,
) -> tuple[dict[str, str], Path, Path]:
    fake_bin = tmp_path / "runner-bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    calls = tmp_path / "python-calls.jsonl"
    prohibited = tmp_path / "prohibited-command"
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    _make_executable(
        fake_bin / "mountpoint",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "[[ \"$#\" == 3 && \"$1\" == -q && \"$2\" == -- && \"$3\" == \"$FAKE_MOUNT_ROOT\" ]]\n"
        "[[ \"$FAKE_MOUNTED\" == 1 ]]\n",
    )
    _make_executable(
        fake_bin / "df",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "[[ \"$#\" == 3 && \"$2\" == -- && \"$3\" == \"$FAKE_MOUNT_ROOT\" ]]\n"
        "case \"$1\" in\n"
        "  -Pk)\n"
        "    printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'\n"
        "    printf '/dev/fake 1 1 %s 1%% /persistent\\n' \"$FAKE_AVAILABLE_KIB\"\n"
        "    ;;\n"
        "  -Pi)\n"
        "    printf 'Filesystem Inodes IUsed IFree IUse%% Mounted on\\n'\n"
        "    printf '/dev/fake 1 1 %s 1%% /persistent\\n' \"$FAKE_AVAILABLE_INODES\"\n"
        "    ;;\n"
        "  *) exit 98 ;;\n"
        "esac\n",
    )
    _make_executable(
        fake_bin / "git",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "[[ \"$FAKE_GIT_EXIT\" == 0 ]] || exit \"$FAKE_GIT_EXIT\"\n"
        "[[ \"$1\" == -C && \"$2\" == \"$FAKE_REPOSITORY\" ]]\n"
        "case \"$3 $4\" in\n"
        "  'rev-parse HEAD') printf '%s\\n' \"$FAKE_GIT_HEAD\" ;;\n"
        "  'symbolic-ref --short')\n"
        "    [[ \"${5:-}\" == HEAD ]]\n"
        "    printf '%s\\n' \"$FAKE_GIT_BRANCH\"\n"
        "    ;;\n"
        "  'status --porcelain') printf '%s' \"$FAKE_GIT_STATUS\" ;;\n"
        "  *) exit 97 ;;\n"
        "esac\n",
    )
    for name in ("conda", "pip", "curl", "wget"):
        _make_executable(
            fake_bin / name,
            "#!/usr/bin/env bash\ntouch \"$PROHIBITED_COMMAND\"\nexit 99\n",
        )
    environment = {
        **os.environ,
        "FAKE_AVAILABLE_INODES": str(available_inodes),
        "FAKE_AVAILABLE_KIB": str(available_kib),
        "FAKE_GIT_BRANCH": git_branch,
        "FAKE_GIT_EXIT": str(git_exit),
        "FAKE_GIT_HEAD": git_head,
        "FAKE_GIT_STATUS": git_status,
        "FAKE_MOUNT_ROOT": str(storage_root),
        "FAKE_MOUNTED": "1" if mounted else "0",
        "HOME": str(home),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "PROHIBITED_COMMAND": str(prohibited),
    }
    return environment, calls, prohibited


def _fake_base_python(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    launcher = tmp_path / "cloud-base-python"
    _make_executable(
        launcher,
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "with Path(os.environ['FAKE_PYTHON_CALLS']).open('a', encoding='utf-8') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        "print(json.dumps({'stage': args[2]}, sort_keys=True, separators=(',', ':')))\n",
    )
    return launcher


def _stage_arguments(storage_root: Path, stage: str) -> list[str]:
    arguments = [
        "--selection-manifest",
        str(storage_root / "samples.jsonl"),
        "--selection-manifest-sha256",
        "b" * 64,
        "--input-audit",
        str(storage_root / "input-audit.json"),
        "--input-audit-sha256",
        "c" * 64,
    ]
    if stage not in {"replay", "development-replay"}:
        arguments.extend(
            [
                "--sen2srlite-model-dir",
                str(storage_root / "models" / "sen2srlite"),
                "--ldsr-model-dir",
                str(storage_root / "models" / "ldsr"),
            ]
        )
    return [*arguments, "--reviewed-commit", REVISION, "--confirm-cloud-storage"]


def _invoke_runner(
    tmp_path: Path,
    *,
    stage: str = "smoke",
    storage_root: str | Path | None = None,
    repository: Path | None = None,
    base_python: Path | None = None,
    stage_arguments: list[str] | None = None,
    **environment_options: object,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]], Path]:
    root = storage_root if storage_root is not None else tmp_path / "persistent"
    if isinstance(root, Path):
        root.mkdir(parents=True, exist_ok=True)
        (root / "models" / "sen2srlite").mkdir(parents=True, exist_ok=True)
        (root / "models" / "ldsr").mkdir(parents=True, exist_ok=True)
        (root / "samples.jsonl").write_text("{}\n", encoding="utf-8")
        (root / "input-audit.json").write_text("{}", encoding="utf-8")
    repo = repository if repository is not None else _repository(tmp_path)
    environment, calls_path, prohibited = _runner_environment(
        tmp_path,
        root if isinstance(root, Path) else Path("/invalid"),
        **environment_options,
    )
    environment["FAKE_REPOSITORY"] = str(repo)
    environment["FAKE_PYTHON_CALLS"] = str(calls_path)
    python = base_python if base_python is not None else _fake_base_python(tmp_path)
    arguments = (
        stage_arguments
        if stage_arguments is not None
        else _stage_arguments(root if isinstance(root, Path) else Path("/invalid"), stage)
    )
    completed = subprocess.run(
        ["bash", str(RUNNER), str(python), str(root), str(repo), stage, *arguments],
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


@pytest.mark.parametrize(
    "stage",
    ["preflight", "single", "smoke", "replay", "development", "development-replay"],
)
def test_runner_uses_base_python_and_one_exact_stage(tmp_path: Path, stage: str) -> None:
    storage_root = tmp_path / "persistent"
    completed, calls, prohibited = _invoke_runner(
        tmp_path, stage=stage, storage_root=storage_root
    )

    assert completed.returncode == 0, completed.stderr
    assert calls == [
        [
            "-m",
            "trustsr.cli.phase2b3a",
            stage,
            "--storage-root",
            str(storage_root),
            "--project-root",
            str(tmp_path / "reviewed-repository"),
            *_stage_arguments(storage_root, stage),
        ]
    ]
    assert not prohibited.exists()
    assert completed.stdout == f'{{"stage":"{stage}"}}\n'
    log = storage_root / "trustsr" / "phase2b3a" / "logs" / f"{stage}.jsonl"
    assert log.read_text(encoding="utf-8") == completed.stdout


@pytest.mark.parametrize("stage", ["unknown", "SMOKE", "smoke;touch", "--smoke"])
def test_runner_rejects_unknown_stage_before_python(tmp_path: Path, stage: str) -> None:
    completed, calls, _ = _invoke_runner(tmp_path, stage=stage)
    assert completed.returncode == 2
    assert calls == []


@pytest.mark.parametrize(
    "bad_root", ["/", "/root", "relative", "/tmp/*", "/tmp/a:b", "/tmp/a\nnext"]
)
def test_runner_rejects_unsafe_storage_root(tmp_path: Path, bad_root: str) -> None:
    completed, calls, _ = _invoke_runner(tmp_path, storage_root=bad_root)
    assert completed.returncode == 2
    assert calls == []


def test_runner_rejects_symlink_components_for_storage_repository_and_python(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real-root"
    linked_root = tmp_path / "linked-root"
    real_root.mkdir()
    linked_root.symlink_to(real_root, target_is_directory=True)
    completed, calls, _ = _invoke_runner(tmp_path, storage_root=linked_root)
    assert completed.returncode == 2 and calls == []

    repository = _repository(tmp_path / "repo-case")
    repo_link = tmp_path / "repo-link"
    repo_link.symlink_to(repository, target_is_directory=True)
    completed, calls, _ = _invoke_runner(tmp_path / "repo-invoke", repository=repo_link)
    assert completed.returncode == 2 and calls == []

    python = _fake_base_python(tmp_path / "python-real")
    python_link = tmp_path / "python-link"
    python_link.symlink_to(python)
    completed, calls, _ = _invoke_runner(tmp_path / "python-invoke", base_python=python_link)
    assert completed.returncode == 2 and calls == []


def test_runner_rejects_interpreter_from_reviewed_environment(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    python = repository / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    _make_executable(python, "#!/usr/bin/env bash\nexit 99\n")
    completed, calls, _ = _invoke_runner(
        tmp_path, repository=repository, base_python=python
    )
    assert completed.returncode == 2
    assert calls == []


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"mounted": False}, "mountpoint"),
        ({"available_kib": 10 * 1024 * 1024 - 1}, "10 GiB"),
        ({"available_inodes": 1024}, "inodes"),
    ],
)
def test_runner_fails_closed_on_mount_disk_and_inode_checks(
    tmp_path: Path, options: dict[str, object], message: str
) -> None:
    completed, calls, _ = _invoke_runner(tmp_path, **options)
    assert completed.returncode == 2
    assert message in completed.stderr
    assert calls == []


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"git_branch": ""}, "attached"),
        ({"git_status": " M pyproject.toml"}, "clean"),
        ({"git_head": "b" * 40}, "reviewed commit"),
        ({"git_exit": 128}, "Git"),
    ],
)
def test_runner_requires_clean_attached_matching_repository(
    tmp_path: Path, options: dict[str, object], message: str
) -> None:
    completed, calls, _ = _invoke_runner(tmp_path, **options)
    assert completed.returncode == 2
    assert message in completed.stderr
    assert calls == []


@pytest.mark.parametrize("stage", ["smoke", "replay"])
def test_runner_requires_exactly_one_canonical_reviewed_commit(
    tmp_path: Path, stage: str
) -> None:
    root = tmp_path / "persistent"
    default = _stage_arguments(root, stage)
    for arguments in (
        default[:-3] + ["--confirm-cloud-storage"],
        [*default, "--reviewed-commit", REVISION],
        [*default[:-2], "A" * 40, "--confirm-cloud-storage"],
    ):
        completed, calls, _ = _invoke_runner(
            tmp_path / hashlib.sha256("\0".join(arguments).encode()).hexdigest()[:8],
            stage=stage,
            stage_arguments=arguments,
        )
        assert completed.returncode == 2
        assert calls == []


@pytest.mark.parametrize(
    ("stage", "mutation"),
    [
        ("smoke", lambda args: [item for item in args if item != "--confirm-cloud-storage"]),
        ("smoke", lambda args: args[:-5] + args[-3:]),
        ("replay", lambda args: [*args, "--sen2srlite-model-dir", "/models"]),
        ("replay", lambda args: [*args, "--api-token", "secret"]),
        ("smoke", lambda args: [*args, "--storage-root", "/different"]),
    ],
)
def test_runner_rejects_wrong_stage_argument_membership(
    tmp_path: Path, stage: str, mutation: Callable[[list[str]], list[str]]
) -> None:
    root = tmp_path / "persistent"
    completed, calls, _ = _invoke_runner(
        tmp_path, stage=stage, stage_arguments=mutation(_stage_arguments(root, stage))
    )
    assert completed.returncode == 2
    assert calls == []


def test_runner_rejects_option_like_or_non_normalized_stage_paths(tmp_path: Path) -> None:
    root = tmp_path / "persistent"
    for value in ("--checkpoint", str(root / "a" / ".." / "samples.jsonl")):
        arguments = _stage_arguments(root, "smoke")
        arguments[arguments.index("--selection-manifest") + 1] = value
        completed, calls, _ = _invoke_runner(
            tmp_path / hashlib.sha256(value.encode()).hexdigest()[:8],
            stage_arguments=arguments,
        )
        assert completed.returncode == 2
        assert calls == []


def test_runner_rejects_existing_log_and_does_not_evaluate_arguments(tmp_path: Path) -> None:
    root = tmp_path / "persistent"
    log = root / "trustsr" / "phase2b3a" / "logs" / "smoke.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text("old\n", encoding="utf-8")
    completed, calls, _ = _invoke_runner(tmp_path, storage_root=root)
    assert completed.returncode == 2 and calls == []
    assert log.read_text(encoding="utf-8") == "old\n"

    marker = tmp_path / "evaluated"
    arguments = _stage_arguments(root, "smoke")
    arguments[arguments.index("--input-audit") + 1] = f"$(touch {marker})"
    completed, calls, _ = _invoke_runner(
        tmp_path / "evaluation-case", stage_arguments=arguments
    )
    assert completed.returncode == 2 and calls == []
    assert not marker.exists()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _remote_bundle(root: Path, phase: str = "a1") -> dict[str, bytes]:
    root.mkdir(parents=True)
    payloads = {
        name: _canonical({"name": name, "phase": phase}) for name in PHASE_FILES[phase]
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    manifest = {
        "schema": "trustsr.phase2b3a-bundle-manifest.v1",
        "phase": phase,
        "files": [
            {
                "basename": name,
                "size_bytes": len(payloads[name]),
                "sha256": hashlib.sha256(payloads[name]).hexdigest(),
            }
            for name in PHASE_FILES[phase]
        ],
    }
    payloads["phase2b3a-bundle-manifest.json"] = _canonical(manifest)
    (root / "phase2b3a-bundle-manifest.json").write_bytes(
        payloads["phase2b3a-bundle-manifest.json"]
    )
    return payloads


def _pull_environment(tmp_path: Path, remote: Path) -> tuple[dict[str, str], Path, Path]:
    fake_bin = tmp_path / "pull-bin"
    fake_bin.mkdir(exist_ok=True)
    calls = tmp_path / "transport-calls.jsonl"
    prohibited = tmp_path / "shell-evaluated"
    scp = f"""#!{sys.executable}
import json, os, shutil, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['TRANSPORT_CALLS']).open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(['scp', *args]) + '\\n')
source, destination = args[-2:]
name = source.rsplit('/', 1)[-1]
if name == os.environ.get('FAIL_TRANSFER_NAME'):
    raise SystemExit(42)
remote = Path(os.environ['REMOTE_BUNDLE']) / name
target = Path(destination)
target.parent.mkdir(parents=True, exist_ok=True)
if name == os.environ.get('LOCAL_SYMLINK_NAME'):
    target.symlink_to(remote)
else:
    shutil.copyfile(remote, target)
if name == os.environ.get('TAMPER_TRANSFER_NAME'):
    target.write_bytes(target.read_bytes() + b'x')
"""
    ssh = f"""#!{sys.executable}
import hashlib, json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['TRANSPORT_CALLS']).open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(['ssh', *args]) + '\\n')
name = args[-1].rsplit('/', 1)[-1]
path = Path(os.environ['REMOTE_BUNDLE']) / name
if (
    path.is_symlink()
    or not path.is_file()
    or name == os.environ.get('REMOTE_COMPONENT_SYMLINK_NAME')
):
    raise SystemExit(44)
payload = path.read_bytes()
size = len(payload) + (1 if name == os.environ.get('REMOTE_SIZE_MISMATCH_NAME') else 0)
digest = hashlib.sha256(payload).hexdigest()
if name == os.environ.get('REMOTE_DIGEST_MISMATCH_NAME'):
    digest = '0' * 64
print(f'{{size}} {{digest}}')
"""
    _make_executable(fake_bin / "scp", scp)
    _make_executable(fake_bin / "ssh", ssh)
    return (
        {
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "REMOTE_BUNDLE": str(remote),
            "TRANSPORT_CALLS": str(calls),
        },
        calls,
        prohibited,
    )


def _invoke_puller(
    tmp_path: Path,
    *,
    host: str = "operator@gpu.example",
    port: str = "2222",
    remote_root: str = "/mnt/persistent",
    destination: str | Path | None = None,
    phase: str = "a1",
    mutate_remote: Callable[[Path], None] | None = None,
    environment_updates: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]], Path, Path]:
    remote = tmp_path / "remote-bundle"
    _remote_bundle(remote, phase)
    if mutate_remote is not None:
        mutate_remote(remote)
    environment, calls_path, prohibited = _pull_environment(tmp_path, remote)
    environment.update(environment_updates or {})
    output = Path(destination) if destination is not None else tmp_path / "accepted-bundle"
    completed = subprocess.run(
        ["bash", str(PULLER), host, port, remote_root, str(output)],
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
    return completed, calls, prohibited, output


@pytest.mark.parametrize("phase", ["a1", "a2"])
def test_puller_fetches_only_manifest_and_its_four_allowlisted_files(
    tmp_path: Path, phase: str
) -> None:
    def add_remote_internal_files(remote: Path) -> None:
        (remote / f"phase2b3a-{phase}-pair-commit.json").write_text("do not copy")
        (remote / ".phase2b3a.lock").write_text("do not copy")

    completed, calls, prohibited, output = _invoke_puller(
        tmp_path, phase=phase, mutate_remote=add_remote_internal_files
    )
    assert completed.returncode == 0, completed.stderr
    assert {path.name for path in output.iterdir()} == {
        "phase2b3a-bundle-manifest.json",
        *PHASE_FILES[phase],
    }
    scp_calls = [call for call in calls if call[0] == "scp"]
    assert len(scp_calls) == 5
    assert all(call[1:4] == ["-P", "2222", "--"] for call in scp_calls)
    assert scp_calls[0][4].endswith(
        f"/trustsr/phase2b3a/results/{POST_MANIFEST_SHA256}/phase2b3a-bundle-manifest.json"
    )
    assert [call[4].rsplit("/", 1)[-1] for call in scp_calls[1:]] == list(
        PHASE_FILES[phase]
    )
    ssh_calls = [call for call in calls if call[0] == "ssh"]
    assert len(ssh_calls) == 5
    assert all(call[1:5] == ["-p", "2222", "--", "operator@gpu.example"] for call in ssh_calls)
    assert ssh_calls[0][-1].endswith("/phase2b3a-bundle-manifest.json")
    assert not prohibited.exists()


@pytest.mark.parametrize("port", ["", "0", "65536", "22x", "-1", " 22"])
def test_puller_rejects_noncanonical_or_out_of_range_port(
    tmp_path: Path, port: str
) -> None:
    completed, calls, _, output = _invoke_puller(tmp_path, port=port)
    assert completed.returncode == 2
    assert calls == []
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("host", "-oProxyCommand=touch"),
        ("host", "operator @gpu.example"),
        ("host", "operator@gpu.example\nnext"),
        ("remote_root", "/"),
        ("remote_root", "/root"),
        ("remote_root", "relative"),
        ("remote_root", "/mnt/../root"),
        ("remote_root", "/mnt/-option"),
        ("remote_root", "/mnt/a:b"),
        ("remote_root", "/mnt/a b"),
        ("destination", "/"),
        ("destination", "/root"),
        ("destination", "relative"),
        ("destination", "/tmp/a/../b"),
        ("destination", "/tmp/a:b"),
    ],
)
def test_puller_rejects_hostile_connection_and_path_inputs(
    tmp_path: Path, field: str, value: str
) -> None:
    completed, calls, _, output = _invoke_puller(tmp_path, **{field: value})
    assert completed.returncode == 2
    assert calls == []
    if field != "destination":
        assert not output.exists()


def test_puller_rejects_local_symlink_component_without_transport(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    completed, calls, _, _ = _invoke_puller(
        tmp_path / "case", destination=link / "bundle"
    )
    assert completed.returncode == 2
    assert calls == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: {**value, "schema": "wrong"},
        lambda value: {**value, "phase": "a2" if value["phase"] == "a1" else "a1"},
        lambda value: {**value, "extra": True},
        lambda value: {**value, "files": value["files"][:3]},
        lambda value: {**value, "files": [*value["files"], value["files"][0]]},
        lambda value: {**value, "files": list(reversed(value["files"]))},
        lambda value: {
            **value,
            "files": [{**value["files"][0], "basename": "../escape.json"}, *value["files"][1:]],
        },
        lambda value: {
            **value,
            "files": [{**value["files"][0], "size_bytes": 5 * 1024**2 + 1}, *value["files"][1:]],
        },
    ],
)
def test_puller_rejects_noncanonical_schema_phase_order_duplicates_and_traversal(
    tmp_path: Path, mutation: Callable[[dict[str, object]], dict[str, object]]
) -> None:
    def mutate(remote: Path) -> None:
        path = remote / "phase2b3a-bundle-manifest.json"
        value = json.loads(path.read_bytes())
        path.write_bytes(_canonical(mutation(value)))

    completed, _, _, output = _invoke_puller(tmp_path, mutate_remote=mutate)
    assert completed.returncode != 0
    assert not output.exists()


def test_puller_rejects_noncanonical_manifest_bytes(tmp_path: Path) -> None:
    def mutate(remote: Path) -> None:
        path = remote / "phase2b3a-bundle-manifest.json"
        value = json.loads(path.read_bytes())
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    completed, _, _, output = _invoke_puller(tmp_path, mutate_remote=mutate)
    assert completed.returncode != 0
    assert not output.exists()


def test_puller_rejects_remote_symlink_component_for_manifest(tmp_path: Path) -> None:
    completed, _, _, output = _invoke_puller(
        tmp_path,
        environment_updates={
            "REMOTE_COMPONENT_SYMLINK_NAME": "phase2b3a-bundle-manifest.json"
        },
    )
    assert completed.returncode != 0
    assert not output.exists()


@pytest.mark.parametrize(
    "environment_updates",
    [
        {"REMOTE_SIZE_MISMATCH_NAME": PHASE_FILES["a1"][0]},
        {"REMOTE_DIGEST_MISMATCH_NAME": PHASE_FILES["a1"][0]},
        {"TAMPER_TRANSFER_NAME": PHASE_FILES["a1"][0]},
    ],
)
def test_puller_rejects_remote_or_local_size_and_digest_mismatch(
    tmp_path: Path, environment_updates: dict[str, str]
) -> None:
    completed, _, _, output = _invoke_puller(
        tmp_path, environment_updates=environment_updates
    )
    assert completed.returncode != 0
    assert not output.exists()
    assert not list(tmp_path.glob(".phase2b3a-pull.*"))


@pytest.mark.parametrize("location", ["remote", "local"])
def test_puller_rejects_remote_or_local_symlink_file(tmp_path: Path, location: str) -> None:
    name = PHASE_FILES["a1"][0]

    def mutate(remote: Path) -> None:
        path = remote / name
        path.unlink()
        path.symlink_to(remote / PHASE_FILES["a1"][1])

    completed, _, _, output = _invoke_puller(
        tmp_path,
        mutate_remote=mutate if location == "remote" else None,
        environment_updates={"LOCAL_SYMLINK_NAME": name} if location == "local" else None,
    )
    assert completed.returncode != 0
    assert not output.exists()


def test_puller_removes_partial_transfer_and_preserves_existing_destination(
    tmp_path: Path,
) -> None:
    name = PHASE_FILES["a1"][1]
    completed, _, _, output = _invoke_puller(
        tmp_path, environment_updates={"FAIL_TRANSFER_NAME": name}
    )
    assert completed.returncode != 0
    assert not output.exists()
    assert not list(tmp_path.glob(".phase2b3a-pull.*"))

    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("original", encoding="utf-8")
    completed, _, _, _ = _invoke_puller(
        tmp_path / "collision", destination=existing
    )
    assert completed.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "original"


def test_puller_does_not_evaluate_host_or_path_text(tmp_path: Path) -> None:
    marker = tmp_path / "evaluated"
    completed, calls, _, output = _invoke_puller(
        tmp_path,
        host=f"$(touch {marker})",
        remote_root=f"/mnt/$(touch {marker})",
    )
    assert completed.returncode == 2
    assert calls == []
    assert not output.exists()
    assert not marker.exists()
