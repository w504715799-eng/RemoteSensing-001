"""Executable contracts for the Phase 2B3-A workspace checkpoint boundary."""

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
CHECKPOINT = REPOSITORY / "scripts" / "phase2b3a" / "checkpoint_workspace.sh"
REVISION = "a" * 40
SELECTION_BYTES = b'{"fixture":"selection"}\n'
INPUT_AUDIT_BYTES = b'{"fixture":"input-audit"}\n'
SELECTION_DIGEST = hashlib.sha256(SELECTION_BYTES).hexdigest()
INPUT_AUDIT_DIGEST = hashlib.sha256(INPUT_AUDIT_BYTES).hexdigest()
EVIDENCE = {
    stage: tuple(
        f"phase2b3a-{stage}-{suffix}.json"
        for suffix in ("result", "cache-audit", "runtime", "replay")
    )
    for stage in ("a1", "a2")
}


def _make_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_workspace(workspace: Path, stage: str, *, corrupt: str | None = None) -> Path:
    selection = workspace / "trustsr" / "phase2b1b" / "selections" / SELECTION_DIGEST
    selection.mkdir(parents=True)
    selection_bytes = b"wrong" if corrupt == "selection" else SELECTION_BYTES
    (selection / "samples.jsonl").write_bytes(selection_bytes)
    audit = workspace / "trustsr" / "phase2b2a" / "input-audits" / SELECTION_DIGEST
    audit.mkdir(parents=True)
    audit_bytes = b"wrong" if corrupt == "input" else INPUT_AUDIT_BYTES
    (audit / "phase2b2a-input-audit.json").write_bytes(audit_bytes)
    phase_root = workspace / "trustsr" / "phase2b3a"
    phase_root.mkdir(parents=True, exist_ok=True)
    (phase_root / "cache.bin").write_bytes(b"cache")
    if stage == "a0":
        return phase_root
    result = phase_root / "results" / SELECTION_DIGEST
    result.mkdir(parents=True, exist_ok=True)
    payloads = {name: _canonical({"name": name, "phase": stage}) for name in EVIDENCE[stage]}
    for name, payload in payloads.items():
        (result / name).write_bytes(payload)
    manifest = {
        "schema": "trustsr.phase2b3a-bundle-manifest.v1",
        "phase": stage,
        "files": [
            {"basename": name, "sha256": hashlib.sha256(payloads[name]).hexdigest(), "size_bytes": len(payloads[name])}
            for name in sorted(payloads)
        ],
    }
    (result / "phase2b3a-bundle-manifest.json").write_bytes(_canonical(manifest))
    return phase_root


def _checkpoint_environment(
    tmp_path: Path,
    work_root: Path,
    persistent_root: Path,
    repository: Path,
    *,
    work_mounted: bool = True,
    persistent_mounted: bool = True,
    work_available_kib: int = 10 * 1024 * 1024,
    persistent_available_kib: int = 10 * 1024 * 1024,
    persistent_after_build_kib: int | None = None,
    persistent_inodes: int = 4,
    git_head: str = REVISION,
    git_branch: str = "reviewed",
    git_status: str = "",
) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "checkpoint-bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    prohibited = tmp_path / "prohibited-command"
    built = tmp_path / "build-finished"
    _make_executable(
        fake_bin / "mountpoint",
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "[[ \"$#\" == 3 && \"$1\" == -q && \"$2\" == -- ]]\n"
        "case \"$3\" in\n"
        "  \"$FAKE_WORK_ROOT\") [[ \"$FAKE_WORK_MOUNTED\" == 1 ]] ;;\n"
        "  \"$FAKE_PERSISTENT_ROOT\") [[ \"$FAKE_PERSISTENT_MOUNTED\" == 1 ]] ;;\n"
        "  *) exit 98 ;;\n"
        "esac\n",
    )
    _make_executable(
        fake_bin / "df",
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "[[ \"$#\" == 3 && \"$2\" == -- ]]\n"
        "root=\"$3\"\n"
        "case \"$1\" in\n"
        "  -Pk)\n"
        "    if [[ \"$root\" == \"$FAKE_WORK_ROOT\" ]]; then available=\"$FAKE_WORK_KIB\"; "
        "elif [[ -e \"$FAKE_BUILD_FINISHED\" && -n \"$FAKE_PERSISTENT_AFTER_BUILD_KIB\" ]]; then available=\"$FAKE_PERSISTENT_AFTER_BUILD_KIB\"; "
        "elif [[ \"$root\" == \"$FAKE_PERSISTENT_ROOT\" ]]; then available=\"$FAKE_PERSISTENT_KIB\"; else exit 98; fi\n"
        "    printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n/dev/fake 1 1 %s 1%% /fake\\n' \"$available\" ;;\n"
        "  -Pi) [[ \"$root\" == \"$FAKE_PERSISTENT_ROOT\" ]] || exit 98; printf 'Filesystem Inodes IUsed IFree IUse%% Mounted on\\n/dev/fake 1 1 %s 1%% /fake\\n' \"$FAKE_PERSISTENT_INODES\" ;;\n"
        "  *) exit 98 ;;\n"
        "esac\n",
    )
    _make_executable(
        fake_bin / "git",
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "[[ \"$1\" == -C && \"$2\" == \"$FAKE_REPOSITORY\" ]]\n"
        "case \"$3 $4\" in\n"
        "  'rev-parse HEAD') printf '%s\\n' \"$FAKE_GIT_HEAD\" ;;\n"
        "  'symbolic-ref --short') [[ \"${5:-}\" == HEAD ]]; printf '%s\\n' \"$FAKE_GIT_BRANCH\" ;;\n"
        "  'status --porcelain') printf '%s' \"$FAKE_GIT_STATUS\" ;;\n"
        "  *) exit 97 ;;\n"
        "esac\n",
    )
    for command in ("conda", "pip", "curl", "wget"):
        _make_executable(
            fake_bin / command,
            "#!/usr/bin/env bash\ntouch \"$PROHIBITED_COMMAND\"\nexit 99\n",
        )
    return (
        {
            **os.environ,
            "CHECKPOINT_SOURCE": str(REPOSITORY / "src"),
            "FAKE_BUILD_FINISHED": str(built),
            "FAKE_GIT_BRANCH": git_branch,
            "FAKE_GIT_HEAD": git_head,
            "FAKE_GIT_STATUS": git_status,
            "FAKE_PERSISTENT_AFTER_BUILD_KIB": "" if persistent_after_build_kib is None else str(persistent_after_build_kib),
            "FAKE_PERSISTENT_INODES": str(persistent_inodes),
            "FAKE_PERSISTENT_KIB": str(persistent_available_kib),
            "FAKE_PERSISTENT_MOUNTED": "1" if persistent_mounted else "0",
            "FAKE_PERSISTENT_ROOT": str(persistent_root),
            "FAKE_REPOSITORY": str(repository),
            "FAKE_WORK_KIB": str(work_available_kib),
            "FAKE_WORK_MOUNTED": "1" if work_mounted else "0",
            "FAKE_WORK_ROOT": str(work_root),
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "PROHIBITED_COMMAND": str(prohibited),
        },
        prohibited,
    )


def _real_base_python(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    launcher = tmp_path / "cloud-base-python"
    _make_executable(
        launcher,
        f"#!{sys.executable}\n"
        "import contextlib, io, json, os, sys\n"
        "sys.path.insert(0, os.environ['CHECKPOINT_SOURCE'])\n"
        "from trustsr.artifacts import workspace_checkpoint as checkpoint\n"
        "checkpoint.SELECTION_MANIFEST_SHA256 = os.environ['FAKE_SELECTION_DIGEST']\n"
        "checkpoint.INPUT_AUDIT_SHA256 = os.environ['FAKE_INPUT_AUDIT_DIGEST']\n"
        "arguments = sys.argv[1:]\n"
        "if arguments[:2] != ['-m', 'trustsr.artifacts.workspace_checkpoint']:\n"
        "    raise SystemExit(96)\n"
        "captured = io.StringIO()\n"
        "with contextlib.redirect_stdout(captured):\n"
        "    result = checkpoint.main(arguments[2:])\n"
        "output = captured.getvalue()\n"
        "mutation = json.loads(os.environ.get('FAKE_RECORD_MUTATIONS', '{}')).get(arguments[2])\n"
        "if mutation and result == 0:\n"
        "    before, separator, after = output.partition('\\\"archive_size_bytes\\\":')\n"
        "    if mutation in ('zero', 'leading-zero', 'size-one'):\n"
        "        replacement = {'zero': '0', 'leading-zero': '00', 'size-one': '1'}[mutation]\n"
        "        output = before + separator + replacement + ',' + after.split(',', 1)[1]\n"
        "    elif mutation == 'extra-output':\n"
        "        output += '{\\\"unexpected\\\":true}\\n'\n"
        "    else:\n"
        "        raise SystemExit(95)\n"
        "sys.stdout.write(output)\n"
        "if arguments[2:3] == ['build'] and result == 0:\n"
        "    Path = __import__('pathlib').Path\n"
        "    Path(os.environ['FAKE_BUILD_FINISHED']).touch()\n"
        "raise SystemExit(result)\n",
    )
    return launcher


def _invoke_checkpoint(
    tmp_path: Path, *, stage: str = "a0", **boundary_state: object
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    workspace = tmp_path / "work-mount"
    persistent = tmp_path / "persistent-mount"
    repository = workspace / "reviewed-repository"
    corrupt = boundary_state.pop("corrupt", None)
    record_mutations = boundary_state.pop("record_mutations", None)
    paths = boundary_state.pop("paths", None)
    equal_roots = bool(boundary_state.pop("equal_roots", False))
    _write_workspace(workspace, stage, corrupt=corrupt if isinstance(corrupt, str) else None)
    persistent.mkdir(exist_ok=True)
    (repository / "src" / "trustsr").mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    base_python = _real_base_python(tmp_path / "base-python")
    environment, prohibited = _checkpoint_environment(
        tmp_path, workspace, persistent, repository, **boundary_state
    )
    environment.update(
        {
            "FAKE_INPUT_AUDIT_DIGEST": INPUT_AUDIT_DIGEST,
            "FAKE_SELECTION_DIGEST": SELECTION_DIGEST,
        }
    )
    if isinstance(record_mutations, dict):
        environment["FAKE_RECORD_MUTATIONS"] = json.dumps(record_mutations)
    arguments = [str(base_python), str(workspace), str(persistent), str(repository), stage, REVISION]
    if isinstance(paths, dict):
        positions = {"base_python": 0, "workspace": 1, "persistent": 2, "repository": 3}
        for name, value in paths.items():
            arguments[positions[str(name)]] = str(value)
    if equal_roots:
        arguments[2] = str(workspace)
        environment["FAKE_PERSISTENT_ROOT"] = str(workspace)
    completed = subprocess.run(
        ["bash", str(CHECKPOINT), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed, persistent, prohibited


@pytest.mark.parametrize("stage", ["a0", "a1", "a2"])
def test_checkpoint_script_builds_publishes_and_reverifies(stage: str, tmp_path: Path) -> None:
    completed, persistent, prohibited = _invoke_checkpoint(tmp_path, stage=stage)
    assert completed.returncode == 0, completed.stderr
    record = json.loads(completed.stdout)
    assert record["completed_stage"] == stage
    assert record["status"] == "verify"
    checkpoint = persistent / "trustsr-phase2b3a-checkpoints"
    assert len(list(checkpoint.glob("*.tar"))) == 1
    assert len(list(checkpoint.glob("*.json"))) == 1
    assert not list(checkpoint.glob("*.part"))
    assert not (checkpoint / ".checkpoint.lock").exists()
    assert not list((tmp_path / "work-mount").glob(".phase2b3a-checkpoint.*"))
    assert not prohibited.exists()


@pytest.mark.parametrize(
    ("name", "state"),
    [
        ("unmounted-work-root", {"work_mounted": False}),
        ("unmounted-persistent-root", {"persistent_mounted": False}),
        ("equal-roots", {"equal_roots": True}),
        ("dirty-head", {"git_status": " M tracked"}),
        ("detached-head", {"git_branch": ""}),
        ("wrong-head", {"git_head": "b" * 40}),
        ("frozen-selection-digest", {"corrupt": "selection"}),
        ("frozen-input-digest", {"corrupt": "input"}),
        ("insufficient-work-bytes", {"work_available_kib": 10 * 1024 * 1024 - 1}),
        ("insufficient-persistent-bytes", {"persistent_after_build_kib": 1}),
        ("fewer-than-four-persistent-inodes", {"persistent_inodes": 3}),
    ],
)
def test_checkpoint_script_rejects_contract_boundary(
    tmp_path: Path, name: str, state: dict[str, object]
) -> None:
    completed, persistent, prohibited = _invoke_checkpoint(tmp_path, **state)
    assert completed.returncode == 2, (name, completed.stderr)
    assert not list((persistent / "trustsr-phase2b3a-checkpoints").glob("*.json"))
    assert not prohibited.exists()


@pytest.mark.parametrize("path_name", ["base_python", "workspace", "persistent", "repository"])
def test_checkpoint_script_rejects_symlink_component_in_each_positional_path(
    tmp_path: Path, path_name: str
) -> None:
    target = tmp_path / "symlink-target"
    target.mkdir()
    linked = tmp_path / "symlink-component"
    linked.symlink_to(target, target_is_directory=True)
    completed, persistent, prohibited = _invoke_checkpoint(
        tmp_path, paths={path_name: linked / "child"}
    )
    assert completed.returncode == 2, completed.stderr
    assert not list((persistent / "trustsr-phase2b3a-checkpoints").glob("*.json"))
    assert not prohibited.exists()


@pytest.mark.parametrize("reservation", ["stage", "checkpoint"])
def test_checkpoint_script_rejects_active_stage_or_checkpoint_reservation(
    tmp_path: Path, reservation: str
) -> None:
    workspace = tmp_path / "work-mount"
    persistent = tmp_path / "persistent-mount"
    if reservation == "stage":
        (workspace / "trustsr" / "phase2b3a" / "logs" / "a0.jsonl.lock").mkdir(parents=True)
    else:
        (persistent / "trustsr-phase2b3a-checkpoints" / ".checkpoint.lock").mkdir(parents=True)
    completed, observed_persistent, prohibited = _invoke_checkpoint(tmp_path)
    assert completed.returncode == 2, completed.stderr
    assert not list((observed_persistent / "trustsr-phase2b3a-checkpoints").glob("*.json"))
    assert not prohibited.exists()


def test_checkpoint_script_never_attempts_prohibited_environment_bootstrap(tmp_path: Path) -> None:
    completed, persistent, prohibited = _invoke_checkpoint(tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert list((persistent / "trustsr-phase2b3a-checkpoints").glob("*.json"))
    assert not prohibited.exists()


@pytest.mark.parametrize(
    "record_mutations",
    [
        {command: "zero" for command in ("build", "publish", "verify")},
        {command: "leading-zero" for command in ("build", "publish", "verify")},
        {"build": "extra-output"},
        {"publish": "size-one"},
    ],
)
def test_checkpoint_script_rejects_malformed_or_disagreeing_module_records(
    tmp_path: Path, record_mutations: dict[str, str]
) -> None:
    completed, _, prohibited = _invoke_checkpoint(
        tmp_path, record_mutations=record_mutations
    )
    assert completed.returncode == 2, completed.stderr
    assert completed.stdout == ""
    assert not prohibited.exists()
