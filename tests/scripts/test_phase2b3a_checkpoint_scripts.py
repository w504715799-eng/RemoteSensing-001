"""Executable contracts for the Phase 2B3-A workspace checkpoint boundary."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
CHECKPOINT = REPOSITORY / "scripts" / "phase2b3a" / "checkpoint_workspace.sh"
RESTORE = REPOSITORY / "scripts" / "phase2b3a" / "restore_workspace.sh"
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
    result = phase_root / "results" / SELECTION_DIGEST
    result.mkdir(parents=True, exist_ok=True)
    logs = phase_root / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "preflight.jsonl").write_bytes(b'{"stage":"preflight"}\n')
    (result / "phase2b3a-preflight-runtime.json").write_bytes(
        _canonical({"git_commit": REVISION, "stage": "preflight"})
    )
    if stage == "a0":
        return phase_root
    payloads = {name: _canonical({"name": name, "phase": stage}) for name in EVIDENCE[stage]}
    for name, payload in payloads.items():
        (result / name).write_bytes(payload)
    manifest = {
        "schema": "trustsr.phase2b3a-bundle-manifest.v1",
        "phase": stage,
        "files": [
            {
                "basename": name,
                "sha256": hashlib.sha256(payloads[name]).hexdigest(),
                "size_bytes": len(payloads[name]),
            }
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
        "    if [[ \"$root\" == \"$FAKE_WORK_ROOT\" ]]; then "
        "available=\"$FAKE_WORK_KIB\"; "
        "elif [[ -e \"$FAKE_BUILD_FINISHED\" && "
        "-n \"$FAKE_PERSISTENT_AFTER_BUILD_KIB\" ]]; then "
        "available=\"$FAKE_PERSISTENT_AFTER_BUILD_KIB\"; "
        "elif [[ \"$root\" == \"$FAKE_PERSISTENT_ROOT\" ]]; then "
        "available=\"$FAKE_PERSISTENT_KIB\"; else exit 98; fi\n"
        "    printf 'Filesystem 1024-blocks Used Available Capacity "
        "Mounted on\\n/dev/fake 1 1 %s 1%% /fake\\n' \"$available\" ;;\n"
        "  -Pi) [[ \"$root\" == \"$FAKE_PERSISTENT_ROOT\" ]] || exit 98; "
        "printf 'Filesystem Inodes IUsed IFree IUse%% Mounted on\\n"
        "/dev/fake 1 1 %s 1%% /fake\\n' \"$FAKE_PERSISTENT_INODES\" ;;\n"
        "  *) exit 98 ;;\n"
        "esac\n",
    )
    _make_executable(
        fake_bin / "git",
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "[[ \"$1\" == -C && \"$2\" == \"$FAKE_REPOSITORY\" ]]\n"
        "case \"$3 $4\" in\n"
        "  'rev-parse HEAD') printf '%s\\n' \"$FAKE_GIT_HEAD\" ;;\n"
        "  'symbolic-ref --short') [[ \"${5:-}\" == HEAD ]]; "
        "printf '%s\\n' \"$FAKE_GIT_BRANCH\" ;;\n"
        "  'status --porcelain') printf '%s' \"$FAKE_GIT_STATUS\" ;;\n"
        "  'merge-base --is-ancestor') [[ \"$FAKE_CHECKPOINT_ANCESTOR\" == 1 ]] ;;\n"
        "  *) exit 97 ;;\n"
        "esac\n",
    )
    _make_executable(
        fake_bin / "mount",
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "calls_path = os.environ['FAKE_MOUNT_CALLS']\n"
        "with open(calls_path, 'a', encoding='utf-8') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "with open(calls_path, encoding='utf-8') as stream:\n"
        "    call_number = sum(1 for _ in stream)\n"
        "if call_number == int(os.environ.get('FAKE_MOUNT_FAIL_CALL', '0')):\n"
        "    raise SystemExit(32)\n"
        "state_path = os.environ['FAKE_MOUNT_STATE']\n"
        "try:\n"
        "    with open(state_path, encoding='utf-8') as stream:\n"
        "        state = json.load(stream)\n"
        "except FileNotFoundError:\n"
        "    state = {}\n"
        "if sys.argv[1:2] == ['--bind']:\n"
        "    state[sys.argv[3]] = sys.argv[2]\n"
        "with open(state_path, 'w', encoding='utf-8') as stream:\n"
        "    json.dump(state, stream)\n",
    )
    _make_executable(
        fake_bin / "umount",
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['FAKE_UMOUNT_CALLS'], 'a', encoding='utf-8') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "state_path = os.environ['FAKE_MOUNT_STATE']\n"
        "try:\n"
        "    with open(state_path, encoding='utf-8') as stream:\n"
        "        state = json.load(stream)\n"
        "except FileNotFoundError:\n"
        "    state = {}\n"
        "state.pop(sys.argv[-1], None)\n"
        "with open(state_path, 'w', encoding='utf-8') as stream:\n"
        "    json.dump(state, stream)\n",
    )
    _make_executable(
        fake_bin / "findmnt",
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "arguments = sys.argv[1:]\n"
        "with open(os.environ['FAKE_FINDMNT_CALLS'], 'a', encoding='utf-8') as stream:\n"
        "    stream.write(json.dumps(arguments) + '\\n')\n"
        "if (len(arguments) != 5 or arguments[:2] != ['-n', '-o'] "
        "or arguments[3] != '--target'):\n"
        "    raise SystemExit(98)\n"
        "with open(os.environ['FAKE_MOUNT_STATE'], encoding='utf-8') as stream:\n"
        "    state = json.load(stream)\n"
        "target = arguments[4]\n"
        "if arguments[2] == 'MAJ:MIN,FSROOT,TARGET':\n"
        "    if target in state:\n"
        "        relative = os.path.relpath(state[target], os.environ['FAKE_PERSISTENT_ROOT'])\n"
        "        fsroot = '/' + relative if relative != '.' else '/'\n"
        "        if target == os.environ.get('FAKE_FINDMNT_WRONG_IDENTITY_TARGET'):\n"
        "            fsroot = '/unexpected/model'\n"
        "        print(f'0:42 {fsroot} {target}')\n"
        "    elif target in state.values():\n"
        "        print(f\"0:42 / {os.environ['FAKE_PERSISTENT_ROOT']}\")\n"
        "    else:\n"
        "        raise SystemExit(1)\n"
        "elif arguments[2] == 'OPTIONS':\n"
        "    if target not in state:\n"
        "        raise SystemExit(1)\n"
        "    no_ro = target == os.environ.get('FAKE_FINDMNT_NO_RO_TARGET')\n"
        "    print('rw,relatime,bind' if no_ro else 'rw,relatime,bind,ro')\n"
        "else:\n"
        "    raise SystemExit(98)\n",
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
            "FAKE_FINDMNT_CALLS": str(tmp_path / "findmnt-calls.jsonl"),
            "FAKE_CHECKPOINT_ANCESTOR": "1",
            "FAKE_MOUNT_CALLS": str(tmp_path / "mount-calls.jsonl"),
            "FAKE_MOUNT_STATE": str(tmp_path / "mount-state.json"),
            "FAKE_PERSISTENT_AFTER_BUILD_KIB": (
                "" if persistent_after_build_kib is None else str(persistent_after_build_kib)
            ),
            "FAKE_PERSISTENT_INODES": str(persistent_inodes),
            "FAKE_PERSISTENT_KIB": str(persistent_available_kib),
            "FAKE_PERSISTENT_MOUNTED": "1" if persistent_mounted else "0",
            "FAKE_PERSISTENT_ROOT": str(persistent_root),
            "FAKE_REPOSITORY": str(repository),
            "FAKE_WORK_KIB": str(work_available_kib),
            "FAKE_WORK_MOUNTED": "1" if work_mounted else "0",
            "FAKE_WORK_ROOT": str(work_root),
            "FAKE_UMOUNT_CALLS": str(tmp_path / "umount-calls.jsonl"),
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
        "from trustsr.artifacts import model_restore, workspace_checkpoint as checkpoint\n"
        "checkpoint.SELECTION_MANIFEST_SHA256 = os.environ['FAKE_SELECTION_DIGEST']\n"
        "checkpoint.INPUT_AUDIT_SHA256 = os.environ['FAKE_INPUT_AUDIT_DIGEST']\n"
        "arguments = sys.argv[1:]\n"
        "if arguments[:2] == ['-m', 'trustsr.artifacts.model_restore']:\n"
        "    if os.environ.get('FAKE_MODEL_COPY_FAIL_AFTER_PUBLISH') == '1':\n"
        "        Path = __import__('pathlib').Path\n"
        "        target = Path(arguments[2])\n"
        "        (target / 'sen2srlite').mkdir(parents=True)\n"
        "        (target / 'ldsr-s2').mkdir()\n"
        "        raise SystemExit(2)\n"
        "    raise SystemExit(model_restore.main(arguments[2:]))\n"
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
    arguments = [
        str(base_python),
        str(workspace),
        str(persistent),
        str(repository),
        stage,
        REVISION,
    ]
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


def _recorded_calls(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _invoke_restore(
    tmp_path: Path, **boundary_state: object
) -> tuple[subprocess.CompletedProcess[str], dict[str, Path], Path]:
    workspace = tmp_path / "work-mount"
    persistent = tmp_path / "persistent-mount"
    repository = workspace / "reviewed-repository"
    checkpoint_stage = str(boundary_state.pop("checkpoint_stage", "a0"))
    _write_workspace(workspace, checkpoint_stage)
    checkpoint_preflight_state = boundary_state.pop("checkpoint_preflight_state", None)
    phase_root = workspace / "trustsr" / "phase2b3a"
    if checkpoint_preflight_state == "extra-runtime":
        extra_result = phase_root / "results" / ("f" * 64)
        extra_result.mkdir(parents=True)
        (extra_result / "phase2b3a-preflight-runtime.json").write_bytes(
            _canonical({"git_commit": REVISION, "stage": "preflight"})
        )
    elif checkpoint_preflight_state == "archive-collision":
        (
            phase_root / "logs" / f"preflight-{checkpoint_stage}-{REVISION}.jsonl"
        ).write_bytes(b"preserved")
    elif checkpoint_preflight_state is not None:
        raise AssertionError(f"unknown checkpoint preflight state: {checkpoint_preflight_state}")
    persistent.mkdir()
    (repository / "src" / "trustsr").mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    base_python = _real_base_python(tmp_path / "base-python")
    restore_git_head = str(boundary_state.pop("git_head", REVISION))
    restore_mode = boundary_state.pop("restore_mode", None)
    checkpoint_reviewed_commit = boundary_state.pop("checkpoint_reviewed_commit", None)
    checkpoint_is_ancestor = bool(boundary_state.pop("checkpoint_is_ancestor", True))
    model_copy_fail_after_publish = bool(
        boundary_state.pop("model_copy_fail_after_publish", False)
    )
    environment, prohibited = _checkpoint_environment(
        tmp_path, workspace, persistent, repository
    )
    environment.update(
        {
            "FAKE_INPUT_AUDIT_DIGEST": INPUT_AUDIT_DIGEST,
            "FAKE_MODEL_COPY_FAIL_AFTER_PUBLISH": (
                "1" if model_copy_fail_after_publish else "0"
            ),
            "FAKE_SELECTION_DIGEST": SELECTION_DIGEST,
        }
    )
    built = subprocess.run(
        [
            "bash",
            str(CHECKPOINT),
            str(base_python),
            str(workspace),
            str(persistent),
            str(repository),
            checkpoint_stage,
            REVISION,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if built.returncode != 0:
        raise AssertionError(f"restore fixture checkpoint failed: {built.stderr}")
    manifest_basename = next(
        (persistent / "trustsr-phase2b3a-checkpoints").glob("*.json")
    ).name
    manifest_basename = str(
        boundary_state.pop("manifest_basename", manifest_basename)
    )
    shutil.rmtree(workspace / "trustsr")
    environment["FAKE_GIT_HEAD"] = restore_git_head
    environment["FAKE_CHECKPOINT_ANCESTOR"] = "1" if checkpoint_is_ancestor else "0"

    models = persistent / "models"
    sen2_source = models / "sen2srlite"
    ldsr_source = models / "ldsr-s2"
    sen2_source.mkdir(parents=True)
    ldsr_source.mkdir()
    (sen2_source / "weights.bin").write_bytes(b"sen2")
    (ldsr_source / "weights.bin").write_bytes(b"ldsr")
    paths = {
        "workspace": workspace,
        "persistent": persistent,
        "repository": repository,
        "sen2_source": sen2_source,
        "ldsr_source": ldsr_source,
        "mount_calls": tmp_path / "mount-calls.jsonl",
        "mount_state": tmp_path / "mount-state.json",
        "umount_calls": tmp_path / "umount-calls.jsonl",
        "findmnt_calls": tmp_path / "findmnt-calls.jsonl",
    }

    state = str(boundary_state.pop("state", ""))
    if state == "live-trustsr":
        (workspace / "trustsr").mkdir()
        (workspace / "trustsr" / "collision").write_text("live", encoding="utf-8")
    elif state == "missing-model-source":
        shutil.rmtree(sen2_source)
    elif state == "outside-model-source":
        sen2_source = tmp_path / "outside-model"
        sen2_source.mkdir()
        paths["sen2_source"] = sen2_source
    elif state == "symlink-model-source":
        real_source = models / "real-sen2srlite"
        sen2_source.rename(real_source)
        sen2_source.symlink_to(real_source, target_is_directory=True)
    elif state == "symlink-model-target":
        target_parent = workspace / "model-mounts"
        target_parent.mkdir()
        outside_target = tmp_path / "outside-target"
        outside_target.mkdir()
        (target_parent / "sen2srlite").symlink_to(outside_target, target_is_directory=True)
    elif state == "nonempty-model-target":
        target = workspace / "model-mounts" / "sen2srlite"
        target.mkdir(parents=True)
        (target / "collision").write_text("occupied", encoding="utf-8")
    elif state == "hard-linked-model-source":
        os.link(sen2_source / "weights.bin", sen2_source / "weights-copy.bin")

    fail_call = boundary_state.pop("mount_fail_call", None)
    if fail_call is not None:
        environment["FAKE_MOUNT_FAIL_CALL"] = str(fail_call)
    no_ro_target = boundary_state.pop("no_ro_target", None)
    if no_ro_target is not None:
        environment["FAKE_FINDMNT_NO_RO_TARGET"] = str(
            workspace / "model-mounts" / str(no_ro_target)
        )
    wrong_identity_target = boundary_state.pop("wrong_identity_target", None)
    if wrong_identity_target is not None:
        environment["FAKE_FINDMNT_WRONG_IDENTITY_TARGET"] = str(
            workspace / "model-mounts" / str(wrong_identity_target)
        )
    record_mutations = boundary_state.pop("record_mutations", None)
    if isinstance(record_mutations, dict):
        environment["FAKE_RECORD_MUTATIONS"] = json.dumps(record_mutations)
    if boundary_state:
        raise AssertionError(f"unknown restore boundary state: {boundary_state}")

    arguments = [
        "bash",
        str(RESTORE),
        str(base_python),
        str(workspace),
        str(persistent),
        str(repository),
        manifest_basename,
        str(sen2_source),
        str(ldsr_source),
    ]
    if restore_mode is not None:
        arguments.append(str(restore_mode))
    if checkpoint_reviewed_commit is not None:
        if restore_mode is None:
            arguments.append("bind")
        arguments.append(str(checkpoint_reviewed_commit))
    completed = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed, paths, prohibited


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


def test_restore_script_mounts_models_read_only_then_restores_explicit_checkpoint(
    tmp_path: Path,
) -> None:
    completed, paths, prohibited = _invoke_restore(tmp_path)
    workspace = paths["workspace"]
    sen2_target = workspace / "model-mounts" / "sen2srlite"
    ldsr_target = workspace / "model-mounts" / "ldsr-s2"

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("\n") == 1
    assert json.loads(completed.stdout)["status"] == "restore"
    assert json.loads(completed.stdout)["model_restore_mode"] == "bind"
    assert _recorded_calls(paths["mount_calls"]) == [
        ["--bind", str(paths["sen2_source"]), str(sen2_target)],
        ["-o", "remount,bind,ro", str(sen2_target)],
        ["--bind", str(paths["ldsr_source"]), str(ldsr_target)],
        ["-o", "remount,bind,ro", str(ldsr_target)],
    ]
    assert _recorded_calls(paths["findmnt_calls"]) == [
        [
            "-n",
            "-o",
            "MAJ:MIN,FSROOT,TARGET",
            "--target",
            str(paths["sen2_source"]),
        ],
        ["-n", "-o", "MAJ:MIN,FSROOT,TARGET", "--target", str(sen2_target)],
        ["-n", "-o", "OPTIONS", "--target", str(sen2_target)],
        [
            "-n",
            "-o",
            "MAJ:MIN,FSROOT,TARGET",
            "--target",
            str(paths["ldsr_source"]),
        ],
        ["-n", "-o", "MAJ:MIN,FSROOT,TARGET", "--target", str(ldsr_target)],
        ["-n", "-o", "OPTIONS", "--target", str(ldsr_target)],
    ]
    assert _recorded_calls(paths["umount_calls"]) == []
    assert stat.S_IMODE(sen2_target.stat().st_mode) == 0o700
    assert stat.S_IMODE(ldsr_target.stat().st_mode) == 0o700
    assert (workspace / "trustsr/phase2b1b").is_dir()
    assert (workspace / "trustsr/phase2b2a").is_dir()
    assert (workspace / "trustsr/phase2b3a").is_dir()
    assert not prohibited.exists()


def test_restore_script_copies_verified_models_without_mount_privilege(tmp_path: Path) -> None:
    completed, paths, prohibited = _invoke_restore(tmp_path, restore_mode="copy")
    workspace = paths["workspace"]
    sen2_target = workspace / "model-mounts" / "sen2srlite"
    ldsr_target = workspace / "model-mounts" / "ldsr-s2"

    assert completed.returncode == 0, completed.stderr
    record = json.loads(completed.stdout)
    assert record["status"] == "restore"
    assert record["model_restore_mode"] == "copy"
    assert (sen2_target / "weights.bin").read_bytes() == b"sen2"
    assert (ldsr_target / "weights.bin").read_bytes() == b"ldsr"
    assert stat.S_IMODE((workspace / "model-mounts").stat().st_mode) == 0o500
    assert stat.S_IMODE(sen2_target.stat().st_mode) & 0o222 == 0
    assert stat.S_IMODE((sen2_target / "weights.bin").stat().st_mode) & 0o222 == 0
    assert stat.S_IMODE(ldsr_target.stat().st_mode) & 0o222 == 0
    assert stat.S_IMODE((ldsr_target / "weights.bin").stat().st_mode) & 0o222 == 0
    assert _recorded_calls(paths["mount_calls"]) == []
    assert _recorded_calls(paths["umount_calls"]) == []
    assert _recorded_calls(paths["findmnt_calls"]) == []
    assert (workspace / "trustsr/phase2b3a").is_dir()
    assert not list(workspace.glob(".phase2b3a-model-copy.*"))
    assert not prohibited.exists()


def test_restore_script_explicitly_binds_checkpoint_and_restore_code_commits(
    tmp_path: Path,
) -> None:
    restore_code_commit = "b" * 40
    completed, paths, prohibited = _invoke_restore(
        tmp_path,
        git_head=restore_code_commit,
        restore_mode="copy",
        checkpoint_reviewed_commit=REVISION,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(completed.stdout)
    assert set(record) == {
        "archive_basename",
        "archive_sha256",
        "archive_size_bytes",
        "checkpoint_reviewed_commit",
        "completed_stage",
        "manifest_basename",
        "model_restore_mode",
        "restore_code_commit",
        "status",
    }
    assert record["checkpoint_reviewed_commit"] == REVISION
    assert record["restore_code_commit"] == restore_code_commit
    assert record["model_restore_mode"] == "copy"
    assert record["completed_stage"] == "a0"
    assert record["status"] == "restore"
    assert (paths["workspace"] / "trustsr/phase2b3a").is_dir()
    assert not prohibited.exists()


def test_restore_script_archives_fixed_preflight_outputs_before_resumed_stage(
    tmp_path: Path,
) -> None:
    restore_code_commit = "b" * 40
    completed, paths, prohibited = _invoke_restore(
        tmp_path,
        checkpoint_stage="a1",
        git_head=restore_code_commit,
        restore_mode="copy",
        checkpoint_reviewed_commit=REVISION,
    )

    assert completed.returncode == 0, completed.stderr
    phase_root = paths["workspace"] / "trustsr" / "phase2b3a"
    result = phase_root / "results" / SELECTION_DIGEST
    assert not (phase_root / "logs" / "preflight.jsonl").exists()
    assert (
        phase_root / "logs" / f"preflight-a1-{REVISION}.jsonl"
    ).read_bytes() == b'{"stage":"preflight"}\n'
    assert not (result / "phase2b3a-preflight-runtime.json").exists()
    assert (
        result / f"phase2b3a-a1-preflight-runtime-{REVISION}.json"
    ).read_bytes() == _canonical({"git_commit": REVISION, "stage": "preflight"})
    assert json.loads(completed.stdout)["completed_stage"] == "a1"
    assert not prohibited.exists()


@pytest.mark.parametrize("preflight_state", ["extra-runtime", "archive-collision"])
def test_restore_script_rolls_back_ambiguous_preflight_archive(
    tmp_path: Path, preflight_state: str
) -> None:
    completed, paths, prohibited = _invoke_restore(
        tmp_path,
        checkpoint_stage="a1",
        git_head="b" * 40,
        restore_mode="copy",
        checkpoint_reviewed_commit=REVISION,
        checkpoint_preflight_state=preflight_state,
    )

    assert completed.returncode == 2, completed.stderr
    assert completed.stdout == ""
    assert not (paths["workspace"] / "trustsr").exists()
    assert not (paths["workspace"] / "model-mounts").exists()
    assert not prohibited.exists()


def test_restore_script_rejects_checkpoint_commit_outside_restore_history(
    tmp_path: Path,
) -> None:
    completed, paths, prohibited = _invoke_restore(
        tmp_path,
        git_head="b" * 40,
        restore_mode="copy",
        checkpoint_reviewed_commit=REVISION,
        checkpoint_is_ancestor=False,
    )

    assert completed.returncode == 2, completed.stderr
    assert completed.stdout == ""
    assert not (paths["workspace"] / "trustsr").exists()
    assert not (paths["workspace"] / "model-mounts").exists()
    assert _recorded_calls(paths["mount_calls"]) == []
    assert not prohibited.exists()


@pytest.mark.parametrize(
    ("name", "state"),
    [
        ("unknown-copy-mode", {"restore_mode": "automatic"}),
        (
            "hard-linked-copy-source",
            {"restore_mode": "copy", "state": "hard-linked-model-source"},
        ),
    ],
)
def test_restore_script_rejects_invalid_copy_restore(
    tmp_path: Path, name: str, state: dict[str, object]
) -> None:
    completed, paths, prohibited = _invoke_restore(tmp_path, **state)

    assert completed.returncode == 2, (name, completed.stderr)
    assert completed.stdout == ""
    assert not (paths["workspace"] / "trustsr").exists()
    assert not (paths["workspace"] / "model-mounts").exists()
    assert _recorded_calls(paths["mount_calls"]) == []
    assert not prohibited.exists()


def test_restore_script_cleans_copy_published_by_failing_subprocess(tmp_path: Path) -> None:
    completed, paths, prohibited = _invoke_restore(
        tmp_path,
        restore_mode="copy",
        model_copy_fail_after_publish=True,
    )

    assert completed.returncode == 2, completed.stderr
    assert completed.stdout == ""
    assert not (paths["workspace"] / "trustsr").exists()
    assert not (paths["workspace"] / "model-mounts").exists()
    assert not prohibited.exists()


@pytest.mark.parametrize(
    ("name", "state"),
    [
        ("unsafe-manifest", {"manifest_basename": "../checkpoint.json"}),
        ("mismatched-git-commit", {"git_head": "b" * 40}),
        ("live-trustsr", {"state": "live-trustsr"}),
        ("missing-model-source", {"state": "missing-model-source"}),
        ("outside-model-source", {"state": "outside-model-source"}),
        ("symlink-model-source", {"state": "symlink-model-source"}),
        ("symlink-model-target", {"state": "symlink-model-target"}),
        ("nonempty-model-target", {"state": "nonempty-model-target"}),
    ],
)
def test_restore_script_rejects_pre_mount_boundary(
    tmp_path: Path, name: str, state: dict[str, object]
) -> None:
    completed, paths, prohibited = _invoke_restore(tmp_path, **state)

    assert completed.returncode == 2, (name, completed.stderr)
    assert completed.stdout == ""
    assert _recorded_calls(paths["mount_calls"]) == []
    if name == "live-trustsr":
        assert (paths["workspace"] / "trustsr/collision").read_text(encoding="utf-8") == "live"
    else:
        assert not (paths["workspace"] / "trustsr").exists()
    assert not prohibited.exists()


@pytest.mark.parametrize(
    ("name", "state", "expected_mount_count", "expected_unmount_targets"),
    [
        (
            "second-bind-failure",
            {"mount_fail_call": 3},
            3,
            ["sen2srlite"],
        ),
        (
            "second-read-only-remount-failure",
            {"mount_fail_call": 4},
            4,
            ["ldsr-s2", "sen2srlite"],
        ),
        (
            "reported-options-lack-ro",
            {"no_ro_target": "sen2srlite"},
            2,
            ["sen2srlite"],
        ),
        (
            "reported-bind-identity-mismatch",
            {"wrong_identity_target": "sen2srlite"},
            2,
            ["sen2srlite"],
        ),
    ],
)
def test_restore_script_unwinds_failed_mount_transaction_in_reverse_order(
    tmp_path: Path,
    name: str,
    state: dict[str, object],
    expected_mount_count: int,
    expected_unmount_targets: list[str],
) -> None:
    completed, paths, prohibited = _invoke_restore(tmp_path, **state)
    workspace = paths["workspace"]

    assert completed.returncode == 2, (name, completed.stderr)
    assert completed.stdout == ""
    assert len(_recorded_calls(paths["mount_calls"])) == expected_mount_count
    assert _recorded_calls(paths["umount_calls"]) == [
        ["--", str(workspace / "model-mounts" / basename)]
        for basename in expected_unmount_targets
    ]
    assert not (workspace / "trustsr").exists()
    assert not (workspace / "model-mounts").exists()
    assert not prohibited.exists()


def test_restore_script_keeps_mounts_after_publication_when_output_is_malformed(
    tmp_path: Path,
) -> None:
    completed, paths, prohibited = _invoke_restore(
        tmp_path, record_mutations={"restore": "extra-output"}
    )
    workspace = paths["workspace"]
    sen2_target = workspace / "model-mounts" / "sen2srlite"
    ldsr_target = workspace / "model-mounts" / "ldsr-s2"

    assert completed.returncode == 2, completed.stderr
    assert completed.stdout == ""
    assert (workspace / "trustsr/phase2b1b").is_dir()
    assert (workspace / "trustsr/phase2b2a").is_dir()
    assert (workspace / "trustsr/phase2b3a").is_dir()
    assert _recorded_calls(paths["umount_calls"]) == []
    assert json.loads(paths["mount_state"].read_text(encoding="utf-8")) == {
        str(sen2_target): str(paths["sen2_source"]),
        str(ldsr_target): str(paths["ldsr_source"]),
    }
    assert sen2_target.is_dir()
    assert ldsr_target.is_dir()
    assert not prohibited.exists()
