"""Executable contracts for the Phase 2B1A cloud operator scripts."""

from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPTS = REPOSITORY / "scripts" / "phase2b1a"
BOOTSTRAP = SCRIPTS / "bootstrap_reader.sh"
RUNNER = SCRIPTS / "run_cloud.sh"
README = REPOSITORY / "README.md"
REQUIREMENTS = REPOSITORY / "requirements" / "cloud-taco-v1.txt"
RUNTIME_REQUIREMENTS = REPOSITORY / "requirements" / "cloud-phase2b1a-runtime.txt"
PINNED_REQUIREMENTS = (
    "tacoreader==0.4.5",
    "geopandas==1.1.4",
    "pyarrow==25.0.1",
    "shapely==2.1.2",
    "pyproj==3.7.2",
    "pyogrio==0.13.0",
)
PINNED_RUNTIME_REQUIREMENTS = (
    "numpy==2.5.2",
    "pandas==2.3.3",
    "scipy==1.18.1",
    "rasterio==1.5.1",
)
STORAGE_ROOT_PREFIXES = (
    "--st",
    "--sto",
    "--stor",
    "--stora",
    "--storag",
    "--storage",
    "--storage-",
    "--storage-r",
    "--storage-ro",
    "--storage-roo",
    "--storage-root",
)


def _make_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _environment(
    tmp_path: Path, *, mount_root: Path, available_kib: int = 30_000_000
) -> tuple[dict[str, str], Path]:
    """Build harmless command fakes used by the real shell scripts."""
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    command_log = tmp_path / "command-log"
    _make_executable(
        fake_bin / "mountpoint",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf 'mountpoint\\n' >> \"$COMMAND_LOG\"\n"
        "[[ \"$*\" == \"-q -- $FAKE_MOUNT_ROOT\" ]] || exit 99\n"
        "[[ \"${FAKE_MOUNT_AVAILABLE:-1}\" == 1 ]]\n",
    )
    _make_executable(
        fake_bin / "df",
        "#!/usr/bin/env bash\n"
        "printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'\n"
        "printf '/dev/fake 30000000 1 %s 1%% /persistent\\n' \"$FAKE_AVAILABLE_KIB\"\n",
    )
    _make_executable(
        fake_bin / "conda",
        "#!/usr/bin/env bash\n"
        "touch \"$CONDA_CALLED\"\n"
        "printf 'conda was called unexpectedly\\n' >&2\n"
        "exit 99\n",
    )
    environment = {
        **os.environ,
        "COMMAND_LOG": str(command_log),
        "CONDA_CALLED": str(tmp_path / "conda-called"),
        "FAKE_AVAILABLE_KIB": str(available_kib),
        "FAKE_MOUNT_ROOT": str(mount_root),
        "HOME": str(home),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    }
    return environment, command_log


def _fake_base_python(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    """Create a Python shim that records pip and CLI argv without installing packages."""
    modules = tmp_path / "fake-modules"
    modules.mkdir()
    (modules / "sitecustomize.py").write_text(
        "import importlib.metadata as metadata\n"
        "import json\n"
        "import os\n"
        "_real_version = metadata.version\n"
        "_fixed = {\n"
        "    'tacoreader': '0.4.5', 'geopandas': '1.1.4', 'pyarrow': '25.0.1',\n"
        "    'shapely': '2.1.2', 'pyproj': '3.7.2', 'pyogrio': '0.13.0',\n"
        "    'numpy': '2.5.2', 'pandas': '2.3.3', 'scipy': '1.18.1',\n"
        "    'rasterio': '1.5.1',\n"
        "}\n"
        "_overrides = json.loads(os.environ.get('FAKE_VERSION_OVERRIDES', '{}'))\n"
        "def _version(name):\n"
        "    if name in _overrides: return _overrides[name]\n"
        "    if name in _fixed: return _fixed[name]\n"
        "    return _real_version(name)\n"
        "metadata.version = _version\n"
        "import builtins\n"
        "_real_import = builtins.__import__\n"
        "def _import(name, *args, **kwargs):\n"
        "    if name == os.environ.get('FAKE_IMPORT_FAILURE'):\n"
        "        raise ImportError(f'blocked test import: {name}')\n"
        "    return _real_import(name, *args, **kwargs)\n"
        "builtins.__import__ = _import\n",
        encoding="utf-8",
    )
    (modules / "tacoreader.py").write_text(
        "import os\n"
        "if os.environ.get('FAKE_TACOREADER_CALLABLE', '1') == '1':\n"
        "    def load(*args, **kwargs): pass\n"
        "    def load_metadata(*args, **kwargs): pass\n"
        "else:\n"
        "    load = None\n"
        "    load_metadata = None\n",
        encoding="utf-8",
    )
    calls = tmp_path / "base-python-calls.jsonl"
    launcher = tmp_path / "base-python"
    launcher_source = f"""#!{sys.executable}
import json
import os
import subprocess
import sys
from pathlib import Path

arguments = sys.argv[1:]
with Path(os.environ['FAKE_PYTHON_CALLS']).open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(arguments) + '\\n')
if arguments == ['--version']:
    print(os.environ.get('FAKE_PYTHON_VERSION', 'Python 3.12.8'))
    raise SystemExit(0)
if arguments[:4] == ['-m', 'pip', 'install', '--dry-run']:
    report_path = Path(arguments[arguments.index('--report') + 1])
    protected = os.environ.get('FAKE_DRY_RUN_PACKAGE')
    installs = [] if protected is None else [{{'metadata': {{'name': protected}}}}]
    report_path.write_text(json.dumps({{'install': installs}}), encoding='utf-8')
    raise SystemExit(0)
if arguments[:3] == ['-m', 'pip', 'install'] or arguments == ['-m', 'pip', 'check']:
    raise SystemExit(0)
if arguments[:2] == ['-m', 'trustsr.cli.phase2b1a']:
    raise SystemExit(0)
if arguments and arguments[0] == '-':
    completed = subprocess.run(
        [{sys.executable!r}, *arguments],
        input=sys.stdin.read(),
        text=True,
        env=os.environ.copy(),
        check=False,
    )
    raise SystemExit(completed.returncode)
raise SystemExit(f'unexpected fake Python argv: {{arguments!r}}')
"""
    _make_executable(launcher, launcher_source)
    environment = {
        "FAKE_PYTHON_CALLS": str(calls),
        "PYTHONPATH": str(modules),
    }
    return launcher, environment, calls


def _invoke_internal(
    script: Path,
    function: str,
    base_python: Path,
    arguments: list[str],
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; shift; function="$1"; shift; "$function" "$@"',
            "phase2b1a-script-test",
            str(script),
            function,
            str(base_python),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def _calls(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _phase2b1a_readme_shell_block() -> str:
    section = README.read_text(encoding="utf-8").split(
        "## Phase 2B1A cloud crosssensor pilot", maxsplit=1
    )[1]
    fenced = section.split("```bash\n", maxsplit=1)[1]
    return fenced.split("\n```", maxsplit=1)[0]


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    (repository / "src" / "trustsr").mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
    (repository / "requirements").mkdir()
    (repository / "requirements" / "cloud-taco-v1.txt").write_text(
        "fixture\n", encoding="utf-8"
    )
    (repository / "requirements" / "cloud-phase2b1a-runtime.txt").write_text(
        "runtime-fixture\n", encoding="utf-8"
    )
    _write_plain_stage_module(repository / "src")
    return repository


def _write_plain_stage_module(source_root: Path) -> None:
    package = source_root / "trustsr" / "cli"
    package.mkdir(parents=True, exist_ok=True)
    (package.parent / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "phase2b1a.py").write_text("VALUE = 'fixture'\n", encoding="utf-8")


def _write_stage_module(source_root: Path, marker: Path, label: str) -> None:
    """Write a runnable stage module whose import origin is externally observable."""
    package = source_root / "trustsr" / "cli"
    package.mkdir(parents=True, exist_ok=True)
    (package.parent / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "phase2b1a.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        f"Path(os.environ['PHASE2B1A_MODULE_MARKER']).write_text({label!r}, encoding='utf-8')\n",
        encoding="utf-8",
    )


def _passthrough_base_python(tmp_path: Path) -> Path:
    """Run the passed module in a fresh real Python process without package installation."""
    launcher = tmp_path / "passthrough-base-python"
    _make_executable(
        launcher,
        f"#!{sys.executable}\n"
        "import os\n"
        "import sys\n"
        f"os.execvpe({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]], os.environ)\n",
    )
    return launcher


def test_cloud_requirements_are_an_exact_isolated_snapshot() -> None:
    """A version drift would change the reader environment used for extraction."""
    assert REQUIREMENTS.read_text(encoding="utf-8").splitlines() == list(PINNED_REQUIREMENTS)
    assert RUNTIME_REQUIREMENTS.read_text(encoding="utf-8").splitlines() == list(
        PINNED_RUNTIME_REQUIREMENTS
    )


def test_bootstrap_requires_python_312_before_any_pip_operation(tmp_path: Path) -> None:
    """Removing the interpreter gate could resolve the pinned snapshot on another ABI."""
    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    repository = _repository(tmp_path)
    base_python, python_environment, calls = _fake_base_python(tmp_path)
    environment, _ = _environment(tmp_path, mount_root=storage_root)
    environment.update(python_environment)
    environment["FAKE_PYTHON_VERSION"] = "Python 3.11.9"

    completed = _invoke_internal(
        BOOTSTRAP, "bootstrap_main", base_python, [str(storage_root), str(repository)], environment
    )

    assert completed.returncode != 0
    assert "python 3.12" in completed.stderr.lower()
    assert _calls(calls) == [["--version"]]


def test_shell_scripts_parse_before_their_implementation_exists() -> None:
    """A syntax error would make the operator entry point unusable before validation."""
    for script in (BOOTSTRAP, RUNNER):
        completed = subprocess.run(
            ["bash", "-n", str(script)], check=False, capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stderr


def test_readme_runs_distinct_pre_and_post_manifest_stage_contract(tmp_path: Path) -> None:
    """Reusing one manifest variable would audit the pre-extraction artifact."""
    checkout = tmp_path / "checkout"
    scripts = checkout / "scripts" / "phase2b1a"
    scripts.mkdir(parents=True)
    call_log = tmp_path / "readme-calls.jsonl"
    fake = f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path

arguments = sys.argv[1:]
with Path(os.environ['PHASE2B1A_CALL_LOG']).open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(arguments) + '\\n')
if len(arguments) == 2:
    raise SystemExit(0)
stage = arguments[2]
digests = {{
    'download': {{'source_sha256': 'c' * 64}},
    'manifest': {{'manifest_sha256': 'a' * 64}},
    'pilot': {{'input_manifest_sha256': 'a' * 64, 'manifest_sha256': 'b' * 64}},
    'audit': {{'audit_sha256': 'd' * 64, 'manifest_sha256': 'b' * 64}},
}}
print(json.dumps({{'stage': stage, 'digests': digests[stage], 'counts': {{}}, 'reused': False}}))
"""
    for name in ("bootstrap_reader.sh", "run_cloud.sh"):
        _make_executable(scripts / name, fake)
    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    block = _phase2b1a_readme_shell_block()
    assert "/opt/conda/bin/python" in block
    portable_block = block.replace("/opt/conda/bin/python", shlex.quote(sys.executable))
    environment = {
        **os.environ,
        "PHASE2B1A_CALL_LOG": str(call_log),
        "PHASE2B1A_STORAGE_ROOT": str(storage_root),
        "PHASE2B1A_TRANSPORT_URL": "https://transport.example.invalid/object.taco",
    }

    completed = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", portable_block],
        cwd=checkout,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    calls = _calls(call_log)
    assert [call[2] for call in calls[1:]] == ["download", "manifest", "pilot", "audit"]
    expected_source = checkout / "artifacts" / "datasets" / "sen2naipv2-source-v1.json"
    for call in calls[1:]:
        assert call[:2] == [str(storage_root), str(checkout)]
        assert "--confirm-cloud-storage" in call
        assert call[call.index("--source") + 1] == str(expected_source)
    assert calls[1][calls[1].index("--transport-url") + 1] == environment[
        "PHASE2B1A_TRANSPORT_URL"
    ]
    pre_manifest = (
        storage_root / "trustsr" / "phase2b1a" / "manifests" / ("a" * 64) / "samples.jsonl"
    )
    post_manifest = (
        storage_root / "trustsr" / "phase2b1a" / "manifests" / ("b" * 64) / "samples.jsonl"
    )
    assert calls[3][calls[3].index("--manifest") + 1] == str(pre_manifest)
    assert calls[4][calls[4].index("--manifest") + 1] == str(post_manifest)
    assert pre_manifest != post_manifest
    audit_path = (
        storage_root
        / "trustsr"
        / "phase2b1a"
        / "audits"
        / ("b" * 64)
        / "phase2b1a-audit.json"
    )
    assert str(audit_path) in completed.stdout


@pytest.mark.parametrize("script,function", ((BOOTSTRAP, "bootstrap_main"), (RUNNER, "run_main")))
def test_cloud_scripts_require_a_mounted_root_before_any_python_call(
    script: Path, function: str, tmp_path: Path
) -> None:
    """Removing the mount gate could install or write on an ephemeral disk."""
    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    repository = _repository(tmp_path)
    base_python, python_environment, calls = _fake_base_python(tmp_path)
    environment, _ = _environment(tmp_path, mount_root=storage_root)
    environment.update(python_environment)
    environment["FAKE_MOUNT_AVAILABLE"] = "0"
    arguments = [str(storage_root), str(repository)]
    if function == "run_main":
        arguments.extend(("manifest", "--confirm-cloud-storage", "--source", "source.json"))

    completed = _invoke_internal(script, function, base_python, arguments, environment)

    assert completed.returncode != 0
    assert "mount" in completed.stderr.lower()
    assert _calls(calls) == []


@pytest.mark.parametrize("invalid_root", ("", "/", "/root", "relative", "root*", "root\nnext"))
@pytest.mark.parametrize("script,function", ((BOOTSTRAP, "bootstrap_main"), (RUNNER, "run_main")))
def test_cloud_scripts_reject_unsafe_storage_roots_before_side_effects(
    invalid_root: str, script: Path, function: str, tmp_path: Path
) -> None:
    """Weak path validation could target a home, root, or shell-expanded location."""
    mount_root = tmp_path / "persistent"
    mount_root.mkdir()
    repository = _repository(tmp_path)
    base_python, python_environment, calls = _fake_base_python(tmp_path)
    environment, _ = _environment(tmp_path, mount_root=mount_root)
    environment.update(python_environment)
    arguments = [invalid_root, str(repository)]
    if function == "run_main":
        arguments.extend(("manifest", "--confirm-cloud-storage", "--source", "source.json"))

    completed = _invoke_internal(script, function, base_python, arguments, environment)

    assert completed.returncode != 0
    assert "storage root" in completed.stderr.lower()
    assert _calls(calls) == []


@pytest.mark.parametrize("script,function", ((BOOTSTRAP, "bootstrap_main"), (RUNNER, "run_main")))
def test_cloud_scripts_reject_storage_symlinks_and_the_current_home(
    script: Path, function: str, tmp_path: Path
) -> None:
    """Canonicalizing a symlink or home would defeat the explicit persistent-root contract."""
    mount_root = tmp_path / "persistent"
    mount_root.mkdir()
    linked_root = tmp_path / "linked-persistent"
    linked_root.symlink_to(mount_root, target_is_directory=True)
    repository = _repository(tmp_path)
    base_python, python_environment, calls = _fake_base_python(tmp_path)
    environment, _ = _environment(tmp_path, mount_root=mount_root)
    environment.update(python_environment)

    for invalid_root in (str(linked_root), str(tmp_path / "home")):
        if invalid_root == str(tmp_path / "home"):
            Path(invalid_root).mkdir(exist_ok=True)
            environment["FAKE_MOUNT_ROOT"] = invalid_root
        completed = _invoke_internal(
            script,
            function,
            base_python,
            [invalid_root, str(repository)]
            + (
                ["manifest", "--confirm-cloud-storage", "--source", "source.json"]
                if function == "run_main"
                else []
            ),
            environment,
        )
        assert completed.returncode != 0
        assert "storage root" in completed.stderr.lower()
    assert _calls(calls) == []


@pytest.mark.parametrize(
    "root_alias",
    ("/tmp/..", "//", "{home}/."),
)
@pytest.mark.parametrize("script,function", ((BOOTSTRAP, "bootstrap_main"), (RUNNER, "run_main")))
def test_cloud_scripts_reject_canonical_root_aliases_before_external_commands(
    root_alias: str, script: Path, function: str, tmp_path: Path
) -> None:
    """Removing post-realpath rejection could treat a lexical root alias as persistent storage."""
    mount_root = tmp_path / "persistent"
    mount_root.mkdir()
    repository = _repository(tmp_path)
    base_python, python_environment, calls = _fake_base_python(tmp_path)
    environment, command_log = _environment(tmp_path, mount_root=mount_root)
    environment.update(python_environment)
    invalid_root = root_alias.format(home=environment["HOME"])
    arguments = [invalid_root, str(repository)]
    if function == "run_main":
        arguments.extend(("manifest", "--confirm-cloud-storage", "--source", "source.json"))

    completed = _invoke_internal(script, function, base_python, arguments, environment)

    assert completed.returncode != 0
    assert "storage root" in completed.stderr.lower()
    assert not command_log.exists()
    assert _calls(calls) == []


@pytest.mark.parametrize("script,function", ((BOOTSTRAP, "bootstrap_main"), (RUNNER, "run_main")))
def test_cloud_scripts_require_strictly_more_than_fifteen_gib(
    script: Path, function: str, tmp_path: Path
) -> None:
    """Changing <= to < would allow the exact minimum despite the required safety margin."""
    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    repository = _repository(tmp_path)
    base_python, python_environment, calls = _fake_base_python(tmp_path)
    environment, _ = _environment(
        tmp_path, mount_root=storage_root, available_kib=15 * 1024 * 1024
    )
    environment.update(python_environment)
    arguments = [str(storage_root), str(repository)]
    if function == "run_main":
        arguments.extend(("manifest", "--confirm-cloud-storage", "--source", "source.json"))

    completed = _invoke_internal(script, function, base_python, arguments, environment)

    assert completed.returncode != 0
    assert "15 gib" in completed.stderr.lower()
    expected_calls = [["--version"]] if function == "bootstrap_main" else []
    assert _calls(calls) == expected_calls


@pytest.mark.parametrize("protected", ("torch", "torchvision", "triton", "NVIDIA.CUBLAS"))
def test_bootstrap_rejects_protected_packages_from_the_dry_run(
    protected: str, tmp_path: Path
) -> None:
    """Dropping normalized dry-run inspection could replace the cloud PyTorch/CUDA stack."""
    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    repository = _repository(tmp_path)
    base_python, python_environment, calls = _fake_base_python(tmp_path)
    environment, _ = _environment(tmp_path, mount_root=storage_root)
    environment.update(python_environment)
    environment["FAKE_DRY_RUN_PACKAGE"] = protected

    completed = _invoke_internal(
        BOOTSTRAP, "bootstrap_main", base_python, [str(storage_root), str(repository)], environment
    )

    assert completed.returncode != 0
    assert "protected" in completed.stderr.lower()
    observed = _calls(calls)
    assert len(observed) == 3
    assert observed[0] == ["--version"]
    assert observed[1][:4] == ["-m", "pip", "install", "--dry-run"]
    assert observed[1][-4:] == [
        "-r",
        str(repository / "requirements" / "cloud-taco-v1.txt"),
        "-r",
        str(repository / "requirements" / "cloud-phase2b1a-runtime.txt"),
    ]
    assert observed[2][0] == "-"
    assert not Path(environment["CONDA_CALLED"]).exists()


@pytest.mark.parametrize("malformed", ("", " torch ", "..."))
def test_bootstrap_rejects_malformed_dry_run_distribution_names_before_install(
    malformed: str, tmp_path: Path
) -> None:
    """An unparseable report name cannot establish that protected packages are untouched."""
    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    repository = _repository(tmp_path)
    base_python, python_environment, calls = _fake_base_python(tmp_path)
    environment, _ = _environment(tmp_path, mount_root=storage_root)
    environment.update(python_environment)
    environment["FAKE_DRY_RUN_PACKAGE"] = malformed

    completed = _invoke_internal(
        BOOTSTRAP, "bootstrap_main", base_python, [str(storage_root), str(repository)], environment
    )

    assert completed.returncode != 0
    assert "invalid package name" in completed.stderr.lower()
    observed = _calls(calls)
    assert observed[0] == ["--version"]
    assert len(observed) == 3


def test_bootstrap_installs_only_after_a_clean_dry_run_and_verifies_the_reader(
    tmp_path: Path,
) -> None:
    """Catch an install-before-dry-run order or omitted reader-contract check."""
    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    repository = _repository(tmp_path)
    base_python, python_environment, calls = _fake_base_python(tmp_path)
    environment, _ = _environment(tmp_path, mount_root=storage_root)
    environment.update(python_environment)

    completed = _invoke_internal(
        BOOTSTRAP, "bootstrap_main", base_python, [str(storage_root), str(repository)], environment
    )

    assert completed.returncode == 0, completed.stderr
    observed = _calls(calls)
    assert observed[0] == ["--version"]
    assert observed[1][:4] == ["-m", "pip", "install", "--dry-run"]
    assert "--report" in observed[1]
    assert observed[1][-4:] == [
        "-r",
        str(repository / "requirements" / "cloud-taco-v1.txt"),
        "-r",
        str(repository / "requirements" / "cloud-phase2b1a-runtime.txt"),
    ]
    assert observed[2][0] == "-"
    assert observed[3] == [
        "-m",
        "pip",
        "install",
        "--upgrade-strategy",
        "only-if-needed",
        "-r",
        str(repository / "requirements" / "cloud-taco-v1.txt"),
        "-r",
        str(repository / "requirements" / "cloud-phase2b1a-runtime.txt"),
    ]
    assert observed[4] == ["-m", "pip", "check"]
    assert observed[5] == ["-", str(repository)]
    assert not Path(environment["CONDA_CALLED"]).exists()


@pytest.mark.parametrize(
    ("environment_name", "value"),
    (
        ("FAKE_VERSION_OVERRIDES", '{"pyogrio": "0.13.1"}'),
        ("FAKE_VERSION_OVERRIDES", '{"rasterio": "1.5.2"}'),
        ("FAKE_TACOREADER_CALLABLE", "0"),
        ("FAKE_IMPORT_FAILURE", "scipy"),
    ),
)
def test_bootstrap_fails_when_an_exact_runtime_contract_is_not_met(
    environment_name: str, value: str, tmp_path: Path
) -> None:
    """Missing exact-version or callable checks would admit an incompatible legacy reader."""
    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    repository = _repository(tmp_path)
    base_python, python_environment, _ = _fake_base_python(tmp_path)
    environment, _ = _environment(tmp_path, mount_root=storage_root)
    environment.update(python_environment)
    environment[environment_name] = value

    completed = _invoke_internal(
        BOOTSTRAP, "bootstrap_main", base_python, [str(storage_root), str(repository)], environment
    )

    assert completed.returncode != 0
    assert "must" in completed.stderr.lower() or "callable" in completed.stderr.lower()


def test_bootstrap_rejects_a_stale_global_cli_when_the_checkout_module_is_missing(
    tmp_path: Path,
) -> None:
    """Dropping the origin proof could bless a stale globally installed stage controller."""
    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    repository = _repository(tmp_path)
    (repository / "src" / "trustsr" / "cli" / "phase2b1a.py").unlink()
    stale_source = tmp_path / "stale-source"
    _write_plain_stage_module(stale_source)
    base_python, python_environment, _ = _fake_base_python(tmp_path)
    environment, _ = _environment(tmp_path, mount_root=storage_root)
    environment.update(python_environment)
    environment["PYTHONPATH"] = os.pathsep.join(
        (python_environment["PYTHONPATH"], str(stale_source))
    )

    completed = _invoke_internal(
        BOOTSTRAP, "bootstrap_main", base_python, [str(storage_root), str(repository)], environment
    )

    assert completed.returncode != 0
    assert "checkout" in completed.stderr.lower()


def test_runner_requires_confirmation_before_it_invokes_the_cli(tmp_path: Path) -> None:
    """Removing confirmation would make a cloud-data stage too easy to start by mistake."""
    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    repository = _repository(tmp_path)
    base_python, python_environment, calls = _fake_base_python(tmp_path)
    environment, _ = _environment(tmp_path, mount_root=storage_root)
    environment.update(python_environment)

    completed = _invoke_internal(
        RUNNER,
        "run_main",
        base_python,
        [str(storage_root), str(repository), "manifest", "--source", "source.json"],
        environment,
    )

    assert completed.returncode != 0
    assert "confirm" in completed.stderr.lower()
    assert _calls(calls) == []


def test_runner_refuses_a_forwarded_storage_root_override(tmp_path: Path) -> None:
    """A second storage-root option would let argparse bypass the mount-validated root."""
    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    repository = _repository(tmp_path)
    base_python, python_environment, calls = _fake_base_python(tmp_path)
    environment, _ = _environment(tmp_path, mount_root=storage_root)
    environment.update(python_environment)

    completed = _invoke_internal(
        RUNNER,
        "run_main",
        base_python,
        [
            str(storage_root),
            str(repository),
            "manifest",
            "--confirm-cloud-storage",
            "--storage-root",
            str(tmp_path / "unvalidated"),
            "--source",
            "source.json",
        ],
        environment,
    )

    assert completed.returncode != 0
    assert "override" in completed.stderr.lower()
    assert _calls(calls) == []


@pytest.mark.parametrize(
    ("prefix", "uses_equals"),
    tuple((prefix, False) for prefix in STORAGE_ROOT_PREFIXES)
    + tuple((prefix, True) for prefix in STORAGE_ROOT_PREFIXES),
)
def test_runner_refuses_every_argparse_storage_root_abbreviation(
    prefix: str, uses_equals: bool, tmp_path: Path
) -> None:
    """An argparse prefix must not replace the wrapper's mount-validated root."""
    from trustsr.cli.phase2b1a import build_parser

    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    unvalidated = tmp_path / "unvalidated"
    override = (
        (f"{prefix}={unvalidated}",)
        if uses_equals
        else (prefix, str(unvalidated))
    )
    parser_arguments = [
        "manifest",
        "--source",
        "source.json",
        "--storage-root",
        str(storage_root),
        "--confirm-cloud-storage",
        *override,
    ]
    assert build_parser().parse_args(parser_arguments).storage_root == unvalidated

    repository = _repository(tmp_path)
    base_python, python_environment, calls = _fake_base_python(tmp_path)
    environment, _ = _environment(tmp_path, mount_root=storage_root)
    environment.update(python_environment)
    completed = _invoke_internal(
        RUNNER,
        "run_main",
        base_python,
        [
            str(storage_root),
            str(repository),
            "manifest",
            "--confirm-cloud-storage",
            "--source",
            "source.json",
            *override,
        ],
        environment,
    )

    assert completed.returncode != 0
    assert "override" in completed.stderr.lower()
    assert _calls(calls) == []


def test_runner_executes_the_phase2b1a_module_from_the_supplied_checkout(
    tmp_path: Path,
) -> None:
    """A stale globally installed module must not run instead of the reviewed checkout."""
    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    repository = _repository(tmp_path)
    marker = tmp_path / "module-origin"
    _write_stage_module(repository / "src", marker, "supplied")
    stale_source = tmp_path / "stale-source"
    _write_stage_module(stale_source, marker, "stale")
    environment, _ = _environment(tmp_path, mount_root=storage_root)
    environment["PHASE2B1A_MODULE_MARKER"] = str(marker)
    environment["PYTHONPATH"] = str(stale_source)

    completed = _invoke_internal(
        RUNNER,
        "run_main",
        _passthrough_base_python(tmp_path),
        [
            str(storage_root),
            str(repository),
            "manifest",
            "--confirm-cloud-storage",
            "--source",
            "source.json",
        ],
        environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert marker.read_text(encoding="utf-8") == "supplied"


def test_runner_rejects_a_colon_bearing_checkout_before_base_python_exec(
    tmp_path: Path,
) -> None:
    """A colon cannot represent one unambiguous POSIX PYTHONPATH entry."""
    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    repository = _repository(tmp_path)
    colon_repository = tmp_path / "repository:colon"
    repository.rename(colon_repository)
    marker = tmp_path / "module-origin"
    _write_stage_module(colon_repository / "src", marker, "supplied")
    stale_source = tmp_path / "stale-source"
    _write_stage_module(stale_source, marker, "stale")
    environment, _ = _environment(tmp_path, mount_root=storage_root)
    environment["PHASE2B1A_MODULE_MARKER"] = str(marker)
    environment["PYTHONPATH"] = str(stale_source)

    completed = _invoke_internal(
        RUNNER,
        "run_main",
        _passthrough_base_python(tmp_path),
        [
            str(storage_root),
            str(colon_repository),
            "manifest",
            "--confirm-cloud-storage",
            "--source",
            "source.json",
        ],
        environment,
    )

    assert completed.returncode != 0
    assert "repository directory" in completed.stderr.lower()
    assert not marker.exists()


def test_runner_passes_distinct_safe_argv_to_the_base_python_cli(tmp_path: Path) -> None:
    """Shell interpolation or a console launcher could change the requested cloud operation."""
    storage_root = tmp_path / "persistent"
    storage_root.mkdir()
    repository = _repository(tmp_path)
    base_python, python_environment, calls = _fake_base_python(tmp_path)
    environment, _ = _environment(tmp_path, mount_root=storage_root)
    environment.update(python_environment)
    sentinel = tmp_path / "must-not-exist"
    transport_url = f"https://transport.example.invalid/file;touch {sentinel}"

    completed = _invoke_internal(
        RUNNER,
        "run_main",
        base_python,
        [
            str(storage_root),
            str(repository),
            "download",
            "--confirm-cloud-storage",
            "--source",
            "source.json",
            "--transport-url",
            transport_url,
        ],
        environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert _calls(calls) == [
        [
            "-m",
            "trustsr.cli.phase2b1a",
            "download",
            "--storage-root",
            str(storage_root),
            "--confirm-cloud-storage",
            "--source",
            "source.json",
            "--transport-url",
            transport_url,
        ]
    ]
    assert not sentinel.exists()
