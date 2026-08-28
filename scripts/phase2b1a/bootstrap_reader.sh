#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'invalid bootstrap input: %s\n' "$1" >&2
  exit 2
}

validate_raw_path() {
  local value="$1"
  [[ -n "$value" && "$value" == /* && "$value" != / && "$value" != /root ]] || return 1
  [[ "$value" != ~* && "$value" != *$'\n'* && "$value" != *$'\r'* ]] || return 1
  [[ "$value" != *[\*\?\[]* ]] || return 1
}

reject_symlink_components() {
  local value="$1"
  local current=/
  local component
  local -a components
  IFS=/ read -r -a components <<< "${value#/}"
  for component in "${components[@]}"; do
    [[ -n "$component" ]] || continue
    current="${current%/}/${component}"
    [[ ! -L "$current" ]] || return 1
  done
}

require_storage_root() {
  local value="$1"
  local resolved
  local home
  validate_raw_path "$value" || die 'storage root'
  reject_symlink_components "$value" || die 'storage root must not contain a symlink'
  [[ -d "$value" && ! -L "$value" ]] || die 'storage root must be an existing directory'
  resolved="$(realpath -e -- "$value")" || die 'storage root'
  home="$(realpath -e -- "${HOME:?HOME must be set}")" || die 'current home cannot be resolved'
  [[ "$resolved" != / && "$resolved" != /root && "$resolved" != "$home" ]] ||
    die 'prohibited storage root'
  mountpoint -q -- "$resolved" || die "persistent mountpoint is unavailable: $resolved"
  printf '%s\n' "$resolved"
}

require_repository() {
  local value="$1"
  local resolved
  validate_raw_path "$value" || die 'repository directory'
  reject_symlink_components "$value" || die 'repository directory must not contain a symlink'
  [[ -d "$value" && ! -L "$value" ]] || die 'repository directory must be an existing directory'
  resolved="$(realpath -e -- "$value")" || die 'repository directory'
  [[ -f "$resolved/requirements/cloud-taco-v1.txt" &&
    ! -L "$resolved/requirements/cloud-taco-v1.txt" ]] ||
    die 'repository directory must contain requirements/cloud-taco-v1.txt'
  [[ -f "$resolved/requirements/cloud-phase2b1a-runtime.txt" &&
    ! -L "$resolved/requirements/cloud-phase2b1a-runtime.txt" ]] ||
    die 'repository directory must contain requirements/cloud-phase2b1a-runtime.txt'
  printf '%s\n' "$resolved"
}

reject_protected_changes() {
  "$1" - "$2" <<'PY'
import json
import re
import sys

report_path = sys.argv[1]
try:
    with open(report_path, encoding="utf-8") as report_file:
        report = json.load(report_file)
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"could not parse pip dry-run report: {error}")

installs = report.get("install")
if not isinstance(installs, list):
    raise SystemExit("pip dry-run report has no install list")

def normalize(name: object) -> str:
    if not isinstance(name, str):
        raise SystemExit("pip dry-run report has an invalid package name")
    if name != name.strip() or not re.fullmatch(
        r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", name
    ):
        raise SystemExit("pip dry-run report has an invalid package name")
    return re.sub(r"[-_.]+", "-", name).lower()

blocked = {"torch", "torchvision", "triton"}
for item in installs:
    if not isinstance(item, dict) or not isinstance(item.get("metadata"), dict):
        raise SystemExit("pip dry-run report has an invalid install entry")
    name = normalize(item["metadata"].get("name"))
    if name in blocked or name.startswith("nvidia-"):
        raise SystemExit(f"pip dry-run would modify protected CUDA package: {name}")
PY
}

verify_runtime_contract() {
  "$1" - "$2" <<'PY'
import importlib
from importlib import metadata
from pathlib import Path
import sys

expected = {
    "tacoreader": "0.4.5",
    "geopandas": "1.1.4",
    "pyarrow": "25.0.1",
    "shapely": "2.1.2",
    "pyproj": "3.7.2",
    "pyogrio": "0.13.0",
    "numpy": "2.5.2",
    "pandas": "2.3.3",
    "scipy": "1.18.1",
    "rasterio": "1.5.1",
}
for distribution, required in expected.items():
    try:
        observed = metadata.version(distribution)
    except metadata.PackageNotFoundError as error:
        raise SystemExit(f"{distribution} must be installed at {required}") from error
    if observed != required:
        raise SystemExit(f"{distribution} must be version {required}, observed {observed}")

import tacoreader

for name in ("load", "load_metadata"):
    if not callable(getattr(tacoreader, name, None)):
        raise SystemExit(f"tacoreader.{name} must be callable")

for name in ("numpy", "pandas", "scipy", "rasterio"):
    try:
        importlib.import_module(name)
    except ImportError as error:
        raise SystemExit(f"{name} must import in the Phase 2B1A base runtime: {error}") from error

checkout = Path(sys.argv[1]).resolve(strict=True)
source_root = (checkout / "src").resolve(strict=True)
try:
    source_root.relative_to(checkout)
except ValueError:
    raise SystemExit("supplied checkout src must be confined under the exact checkout") from None
sys.path.insert(0, str(source_root))
try:
    phase2b1a = importlib.import_module("trustsr.cli.phase2b1a")
except ImportError as error:
    raise SystemExit(
        f"trustsr.cli.phase2b1a must import from the supplied checkout: {error}"
    ) from error
module_file = getattr(phase2b1a, "__file__", None)
if not isinstance(module_file, str):
    raise SystemExit("trustsr.cli.phase2b1a from the supplied checkout must have __file__")
try:
    Path(module_file).resolve(strict=True).relative_to(source_root)
except (OSError, ValueError):
    raise SystemExit(
        "trustsr.cli.phase2b1a must be confined under the exact supplied checkout"
    ) from None
PY
}

bootstrap_main() {
  local base_python="$1"
  local storage_root
  local repo_dir
  local reader_requirements
  local runtime_requirements
  local python_version
  local report
  shift
  [[ $# -eq 2 ]] || die 'argument count; usage: bootstrap_reader.sh STORAGE_ROOT REPO_DIR'
  [[ -x "$base_python" ]] || die "required cloud-image interpreter is unavailable: $base_python"

  storage_root="$(require_storage_root "$1")"
  repo_dir="$(require_repository "$2")"
  reader_requirements="$repo_dir/requirements/cloud-taco-v1.txt"
  runtime_requirements="$repo_dir/requirements/cloud-phase2b1a-runtime.txt"

  python_version="$("$base_python" --version 2>&1)" || die 'could not inspect base Python version'
  [[ "$python_version" =~ ^Python\ 3\.12\.[0-9]+([A-Za-z0-9.+-]*)?$ ]] ||
    die 'base Python must be exactly Python 3.12'

  local available_kib
  available_kib="$(df -Pk -- "$storage_root" | awk 'NR == 2 {print $4}')"
  [[ "$available_kib" =~ ^[0-9]+$ ]] || die 'could not determine free disk space'
  (( available_kib > 15 * 1024 * 1024 )) || die 'more than 15 GiB of free disk space is required'

  report="$(mktemp "${storage_root}/.trustsr-pip-dry-run.XXXXXX")"
  cleanup() {
    rm -f -- "$report"
  }
  trap cleanup EXIT

  "$base_python" -m pip install --dry-run --report "$report" \
    --upgrade-strategy only-if-needed \
    -r "$reader_requirements" -r "$runtime_requirements"
  reject_protected_changes "$base_python" "$report"
  "$base_python" -m pip install --upgrade-strategy only-if-needed \
    -r "$reader_requirements" -r "$runtime_requirements"
  "$base_python" -m pip check
  verify_runtime_contract "$base_python" "$repo_dir"
  rm -f -- "$report"
  trap - EXIT
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  bootstrap_main /opt/conda/bin/python "$@"
fi
