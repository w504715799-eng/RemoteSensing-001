#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'invalid bootstrap input: %s\n' "$1" >&2
  exit 2
}

validate_path() {
  local value="$1"
  [[ -n "$value" && "$value" != / && "$value" != /root ]] || return 1
  [[ "$value" != ~* && "$value" != *$'\n'* && "$value" != *$'\r'* ]] || return 1
  [[ "$value" != *[\*\?\[]* ]] || return 1
}

require_remote_root() {
  local value="$1"
  local resolved
  validate_path "$value" || die 'remote root'
  resolved="$(realpath -e -- "$value")" || die 'remote root'
  [[ "$resolved" == /root/rivermind-fs/* ]] || die 'remote root must resolve under /root/rivermind-fs/'
  [[ -d "$value" && ! -L "$value" ]] || die 'remote root must be an existing directory'
}

require_base_runtime() {
  [[ -x "$base_python" ]] || die "required cloud-image interpreter is unavailable: $base_python"
  [[ "$("$base_python" --version 2>&1)" == Python\ 3.12.* ]] || die 'the cloud-image interpreter must report Python 3.12'
  "$base_python" - <<'PY'
import json
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit("the cloud-image interpreter must be Python 3.12")

import torch
import torchvision

if not torch.cuda.is_available():
    raise SystemExit("the cloud-image PyTorch runtime has no available CUDA device")
if torch.version.cuda is None:
    raise SystemExit("the cloud-image PyTorch runtime has no CUDA version")

print(
    json.dumps(
        {
            "cuda_runtime": torch.version.cuda,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
)
PY
}

reject_cuda_stack_changes() {
  "$base_python" - "$1" <<'PY'
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

verify_installed_requirements() {
  "$base_python" - <<'PY'
import importlib.metadata as metadata
import json
import torch
import torchvision

if torch.cuda.is_available() is not True or torch.version.cuda is None:
    raise SystemExit("CUDA is no longer available after dependency installation")
if metadata.version("opensr-model") != "1.1.1":
    raise SystemExit("opensr-model must be version 1.1.1")
if metadata.version("uv") != "0.12.5":
    raise SystemExit("uv must be version 0.12.5")
import trustsr

print(
    json.dumps(
        {
            "cuda_runtime": torch.version.cuda,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
)
PY
}

if [[ $# -ne 2 ]]; then
  die 'argument count; usage: bootstrap_remote.sh REMOTE_ROOT REPO_DIR'
fi

remote_root="$1"
repo_dir="$2"
base_python=/opt/conda/bin/python
require_remote_root "$remote_root"
validate_path "$repo_dir" || die 'repository directory'
[[ -d "$repo_dir" && ! -L "$repo_dir" && -f "$repo_dir/uv.lock" && ! -L "$repo_dir/uv.lock" ]] || die 'repository directory must contain a regular uv.lock'

available_kib="$(df -Pk -- "$remote_root" | awk 'NR == 2 {print $4}')"
[[ "$available_kib" =~ ^[0-9]+$ ]] || die 'could not determine free disk space'
if (( available_kib < 15 * 1024 * 1024 )); then
  die 'at least 15 GiB of free disk space is required'
fi

pre_fingerprint="$(require_base_runtime)" || die 'cloud-image CUDA runtime validation failed'
lock_digest="$(sha256sum -- "$repo_dir/uv.lock" | awk '{print $1}')"
[[ "$lock_digest" =~ ^[0-9a-f]{64}$ ]] || die 'could not hash uv.lock'

report="$(mktemp "${remote_root}/.trustsr-pip-dry-run.XXXXXX")"
stamp_tmp=''
cleanup() {
  rm -f -- "$report" "$stamp_tmp"
}
trap cleanup EXIT

"$base_python" -m pip install --dry-run --report "$report" --upgrade-strategy only-if-needed -e "${repo_dir}[gpu]"
reject_cuda_stack_changes "$report"

"$base_python" -m pip install --upgrade-strategy only-if-needed "uv==0.12.5"
"$base_python" -m pip install --upgrade-strategy only-if-needed -e "${repo_dir}[gpu]"
post_fingerprint="$(verify_installed_requirements)" || die 'post-install CUDA runtime validation failed'
[[ "$pre_fingerprint" == "$post_fingerprint" ]] || die 'PyTorch/CUDA runtime fingerprint changed during bootstrap'
"$base_python" -m pip check

stamp="${remote_root}/.trustsr-bootstrap-provenance.json"
stamp_tmp="$(mktemp "${remote_root}/.trustsr-bootstrap-provenance.XXXXXX")"
"$base_python" - "$stamp_tmp" "$lock_digest" "$post_fingerprint" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "base_fingerprint": json.loads(sys.argv[3]),
            "schema_version": 1,
            "uv_lock_sha256": sys.argv[2],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
mv -f -- "$stamp_tmp" "$stamp"
stamp_tmp=''
