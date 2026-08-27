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

if [[ $# -ne 2 ]]; then
  die 'argument count; usage: bootstrap_remote.sh REMOTE_ROOT REPO_DIR'
fi

remote_root="$1"
repo_dir="$2"
require_remote_root "$remote_root"
validate_path "$repo_dir" || die 'repository directory'
[[ -d "$repo_dir" && ! -L "$repo_dir" && -f "$repo_dir/uv.lock" && ! -L "$repo_dir/uv.lock" ]] || die 'repository directory must contain a regular uv.lock'

available_kib="$(df -Pk -- "$remote_root" | awk 'NR == 2 {print $4}')"
[[ "$available_kib" =~ ^[0-9]+$ ]] || die 'could not determine free disk space'
if (( available_kib < 15 * 1024 * 1024 )); then
  die 'at least 15 GiB of free disk space is required'
fi

prefix="${remote_root}/conda-env"
lock_digest="$(sha256sum -- "$repo_dir/uv.lock" | awk '{print $1}')"
[[ "$lock_digest" =~ ^[0-9a-f]{64}$ ]] || die 'could not hash uv.lock'
stamp="${prefix}/.trustsr-uv-lock.sha256"

if [[ -e "$prefix" || -L "$prefix" ]]; then
  if [[ ! -d "$prefix" || -L "$prefix" || ! -x "$prefix/bin/python" || ! -x "$prefix/bin/uv" || ! -f "$stamp" || -L "$stamp" ]]; then
    printf 'existing prefix is incompatible; remove it manually and recreate it with this script\n' >&2
    exit 1
  fi
  if [[ "$("$prefix/bin/python" --version 2>&1)" != Python\ 3.12.* ]] || [[ "$("$prefix/bin/uv" --version 2>&1)" != 'uv 0.12.5' ]] || ! printf '%s\n' "$lock_digest" | cmp -s - "$stamp"; then
    printf 'existing prefix is incompatible; remove it manually and recreate it with this script\n' >&2
    exit 1
  fi
  exit 0
fi

conda create --yes --override-channels --channel conda-forge --prefix "$prefix" python=3.12 pip
"$prefix/bin/python" -m pip install 'uv==0.12.5'
UV_PROJECT_ENVIRONMENT="$prefix" "$prefix/bin/uv" sync --directory "$repo_dir" --frozen --no-dev --extra gpu
printf '%s\n' "$lock_digest" > "$stamp"
