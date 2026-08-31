#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'invalid cloud run input: %s\n' "$1" >&2
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
  local current_home
  validate_raw_path "$value" || die 'storage root'
  reject_symlink_components "$value" || die 'storage root must not contain a symlink'
  [[ -d "$value" && ! -L "$value" ]] || die 'storage root must be an existing directory'
  resolved="$(realpath -e -- "$value")" || die 'storage root'
  current_home="$(realpath -e -- "${HOME:?HOME must be set}")" ||
    die 'current home cannot be resolved'
  [[ "$resolved" != / && "$resolved" != /root && "$resolved" != "$current_home" ]] ||
    die 'prohibited storage root'
  mountpoint -q -- "$resolved" || die "persistent mountpoint is unavailable: $resolved"
  printf '%s\n' "$resolved"
}

require_repository() {
  local value="$1"
  local resolved
  validate_raw_path "$value" || die 'repository directory'
  [[ "$value" != *:* ]] || die 'repository directory must not contain a colon'
  reject_symlink_components "$value" || die 'repository directory must not contain a symlink'
  [[ -d "$value" && ! -L "$value" && -f "$value/pyproject.toml" &&
    -d "$value/src/trustsr" ]] || die 'repository directory must be a checked-out project'
  resolved="$(realpath -e -- "$value")" || die 'repository directory'
  [[ "$resolved" != *:* ]] || die 'repository directory must not contain a colon'
  printf '%s\n' "$resolved"
}

run_main() {
  local base_python="$1"
  local storage_root
  local repo_dir
  local argument
  local confirmed=false
  local available_kib
  local available_inodes
  local log_dir
  local log_path
  local output
  shift
  (( $# >= 3 )) ||
    die 'argument count; usage: run_cloud.sh STORAGE_ROOT REPO_DIR STAGE_ARGS'
  [[ -x "$base_python" ]] ||
    die "required cloud-image interpreter is unavailable: $base_python"

  storage_root="$(require_storage_root "$1")"
  repo_dir="$(require_repository "$2")"
  shift 2
  for argument in "$@"; do
    case "$argument" in
      --confirm-cloud-storage) confirmed=true ;;
      --st*) die 'stage arguments must not override storage root' ;;
    esac
  done
  [[ "$confirmed" == true ]] ||
    die 'stage arguments must include --confirm-cloud-storage'

  available_kib="$(df -Pk -- "$storage_root" | awk 'NR == 2 {print $4}')"
  [[ "$available_kib" =~ ^[0-9]+$ ]] || die 'could not determine free disk space'
  (( available_kib > 1024 * 1024 )) || die 'more than 1 GiB of free disk space is required'
  available_inodes="$(df -Pi -- "$storage_root" | awk 'NR == 2 {print $4}')"
  [[ "$available_inodes" =~ ^[0-9]+$ ]] || die 'could not determine free inodes'
  (( available_inodes > 1024 )) || die 'more than 1024 free inodes are required'

  log_dir="$storage_root/trustsr/phase2b2a/logs"
  reject_symlink_components "$log_dir" || die 'log directory must not contain a symlink'
  mkdir -p -- "$log_dir"
  [[ -d "$log_dir" && ! -L "$log_dir" ]] || die 'log directory is invalid'
  log_path="$log_dir/audit-inputs.jsonl"
  if [[ -e "$log_path" || -L "$log_path" ]]; then
    [[ -f "$log_path" && ! -L "$log_path" ]] || die 'stage log is invalid'
  fi

  cd -- "$repo_dir"
  output="$(
    PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}" \
      "$base_python" -m trustsr.cli.phase2b2a audit-inputs \
        --storage-root "$storage_root" "$@"
  )"
  printf '%s\n' "$output" | tee -a -- "$log_path"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  run_main /opt/conda/bin/python "$@"
fi
