#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'invalid Phase 2B3-A live reset input: %s\n' "$1" >&2
  exit 2
}

validate_absolute_path() {
  local value="$1"
  local component
  local -a components
  [[ -n "$value" && "$value" == /* && "$value" != / && "$value" != /root ]] || return 1
  [[ "$value" != *$'\n'* && "$value" != *$'\r'* && "$value" != *$'\t'* ]] || return 1
  [[ "$value" != *' '* && "$value" != *:* && "$value" != *[\*\?\[]* ]] || return 1
  case "$value" in
    *'//'*) return 1 ;;
  esac
  IFS=/ read -r -a components <<< "${value#/}"
  (( ${#components[@]} >= 2 )) || return 1
  for component in "${components[@]}"; do
    [[ -n "$component" && "$component" != . && "$component" != .. && "$component" != -* ]] ||
      return 1
  done
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

require_canonical_directory() {
  local value="$1"
  local label="$2"
  local resolved
  validate_absolute_path "$value" || die "$label path"
  reject_symlink_components "$value" || die "$label path contains a symlink"
  [[ -d "$value" && ! -L "$value" ]] || die "$label path is not a directory"
  resolved="$(realpath -e -- "$value")" || die "$label path cannot be resolved"
  [[ "$resolved" == "$value" ]] || die "$label path is not normalized"
  printf '%s\n' "$resolved"
}

reject_unsafe_entries() {
  local target="$1"
  local unsafe_entry mount_output mount_target
  unsafe_entry="$(
    find -P "$target" -xdev -mindepth 1 \
      \( -type l -o \( ! -type d ! -type f \) \) -print -quit
  )" || die 'live target cannot be inspected'
  [[ -z "$unsafe_entry" ]] || die 'live target contains an unsafe entry'

  mount_output="$(findmnt -R -n -o TARGET --target "$target")" ||
    die 'live target mount identity cannot be inspected'
  while IFS= read -r mount_target; do
    [[ -n "$mount_target" ]] || continue
    case "$mount_target" in
      "$target" | "$target"/*) die 'live target contains a mountpoint' ;;
    esac
  done <<< "$mount_output"
}

reject_active_locks() {
  local target="$1"
  local logs="$target/logs"
  local active_lock
  local -a output_locks=()

  if [[ -e "$logs" || -L "$logs" ]]; then
    [[ -d "$logs" && ! -L "$logs" ]] || die 'stage log directory is unsafe'
    active_lock="$(
      find -P "$logs" -maxdepth 1 -name '*.jsonl.lock' ! -type d -print -quit
    )" || die 'stage log directory cannot be inspected'
    [[ -z "$active_lock" ]] || die 'stage log reservation has an unsafe type'
    active_lock="$(
      find -P "$logs" -maxdepth 1 -type d -name '*.jsonl.lock' -print -quit
    )" || die 'stage log directory cannot be inspected'
    [[ -z "$active_lock" ]] || die 'a stage log is reserved by another process'
  fi

  mapfile -d '' -t output_locks < <(
    find -P "$target" -xdev -type f -name '.*.lock' -print0
  )
  active_lock="$(
    find -P "$target" -xdev -name '.*.lock' ! -type f -print -quit
  )" || die 'deterministic output locks cannot be inspected'
  [[ -z "$active_lock" ]] || die 'deterministic output lock has an unsafe type'
  for active_lock in "${output_locks[@]}"; do
    flock -n "$active_lock" true || die 'a deterministic output is reserved by another process'
  done
}

run_main() {
  (( $# == 1 )) ||
    die 'argument count; usage: reset_live_phase2b3a.sh WORKSPACE_ROOT'

  local workspace_root trustsr_root target workspace_name
  workspace_root="$(require_canonical_directory "$1" 'workspace root')"
  workspace_name="${workspace_root##*/}"
  case "$workspace_name" in
    *phase2b3a-checkpoints*)
      die 'workspace root resembles a durable checkpoint root'
      ;;
  esac
  [[ ! -e "$workspace_root/trustsr-phase2b3a-checkpoints" &&
    ! -L "$workspace_root/trustsr-phase2b3a-checkpoints" ]] ||
    die 'workspace root contains a durable checkpoint store'

  trustsr_root="$(require_canonical_directory "$workspace_root/trustsr" 'live trustsr root')"
  [[ "$trustsr_root" == "$workspace_root/trustsr" ]] || die 'unexpected live trustsr root'
  target="$(require_canonical_directory "$trustsr_root/phase2b3a" 'live target')"
  [[ "$target" == "$workspace_root/trustsr/phase2b3a" ]] || die 'unexpected live target'

  reject_unsafe_entries "$target"
  reject_active_locks "$target"

  rm -rf --one-file-system -- "$target" || die 'live target could not be removed'
  [[ ! -e "$target" && ! -L "$target" ]] || die 'live target was not fully removed'
  mkdir -m 0700 -- "$target" || die 'live target could not be recreated'
  [[ -d "$target" && ! -L "$target" ]] || die 'recreated live target is invalid'
  [[ -z "$(find -P "$target" -mindepth 1 -print -quit)" ]] ||
    die 'recreated live target is not empty'

  printf '{"status":"reset","target":"trustsr/phase2b3a"}\n'
}

run_main "$@"
