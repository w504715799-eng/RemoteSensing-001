#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'invalid Phase 2B3-A workspace checkpoint input: %s\n' "$1" >&2
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
  for component in "${components[@]}"; do
    [[ -n "$component" && "$component" != . && "$component" != .. && "$component" != -* ]] || return 1
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

require_existing_path() {
  local value="$1"
  local kind="$2"
  local resolved
  validate_absolute_path "$value" || die "$kind path"
  reject_symlink_components "$value" || die "$kind path contains a symlink"
  case "$kind" in
    directory) [[ -d "$value" && ! -L "$value" ]] || die "$kind path is not a directory" ;;
    executable) [[ -f "$value" && -x "$value" && ! -L "$value" ]] || die "$kind path is not executable" ;;
    *) die 'internal path-kind error' ;;
  esac
  resolved="$(realpath -e -- "$value")" || die "$kind path cannot be resolved"
  [[ "$resolved" == "$value" ]] || die "$kind path is not normalized"
  printf '%s\n' "$resolved"
}

require_lower_hex() {
  local value="$1"
  local length="$2"
  [[ ${#value} -eq "$length" && "$value" =~ ^[0-9a-f]+$ ]] ||
    die "expected lowercase ${length}-hex value"
}

available_kib() {
  local root="$1"
  local value
  value="$(df -Pk -- "$root" | awk 'NR == 2 {print $4}')" || return 1
  [[ "$value" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$value"
}

available_inodes() {
  local root="$1"
  local value
  value="$(df -Pi -- "$root" | awk 'NR == 2 {print $4}')" || return 1
  [[ "$value" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$value"
}

record_archive_basename=
record_archive_sha256=
record_archive_size_bytes=
record_completed_stage=
record_manifest_basename=
record_reviewed_commit=
record_status=
record_output=

parse_record() {
  local record="$1"
  local expected_status="$2"
  local pattern
  local archive archive_stage archive_digest digest size stage manifest commit status
  pattern='^\{"archive_basename":"(phase2b3a-workspace-(a0|a1|a2)-([0-9a-f]{64})[.]tar)","archive_sha256":"([0-9a-f]{64})","archive_size_bytes":([0-9]+),"completed_stage":"(a0|a1|a2)","manifest_basename":"(phase2b3a-workspace-(a0|a1|a2)-[0-9a-f]{64}[.]json)","reviewed_commit":"([0-9a-f]{40})","status":"(build|publish|verify)"\}$'
  [[ "$record" =~ $pattern ]] || return 1
  archive="${BASH_REMATCH[1]}"
  archive_stage="${BASH_REMATCH[2]}"
  archive_digest="${BASH_REMATCH[3]}"
  digest="${BASH_REMATCH[4]}"
  size="${BASH_REMATCH[5]}"
  stage="${BASH_REMATCH[6]}"
  manifest="${BASH_REMATCH[7]}"
  commit="${BASH_REMATCH[9]}"
  status="${BASH_REMATCH[10]}"
  [[ "$archive_stage" == "$stage" ]] || return 1
  [[ "$archive_digest" == "$digest" ]] || return 1
  [[ "$manifest" == "${archive%.tar}.json" ]] || return 1
  [[ "$status" == "$expected_status" ]] || return 1
  (( ${#size} <= 18 )) || return 1
  (( size <= 9223372035781033983 )) || return 1
  record_archive_basename="$archive"
  record_archive_sha256="$digest"
  record_archive_size_bytes="$size"
  record_completed_stage="$stage"
  record_manifest_basename="$manifest"
  record_reviewed_commit="$commit"
  record_status="$status"
}

require_one_record() {
  local output_path="$1"
  local expected_status="$2"
  local output
  local byte_count
  local line_count
  [[ -f "$output_path" && ! -L "$output_path" ]] || return 1
  output="$(<"$output_path")"
  byte_count="$(wc -c < "$output_path")" || return 1
  line_count="$(wc -l < "$output_path")" || return 1
  [[ "$byte_count" =~ ^[0-9]+$ && "$line_count" == 1 ]] || return 1
  (( byte_count == ${#output} + 1 )) || return 1
  parse_record "$output" "$expected_status" || return 1
  record_output="$output"
}

records_agree() {
  [[ "$1" == "$2" && "$1" == "$3" &&
    "$4" == "$5" && "$4" == "$6" &&
    "$7" == "$8" && "$7" == "$9" &&
    "${10}" == "${11}" && "${10}" == "${12}" &&
    "${13}" == "${14}" && "${13}" == "${15}" ]]
}

run_main() {
  (( $# == 6 )) || die 'argument count; usage: checkpoint_workspace.sh BASE_PYTHON WORKSPACE_ROOT PERSISTENT_ROOT REPOSITORY COMPLETED_STAGE REVIEWED_COMMIT'

  local base_python workspace_root persistent_root repository completed_stage reviewed_commit
  base_python="$(require_existing_path "$1" executable)"
  workspace_root="$(require_existing_path "$2" directory)"
  persistent_root="$(require_existing_path "$3" directory)"
  repository="$(require_existing_path "$4" directory)"
  completed_stage="$5"
  reviewed_commit="$6"
  case "$completed_stage" in a0 | a1 | a2) ;; *) die 'completed stage must be a0, a1, or a2' ;; esac
  require_lower_hex "$reviewed_commit" 40
  [[ -f "$repository/pyproject.toml" && -f "$repository/uv.lock" && -d "$repository/src/trustsr" ]] ||
    die 'repository is not the reviewed project checkout'
  [[ "$workspace_root" != "$persistent_root" ]] || die 'workspace and persistent roots must differ'
  [[ "$repository" == "$workspace_root"/* ]] || die 'repository is outside workspace root'
  [[ "$base_python" != "$repository"/* && "$base_python" != "$workspace_root"/* ]] ||
    die 'interpreter is not the cloud base Python'
  mountpoint -q -- "$workspace_root" || die 'workspace mountpoint is unavailable'
  mountpoint -q -- "$persistent_root" || die 'persistent mountpoint is unavailable'

  local git_head git_branch git_status
  git_head="$(git -C "$repository" rev-parse HEAD)" || die 'reviewed Git checkout cannot be inspected'
  git_branch="$(git -C "$repository" symbolic-ref --short HEAD)" || die 'reviewed Git checkout cannot be inspected'
  git_status="$(git -C "$repository" status --porcelain)" || die 'reviewed Git checkout cannot be inspected'
  require_lower_hex "$git_head" 40
  [[ -n "$git_branch" ]] || die 'reviewed Git checkout must be attached'
  [[ -z "$git_status" ]] || die 'reviewed Git checkout must be clean'
  [[ "$git_head" == "$reviewed_commit" ]] || die 'reviewed commit does not match checkout HEAD'

  local work_kib
  work_kib="$(available_kib "$workspace_root")" || die 'could not determine workspace free disk space'
  (( work_kib >= 10 * 1024 * 1024 )) || die 'at least 10 GiB free workspace space is required'

  local log_directory active_lock
  log_directory="$workspace_root/trustsr/phase2b3a/logs"
  reject_symlink_components "$log_directory" || die 'stage log directory contains a symlink'
  if [[ -e "$log_directory" || -L "$log_directory" ]]; then
    [[ -d "$log_directory" && ! -L "$log_directory" ]] || die 'stage log directory is invalid'
    active_lock="$(find -P "$log_directory" -maxdepth 1 -type d -name '*.jsonl.lock' -print -quit)" ||
      die 'stage log directory cannot be inspected'
    [[ -z "$active_lock" ]] || die 'a stage log is reserved by another process'
  fi

  local checkpoint_directory reservation scratch=
  checkpoint_directory="$persistent_root/trustsr-phase2b3a-checkpoints"
  reject_symlink_components "$checkpoint_directory" || die 'checkpoint directory contains a symlink'
  mkdir -p -- "$checkpoint_directory" || die 'checkpoint directory cannot be created'
  [[ -d "$checkpoint_directory" && ! -L "$checkpoint_directory" ]] || die 'checkpoint directory is invalid'
  reservation="$checkpoint_directory/.checkpoint.lock"
  mkdir -- "$reservation" 2>/dev/null || die 'checkpoint publication is already reserved'

  cleanup_checkpoint_transaction() {
    [[ -z "${scratch:-}" ]] || rm -rf -- "$scratch" 2>/dev/null || true
    rmdir -- "$reservation" 2>/dev/null || true
  }
  trap cleanup_checkpoint_transaction EXIT

  scratch="$(mktemp -d -- "$workspace_root/.phase2b3a-checkpoint.XXXXXX")" || die 'workspace scratch directory cannot be created'
  reject_symlink_components "$scratch" || die 'workspace scratch directory contains a symlink'

  local build_file publish_file verify_file build_record publish_record verify_record
  build_file="$scratch/build.json"
  publish_file="$scratch/publish.json"
  verify_file="$scratch/verify.json"
  if ! PYTHONPATH="$repository/src" "$base_python" -m trustsr.artifacts.workspace_checkpoint \
    build "$workspace_root" "$scratch" "$completed_stage" "$reviewed_commit" > "$build_file"; then
    die 'checkpoint build failed'
  fi
  require_one_record "$build_file" build || die 'checkpoint build emitted an invalid record'
  build_record="$record_output"
  local build_archive build_digest build_size build_stage build_manifest build_commit
  build_archive="$record_archive_basename"
  build_digest="$record_archive_sha256"
  build_size="$record_archive_size_bytes"
  build_stage="$record_completed_stage"
  build_manifest="$record_manifest_basename"
  build_commit="$record_reviewed_commit"
  [[ "$build_stage" == "$completed_stage" && "$build_commit" == "$reviewed_commit" ]] ||
    die 'checkpoint build record does not match requested stage or commit'

  local persistent_kib persistent_inodes required_bytes available_bytes
  persistent_kib="$(available_kib "$persistent_root")" || die 'could not determine persistent free disk space'
  persistent_inodes="$(available_inodes "$persistent_root")" || die 'could not determine persistent free inodes'
  required_bytes=$((build_size + 1024 * 1024 * 1024))
  available_bytes=$((persistent_kib * 1024))
  (( available_bytes >= required_bytes )) || die 'persistent storage lacks checkpoint capacity'
  (( persistent_inodes >= 4 )) || die 'persistent storage lacks checkpoint inodes'

  if ! PYTHONPATH="$repository/src" "$base_python" -m trustsr.artifacts.workspace_checkpoint \
    publish "$scratch" "$build_manifest" "$checkpoint_directory" > "$publish_file"; then
    die 'checkpoint publish failed'
  fi
  require_one_record "$publish_file" publish || die 'checkpoint publish emitted an invalid record'
  publish_record="$record_output"
  local publish_archive publish_digest publish_size publish_stage publish_manifest publish_commit
  publish_archive="$record_archive_basename"
  publish_digest="$record_archive_sha256"
  publish_size="$record_archive_size_bytes"
  publish_stage="$record_completed_stage"
  publish_manifest="$record_manifest_basename"
  publish_commit="$record_reviewed_commit"

  if ! PYTHONPATH="$repository/src" "$base_python" -m trustsr.artifacts.workspace_checkpoint \
    verify "$checkpoint_directory" "$build_manifest" > "$verify_file"; then
    die 'checkpoint verification failed'
  fi
  require_one_record "$verify_file" verify || die 'checkpoint verification emitted an invalid record'
  verify_record="$record_output"
  local verify_archive verify_digest verify_size verify_stage verify_manifest verify_commit
  verify_archive="$record_archive_basename"
  verify_digest="$record_archive_sha256"
  verify_size="$record_archive_size_bytes"
  verify_stage="$record_completed_stage"
  verify_manifest="$record_manifest_basename"
  verify_commit="$record_reviewed_commit"
  records_agree \
    "$build_archive" "$publish_archive" "$verify_archive" \
    "$build_digest" "$publish_digest" "$verify_digest" \
    "$build_size" "$publish_size" "$verify_size" \
    "$build_stage" "$publish_stage" "$verify_stage" \
    "$build_manifest" "$publish_manifest" "$verify_manifest" ||
    die 'checkpoint records disagree'
  [[ "$verify_commit" == "$build_commit" && "$verify_commit" == "$publish_commit" &&
    "$verify_commit" == "$reviewed_commit" ]] || die 'checkpoint records disagree on reviewed commit'

  rm -rf -- "$scratch" || die 'workspace scratch directory cannot be removed'
  scratch=
  rmdir -- "$reservation" || die 'checkpoint reservation cannot be released'
  trap - EXIT
  printf '%s\n' "$verify_record"
}

run_main "$@"
