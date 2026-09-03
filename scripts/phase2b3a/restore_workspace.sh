#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'invalid Phase 2B3-A workspace restore input: %s\n' "$1" >&2
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
  local archive archive_stage archive_digest digest size stage manifest manifest_stage commit status
  pattern='^\{"archive_basename":"(phase2b3a-workspace-(a0|a1|a2)-([0-9a-f]{64})[.]tar)","archive_sha256":"([0-9a-f]{64})","archive_size_bytes":([1-9][0-9]*),"completed_stage":"(a0|a1|a2)","manifest_basename":"(phase2b3a-workspace-(a0|a1|a2)-[0-9a-f]{64}[.]json)","reviewed_commit":"([0-9a-f]{40})","status":"(verify|restore)"\}$'
  [[ "$record" =~ $pattern ]] || return 1
  archive="${BASH_REMATCH[1]}"
  archive_stage="${BASH_REMATCH[2]}"
  archive_digest="${BASH_REMATCH[3]}"
  digest="${BASH_REMATCH[4]}"
  size="${BASH_REMATCH[5]}"
  stage="${BASH_REMATCH[6]}"
  manifest="${BASH_REMATCH[7]}"
  manifest_stage="${BASH_REMATCH[8]}"
  commit="${BASH_REMATCH[9]}"
  status="${BASH_REMATCH[10]}"
  [[ "$archive_stage" == "$stage" && "$manifest_stage" == "$stage" ]] || return 1
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
  local output byte_count line_count
  [[ -f "$output_path" && ! -L "$output_path" ]] || return 1
  output="$(<"$output_path")"
  byte_count="$(wc -c < "$output_path")" || return 1
  line_count="$(wc -l < "$output_path")" || return 1
  [[ "$byte_count" =~ ^[0-9]+$ && "$line_count" == 1 ]] || return 1
  (( byte_count == ${#output} + 1 )) || return 1
  parse_record "$output" "$expected_status" || return 1
  record_output="$output"
}

mount_identity_major_minor=
mount_identity_fsroot=
mount_identity_target=

read_mount_identity() {
  local path="$1"
  local output extra
  output="$(findmnt -n -o MAJ:MIN,FSROOT,TARGET --target "$path")" || return 1
  [[ -n "$output" && "$output" != *$'\n'* && "$output" != *$'\r'* ]] || return 1
  read -r mount_identity_major_minor mount_identity_fsroot mount_identity_target extra <<< "$output"
  [[ -z "$extra" && "$mount_identity_major_minor" =~ ^[0-9]+:[0-9]+$ ]] || return 1
  [[ "$mount_identity_fsroot" == /* && "$mount_identity_target" == /* ]] || return 1
}

archive_resumed_preflight_outputs() {
  local workspace_root="$1"
  local completed_stage="$2"
  local checkpoint_commit="$3"
  [[ "$completed_stage" == a0 || "$completed_stage" == a1 ]] || return 0

  local phase_root="$workspace_root/trustsr/phase2b3a"
  local log_source="$phase_root/logs/preflight.jsonl"
  local log_target="$phase_root/logs/preflight-$completed_stage-$checkpoint_commit.jsonl"
  local -a runtime_candidates=()
  local runtime_source runtime_target
  shopt -s nullglob
  runtime_candidates=(
    "$phase_root"/results/*/phase2b3a-preflight-runtime.json
  )
  shopt -u nullglob

  local log_present=0
  [[ -e "$log_source" || -L "$log_source" ]] && log_present=1
  (( log_present && ${#runtime_candidates[@]} == 1 )) ||
    die 'restored preflight evidence is incomplete or ambiguous'
  runtime_source="${runtime_candidates[0]}"
  runtime_target="${runtime_source%/*}/phase2b3a-$completed_stage-preflight-runtime-$checkpoint_commit.json"

  reject_symlink_components "$log_source" || die 'restored preflight log contains a symlink'
  reject_symlink_components "$runtime_source" || die 'restored preflight runtime contains a symlink'
  [[ -f "$log_source" && ! -L "$log_source" ]] || die 'restored preflight log is invalid'
  [[ -f "$runtime_source" && ! -L "$runtime_source" ]] ||
    die 'restored preflight runtime is invalid'
  [[ ! -e "$log_target" && ! -L "$log_target" ]] ||
    die 'resumed preflight log archive collision'
  [[ ! -e "$runtime_target" && ! -L "$runtime_target" ]] ||
    die 'resumed preflight runtime archive collision'

  mv -- "$log_source" "$log_target" || die 'restored preflight log cannot be archived'
  if ! mv -- "$runtime_source" "$runtime_target"; then
    mv -- "$log_target" "$log_source" 2>/dev/null || true
    die 'restored preflight runtime cannot be archived'
  fi
}

run_main() {
  (( $# >= 7 && $# <= 9 )) || die 'argument count; usage: restore_workspace.sh BASE_PYTHON WORKSPACE_ROOT PERSISTENT_ROOT REPOSITORY MANIFEST_BASENAME SEN2SRLITE_SOURCE LDSR_SOURCE [bind|copy] [CHECKPOINT_REVIEWED_COMMIT]'

  local base_python workspace_root persistent_root repository manifest_basename
  local sen2srlite_source ldsr_source model_restore_mode checkpoint_reviewed_commit
  local checkpoint_directory
  base_python="$(require_existing_path "$1" executable)"
  workspace_root="$(require_existing_path "$2" directory)"
  persistent_root="$(require_existing_path "$3" directory)"
  repository="$(require_existing_path "$4" directory)"
  manifest_basename="$5"
  sen2srlite_source="$(require_existing_path "$6" directory)"
  ldsr_source="$(require_existing_path "$7" directory)"
  model_restore_mode="${8:-bind}"
  checkpoint_reviewed_commit="${9:-}"
  [[ "$model_restore_mode" == bind || "$model_restore_mode" == copy ]] ||
    die 'model restore mode must be bind or copy'
  [[ -z "$checkpoint_reviewed_commit" ||
    (${#checkpoint_reviewed_commit} -eq 40 && "$checkpoint_reviewed_commit" =~ ^[0-9a-f]+$) ]] ||
    die 'checkpoint reviewed commit must be a lowercase 40-character digest'
  [[ "$manifest_basename" =~ ^phase2b3a-workspace-(a0|a1|a2)-[0-9a-f]{64}[.]json$ ]] ||
    die 'checkpoint manifest basename'
  [[ -f "$repository/pyproject.toml" && -f "$repository/uv.lock" && -d "$repository/src/trustsr" ]] ||
    die 'repository is not the reviewed project checkout'
  [[ "$workspace_root" != "$persistent_root" ]] || die 'workspace and persistent roots must differ'
  [[ "$repository" == "$workspace_root"/* ]] || die 'repository is outside workspace root'
  [[ "$base_python" != "$repository"/* && "$base_python" != "$workspace_root"/* ]] ||
    die 'interpreter is not the cloud base Python'
  [[ "$sen2srlite_source" == "$persistent_root"/* ]] || die 'Sen2SRLite source is outside persistent root'
  [[ "$ldsr_source" == "$persistent_root"/* ]] || die 'LDSR source is outside persistent root'
  mountpoint -q -- "$workspace_root" || die 'workspace mountpoint is unavailable'
  mountpoint -q -- "$persistent_root" || die 'persistent mountpoint is unavailable'
  [[ ! -e "$workspace_root/trustsr" && ! -L "$workspace_root/trustsr" ]] ||
    die 'live trustsr destination already exists'

  checkpoint_directory="$persistent_root/trustsr-phase2b3a-checkpoints"
  checkpoint_directory="$(require_existing_path "$checkpoint_directory" directory)"

  local git_head git_branch git_status
  git_head="$(git -C "$repository" rev-parse HEAD)" || die 'reviewed Git checkout cannot be inspected'
  git_branch="$(git -C "$repository" symbolic-ref --short HEAD)" || die 'reviewed Git checkout cannot be inspected'
  git_status="$(git -C "$repository" status --porcelain)" || die 'reviewed Git checkout cannot be inspected'
  [[ ${#git_head} -eq 40 && "$git_head" =~ ^[0-9a-f]+$ ]] || die 'reviewed Git HEAD is invalid'
  [[ -n "$git_branch" ]] || die 'reviewed Git checkout must be attached'
  [[ -z "$git_status" ]] || die 'reviewed Git checkout must be clean'
  if [[ -z "$checkpoint_reviewed_commit" ]]; then
    checkpoint_reviewed_commit="$git_head"
  fi

  local verify_file= restore_file= model_copy_file=
  local model_mount_directory="$workspace_root/model-mounts"
  local created_model_mount_directory=0
  local copied_models_published=0
  local restored_workspace_published=0
  local rollback_models=1
  local -a created_targets=()
  local -a mounted_targets=()

  cleanup_restore_transaction() {
    local index target
    if (( rollback_models )); then
      for ((index = ${#mounted_targets[@]} - 1; index >= 0; index--)); do
        umount -- "${mounted_targets[index]}" 2>/dev/null || true
      done
      for ((index = ${#created_targets[@]} - 1; index >= 0; index--)); do
        target="${created_targets[index]}"
        rmdir -- "$target" 2>/dev/null || true
      done
      if (( created_model_mount_directory )); then
        rmdir -- "$model_mount_directory" 2>/dev/null || true
      fi
      if (( copied_models_published )) && [[ -d "$model_mount_directory" && ! -L "$model_mount_directory" ]]; then
        chmod -R u+rwX -- "$model_mount_directory" 2>/dev/null || true
        rm -rf --one-file-system -- "$model_mount_directory" 2>/dev/null || true
      fi
      if (( restored_workspace_published )) &&
        [[ -d "$workspace_root/trustsr" && ! -L "$workspace_root/trustsr" ]]; then
        chmod -R u+rwX -- "$workspace_root/trustsr" 2>/dev/null || true
        rm -rf --one-file-system -- "$workspace_root/trustsr" 2>/dev/null || true
      fi
    fi
    [[ -z "$verify_file" ]] || rm -f -- "$verify_file" 2>/dev/null || true
    [[ -z "$restore_file" ]] || rm -f -- "$restore_file" 2>/dev/null || true
    [[ -z "$model_copy_file" ]] || rm -f -- "$model_copy_file" 2>/dev/null || true
  }
  trap cleanup_restore_transaction EXIT

  verify_file="$(mktemp -- "$workspace_root/.phase2b3a-verify.XXXXXX")" ||
    die 'checkpoint verification output cannot be created'
  if ! PYTHONPATH="$repository/src" "$base_python" -m trustsr.artifacts.workspace_checkpoint \
    verify "$checkpoint_directory" "$manifest_basename" > "$verify_file"; then
    die 'checkpoint verification failed'
  fi
  require_one_record "$verify_file" verify || die 'checkpoint verify emitted an invalid record'
  local verify_archive="$record_archive_basename"
  local verify_digest="$record_archive_sha256"
  local verify_size="$record_archive_size_bytes"
  local verify_stage="$record_completed_stage"
  local verify_manifest="$record_manifest_basename"
  local verify_commit="$record_reviewed_commit"
  [[ "$verify_manifest" == "$manifest_basename" ]] || die 'checkpoint verify record names another manifest'
  [[ "$verify_commit" == "$checkpoint_reviewed_commit" ]] ||
    die 'checkpoint reviewed commit does not match the explicit expected commit'
  git -C "$repository" merge-base --is-ancestor "$checkpoint_reviewed_commit" "$git_head" ||
    die 'checkpoint reviewed commit is outside restore code history'
  rm -f -- "$verify_file" || die 'checkpoint verification output cannot be removed'
  verify_file=

  reject_symlink_components "$model_mount_directory" || die 'model directory contains a symlink'
  if [[ "$model_restore_mode" == copy ]]; then
    [[ ! -e "$model_mount_directory" && ! -L "$model_mount_directory" ]] ||
      die 'model copy target collision'
    model_copy_file="$(mktemp -- "$workspace_root/.phase2b3a-model-copy-output.XXXXXX")" ||
      die 'model copy output cannot be created'
    if ! PYTHONPATH="$repository/src" "$base_python" -m trustsr.artifacts.model_restore \
      "$model_mount_directory" "$sen2srlite_source" "$ldsr_source" > "$model_copy_file"; then
      if [[ -d "$model_mount_directory" && ! -L "$model_mount_directory" ]]; then
        copied_models_published=1
      fi
      die 'verified model copy failed'
    fi
    copied_models_published=1
    local model_copy_output model_copy_bytes model_copy_lines model_copy_pattern
    model_copy_output="$(<"$model_copy_file")"
    model_copy_bytes="$(wc -c < "$model_copy_file")" || die 'model copy output is invalid'
    model_copy_lines="$(wc -l < "$model_copy_file")" || die 'model copy output is invalid'
    model_copy_pattern='^\{"ldsr_inventory_sha256":"[0-9a-f]{64}","mode":"copy","sen2srlite_inventory_sha256":"[0-9a-f]{64}","status":"models-restored"\}$'
    [[ "$model_copy_lines" == 1 && "$model_copy_bytes" =~ ^[0-9]+$ ]] ||
      die 'model copy emitted an invalid record'
    (( model_copy_bytes == ${#model_copy_output} + 1 )) ||
      die 'model copy emitted an invalid record'
    [[ "$model_copy_output" =~ $model_copy_pattern ]] ||
      die 'model copy emitted an invalid record'
    rm -f -- "$model_copy_file" || die 'model copy output cannot be removed'
    model_copy_file=
  else
    if [[ -e "$model_mount_directory" || -L "$model_mount_directory" ]]; then
      [[ -d "$model_mount_directory" && ! -L "$model_mount_directory" ]] ||
        die 'model mount directory is not an ordinary directory'
    else
      mkdir -m 0700 -- "$model_mount_directory" || die 'model mount directory cannot be created'
      created_model_mount_directory=1
    fi

    local source target options index
    local source_major_minor source_fsroot source_mount_target
    local target_major_minor target_fsroot target_mount_target relative expected_fsroot
    local -a sources=("$sen2srlite_source" "$ldsr_source")
    local -a targets=("$model_mount_directory/sen2srlite" "$model_mount_directory/ldsr-s2")
    for target in "${targets[@]}"; do
      reject_symlink_components "$target" || die 'model mount target contains a symlink'
      [[ ! -e "$target" && ! -L "$target" ]] || die 'model mount target collision'
      mkdir -m 0700 -- "$target" || die 'model mount target cannot be created'
      created_targets+=("$target")
    done
    for ((index = 0; index < ${#sources[@]}; index++)); do
      source="${sources[index]}"
      target="${targets[index]}"
      if ! mount --bind "$source" "$target"; then
        die 'model bind mount failed'
      fi
      mounted_targets+=("$target")
      if ! mount -o remount,bind,ro "$target"; then
        die 'model bind mount could not be made read-only'
      fi
      read_mount_identity "$source" || die 'model source mount identity cannot be inspected'
      source_major_minor="$mount_identity_major_minor"
      source_fsroot="$mount_identity_fsroot"
      source_mount_target="$mount_identity_target"
      [[ "$source_mount_target" == / || "$source" == "$source_mount_target" ||
        "$source" == "$source_mount_target"/* ]] || die 'model source mount identity mismatch'
      if [[ "$source_mount_target" == / ]]; then
        relative="$source"
      elif [[ "$source" == "$source_mount_target" ]]; then
        relative=
      else
        relative="${source#"$source_mount_target"}"
      fi
      if [[ -z "$relative" ]]; then
        expected_fsroot="$source_fsroot"
      elif [[ "$source_fsroot" == / ]]; then
        expected_fsroot="$relative"
      else
        expected_fsroot="${source_fsroot%/}$relative"
      fi
      read_mount_identity "$target" || die 'model bind mount identity cannot be inspected'
      target_major_minor="$mount_identity_major_minor"
      target_fsroot="$mount_identity_fsroot"
      target_mount_target="$mount_identity_target"
      [[ "$target_mount_target" == "$target" &&
        "$target_major_minor" == "$source_major_minor" &&
        "$target_fsroot" == "$expected_fsroot" ]] || die 'model bind mount identity mismatch'
      options="$(findmnt -n -o OPTIONS --target "$target")" ||
        die 'model bind mount options cannot be inspected'
      [[ -n "$options" && "$options" != *$'\n'* && "$options" != *$'\r'* ]] ||
        die 'model bind mount options are invalid'
      [[ ",$options," == *,ro,* ]] || die 'model bind mount is not read-only'
    done
  fi

  restore_file="$(mktemp -- "$workspace_root/.phase2b3a-restore-output.XXXXXX")" ||
    die 'checkpoint restore output cannot be created'
  if ! PYTHONPATH="$repository/src" "$base_python" -m trustsr.artifacts.workspace_checkpoint \
    restore "$checkpoint_directory" "$manifest_basename" "$workspace_root" \
    "$checkpoint_reviewed_commit" > "$restore_file"; then
    [[ ! -d "$workspace_root/trustsr" || -L "$workspace_root/trustsr" ]] ||
      restored_workspace_published=1
    die 'checkpoint restore failed'
  fi
  restored_workspace_published=1
  rollback_models=0
  require_one_record "$restore_file" restore || die 'checkpoint restore emitted an invalid record'
  [[ "$record_archive_basename" == "$verify_archive" &&
    "$record_archive_sha256" == "$verify_digest" &&
    "$record_archive_size_bytes" == "$verify_size" &&
    "$record_completed_stage" == "$verify_stage" &&
    "$record_manifest_basename" == "$verify_manifest" &&
    "$record_reviewed_commit" == "$verify_commit" ]] ||
    die 'checkpoint verify and restore records disagree'
  rollback_models=1
  archive_resumed_preflight_outputs \
    "$workspace_root" "$record_completed_stage" "$checkpoint_reviewed_commit"
  rm -f -- "$restore_file" || die 'checkpoint restore output cannot be removed'
  restore_file=
  rollback_models=0
  trap - EXIT
  printf '{"archive_basename":"%s","archive_sha256":"%s","archive_size_bytes":%s,"checkpoint_reviewed_commit":"%s","completed_stage":"%s","manifest_basename":"%s","model_restore_mode":"%s","restore_code_commit":"%s","status":"restore"}\n' \
    "$record_archive_basename" "$record_archive_sha256" "$record_archive_size_bytes" \
    "$checkpoint_reviewed_commit" "$record_completed_stage" "$record_manifest_basename" \
    "$model_restore_mode" "$git_head"
}

run_main "$@"
