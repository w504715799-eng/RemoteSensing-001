#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'invalid Phase 2B3-A cloud run input: %s\n' "$1" >&2
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

require_existing_path() {
  local value="$1"
  local kind="$2"
  validate_absolute_path "$value" || die "$kind path"
  reject_symlink_components "$value" || die "$kind path contains a symlink"
  case "$kind" in
    file) [[ -f "$value" && ! -L "$value" ]] || die "$kind path is not a regular file" ;;
    directory) [[ -d "$value" && ! -L "$value" ]] || die "$kind path is not a directory" ;;
    executable) [[ -f "$value" && -x "$value" && ! -L "$value" ]] || die "$kind path is not executable" ;;
    *) die 'internal path-kind error' ;;
  esac
  realpath -e -- "$value" || die "$kind path cannot be resolved"
}

require_stage() {
  case "$1" in
    preflight | single | smoke | replay | development | development-replay)
      printf '%s\n' "$1"
      ;;
    *) die 'stage must be preflight, single, smoke, replay, development, or development-replay' ;;
  esac
}

require_lower_hex() {
  local value="$1"
  local length="$2"
  [[ ${#value} -eq "$length" && "$value" =~ ^[0-9a-f]+$ ]] ||
    die "expected lowercase ${length}-hex value"
}

run_main() {
  (( $# >= 4 )) ||
    die 'argument count; usage: run_cloud.sh BASE_PYTHON STORAGE_ROOT REPOSITORY STAGE STAGE_ARGS'

  local base_python="$1"
  local storage_root="$2"
  local repository="$3"
  local stage
  shift 3
  stage="$(require_stage "$1")"
  shift

  base_python="$(require_existing_path "$base_python" executable)"
  storage_root="$(require_existing_path "$storage_root" directory)"
  repository="$(require_existing_path "$repository" directory)"
  [[ -f "$repository/pyproject.toml" && -f "$repository/uv.lock" &&
    -d "$repository/src/trustsr" ]] || die 'repository is not the reviewed project checkout'
  case "$base_python" in
    "$repository"/* | "$storage_root"/*) die 'interpreter is not the cloud base Python' ;;
  esac

  local current_home
  current_home="$(realpath -e -- "${HOME:?HOME must be set}")" || die 'home cannot be resolved'
  [[ "$storage_root" != "$current_home" ]] || die 'storage root must not be home'
  mountpoint -q -- "$storage_root" || die 'persistent storage mountpoint is unavailable'

  local available_kib
  local available_inodes
  available_kib="$(df -Pk -- "$storage_root" | awk 'NR == 2 {print $4}')"
  [[ "$available_kib" =~ ^[0-9]+$ ]] || die 'could not determine free disk space'
  (( available_kib >= 10 * 1024 * 1024 )) || die 'at least 10 GiB free space is required'
  available_inodes="$(df -Pi -- "$storage_root" | awk 'NR == 2 {print $4}')"
  [[ "$available_inodes" =~ ^[0-9]+$ ]] || die 'could not determine free inodes'
  (( available_inodes > 1024 )) || die 'more than 1024 free inodes are required'

  local selection_manifest=
  local selection_digest=
  local input_audit=
  local input_digest=
  local sen2srlite_directory=
  local ldsr_directory=
  local reviewed_commit=
  local confirm_count=0
  local argument
  local value
  local -a stage_arguments=("$@")
  while (( $# > 0 )); do
    argument="$1"
    shift
    case "$argument" in
      --confirm-cloud-storage)
        ((confirm_count += 1))
        ;;
      --selection-manifest | --selection-manifest-sha256 | --input-audit | --input-audit-sha256 | --sen2srlite-model-dir | --ldsr-model-dir | --reviewed-commit)
        (( $# > 0 )) || die "missing value for $argument"
        value="$1"
        shift
        [[ "$value" != --* ]] || die "option-like value for $argument"
        case "$argument" in
          --selection-manifest)
            [[ -z "$selection_manifest" ]] || die 'duplicate selection manifest'
            selection_manifest="$value"
            ;;
          --selection-manifest-sha256)
            [[ -z "$selection_digest" ]] || die 'duplicate selection manifest digest'
            selection_digest="$value"
            ;;
          --input-audit)
            [[ -z "$input_audit" ]] || die 'duplicate input audit'
            input_audit="$value"
            ;;
          --input-audit-sha256)
            [[ -z "$input_digest" ]] || die 'duplicate input audit digest'
            input_digest="$value"
            ;;
          --sen2srlite-model-dir)
            [[ -z "$sen2srlite_directory" ]] || die 'duplicate SEN2SRLite model directory'
            sen2srlite_directory="$value"
            ;;
          --ldsr-model-dir)
            [[ -z "$ldsr_directory" ]] || die 'duplicate LDSR model directory'
            ldsr_directory="$value"
            ;;
          --reviewed-commit)
            [[ -z "$reviewed_commit" ]] || die 'duplicate reviewed commit'
            reviewed_commit="$value"
            ;;
        esac
        ;;
      *) die "unknown or prohibited stage argument: $argument" ;;
    esac
  done

  (( confirm_count == 1 )) || die 'exactly one cloud-storage confirmation is required'
  [[ -n "$selection_manifest" && -n "$selection_digest" && -n "$input_audit" &&
    -n "$input_digest" && -n "$reviewed_commit" ]] ||
    die 'stage arguments are missing frozen manifest, audit, or reviewed commit values'
  require_lower_hex "$selection_digest" 64
  require_lower_hex "$input_digest" 64
  require_lower_hex "$reviewed_commit" 40
  selection_manifest="$(require_existing_path "$selection_manifest" file)"
  input_audit="$(require_existing_path "$input_audit" file)"
  case "$selection_manifest" in "$storage_root"/*) ;; *) die 'selection manifest is outside storage root' ;; esac
  case "$input_audit" in "$storage_root"/*) ;; *) die 'input audit is outside storage root' ;; esac

  case "$stage" in
    replay | development-replay)
      [[ -z "$sen2srlite_directory" && -z "$ldsr_directory" ]] ||
        die 'replay stages must omit model directories'
      ;;
    *)
      [[ -n "$sen2srlite_directory" && -n "$ldsr_directory" ]] ||
        die 'compute stages require both model directories'
      sen2srlite_directory="$(require_existing_path "$sen2srlite_directory" directory)"
      ldsr_directory="$(require_existing_path "$ldsr_directory" directory)"
      case "$sen2srlite_directory" in "$storage_root"/*) ;; *) die 'model directory is outside storage root' ;; esac
      case "$ldsr_directory" in "$storage_root"/*) ;; *) die 'model directory is outside storage root' ;; esac
      ;;
  esac

  local git_head
  local git_branch
  local git_status
  git_head="$(git -C "$repository" rev-parse HEAD)" || die 'reviewed Git checkout cannot be inspected'
  git_branch="$(git -C "$repository" symbolic-ref --short HEAD)" ||
    die 'reviewed Git checkout cannot be inspected'
  git_status="$(git -C "$repository" status --porcelain)" ||
    die 'reviewed Git checkout cannot be inspected'
  require_lower_hex "$git_head" 40
  [[ -n "$git_branch" ]] || die 'reviewed Git checkout must be attached'
  [[ -z "$git_status" ]] || die 'reviewed Git checkout must be clean'
  [[ "$git_head" == "$reviewed_commit" ]] || die 'reviewed commit does not match checkout HEAD'

  local log_directory="$storage_root/trustsr/phase2b3a/logs"
  local log_path="$log_directory/$stage.jsonl"
  reject_symlink_components "$log_directory" || die 'log directory contains a symlink'
  mkdir -p -- "$log_directory"
  [[ -d "$log_directory" && ! -L "$log_directory" ]] || die 'log directory is invalid'
  [[ ! -e "$log_path" && ! -L "$log_path" ]] || die 'stage log collision'

  local temporary_log
  temporary_log="$(mktemp -- "$log_directory/.${stage}.XXXXXX")"
  trap 'rm -f -- "$temporary_log"' EXIT
  cd -- "$repository"
  PYTHONPATH="$repository/src" "$base_python" -m trustsr.cli.phase2b3a "$stage" \
    --storage-root "$storage_root" \
    --project-root "$repository" \
    "${stage_arguments[@]}" | tee -- "$temporary_log"
  mv -- "$temporary_log" "$log_path"
  trap - EXIT
}

run_main "$@"
