#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'invalid run input: %s\n' "$1" >&2
  exit 2
}

validate_path() {
  local value="$1"
  [[ -n "$value" && "$value" != / && "$value" != /root ]] || return 1
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
    [[ ! -L "$current" ]] || die "symlink component escapes remote root: $current"
  done
}

validate_derived_directory() {
  local relative="$1"
  local current="$remote_root"
  local component
  local -a components
  IFS=/ read -r -a components <<< "$relative"
  for component in "${components[@]}"; do
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || die 'derived path escape'
    current="${current}/${component}"
    [[ ! -L "$current" ]] || die "derived path symlink escape: $relative"
    [[ ! -e "$current" || -d "$current" ]] || die "derived path is not a directory: $relative"
  done
}

create_derived_directory() {
  local relative="$1"
  local current="$remote_root"
  local component
  local resolved
  local -a components
  IFS=/ read -r -a components <<< "$relative"
  for component in "${components[@]}"; do
    current="${current}/${component}"
    [[ ! -L "$current" ]] || die "derived path symlink escape: $relative"
    if [[ ! -e "$current" ]]; then
      mkdir -- "$current"
    fi
    [[ -d "$current" && ! -L "$current" ]] || die "derived path is not a safe directory: $relative"
    resolved="$(realpath -e -- "$current")" || die "cannot resolve derived path: $relative"
    [[ "$resolved" == "$remote_root"/* ]] || die "derived path escapes canonical remote root: $relative"
  done
}

if [[ $# -ne 2 ]]; then
  die 'argument count; usage: run_remote.sh REMOTE_ROOT preflight|single|benchmark|manifest'
fi

remote_root="$1"
stage="$2"
base_python=/opt/conda/bin/python
validate_path "$remote_root" || die 'remote root'
[[ "$remote_root" == /* ]] || die 'remote root must be absolute'
reject_symlink_components "$remote_root"
resolved_root="$(realpath -e -- "$remote_root")" || die 'remote root'
[[ "$resolved_root" == /root/rivermind-fs/* ]] || die 'remote root must resolve under /root/rivermind-fs/'
[[ -d "$remote_root" && ! -L "$remote_root" ]] || die 'remote root must be an existing directory'
remote_root="$(cd -- "$remote_root" && pwd -P)" || die 'remote root'
case "$stage" in
  preflight|single|benchmark|manifest) ;;
  *) die 'stage must be preflight, single, benchmark, or manifest' ;;
esac

derived_directories=(
  repo
  data
  data/opensr
  models
  models/sen2srlite
  models/ldsr-s2
  artifacts
  artifacts/cache
  artifacts/cache/predictions
  artifacts/phase1b
  artifacts/phase1b/cache
)
for relative in "${derived_directories[@]}"; do
  validate_derived_directory "$relative"
done

project_root="${remote_root}/repo"
[[ -d "$project_root" && ! -L "$project_root" ]] || die 'canonical project root is required'
[[ -x "$base_python" ]] || die "required cloud-image interpreter is unavailable: $base_python"

for relative in \
  data/opensr \
  models/sen2srlite \
  models/ldsr-s2 \
  artifacts/cache/predictions \
  artifacts/phase1b/cache; do
  create_derived_directory "$relative"
done

export TRUSTSR_DATA_CACHE_DIR="${remote_root}/data/opensr"
export TRUSTSR_SEN2SR_MODEL_DIR="${remote_root}/models/sen2srlite"
export TRUSTSR_LDSR_MODEL_DIR="${remote_root}/models/ldsr-s2"
export TRUSTSR_ARTIFACT_ROOT="${remote_root}/artifacts"

cd -- "$project_root"
exec "$base_python" -m trustsr.cli.ldsr_gpu "$stage" \
  --project-root "$project_root" \
  --dataset-cache-dir "$TRUSTSR_DATA_CACHE_DIR" \
  --ldsr-model-dir "$TRUSTSR_LDSR_MODEL_DIR" \
  --sen2srlite-model-dir "$TRUSTSR_SEN2SR_MODEL_DIR" \
  --artifacts-dir "$TRUSTSR_ARTIFACT_ROOT" \
  --prediction-cache-dir "${TRUSTSR_ARTIFACT_ROOT}/cache/predictions"
