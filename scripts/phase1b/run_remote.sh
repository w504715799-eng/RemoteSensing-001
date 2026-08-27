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

if [[ $# -ne 2 ]]; then
  die 'argument count; usage: run_remote.sh REMOTE_ROOT preflight|single|benchmark|manifest'
fi

remote_root="$1"
stage="$2"
validate_path "$remote_root" || die 'remote root'
resolved_root="$(realpath -e -- "$remote_root")" || die 'remote root'
[[ "$resolved_root" == /root/rivermind-data/* ]] || die 'remote root must resolve under /root/rivermind-data/'
[[ -d "$remote_root" && ! -L "$remote_root" ]] || die 'remote root must be an existing directory'
case "$stage" in
  preflight|single|benchmark|manifest) ;;
  *) die 'stage must be preflight, single, benchmark, or manifest' ;;
esac

export TRUSTSR_DATA_CACHE_DIR="${remote_root}/data/opensr"
export TRUSTSR_SEN2SR_MODEL_DIR="${remote_root}/models/sen2srlite"
export TRUSTSR_LDSR_MODEL_DIR="${remote_root}/models/ldsr-s2"
export TRUSTSR_ARTIFACT_ROOT="${remote_root}/artifacts"

cli="${remote_root}/conda-env/bin/trustsr-ldsr-gpu"
[[ -x "$cli" && ! -L "$cli" ]] || die 'compatible prefix with trustsr-ldsr-gpu is required'
exec "$cli" "$stage" \
  --dataset-cache-dir "$TRUSTSR_DATA_CACHE_DIR" \
  --ldsr-model-dir "$TRUSTSR_LDSR_MODEL_DIR" \
  --sen2srlite-model-dir "$TRUSTSR_SEN2SR_MODEL_DIR" \
  --artifacts-dir "$TRUSTSR_ARTIFACT_ROOT" \
  --prediction-cache-dir "${TRUSTSR_ARTIFACT_ROOT}/cache/predictions"
