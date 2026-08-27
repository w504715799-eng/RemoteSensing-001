#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'invalid pull input: %s\n' "$1" >&2
  exit 2
}

validate_path() {
  local value="$1"
  [[ -n "$value" && "$value" != / && "$value" != /root ]] || return 1
  [[ "$value" != ~* && "$value" != *$'\n'* && "$value" != *$'\r'* ]] || return 1
  [[ "$value" != *[\*\?\[]* ]] || return 1
}

if [[ $# -ne 3 ]]; then
  die 'argument count; usage: pull_artifacts.sh SSH_ALIAS REMOTE_ROOT LOCAL_OUTPUT_DIR'
fi

ssh_alias="$1"
remote_root="$2"
local_output_dir="$3"
[[ "$ssh_alias" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die 'SSH config alias'
validate_path "$remote_root" || die 'remote root'
[[ "$remote_root" == /root/rivermind-data/* ]] || die 'remote root must resolve under /root/rivermind-data/'
case "$remote_root" in
  *'//'* | */./* | */../* | */. | */..) die 'remote root must be a normalized path' ;;
esac
[[ "$remote_root" =~ ^/root/rivermind-data/[A-Za-z0-9._/-]+$ ]] || die 'remote root must use safe path characters'
validate_path "$local_output_dir" || die 'local output directory'

printf -v quoted_remote_root '%q' "$remote_root"
canonical_remote_root="$(ssh -- "$ssh_alias" "realpath -e -- ${quoted_remote_root}")" || die 'could not resolve remote root through SSH alias'
validate_path "$canonical_remote_root" || die 'remote root returned an invalid canonical path'
[[ "$canonical_remote_root" == /root/rivermind-data/* ]] || die 'remote root must resolve under /root/rivermind-data/'
remote_root="$canonical_remote_root"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repository_dir="$(cd -- "${script_dir}/../.." && pwd -P)"
mkdir -p -- "$local_output_dir/phase1b"
manifest_path="${local_output_dir}/phase1b/artifact-manifest.json"

rsync --archive --protect-args -- \
  "${ssh_alias}:${remote_root}/artifacts/phase1b/artifact-manifest.json" \
  "${local_output_dir}/phase1b"

artifact_paths_output="$(uv run --directory "$repository_dir" python -c '
import json
import sys
from pathlib import PurePosixPath

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "files"} or manifest["schema_version"] != 1 or not isinstance(manifest["files"], list):
    raise ValueError("invalid artifact manifest")
seen = set()
paths = []
for entry in manifest["files"]:
    if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
        raise ValueError("invalid artifact manifest entry")
    relative, size, digest = entry["path"], entry["size"], entry["sha256"]
    path = PurePosixPath(relative) if isinstance(relative, str) else None
    if not relative or "\0" in relative or "\n" in relative or "\r" in relative or path is None or path.is_absolute() or ".." in path.parts or "." in path.parts or path.as_posix() != relative or relative in seen or type(size) is not int or size < 0 or not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("artifact path is not a confined relative POSIX path")
    seen.add(relative)
    paths.append(relative)
    print(relative)
if paths != sorted(paths):
    raise ValueError("artifact manifest paths must be sorted")
' "$manifest_path")"

if [[ -n "$artifact_paths_output" ]]; then
  mapfile -t artifact_paths <<< "$artifact_paths_output"
  for relative_path in "${artifact_paths[@]}"; do
    destination_dir="${local_output_dir}/$(dirname -- "$relative_path")"
    mkdir -p -- "$destination_dir"
    rsync --archive --protect-args -- \
      "${ssh_alias}:${remote_root}/artifacts/${relative_path}" \
      "$destination_dir"
  done
fi

uv run --directory "$repository_dir" python -c 'from pathlib import Path; import sys; from trustsr.artifacts import verify_artifact_manifest; verify_artifact_manifest(Path(sys.argv[1]), Path(sys.argv[2]))' "$local_output_dir" "$manifest_path"
