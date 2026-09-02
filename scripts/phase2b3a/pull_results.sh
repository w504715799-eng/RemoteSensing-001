#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'invalid Phase 2B3-A pull input: %s\n' "$1" >&2
  exit 2
}

validate_absolute_path() {
  local value="$1"
  local component
  local -a components
  [[ -n "$value" && "$value" == /* && "$value" != / && "$value" != /root ]] || return 1
  [[ "$value" != *$'\n'* && "$value" != *$'\r'* && "$value" != *$'\t'* ]] || return 1
  [[ "$value" != *' '* && "$value" != *:* && "$value" != *[\*\?\[]* ]] || return 1
  case "$value" in *'//'*) return 1 ;; esac
  IFS=/ read -r -a components <<< "${value#/}"
  for component in "${components[@]}"; do
    [[ -n "$component" && "$component" != . && "$component" != .. && "$component" != -* ]] ||
      return 1
  done
}

validate_remote_storage_root() {
  local value="$1"
  local component
  local -a components
  [[ -n "$value" && "$value" == /* && "$value" != / && "$value" != /root ]] || return 1
  [[ "$value" =~ ^(/[A-Za-z0-9._-]+)+$ ]] || return 1
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

remote_metadata() {
  local remote_path="$1"
  ssh -p "$ssh_port" -- "$ssh_host" bash -s -- "$remote_path" <<'REMOTE'
set -euo pipefail
path="$1"
[[ -f "$path" && ! -L "$path" ]]
resolved="$(realpath -e -- "$path")"
[[ "$resolved" == "$path" ]]
size="$(stat -Lc '%s' -- "$path")"
digest="$(sha256sum -- "$path")"
digest="${digest%% *}"
printf '%s %s\n' "$size" "$digest"
REMOTE
}

if (( $# != 4 )); then
  die 'argument count; usage: pull_results.sh SSH_HOST SSH_PORT REMOTE_STORAGE_ROOT LOCAL_DESTINATION'
fi

ssh_host="$1"
ssh_port="$2"
remote_storage_root="$3"
local_destination="$4"

[[ "$ssh_host" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9.-]*$ ]] ||
  die 'SSH host must be a user@host value using safe characters'
[[ "$ssh_host" != *..* ]] || die 'SSH host is not normalized'
[[ "$ssh_port" =~ ^[0-9]+$ ]] || die 'SSH port must be numeric'
(( 10#$ssh_port >= 1 && 10#$ssh_port <= 65535 )) || die 'SSH port is outside 1..65535'
validate_remote_storage_root "$remote_storage_root" || die 'remote storage root'
validate_absolute_path "$local_destination" || die 'local destination'
reject_symlink_components "$local_destination" || die 'local destination contains a symlink'

local_parent="$(dirname -- "$local_destination")"
[[ -d "$local_parent" && ! -L "$local_parent" ]] || die 'local destination parent must exist'
local_parent="$(realpath -e -- "$local_parent")" || die 'local destination parent cannot be resolved'
[[ "$local_destination" == "$local_parent/"* ]] || die 'local destination is not canonical'

post_manifest_sha256=c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a
remote_bundle_root="${remote_storage_root%/}/trustsr/phase2b3a/results/$post_manifest_sha256"
manifest_name=phase2b3a-bundle-manifest.json
publication_lock="${local_destination}.lock"

mkdir -- "$publication_lock" 2>/dev/null || die 'local destination is reserved by another pull'
staging_directory=
cleanup() {
  [[ -z "$staging_directory" ]] || rm -rf -- "$staging_directory"
  rmdir -- "$publication_lock" 2>/dev/null || true
}
trap cleanup EXIT
staging_directory="$(mktemp -d -- "$local_parent/.phase2b3a-pull.XXXXXX")"

manifest_path="$staging_directory/$manifest_name"
scp -P "$ssh_port" -- "$ssh_host:$remote_bundle_root/$manifest_name" "$manifest_path" ||
  die 'bundle manifest transfer failed'
[[ -f "$manifest_path" && ! -L "$manifest_path" ]] || die 'bundle manifest is not a regular file'
manifest_size="$(stat -Lc '%s' -- "$manifest_path")" || die 'bundle manifest size is unavailable'
[[ "$manifest_size" =~ ^[0-9]+$ ]] || die 'bundle manifest size is invalid'
(( manifest_size <= 5 * 1024 * 1024 )) || die 'bundle manifest exceeds 5 MiB'
manifest_digest="$(sha256sum -- "$manifest_path")"
manifest_digest="${manifest_digest%% *}"
remote_manifest_metadata="$(remote_metadata "$remote_bundle_root/$manifest_name")" ||
  die 'remote bundle manifest inspection failed'
[[ "$remote_manifest_metadata" == "$manifest_size $manifest_digest" ]] ||
  die 'remote and local bundle manifests differ'

manifest_output="$(python3 - "$manifest_path" <<'PY'
import json
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

path = Path(sys.argv[1])
raw = path.read_bytes()
value = json.loads(raw.decode("utf-8"))
canonical = json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
if canonical != raw or not isinstance(value, dict):
    raise ValueError("bundle manifest is not canonical JSON")
if set(value) != {"schema", "phase", "files"}:
    raise ValueError("bundle manifest schema is invalid")
if value["schema"] != "trustsr.phase2b3a-bundle-manifest.v1":
    raise ValueError("bundle manifest schema is invalid")
phase = value["phase"]
if phase not in {"a1", "a2"}:
    raise ValueError("bundle manifest phase is invalid")
expected = sorted(
    [
        f"phase2b3a-{phase}-result.json",
        f"phase2b3a-{phase}-cache-audit.json",
        f"phase2b3a-{phase}-runtime.json",
        f"phase2b3a-{phase}-replay.json",
    ]
)
entries = value["files"]
if not isinstance(entries, list) or len(entries) != 4:
    raise ValueError("bundle manifest must contain four entries")
names = []
for entry in entries:
    if not isinstance(entry, dict) or set(entry) != {"basename", "size_bytes", "sha256"}:
        raise ValueError("bundle manifest entry schema is invalid")
    name = entry["basename"]
    size = entry["size_bytes"]
    digest = entry["sha256"]
    if (
        type(name) is not str
        or PurePosixPath(name).name != name
        or PureWindowsPath(name).name != name
        or type(size) is not int
        or not 0 <= size <= 5 * 1024**2
        or type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("bundle manifest entry is invalid")
    names.append(name)
    print(f"{name}\t{size}\t{digest}")
if names != expected or len(set(names)) != 4:
    raise ValueError("bundle manifest membership or order is invalid")
PY
)" || die 'bundle manifest validation failed'
mapfile -t manifest_entries <<< "$manifest_output"
(( ${#manifest_entries[@]} == 4 )) || die 'bundle manifest did not yield four files'

phase_files=()
for entry in "${manifest_entries[@]}"; do
  IFS=$'\t' read -r basename expected_size expected_digest extra <<< "$entry"
  [[ -n "$basename" && -n "$expected_size" && -n "$expected_digest" && -z "${extra:-}" ]] ||
    die 'bundle manifest entry encoding is invalid'
  remote_path="$remote_bundle_root/$basename"
  observed_remote_metadata="$(remote_metadata "$remote_path")" ||
    die "remote evidence inspection failed: $basename"
  [[ "$observed_remote_metadata" == "$expected_size $expected_digest" ]] ||
    die "remote evidence size or digest mismatch: $basename"

  local_path="$staging_directory/$basename"
  scp -P "$ssh_port" -- "$ssh_host:$remote_path" "$local_path" ||
    die "evidence transfer failed: $basename"
  [[ -f "$local_path" && ! -L "$local_path" ]] ||
    die "transferred evidence is not a regular file: $basename"
  local_size="$(stat -Lc '%s' -- "$local_path")" || die "local evidence size failed: $basename"
  local_digest="$(sha256sum -- "$local_path")"
  local_digest="${local_digest%% *}"
  [[ "$local_size" == "$expected_size" && "$local_digest" == "$expected_digest" ]] ||
    die "local evidence size or digest mismatch: $basename"
  phase_files+=("$basename")
done

published_bundle_is_identical() {
  [[ -d "$local_destination" && ! -L "$local_destination" ]] || return 1
  existing_count="$(find "$local_destination" -mindepth 1 -maxdepth 1 -printf . | wc -c)"
  (( existing_count == 5 )) || return 1
  for basename in "$manifest_name" "${phase_files[@]}"; do
    [[ -f "$local_destination/$basename" && ! -L "$local_destination/$basename" ]] || return 1
    cmp -s -- "$staging_directory/$basename" "$local_destination/$basename" || return 1
  done
}

if [[ -e "$local_destination" || -L "$local_destination" ]]; then
  published_bundle_is_identical || die 'existing local destination contains different evidence'
else
  publication_status=0
  mv -T --no-clobber -- "$staging_directory" "$local_destination" || publication_status=$?
  if [[ -e "$staging_directory" ]]; then
    published_bundle_is_identical || die 'local bundle publication lost a conflicting race'
    rm -rf -- "$staging_directory"
  elif (( publication_status != 0 )); then
    die 'local bundle publication failed'
  fi
fi

staging_directory=
rmdir -- "$publication_lock" || die 'local destination reservation could not be released'
trap - EXIT
printf '%s\n' "$local_destination"
