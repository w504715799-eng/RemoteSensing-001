# Phase 2B3-A staged cloud runbook

Phase 2B3-A audits the frozen `development` split. A1 is the four-ROI stability/resource gate and
A2 is the exact 120-ROI audit; neither is calibration or `internal_test` evaluation. Historical
paths retain `uint16_divide_10000_no_clip_v1`; every new Phase 2B3-A pixel stage explicitly uses
`uint16_saturate_10000_divide_10000_v2`. The disposable
work mount holds live pixels, tensors, caches, logs, and the checkpointed `trustsr` trees. The
durable mount holds immutable checkpoint pairs and durable model sources. GitHub is the durable
source for code and Git history. Local development, review, and synthetic/CPU verification finish
before the rented GPU instance is started. On the current unprivileged cloud container, model
restore uses an explicit verified copy into disposable storage; bind mode is allowed only when the
provider explicitly grants mount capability.

No command below installs software, downloads data, or controls an instance. Instance-identifying
values remain in the operator shell and never enter tracked files. The final reviewed integration
commit is pending. Keep the GPU rerun paused until local integration passes, that exact commit is
pushed, and the user reports that the GPU has been restarted.

## A0: local review and immutable handoff

A0 requires no cloud or GPU connection. Commit, review, and push the exact local code before A1.

```bash
uv run pytest -q
uv run ruff check .
bash -n scripts/phase2b3a/*.sh
uv run trustsr-phase2b3a --help >/dev/null
PYTHONPATH=src uv run python -m trustsr.artifacts.workspace_checkpoint --help >/dev/null
PYTHONPATH=src uv run python -m trustsr.artifacts.model_restore --help >/dev/null
git diff --check
git status --short --branch
git rev-parse HEAD
```

The A0 pin is the exact reviewed and pushed commit, not a branch name or abbreviation. Set
runtime-only values in the cloud operator shell:

```bash
: "${PHASE2B3A_WORK_ROOT:?set disposable mounted work root}"
: "${PHASE2B3A_PERSISTENT_ROOT:?set durable mounted checkpoint/model root}"
: "${PHASE2B3A_REPOSITORY:?set clean attached checkout below work root}"
: "${PHASE2B3A_GIT_REMOTE:?set reviewed project Git remote}"
: "${PHASE2B3A_BASE_PYTHON:?set existing cloud base Python}"
: "${PHASE2B3A_REVIEWED_COMMIT:?set exact reviewed and pushed 40-hex SHA}"
: "${PHASE2B3A_SEN2SRLITE_SOURCE:?set durable SEN2SRLite model directory}"
: "${PHASE2B3A_LDSR_SOURCE:?set durable LDSR-S2 model directory}"
PHASE2B3A_MODEL_RESTORE_MODE=copy
PHASE2B3A_A1_CHECKPOINT_COMMIT=4df5195e0a28701391c3951659a42409f81a11c2
PHASE2B3A_STORAGE_ROOT="$PHASE2B3A_WORK_ROOT"
PHASE2B3A_SEN2SRLITE_DIR="$PHASE2B3A_WORK_ROOT/model-mounts/sen2srlite"
PHASE2B3A_LDSR_DIR="$PHASE2B3A_WORK_ROOT/model-mounts/ldsr-s2"
[[ "$PHASE2B3A_MODEL_RESTORE_MODE" == copy ]]
[[ "$PHASE2B3A_A1_CHECKPOINT_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -f "$PHASE2B3A_BASE_PYTHON" && -x "$PHASE2B3A_BASE_PYTHON" && ! -L "$PHASE2B3A_BASE_PYTHON" ]]
[[ "$(realpath -e -- "$PHASE2B3A_BASE_PYTHON")" == "$PHASE2B3A_BASE_PYTHON" ]]
```

Use the canonical, non-symlink base interpreter path. On the current cloud image this is
`/opt/conda/bin/python3.12`; `/opt/conda/bin/python` is a symlink and is intentionally rejected by
the restore boundary.

The saturation-v2 rerun transitions are exactly:

```text
UNVERIFIED -> LEGACY_A1_RESTORED -> LIVE_PHASE2B3A_RESET -> PREFLIGHT_OK
PREFLIGHT_OK -> A1_V2_OK -> A1_V2_CHECKPOINTED -> A2_OK -> A2_CHECKPOINTED
```

Only `UNVERIFIED` with no mutation, `A1_V2_CHECKPOINTED`, and `A2_CHECKPOINTED` are safe pause
points. A resumed compute session may restore the newly produced v2 A1 checkpoint, but must never
resume A2 directly from the accepted historical v1 A1. `A2_CHECKPOINTED` is terminal for compute.
A later offline inspection restore is not a compute-state transition.

## One-time inode recovery before the initial session

Perform this recovery only if the old checkout blocks inode preflight. It is one fail-closed guarded
operation: any failed command or predicate exits before deletion. It targets only the approved clean
checkout and proves both reclaimed and remaining free inodes.

```bash
set -euo pipefail
old_repo=/root/rivermind-fs/trustsr-phase1b/repo
old_filesystem_root=/root/rivermind-fs
[[ "$old_repo" == "$old_filesystem_root"/* ]]
[[ -d "$old_repo" && ! -L "$old_repo" ]]
[[ "$(realpath -e -- "$old_repo")" == "$old_repo" ]]
old_worktree="$(git -C "$old_repo" rev-parse --show-toplevel)"
[[ "$old_worktree" == "$old_repo" ]]
old_status="$(git -C "$old_repo" status --porcelain=v1 --untracked-files=all)"
[[ -z "$old_status" ]]
old_head="$(git -C "$old_repo" rev-parse HEAD)"
[[ "$old_head" =~ ^[0-9a-f]{40}$ ]]
git -C "$old_repo" fetch --prune --tags origin
old_branch="$(git -C "$old_repo" branch -r --contains "$old_head")"
old_tag="$(git -C "$old_repo" tag --contains "$old_head")"
[[ -n "$old_branch" || -n "$old_tag" ]]
old_inode_count="$(find "$old_repo" -xdev -printf . | wc -c)"
[[ "$old_inode_count" =~ ^[0-9]+$ && "$old_inode_count" -ge 1229 ]]
df -Pi -- "$old_filesystem_root"
before_free_inodes="$(df -Pi -- "$old_filesystem_root" | awk 'NR == 2 {print $4}')"
[[ "$before_free_inodes" =~ ^[0-9]+$ ]]
[[ "$old_repo" == /root/rivermind-fs/trustsr-phase1b/repo ]]
[[ "$(realpath -e -- "$old_repo")" == "$old_repo" ]]
rm -rf --one-file-system -- "$old_repo"
df -Pi -- "$old_filesystem_root"
after_free_inodes="$(df -Pi -- "$old_filesystem_root" | awk 'NR == 2 {print $4}')"
[[ "$after_free_inodes" =~ ^[0-9]+$ ]]
(( after_free_inodes - before_free_inodes >= 1229 ))
(( after_free_inodes >= 1229 ))
```

The threshold is `ceil(1024 * 1.20) = 1229`. This removes only a clean GitHub-reachable checkout,
recoverable by cloning the verified revision; it never targets a model, dataset, audit, cache, or
evidence directory. Do not broaden the target or use a wildcard.

## Every new disposable cloud session: bootstrap exact A0

Run this before initial setup or any restore. It canonicalizes non-symlink paths before mutation,
requires distinct mounted roots, recreates the code checkout from the quoted Git remote, and proves
the clean attached A0 checkout. Execute later cloud commands through this exact checkout.

```bash
set -euo pipefail
[[ "$PHASE2B3A_REVIEWED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -d "$PHASE2B3A_WORK_ROOT" && ! -L "$PHASE2B3A_WORK_ROOT" ]]
[[ -d "$PHASE2B3A_PERSISTENT_ROOT" && ! -L "$PHASE2B3A_PERSISTENT_ROOT" ]]
[[ -d "$PHASE2B3A_SEN2SRLITE_SOURCE" && ! -L "$PHASE2B3A_SEN2SRLITE_SOURCE" ]]
[[ -d "$PHASE2B3A_LDSR_SOURCE" && ! -L "$PHASE2B3A_LDSR_SOURCE" ]]
PHASE2B3A_WORK_ROOT="$(realpath -e -- "$PHASE2B3A_WORK_ROOT")"
PHASE2B3A_PERSISTENT_ROOT="$(realpath -e -- "$PHASE2B3A_PERSISTENT_ROOT")"
PHASE2B3A_SEN2SRLITE_SOURCE="$(realpath -e -- "$PHASE2B3A_SEN2SRLITE_SOURCE")"
PHASE2B3A_LDSR_SOURCE="$(realpath -e -- "$PHASE2B3A_LDSR_SOURCE")"
PHASE2B3A_STORAGE_ROOT="$PHASE2B3A_WORK_ROOT"
PHASE2B3A_SEN2SRLITE_DIR="$PHASE2B3A_WORK_ROOT/model-mounts/sen2srlite"
PHASE2B3A_LDSR_DIR="$PHASE2B3A_WORK_ROOT/model-mounts/ldsr-s2"
[[ "$PHASE2B3A_WORK_ROOT" != "$PHASE2B3A_PERSISTENT_ROOT" ]]
[[ "$PHASE2B3A_SEN2SRLITE_SOURCE" == "$PHASE2B3A_PERSISTENT_ROOT"/* ]]
[[ "$PHASE2B3A_LDSR_SOURCE" == "$PHASE2B3A_PERSISTENT_ROOT"/* ]]
mountpoint -q -- "$PHASE2B3A_WORK_ROOT"
mountpoint -q -- "$PHASE2B3A_PERSISTENT_ROOT"
repository_parent="$(realpath -e -- "$(dirname -- "$PHASE2B3A_REPOSITORY")")"
repository_name="$(basename -- "$PHASE2B3A_REPOSITORY")"
[[ "$repository_name" != . && "$repository_name" != .. ]]
PHASE2B3A_REPOSITORY="$repository_parent/$repository_name"
[[ "$PHASE2B3A_REPOSITORY" == "$PHASE2B3A_WORK_ROOT"/* ]]
[[ ! -e "$PHASE2B3A_REPOSITORY" && ! -L "$PHASE2B3A_REPOSITORY" ]]
git clone --no-checkout "$PHASE2B3A_GIT_REMOTE" "$PHASE2B3A_REPOSITORY"
git -C "$PHASE2B3A_REPOSITORY" fetch --prune --tags origin
git -C "$PHASE2B3A_REPOSITORY" switch --create phase2b3a-a0 "$PHASE2B3A_REVIEWED_COMMIT"
[[ "$(realpath -e -- "$PHASE2B3A_REPOSITORY")" == "$PHASE2B3A_REPOSITORY" ]]
[[ "$(git -C "$PHASE2B3A_REPOSITORY" rev-parse --show-toplevel)" == "$PHASE2B3A_REPOSITORY" ]]
[[ "$(git -C "$PHASE2B3A_REPOSITORY" rev-parse HEAD)" == "$PHASE2B3A_REVIEWED_COMMIT" ]]
[[ -n "$(git -C "$PHASE2B3A_REPOSITORY" symbolic-ref --short HEAD)" ]]
[[ -z "$(git -C "$PHASE2B3A_REPOSITORY" status --porcelain=v1 --untracked-files=all)" ]]
df -h -- "$PHASE2B3A_WORK_ROOT" "$PHASE2B3A_PERSISTENT_ROOT"
df -Pi -- "$PHASE2B3A_WORK_ROOT" "$PHASE2B3A_PERSISTENT_ROOT"
```

## Greenfield-only setup and A0 checkpoint

The initial verified input tree provides ordinary `trustsr/phase2b1b` and `trustsr/phase2b2a`
under work storage. Create the empty ordinary `trustsr/phase2b3a` directory. For the current
unprivileged provider, copy both durable model trees through the reviewed verifier; it compares a
canonical type/mode/per-file-SHA-256 inventory before publication, removes all write permission
from the copies, and leaves the durable sources unchanged. The restore script is used once an
existing workspace checkpoint is available.

The directory creation and model publication commands in this subsection are only for a greenfield
A0. The current saturation-v2 recovery skips them and uses the accepted legacy A1 restore in the
next subsection. In either case, define the frozen inputs and command wrappers below; defining them
does not read pixels or run a stage.

```bash
set -euo pipefail
trustsr_parent="$PHASE2B3A_WORK_ROOT/trustsr"
phase2b3a_target="$trustsr_parent/phase2b3a"
model_mount_parent="$PHASE2B3A_WORK_ROOT/model-mounts"
[[ -d "$trustsr_parent" && ! -L "$trustsr_parent" ]]
[[ "$(realpath -e -- "$trustsr_parent")" == "$trustsr_parent" ]]
[[ ! -e "$phase2b3a_target" && ! -L "$phase2b3a_target" ]]
[[ ! -e "$model_mount_parent" && ! -L "$model_mount_parent" ]]
mkdir -- "$phase2b3a_target"
[[ "$(realpath -e -- "$phase2b3a_target")" == "$phase2b3a_target" ]]
[[ -z "$(find "$phase2b3a_target" -mindepth 1 -print -quit)" ]]
PYTHONPATH="$PHASE2B3A_REPOSITORY/src" "$PHASE2B3A_BASE_PYTHON" \
  -m trustsr.artifacts.model_restore "$model_mount_parent" \
  "$PHASE2B3A_SEN2SRLITE_SOURCE" "$PHASE2B3A_LDSR_SOURCE"
[[ "$(realpath -e -- "$model_mount_parent")" == "$model_mount_parent" ]]
[[ -d "$PHASE2B3A_SEN2SRLITE_DIR" && ! -L "$PHASE2B3A_SEN2SRLITE_DIR" ]]
[[ -d "$PHASE2B3A_LDSR_DIR" && ! -L "$PHASE2B3A_LDSR_DIR" ]]
[[ -z "$(find "$model_mount_parent" -perm /222 -print -quit)" ]]
```

This verified initial setup establishes `RESTORED`. Define the frozen inputs and commands from the
reviewed checkout:

```bash
PHASE2B3A_POST_MANIFEST_SHA256=c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a
PHASE2B3A_INPUT_AUDIT_SHA256=fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b
PHASE2B3A_POST_MANIFEST="${PHASE2B3A_STORAGE_ROOT%/}/trustsr/phase2b1b/selections/${PHASE2B3A_POST_MANIFEST_SHA256}/samples.jsonl"
PHASE2B3A_INPUT_AUDIT="${PHASE2B3A_STORAGE_ROOT%/}/trustsr/phase2b2a/input-audits/${PHASE2B3A_POST_MANIFEST_SHA256}/phase2b2a-input-audit.json"
PHASE2B3A_COMMON_ARGS=(--selection-manifest "$PHASE2B3A_POST_MANIFEST" --selection-manifest-sha256 "$PHASE2B3A_POST_MANIFEST_SHA256" --input-audit "$PHASE2B3A_INPUT_AUDIT" --input-audit-sha256 "$PHASE2B3A_INPUT_AUDIT_SHA256" --reviewed-commit "$PHASE2B3A_REVIEWED_COMMIT" --confirm-cloud-storage)
PHASE2B3A_COMPUTE_ONLY_ARGS=(--sen2srlite-model-dir "$PHASE2B3A_SEN2SRLITE_DIR" --ldsr-model-dir "$PHASE2B3A_LDSR_DIR")
phase2b3a_compute() {
  "$PHASE2B3A_REPOSITORY/scripts/phase2b3a/run_cloud.sh" "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_STORAGE_ROOT" "$PHASE2B3A_REPOSITORY" "$1" "${PHASE2B3A_COMMON_ARGS[@]}" "${PHASE2B3A_COMPUTE_ONLY_ARGS[@]}"
}
phase2b3a_replay() {
  "$PHASE2B3A_REPOSITORY/scripts/phase2b3a/run_cloud.sh" "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_STORAGE_ROOT" "$PHASE2B3A_REPOSITORY" "$1" "${PHASE2B3A_COMMON_ARGS[@]}"
}
```

For a greenfield A0 only, run preflight, then create and independently verify the full three-root
checkpoint before GPU compute. The saturation-v2 recovery does not execute this A0 block. Copy the
emitted manifest basename exactly; never glob for a newest file.

```bash
phase2b3a_compute preflight
"$PHASE2B3A_REPOSITORY/scripts/phase2b3a/checkpoint_workspace.sh" "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_WORK_ROOT" "$PHASE2B3A_PERSISTENT_ROOT" "$PHASE2B3A_REPOSITORY" a0 "$PHASE2B3A_REVIEWED_COMMIT"
: "${PHASE2B3A_A0_MANIFEST_BASENAME:?copy emitted a0 manifest basename exactly}"
PYTHONPATH="$PHASE2B3A_REPOSITORY/src" "$PHASE2B3A_BASE_PYTHON" -m trustsr.artifacts.workspace_checkpoint verify "$PHASE2B3A_PERSISTENT_ROOT/trustsr-phase2b3a-checkpoints" "$PHASE2B3A_A0_MANIFEST_BASENAME"
```

## Saturation-v2 A1 and A2 rerun

The first exact A2 attempt stopped on the ninth development ROI before model construction or
inference because a raw reflectance value exceeded the former `[0,10000]` assumption. No A2
result, runtime, or replay was produced, and GPU processes returned to zero. The host-free diagnosis
found exactly one affected ROI out of 120: LR has eight values above `10000` in ordered bands
B04/B03/B02/B08 as `[4,0,0,4]`; HR has 117 as `[56,0,0,61]`; the raw crop maximum is `11968`.
The v2 policy saturates aligned-crop values at `10000`, records these counts, and rejects the full
raw input when any value exceeds `32767`. It preserves all 120 development ROIs and never reads
calibration or `internal_test` pixels.

The immutable accepted legacy A1 is recovery input only. Its exact manifest is
`phase2b3a-workspace-a1-623535c33fee50e7d05b83386158b349c4056d1f4aa256efda1189933e9993f8.json`,
its archive is `933263360` bytes, and its producer is
`4df5195e0a28701391c3951659a42409f81a11c2`. Verify that pair, restore it only to recover frozen data
and verified models, then delete and recreate only the disposable live `trustsr/phase2b3a` stage.
The durable checkpoint pair and durable model sources are never reset targets.

```bash
set -euo pipefail
: "${PHASE2B3A_A1_MANIFEST_BASENAME:=phase2b3a-workspace-a1-623535c33fee50e7d05b83386158b349c4056d1f4aa256efda1189933e9993f8.json}"
test "$PHASE2B3A_A1_MANIFEST_BASENAME" = phase2b3a-workspace-a1-623535c33fee50e7d05b83386158b349c4056d1f4aa256efda1189933e9993f8.json
PYTHONPATH="$PHASE2B3A_REPOSITORY/src" "$PHASE2B3A_BASE_PYTHON" \
  -m trustsr.artifacts.workspace_checkpoint verify \
  "$PHASE2B3A_PERSISTENT_ROOT/trustsr-phase2b3a-checkpoints" \
  "$PHASE2B3A_A1_MANIFEST_BASENAME"
"$PHASE2B3A_REPOSITORY/scripts/phase2b3a/restore_workspace.sh" \
  "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_WORK_ROOT" "$PHASE2B3A_PERSISTENT_ROOT" \
  "$PHASE2B3A_REPOSITORY" "$PHASE2B3A_A1_MANIFEST_BASENAME" \
  "$PHASE2B3A_SEN2SRLITE_SOURCE" "$PHASE2B3A_LDSR_SOURCE" \
  "$PHASE2B3A_MODEL_RESTORE_MODE" "$PHASE2B3A_A1_CHECKPOINT_COMMIT"
"$PHASE2B3A_REPOSITORY/scripts/phase2b3a/reset_live_phase2b3a.sh" \
  "$PHASE2B3A_WORK_ROOT"
test -d "$PHASE2B3A_WORK_ROOT/trustsr/phase2b3a"
test ! -L "$PHASE2B3A_WORK_ROOT/trustsr/phase2b3a"
test -z "$(find "$PHASE2B3A_WORK_ROOT/trustsr/phase2b3a" -mindepth 1 -print -quit)"
```

The restore success JSON must record copy mode, the exact historical checkpoint producer, and the
new pushed restore-code commit. The historical producer must be an ancestor of that code commit.
Do not retry with bind mode after a permission failure, replace model directories with symlinks,
broaden the reset target, or bypass any path, mount, digest, inventory, ancestry, lock, capacity, or
GPU-idleness guard. Successful reset establishes `LIVE_PHASE2B3A_RESET`.

Run formal preflight, then produce a completely fresh A1 under v2. Do not reuse the historical A1
result, cache, runtime, or replay. Checkpoint and independently verify the new A1 before A2.

```bash
phase2b3a_compute preflight
phase2b3a_compute single
phase2b3a_compute smoke
phase2b3a_replay replay
"$PHASE2B3A_REPOSITORY/scripts/phase2b3a/checkpoint_workspace.sh" \
  "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_WORK_ROOT" "$PHASE2B3A_PERSISTENT_ROOT" \
  "$PHASE2B3A_REPOSITORY" a1 "$PHASE2B3A_REVIEWED_COMMIT"
: "${PHASE2B3A_A1_V2_MANIFEST_BASENAME:?copy emitted v2 a1 manifest basename exactly}"
PYTHONPATH="$PHASE2B3A_REPOSITORY/src" "$PHASE2B3A_BASE_PYTHON" \
  -m trustsr.artifacts.workspace_checkpoint verify \
  "$PHASE2B3A_PERSISTENT_ROOT/trustsr-phase2b3a-checkpoints" \
  "$PHASE2B3A_A1_V2_MANIFEST_BASENAME"
```

Before A2 can replace the live bundle manifest, use the existing pull script from the clean local
reviewed checkout to fetch and offline-verify the A1-v2 bundle into a new destination. The
`.gitignore` allowlist for the three new A1-v2 publication paths must already be present in the
reviewed integration commit. Materialize exactly those three files, but do not stage them yet.

```bash
set -euo pipefail
: "${PHASE2B3A_LOCAL_REPOSITORY:?set clean local reviewed checkout}"
: "${PHASE2B3A_LOCAL_BRANCH:?set intended attached local evidence branch}"
: "${PHASE2B3A_LOCAL_REVIEWED_COMMIT:?set exact reviewed 40-hex A0 commit}"
: "${PHASE2B3A_REMOTE_STORAGE_ROOT:?set remote storage root only for this pull}"
: "${PHASE2B3A_SSH_HOST:?set operator-provided endpoint only in this shell}"
: "${PHASE2B3A_SSH_PORT:?set operator-provided numeric port only in this shell}"
: "${PHASE2B3A_A1_V2_BUNDLE:?set new absolute local A1-v2 bundle destination}"
[[ "$PHASE2B3A_LOCAL_REVIEWED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
PHASE2B3A_LOCAL_REPOSITORY="$(realpath -e -- "$PHASE2B3A_LOCAL_REPOSITORY")"
test "$(git -C "$PHASE2B3A_LOCAL_REPOSITORY" symbolic-ref --short HEAD)" = \
  "$PHASE2B3A_LOCAL_BRANCH"
test -z "$(git -C "$PHASE2B3A_LOCAL_REPOSITORY" status --porcelain=v1 --untracked-files=all)"
git -C "$PHASE2B3A_LOCAL_REPOSITORY" merge-base --is-ancestor \
  "$PHASE2B3A_LOCAL_REVIEWED_COMMIT" HEAD
cd -- "$PHASE2B3A_LOCAL_REPOSITORY"
a1_result=artifacts/phase2b3a/sen2naipv2-development-smoke-v2.json
a1_audit=artifacts/phase2b3a/sen2naipv2-development-smoke-cache-audit-v2.json
a1_acceptance=artifacts/phase2b3a/sen2naipv2-development-smoke-acceptance-v2.json
for path in "$a1_result" "$a1_audit" "$a1_acceptance"; do
  test ! -e "$path" && test ! -L "$path"
done
scripts/phase2b3a/pull_results.sh "$PHASE2B3A_SSH_HOST" "$PHASE2B3A_SSH_PORT" \
  "$PHASE2B3A_REMOTE_STORAGE_ROOT" "$PHASE2B3A_A1_V2_BUNDLE"
uv run trustsr-phase2b3a-verify a1 --bundle "$PHASE2B3A_A1_V2_BUNDLE" \
  --output "$a1_acceptance"
cp -- "$PHASE2B3A_A1_V2_BUNDLE/phase2b3a-a1-result.json" "$a1_result"
cp -- "$PHASE2B3A_A1_V2_BUNDLE/phase2b3a-a1-cache-audit.json" "$a1_audit"
expected_a1_status="$(printf '?? %s\n' "$a1_result" "$a1_audit" "$a1_acceptance" | sort)"
test "$(git status --porcelain=v1 --untracked-files=all | sort)" = "$expected_a1_status"
test -z "$(git diff --cached --name-only)"
```

Return to the cloud shell for exact A2, replay, checkpoint, and independent checkpoint
verification:

```bash
phase2b3a_compute development
phase2b3a_replay development-replay
"$PHASE2B3A_REPOSITORY/scripts/phase2b3a/checkpoint_workspace.sh" \
  "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_WORK_ROOT" "$PHASE2B3A_PERSISTENT_ROOT" \
  "$PHASE2B3A_REPOSITORY" a2 "$PHASE2B3A_REVIEWED_COMMIT"
: "${PHASE2B3A_A2_MANIFEST_BASENAME:?copy emitted a2 manifest basename exactly}"
PYTHONPATH="$PHASE2B3A_REPOSITORY/src" "$PHASE2B3A_BASE_PYTHON" \
  -m trustsr.artifacts.workspace_checkpoint verify \
  "$PHASE2B3A_PERSISTENT_ROOT/trustsr-phase2b3a-checkpoints" \
  "$PHASE2B3A_A2_MANIFEST_BASENAME"
```

New A1 checkpoint builds and all A2 checkpoint builds require
`uint16_saturate_10000_divide_10000_v2` in their current runtime evidence. The sole legacy restore
exception is the exact accepted historical A1 identity above, under its verified producer lineage.
That legacy checkpoint can recover data and models, but cannot be re-checkpointed or presented as
current A1 evidence. If a compute session pauses after `A1_V2_CHECKPOINTED`, resume from the exact
new v2 A1 checkpoint, never from the historical exception.

The expected scientific schemas are:

- A1 result: `trustsr.phase2b3a-development-smoke.v2`
- A1 cache audit: `trustsr.phase2b3a-development-smoke-cache-audit.v2`
- A1 acceptance: `trustsr.phase2b3a-development-smoke-acceptance.v2`
- A1 runtime: `trustsr.phase2b3a-a1-runtime.v2`
- A1 replay: `trustsr.phase2b3a-a1-replay.v2`
- A1 bundle manifest: `trustsr.phase2b3a-bundle-manifest.v2`
- A2 result: `trustsr.phase2b3a-development-score-audit.v1`
- A2 cache audit: `trustsr.phase2b3a-development-score-cache-audit.v1`
- A2 acceptance: `trustsr.phase2b3a-development-score-acceptance.v1`
- A2 runtime: `trustsr.phase2b3a-a2-runtime.v1`
- A2 replay: `trustsr.phase2b3a-a2-replay.v1`
- A2 bundle manifest: `trustsr.phase2b3a-bundle-manifest.v1`

Both stages require top-level `normalization_policy`, per-sample `radiometric_saturation`, and the
derived `radiometric_policy`; A2 retains v1 only because no A2 evidence was previously published.
Existing tracked Phase 2B2-A evidence and the accepted A1 v1 publication remain byte-for-byte
historical and must not be overwritten or relabelled.

After the A2 checkpoint verifies, pull and offline-verify its v1 bundle into a second new local
destination. The A1-v2 bundle was already captured before A2 replaced the live manifest. Verify
both local bundle manifests again before staging any allowlisted JSON:

```bash
set -euo pipefail
: "${PHASE2B3A_LOCAL_REPOSITORY:?set clean local reviewed checkout}"
: "${PHASE2B3A_LOCAL_BRANCH:?set intended attached local evidence branch}"
: "${PHASE2B3A_LOCAL_REVIEWED_COMMIT:?set exact reviewed 40-hex A0 commit}"
: "${PHASE2B3A_REMOTE_STORAGE_ROOT:?set remote storage root only for this pull}"
: "${PHASE2B3A_SSH_HOST:?set operator-provided endpoint only in this shell}"
: "${PHASE2B3A_SSH_PORT:?set operator-provided numeric port only in this shell}"
: "${PHASE2B3A_A1_V2_BUNDLE:?set previously verified A1-v2 bundle destination}"
: "${PHASE2B3A_A2_V1_BUNDLE:?set new absolute local A2-v1 bundle destination}"
[[ "$PHASE2B3A_LOCAL_REVIEWED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
PHASE2B3A_LOCAL_REPOSITORY="$(realpath -e -- "$PHASE2B3A_LOCAL_REPOSITORY")"
test "$(git -C "$PHASE2B3A_LOCAL_REPOSITORY" symbolic-ref --short HEAD)" = \
  "$PHASE2B3A_LOCAL_BRANCH"
git -C "$PHASE2B3A_LOCAL_REPOSITORY" merge-base --is-ancestor \
  "$PHASE2B3A_LOCAL_REVIEWED_COMMIT" HEAD
cd -- "$PHASE2B3A_LOCAL_REPOSITORY"
a1_result=artifacts/phase2b3a/sen2naipv2-development-smoke-v2.json
a1_audit=artifacts/phase2b3a/sen2naipv2-development-smoke-cache-audit-v2.json
a1_acceptance=artifacts/phase2b3a/sen2naipv2-development-smoke-acceptance-v2.json
expected_a1_status="$(printf '?? %s\n' "$a1_result" "$a1_audit" "$a1_acceptance" | sort)"
test "$(git status --porcelain=v1 --untracked-files=all | sort)" = "$expected_a1_status"
test -z "$(git diff --cached --name-only)"
for path in "$a1_result" "$a1_audit" "$a1_acceptance"; do
  test -f "$path" && test ! -L "$path"
done
a2_result=artifacts/phase2b3a/sen2naipv2-development-score-audit-v1.json
a2_audit=artifacts/phase2b3a/sen2naipv2-development-score-cache-audit-v1.json
a2_acceptance=artifacts/phase2b3a/sen2naipv2-development-score-acceptance-v1.json
for path in "$a2_result" "$a2_audit" "$a2_acceptance"; do
  test ! -e "$path" && test ! -L "$path"
done
scripts/phase2b3a/pull_results.sh "$PHASE2B3A_SSH_HOST" "$PHASE2B3A_SSH_PORT" \
  "$PHASE2B3A_REMOTE_STORAGE_ROOT" "$PHASE2B3A_A2_V1_BUNDLE"
uv run trustsr-phase2b3a-verify a2 --bundle "$PHASE2B3A_A2_V1_BUNDLE" \
  --output "$a2_acceptance"
uv run trustsr-phase2b3a-verify a1 --bundle "$PHASE2B3A_A1_V2_BUNDLE" \
  --output "$a1_acceptance"
cp -- "$PHASE2B3A_A2_V1_BUNDLE/phase2b3a-a2-result.json" "$a2_result"
cp -- "$PHASE2B3A_A2_V1_BUNDLE/phase2b3a-a2-cache-audit.json" "$a2_audit"
expected_all_status="$(printf '?? %s\n' \
  "$a1_result" "$a1_audit" "$a1_acceptance" \
  "$a2_result" "$a2_audit" "$a2_acceptance" | sort)"
test "$(git status --porcelain=v1 --untracked-files=all | sort)" = "$expected_all_status"
git add -- "$a1_result" "$a1_audit" "$a1_acceptance" \
  "$a2_result" "$a2_audit" "$a2_acceptance"
expected_staged="$(printf '%s\n' \
  "$a1_result" "$a1_audit" "$a1_acceptance" \
  "$a2_result" "$a2_audit" "$a2_acceptance" | sort)"
test "$(git diff --cached --name-only | sort)" = "$expected_staged"
test -z "$(git diff --name-only)"
git diff --cached --check
```

Inspect each pulled `phase2b3a-bundle-manifest.json`, confirm the schemas above, and review the
six-file staged diff before commit. Commit, push, and prove local and remote branch SHAs match.
Cloud pixels, tensors, caches, checkpoint archives, models, logs, remote markers, endpoints,
credentials, and host runtime manifests are never downloaded or committed.

## Post-publication LDSR worker benchmark

The saturation-v2 evidence published at `b386d4b38c9f3725107eed178829955d442f5601` is complete and
remains the audit baseline. A later performance-only run may benchmark one, two, three, and four
LDSR workers. Omission of `--ldsr-workers` remains compatible with the default of one. The option
is valid only for the exact `development` compute stage; preflight, single, smoke, replay, and
development-replay must omit it.

Run each candidate once from the same independently verified checkpoint in a fresh, isolated
disposable workspace so result paths, caches, locks, and `development.jsonl` cannot collide. Pin
the same reviewed commit, frozen inputs, and model inventories for every candidate. In that
candidate's operator shell, invoke the existing runner directly:

```bash
workers=4
"$PHASE2B3A_REPOSITORY/scripts/phase2b3a/run_cloud.sh" \
  "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_STORAGE_ROOT" "$PHASE2B3A_REPOSITORY" \
  development "${PHASE2B3A_COMMON_ARGS[@]}" "${PHASE2B3A_COMPUTE_ONLY_ARGS[@]}" \
  --ldsr-workers "$workers"
phase2b3a_replay development-replay
```

Benchmark `workers=1`, `2`, `3`, and `4`; record elapsed time and peak GPU memory for each isolated
run. Promote a worker count only when every per-prediction SHA matches the one-worker baseline,
the replay remains byte-identical, and measured GPU memory retains operational headroom. Prefer
four workers when all gates pass; if four is unsafe, fall back to three. Keep one worker as the
compatibility default and use it whenever no benchmarked parallel count passes every gate.
