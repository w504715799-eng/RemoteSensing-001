# Phase 2B3-A staged cloud runbook

Phase 2B3-A audits the frozen `development` split. A1 is the four-ROI stability/resource gate and
A2 is the exact 120-ROI audit; neither is calibration or `internal_test` evaluation. The disposable
work mount holds live pixels, tensors, caches, logs, and the checkpointed `trustsr` trees. The
durable mount holds immutable checkpoint pairs and durable model sources. GitHub is the durable
source for code and Git history. Local development, review, and synthetic/CPU verification finish
before the rented GPU instance is started. On the current unprivileged cloud container, model
restore uses an explicit verified copy into disposable storage; bind mode is allowed only when the
provider explicitly grants mount capability.

No command below installs software, downloads data, or controls an instance. Instance-identifying
values remain in the operator shell and never enter tracked files.

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
```

The allowed compute-session transitions are exactly:

```text
UNVERIFIED -> RESTORED -> PREFLIGHT_OK -> A1_OK -> A1_CHECKPOINTED
A1_CHECKPOINTED -> RESTORED -> PREFLIGHT_OK -> A2_OK -> A2_CHECKPOINTED
```

Only `UNVERIFIED` with no mutation, `A1_CHECKPOINTED`, and `A2_CHECKPOINTED` are safe pause
points. `A2_CHECKPOINTED` is terminal for compute. A later offline inspection restore is not a
compute-state transition.

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

## Initial setup, preflight, and A0 checkpoint

The initial verified input tree provides ordinary `trustsr/phase2b1b` and `trustsr/phase2b2a`
under work storage. Create the empty ordinary `trustsr/phase2b3a` directory. For the current
unprivileged provider, copy both durable model trees through the reviewed verifier; it compares a
canonical type/mode/per-file-SHA-256 inventory before publication, removes all write permission
from the copies, and leaves the durable sources unchanged. The restore script is used once an
existing workspace checkpoint is available.

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
phase2b3a_compute preflight
```

Successful preflight establishes `PREFLIGHT_OK`. Create and independently verify the full
three-root `a0` checkpoint before GPU compute. Copy the emitted manifest basename exactly; never
glob for a newest file.

```bash
"$PHASE2B3A_REPOSITORY/scripts/phase2b3a/checkpoint_workspace.sh" "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_WORK_ROOT" "$PHASE2B3A_PERSISTENT_ROOT" "$PHASE2B3A_REPOSITORY" a0 "$PHASE2B3A_REVIEWED_COMMIT"
: "${PHASE2B3A_A0_MANIFEST_BASENAME:?copy emitted a0 manifest basename exactly}"
PYTHONPATH="$PHASE2B3A_REPOSITORY/src" "$PHASE2B3A_BASE_PYTHON" -m trustsr.artifacts.workspace_checkpoint verify "$PHASE2B3A_PERSISTENT_ROOT/trustsr-phase2b3a-checkpoints" "$PHASE2B3A_A0_MANIFEST_BASENAME"
```

## A1 and A2

For A1, in the cloud shell and from `PREFLIGHT_OK`, execute the reviewed-A0 wrappers in this exact
order. Successful scientific verification establishes `A1_OK`.

```bash
phase2b3a_compute single
phase2b3a_compute smoke
phase2b3a_replay replay
```

Then, in the **local reviewed checkout**, pull, verify, and commit only the exact allowlist; do not
run these commands in the cloud shell.

```bash
set -euo pipefail
: "${PHASE2B3A_LOCAL_REPOSITORY:?set clean local reviewed checkout}"
: "${PHASE2B3A_LOCAL_BRANCH:?set intended attached local evidence branch}"
: "${PHASE2B3A_LOCAL_REVIEWED_COMMIT:?set exact reviewed 40-hex A0 commit}"
: "${PHASE2B3A_REMOTE_STORAGE_ROOT:?set remote cloud storage root only for this pull}"
: "${PHASE2B3A_SSH_HOST:?set user-provided endpoint only for this pull}"
: "${PHASE2B3A_SSH_PORT:?set user-provided numeric port only for this pull}"
: "${PHASE2B3A_A1_BUNDLE:?set new absolute local A1 bundle destination}"
[[ -d "$PHASE2B3A_LOCAL_REPOSITORY" && ! -L "$PHASE2B3A_LOCAL_REPOSITORY" ]]
[[ "$PHASE2B3A_LOCAL_REVIEWED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
PHASE2B3A_LOCAL_REPOSITORY="$(realpath -e -- "$PHASE2B3A_LOCAL_REPOSITORY")"
[[ "$(git -C "$PHASE2B3A_LOCAL_REPOSITORY" rev-parse --show-toplevel)" == "$PHASE2B3A_LOCAL_REPOSITORY" ]]
[[ "$(git -C "$PHASE2B3A_LOCAL_REPOSITORY" symbolic-ref --short HEAD)" == "$PHASE2B3A_LOCAL_BRANCH" ]]
[[ -z "$(git -C "$PHASE2B3A_LOCAL_REPOSITORY" status --porcelain=v1 --untracked-files=all)" ]]
[[ "$(git -C "$PHASE2B3A_LOCAL_REPOSITORY" rev-parse HEAD)" =~ ^[0-9a-f]{40}$ ]]
git -C "$PHASE2B3A_LOCAL_REPOSITORY" merge-base --is-ancestor "$PHASE2B3A_LOCAL_REVIEWED_COMMIT" HEAD
cd -- "$PHASE2B3A_LOCAL_REPOSITORY"
"$PHASE2B3A_LOCAL_REPOSITORY/scripts/phase2b3a/pull_results.sh" "$PHASE2B3A_SSH_HOST" "$PHASE2B3A_SSH_PORT" "$PHASE2B3A_REMOTE_STORAGE_ROOT" "$PHASE2B3A_A1_BUNDLE"
uv run trustsr-phase2b3a-verify a1 --bundle "$PHASE2B3A_A1_BUNDLE" --output artifacts/phase2b3a/sen2naipv2-development-smoke-acceptance-v1.json
cp -- "$PHASE2B3A_A1_BUNDLE/phase2b3a-a1-result.json" artifacts/phase2b3a/sen2naipv2-development-smoke-v1.json
cp -- "$PHASE2B3A_A1_BUNDLE/phase2b3a-a1-cache-audit.json" artifacts/phase2b3a/sen2naipv2-development-smoke-cache-audit-v1.json
[[ -z "$(git diff --cached --name-only)" ]]
git add -- artifacts/phase2b3a/sen2naipv2-development-smoke-v1.json artifacts/phase2b3a/sen2naipv2-development-smoke-acceptance-v1.json artifacts/phase2b3a/sen2naipv2-development-smoke-cache-audit-v1.json
expected_a1="$(printf '%s\n' artifacts/phase2b3a/sen2naipv2-development-smoke-v1.json artifacts/phase2b3a/sen2naipv2-development-smoke-acceptance-v1.json artifacts/phase2b3a/sen2naipv2-development-smoke-cache-audit-v1.json | sort)"
[[ "$(git diff --cached --name-only | sort)" == "$expected_a1" ]]
git commit -m "docs: record phase2b3a A1 evidence"
git push
stage_local_sha="$(git rev-parse HEAD)"
stage_remote_sha="$(git ls-remote --heads origin "$(git branch --show-current)" | awk '{print $1}')"
test "$stage_local_sha" = "$stage_remote_sha"
```

Back in the **cloud shell**, checkpoint and reverify the full `trustsr` tree. Successful reverify
establishes `A1_CHECKPOINTED`, the first safe pause point.

```bash
"$PHASE2B3A_REPOSITORY/scripts/phase2b3a/checkpoint_workspace.sh" "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_WORK_ROOT" "$PHASE2B3A_PERSISTENT_ROOT" "$PHASE2B3A_REPOSITORY" a1 "$PHASE2B3A_REVIEWED_COMMIT"
: "${PHASE2B3A_A1_MANIFEST_BASENAME:?copy emitted a1 manifest basename exactly}"
PYTHONPATH="$PHASE2B3A_REPOSITORY/src" "$PHASE2B3A_BASE_PYTHON" -m trustsr.artifacts.workspace_checkpoint verify "$PHASE2B3A_PERSISTENT_ROOT/trustsr-phase2b3a-checkpoints" "$PHASE2B3A_A1_MANIFEST_BASENAME"
```

For A2, every new disposable session first runs the bootstrap above, then restores the exact A1
manifest; this successful restore establishes `RESTORED`. Rerun preflight to establish
`PREFLIGHT_OK`, then run `development` and `development-replay`. Successful scientific
verification establishes `A2_OK`.

```bash
: "${PHASE2B3A_A1_MANIFEST_BASENAME:?copy recorded a1 manifest basename exactly}"
"$PHASE2B3A_REPOSITORY/scripts/phase2b3a/restore_workspace.sh" "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_WORK_ROOT" "$PHASE2B3A_PERSISTENT_ROOT" "$PHASE2B3A_REPOSITORY" "$PHASE2B3A_A1_MANIFEST_BASENAME" "$PHASE2B3A_SEN2SRLITE_SOURCE" "$PHASE2B3A_LDSR_SOURCE" "$PHASE2B3A_MODEL_RESTORE_MODE" "$PHASE2B3A_A1_CHECKPOINT_COMMIT"
phase2b3a_compute preflight
phase2b3a_compute development
phase2b3a_replay development-replay
```

Require the restore JSON to contain `"model_restore_mode":"copy"`, the exact old
`"checkpoint_reviewed_commit":"4df5195e..."`, and the new pushed A2 code identity in
`"restore_code_commit"`. The old checkpoint commit must be an ancestor of the restore code commit;
the script fails before model publication otherwise. Do not retry with bind mode after a permission
error, and do not replace either model directory with a symlink.

Repeat the labelled **local reviewed checkout** evidence sequence for the exact A2 filenames, then
return to the cloud shell to checkpoint `a2` and reverify. A complete reverified A2 checkpoint
establishes terminal `A2_CHECKPOINTED`.

```bash
set -euo pipefail
: "${PHASE2B3A_LOCAL_REPOSITORY:?set clean local reviewed checkout}"
: "${PHASE2B3A_LOCAL_BRANCH:?set intended attached local evidence branch}"
: "${PHASE2B3A_LOCAL_REVIEWED_COMMIT:?set exact reviewed 40-hex A0 commit}"
: "${PHASE2B3A_REMOTE_STORAGE_ROOT:?set remote cloud storage root only for this pull}"
: "${PHASE2B3A_SSH_HOST:?set user-provided endpoint only for this pull}"
: "${PHASE2B3A_SSH_PORT:?set user-provided numeric port only for this pull}"
: "${PHASE2B3A_A2_BUNDLE:?set new absolute local A2 bundle destination}"
[[ -d "$PHASE2B3A_LOCAL_REPOSITORY" && ! -L "$PHASE2B3A_LOCAL_REPOSITORY" ]]
[[ "$PHASE2B3A_LOCAL_REVIEWED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
PHASE2B3A_LOCAL_REPOSITORY="$(realpath -e -- "$PHASE2B3A_LOCAL_REPOSITORY")"
[[ "$(git -C "$PHASE2B3A_LOCAL_REPOSITORY" rev-parse --show-toplevel)" == "$PHASE2B3A_LOCAL_REPOSITORY" ]]
[[ "$(git -C "$PHASE2B3A_LOCAL_REPOSITORY" symbolic-ref --short HEAD)" == "$PHASE2B3A_LOCAL_BRANCH" ]]
[[ -z "$(git -C "$PHASE2B3A_LOCAL_REPOSITORY" status --porcelain=v1 --untracked-files=all)" ]]
[[ "$(git -C "$PHASE2B3A_LOCAL_REPOSITORY" rev-parse HEAD)" =~ ^[0-9a-f]{40}$ ]]
git -C "$PHASE2B3A_LOCAL_REPOSITORY" merge-base --is-ancestor "$PHASE2B3A_LOCAL_REVIEWED_COMMIT" HEAD
cd -- "$PHASE2B3A_LOCAL_REPOSITORY"
"$PHASE2B3A_LOCAL_REPOSITORY/scripts/phase2b3a/pull_results.sh" "$PHASE2B3A_SSH_HOST" "$PHASE2B3A_SSH_PORT" "$PHASE2B3A_REMOTE_STORAGE_ROOT" "$PHASE2B3A_A2_BUNDLE"
uv run trustsr-phase2b3a-verify a2 --bundle "$PHASE2B3A_A2_BUNDLE" --output artifacts/phase2b3a/sen2naipv2-development-score-acceptance-v1.json
cp -- "$PHASE2B3A_A2_BUNDLE/phase2b3a-a2-result.json" artifacts/phase2b3a/sen2naipv2-development-score-audit-v1.json
cp -- "$PHASE2B3A_A2_BUNDLE/phase2b3a-a2-cache-audit.json" artifacts/phase2b3a/sen2naipv2-development-score-cache-audit-v1.json
[[ -z "$(git diff --cached --name-only)" ]]
git add -- artifacts/phase2b3a/sen2naipv2-development-score-audit-v1.json artifacts/phase2b3a/sen2naipv2-development-score-acceptance-v1.json artifacts/phase2b3a/sen2naipv2-development-score-cache-audit-v1.json
expected_a2="$(printf '%s\n' artifacts/phase2b3a/sen2naipv2-development-score-audit-v1.json artifacts/phase2b3a/sen2naipv2-development-score-acceptance-v1.json artifacts/phase2b3a/sen2naipv2-development-score-cache-audit-v1.json | sort)"
[[ "$(git diff --cached --name-only | sort)" == "$expected_a2" ]]
git commit -m "docs: record phase2b3a A2 evidence"
git push
stage_local_sha="$(git rev-parse HEAD)"
stage_remote_sha="$(git ls-remote --heads origin "$(git branch --show-current)" | awk '{print $1}')"
test "$stage_local_sha" = "$stage_remote_sha"
```

```bash
"$PHASE2B3A_REPOSITORY/scripts/phase2b3a/checkpoint_workspace.sh" "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_WORK_ROOT" "$PHASE2B3A_PERSISTENT_ROOT" "$PHASE2B3A_REPOSITORY" a2 "$PHASE2B3A_REVIEWED_COMMIT"
: "${PHASE2B3A_A2_MANIFEST_BASENAME:?copy emitted a2 manifest basename exactly}"
PYTHONPATH="$PHASE2B3A_REPOSITORY/src" "$PHASE2B3A_BASE_PYTHON" -m trustsr.artifacts.workspace_checkpoint verify "$PHASE2B3A_PERSISTENT_ROOT/trustsr-phase2b3a-checkpoints" "$PHASE2B3A_A2_MANIFEST_BASENAME"
```

Cloud pixels, tensors, caches, checkpoint tar files, models, logs, remote markers, endpoints, and
credentials are never downloaded locally or committed. Only allowlisted host-free JSON evidence
enters Git.
