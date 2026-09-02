# Phase 2B3-A staged cloud runbook

Phase 2B3-A audits score candidates on the frozen `development` split: A1 is the four-ROI
stability/resource gate and A2 is the exact 120-ROI audit. Neither is calibration or
`internal_test` evaluation. The disposable work mount holds live pixels, tensors, caches, logs,
and the checkpointed `trustsr` trees. The durable mount holds immutable checkpoint pairs and
model sources. GitHub is the durable source for code and Git history.

No command below installs software, downloads data, or controls an instance. Instance-identifying
values stay in the operator shell and never enter tracked files.

## A0: local review and immutable handoff

A0 requires no GPU or cloud connection. Commit, review, and push the exact local code before A1; do
not use an uncommitted, detached, or different checkout.

```bash
uv run pytest -q
uv run ruff check .
bash -n scripts/phase2b3a/*.sh
uv run trustsr-phase2b3a --help >/dev/null
PYTHONPATH=src uv run python -m trustsr.artifacts.workspace_checkpoint --help >/dev/null
git diff --check
git status --short --branch
git rev-parse HEAD
```

The A0 pin is the exact reviewed and pushed commit, not a branch name or abbreviated revision. Set
these runtime-only values in the operator shell:

```bash
: "${PHASE2B3A_WORK_ROOT:?set disposable mounted work root}"
: "${PHASE2B3A_PERSISTENT_ROOT:?set durable mounted checkpoint/model root}"
: "${PHASE2B3A_REPOSITORY:?set clean attached checkout below work root}"
: "${PHASE2B3A_BASE_PYTHON:?set existing cloud base Python}"
: "${PHASE2B3A_REVIEWED_COMMIT:?set exact reviewed and pushed 40-hex SHA}"
: "${PHASE2B3A_SEN2SRLITE_SOURCE:?set durable SEN2SRLite model directory}"
: "${PHASE2B3A_LDSR_SOURCE:?set durable LDSR-S2 model directory}"
PHASE2B3A_STORAGE_ROOT="$PHASE2B3A_WORK_ROOT"
PHASE2B3A_SEN2SRLITE_DIR="$PHASE2B3A_WORK_ROOT/model-mounts/sen2srlite"
PHASE2B3A_LDSR_DIR="$PHASE2B3A_WORK_ROOT/model-mounts/ldsr-s2"
```

The roots must be distinct mountpoints; the repository belongs below work and model sources below
durable storage. Before mutation, validate the mounts, attached clean A0 checkout, capacity, and
model-source containment:

```bash
[[ "$PHASE2B3A_REVIEWED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$PHASE2B3A_REPOSITORY" == "$PHASE2B3A_WORK_ROOT"/* ]]
[[ "$PHASE2B3A_SEN2SRLITE_SOURCE" == "$PHASE2B3A_PERSISTENT_ROOT"/* ]]
[[ "$PHASE2B3A_LDSR_SOURCE" == "$PHASE2B3A_PERSISTENT_ROOT"/* ]]
mountpoint -q -- "$PHASE2B3A_WORK_ROOT"
mountpoint -q -- "$PHASE2B3A_PERSISTENT_ROOT"
[[ "$(git -C "$PHASE2B3A_REPOSITORY" rev-parse HEAD)" == "$PHASE2B3A_REVIEWED_COMMIT" ]]
[[ -n "$(git -C "$PHASE2B3A_REPOSITORY" symbolic-ref --short HEAD)" ]]
[[ -z "$(git -C "$PHASE2B3A_REPOSITORY" status --porcelain)" ]]
df -h -- "$PHASE2B3A_WORK_ROOT" "$PHASE2B3A_PERSISTENT_ROOT"
df -Pi -- "$PHASE2B3A_WORK_ROOT" "$PHASE2B3A_PERSISTENT_ROOT"
```

The allowed session transitions are exactly:

```text
UNVERIFIED -> RESTORED -> PREFLIGHT_OK -> A1_OK -> A1_CHECKPOINTED
A1_CHECKPOINTED -> RESTORED -> PREFLIGHT_OK -> A2_OK -> A2_CHECKPOINTED
```

Only `UNVERIFIED` with no mutation, `A1_CHECKPOINTED`, and `A2_CHECKPOINTED` are safe pause
points. Stop on every failed prerequisite or transition.

## One-time inode recovery before the initial session

Do this only if the old checkout prevents inode preflight. It is limited to this one directory. Record
free inodes and prove the directory is ordinary, clean, fully identified, and recoverable from a
named remote branch or tag:

```bash
old_repo=/root/rivermind-fs/trustsr-phase1b/repo
df -Pi -- "$old_repo"
[[ -d "$old_repo" && ! -L "$old_repo" ]]
[[ "$(realpath -e -- "$old_repo")" == "$old_repo" ]]
[[ -z "$(git -C "$old_repo" status --porcelain=v1 --untracked-files=all)" ]]
old_head="$(git -C "$old_repo" rev-parse HEAD)"
[[ "$old_head" =~ ^[0-9a-f]{40}$ ]]
git -C "$old_repo" fetch --prune --tags origin
old_branch="$(git -C "$old_repo" branch -r --contains "$old_head" | sed -n '/[^[:space:]]/p')"
old_tag="$(git -C "$old_repo" tag --contains "$old_head" | sed -n '/[^[:space:]]/p')"
[[ -n "$old_branch" || -n "$old_tag" ]]
old_inode_count="$(find "$old_repo" -xdev -printf . | wc -c)"
[[ "$old_inode_count" =~ ^[0-9]+$ && "$old_inode_count" -ge 1229 ]]
```

The required count is `ceil(1024 * 1.20) = 1229`. Only after every prerequisite succeeds, record
the before value, run this exact guarded deletion, and record the after value:

```bash
[[ "$old_repo" == /root/rivermind-fs/trustsr-phase1b/repo ]]
[[ "$(realpath -e -- "$old_repo")" == "$old_repo" ]]
rm -rf --one-file-system -- "$old_repo"
df -Pi -- /root/rivermind-fs
```

It removes only a clean GitHub-reachable checkout and is recoverable by cloning the verified revision.
It never targets a model, dataset, audit, cache, or evidence directory. Stop if any prerequisite
fails; do not broaden the target or use a wildcard.

## First session: initial tree, preflight, and baseline checkpoint

Clone/fetch the reviewed project remote below work storage and attach a local branch at exact A0. The
remote location is operator-provided and must not be recorded here.

```bash
git clone --no-checkout <reviewed-project-remote> "$PHASE2B3A_REPOSITORY"
git -C "$PHASE2B3A_REPOSITORY" fetch --prune --tags origin
git -C "$PHASE2B3A_REPOSITORY" switch --create phase2b3a-a0 "$PHASE2B3A_REVIEWED_COMMIT"
[[ "$(git -C "$PHASE2B3A_REPOSITORY" rev-parse HEAD)" == "$PHASE2B3A_REVIEWED_COMMIT" ]]
[[ -n "$(git -C "$PHASE2B3A_REPOSITORY" symbolic-ref --short HEAD)" ]]
[[ -z "$(git -C "$PHASE2B3A_REPOSITORY" status --porcelain)" ]]
```

The initial verified input tree provides ordinary `trustsr/phase2b1b` and `trustsr/phase2b2a`
beneath work storage. Create the empty ordinary `trustsr/phase2b3a` tree and model targets. For an
initial uncheckpointed tree, use these four explicit bind commands; use the restore script only for
an existing checkpoint:

```bash
mkdir -p -- "$PHASE2B3A_WORK_ROOT/trustsr/phase2b3a" "$PHASE2B3A_WORK_ROOT/model-mounts"
[[ -d "$PHASE2B3A_WORK_ROOT/trustsr/phase2b3a" && ! -L "$PHASE2B3A_WORK_ROOT/trustsr/phase2b3a" ]]
[[ -z "$(find "$PHASE2B3A_WORK_ROOT/trustsr/phase2b3a" -mindepth 1 -print -quit)" ]]
mkdir -- "$PHASE2B3A_SEN2SRLITE_DIR" "$PHASE2B3A_LDSR_DIR"
mount --bind "$PHASE2B3A_SEN2SRLITE_SOURCE" "$PHASE2B3A_SEN2SRLITE_DIR"
mount -o remount,bind,ro "$PHASE2B3A_SEN2SRLITE_DIR"
mount --bind "$PHASE2B3A_LDSR_SOURCE" "$PHASE2B3A_LDSR_DIR"
mount -o remount,bind,ro "$PHASE2B3A_LDSR_DIR"
```

Use the frozen paths and separate common from compute-only arguments. Replay stages are model-free.

```bash
PHASE2B3A_POST_MANIFEST_SHA256=c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a
PHASE2B3A_INPUT_AUDIT_SHA256=fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b
PHASE2B3A_POST_MANIFEST="${PHASE2B3A_STORAGE_ROOT%/}/trustsr/phase2b1b/selections/${PHASE2B3A_POST_MANIFEST_SHA256}/samples.jsonl"
PHASE2B3A_INPUT_AUDIT="${PHASE2B3A_STORAGE_ROOT%/}/trustsr/phase2b2a/input-audits/${PHASE2B3A_POST_MANIFEST_SHA256}/phase2b2a-input-audit.json"
PHASE2B3A_COMMON_ARGS=(
  --selection-manifest "$PHASE2B3A_POST_MANIFEST"
  --selection-manifest-sha256 "$PHASE2B3A_POST_MANIFEST_SHA256"
  --input-audit "$PHASE2B3A_INPUT_AUDIT"
  --input-audit-sha256 "$PHASE2B3A_INPUT_AUDIT_SHA256"
  --reviewed-commit "$PHASE2B3A_REVIEWED_COMMIT"
  --confirm-cloud-storage
)
PHASE2B3A_COMPUTE_ONLY_ARGS=(
  --sen2srlite-model-dir "$PHASE2B3A_SEN2SRLITE_DIR"
  --ldsr-model-dir "$PHASE2B3A_LDSR_DIR"
)
phase2b3a_compute() {
  local stage="$1"
  scripts/phase2b3a/run_cloud.sh \
    "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_STORAGE_ROOT" "$PHASE2B3A_REPOSITORY" "$stage" \
    "${PHASE2B3A_COMMON_ARGS[@]}" "${PHASE2B3A_COMPUTE_ONLY_ARGS[@]}"
}
phase2b3a_replay() {
  local stage="$1"
  scripts/phase2b3a/run_cloud.sh \
    "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_STORAGE_ROOT" "$PHASE2B3A_REPOSITORY" "$stage" \
    "${PHASE2B3A_COMMON_ARGS[@]}"
}
```

Run normal preflight, then create and independently verify the full three-root `a0` checkpoint
before GPU compute. Copy its manifest basename from the sole JSON output; never glob for it.

```bash
phase2b3a_compute preflight
scripts/phase2b3a/checkpoint_workspace.sh \
  "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_WORK_ROOT" "$PHASE2B3A_PERSISTENT_ROOT" \
  "$PHASE2B3A_REPOSITORY" a0 "$PHASE2B3A_REVIEWED_COMMIT"
: "${PHASE2B3A_A0_MANIFEST_BASENAME:?copy the emitted a0 manifest basename exactly}"
PYTHONPATH="$PHASE2B3A_REPOSITORY/src" "$PHASE2B3A_BASE_PYTHON" -m trustsr.artifacts.workspace_checkpoint \
  verify "$PHASE2B3A_PERSISTENT_ROOT/trustsr-phase2b3a-checkpoints" "$PHASE2B3A_A0_MANIFEST_BASENAME"
```

## Restore a later session

Begin only from `A1_CHECKPOINTED` or `A2_CHECKPOINTED`. Copy the desired recorded manifest
basename and run the explicit restore command; it verifies the durable pair, establishes read-only
model binds, and restores `trustsr` into an empty work destination. Never glob for the newest file.

```bash
: "${PHASE2B3A_MANIFEST_BASENAME:?copy one recorded checkpoint manifest basename exactly}"
scripts/phase2b3a/restore_workspace.sh \
  "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_WORK_ROOT" "$PHASE2B3A_PERSISTENT_ROOT" \
  "$PHASE2B3A_REPOSITORY" "$PHASE2B3A_MANIFEST_BASENAME" \
  "$PHASE2B3A_SEN2SRLITE_SOURCE" "$PHASE2B3A_LDSR_SOURCE"
```

The successful command yields `RESTORED`; rerun `phase2b3a_compute preflight` before the next
stage.

## A1 and A2 stage shutdown

From `PREFLIGHT_OK`, A1 runs in order:

```bash
phase2b3a_compute single
phase2b3a_compute smoke
phase2b3a_replay replay
```

For A2, restore the A1 checkpoint, rerun preflight, then run:

```bash
phase2b3a_compute development
phase2b3a_replay development-replay
```

Stop on any digest, checkout, model, cache, resource, scientific-decision, or replay failure. Do not
weaken thresholds, change inputs/seeds, or copy partial results. For either completed stage, shutdown
order is exact: verify the scientific stage; pull, locally verify, review, commit only allowlisted
JSON evidence; push Git and verify the remote SHA; checkpoint the full `trustsr` tree; independently
reverify the named persistent pair; then the user may pause.

For A1, use a new private local bundle destination, verify its evidence, then commit and push only
the three allowlisted JSON files before checkpointing:

```bash
: "${PHASE2B3A_A1_BUNDLE:?set new absolute local A1 bundle destination}"
scripts/phase2b3a/pull_results.sh \
  "$PHASE2B3A_SSH_HOST" "$PHASE2B3A_SSH_PORT" "$PHASE2B3A_STORAGE_ROOT" "$PHASE2B3A_A1_BUNDLE"
uv run trustsr-phase2b3a-verify a1 --bundle "$PHASE2B3A_A1_BUNDLE" \
  --output artifacts/phase2b3a/sen2naipv2-development-smoke-acceptance-v1.json
cp -- "$PHASE2B3A_A1_BUNDLE/phase2b3a-a1-result.json" \
  artifacts/phase2b3a/sen2naipv2-development-smoke-v1.json
cp -- "$PHASE2B3A_A1_BUNDLE/phase2b3a-a1-cache-audit.json" \
  artifacts/phase2b3a/sen2naipv2-development-smoke-cache-audit-v1.json
git add -- artifacts/phase2b3a/sen2naipv2-development-smoke-*-v1.json
git commit -m "docs: record phase2b3a A1 evidence"
git push
stage_local_sha="$(git rev-parse HEAD)"
stage_remote_sha="$(git ls-remote --heads origin "$(git branch --show-current)" | awk '{print $1}')"
test "$stage_local_sha" = "$stage_remote_sha"
scripts/phase2b3a/checkpoint_workspace.sh \
  "$PHASE2B3A_BASE_PYTHON" \
  "$PHASE2B3A_WORK_ROOT" \
  "$PHASE2B3A_PERSISTENT_ROOT" \
  "$PHASE2B3A_REPOSITORY" \
  a1 \
  "$PHASE2B3A_REVIEWED_COMMIT"
: "${PHASE2B3A_A1_MANIFEST_BASENAME:?copy the emitted a1 manifest basename exactly}"
PYTHONPATH="$PHASE2B3A_REPOSITORY/src" "$PHASE2B3A_BASE_PYTHON" -m trustsr.artifacts.workspace_checkpoint \
  verify "$PHASE2B3A_PERSISTENT_ROOT/trustsr-phase2b3a-checkpoints" "$PHASE2B3A_A1_MANIFEST_BASENAME"
```

For A2, use a distinct new bundle destination; verify, commit, push, and verify the remote SHA for
only its allowlisted JSON evidence before checkpointing:

```bash
: "${PHASE2B3A_A2_BUNDLE:?set new absolute local A2 bundle destination}"
scripts/phase2b3a/pull_results.sh \
  "$PHASE2B3A_SSH_HOST" "$PHASE2B3A_SSH_PORT" "$PHASE2B3A_STORAGE_ROOT" "$PHASE2B3A_A2_BUNDLE"
uv run trustsr-phase2b3a-verify a2 --bundle "$PHASE2B3A_A2_BUNDLE" \
  --output artifacts/phase2b3a/sen2naipv2-development-score-acceptance-v1.json
cp -- "$PHASE2B3A_A2_BUNDLE/phase2b3a-a2-result.json" \
  artifacts/phase2b3a/sen2naipv2-development-score-audit-v1.json
cp -- "$PHASE2B3A_A2_BUNDLE/phase2b3a-a2-cache-audit.json" \
  artifacts/phase2b3a/sen2naipv2-development-score-cache-audit-v1.json
git add -- artifacts/phase2b3a/sen2naipv2-development-score-*-v1.json
git commit -m "docs: record phase2b3a A2 evidence"
git push
stage_local_sha="$(git rev-parse HEAD)"
stage_remote_sha="$(git ls-remote --heads origin "$(git branch --show-current)" | awk '{print $1}')"
test "$stage_local_sha" = "$stage_remote_sha"
scripts/phase2b3a/checkpoint_workspace.sh \
  "$PHASE2B3A_BASE_PYTHON" \
  "$PHASE2B3A_WORK_ROOT" \
  "$PHASE2B3A_PERSISTENT_ROOT" \
  "$PHASE2B3A_REPOSITORY" \
  a2 \
  "$PHASE2B3A_REVIEWED_COMMIT"
: "${PHASE2B3A_A2_MANIFEST_BASENAME:?copy the emitted a2 manifest basename exactly}"
PYTHONPATH="$PHASE2B3A_REPOSITORY/src" "$PHASE2B3A_BASE_PYTHON" -m trustsr.artifacts.workspace_checkpoint \
  verify "$PHASE2B3A_PERSISTENT_ROOT/trustsr-phase2b3a-checkpoints" "$PHASE2B3A_A2_MANIFEST_BASENAME"
```

A complete reverified A1 checkpoint establishes `A1_CHECKPOINTED`; a complete reverified A2
checkpoint establishes `A2_CHECKPOINTED`. The scripts never pause or shut down an instance.

Cloud pixels, tensors, caches, checkpoint tar files, models, logs, remote markers, endpoints, and
credentials are never downloaded locally or committed. Only allowlisted host-free JSON evidence
enters Git.
