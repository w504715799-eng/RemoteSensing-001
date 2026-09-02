# Phase 2B3-A staged cloud runbook

Phase 2B3-A audits score candidates on the frozen `development` split. A1 is a four-ROI
stability/resource gate; A2 is the exact 120-ROI development audit. Neither phase is a calibration
or `internal_test` evaluation. Large manifests, pixels, predictions, score tensors, model assets,
logs, remote pair commit markers, and lock files remain on persistent cloud storage.

No command below creates an environment, installs a package, downloads data, or controls the cloud
instance. Use only the already reviewed cloud base Python and already verified assets. Values that
identify the current instance or checkout stay in the operator's shell and never enter tracked
files.

## A0: local review and immutable handoff

A0 requires no GPU or network connection to a cloud instance. From the local reviewed checkout,
complete the quality gate, commit it, review it, push that exact commit through the normal project
workflow, and record its full lowercase SHA. Do not start A1 from an uncommitted, detached, or
different checkout.

```bash
uv run pytest -q
uv run ruff check .
bash -n scripts/phase2b3a/run_cloud.sh scripts/phase2b3a/pull_results.sh
uv run trustsr-phase2b3a --help
uv run trustsr-phase2b3a-verify --help
git status --short --branch
git rev-parse HEAD
```

On the cloud instance, check out that reviewed/pushed A0 commit and set these runtime-only values.
`PHASE2B3A_REVIEWED_COMMIT` is the pin: copy the exact reviewed/pushed A0 SHA, not a branch name or
an abbreviated revision.

```bash
: "${PHASE2B3A_STORAGE_ROOT:?set mounted persistent root}"
: "${PHASE2B3A_REPOSITORY:?set reviewed checkout}"
: "${PHASE2B3A_BASE_PYTHON:?set cloud base Python}"
: "${PHASE2B3A_REVIEWED_COMMIT:?set exact reviewed and pushed A0 40-hex SHA}"
: "${PHASE2B3A_SEN2SRLITE_DIR:?set verified SEN2SRLite model directory}"
: "${PHASE2B3A_LDSR_DIR:?set verified LDSR-S2 model directory}"

[[ "$PHASE2B3A_REVIEWED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$(git -C "$PHASE2B3A_REPOSITORY" rev-parse HEAD)" == "$PHASE2B3A_REVIEWED_COMMIT" ]]
[[ -n "$(git -C "$PHASE2B3A_REPOSITORY" symbolic-ref --short HEAD)" ]]
[[ -z "$(git -C "$PHASE2B3A_REPOSITORY" status --porcelain)" ]]
mountpoint -q -- "$PHASE2B3A_STORAGE_ROOT"
df -h -- "$PHASE2B3A_STORAGE_ROOT"
df -ih -- "$PHASE2B3A_STORAGE_ROOT"
```

Freeze the upstream locations and keep common non-model arguments separate from compute-only model
arguments. Replay stages intentionally receive the common array only.

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
    "$PHASE2B3A_BASE_PYTHON" \
    "$PHASE2B3A_STORAGE_ROOT" \
    "$PHASE2B3A_REPOSITORY" \
    "$stage" \
    "${PHASE2B3A_COMMON_ARGS[@]}" \
    "${PHASE2B3A_COMPUTE_ONLY_ARGS[@]}"
}

phase2b3a_replay() {
  local stage="$1"
  scripts/phase2b3a/run_cloud.sh \
    "$PHASE2B3A_BASE_PYTHON" \
    "$PHASE2B3A_STORAGE_ROOT" \
    "$PHASE2B3A_REPOSITORY" \
    "$stage" \
    "${PHASE2B3A_COMMON_ARGS[@]}"
}
```

The runner resolves all paths without symlinks, requires the storage root to be a mountpoint with at
least 10 GiB free and more than 1024 free inodes, and verifies the pinned commit against clean,
attached `HEAD` before invoking Python. Each successful stage creates exactly one immutable log at
`trustsr/phase2b3a/logs/<stage>.jsonl`. An existing log is a collision: inspect it and choose whether
the old successful stage is authoritative; never delete or overwrite it merely to force a rerun.
The runner atomically reserves that name with a sibling `<stage>.jsonl.lock` directory before Python
starts and publishes through a no-replace hard link. Concurrent invocations therefore cannot both
run the CLI or replace an existing log.

An ordinary failure removes its temporary log and reservation. Process death can leave a stale lock;
the runner fails closed instead of guessing that it is stale. Recovery requires an operator to prove
that no matching runner or Python process exists, inspect whether the final stage log or a
`.stage.XXXXXX` temporary remains, preserve any completed evidence, and only then remove the exact
empty lock with `rmdir -- "$PHASE2B3A_STORAGE_ROOT/trustsr/phase2b3a/logs/<stage>.jsonl.lock"`.
Never recursively remove the log directory or clear a live reservation.

## A1: four-ROI stability and resource gate

Run each command only after the preceding command succeeds:

```bash
phase2b3a_compute preflight
phase2b3a_compute single
phase2b3a_compute smoke
phase2b3a_replay replay
```

Stop immediately on a manifest/audit digest mismatch, a dirty or mismatched checkout, an invalid
model asset, a foreign GPU process, non-repeatable single prediction, invalid cache entry, OOM,
non-finite output, replay mismatch, or failed resource integrity check. Do not weaken a threshold,
change samples/seeds, install into the base environment, or substitute another checkout. A failed
K=5 statistical stability decision removes only `ldsr_variance_k5` from A2; an integrity/resource
failure stops the workflow.

After `replay` reports `byte_identical=true`, pull and verify the A1 bundle from the local reviewed
checkout. Use a new, private local destination whose parent already exists.

```bash
: "${PHASE2B3A_SSH_HOST:?set current user-provided SSH user@host only when pulling}"
: "${PHASE2B3A_SSH_PORT:?set current user-provided numeric SSH port only when pulling}"
: "${PHASE2B3A_STORAGE_ROOT:?set mounted persistent root}"
: "${PHASE2B3A_A1_BUNDLE:?set new absolute local A1 bundle destination}"

scripts/phase2b3a/pull_results.sh \
  "$PHASE2B3A_SSH_HOST" "$PHASE2B3A_SSH_PORT" \
  "$PHASE2B3A_STORAGE_ROOT" "$PHASE2B3A_A1_BUNDLE"

mkdir -p artifacts/phase2b3a
uv run trustsr-phase2b3a-verify a1 \
  --bundle "$PHASE2B3A_A1_BUNDLE" \
  --output artifacts/phase2b3a/sen2naipv2-development-smoke-acceptance-v1.json
cp -- "$PHASE2B3A_A1_BUNDLE/phase2b3a-a1-result.json" \
  artifacts/phase2b3a/sen2naipv2-development-smoke-v1.json
cp -- "$PHASE2B3A_A1_BUNDLE/phase2b3a-a1-cache-audit.json" \
  artifacts/phase2b3a/sen2naipv2-development-smoke-cache-audit-v1.json
sha256sum -- artifacts/phase2b3a/sen2naipv2-development-smoke-*-v1.json
```

The puller first stages the digest-addressed completed bundle manifest, accepts only the four files
listed for its declared phase, checks their remote and local sizes/SHA-256 values, and publishes the
five-file bundle only after all checks pass. It does not enumerate the result directory and therefore
does not copy remote pair commit markers, lock files, caches, logs, or temporary files.
The remote storage root is deliberately narrower than a general shell path: every component may use
only ASCII letters, digits, `.`, `_`, and `-`, and may not be empty, `.`, `..`, or begin with `-`.
This is required because OpenSSH and legacy SCP serialize remote commands through a shell. The remote
metadata probe explicitly uses Bash; punctuation-heavy mount paths must be rejected, not quoted into
remote commands.

The puller also reserves `${PHASE2B3A_A1_BUNDLE}.lock` (or the corresponding A2 destination) and
publishes with no directory following and no replacement. A concurrent winner is accepted only if
its exact five-file membership and every byte match the fully verified staging bundle. Ordinary
failures remove the staging directory and lock. After process death, leave a stale lock in place
until no puller, SSH, or SCP process exists and the destination has been independently verified or
shown absent; then remove only the exact empty sibling lock with
`rmdir -- "${PHASE2B3A_A1_BUNDLE}.lock"` (or the guarded A2 destination lock).

Do not begin A2 until the local A1 verifier succeeds and the acceptance JSON confirms the resource
gate. If A1 verification fails, keep the remote instance and persistent evidence available for
diagnosis. Once A1 verification succeeds, no stage process remains, and `nvidia-smi` reports no
compute PID, tell the user that the GPU can be paused while A2 is reviewed or scheduled; the scripts
never pause or shut down the instance.

## A2: exact 120-ROI development audit

Resume the same persistent storage and the exact pinned reviewed commit. Do not edit either argument
array. The development stage consumes A1 acceptance and either one or five LDSR seeds exactly as the
A1 decision requires.

```bash
phase2b3a_compute development
phase2b3a_replay development-replay
```

Stop on any A1 reference mismatch, sample/stratum count other than 120 and 12×10, cache corruption,
unexpected calibration/`internal_test` evidence, compute failure, or non-byte-identical replay. Do
not copy partial results or the remote internal pair commit marker.

After `development-replay` succeeds, use a different new local destination. The completed remote
bundle manifest now declares phase A2 and exactly four A2 evidence files.

```bash
: "${PHASE2B3A_SSH_HOST:?set current user-provided SSH user@host only when pulling}"
: "${PHASE2B3A_SSH_PORT:?set current user-provided numeric SSH port only when pulling}"
: "${PHASE2B3A_STORAGE_ROOT:?set mounted persistent root}"
: "${PHASE2B3A_A2_BUNDLE:?set new absolute local A2 bundle destination}"

scripts/phase2b3a/pull_results.sh \
  "$PHASE2B3A_SSH_HOST" "$PHASE2B3A_SSH_PORT" \
  "$PHASE2B3A_STORAGE_ROOT" "$PHASE2B3A_A2_BUNDLE"

uv run trustsr-phase2b3a-verify a2 \
  --bundle "$PHASE2B3A_A2_BUNDLE" \
  --output artifacts/phase2b3a/sen2naipv2-development-score-acceptance-v1.json
cp -- "$PHASE2B3A_A2_BUNDLE/phase2b3a-a2-result.json" \
  artifacts/phase2b3a/sen2naipv2-development-score-audit-v1.json
cp -- "$PHASE2B3A_A2_BUNDLE/phase2b3a-a2-cache-audit.json" \
  artifacts/phase2b3a/sen2naipv2-development-score-cache-audit-v1.json
sha256sum -- artifacts/phase2b3a/sen2naipv2-development-score-*-v1.json
```

Only the six allowlisted host-free JSON files under `artifacts/phase2b3a/` may enter Git. Never
transfer local pixels, the 360-row manifest, GeoTIFFs, predictions, score tensors, models, runtime
logs, SSH configuration, or credentials. After the local A2 verifier and local regression suite
succeed, all expected digests are recorded, no cloud stage is running, and `nvidia-smi` reports no
compute PID, tell the user the GPU can be paused. Pausing remains a user-controlled cloud action.
