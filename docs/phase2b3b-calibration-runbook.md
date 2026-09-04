# Phase 2B3-B formal calibration runbook

This runbook executes the one preregistered calibration operating point:

- `alpha = 0.05`;
- minimum aggregate calibration pixel coverage `0.10`.

Neither value is a CLI option. The formal result composer, replay verifier, acceptance builder, and
independent verifier reject every other value. Do not edit the constants or relax the gate after
observing calibration.

Phase 2B3-B may read only the 120 frozen `calibration` ROIs. It must never read any
`internal_test` pixel, prediction, cache, score, risk, or metric. Phase 2B3-C remains out of scope.

## 1. Local CPU readiness gate

Run this gate from the final clean attached `main` checkout before requesting GPU startup:

```bash
uv run pytest -q \
  tests/cli/test_phase2b3b.py \
  tests/cli/test_phase2b3b_verify.py \
  tests/evaluation/test_phase2b3b_*.py \
  tests/evaluation/test_calibration_*.py \
  tests/data/test_calibration_subset.py \
  tests/data/test_calibration_pairs.py \
  tests/calibration/test_conformal.py
uv run ruff check \
  src/trustsr/cli/phase2b3b.py \
  src/trustsr/cli/phase2b3b_verify.py \
  src/trustsr/evaluation \
  tests/cli/test_phase2b3b.py \
  tests/cli/test_phase2b3b_verify.py \
  tests/evaluation
uv run python -m compileall -q src
uv run trustsr-phase2b3b --help >/dev/null
uv run trustsr-phase2b3b preflight --help >/dev/null
uv run trustsr-phase2b3b calibration --help >/dev/null
uv run trustsr-phase2b3b calibration-replay --help >/dev/null
uv run trustsr-phase2b3b-verify --help >/dev/null
git diff --check
git status --short --branch
PHASE2B3B_REVIEWED_COMMIT=$(git rev-parse HEAD)
```

`git status` must be clean and attached to `main`. Record the exact
`PHASE2B3B_REVIEWED_COMMIT`; the command itself revalidates that the checkout is clean, attached,
and descended from both frozen Phase 2B3-A trust anchors.

The checks above use generated metadata and tiny CPU tensors only. They do not authorize GPU
startup or a real calibration run.

## 2. Fixed paths

Define task-specific variables on the machine holding the persistent SEN2NAIPv2 assets and caches:

```bash
PHASE2B3B_PROJECT_ROOT=/absolute/path/to/clean/RemoteSensing001
PHASE2B3B_STORAGE_ROOT=/absolute/path/to/persistent/storage
PHASE2B3B_EVIDENCE_DIR="$PHASE2B3B_PROJECT_ROOT/artifacts/phase2b3a"
PHASE2B3B_POST_MANIFEST_SHA256=c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a
PHASE2B3B_MANIFEST="$PHASE2B3B_STORAGE_ROOT/trustsr/phase2b1b/selections/$PHASE2B3B_POST_MANIFEST_SHA256/samples.jsonl"
PHASE2B3B_BUNDLE="$PHASE2B3B_STORAGE_ROOT/trustsr/phase2b3b/bundles/$PHASE2B3B_POST_MANIFEST_SHA256"
PHASE2B3B_COMMON_ARGS=(
  --project-root "$PHASE2B3B_PROJECT_ROOT"
  --evidence-dir "$PHASE2B3B_EVIDENCE_DIR"
  --storage-root "$PHASE2B3B_STORAGE_ROOT"
  --manifest "$PHASE2B3B_MANIFEST"
  --confirm-persistent-storage
)
```

The storage root must be an absolute canonical non-symlink directory, must not be `/` or the user
home, and must have more than 10 GiB free. The commands derive these paths and accept no alternate
layout:

```text
trustsr/phase2b3b/predictions/<post-manifest-sha256>/
trustsr/phase2b3b/scores/<post-manifest-sha256>/
trustsr/phase2b3b/bundles/<post-manifest-sha256>/
```

All formal stages share one non-blocking exclusive lock. A concurrent preflight, calibration,
replay, or independent verification attempt fails closed.

## 3. Metadata-only preflight

This command verifies Git, the six frozen Phase 2B3-A evidence files, the complete 360-row
manifest, and the exact 120-member calibration selection. It does not load image pixels or
construct a model:

```bash
cd "$PHASE2B3B_PROJECT_ROOT"
uv run trustsr-phase2b3b preflight "${PHASE2B3B_COMMON_ARGS[@]}"
```

Stop on any nonzero exit, revision mismatch, evidence mismatch, manifest mismatch, path rejection,
or capacity rejection.

## 4. CPU cache probe and formal calibration

Do not run this section until the local readiness gate has passed. The first invocation deliberately
omits the model path. It loads only the 120 frozen calibration inputs, verifies whether one coherent
set of all 600 fixed K5 prediction-cache entries exists, and never imports or constructs LDSR:

```bash
uv run trustsr-phase2b3b calibration "${PHASE2B3B_COMMON_ARGS[@]}"
```

If the complete cache exists, this CPU-only command computes the fixed K5 variance score and R9
risk, fits `alpha=0.05`, immediately reconstructs score, risk, fit, result, and cache audit without
inference, and atomically publishes the bundle after byte identity succeeds. Continue to replay;
do not start a GPU.

If it stops with `verified K5 prediction caches are missing`, stop and request the user's separate
GPU-server authorization. Only after that authorization define the verified model path and rerun:

```bash
PHASE2B3B_LDSR_MODEL_DIR=/absolute/path/to/verified/ldsr/model
uv run trustsr-phase2b3b calibration \
  "${PHASE2B3B_COMMON_ARGS[@]}" \
  --ldsr-model-dir "$PHASE2B3B_LDSR_MODEL_DIR"
```

Supplying `--ldsr-model-dir` is the only path that imports and constructs LDSR or may call
prediction. It happens only after metadata, storage, lock, calibration-input, and cache-integrity
gates pass, and it generates only missing exact K5 entries. The default is one worker; no worker
override exists.

An interruption may leave individually verified cache entries, but it must not leave an accepted
bundle. Cache entries are reusable only through their full identities.

## 5. Explicit inference-free replay

Run replay with no model path. This is a deliberate interface guarantee: the replay command does
not import or construct LDSR and cannot call prediction.

```bash
uv run trustsr-phase2b3b calibration-replay "${PHASE2B3B_COMMON_ARGS[@]}"
```

Replay reloads the authoritative calibration pairs and verified prediction/score caches, recomputes
the ensemble-variance score, R9 risk, conformal fit, result, and cache audit, and requires the
result, audit, runtime binding, replay receipt, and bundle manifest to remain byte-identical.

## 6. Copied-bundle independent verification and publication

Copy the completed bundle to a separate canonical non-symlink directory before independent
verification:

```bash
PHASE2B3B_VERIFY_PARENT=/absolute/path/to/independent/verification
PHASE2B3B_COPIED_BUNDLE="$PHASE2B3B_VERIFY_PARENT/phase2b3b-bundle"
mkdir -p "$PHASE2B3B_VERIFY_PARENT"
test ! -e "$PHASE2B3B_COPIED_BUNDLE"
cp -a -- "$PHASE2B3B_BUNDLE" "$PHASE2B3B_COPIED_BUNDLE"
uv run trustsr-phase2b3b-verify \
  --bundle "$PHASE2B3B_COPIED_BUNDLE" \
  "${PHASE2B3B_COMMON_ARGS[@]}"
```

The verifier does not trust the producing CLI. Before calibration pixel loading it independently
validates Git ancestry, upstream evidence, authoritative membership, the copied bundle allowlist,
canonical bytes, runtime inventory, and all cross-document digests. It then reconstructs exact
input/radiometry authority, independently recomputes cache-derived score, risk, fit, audit, and
result without inference, and requires byte-identical replay.

On success it atomically publishes exactly these three Git-safe files under
`artifacts/phase2b3b/`:

```text
sen2naipv2-calibration-conformal-v1.json
sen2naipv2-calibration-conformal-cache-audit-v1.json
sen2naipv2-calibration-conformal-acceptance-v1.json
```

An existing byte-identical three-file publication is reused. A partial directory, symlink, extra
file, or different byte fails closed. No runtime manifest, replay receipt, raw tensor, cache file,
model weight, host path, endpoint, credential, or log may be added to Git.

## 7. Review and decision

Review before committing:

```bash
sha256sum "$PHASE2B3B_PROJECT_ROOT"/artifacts/phase2b3b/*.json
git -C "$PHASE2B3B_PROJECT_ROOT" diff --check
git -C "$PHASE2B3B_PROJECT_ROOT" diff -- artifacts/phase2b3b
```

The acceptance record contains exactly one observed decision:

- `freeze_calibration`: the threshold is finite and aggregate calibration pixel coverage is at
  least `0.10`. Commit the three reviewed files together. The next action is a separate Phase 2B3-C
  design and approval; do not access `internal_test` yet.
- `stop_insufficient_coverage`: the result is all-abstain or coverage is below `0.10`. Commit the
  three reviewed files together as a valid negative result and stop. Do not relax alpha, coverage,
  score, seeds, risk window, or threshold, and do not design Phase 2B3-C from this result.
