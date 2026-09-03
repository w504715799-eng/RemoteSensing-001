# Phase 2B3-A Radiometric Saturation v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit, auditable saturation policy for the exact Phase 2B3-A development inputs, rerun A1 under that policy, and unblock the 120-ROI A2 audit without changing its sample set.

**Architecture:** Keep historical Phase 2B2-A/2B2-B loading on the legacy no-clip policy while Phase 2B3-A explicitly selects v2. The loader records immutable per-crop saturation statistics; Phase 2B3-A result builders aggregate them, cache identities bind the policy, and online replay plus the offline verifier independently validate the evidence.

**Tech Stack:** Python 3.12, dataclasses, NumPy, Rasterio, PyTorch, pytest, Ruff, Bash, canonical JSON.

**Spec:** `docs/superpowers/specs/2026-09-03-phase2b3a-radiometric-saturation-v2.md`

## Global Constraints

- Preserve the exact frozen 120 development ROI and all existing asset-integrity checks.
- Do not inspect calibration or `internal_test` pixels and do not alter their split membership.
- Keep tracked Phase 2B2-A and A1 publication JSON byte-for-byte unchanged.
- Legacy policy is `uint16_divide_10000_no_clip_v1`; Phase 2B3-A policy is `uint16_saturate_10000_divide_10000_v2`.
- Saturate aligned crop values above `10000`, reject raw values above `32767`, and divide by `10000.0` into contiguous CPU float32 tensors.
- Record total and B04/B03/B02/B08 per-band clipped counts for LR and HR; all aggregate evidence is derived and independently revalidated.
- Bind prediction caches, A1 v2 evidence, A2 evidence, replay, checkpoint, and offline acceptance to the v2 policy.
- No endpoint, port, credentials, host paths, pixels, tensors, models, caches, logs, or checkpoints enter Git.

---

### Task 1: Versioned loader and saturation statistics

**Files:**
- Modify: `src/trustsr/data/crosssensor_pairs.py`
- Modify: `tests/data/test_crosssensor_pairs.py`

**Interfaces:**
- Produces: `LEGACY_NORMALIZATION_POLICY`, `PHASE2B3A_NORMALIZATION_POLICY`, `RAW_RADIOMETRIC_MAX`, and frozen `RadiometricSaturation`.
- Produces: `load_crosssensor_pair(..., normalization_policy: str = LEGACY_NORMALIZATION_POLICY)`.
- Produces: `CrosssensorPairMetadata.lr_saturation` and `.hr_saturation`.
- Consumes: unchanged frozen sidecar shape/dtype/CRS/transform/nodata/min/max identity.

- [ ] **Step 1: Write failing real-GeoTIFF tests**

Add tests whose production-breaking mutations are: removing the v2 branch, clipping the wrong
window, aggregating bands in the wrong order, mutating the source, accepting `32768`, or weakening
legacy v1. Build a literal fixture with aligned-crop B04/B08 values above `10000`; assert exact
clipped tensor values and exact statistics.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest tests/data/test_crosssensor_pairs.py -q`

Expected: the new tests fail because the policy argument, v2 constants, and saturation metadata do
not exist; all pre-existing tests retain their prior behavior.

- [ ] **Step 3: Implement the minimal versioned transform**

Implement the exact interfaces above. Validate the full raw raster against `32767` for v2, compute
statistics on the aligned crop, clip a copy rather than the source array, and keep the legacy branch
byte-compatible. Reject booleans or non-built-in integers in statistics.

- [ ] **Step 4: Verify GREEN and regression scope**

Run:

```bash
uv run pytest tests/data/test_crosssensor_pairs.py tests/data/test_input_audit.py \
  tests/cli/test_phase2b2a.py tests/cli/test_phase2b2b.py -q
uv run ruff check src/trustsr/data/crosssensor_pairs.py tests/data/test_crosssensor_pairs.py
```

Expected: all pass; historical callers still select legacy v1.

- [ ] **Step 5: Commit**

```bash
git add src/trustsr/data/crosssensor_pairs.py tests/data/test_crosssensor_pairs.py
git commit -m "feat: add phase2b3a saturation input policy"
```

### Task 2: Phase 2B3-A result and cache provenance

**Files:**
- Modify: `src/trustsr/cli/phase2b3a.py`
- Modify: `src/trustsr/evaluation/development_predictions.py`
- Modify: `src/trustsr/evaluation/development_score_audit.py`
- Modify: `tests/cli/test_phase2b3a.py`
- Modify: `tests/evaluation/test_development_predictions.py`
- Modify: `tests/evaluation/test_development_score_audit.py`

**Interfaces:**
- Consumes: Task 1 v2 constants, policy argument, and `RadiometricSaturation` fields.
- Produces: Phase 2B3-A loaders always request `PHASE2B3A_NORMALIZATION_POLICY`.
- Produces: prediction provenance key `normalization_policy`.
- Produces: A1 v2 and A2 per-sample `radiometric_saturation` plus aggregate
  `radiometric_policy` with literal keys from the spec.

- [ ] **Step 1: Write failing policy-selection and evidence tests**

Assert that every Phase 2B3-A loading stage passes v2 explicitly, prediction cache keys differ by
policy, the known sample statistics aggregate to affected-sample count 1, affected-asset count 2,
LR clipped total 8, HR clipped total 117, and maximum 11968, while zero-saturation A1 fixtures remain
valid. Add tampering cases for negative counts, wrong per-band length, wrong totals, and wrong policy.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
uv run pytest tests/cli/test_phase2b3a.py \
  tests/evaluation/test_development_predictions.py \
  tests/evaluation/test_development_score_audit.py -q
```

Expected: failures identify missing v2 selection, cache provenance, schemas, and radiometric fields.

- [ ] **Step 3: Implement v2 online provenance**

Add one private serializer and one aggregate validator/builder; reuse them for A1 and A2. Bump A1
result/cache/replay/runtime/bundle schemas to v2. Keep A2 at pre-publication v1 while making the new
fields mandatory. Ensure replay rebuilds and byte-compares the same objects.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command, then:

```bash
uv run ruff check src/trustsr/cli/phase2b3a.py \
  src/trustsr/evaluation/development_predictions.py \
  src/trustsr/evaluation/development_score_audit.py \
  tests/cli/test_phase2b3a.py tests/evaluation/test_development_predictions.py \
  tests/evaluation/test_development_score_audit.py
```

- [ ] **Step 5: Commit**

```bash
git add src/trustsr/cli/phase2b3a.py src/trustsr/evaluation/development_predictions.py \
  src/trustsr/evaluation/development_score_audit.py tests/cli/test_phase2b3a.py \
  tests/evaluation/test_development_predictions.py tests/evaluation/test_development_score_audit.py
git commit -m "feat: bind phase2b3a evidence to saturation policy"
```

### Task 3: Offline verifier and checkpoint boundary

**Files:**
- Modify: `src/trustsr/cli/phase2b3a_verify.py`
- Modify: `src/trustsr/artifacts/workspace_checkpoint.py`
- Modify: `tests/cli/test_phase2b3a_verify.py`
- Modify: `tests/artifacts/test_workspace_checkpoint.py`

**Interfaces:**
- Consumes: Task 2 exact A1 v2 and A2 radiometric JSON shapes.
- Produces: offline recomputation and fail-closed validation of all radiometric aggregates.
- Produces: new A1/A2 checkpoints require the v2 runtime policy while old A1 manifests remain
  restorable only under their exact producer lineage.

- [ ] **Step 1: Write failing verifier and checkpoint tests**

Extend literal bundle fixtures with v2 A1 and required A2 fields. Add one test per malformed scalar,
wrong band vector, inconsistent aggregate, missing policy, legacy A1 presented as current A1, and v2
checkpoint runtime mismatch. Preserve existing tests that validate the immutable legacy checkpoint.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
uv run pytest tests/cli/test_phase2b3a_verify.py \
  tests/artifacts/test_workspace_checkpoint.py -q
```

- [ ] **Step 3: Implement minimal offline and checkpoint validation**

Mirror Task 2's public JSON contract without importing online private helpers. Require exact built-in
integer types, nonnegative counts, four ordered band counts summing to totals, raw min/max ordering,
raw maximum no greater than `32767`, and aggregate equality. Keep host-free allowlists unchanged.

- [ ] **Step 4: Verify GREEN**

Run the Step 2 command and Ruff on all four files.

- [ ] **Step 5: Commit**

```bash
git add src/trustsr/cli/phase2b3a_verify.py src/trustsr/artifacts/workspace_checkpoint.py \
  tests/cli/test_phase2b3a_verify.py tests/artifacts/test_workspace_checkpoint.py
git commit -m "feat: verify phase2b3a saturation provenance"
```

### Task 4: Cloud reset workflow and documentation

**Files:**
- Modify: `scripts/phase2b3a/restore_workspace.sh`
- Modify: `tests/scripts/test_phase2b3a_checkpoint_scripts.py`
- Modify: `docs/phase2b3a-cloud-runbook.md`
- Modify: `docs/codex-handoff.md`

**Interfaces:**
- Consumes: Task 2 A1 v2 requirement and Task 3 checkpoint validation.
- Produces: an exact guarded disposable Phase 2B3-A reset after legacy A1 restore, followed by v2 A1
  before A2. Durable A1 checkpoint and model sources are never deletion targets.

- [ ] **Step 1: Write a failing executable script test**

Exercise the reset behavior in a temporary mounted-layout fixture: only the exact disposable
`trustsr/phase2b3a` target can be removed; symlink, missing parent, unexpected path, durable-root, and
active-lock cases fail before mutation.

- [ ] **Step 2: Run the script tests and verify RED**

Run: `uv run pytest tests/scripts/test_phase2b3a_checkpoint_scripts.py -q`

- [ ] **Step 3: Implement and document the guarded reset**

Keep the deletion target explicit and canonical, require it below the disposable `trustsr` root, and
recreate an empty non-symlink directory. Update the runbook sequence to restore legacy A1, reset only
live Phase 2B3-A state, preflight, v2 A1, checkpoint/reverify A1, A2, replay, and checkpoint/reverify
A2. Record the original stop evidence and new checkpoint in the handoff without endpoint details.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/scripts/test_phase2b3a_checkpoint_scripts.py -q
bash -n scripts/phase2b3a/*.sh
git diff --check
```

- [ ] **Step 5: Commit**

```bash
git add scripts/phase2b3a/restore_workspace.sh tests/scripts/test_phase2b3a_checkpoint_scripts.py \
  docs/phase2b3a-cloud-runbook.md docs/codex-handoff.md
git commit -m "docs: add phase2b3a saturation rerun workflow"
```

### Task 5: Integrated local acceptance

**Files:**
- Modify only if a verified integration defect requires the smallest targeted repair.

**Interfaces:**
- Consumes: Tasks 1-4 merged in dependency order.
- Produces: one reviewed and locally accepted commit lineage ready to push and deploy.

- [ ] **Step 1: Run complete verification**

```bash
uv sync --frozen
uv run pytest -q
uv run ruff check .
bash -n scripts/phase2b3a/*.sh
uv run trustsr-phase2b3a --help >/dev/null
PYTHONPATH=src uv run python -m trustsr.artifacts.workspace_checkpoint --help >/dev/null
git diff --check
git status --short --branch
```

- [ ] **Step 2: Review provenance and sensitive strings**

```bash
: "${PHASE2B3A_SENSITIVE_PATTERN:?set from untracked operator context}"
rg -n -i "$PHASE2B3A_SENSITIVE_PATTERN" src tests scripts docs artifacts
git diff --stat HEAD~4..HEAD
```

Expected: no endpoint, port, credential, host runtime, real pixel, cache, or checkpoint payload is
tracked; only the runbook's generic placeholders remain.

- [ ] **Step 3: Independent whole-branch review**

Review the complete base-to-head diff for spec compliance, code quality, historical-evidence
preservation, replay identity, checkpoint safety, and exact 120-development-only scope. Resolve every
Critical or Important finding and re-run its covering tests.
