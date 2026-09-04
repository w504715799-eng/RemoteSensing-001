# Phase 2B3-B Formal Calibration Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the fixed Phase 2B3-B calibration, inference-free replay, independent acceptance, and Git-safe publication surfaces without reading real calibration pixels or running LDSR during local engineering.

**Architecture:** Keep the existing verified scientific units as the only implementation of selection, loading, K5 cache identity, score/risk construction, conformal fitting, result composition, runtime inventory, replay receipt, and atomic bundle I/O. Add one frozen operating-point authority, one formal workflow orchestrator, and one independent acceptance/publication verifier; command-line modules remain thin adapters with no scientific override flags.

**Tech Stack:** Python 3.12, PyTorch CPU tensors in tests, pytest, canonical JSON, SHA-256, atomic filesystem publication, existing TrustSR cache and verification modules.

**Spec:** `docs/superpowers/specs/2026-09-03-phase2b3b-calibration-design.md`

## Global Constraints

- Formal `alpha` is exactly the built-in float `0.05`; no command-line or environment override is allowed.
- Formal minimum calibration pixel coverage is exactly the built-in float `0.10`; no command-line or environment override is allowed.
- Accept exactly 120 frozen `calibration` records in canonical manifest order and never load `development` or `internal_test` pixels.
- Formal inference uses only LDSR-S2-x4 seeds `3407,3408,3409,3410,3411`; replay and verification never construct a model or call prediction.
- Local engineering and tests use only generated metadata and tiny synthetic CPU tensors. Do not inspect real calibration pixels or any `internal_test` artifact.
- Real calibration and GPU execution remain forbidden until all local command, replay, acceptance, publication, and independent-verification checks pass and the user separately authorizes GPU startup.
- The current attached `main` worktree is the sole write target; do not create a branch or worktree and do not delegate writes.

---

### Task 1: Freeze the approved operating point across scientific result paths

**Files:**
- Create: `src/trustsr/evaluation/phase2b3b_policy.py`
- Create: `tests/evaluation/test_phase2b3b_policy.py`
- Modify: `src/trustsr/evaluation/phase2b3b_result.py`
- Modify: `src/trustsr/evaluation/phase2b3b_result_verify.py`
- Modify: `src/trustsr/evaluation/phase2b3b_computation_verify.py`
- Modify: `tests/evaluation/test_phase2b3b_result.py`
- Modify: `tests/evaluation/test_phase2b3b_result_verify.py`
- Modify: `tests/evaluation/test_phase2b3b_computation_verify.py`

**Interfaces:**
- Produces: `APPROVED_ALPHA: Final[float]`, `APPROVED_MINIMUM_COVERAGE: Final[float]`, and `require_approved_operating_point(alpha: object, minimum_coverage: object) -> tuple[float, float]`.
- Consumes: Existing `CalibrationFit`, result composer, metadata verifier, and computation replay verifier.

- [ ] **Step 1: Write failing policy and integration tests**

```python
def test_approved_operating_point_is_exact() -> None:
    assert require_approved_operating_point(0.05, 0.10) == (0.05, 0.10)

@pytest.mark.parametrize("alpha,coverage", [(0.0500001, 0.10), (0.05, 0.100001), (True, 0.10)])
def test_other_operating_points_are_rejected(alpha: object, coverage: object) -> None:
    with pytest.raises(ValueError, match="approved"):
        require_approved_operating_point(alpha, coverage)
```

Update result, result-verifier, and computation-verifier tests so a result carrying any other target fails before it can be accepted or replayed.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run pytest -q tests/evaluation/test_phase2b3b_policy.py tests/evaluation/test_phase2b3b_result.py tests/evaluation/test_phase2b3b_result_verify.py tests/evaluation/test_phase2b3b_computation_verify.py
```

Expected: failures because the policy module does not exist and existing result paths accept caller-selected values.

- [ ] **Step 3: Implement the frozen policy and enforce it**

```python
APPROVED_ALPHA: Final[float] = 0.05
APPROVED_MINIMUM_COVERAGE: Final[float] = 0.10

def require_approved_operating_point(
    alpha: object, minimum_coverage: object
) -> tuple[float, float]:
    if type(alpha) is not float or alpha != APPROVED_ALPHA:
        raise ValueError("alpha must equal the approved Phase 2B3-B value 0.05")
    if type(minimum_coverage) is not float or minimum_coverage != APPROVED_MINIMUM_COVERAGE:
        raise ValueError("minimum coverage must equal the approved Phase 2B3-B value 0.10")
    return alpha, minimum_coverage
```

Call this validator from result composition, independent result verification, and cache-computation replay. Leave the generic conformal primitive and synthetic `fit_calibration_maps` API configurable; only the formal result boundary is frozen.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Commit the policy boundary**

```bash
git add src/trustsr/evaluation/phase2b3b_policy.py src/trustsr/evaluation/phase2b3b_result.py src/trustsr/evaluation/phase2b3b_result_verify.py src/trustsr/evaluation/phase2b3b_computation_verify.py tests/evaluation/test_phase2b3b_policy.py tests/evaluation/test_phase2b3b_result.py tests/evaluation/test_phase2b3b_result_verify.py tests/evaluation/test_phase2b3b_computation_verify.py
git commit -m "feat: freeze phase2b3b operating point"
```

### Task 2: Add the formal calibration and inference-free replay workflow

**Files:**
- Create: `src/trustsr/evaluation/phase2b3b_workflow.py`
- Create: `tests/evaluation/test_phase2b3b_workflow.py`
- Modify: `src/trustsr/cli/phase2b3b.py`
- Modify: `tests/cli/test_phase2b3b.py`

**Interfaces:**
- Consumes: `load_phase2b3b_preflight`, `load_calibration_records`, `load_calibration_pairs`, `load_or_generate_calibration_bundle`, `load_or_compute_calibration_maps`, `fit_calibration_maps`, the receipt/result/runtime builders, cache replay, and `write_phase2b3b_bundle`.
- Produces: `Phase2B3BStoragePaths`, `run_formal_calibration(...) -> Phase2B3BWorkflowReceipt`, and `run_formal_calibration_replay(...) -> Phase2B3BWorkflowReceipt`.

- [ ] **Step 1: Write failing workflow and CLI tests**

Tests must prove the following observable sequence with injected synthetic dependencies:

```text
revision -> preflight metadata -> exact calibration records -> pixel loader
-> input/radiometry receipts -> K5 bundles -> score/risk maps -> approved fit
-> audit/result/runtime -> inference-free reconstruction -> atomic bundle
```

The replay test must install a prediction function that raises if called, reconstruct result and audit from existing caches, require byte identity, and reuse the existing bundle. Parser tests must expose exactly `preflight`, `calibration`, and `calibration-replay`, require explicit persistent-storage confirmation for formal stages, accept an optional LDSR model directory only for calibration, and reject `--alpha`, `--coverage`, `--seed`, `--score`, `--sample`, and worker overrides.

- [ ] **Step 2: Run workflow and CLI tests and verify RED**

Run:

```bash
uv run pytest -q tests/evaluation/test_phase2b3b_workflow.py tests/cli/test_phase2b3b.py
```

Expected: failures because the workflow and formal subcommands do not exist.

- [ ] **Step 3: Implement deterministic storage and orchestration**

Derive all mutable paths below the confirmed canonical storage root:

```text
trustsr/phase2b3b/predictions/<post-manifest-sha256>/
trustsr/phase2b3b/scores/<post-manifest-sha256>/
trustsr/phase2b3b/bundles/<post-manifest-sha256>/
```

Require more than 10 GiB free space, reject symlink components and unsafe roots, and take one advisory phase lock before pixel loading. Calibration first verifies whether one coherent complete K5 cache set exists; it may construct `LDSRS2X4` and call prediction only after a cache miss and after revision, metadata, path, capacity, and lock gates pass. A missing cache with no model path stops before model construction so GPU permission can be requested. It must immediately rebuild score/risk/fit/result from committed caches without inference, create the replay receipt, and atomically publish the complete five-file bundle. Replay must never import or construct the model and must fail unless the rebuilt result, cache audit, replay receipt, and existing bundle are byte-identical.

- [ ] **Step 4: Wire the two formal CLI stages**

Each stage emits one host-free canonical JSON receipt on stdout and redirects diagnostics to stderr. Formal handlers pass `APPROVED_ALPHA` and `APPROVED_MINIMUM_COVERAGE` directly; the parser contains no scientific overrides.

- [ ] **Step 5: Run focused workflow and CLI tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass without CUDA use and without any real data fixture.

- [ ] **Step 6: Commit the formal workflow**

```bash
git add src/trustsr/evaluation/phase2b3b_workflow.py src/trustsr/cli/phase2b3b.py tests/evaluation/test_phase2b3b_workflow.py tests/cli/test_phase2b3b.py
git commit -m "feat: add phase2b3b calibration replay workflow"
```

### Task 3: Add independent acceptance and atomic Git-safe publication

**Files:**
- Create: `src/trustsr/evaluation/phase2b3b_acceptance.py`
- Create: `tests/evaluation/test_phase2b3b_acceptance.py`
- Modify: `src/trustsr/cli/phase2b3b_verify.py`
- Modify: `tests/cli/test_phase2b3b_verify.py`

**Interfaces:**
- Consumes: the copied bundle reader, metadata bundle verifier, cache-computation verifier, authoritative preflight/input/radiometry reconstruction, frozen operating point, and fixed storage cache paths.
- Produces: `build_phase2b3b_acceptance(...) -> VerifiedPhase2B3BAcceptance`, `publish_phase2b3b_evidence(...) -> Phase2B3BPublicationReceipt`, and an acceptance-authorizing `trustsr-phase2b3b-verify` command.

- [ ] **Step 1: Write failing acceptance and verifier CLI tests**

Tests cover both decisions, exact target rejection, digest cross-binding, forged receipt types, metadata-only receipts that cannot authorize, mismatched computation digests, non-byte-identical replay, extra/missing bundle files, forbidden host/path/secret content, fixed publication filenames, collision rejection, idempotent identical publication, concurrent identical writers, and cleanup after an injected staging failure.

- [ ] **Step 2: Run acceptance tests and verify RED**

Run:

```bash
uv run pytest -q tests/evaluation/test_phase2b3b_acceptance.py tests/cli/test_phase2b3b_verify.py
```

Expected: failures because no acceptance authority or three-file publisher exists and the CLI is metadata-only.

- [ ] **Step 3: Implement the acceptance schema**

The canonical acceptance record must bind actual bundle-manifest, result, cache-audit, runtime, and replay bytes; require successful metadata authority, cache-computation replay, and byte-identical replay; copy the exact approved operating point; record the observed decision; and expose a non-null `frozen_calibration` payload only for `freeze_calibration`. The frozen payload carries the upstream six evidence digests, score/risk identities, threshold, risk bound, counts, coverage, input identities, model inventory, and producer revision. `stop_insufficient_coverage` publishes the negative result with `frozen_calibration: null`.

- [ ] **Step 4: Implement transaction-style publication**

Atomically publish exactly these canonical files under `artifacts/phase2b3b` in the reviewed checkout:

```text
sen2naipv2-calibration-conformal-v1.json
sen2naipv2-calibration-conformal-cache-audit-v1.json
sen2naipv2-calibration-conformal-acceptance-v1.json
```

Use a same-parent staging directory, fsync written files and directories, rename without replacement, reuse an existing byte-identical publication, reject partial/different/symlinked output, and remove staging state after failures.

- [ ] **Step 5: Upgrade the verifier CLI**

The command validates the clean reviewed checkout and copied bundle, loads only the 120 authoritative calibration pairs, reconstructs input/radiometry authority, replays prediction and score caches without model inference, builds acceptance, publishes the three files, and emits one canonical host-free success receipt. Its parser has no scientific, sample, split, or model flags.

- [ ] **Step 6: Run focused acceptance and CLI tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass using only synthetic CPU data.

- [ ] **Step 7: Commit independent acceptance**

```bash
git add src/trustsr/evaluation/phase2b3b_acceptance.py src/trustsr/cli/phase2b3b_verify.py tests/evaluation/test_phase2b3b_acceptance.py tests/cli/test_phase2b3b_verify.py
git commit -m "feat: verify and publish phase2b3b acceptance"
```

### Task 4: Document the exact formal commands and local readiness gate

**Files:**
- Create: `docs/phase2b3b-calibration-runbook.md`
- Modify: `docs/codex-handoff.md`

**Interfaces:**
- Consumes: the final CLI parsers and fixed storage layout.
- Produces: copyable preflight, calibration, calibration-replay, verifier/publication, digest-review, and GPU-permission procedures.

- [ ] **Step 1: Write the runbook**

Document `REVIEWED_COMMIT=$(git rev-parse HEAD)` as an operator variable captured only after the local readiness gate, the explicit persistent-storage confirmation, the fact that only `calibration` may invoke LDSR, the copied-bundle verification flow, the three allowed Git-safe files, the `freeze_calibration` and `stop_insufficient_coverage` branches, and the continuing prohibition on all `internal_test` access.

- [ ] **Step 2: Update the handoff**

Replace statements that formal commands are missing with the implemented command surface and list the exact local readiness commands. Do not claim that real calibration, replay, acceptance, publication, or GPU work has occurred.

- [ ] **Step 3: Check documentation and commit**

Run:

```bash
git diff --check
rg -n "0\.05|0\.10|calibration-replay|stop_insufficient_coverage|internal_test" docs/phase2b3b-calibration-runbook.md docs/codex-handoff.md
```

Expected: no whitespace errors and all frozen decisions/boundaries are explicit.

```bash
git add docs/phase2b3b-calibration-runbook.md docs/codex-handoff.md
git commit -m "docs: add phase2b3b calibration runbook"
```

### Task 5: Execute the scoped local readiness and independent-verification checks

**Files:**
- Modify only if a failing check exposes a defect in the files above.

**Interfaces:**
- Consumes: all Phase 2B3-B modules and CLI entry points.
- Produces: fresh local CPU evidence that the formal surfaces are ready; does not produce scientific calibration evidence.

- [ ] **Step 1: Run the Phase 2B3-B and calibration test scope**

```bash
uv run pytest -q \
  tests/cli/test_phase2b3b.py \
  tests/cli/test_phase2b3b_verify.py \
  tests/evaluation/test_phase2b3b_*.py \
  tests/evaluation/test_calibration_*.py \
  tests/data/test_calibration_subset.py \
  tests/data/test_calibration_pairs.py \
  tests/calibration/test_conformal.py
```

Expected: all selected tests pass; any TorchScript deprecation warnings are identified as third-party warnings rather than silently ignored.

- [ ] **Step 2: Run focused static and command-boundary checks**

```bash
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
```

Expected: all commands exit zero and formal help exposes no `--alpha`, `--coverage`, `--seed`, `--score`, `--sample`, or `internal_test` option.

- [ ] **Step 3: Verify repository state and stop before real data**

```bash
git status --short --branch
git log -1 --format=%H
```

Confirm the checkout is attached to `main`, the only unpushed commits are the intended local Phase 2B3-B work, no real calibration/publication artifact was created during engineering, and no `internal_test` path was accessed. If verified K5 calibration cache entries are not already present, report that GPU permission is the next required authorization; do not start a server or run a model.
