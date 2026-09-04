# Phase 2B3-C One-Time Internal-Test Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` task-by-task.
> The attached `main` worktree is the sole write target; do not create a branch/worktree or delegate
> writes.

**Goal:** Implement and locally verify the frozen Phase 2B3-C one-time evaluation, replay,
independent acceptance, and publication surfaces without accessing any real `internal_test` pixel,
cache, prediction, score, risk, or metric.

**Architecture:** Build a Phase 2B3-C namespace around the accepted Phase 2B3-B frozen payload.
Keep metadata preflight separate from the permit-gated pixel boundary, record authorized access as
immutable digest-chained events, compute one loss per ROI, use the preregistered order-invariant
grid-Kelly UCB, and require inference-free independent replay before publication. Reuse generic
scientific primitives and hardened I/O patterns, but do not parameterize B's split-specific paths.

**Tech Stack:** Python 3.12, NumPy, PyTorch CPU in tests, pytest, canonical JSON, SHA-256, Git
ancestry/tree validation, `fcntl` locks, descriptor-relative I/O, atomic no-replace writes.

**Spec:** `docs/superpowers/specs/2026-09-04-phase2b3c-one-time-internal-test-design.md`

## Global constraints

- Freeze threshold `7.970395366024563e-06`, risk target `0.05`, coverage `0.10`, confidence error
  `0.05`, and grid size `20`; expose no overrides.
- Accept exactly 120 frozen test records, 12 strata by 10, in canonical manifest order.
- Use generated metadata and tiny synthetic CPU tensors locally. Tripwire every real-data,
  test-cache, model-construction, and GPU boundary.
- Do not create the access permit during implementation. That requires later explicit user
  authorization dedicated to the one-time evaluation.
- Do not inspect/start cloud resources locally. Later remote work uses `/opt/conda/bin/python` in
  base, never remote `uv`, `.venv`, or environment mutation.
- Run focused tests per task and the complete C scope once at the final local gate.

---

### Task 1: Freeze policy, ROI statistics, and the grid-Kelly UCB

**Files:**
- Create: `src/trustsr/evaluation/phase2b3c_policy.py`
- Create: `src/trustsr/evaluation/phase2b3c_statistics.py`
- Create: `tests/evaluation/test_phase2b3c_policy.py`
- Create: `tests/evaluation/test_phase2b3c_statistics.py`

**Interfaces:** exact constants/validators;
`evaluate_roi_loss(score, risk, threshold) -> ROIEvaluation`;
`grid_kelly_log_evalue(losses, candidate_mean) -> float`;
`grid_kelly_upper_bound(losses) -> float`;
`build_phase2b3c_statistics(...) -> Phase2B3CStatistics`.

- [ ] Write failing tests for strict built-in types, exact constants, score ties, empty trusted
  masks, range/non-finite failures, aggregate coverage, signed violations, Hoeffding diagnostic,
  12 ordered strata, 120 ROI count, and all three decisions.
- [ ] Add a structurally independent `decimal.Decimal` oracle for small direct products. Cover
  `rho_j=j/21`, equal weights, permutation invariance, monotonicity, endpoints, mixed losses, and
  the conservative 128-step bisection.
- [ ] Run and verify RED:

  ```bash
  uv run pytest -q tests/evaluation/test_phase2b3c_policy.py \
    tests/evaluation/test_phase2b3c_statistics.py
  ```

- [ ] Implement binary64 log wealth with `math.log1p`/`math.fsum`, max-shifted log-sum-exp, and
  exactly 128 bisections returning the maintained rejected upper endpoint. Do not call an author
  package or a generic root finder.
- [ ] Rerun GREEN and commit:

  ```bash
  git add src/trustsr/evaluation/phase2b3c_policy.py \
    src/trustsr/evaluation/phase2b3c_statistics.py \
    tests/evaluation/test_phase2b3c_policy.py \
    tests/evaluation/test_phase2b3c_statistics.py
  git commit -m "feat: add phase2b3c evaluation statistics"
  ```

### Task 2: Verify the frozen B interface and implementation revision

**Files:**
- Create: `src/trustsr/evaluation/phase2b3c_evidence.py`
- Create: `src/trustsr/evaluation/phase2b3c_revision.py`
- Create: `tests/evaluation/test_phase2b3c_evidence.py`
- Create: `tests/evaluation/test_phase2b3c_revision.py`

**Interfaces:** `load_frozen_phase2b3b_evidence(...) -> FrozenPhase2B3BEvidence`;
`phase2b3c_computation_tree_sha256(...) -> str`;
`verify_phase2b3c_implementation_revision(...) -> VerifiedRevision`.

- [ ] Write failing evidence tests with canonical synthetic copies of all three B schemas. Mutate
  every frozen identity class. Require exact file digests, publication ancestry,
  `freeze_calibration`, `acceptance_authorized`, cross-bindings, and the full frozen payload.
- [ ] Write failing revision tests. Define the computation-tree digest as canonical Git blob
  identities for `pyproject.toml`, `uv.lock`, and `src/trustsr/**`. Reject dirty tracked compute
  files, untrusted/detached revisions, source drift, symlink roots, malformed hashes, and later
  changes other than the future exact permit artifact.
- [ ] Run RED, implement independent byte/schema/Git verification, and rerun GREEN:

  ```bash
  uv run pytest -q tests/evaluation/test_phase2b3c_evidence.py \
    tests/evaluation/test_phase2b3c_revision.py
  ```

- [ ] Commit:

  ```bash
  git add src/trustsr/evaluation/phase2b3c_evidence.py \
    src/trustsr/evaluation/phase2b3c_revision.py \
    tests/evaluation/test_phase2b3c_evidence.py \
    tests/evaluation/test_phase2b3c_revision.py
  git commit -m "feat: verify frozen phase2b3b evidence"
  ```

### Task 3: Add metadata-only selection and permit-gated pair loading

**Files:**
- Create: `src/trustsr/data/internal_test_subset.py`
- Create: `src/trustsr/data/internal_test_pairs.py`
- Create: `src/trustsr/evaluation/internal_test_input_receipt.py`
- Create: `tests/data/test_internal_test_subset.py`
- Create: `tests/data/test_internal_test_pairs.py`
- Create: `tests/evaluation/test_internal_test_input_receipt.py`

**Interfaces:** `select_internal_test_records(...)`; `load_internal_test_records(...)`;
`load_internal_test_pairs(..., access_guard)`; `build_internal_test_input_receipt(...)`.

- [ ] Write selector tests using a generated 360-row manifest. Require exact 120 membership,
  12-by-10 strata, rounds 1–10, unique identities/groups, canonical order, and no pixel/cache/model
  action. Reject incomplete, duplicated, reordered, development, and calibration cases.
- [ ] Write loader/receipt tests mirroring B's byte/sidecar/shape/dtype/CRS/transform/nodata/range/
  crop/saturation checks, with descriptor-relative confinement and an authenticated
  `pixels_opened` guard required before the first asset open.
- [ ] Run RED, implement without weakening B's split guard, and rerun GREEN:

  ```bash
  uv run pytest -q tests/data/test_internal_test_subset.py \
    tests/data/test_internal_test_pairs.py \
    tests/evaluation/test_internal_test_input_receipt.py
  ```

- [ ] Commit:

  ```bash
  git add src/trustsr/data/internal_test_subset.py \
    src/trustsr/data/internal_test_pairs.py \
    src/trustsr/evaluation/internal_test_input_receipt.py \
    tests/data/test_internal_test_subset.py tests/data/test_internal_test_pairs.py \
    tests/evaluation/test_internal_test_input_receipt.py
  git commit -m "feat: add phase2b3c test input boundary"
  ```

### Task 4: Implement readiness, permit verification, and immutable ledger

**Files:**
- Create: `src/trustsr/evaluation/phase2b3c_access.py`
- Create: `tests/evaluation/test_phase2b3c_access.py`

**Interfaces:** `build_phase2b3c_readiness(...)`; `verify_phase2b3c_access_permit(...)`;
`phase2b3c_access_lock(...)`; `load_access_ledger(...)`; `advance_access_ledger(...)`.

- [ ] Write failing readiness/permit tests for deterministic evaluation ID, implementation tree,
  B/manifest identities, exact parameters, `one_time_evaluation=true`, canonical bytes, explicit
  authorization statement, repository location, and absence of paths/hosts/times/secrets.
  Production exposes permit verification, not a self-approving builder.
- [ ] Write failing state tests using immutable no-replace event files from `000-reserved.json`
  through `004-accepted.json`, plus terminal invalidation. Bind each predecessor digest. Cover
  exact resume, skipped/duplicate/rollback/deleted states, root drift, lock contention, crash
  injection, cleanup, and post-access invalidation.
- [ ] Run RED, implement same-parent staging/fsync/no-replace/descriptor-relative I/O, and rerun:

  ```bash
  uv run pytest -q tests/evaluation/test_phase2b3c_access.py
  ```

- [ ] Commit:

  ```bash
  git add src/trustsr/evaluation/phase2b3c_access.py \
    tests/evaluation/test_phase2b3c_access.py
  git commit -m "feat: enforce one-time phase2b3c access"
  ```

### Task 5: Add test caches, maps, metrics, and cache audit

**Files:**
- Create: `src/trustsr/evaluation/internal_test_predictions.py`
- Create: `src/trustsr/evaluation/internal_test_maps.py`
- Create: `src/trustsr/evaluation/internal_test_metrics.py`
- Create: `src/trustsr/evaluation/internal_test_cache_audit.py`
- Create: `tests/evaluation/test_internal_test_predictions.py`
- Create: `tests/evaluation/test_internal_test_maps.py`
- Create: `tests/evaluation/test_internal_test_metrics.py`
- Create: `tests/evaluation/test_internal_test_cache_audit.py`

- [ ] Write prediction tests for 120 samples, exact K5 seeds/model/LR digest, canonical cache paths,
  tensor identity, atomic entries, and complete-set probing before model import/construction. One
  missing entry without a model path raises a redacted typed stop.
- [ ] Write map/metric/audit tests for population variance, band mean, seed-3407 risk, R9,
  inclusive threshold, ROI loss/coverage, 12 diagnostics, and exact map/cache digests. Mutate score
  correction, seed order, tie, window, central seed, dtype/range, sample order, and extra entries.
- [ ] Run RED, implement with split-neutral primitives only, and rerun GREEN:

  ```bash
  uv run pytest -q tests/evaluation/test_internal_test_predictions.py \
    tests/evaluation/test_internal_test_maps.py \
    tests/evaluation/test_internal_test_metrics.py \
    tests/evaluation/test_internal_test_cache_audit.py
  ```

- [ ] Commit:

  ```bash
  git add src/trustsr/evaluation/internal_test_*.py \
    tests/evaluation/test_internal_test_*.py
  git commit -m "feat: add phase2b3c cache computation"
  ```

### Task 6: Define result, runtime, replay, and atomic bundle schemas

**Files:**
- Create: `src/trustsr/evaluation/phase2b3c_result.py`
- Create: `src/trustsr/evaluation/phase2b3c_result_verify.py`
- Create: `src/trustsr/evaluation/phase2b3c_runtime.py`
- Create: `src/trustsr/evaluation/phase2b3c_replay_receipt.py`
- Create: `src/trustsr/evaluation/phase2b3c_bundle.py`
- Create: corresponding `tests/evaluation/test_phase2b3c_{result,result_verify,runtime,replay_receipt,bundle}.py`

- [ ] Write strict schema/DAG tests for three decisions, reasons, 12 diagnostics, aggregates, no
  per-sample numerical metrics, all identity bindings, host-free runtime, non-authorizing scopes,
  and the acyclic result → runtime → replay → bundle graph.
- [ ] Write bundle filesystem tests for canonical bytes, six exact files, manifest membership,
  copied reads, identical concurrency, collision/partial/extra/symlink/race rejection, no-replace
  writes, fsync, and cleanup after injected failure.
- [ ] Run RED, implement independent C schemas using B's hardened I/O pattern, rerun GREEN:

  ```bash
  uv run pytest -q tests/evaluation/test_phase2b3c_result.py \
    tests/evaluation/test_phase2b3c_result_verify.py \
    tests/evaluation/test_phase2b3c_runtime.py \
    tests/evaluation/test_phase2b3c_replay_receipt.py \
    tests/evaluation/test_phase2b3c_bundle.py
  ```

- [ ] Commit:

  ```bash
  git add src/trustsr/evaluation/phase2b3c_*.py \
    tests/evaluation/test_phase2b3c_*.py
  git commit -m "feat: add phase2b3c evidence bundle"
  ```

### Task 7: Add metadata preflight and independent computation replay

**Files:**
- Create: `src/trustsr/evaluation/phase2b3c_preflight.py`
- Create: `src/trustsr/evaluation/phase2b3c_computation_verify.py`
- Create: `tests/evaluation/test_phase2b3c_preflight.py`
- Create: `tests/evaluation/test_phase2b3c_computation_verify.py`

- [ ] Write preflight tripwire tests. It validates Git, B evidence, full manifest, exact metadata
  membership, storage/environment/readiness, and an optional permit, but never opens an image,
  enumerates test caches, imports model backends, queries CUDA, creates a ledger, or writes output.
- [ ] Write independent replay tests. Given caller-loaded authoritative pairs/caches, reimplement
  score, risk, ROI statistics, grid-Kelly inversion, diagnostics, audit, result, and runtime
  projections without calling producer result/statistic helpers. Reject forged receipts and every
  value/digest mutation; never construct LDSR or authorize acceptance.
- [ ] Run RED, implement, and rerun GREEN:

  ```bash
  uv run pytest -q tests/evaluation/test_phase2b3c_preflight.py \
    tests/evaluation/test_phase2b3c_computation_verify.py
  ```

- [ ] Commit:

  ```bash
  git add src/trustsr/evaluation/phase2b3c_preflight.py \
    src/trustsr/evaluation/phase2b3c_computation_verify.py \
    tests/evaluation/test_phase2b3c_preflight.py \
    tests/evaluation/test_phase2b3c_computation_verify.py
  git commit -m "feat: verify phase2b3c computation"
  ```

### Task 8: Implement the formal workflow and command surface

**Files:**
- Create: `src/trustsr/evaluation/phase2b3c_workflow.py`
- Create: `src/trustsr/cli/phase2b3c.py`
- Create: `tests/evaluation/test_phase2b3c_workflow.py`
- Create: `tests/cli/test_phase2b3c.py`
- Modify: `pyproject.toml`

**Interfaces:** `Phase2B3CStoragePaths`; `run_formal_evaluation(...)`;
`run_formal_evaluation_replay(...)`; console script `trustsr-phase2b3c`.

- [ ] Write injected workflow-order tests for:

  ```text
  revision/evidence -> metadata preflight -> permit -> lock/reserved -> pixels_opened
  -> exact records/pairs/receipts -> complete K5 probe -> optional generation
  -> maps/metrics/statistics -> audit/result/runtime -> inference-free reconstruction
  -> replay/ledger snapshot -> atomic bundle
  ```

  Cover complete-cache CPU, missing-cache stop before model construction, permitted generation,
  partial resume, redacted logs, interruptions on both sides of pixel access, and no partial metric.
- [ ] Write parser/replay tests. Expose exactly `preflight`, `evaluate`, and `evaluation-replay`;
  only `evaluate` accepts an optional model directory. Reject alpha/threshold/coverage/delta/grid/
  score/risk/seed/split/sample/device/worker/output-name flags. Replay model factories must raise.
- [ ] Run RED, implement thin canonical CLI handlers, and rerun GREEN:

  ```bash
  uv run pytest -q tests/evaluation/test_phase2b3c_workflow.py \
    tests/cli/test_phase2b3c.py
  ```

- [ ] Commit:

  ```bash
  git add pyproject.toml src/trustsr/evaluation/phase2b3c_workflow.py \
    src/trustsr/cli/phase2b3c.py \
    tests/evaluation/test_phase2b3c_workflow.py tests/cli/test_phase2b3c.py
  git commit -m "feat: add phase2b3c evaluation workflow"
  ```

### Task 9: Add independent acceptance and transaction publication

**Files:**
- Create: `src/trustsr/evaluation/phase2b3c_bundle_verify.py`
- Create: `src/trustsr/evaluation/phase2b3c_acceptance.py`
- Create: `src/trustsr/cli/phase2b3c_verify.py`
- Create: `tests/evaluation/test_phase2b3c_bundle_verify.py`
- Create: `tests/evaluation/test_phase2b3c_acceptance.py`
- Create: `tests/cli/test_phase2b3c_verify.py`
- Modify: `pyproject.toml`

- [ ] Write metadata verifier tests that cross-bind all six bundle documents, permit, ledger,
  evidence, membership, model, and revision identities while explicitly remaining metadata-only.
- [ ] Write acceptance/publication tests. Require a fresh independent computation receipt and
  byte-identical replay; cover all decisions, copied-bundle requirement, terminal ledger update,
  forged receipt rejection, exact three publication files, idempotent identical bytes, and atomic
  collision/partial/symlink/race cleanup without altering the preexisting permit.
- [ ] Write verifier CLI tests: reacquire the same evaluation lock, verify ledger state, reload
  authorized inputs, replay caches without inference, publish, then advance to `accepted`. Expose
  no model or scientific flag.
- [ ] Run RED, implement, and rerun GREEN:

  ```bash
  uv run pytest -q tests/evaluation/test_phase2b3c_bundle_verify.py \
    tests/evaluation/test_phase2b3c_acceptance.py \
    tests/cli/test_phase2b3c_verify.py
  ```

- [ ] Commit:

  ```bash
  git add pyproject.toml src/trustsr/evaluation/phase2b3c_bundle_verify.py \
    src/trustsr/evaluation/phase2b3c_acceptance.py \
    src/trustsr/cli/phase2b3c_verify.py \
    tests/evaluation/test_phase2b3c_bundle_verify.py \
    tests/evaluation/test_phase2b3c_acceptance.py tests/cli/test_phase2b3c_verify.py
  git commit -m "feat: verify and publish phase2b3c acceptance"
  ```

### Task 10: Enforce artifact policy and synthetic end-to-end readiness

**Files:**
- Create: `src/trustsr/evaluation/phase2b3c_policy_scan.py`
- Create: `tests/evaluation/test_phase2b3c_policy_scan.py`
- Modify: `src/trustsr/data/local_policy.py`
- Modify: `tests/data/test_local_data_policy.py`
- Modify: `.gitignore`
- Create: `tests/evaluation/test_phase2b3c_synthetic_e2e.py`

- [ ] Write allowlist/leak tests. Permit only the future exact access-authorization filename and
  three exact evaluation publication names. Reject other files, tensors, manifests, weights,
  paths, hosts, endpoints, credentials, raw timestamps, and per-sample metrics.
- [ ] Write a tiny synthetic end-to-end test with 360 generated metadata rows, 120 tiny CPU pairs,
  fake K5 predictions, test-only permit, and temporary ledger. Exercise evaluate, no-inference
  replay, copied-bundle independent verification, each decision, and canonical publication.
- [ ] Run RED, implement, and rerun GREEN:

  ```bash
  uv run pytest -q tests/evaluation/test_phase2b3c_policy_scan.py \
    tests/data/test_local_data_policy.py \
    tests/evaluation/test_phase2b3c_synthetic_e2e.py
  ```

- [ ] Commit:

  ```bash
  git add .gitignore src/trustsr/data/local_policy.py \
    src/trustsr/evaluation/phase2b3c_policy_scan.py \
    tests/data/test_local_data_policy.py \
    tests/evaluation/test_phase2b3c_policy_scan.py \
    tests/evaluation/test_phase2b3c_synthetic_e2e.py
  git commit -m "test: harden phase2b3c publication boundary"
  ```

### Task 11: Document operations and run the final local-only gate

**Files:**
- Create: `docs/phase2b3c-one-time-evaluation-runbook.md`
- Modify: `docs/codex-handoff.md`

- [ ] Write the runbook: metadata preflight, external readiness capture, dedicated authorization
  stop, manual permit review/commit, ledger states, CPU cache path, missing-cache GPU stop, base
  Python remote commands, redacted progress, replay, copied verifier, decisions, publication,
  invalidation, and server shutdown.
- [ ] Update handoff only after implementation exists. Record exact commits/commands and state that
  no test pixel/cache/model/GPU was accessed and no permit exists.
- [ ] Run the full Phase 2B3-C scope once:

  ```bash
  uv run pytest -q \
    tests/cli/test_phase2b3c.py tests/cli/test_phase2b3c_verify.py \
    tests/evaluation/test_phase2b3c_*.py \
    tests/evaluation/test_internal_test_*.py \
    tests/data/test_internal_test_subset.py tests/data/test_internal_test_pairs.py \
    tests/data/test_local_data_policy.py tests/models/test_ldsr_s2.py tests/risk/test_local.py
  ```

- [ ] Run lint, compile, CLI, and repository gates:

  ```bash
  uv run ruff check \
    src/trustsr/cli/phase2b3c.py src/trustsr/cli/phase2b3c_verify.py \
    src/trustsr/data/internal_test_subset.py src/trustsr/data/internal_test_pairs.py \
    src/trustsr/evaluation/phase2b3c_*.py src/trustsr/evaluation/internal_test_*.py \
    tests/cli/test_phase2b3c.py tests/cli/test_phase2b3c_verify.py \
    tests/data/test_internal_test_*.py tests/evaluation/test_phase2b3c_*.py \
    tests/evaluation/test_internal_test_*.py
  uv run python -m compileall -q src
  uv run trustsr-phase2b3c --help >/dev/null
  uv run trustsr-phase2b3c preflight --help >/dev/null
  uv run trustsr-phase2b3c evaluate --help >/dev/null
  uv run trustsr-phase2b3c evaluation-replay --help >/dev/null
  uv run trustsr-phase2b3c-verify --help >/dev/null
  git diff --check
  git status --short --branch
  ```

- [ ] Commit documentation/readiness and verify clean attached `main`:

  ```bash
  git add docs/phase2b3c-one-time-evaluation-runbook.md docs/codex-handoff.md
  git commit -m "docs: record phase2b3c local readiness"
  git status --short --branch
  git rev-parse HEAD
  ```

### Task 12: Mandatory stop before real data or GPU

- [ ] Produce only the metadata preflight/readiness JSON outside Git and report its digest,
  evaluation ID, implementation revision/tree, and local gate results. Do not create a permit or
  ledger.
- [ ] Request explicit one-time `internal_test` authorization. Routine recommended-choice approval
  does not satisfy this stop.
- [ ] After that authorization only, write/review/commit the exact permit and run the authorized
  cache probe. If any of 600 predictions is absent, stop before LDSR construction and request GPU
  startup/usage authorization.

No Task 12 action belongs to the present local implementation execution.
