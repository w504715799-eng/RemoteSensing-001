# Phase 2B3-C: One-time internal-test evaluation design

**Date:** 2026-09-04

**Status:** Approved for local implementation on 2026-09-04; real `internal_test` access and GPU
execution remain separately unauthorized

**Upstream:**
`docs/superpowers/specs/2026-09-03-phase2b3b-calibration-design.md`

## 1. Purpose and hard boundary

Phase 2B3-C evaluates the single frozen Phase 2B3-B trusted-pixel rule exactly once on the 120
frozen SEN2NAIPv2 `internal_test` ROIs. It does not select a threshold, compare methods, or create a
fallback product. Its primary question is whether the balanced-design expected ROI loss is bounded
by the already approved risk target while the frozen rule retains the already approved minimum
pixel coverage.

This specification preregisters the test statistic, finite-sample bound, decision rule, access
state machine, evidence graph, and recovery policy before any `internal_test` pixel, prediction,
score, risk, or metric is read. Local implementation and all pre-access verification use metadata
and synthetic tensors only. Real access requires a later, explicit user authorization dedicated to
the one-time evaluation. The user's authorization to follow recommended local engineering choices
does not count as that data-access authorization.

The phase creates Phase 2B3-C-specific data, evaluation, command, policy, and verification modules.
It must not generalize Phase 2B3-B commands to accept arbitrary splits.

## 2. Immutable upstream interface

Phase 2B3-C consumes only the accepted three-file Phase 2B3-B publication at commit
`f8f49a820d22b7dea2e003736ee465f9d7788f7d`. The exact files and SHA-256 digests are:

| File | SHA-256 |
|---|---|
| `sen2naipv2-calibration-conformal-v1.json` | `5fe01daaaa27443d5289cc366eacbe1fb628492c25bab16fadd164d3a8eb3174` |
| `sen2naipv2-calibration-conformal-cache-audit-v1.json` | `40aff8aaccb27719d958d42026d68d8efc0c7f79479e3bd99777053ee1d6399f` |
| `sen2naipv2-calibration-conformal-acceptance-v1.json` | `ce7f1a91e9954235e289dfc6eececb67261ff57110caaf15fe6640a0fb3b69ab` |

Before real-data access, the Phase 2B3-C preflight and evaluator independently verify:

- canonical bytes, exact schemas, all three digests, and the publication commit's trusted Git
  ancestry;
- `acceptance_authorized=true`, `phase_decision="freeze_calibration"`, and the complete
  `frozen_calibration` payload;
- equality between the result, cache audit, acceptance record, and frozen payload wherever their
  identities overlap; and
- an unchanged Phase 2B1-B post-manifest and Phase 2B2-A input audit.

The frozen scientific values are:

```text
score                   ldsr_variance_k5
seeds                   3407, 3408, 3409, 3410, 3411
score variance          population variance, correction=0, then RGBN band mean
risk                    local_l1_risk(window=9), upper bound 1.0
central prediction      seed 3407
threshold               7.970395366024563e-06, inclusive score <= threshold
risk target alpha       0.05
minimum pixel coverage  0.10
normalization           uint16_saturate_10000_divide_10000_v2
crop                    center_crop_lr_1_hr_4_v1
bands                   B04, B03, B02, B08
scale                   4
```

No command exposes an override for any of these values. Phase 2B3-C may project them from the
verified `frozen_calibration` payload but may not infer, refit, round, amend, or replace them.

## 3. Evaluation population, estimand, and assumptions

### 3.1 Frozen balanced evaluation design

The strict Phase 2B1-B 360-row post-extraction manifest is the only membership source. The Phase
2B3-C selector validates the complete manifest without opening assets, then returns exactly the 120
records having `split="internal_test"` in canonical manifest order. It proves:

- 120 unique sample IDs, selection digests, and spatial group IDs;
- exactly 12 `(days_between, correlation_bin)` strata with 10 ROIs per stratum;
- selection rounds 1 through 10, one item per stratum per round;
- unchanged asset byte and geospatial metadata; and
- no development or calibration record crosses the Phase 2B3-C pixel-loading boundary.

The design weights every selected ROI equally. Consequently every stratum has weight `10/120`,
and the estimand is the equally weighted mean of the 12 stratum-specific expected losses, not the
unstratified prevalence-weighted SEN2NAIPv2 population risk.

### 3.2 Primary loss and estimand

For ROI `i`, frozen K5 predictions produce the Phase 2B3-B score and risk maps. Define:

```text
trusted_i[p] = (score_i[p] <= frozen_threshold)
L_i          = max({risk_i[p] : trusted_i[p]} union {0})
C_i_num      = count(p : trusted_i[p])
C_i_den      = count(all evaluated pixels p)
```

`L_i` is a single bounded observation in `[0,1]`; pixels are never treated as independent test
units. An ROI with no trusted pixels has `L_i=0`, exactly matching calibration, while its zero
coverage still counts against the independent coverage gate.

For frozen design positions `i=1,...,120`, let `mu_i = E[L_i]`. The primary inferential estimand is

```text
mu_balanced = (1/120) * sum_i mu_i
```

and the observed empirical mean is

```text
mean_loss = (1/120) * sum_i L_i.
```

The preregistered signed empirical risk violation statistic is

```text
empirical_risk_delta = mean_loss - 0.05
empirical_risk_violation = max(0, empirical_risk_delta).
```

Aggregate pixel coverage is

```text
coverage = sum_i C_i_num / sum_i C_i_den.
```

### 3.3 Scope of the finite-sample statement

The finite-sample bound assumes the 120 ROI losses are mutually independent, each lies in `[0,1]`,
and each frozen design position is sampled without outcome-dependent selection. It permits
different means across the 12 fixed strata. The 5 km connected-component split and unique spatial
groups reduce obvious leakage but cannot prove independence, representativeness, or absence of
large-scale spatial correlation.

Therefore `confirmed` means confirmed for `mu_balanced` under these stated assumptions. It is not a
per-pixel guarantee, a per-ROI guarantee, a natural-prevalence population estimate, a universal
Sentinel-2 guarantee, or proof of radiometric compliance. Phase 2B3-B's conformal statement likewise
controls an expected bounded ROI loss under its exchangeability assumptions; it does not promise
that every ROI or trusted pixel has risk at most `0.05`.

## 4. Finite-sample method

### 4.1 Considered methods

Three methods were considered before test access:

1. **Predictable fixed-time Hedged-CI.** This is variance-adaptive and attractive when every
   observation has the same conditional mean. The published construction uses predictable bets,
   hypothesis-dependent truncation, and inversion of a capital process. It is not the primary
   method here because the 120-row design fixes 10 observations in each of 12 strata and does not
   justify a common conditional mean.
2. **One-sided Hoeffding UCB.** This remains valid for independent, non-identically distributed
   `[0,1]` observations, but at `n=120` and confidence error `0.05` its additive radius is about
   `0.112`. It is retained as a transparent diagnostic and cannot authorize the decision.
3. **Fixed-time one-sided diversified grid-Kelly betting UCB.** This is the selected primary
   method. It uses only constant, preregistered bets, is permutation invariant, and remains valid
   for the average of heterogeneous independent means by independence plus the arithmetic-geometric
   mean inequality. It preserves the useful betting construction without imposing a false common-
   mean assumption.

This replaces the provisional predictable Hedged-CI recommendation made during design discussion.
The change is made before test access because the fixed-stratum contract is scientifically
material; it does not use any observed `internal_test` value.

### 4.2 Exact one-sided grid-Kelly construction

Use confidence error `delta=0.05` and exactly `D=20` equal-weight negative betting strategies. For
candidate mean `m` in `[0,1)`, strategy `j=1,...,20` uses

```text
rho_j       = j / 21
lambda_j(m) = rho_j / (1 - m)
K_j(m)      = product_i [1 - lambda_j(m) * (L_i - m)]
E(m)        = (1/20) * sum_j K_j(m).
```

The grid matches the constant-bet diversified/grid-Kelly construction, with the boundary excluded
by `j/(D+1)`; `D=20` is frozen before access and matches the paper's reported hedged-grid simulation
setting. It has no learned bet, data-dependent permutation, random seed, tuning grid, or runtime
parameter.

For independent `L_i`, the expectation of one component factorizes. At the true candidate
`m=mu_balanced`, all factors are nonnegative and

```text
E[K_j(m)]
  = product_i [1 - lambda_j(m) * (mu_i - m)]
  <= ((1/120) * sum_i [1 - lambda_j(m) * (mu_i - m)])^120
  = 1.
```

Thus each `K_j` and their fixed convex mixture `E` are e-values for the balanced average mean.
Markov's inequality gives error at most `delta` when rejecting at `1/delta`. Every factor and hence
`E(m)` is nondecreasing in `m`, so inversion produces an interval.

The primary upper confidence bound is

```text
risk_ucb = sup {m in [0,1] : log(E(m)) < log(1/delta)}.
```

Computation is frozen as follows:

- accept only 120 finite binary64 losses in `[0,1]` in canonical manifest order;
- compute each `log(K_j)` with `math.log1p` and `math.fsum`;
- compute `log(E)` with the max-shifted log-sum-exp and `math.fsum`;
- use the analytical endpoint behavior at `m=0` and `m=1`;
- perform exactly 128 binary64 bisection iterations on `[0,1]`;
- update the lower endpoint for an accepted midpoint and the upper endpoint for a rejected midpoint;
  return the rejected upper endpoint, making numerical approximation conservative; and
- fail closed on a non-finite intermediate, loss-range violation, non-monotonicity witness, or a
  disagreement with the independent implementation.

The finite-sample margin and signed bound violation are

```text
finite_sample_margin = risk_ucb - mean_loss
finite_sample_delta  = risk_ucb - 0.05.
```

The diagnostic Hoeffding upper bound is fixed as

```text
hoeffding_ucb = min(1, mean_loss + sqrt(log(1/delta) / (2*120))).
```

It is published for interpretability only and never changes `phase_decision`.

Primary method references:

- Waudby-Smith and Ramdas, *Estimating means of bounded random variables by betting*, JRSS-B
  2024: https://doi.org/10.1093/jrsssb/qkad009
- Maurer and Pontil, *Empirical Bernstein Bounds and Sample Variance Penalization*, COLT 2009:
  https://arxiv.org/abs/0907.3740
- Angelopoulos et al., *Conformal Risk Control*, ICLR 2024:
  https://openreview.net/pdf?id=33XGfHLtZg

## 5. Preregistered decision rule

The decision is one of exactly three values:

```text
confirmed
empirically_met_but_inconclusive
failed
```

The rule is evaluated without rounding:

| Condition | Decision |
|---|---|
| `coverage >= 0.10` and `risk_ucb <= 0.05` | `confirmed` |
| `coverage >= 0.10`, `mean_loss <= 0.05`, and `risk_ucb > 0.05` | `empirically_met_but_inconclusive` |
| `coverage < 0.10` or `mean_loss > 0.05` | `failed` |

`failed` carries the exact applicable reasons from
`insufficient_coverage` and `empirical_risk_exceeds_target`. The inconclusive outcome carries
`finite_sample_upper_bound_exceeds_target`. The confirmed outcome has no reason strings.

Exactly 12 preregistered stratum diagnostics are also reported: stratum identity, ROI count,
trusted and total pixel counts, coverage, mean loss, and maximum loss. They are descriptive only.
No per-stratum threshold, confidence interval, gate, weighting change, multiple-comparison claim,
or decision override is allowed.

An unfavorable or inconclusive outcome is final. Alpha, coverage, threshold, score, seeds, loss,
bet grid, confidence error, and sample membership may not be changed to rescue it.

## 6. One-time access authorization and state machine

### 6.1 Pre-access readiness and permit

Implementation ends first at a clean, committed local `main` revision whose scoped tests, synthetic
end-to-end run, replay, copied-bundle verifier, CLI help, lint, compilation, policy checks, and
canonical repeatability all pass without reading `internal_test` data.

That revision produces a Git-safe readiness request containing only code/evidence/manifest metadata
digests and the proposed evaluation ID. The readiness request remains outside Git and is not an
access credential. After reviewing readiness, the user must explicitly authorize real
`internal_test` access. Only then may a separate canonical
access permit be written as
`artifacts/phase2b3c/sen2naipv2-internal-test-access-authorization-v1.json` and committed. The permit
binds:

- schema and one immutable evaluation ID;
- exact implementation revision and allowlisted computation-tree digest;
- exact B publication identities and post-manifest/input-audit digests;
- fixed `alpha`, minimum coverage, `delta`, `D`, score, risk, threshold, seeds, and sample count;
- `one_time_evaluation=true`; and
- the statement that explicit user authorization was recorded before access.

No routine CLI auto-generates or self-approves this permit. Formal commands require it and prove
that no computation file or dependency lock changed since its implementation revision. A commit
that adds only the reviewed permit may descend from that revision.

### 6.2 Persistent access ledger

The first authorized real-data command atomically creates a locked ledger under the confirmed
persistent storage root before it can reach any test asset or test cache. The ledger is bound to
the permit, evaluation ID, code tree, environment inventory, manifest, and B acceptance digest.
Its monotone states are:

```text
reserved -> pixels_opened -> caches_complete -> bundle_complete -> accepted
                                      \-> invalidated
```

`pixels_opened` is committed immediately before the first permitted pixel or test-cache read, so a
crash in the small gap is conservatively treated as consumed access. State transitions are atomic,
append-only in content, digest chained, and guarded by one lock. Existing state may never be
deleted, reset, or replaced by the workflow.

Before `pixels_opened`, an exact-identity restart is allowed. Afterwards, only the same permit,
code tree, dependencies, scientific configuration, membership, and storage root may resume. A
power loss, OOM, or network interruption may reuse individually atomic cache entries after their
full identity and tensor digests pass. The formal replay and independent verifier are declared
parts of this same one-time evaluation and may reread the same fixed data solely to recompute the
same preregistered result.

Any scientific/code change, mismatched ledger, alternate storage root, second evaluation ID,
unexpected output, or attempt to create a second formal result after `pixels_opened` invalidates
the evaluation. A code defect discovered after access may be documented, but fixing it and rerunning
cannot yield a Phase 2B3-C confirmation without a newly sourced external holdout and a new phase.

The ledger prevents accidental repeated evaluation within the supported workflow; it is not a
claim that software can stop a privileged operator from copying or bypassing the data.

## 7. Commands and execution stages

Phase 2B3-C implements these fixed surfaces:

```text
trustsr-phase2b3c preflight
trustsr-phase2b3c evaluate
trustsr-phase2b3c evaluation-replay
trustsr-phase2b3c-verify
```

Only operational paths are accepted: project root, evidence directory, persistent storage root,
manifest, access permit, candidate bundle, and optionally an LDSR model directory on `evaluate`.
Every storage-using command requires explicit persistent-storage confirmation. No scientific,
sample-limit, sample-ID, split, seed, device, or statistic override exists.

`preflight` is metadata-only. It validates Git, B evidence, permit/readiness state when present,
the complete manifest, exact 120-member test selection, storage/mount capacity, environment, locks,
and output boundaries. It does not open image assets, enumerate test caches, construct LDSR, or
advance the access ledger.

After authorization, `evaluate` advances the ledger, loads only the frozen test pairs, derives
authoritative input receipts, and probes all 600 exact K5 prediction identities before constructing
LDSR. If the cache is complete, it completes on CPU. If any prediction is missing, it stops before
model construction and reports the exact count without emitting metrics. A later same-identity
resume may use `--ldsr-model-dir` only after separate GPU authorization.

Once predictions are complete, `evaluate` computes the frozen score/risk maps, per-ROI loss and
coverage, statistics, diagnostics, result, cache audit, runtime inventory, and an immediate
inference-free reconstruction. It publishes an atomic external bundle only if reconstruction is
byte-identical.

`evaluation-replay` cannot accept a model path and cannot run inference. It reloads authoritative
inputs and verified K5 caches, independently recomputes every downstream map/statistic/result, and
requires byte-identical scientific outputs.

`trustsr-phase2b3c-verify` first performs metadata-only candidate verification, then uses its own
pre-access-frozen implementations to reload the same 120 inputs and replay cache-derived
computation without model inference. Only this verifier may set `acceptance_authorized=true` and
publish Git-safe artifacts. Verification cannot calculate or report an alternative threshold,
score, loss, confidence method, or decision.

## 8. Evidence graph and publication

The external atomic bundle contains canonical documents with these schemas:

- `trustsr.phase2b3c-evaluation.v1`;
- `trustsr.phase2b3c-evaluation-cache-audit.v1`;
- `trustsr.phase2b3c-evaluation-runtime.v1`;
- `trustsr.phase2b3c-evaluation-replay.v1`;
- `trustsr.phase2b3c-access-ledger-snapshot.v1`; and
- `trustsr.phase2b3c-bundle-manifest.v1`.

The result contains frozen upstream identity, test sample/stratum counts, aggregate counts and
coverage, ROI loss aggregate, the exact betting and Hoeffding fields, 12 fixed stratum diagnostics,
radiometric aggregates, cache/map/input digests, producer revision, and the three-way decision. It
contains no raw pixels, tensors, model weights, per-pixel values, host paths, timestamps, endpoints,
credentials, or per-sample numerical loss/coverage.

The cache audit enumerates exactly the expected 600 prediction and 120 score/risk identities,
including per-sample tensor and cache digests but no numerical loss values. The runtime manifest is
a host-free inventory and cross-binding, not a computation receipt. The replay receipt binds the
byte-identical recomputation. The ledger snapshot binds the one-time state immediately before
bundle publication without exposing host paths or wall-clock times.

The digest graph remains acyclic:

```text
permit + ledger + verified inputs + verified caches -> result + cache audit + runtime
result + cache audit + runtime -> replay
permit + ledger snapshot + result + cache audit + runtime + replay -> bundle manifest
independently verified copied bundle + computation replay -> acceptance
```

After successful independent verification, the evaluation publication adds only:

```text
artifacts/phase2b3c/sen2naipv2-internal-test-evaluation-v1.json
artifacts/phase2b3c/sen2naipv2-internal-test-evaluation-cache-audit-v1.json
artifacts/phase2b3c/sen2naipv2-internal-test-evaluation-acceptance-v1.json
```

The acceptance record binds all bundle digests, the permit and implementation revision, verifier
revision, one-time ledger terminal state, independent cache-computation replay, and the exact phase
decision. Publication of any of the three decision values completes Phase 2B3-C; only `confirmed`
supports a later claim that the preregistered internal check confirmed the balanced-design target.
The earlier access permit remains a fourth, non-result authorization artifact; the external
readiness request is never committed.

## 9. Fail-closed and leakage policy

The phase stops before scientific publication on any mismatch in evidence, ancestry, code tree,
permit, ledger, membership, asset, tensor, model, cache, threshold, map computation, statistic,
runtime, replay, canonical JSON, path, or digest.

Before dedicated access authorization, tests and commands must reject attempts to:

- open any selected test image or test-cache path;
- construct a model or inspect a GPU;
- infer test metrics from filenames, stale output, or partial caches;
- pass development/calibration rows into the evaluator; or
- write an access permit, advance the ledger, or publish a scientific result.

After `pixels_opened`, logs may contain only progress counts, state transitions, and opaque digests.
They must not print per-ROI loss, partial means, partial coverage, stratum values, running bounds, or
intermediate decisions. Statistics become visible together only in the completed atomic bundle.

No cloud path, endpoint, credential, token, hostname, user name, GPU UUID, raw timestamp, raw tensor,
or model weight may enter Git. Cloud code and environments are disposable and never a merge source.
Remote execution must use the server's base Python environment; `uv`, `.venv`, environment creation,
and dependency mutation are prohibited there.

## 10. CPU/GPU boundary and completion gates

All implementation work is local CPU work: evidence and permit validation, selector/loader gates,
one-time state machine, statistical code, schemas, canonical I/O, cache probe, replay, verifier,
hostile-input tests, and tiny synthetic end-to-end tests. Tests must construct CPU tensors even if
CUDA is reported available.

Before requesting real-data or GPU authorization, local verification must prove:

- exact upstream evidence and frozen payload validation with mutation coverage;
- metadata-only preflight and pre-authorization pixel/cache/model tripwires;
- exact test-only selection and all 120/12-by-10/round/group invariants using synthetic manifests;
- hand-calculated e-values, monotonic inversion, endpoint cases, bisection conservatism, permutation
  invariance, heterogeneous-mean simulations, and an implementation-independent statistical oracle;
- exact loss, coverage, stratum diagnostics, boundary comparisons, and all three decisions;
- one-time ledger transitions, restart rules, invalidation, lock contention, and partial-write
  recovery;
- complete-cache CPU operation, missing-cache stop before model construction, and inference-free
  replay;
- copied-bundle independent verification, digest DAG, canonical repeatability, artifact allowlist,
  and secret/path scans; and
- scoped pytest, Ruff, `compileall`, every CLI help surface, `git diff --check`, and a clean attached
  `main` checkpoint.

GPU is needed only if the authorized test K5 cache probe reports missing predictions. At that point
the operator must ask the user to start/authorize the GPU server; until then the server may remain
off. The remote command must use `/opt/conda/bin/python` in base, never `uv` or a new environment.

Real Phase 2B3-C is complete only after the single authorized evaluation, immediate reconstruction,
explicit inference-free replay, copied-bundle independent verification, acceptance, and three-file
publication all succeed and the ledger reaches `accepted`.

## 11. Explicitly deferred work

- changing or comparing thresholds, alphas, confidence errors, bet grids, coverage gates, scores,
  risks, seeds, models, or crops;
- tuning from any `internal_test` aggregate, stratum, per-ROI, pixel, or cache observation;
- per-stratum inferential claims or multiple-comparison procedures;
- grouped, Mondrian, weighted, adaptive, or cross-domain conformal calibration;
- trusted/untrusted blending, fallback products, mosaics, and downstream tasks;
- reusing this holdout for model selection or a second confirmation; and
- external OpenSR-Test, SEN2NEON, or paper-table experiments.
