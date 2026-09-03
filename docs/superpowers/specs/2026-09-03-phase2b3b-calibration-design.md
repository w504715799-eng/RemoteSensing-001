# Phase 2B3-B: Calibration-only conformal threshold design

**Date:** 2026-09-03

**Status:** Draft for scientific-parameter approval; engineering boundaries are frozen

**Upstream:**
`docs/superpowers/specs/2026-09-01-phase2b3a-development-score-audit-design.md`

**Next phase:** Phase 2B3-C, a separately specified one-time `internal_test` evaluation

## 1. Purpose and hard boundary

Phase 2B3-B fits exactly one global ROI-level conformal threshold on the 120 frozen
SEN2NAIPv2 `calibration` ROIs. It consumes the score selected by Phase 2B3-A and may not
compare, replace, or tune that score. The phase ends after publishing a compact, replayable
calibration result and acceptance record.

This phase may read `calibration` LR/HR pixels only after this specification's two numerical
decisions in Section 5 are approved. It must never read `internal_test` pixels, predictions,
scores, risks, or metrics. Development pixels are also out of scope; Phase 2B3-B consumes only
the immutable Git-safe Phase 2B3-A evidence.

The design deliberately creates Phase 2B3-B-specific data, evaluation, CLI, and verification
modules. Existing Phase 2B3-A modules remain development-only and are not parameterized to accept
arbitrary splits.

## 2. Frozen upstream evidence

The scientific trust anchor is the six-file publication commit
`b386d4b38c9f3725107eed178829955d442f5601`, not the current branch prose or a cloud checkout.
The following repository files and exact SHA-256 digests are immutable inputs:

| File | SHA-256 |
|---|---|
| `sen2naipv2-development-smoke-v2.json` | `2c962de9651f3d2cc65f321877564c3509d8d4414801fd5b445503aed5dbb947` |
| `sen2naipv2-development-smoke-cache-audit-v2.json` | `88144cb6dcfc4d8fc68289188aa909fd2e597304b95e47d23f9d0f0c17127a47` |
| `sen2naipv2-development-smoke-acceptance-v2.json` | `5ac7bd232ce2a0897b9b93a35f896de4f5641a0adc9f42ce3d1f6986f1a054d2` |
| `sen2naipv2-development-score-audit-v1.json` | `5bb0e5138d6ed1df6c65744556be02ccd48b77d3288df39630d16fbd9cd2dce9` |
| `sen2naipv2-development-score-cache-audit-v1.json` | `d61c36e2180a2dc3468d4d9aba083ac0925d163ac2bb910e0227138e9fa249f1` |
| `sen2naipv2-development-score-acceptance-v1.json` | `34741fe788cac6e28c6d8b1ce2fd96335b608e1b3e6ffb29e82ac064a2118227` |

Before any model construction or pixel loading, Phase 2B3-B must independently verify all six
file bytes, schemas, internal result/cache digests, A1 gates, A2 `phase_decision="freeze_score"`,
and equality between the A2 result and acceptance `frozen_score`. A later code revision is allowed
only when Git proves it descends from the recorded calculation commit; a newer revision does not
replace any frozen evidence value.

The exact selected score identity is:

- name: `ldsr_variance_k5`;
- operator: population ensemble variance (`correction=0`) followed by mean over RGBN bands;
- seeds: exactly `3407, 3408, 3409, 3410, 3411`;
- score-selection risk: `local_l1_risk(window=9)`;
- risk upper bound: `1.0`;
- normalization: `uint16_saturate_10000_divide_10000_v2`;
- crop policy: `center_crop_lr_1_hr_4_v1`;
- bands: `[B04, B03, B02, B08]`;
- scale: `4`;
- Phase 2B1-B post-manifest SHA-256:
  `c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a`;
- Phase 2B2-A input-audit SHA-256:
  `fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b`;
- Phase 2B3-A calculation revision:
  `58694420c3c0e11d495953a1963c71b997261601`.

Changing any identity field is a hard failure, not a new Phase 2B3-B run.

## 3. Calibration sample contract

The strict Phase 2B1-B 360-row post-extraction manifest remains the only sample source. A
Phase 2B3-B selector validates the complete manifest and then returns exactly the 120 records with
`split="calibration"`, in canonical manifest order. It must establish:

- 120 unique sample IDs, selection hashes, and spatial group IDs;
- 12 exact `(days_between, correlation_bin)` strata with 10 ROIs each;
- selection rounds 1 through 10, with one item per stratum per round;
- both assets present and still matching their size, digest, shape, dtype, CRS, transform, nodata,
  range, timestamp, and canonical relative path;
- no accepted record has split `development` or `internal_test`.

The complete 360-row metadata may be read to verify the frozen manifest. The selector must not
open image paths while validating metadata. Only the returned calibration records may cross the
pixel-loading boundary. The loader then applies the same byte, sidecar, geometry, dtype, nodata,
mask, alignment, crop, radiometric saturation, and normalization checks used by Phase 2B3-A.

Tests use generated metadata and tiny tensors only. No repository test fixture may contain a real
SEN2NAIPv2 pixel, host path, endpoint, credential, model weight, or prediction cache.

## 4. Calibration mathematics

For each calibration ROI `i`, five frozen LDSR predictions produce

```text
score_i[p] = mean_band Var_seed(prediction_i[seed, band, p], correction=0)
risk_i[p]  = box_mean(mean_band |central_prediction_i - hr_i|, window=9)
```

The central prediction used for risk is the seed-3407 LDSR prediction, matching the Phase 2B3-A
score audit. Each ROI, not each pixel, is one calibration unit. For a candidate score threshold
`t`:

```text
worst_i(t) = max({risk_i[p] : score_i[p] <= t} union {0})
bound(t)   = (sum_i worst_i(t) + 1) / 121
```

The calibrated threshold is the greatest finite observed score with `bound(t) <= alpha`. If no
finite score is admissible, the internal threshold is `-inf`, JSON uses `null`, and
`all_abstain=true`. Score ties are admitted together through `score <= threshold`; they may not be
split to manufacture coverage.

The implementation reuses `trustsr.calibration.conformal` only after its optimized exact scan is
shown equivalent to the original definition on hand-calculated cases, ties, non-monotonic
cross-ROI risk events, all-abstain inputs, and deterministic randomized small examples. It must not
flatten pixels and treat them as independent calibration samples.

## 5. Numerical decisions requiring approval

The approved roadmap and Phase 2B3-A specification intentionally did not choose a formal target
`alpha` or a minimum useful coverage. These are scientific parameters, so the synthetic Phase 2A
CLI default `0.27` is prohibited as a formal value.

The recommended preregistration is:

- `alpha = 0.05`, interpreted as a five-percentage-point reflectance local-L1 ROI risk target;
- minimum calibration pixel coverage `0.10` for permission to design Phase 2B3-C.

This recommendation is intentionally simple, application-interpretable, and selected before any
calibration pixel or score is read. Alternatives considered were `alpha=0.10` (more permissive but
weaker hallucination control) and a grid of alphas (more descriptive but creates a calibration-set
selection degree of freedom). A formal grid may be reported only in a later exploratory artifact;
it cannot select the Phase 2B3-C threshold.

Until both recommended values are explicitly approved or replaced, implementation may cover
evidence validation, metadata isolation, exact conformal mathematics, schemas, synthetic tests,
and command dry-runs. It must stop before real calibration pixel loading, GPU prediction, threshold
fitting, or B acceptance publication.

All-abstain is a statistically valid calibration result and must be published honestly, but it
fails the minimum-coverage gate and stops before Phase 2B3-C. Any finite threshold with calibration
coverage below the approved minimum also stops before C; the threshold must not be relaxed.

## 6. Commands and immutable stages

Phase 2B3-B uses fixed commands with no score, seed, risk-window, sample-limit, sample-ID, alpha, or
coverage override flags:

```text
trustsr-phase2b3b preflight
trustsr-phase2b3b calibration
trustsr-phase2b3b calibration-replay
trustsr-phase2b3b-verify
```

Operator arguments may identify the storage root, exact frozen manifests/evidence, model roots,
and output root. These paths are runtime-only and absent from Git-safe artifacts. Every stage
requires the same explicit persistent-storage confirmation, canonical non-symlink paths, mount and
capacity checks, lock, empty/resumable stage boundary, exact Git ancestry, dependency versions,
and model inventories established by Phase 2B3-A.

`preflight` validates identities and resources without loading image pixels or constructing a
model. `calibration` validates and loads only calibration assets, generates or verifies the frozen
K5 prediction and score caches, fits the threshold once, and writes an atomic result bundle.
`calibration-replay` performs no inference: it independently reloads verified caches and must emit
scientific JSON byte-identical to `calibration`. The independent verifier accepts a copied bundle
without trusting the producing CLI implementation.

## 7. Result and evidence schemas

The runtime bundle contains:

- `trustsr.phase2b3b-calibration.v1`;
- `trustsr.phase2b3b-calibration-cache-audit.v1`;
- `trustsr.phase2b3b-calibration-runtime.v1`;
- `trustsr.phase2b3b-calibration-replay.v1`;
- `trustsr.phase2b3b-bundle-manifest.v1`.

After offline verification, Git may receive only a small result, cache audit, and acceptance JSON:

- `sen2naipv2-calibration-conformal-v1.json`;
- `sen2naipv2-calibration-conformal-cache-audit-v1.json`;
- `sen2naipv2-calibration-conformal-acceptance-v1.json`.

The result contains exactly the frozen upstream identities, calibration-only sample/stratum counts,
score/risk configuration, numerical target, `threshold` or `null`, `all_abstain`, `risk_bound`,
calibration pixel coverage, trusted/total pixel counts, per-sample cache identities and tensor
digests, radiometric saturation aggregates, environment package/model provenance, and producer
revision. It contains no raw pixels, tensors, model weights, per-pixel values, host paths,
timestamps, endpoints, credentials, or `internal_test` metadata.

The cache audit enumerates every expected prediction/score identity and proves no extra split or
seed entered the phase. The acceptance record binds result/cache/runtime/replay/bundle digests,
records offline-verifier success and byte-identical replay, and exposes one explicit phase decision:

- `freeze_calibration` when threshold is finite and coverage meets the approved minimum;
- `stop_insufficient_coverage` otherwise.

The `freeze_calibration` payload is the only B-to-C interface. It includes the six A evidence
digests, frozen score/risk identities, approved alpha and coverage gate, threshold, risk bound,
calibration size, trusted/total counts, coverage, B result digest, post-manifest/input-audit digests,
normalization/crop policies, model inventories, and producer revision. Phase 2B3-C may not infer,
recompute, or amend any of these values.

## 8. Fail-closed behavior and replay

The phase stops before writing scientific output if any of the following occurs:

- an upstream file, schema, digest, A gate, frozen-score field, manifest, input audit, code ancestry,
  model inventory, or policy differs;
- a selected record is not calibration, the 120/12-by-10/round/group contract fails, or an asset
  changes;
- any development or `internal_test` pixel path is opened;
- a cache identity has the wrong sample, seed, model, tensor digest, score operator, or risk window;
- a tensor is malformed, non-finite, outside the allowed reflectance/risk range, or inconsistently
  saturated;
- the calibration implementation disagrees with the independent verifier;
- output already exists with non-identical bytes, a lock is held, or replay is not byte-identical.

OOM, interruption, or partial writes leave no accepted result. Atomic cache entries may be resumed
only after full identity verification. A failed run cannot lower alpha, lower the coverage gate,
change seeds, switch score, or inspect `internal_test` to rescue the result.

## 9. CPU and GPU execution split

Local CPU work includes all strict loaders, metadata selectors, evidence validators, schemas,
canonical writers, optimized conformal math, replay, offline verification, hostile-input tests,
shell boundary tests, and tiny synthetic end-to-end tests. Local tests must remain valid when CUDA
is reported available but must construct CPU tensors only.

GPU authorization is required only to generate missing K5 LDSR predictions for the 120 calibration
ROIs. A future cloud run may use up to four isolated workers if a benchmark justifies it, but one
worker remains the default because the Phase 2B3-A four-worker measurement used 22,687 MiB without
reducing elapsed time. Cloud code is disposable and never a merge source; local `main` remains
authoritative.

## 10. Verification and completion

Before real calibration begins, tests must prove:

- exact A evidence and Git ancestry validation, including one mutation test per frozen identity;
- complete-manifest validation followed by calibration-only selection, with hostile attempts to
  pass development or `internal_test` records rejected before pixel loading;
- K5 cache identity and R9 risk identity, including seed/order/tie mutations;
- hand-calculated and randomized equivalence of the optimized conformal scan;
- strict JSON fields, `-inf`/`null` conversion, all-abstain and insufficient-coverage decisions;
- inference-free replay and an implementation-independent bundle verifier;
- canonical repeatability, artifact allowlisting, secret/path scanning, and partial-write cleanup.

Phase 2B3-B is complete only when the numerical decisions are approved, all 120 calibration ROIs
are present, the formal run and replay agree byte-for-byte, the offline verifier passes, the three
Git-safe files are reviewed, and the acceptance decision is published. A
`stop_insufficient_coverage` result completes B as a negative result but forbids Phase 2B3-C.

If `freeze_calibration` is published, the next action is to write and approve a separate
Phase 2B3-C specification. That specification must preregister the one-time empirical risk
violation statistic and finite-sample tolerance before any `internal_test` pixel is opened.

## 11. Explicitly deferred work

- all `internal_test` evaluation;
- grouped, Mondrian, weighted, adaptive, or cross-domain conformal methods;
- alpha grids used for model or threshold selection;
- new scores, risk functions, trained models, or model fine-tuning;
- trusted/untrusted pixel fallback products and downstream tasks;
- external OpenSR-Test, SEN2NEON, or paper-table experiments.
