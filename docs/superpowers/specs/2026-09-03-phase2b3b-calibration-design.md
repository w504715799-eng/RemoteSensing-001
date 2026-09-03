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

### 7.1 Runtime manifest v1: exact metadata inventory, not a computation receipt

`trustsr.phase2b3b-calibration-runtime.v1` is one canonical JSON object. It is produced only
after the input receipt, cache audit, result, and recorded producer revision have each passed their
independent verifier. It is a host-free inventory and cross-document binding, not a second cache
audit and not a source of scientific parameters. Every object below has **exactly** the shown keys;
unknown keys, omitted keys, non-JSON values, non-canonical JSON, non-lowercase 64-hex digests, and
values with the wrong built-in JSON type fail closed.

```json
{
  "schema": "trustsr.phase2b3b-calibration-runtime.v1",
  "phase": "calibration",
  "verification_scope": "metadata_inventory_only",
  "cache_computation_verified": false,
  "dependencies": {
    "python": {"major_minor": "3.12"},
    "uv_lock_sha256": "<64 lowercase hex>",
    "packages": {
      "numpy": "<version>",
      "opensr-model": "<version>",
      "rasterio": "<version>",
      "torch": "<version>",
      "trustsr": "<version>"
    }
  },
  "model_inventory": {
    "identity": {
      "name": "ldsr-s2-x4",
      "scale": 4,
      "implementation_schema_version": 1,
      "opensr_model_version": "<version>",
      "torch_version": "<version>",
      "cuda_runtime": "<version>",
      "checkpoint_name": "opensr-ldsrs2_v1_0_0.ckpt",
      "checkpoint_size": 1130715795,
      "checkpoint_sha256": "e2621e3912eb7c14867c3d20c9029607ba941be8e166dc09621860fcac27dc3a",
      "config_sha256": "ac76685d354bfec32e3e0641aef574bedd7d650402c97dbd0ade86304e69ca6f",
      "sampling_steps": 100,
      "sampling_eta": 0.95,
      "sampling_temperature": 1.0,
      "histogram_matching": true,
      "output_policy": "clip_to_[0,1]"
    },
    "seeds": [3407, 3408, 3409, 3410, 3411]
  },
  "inputs": {
    "post_manifest_sha256": "<64 lowercase hex>",
    "input_audit_sha256": "<64 lowercase hex>",
    "normalization_policy": "uint16_saturate_10000_divide_10000_v2",
    "crop_policy": "center_crop_lr_1_hr_4_v1",
    "bands": ["B04", "B03", "B02", "B08"],
    "scale": 4,
    "ordered_sample_ids_sha256": "<64 lowercase hex>",
    "ordered_membership_sha256": "<64 lowercase hex>",
    "input_receipt_sha256": "<64 lowercase hex>",
    "ordered_inputs_sha256": "<64 lowercase hex>"
  },
  "artifacts": {
    "result_sha256": "<64 lowercase hex>",
    "cache_audit_sha256": "<64 lowercase hex>",
    "map_evidence_sha256": "<64 lowercase hex>",
    "cache_audit_identity_digests": {
      "prediction_identities_sha256": "<64 lowercase hex>",
      "score_identities_sha256": "<64 lowercase hex>",
      "risk_receipts_sha256": "<64 lowercase hex>"
    }
  },
  "revision": {
    "producer_revision": "<40 lowercase hex>",
    "phase2b3a_calculation_revision": "<40 lowercase hex>",
    "phase2b3a_publication_commit": "<40 lowercase hex>"
  }
}
```

The angle-bracket strings above denote validated values, not literal JSON values. The top-level
key set is exactly `schema`, `phase`, `verification_scope`, `cache_computation_verified`,
`dependencies`, `model_inventory`, `inputs`, `artifacts`, and `revision`. The nested key sets are
also exactly those shown in the schema. `phase` is exactly `calibration`;
`verification_scope` is exactly `metadata_inventory_only`; and
`cache_computation_verified` is the built-in JSON boolean `false` (not `0`, a string, or a caller
choice).

Dependency values have one authority source each:

- `dependencies.python.major_minor` is derived from `sys.version_info.major` and `.minor`, and
  must match `^[0-9]+\\.[0-9]+$`.
- `dependencies.uv_lock_sha256` is SHA-256 of the raw repository-root `uv.lock` bytes in the
  clean verifier checkout, not a path, URL, lockfile excerpt, or resolver report.
- `dependencies.packages` has exactly the five allowlisted distribution names shown. Their values
  come from `importlib.metadata.version` for those distributions in the producing environment.
  Each package version, `model_inventory.identity.opensr_model_version`, `torch_version`, and
  `cuda_runtime` must match `^[0-9][0-9A-Za-z.+_-]*$` and must not contain, case-insensitively,
  `internal_test`, `token`, `secret`, or `host`. This preserves legitimate values such as
  `2.7.1+cu128`, `2.12.1+cu130`, and `13.0` without admitting paths, endpoints, credentials, or
  free-form host labels.

`model_inventory` is a compact projection, never a 600-entry duplicate. The runtime builder must
first independently validate the parsed cache audit, then revalidate every prediction provenance
with the cached calibration-model contract. It removes only `seed`, requires all 600 resulting
scientific identities to be equal, emits that one seed-independent identity, and emits the exact
ordered K5 seed list. `identity` is therefore the existing host-free
`CalibrationModelIdentity.as_dict()` projection with `seed` removed; it deliberately excludes raw
`checkpoint_url`, `device`, GPU identifiers, and all per-worker state. The current audit document
does not contain a top-level model-inventory field: this is a verifier-derived projection from its
already validated prediction identities, not an invented or separately trusted audit value.

`inputs`, `artifacts`, and `revision` are likewise projections with fixed sources, not caller
claims. A runtime builder must obtain them as follows:

| Runtime field(s) | Required verified source |
|---|---|
| `inputs.post_manifest_sha256`, `input_audit_sha256`, normalization/crop/bands/scale | frozen preflight/result `upstream` and `frozen.input`, checked for equality with the fixed Phase 2B3-A evidence |
| `inputs.ordered_sample_ids_sha256`, `ordered_membership_sha256`, `input_receipt_sha256`, `ordered_inputs_sha256` | `VerifiedCalibrationInputReceipt`, then checked equal to the result projection and authoritative preflight membership |
| `artifacts.result_sha256` | SHA-256 of the canonical committed `trustsr.phase2b3b-calibration.v1` bytes |
| `artifacts.cache_audit_sha256` and `cache_audit_identity_digests` | `verify_calibration_cache_audit` receipt; the first is its audit digest and the nested three are its existing prediction/score/risk identity digest projections |
| `artifacts.map_evidence_sha256` | result field, recomputed from the same verified audit score/risk projection |
| `revision.producer_revision` | result field after `verify_recorded_phase2b3b_revision` proves it exists, descends from both frozen A revisions, and is an ancestor of the clean verifier HEAD |
| `revision.phase2b3a_calculation_revision`, `phase2b3a_publication_commit` | frozen Phase 2B3-A evidence/result upstream projection |

No field in this schema carries alpha, a minimum-coverage gate, threshold, coverage, trusted or
total pixels, a risk bound, or a phase decision. Those values remain result data and are not
scientifically approved merely by appearing in a runtime inventory.

The digest DAG is intentionally one-way:

```text
verified input receipt ─┐
verified cache audit ───┼─> verified result ─┐
verified model identity ┤                    ├─> runtime manifest
verified revision ──────┘                    │
verified input/model/revision ───────────────┘

canonical result + canonical cache audit + canonical runtime ─> replay receipt
canonical result + canonical cache audit + canonical runtime + replay receipt ─> bundle manifest
verified bundle + replay receipt ─> any later acceptance record
```

Runtime creation must not read or contain a replay digest, replay receipt, bundle manifest digest,
acceptance decision, or acceptance digest. Replay is allowed to bind the runtime-manifest digest
only after the runtime bytes exist; bundle and any later acceptance record bind replay only after
replay exists. Result composition, cache-audit construction, input receipt construction, model
identity construction, and conformal fitting must not accept runtime bytes or a runtime digest as
an input. This forbids self-consistent circular evidence.

The manifest must contain no host or operational data: no literal paths, working directories,
filenames, timestamps, timezones, hostnames, user names, process IDs, GPU/CUDA UUIDs or device
ordinals, driver versions, endpoints, URLs, credentials, tokens, secrets, environment-variable
values, `development`, or `internal_test`. It may not contain raw pixels, tensors, cache locations,
prediction/score/risk values, or per-sample model entries.

The runtime manifest proves only that these verified metadata projections were assembled under one
declared dependency/model inventory. It does **not** prove cache pixels, prediction execution,
score computation, risk computation, byte-identical replay, or scientific acceptance, and it must
never independently authorize acceptance.

After offline verification, Git may receive only a small result, cache audit, and acceptance JSON:

- `sen2naipv2-calibration-conformal-v1.json`;
- `sen2naipv2-calibration-conformal-cache-audit-v1.json`;
- `sen2naipv2-calibration-conformal-acceptance-v1.json`.

The result contains exactly the frozen upstream identities, calibration-only sample/stratum counts,
score/risk configuration, numerical target, `threshold` or `null`, `all_abstain`, `risk_bound`,
calibration pixel coverage, trusted/total pixel counts, per-sample cache identities and tensor
digests, radiometric saturation aggregates, and producer revision. Package and model inventory
belong only to the runtime manifest above. It contains no raw pixels, tensors, model weights,
per-pixel values, host paths, timestamps, endpoints, credentials, or `internal_test` metadata.

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
