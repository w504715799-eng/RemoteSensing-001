# Codex handoff: Phase 2B3-B local CPU engineering

Date: 2026-09-03 (Asia/Shanghai)

## Repository checkpoint

- Integration branch: `main`; local Git is authoritative. Cloud code is disposable and must never
  be merged back.
- Code checkpoint immediately before this handoff-document update:
  `a7ff6a044245c7ab35e290b364c77fcd460573a1`.
- The handoff commit is necessarily a child of that code checkpoint. Runtime wiring was subsequently
  integrated as `d8b3cac`; the main coordinator must record the final integrated `main` SHA in the
  final report. This document does not identify its own commit as a code checkpoint.
- Do not infer readiness from a branch name or this document. Resume only from a clean, attached
  commit and rerun the scoped gates below.

## Frozen Phase 2B3-A baseline

Phase 2B3-A saturation-v2 compute, replay, offline verification, evidence publication, and the
post-publication worker benchmark are complete. The GPU can remain off. The operational default is
one worker: the four-worker benchmark used `22687 MiB` and reached `100%` utilization without
improving end-to-end elapsed time.

The immutable trust anchors are:

- Phase 2B3-A calculation revision:
  `58694420c3c0e11d495953a1963c71b997261601`;
- six-file evidence publication commit:
  `b386d4b38c9f3725107eed178829955d442f5601`;
- post-publication worker integration commit:
  `b444c2d64bb4bc512a2b3bc06e04e16af07df612`;
- Phase 2B1-B post-manifest SHA-256:
  `c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a`;
- Phase 2B2-A input-audit SHA-256:
  `fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b`;
- normalization policy `uint16_saturate_10000_divide_10000_v2`;
- crop policy `center_crop_lr_1_hr_4_v1` and bands `B04,B03,B02,B08` at scale 4.

The six immutable Git-safe evidence files are:

| File | SHA-256 |
|---|---|
| `sen2naipv2-development-smoke-v2.json` | `2c962de9651f3d2cc65f321877564c3509d8d4414801fd5b445503aed5dbb947` |
| `sen2naipv2-development-smoke-cache-audit-v2.json` | `88144cb6dcfc4d8fc68289188aa909fd2e597304b95e47d23f9d0f0c17127a47` |
| `sen2naipv2-development-smoke-acceptance-v2.json` | `5ac7bd232ce2a0897b9b93a35f896de4f5641a0adc9f42ce3d1f6986f1a054d2` |
| `sen2naipv2-development-score-audit-v1.json` | `5bb0e5138d6ed1df6c65744556be02ccd48b77d3288df39630d16fbd9cd2dce9` |
| `sen2naipv2-development-score-cache-audit-v1.json` | `d61c36e2180a2dc3468d4d9aba083ac0925d163ac2bb910e0227138e9fa249f1` |
| `sen2naipv2-development-score-acceptance-v1.json` | `34741fe788cac6e28c6d8b1ce2fd96335b608e1b3e6ffb29e82ac064a2118227` |

Do not overwrite, relabel, or recompute these artifacts. The completed recovery, rerun, benchmark,
and reproduction procedure remains in [the Phase 2B3-A cloud runbook](phase2b3a-cloud-runbook.md).

## Current Phase 2B3-B boundary

The active design is
[Phase 2B3-B: Calibration-only conformal threshold design](superpowers/specs/2026-09-03-phase2b3b-calibration-design.md).
It remains a draft pending scientific-parameter approval.

At the code checkpoint, local CPU engineering and synthetic tests cover:

- frozen Phase 2B3-A evidence validation and clean Git revision/ancestry gates;
- complete-manifest validation, exact 120-member calibration selection, strict pair loading, and
  authoritative input receipts binding frozen membership to loaded LR/HR tensor identities;
- frozen LDSR-S2-x4 scientific identity, fixed K5 seeds `3407..3411`, prediction/cache identities,
  score and risk maps, cache audit, and inference-free cache replay;
- radiometric receipts plus an independent verifier, exact conformal fitting, final result
  composition, replay receipt composition, and canonical atomic bundle I/O;
- an independent final-result verifier that revalidates trusted local Git ancestry;
- a semantic bundle verifier that cross-checks candidate documents but is explicitly metadata-only,
  including exact runtime verification and digest cross-binding;
- a downstream computation replay verifier that recomputes score, risk, fit, audit, and canonical
  result from caller-supplied loaded inputs and cache entries; and
- the exact design section 7.1 runtime manifest module.

The runtime builder internally reverifies the raw result and raw input authority. Offline runtime
verification invokes the authoritative result verifier again and reconstructs input authority from
that verified result. Runtime contains no replay, bundle, or acceptance digest, so the evidence graph
remains acyclic. Runtime is host-free: it contains no operational path or arbitrary filename; the
fixed scientific checkpoint basename in `checkpoint_name` is the sole schema-approved
filename-shaped exception.

Runtime wiring is integrated: candidate bundle verification calls the exact runtime verifier and
cross-binds its runtime, result, cache-audit, input, map-evidence, and revision identities. This does
not elevate the bundle beyond metadata consistency or authorize acceptance.

## Verification scopes are not interchangeable

| Layer | Positive scope | Explicitly not authorized or proved |
|---|---|---|
| final-result verifier | `metadata_consistency_only`; `cache_computation_verified=false` | cache computations and acceptance |
| semantic bundle verifier | `metadata_consistency_only`; `cache_computation_verified=false` | pixel/model computation and acceptance |
| computation replay verifier | `cache_computation_replay`; `cache_computation_verified=true` | LDSR inference, independent membership authority, and acceptance |
| runtime verifier | `metadata_inventory_only`; `cache_computation_verified=false` | computation and acceptance |

For computation replay, `prediction_inference_verified=false`,
`membership_authority_verified=false`, and `acceptance_authorized=false`. Its positive flag means
only that cache-derived downstream score/risk/fit/result computations were replayed. A structurally
self-consistent receipt is never a scientific authorization credential.

## Implemented command surface

Metadata-only preflight is available:

```text
trustsr-phase2b3b preflight \
  --project-root PROJECT_ROOT \
  --evidence-dir EVIDENCE_DIR \
  --storage-root STORAGE_ROOT \
  --manifest MANIFEST
```

Candidate bundle verification is also implemented:

```text
trustsr-phase2b3b-verify \
  --bundle BUNDLE \
  --project-root PROJECT_ROOT \
  --evidence-dir EVIDENCE_DIR \
  --storage-root STORAGE_ROOT \
  --manifest MANIFEST
```

This verifier reports metadata consistency only and always emits
`acceptance_authorized=false`. There is still no formal `calibration` or
`calibration-replay` command and no acceptance/publication command. Library functions exercised
with synthetic inputs do not authorize a real run.

## Unapproved scientific parameters and hard stop

The design recommends, but the user has not approved:

- `alpha = 0.05`;
- minimum calibration pixel coverage `0.10` for permission to design Phase 2B3-C.

The synthetic Phase 2A default `0.27` is prohibited. Neither value may be selected after observing
calibration pixels, scores, risks, coverage, or the fitted threshold.

Until the user explicitly approves or replaces both values:

- do not open real calibration pixel files;
- do not construct or run the model;
- do not use the GPU or cloud server;
- do not generate real K5 calibration predictions or scores;
- do not fit or publish a formal threshold, runtime bundle, replay, result, or acceptance record;
- do not inspect any `internal_test` pixels, caches, predictions, scores, risks, or metrics.

## GPU and cloud status

No GPU or cloud server is needed for the remaining local CPU implementation and verification work.
Keep the server off. Request the user's permission before starting it, and only when all of the
following are true:

1. `alpha` and minimum coverage have been explicitly approved;
2. the formal calibration, replay, acceptance, and independent verification surfaces are integrated
   and locally verified;
3. the exact clean reviewed commit is ready for a disposable cloud checkout; and
4. verified calibration K5 cache entries are missing and must be generated.

Cloud-side code, logs, tensors, caches, models, paths, endpoints, and credentials must never enter
Git or become a merge source.

## Next local work

1. Run the final integrated CPU tests and static gates and record the exact final `main` SHA.
2. Obtain explicit approval or replacement of `alpha` and minimum coverage before opening real
   calibration data or exposing any executable scientific path.
3. Only after approval, finish and verify the fixed formal calibration, inference-free replay,
   acceptance, and publication command surfaces without scientific override flags.
4. Request separate GPU/cloud permission only if verified K5 calibration cache entries are missing.
5. Run calibration once, replay without inference, independently verify the copied bundle, review
   the Git-safe publication files, and publish either `freeze_calibration` or
   `stop_insufficient_coverage` without relaxing preregistered gates.

Phase 2B3-C remains out of scope. It requires a separate written and approved specification before
any one-time `internal_test` access.

## Verification at handoff

Individual workstreams ran targeted tests, Ruff, and `git diff --check` before their commits. The
main coordinator owns the final integrated run:

```bash
uv run pytest -q \
  tests/cli/test_phase2b3b.py \
  tests/cli/test_phase2b3b_verify.py \
  tests/evaluation/test_phase2b3b_*.py \
  tests/evaluation/test_calibration_*.py
uv run ruff check \
  src/trustsr/cli/phase2b3b.py \
  src/trustsr/cli/phase2b3b_verify.py \
  src/trustsr/evaluation \
  tests/cli/test_phase2b3b.py \
  tests/cli/test_phase2b3b_verify.py \
  tests/evaluation
uv run trustsr-phase2b3b --help >/dev/null
uv run trustsr-phase2b3b preflight --help >/dev/null
uv run trustsr-phase2b3b-verify --help >/dev/null
git diff --check
git status --short --branch
git rev-parse HEAD
```

Final integrated/full-suite status: **PENDING FINAL COORDINATOR RUN**.

Do not claim Phase 2B3-B complete from targeted workstream tests, metadata-only receipts, or this
handoff draft.

## Persistent stop conditions

- Accept only the exact 120 frozen `calibration` records in canonical manifest order.
- Fail closed on any evidence, revision, membership, asset, tensor, model, policy, cache, replay,
  runtime, canonical JSON, path, or digest mismatch.
- Never use development or `internal_test` observations to tune Phase 2B3-B.
- Never lower alpha or minimum coverage to rescue an unfavorable result.
- Never treat a structurally self-consistent receipt or transport-valid bundle as scientific
  authority without independent manifest, Git ancestry, and semantic verification.
- Preserve the six published Phase 2B3-A evidence files as the immutable upstream baseline.
