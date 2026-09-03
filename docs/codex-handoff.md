# Codex handoff: Phase 2B3-B local CPU engineering

Date: 2026-09-03 (Asia/Shanghai)

## Repository checkpoint

- Integration branch: `main`; local Git is authoritative. Cloud code is disposable and must never
  be merged back.
- This handoff was written from
  `adb15824cc7381244195098968c8dc98344cea37`.
- The main coordinator must replace that checkpoint with the final integrated `main` SHA after the
  remaining parallel workflow is reviewed and merged.
- Do not infer readiness from a branch name or this document; resume only from the recorded clean,
  attached commit and re-run the scoped gates below.

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

At this checkpoint, local CPU engineering and synthetic tests cover:

- frozen Phase 2B3-A evidence validation and clean Git revision/ancestry gates;
- complete-manifest validation, exact 120-member calibration selection, and strict
  calibration-only pair loading;
- frozen LDSR-S2-x4 scientific model identity and fixed K5 seeds `3407..3411`;
- prediction/score cache identities, ensemble-variance score maps, R9 local-L1 risk maps, cache
  audit, and inference-free cache replay;
- radiometric saturation receipts and an independent radiometry verifier;
- exact ROI-level conformal fitting, including all-abstain and minimum-coverage decisions;
- authoritative input receipts that bind frozen manifest membership to loaded LR/HR tensor
  identities;
- final result composition with per-sample input/cache/radiometry bindings;
- an independent final result verifier with trusted local Git ancestry revalidation;
- replay receipt composition; and
- canonical atomic bundle I/O hardened against symlinks, FIFOs, overwrite races, partial staging,
  noncanonical files, and mismatched digests.

The semantic cross-document bundle verifier is still being developed in a separate isolated
workflow. It is not part of this checkpoint and must not be described as complete until its commit
is reviewed, integrated, and tested by the main coordinator.

## Implemented command surface

The only formal Phase 2B3-B command currently implemented is metadata-only preflight:

```text
trustsr-phase2b3b preflight \
  --project-root PROJECT_ROOT \
  --evidence-dir EVIDENCE_DIR \
  --storage-root STORAGE_ROOT \
  --manifest MANIFEST
```

There is no formal `calibration` or `calibration-replay` subcommand and no
`trustsr-phase2b3b-verify` entry point yet. The similarly named functions and modules are library
boundaries exercised with synthetic inputs; they do not authorize a real run.

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
2. the formal calibration, replay, and independent verifier commands are integrated and locally
   verified;
3. the exact clean reviewed commit is ready for a disposable cloud checkout; and
4. verified calibration K5 cache entries are missing and must be generated.

Cloud-side code, logs, tensors, caches, models, paths, endpoints, and credentials must never enter
Git or become a merge source.

## Next local work

1. Review and integrate the in-progress semantic bundle verifier.
2. Add the exact host-free calibration runtime manifest and cross-bind model, dependency, input,
   result, cache-audit, replay, and revision identities.
3. Implement the formal fixed `calibration`, inference-free `calibration-replay`, and independent
   `verify` command surfaces without alpha, coverage, score, seed, window, split, sample-limit, or
   sample-ID override flags.
4. Add the acceptance record and the strict three-file Git-safe publication boundary.
5. Run the final integrated CPU test/static gates. Only then ask the user to approve or replace the
   two preregistered numerical values.
6. After approval, separately request GPU/cloud authorization if cache generation is necessary.
7. Run calibration once, replay without inference, independently verify the copied bundle, review
   the three Git-safe files, and publish either `freeze_calibration` or
   `stop_insufficient_coverage` without relaxing the preregistered gates.

Phase 2B3-C remains out of scope. It requires a separate written and approved specification before
any one-time `internal_test` access.

## Verification at handoff

Individual workstreams ran targeted tests, Ruff, and `git diff --check` before their commits. The
main coordinator owns the final integrated run after all pending commits are merged:

```bash
uv run pytest -q \
  tests/cli/test_phase2b3b.py \
  tests/evaluation/test_phase2b3b_*.py \
  tests/evaluation/test_calibration_*.py
uv run ruff check \
  src/trustsr/cli/phase2b3b.py \
  src/trustsr/evaluation \
  tests/cli/test_phase2b3b.py \
  tests/evaluation
uv run trustsr-phase2b3b --help >/dev/null
uv run trustsr-phase2b3b preflight --help >/dev/null
git diff --check
git status --short --branch
git rev-parse HEAD
```

A final full `uv run pytest -q` is intentionally deferred to the main coordinator after the
semantic bundle verifier and any final integration repairs land. Do not claim Phase 2B3-B complete
from targeted tests alone.

## Persistent stop conditions

- Accept only the exact 120 frozen `calibration` records in canonical manifest order.
- Fail closed on any evidence, revision, membership, asset, tensor, model, policy, cache, replay,
  canonical JSON, path, or digest mismatch.
- Never use development or `internal_test` observations to tune Phase 2B3-B.
- Never lower alpha or minimum coverage to rescue an unfavorable result.
- Never treat a structurally self-consistent receipt or transport-valid bundle as scientific
  authority without independent manifest, Git ancestry, and semantic verification.
- Preserve the six published Phase 2B3-A evidence files as the immutable upstream baseline.
