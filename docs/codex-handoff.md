# Codex handoff: Phase 2B3-C local implementation paused during Task 9

Date: 2026-09-04 (Asia/Shanghai)

## Repository checkpoint

- Integration branch: `main`; local Git is authoritative. Cloud code is disposable and must never
  be merged back.
- Last complete Phase 2B3-C implementation checkpoint:
  `abffa93` (`feat: add phase2b3c evaluation workflow`).
- The working tree is intentionally dirty because the user paused the session during Task 9.
  Do not discard, reset, or overwrite the six uncommitted Task 9 source/test files listed below.
- Phase 2B3-B Git-safe evidence publication commit:
  `f8f49a820d22b7dea2e003736ee465f9d7788f7d`.
- Work directly in the current attached `main`. Do not create a branch or worktree. This repository
  must have only one code-writing session; close the old session before a new terminal writes.
- Do not infer readiness from a branch name or this document. On resume, first confirm
  `git status --short`, `git branch --show-current`, and `git log -1 --oneline`; preserve the dirty
  Task 9 files and continue from their actual bytes.

## Active Phase 2B3-C state at pause

The governing documents are:

- [Phase 2B3-C one-time internal-test design](superpowers/specs/2026-09-04-phase2b3c-one-time-internal-test-design.md);
- [Phase 2B3-C implementation plan](superpowers/plans/2026-09-04-phase2b3c-one-time-internal-test.md).

The design and plan are committed at `1ae7bd5` and `1c73c2d`. The approved scientific parameters
remain frozen at `alpha=0.05` and `minimum_coverage=0.10`; the accepted Phase 2B3-B threshold is
`7.970395366024563e-06`. No scientific or execution override may be introduced.

Completed local CPU checkpoints, in order:

| Commit | Scope |
|---|---|
| `c2d2f0b` | Phase 2B3-C evaluation statistics |
| `ca9b54a` | frozen Phase 2B3-B evidence verification |
| `c542c7d` | strict `internal_test` metadata and pixel boundary |
| `1d11160` | reviewed permit and append-only one-time access ledger |
| `906ee21` | frozen manifest-order statistics correction |
| `213d686` | K5 prediction, score/risk, metrics, and cache audit computation |
| `7657c39` | result/runtime/replay and atomic six-file external bundle |
| `4b578e9` | metadata-only preflight and independent cache-computation replay |
| `739cb94` | exact 600-entry cache completeness probe |
| `abffa93` | formal evaluate/replay workflow and `trustsr-phase2b3c` CLI |

Task 8 is complete. Its focused test run passed all 26 tests:

```bash
uv run pytest -q tests/evaluation/test_phase2b3c_workflow.py \
  tests/cli/test_phase2b3c.py
```

Ruff, `compileall`, all four `trustsr-phase2b3c` help surfaces, and `git diff --check` also passed
before commit `abffa93`. The workflow now has explicit regression coverage for complete-cache CPU
operation, exact missing-cache counts before model construction, separately authorized generation,
`caches_complete` resume, storage confirmation, optional-permit metadata preflight, interruptions
on both sides of pixel access, no partial metrics, and inference-free replay.

No real `internal_test` image, prediction cache, score, risk, metric, or aggregate has been read.
No real Phase 2B3-C preflight/evaluate/replay command has run. No model was constructed and no GPU
or remote server was accessed during Phase 2B3-C implementation.

## Dirty Task 9 work that must be preserved

The user paused after the initial implementation and partial testing of Task 9. These paths are
intentionally uncommitted:

```text
M  pyproject.toml
?? src/trustsr/cli/phase2b3c_verify.py
?? src/trustsr/evaluation/phase2b3c_acceptance.py
?? src/trustsr/evaluation/phase2b3c_bundle_verify.py
?? tests/cli/test_phase2b3c_verify.py
?? tests/evaluation/test_phase2b3c_acceptance.py
?? tests/evaluation/test_phase2b3c_bundle_verify.py
```

The current draft provides:

- a metadata-only candidate-bundle receipt and verifier intended to bind all six bundle files to
  the reviewed permit, `bundle_complete` ledger, frozen evidence, exact membership, local runtime,
  model identity, and producer revision;
- an opaque acceptance capability requiring exact metadata and independent computation receipts;
- preservation of all three preregistered decisions: `confirmed`,
  `empirically_met_but_inconclusive`, and `failed`;
- a three-result-file publication transaction inside the preexisting `artifacts/phase2b3c`
  permit directory, plus an independent verifier orchestration and terminal `accepted` ledger
  transition; and
- a narrow `trustsr-phase2b3c-verify` CLI entry in `pyproject.toml` with no model, scientific,
  sample, seed, device, or worker override.

Testing status at the exact pause:

- the first minimal Task 9 run passed 12 tests after the three modules were created;
- `tests/evaluation/test_phase2b3c_acceptance.py` then passed 12 substantive tests;
- the combined substantive bundle/acceptance run reached 21 passes and one test-only `NameError`;
  that variable name was corrected immediately afterward, but the command was **not rerun**;
- Ruff initially reported four line-length findings in the new acceptance module; those lines were
  reformatted, but Ruff was **not rerun** after the substantive test additions;
- no Task 9 `compileall`, CLI help, `git diff --check`, or commit has been completed.

Treat all Task 9 code as an in-progress draft, not as verified or accepted. On resume, begin with:

```bash
uv run pytest -q \
  tests/evaluation/test_phase2b3c_bundle_verify.py \
  tests/evaluation/test_phase2b3c_acceptance.py \
  tests/cli/test_phase2b3c_verify.py
uv run ruff check \
  src/trustsr/evaluation/phase2b3c_bundle_verify.py \
  src/trustsr/evaluation/phase2b3c_acceptance.py \
  src/trustsr/cli/phase2b3c_verify.py \
  tests/evaluation/test_phase2b3c_bundle_verify.py \
  tests/evaluation/test_phase2b3c_acceptance.py \
  tests/cli/test_phase2b3c_verify.py
```

Before committing Task 9, review and close these known gaps:

- add injected orchestration tests proving the verifier reacquires the formal lock, rejects the
  producer bundle before authority/pixel reads, requires `bundle_complete`, never constructs a
  model, uses a complete exact K5 cache, publishes before advancing to `accepted`, and does not
  advance the ledger if publication fails;
- add or strengthen collision, preexisting-partial, symlink, concurrent-race, and rollback tests.
  The permit must never be removed, replaced, rewritten, or included in rollback;
- review the publication algorithm carefully. Because the reviewed permit already occupies the
  final directory, the draft links three staged files and rolls back on failure; this gives an
  all-or-clean transaction under its lock but not a single POSIX directory-rename visibility
  point. Reconcile that limitation explicitly with the design before claiming atomic publication;
- harden rollback against adversarial replacement by comparing descriptor/inode identity rather
  than deleting a same-byte destination, and avoid any unsafe path-following race;
- decide whether the private `_metadata_authority` shared by bundle verification and acceptance
  should become a reviewed public capability, and review its preliminary permit read before the
  authoritative descriptor-hardened permit verifier;
- make publication revalidation independently check every acceptance field and cross-binding,
  including target, decision, implementation revisions, permit, ledger, and all bundle digests;
- add explicit tests for forged receipt instances and byte-identical replay, plus a real synthetic
  independent computation receipt rather than relying only on private test constructors; and
- inspect type assumptions around nested JSON mappings before static checks.

After Task 9 passes and is committed, complete plan Tasks 10 and 11: Phase 2B3-C artifact/leak
policy, local-data policy and `.gitignore`, full synthetic end-to-end coverage, runbook/handoff
updates, scoped integrated tests, Ruff, `compileall`, every CLI help surface, `git diff --check`,
and a clean attached `main`. Only then reach the dedicated real-data authorization gate in Task 12.

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

## Implemented Phase 2B3-B boundary

The active design is
[Phase 2B3-B: Calibration-only conformal threshold design](superpowers/specs/2026-09-03-phase2b3b-calibration-design.md).
Its two scientific parameters were explicitly approved by the user on 2026-09-04, before any
Phase 2B3-B calibration or `internal_test` pixel was read.

Local CPU engineering, synthetic tests, and the completed formal workflow cover:

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
  --manifest MANIFEST \
  --confirm-persistent-storage
```

Formal calibration and inference-free replay are implemented with the same fixed path arguments,
explicit `--confirm-persistent-storage`, and no scientific override flags:

```text
trustsr-phase2b3b calibration \
  --project-root PROJECT_ROOT \
  --evidence-dir EVIDENCE_DIR \
  --storage-root STORAGE_ROOT \
  --manifest MANIFEST \
  --confirm-persistent-storage \
  [--ldsr-model-dir LDSR_MODEL_DIR]

trustsr-phase2b3b calibration-replay \
  --project-root PROJECT_ROOT \
  --evidence-dir EVIDENCE_DIR \
  --storage-root STORAGE_ROOT \
  --manifest MANIFEST \
  --confirm-persistent-storage
```

`calibration` first probes the complete fixed K5 cache identity without constructing LDSR. A fully
verified 600-entry cache completes on CPU with no model path. If an entry is missing it stops before
model loading; only a separately authorized rerun with `--ldsr-model-dir` may construct LDSR or
generate predictions. Before atomic bundle publication it immediately performs an inference-free
reconstruction. The explicit replay surface does not accept a model path, recomputes
ensemble-variance score and R9 risk from verified caches, and requires byte-identical
result/audit/replay/bundle evidence.

The verifier is now the acceptance-authorizing independent computation and publication surface:

```text
trustsr-phase2b3b-verify \
  --bundle BUNDLE \
  --project-root PROJECT_ROOT \
  --evidence-dir EVIDENCE_DIR \
  --storage-root STORAGE_ROOT \
  --manifest MANIFEST \
  --confirm-persistent-storage
```

It retains the metadata-only candidate verifier as its first gate, then loads only authoritative
calibration inputs, independently replays cache-derived computations without model inference, and
atomically publishes the exact result, cache-audit, and acceptance JSON files under
`artifacts/phase2b3b`. Only this complete path may emit `acceptance_authorized=true`.

The exact operator sequence and stop conditions are in
[the Phase 2B3-B calibration runbook](phase2b3b-calibration-runbook.md).

## Completed formal calibration and publication

The authorized formal calibration, explicit inference-free replay, copied-bundle independent
verification, acceptance, and three-file publication completed on 2026-09-04. Only the 120 frozen
`calibration` records were loaded. No `internal_test` pixel, cache, prediction, score, risk, or
metric was accessed.

Formal identities and results:

- formal result producer revision: `2d17c141174c1062f8a056b491326487202a6deb`;
- independent verifier revision: `aad2b7e1ea4ec6599db40015d1745c40312dd0af`;
- result SHA-256: `5fe01daaaa27443d5289cc366eacbe1fb628492c25bab16fadd164d3a8eb3174`;
- cache-audit SHA-256: `40aff8aaccb27719d958d42026d68d8efc0c7f79479e3bd99777053ee1d6399f`;
- runtime-manifest SHA-256: `33ce35ba1065afa27f633a703c2e3c7bb4ce6438a1a4688e937b9ba76cad1a04`;
- replay SHA-256: `203c3da04882c5dc538af799c38f24110341e792768edde51eaca413df028bf4`;
- bundle-manifest SHA-256: `2b08366c1109bfa119eb205360a9127f9b3aeb8ce44ba5940dec76ee035ae31d`;
- acceptance SHA-256: `ce7f1a91e9954235e289dfc6eececb67261ff57110caaf15fe6640a0fb3b69ab`;
- three-file publication SHA-256: `f59d8589425e41d7a0bf3f9913c4db8f7d413a82c97305c882388f522389960f`.

The observed result is `freeze_calibration`: threshold
`7.970395366024563e-06`, aggregate calibration coverage `0.5832304954528809`, and risk bound
`0.04999883134510526`. The explicit replay was byte-identical. The independent verifier emitted
`acceptance_authorized=true` and `cache_computation_verified=true`; as designed, it did not claim
to rerun LDSR inference.

The only published Git artifacts are:

| File | SHA-256 |
|---|---|
| `sen2naipv2-calibration-conformal-v1.json` | `5fe01daaaa27443d5289cc366eacbe1fb628492c25bab16fadd164d3a8eb3174` |
| `sen2naipv2-calibration-conformal-cache-audit-v1.json` | `40aff8aaccb27719d958d42026d68d8efc0c7f79479e3bd99777053ee1d6399f` |
| `sen2naipv2-calibration-conformal-acceptance-v1.json` | `ce7f1a91e9954235e289dfc6eececb67261ff57110caaf15fe6640a0fb3b69ab` |

## Approved scientific parameters and interpretation

The user explicitly approved the following preregistered primary operating point on 2026-09-04:

- `alpha = 0.05`;
- minimum calibration pixel coverage `0.10` for permission to design Phase 2B3-C.

The approval followed a read-only scientific review. `alpha=0.05` is an application-defined target
for the expected ROI-level maximum local mean absolute RGBN reflectance risk under the conformal
exchangeability assumptions. It is not a significance level, a 95% pixel guarantee, a data-driven
optimum, or a Sentinel-2 radiometric-compliance claim. With 120 calibration ROIs and risk upper
bound 1, the implemented finite-sample rule requires
`(sum(worst_i) + 1) / 121 <= 0.05`, so the empirical mean of the 120 ROI worst risks must be at most
approximately `0.04208` for a finite threshold.

The `0.10` value is a preregistered non-degeneracy and minimum-utility gate over aggregate trusted
calibration pixels. It is not a conformal coverage guarantee, a per-ROI guarantee, a geographic
representativeness guarantee, or a universal remote-sensing standard. A finite threshold below this
coverage must produce `stop_insufficient_coverage`; neither value may be relaxed after observing the
result. The synthetic Phase 2A default `0.27` remains prohibited for the formal run.

Method and context sources:

- Angelopoulos et al., *Conformal Risk Control*, ICLR 2024:
  https://research.google/pubs/conformal-risk-control/
- Adame et al., *Image Super-Resolution with Guarantees via Conformalized Generative Models*,
  NeurIPS 2025: https://arxiv.org/abs/2502.09664
- ESA/Copernicus Sentinel-2 radiometric requirements, used only as order-of-magnitude context:
  https://sentiwiki.copernicus.eu/web/s2-mission
- Aybar et al., *SEN2NAIP*, documenting cross-sensor harmonization and reference limitations:
  https://doi.org/10.1038/s41597-024-04214-y

The approval and completed workflow do not authorize parameter changes or later-split access.
Continue to enforce:

- do not expose `alpha` or minimum coverage as runtime override flags;
- do not inspect any `internal_test` pixels, caches, predictions, scores, risks, or metrics.

## GPU and cloud status

No GPU or cloud server is needed for the remaining local Phase 2B3-C Tasks 9-11. The server may
remain off. Do not connect merely to inspect caches: the exact K5 cache probe is itself protected
`internal_test` access and may run only after the dedicated Task 12 authorization and reviewed
permit exist.

If an authorized real evaluation later reports any of the exact 600 prediction entries missing,
stop before model construction and tell the user GPU is required. Remote execution must use the
server's existing base interpreter (`/opt/conda/bin/python`), never `uv`, `.venv`, environment
creation, or dependency mutation. If all 600 entries verify, complete evaluate/replay/verification
on CPU and do not start LDSR or request GPU.

Cloud-side code, logs, tensors, caches, models, paths, endpoints, and credentials must never enter
Git or become a merge source.

## Next local work

Resume the dirty Task 9 draft exactly as described above. Do not start Task 10 until independent
acceptance/publication and verifier CLI tests pass and Task 9 is committed. Do not start Task 12 or
run real preflight/evaluate/replay/verify commands until Tasks 9-11 and all local gates are complete.

The user's standing instruction to proceed with recommended local steps is not a substitute for the
design's dedicated, one-time Phase 2B3-C real-data authorization. At Task 12, stop and request that
specific authorization before creating the reviewed permit or accessing any real `internal_test`
pixel or cache. Parameter changes remain forbidden after observing any holdout information.

## Verification at handoff

The historical completed Phase 2B3-B integrated run covered:

```bash
uv run pytest -q \
  tests/cli/test_phase2b3b.py \
  tests/cli/test_phase2b3b_verify.py \
  tests/evaluation/test_phase2b3b_*.py \
  tests/evaluation/test_calibration_*.py \
  tests/data/test_calibration_subset.py \
  tests/data/test_calibration_pairs.py \
  tests/data/test_local_data_policy.py \
  tests/models/test_ldsr_s2.py \
  tests/calibration/test_conformal.py
uv run ruff check \
  src/trustsr/cli/phase2b3b.py \
  src/trustsr/cli/phase2b3b_verify.py \
  src/trustsr/evaluation \
  tests/cli/test_phase2b3b.py \
  tests/cli/test_phase2b3b_verify.py \
  tests/evaluation \
  tests/data/test_local_data_policy.py \
  tests/models/test_ldsr_s2.py
uv run python -m compileall -q src
uv run trustsr-phase2b3b --help >/dev/null
uv run trustsr-phase2b3b preflight --help >/dev/null
uv run trustsr-phase2b3b calibration --help >/dev/null
uv run trustsr-phase2b3b calibration-replay --help >/dev/null
uv run trustsr-phase2b3b-verify --help >/dev/null
git diff --check
git status --short --branch
git rev-parse HEAD
```

That scoped pytest run passed at 100%. Ruff, `compileall`, all five CLI help checks, tracked-data
policy, canonical publication/digest checks, and `git diff --check` also passed. The formal command,
explicit replay, independent copied-bundle verifier, local publication review, and post-commit
three-file digest checks all passed before this handoff update.

For Phase 2B3-C, only Tasks 1-8 have completed their local checkpoints. Task 8's latest verified
focused gate was 26 passing tests plus Ruff, compile, CLI help, and diff checks. Task 9 is dirty and
partially tested as recorded above. Do not describe the Phase 2B3-C verifier, acceptance,
publication, policy, integrated suite, or real evaluation as complete.

## Persistent stop conditions

- Preserve the completed Phase 2B3-B evidence and accept only the exact 120 frozen
  `internal_test` records in their canonical post-manifest order for Phase 2B3-C.
- Fail closed on any evidence, revision, membership, asset, tensor, model, policy, cache, replay,
  runtime, permit, ledger, canonical JSON, path, or digest mismatch.
- Before dedicated authorization, do not open any `internal_test` image or cache, run a real C
  command, construct LDSR, inspect CUDA for this phase, or write/advance the C access ledger.
- After authorization, write `pixels_opened` before the first pixel or test-cache access and never
  reset, delete, or replace ledger history. A crash after that event counts as consumed access.
- Never use development, calibration, or `internal_test` observations to tune Phase 2B3-C.
- Never lower alpha or minimum coverage to rescue an unfavorable result.
- Never treat a structurally self-consistent receipt or transport-valid bundle as scientific
  authority without independent manifest, Git ancestry, and semantic verification.
- Preserve the six published Phase 2B3-A evidence files as the immutable upstream baseline.
- Publish whichever preregistered C decision is observed; only `confirmed` supports a confirmation
  claim, while all three decisions permanently complete the one-time phase after acceptance.
