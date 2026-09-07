# Codex handoff: first-paper research plan; Phase 2B3-C remains terminal

Date: 2026-09-07 (Asia/Shanghai)

## Current objective: approved first-paper research direction

The user accepted the recommended research direction on 2026-09-07 and requested a new plan
and Git publication. The first paper now focuses on uncertainty-score cost, error-scale
sensitivity, and frozen calibration transfer for Sentinel-2 RGBN x4 super-resolution.

Read these documents before starting new work:

- [Current first-paper roadmap](research-roadmap.md);
- [Task-by-task research execution plan](superpowers/plans/2026-09-07-first-paper-research.md);
- [Literature review and scope rationale](reports/2026-09-07-research-replan.md).

The user subsequently authorized execution, with a stop to request GPU resources when needed.
Tasks 1–2 now have evidence, metric, manuscript, and rendering-design documents; no new
scientific computation has run. Neighborhood and Spain designs are drafts, not a final freeze.
The next local task is the read-only evidence renderer followed by synthetic neighborhood tests.
See [readiness checkpoint](reports/first-paper-development-readiness.md) and
[Spain metadata audit](reports/first-paper-spain-metadata-audit.md). The latter records unresolved
v2/v3 membership correspondence and incidental exposure to official quality columns in metadata.
Do not describe the external study as completely quality-blind or resolve these gaps using pixels.
The target is a complete draft in approximately 4–6 weeks, subject to data and resource gates.
New networks, grouped calibration, SEN2NEON, downstream segmentation, and fallback products
are outside the first-paper baseline scope. Formal external-test access and paid GPU execution
still require their dedicated execution scope and authorization after the new protocol is frozen.

The historical Phase 2B3-C workflow below is complete and immutable. New work must not rerun
its commands, reopen its protected data, or alter its inconclusive interpretation. Historical
plans are not instructions to repeat completed experiments.

## Repository checkpoint

- Integration branch: `main`; local Git is authoritative. Cloud code is disposable and must never
  be merged back.
- Frozen Phase 2B3-C computation implementation checkpoint:
  `b45ca5e` (`test: harden phase2b3c publication boundary`).
- Independent acceptance/publication checkpoint:
  `d7d0c35` (`feat: verify and publish phase2b3c acceptance`).
- Task 11 local-readiness documentation checkpoint:
  `1f9d07a654d10cd155e32a329150383382ce93b6`.
- Reviewed one-time access-permit commit:
  `a5367ef630d634cd5482f37bf289d35da6ac9119`.
- Three-file Phase 2B3-C result publication commit:
  `ab7fb54e4b293eee7850cb62afe4c498ad18c9d9`.
- Phase 2B3-B Git-safe evidence publication commit:
  `f8f49a820d22b7dea2e003736ee465f9d7788f7d`.
- Work directly in the current attached `main`. Do not create a branch/worktree or allow concurrent
  writers. Confirm `git status --short`, `git branch --show-current`, and `git log -1 --oneline`
  before any later operation.
- Tasks 1–12 and the authorized one-time workflow are complete. The terminal decision is
  `empirically_met_but_inconclusive`; the immutable ledger is `accepted`. The protected evaluation
  is consumed and must never be repeated, reset, rescued, or retuned.

## Historical Phase 2B3-C completion summary

### Task objective

Implement the preregistered Phase 2B3-C one-time `internal_test` evaluation on local `main`, prove
the full workflow with synthetic/local CPU gates, stop for dedicated real-data authorization,
probe the exact 600-entry K5 cache, use GPU only after a separate authorization if entries are
missing, independently verify and publish the observed decision, and leave GitHub `main` aligned
with the reviewed local history.

### Completed modifications

- Tasks 9–11 completed the independent copied-bundle verifier, acceptance/publication boundary,
  exact four-file publication policy, leakage protections, synthetic end-to-end coverage, runbook,
  local gates, and readiness handoff.
- Task 12 produced metadata-only readiness outside Git, committed the exact reviewed permit,
  consumed the one-time ledger only after explicit authorization, and stopped before LDSR when the
  exact cache probe reported `0/600` present and `600/600` missing.
- After separate GPU authorization, the fixed LDSR implementation generated only the missing
  entries. Formal evaluate, immediate reconstruction, explicit inference-free replay, copied-bundle
  independent verification, three-file publication, and terminal `accepted` advancement passed.
- The permit, result publication, and terminal documentation were committed separately as
  `a5367ef`, `ab7fb54`, and `e4fd8c9`. Before the earlier handoff-only edit, local `main` and GitHub
  `origin/main` both resolved to `e4fd8c9b7ae8293bd873625a2e20684832b9ce5d` with no diff.

### Key decisions

- The frozen threshold, alpha, minimum coverage, score, K5 seeds, risk, membership, ordering, and
  confidence method were never changed after protected access.
- The observed decision is permanently `empirically_met_but_inconclusive`, not `confirmed`:
  coverage and empirical mean loss met their fixed targets, but the grid-Kelly UCB did not.
- The one-time authorization is consumed. No rerun, second permit, replacement ledger, retuning,
  or alternate interpretation is allowed.
- No implementation/schema change was made after observing the holdout. Review findings about the
  acceptance v1 terminal-event binding and index-only publication scanner are documented rather
  than repaired through a prohibited second run.

### Principal modified files

- Runtime and CLIs: `src/trustsr/cli/phase2b3c.py`,
  `src/trustsr/cli/phase2b3c_verify.py`, `src/trustsr/data/internal_test_subset.py`,
  `src/trustsr/data/internal_test_pairs.py`, and the Phase 2B3-C/internal-test modules under
  `src/trustsr/evaluation/`.
- Tests: the corresponding Phase 2B3-C CLI, data, model, risk, policy, workflow, verifier,
  acceptance, result, replay, and synthetic end-to-end tests under `tests/`.
- Documentation: `docs/phase2b3c-one-time-evaluation-runbook.md` and this handoff.
- Git-safe artifacts: the permit plus the three canonical JSON documents under
  `artifacts/phase2b3c/`; no runtime, replay, ledger, bundle manifest, cache, tensor, model, path,
  endpoint, credential, timestamp, or per-sample numerical outcome entered Git.

### Verification results

- The planned Phase 2B3-C local CPU pytest gate reached `100%`; Ruff, `compileall`, all five CLI
  help checks, and `git diff --check` passed before real access.
- Real evaluate completed for 120 frozen records and 600 exact predictions. Immediate and explicit
  replay were byte-identical. Independent copied-bundle verification reported
  `acceptance_authorized=true`, `cache_computation_verified=true`, and ledger state `accepted`.
- After publication, the scoped tracked-data, publication-policy, result, and acceptance tests
  reached `100%`; the index policy scanner returned no violations; all four artifacts were
  canonical and their cross-file digests/counts/decision/checks matched.
- A separate read-only review found no critical artifact issue and confirmed the decision
  semantics and absence of forbidden publication data.

### Unfinished items

- No functional Phase 2B3-C implementation, evaluation, replay, verification, publication, or
  GitHub synchronization work remains.
- The earlier handoff-only edit was intentionally staged and left unpublished. The subsequent
  user request now authorizes publishing this reviewed handoff together with the new research plan.
- Outstanding first-paper research work is tracked in the new execution plan, not in the
  completed Phase 2B3-C task list.

### Known risks and immutable cautions

- `empirically_met_but_inconclusive` cannot support a confirmation claim.
- The acceptance v1 artifact binds the `bundle_complete` event and records terminal state
  `accepted`, but does not contain the subsequently appended accepted-event digest. The exact
  deterministic offline reconstruction and limitation are recorded below; do not rerun protected
  data to alter the schema.
- The publication scanner reads Git-index entries, so candidate artifacts must be staged through
  the exact allowlist before the scan. This procedure was followed for the published files.
- The external ledger/cache/bundle/model state is not Git authority and must not be copied back,
  reset, deleted, inspected for another evaluation, or used as a merge source.

## Phase 2B3-C local implementation state

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
| `d7d0c35` | independent verifier, acceptance, and hardened publication |
| `b45ca5e` | artifact/leak policy and full synthetic end-to-end readiness |

Task 9 resumed from the intentionally dirty draft recorded by the previous handoff. Its initial
32-test command and Ruff command passed before further hardening. The completed checkpoint adds:

- a metadata-only candidate-bundle receipt and verifier that bind all six bundle files to
  the reviewed permit, `bundle_complete` ledger, frozen evidence, exact membership, local runtime,
  model identity, and producer revision;
- provenance-marked opaque metadata/computation/acceptance capabilities that reject forged
  instances and independently require byte-identical replay;
- preservation of all three preregistered decisions: `confirmed`,
  `empirically_met_but_inconclusive`, and `failed`;
- descriptor-relative permit/publication reads, no-replace staged hard links, inode-bound rollback,
  collision/partial/symlink/race rejection, and preservation of the preexisting permit;
- an explicit acknowledgement that the existing permit directory prevents a single POSIX
  directory-rename visibility point: the implementation provides all-or-clean failure semantics
  under formal/permit locks for cooperating readers;
- verifier orchestration that reacquires the formal lock, rejects the producer bundle before
  authority/pixel access, requires `bundle_complete`, requires the exact complete K5 cache, never
  constructs a model, publishes before advancing to `accepted`, and leaves the ledger unchanged on
  publication failure; and
- a narrow `trustsr-phase2b3c-verify` CLI entry in `pyproject.toml` with no model, scientific,
  sample, seed, device, or worker override.

Task 10 adds an exact four-name Git allowlist (one future permit plus three future result files),
canonical schema/size checks, path/host/endpoint/credential/token/time/GPU-identity leak scanning,
per-sample numerical-metric rejection, `.gitignore` enforcement, and tracked-data enforcement. Its
three-decision synthetic end-to-end test uses a generated balanced 360-row manifest, 120 tiny CPU
pairs, fake complete K5 caches, formal evaluate, inference-free replay, copied-bundle independent
verification, terminal ledger behavior, canonical publication, and the publication policy scan.

The exact operations and stop conditions are documented in
[the Phase 2B3-C runbook](phase2b3c-one-time-evaluation-runbook.md).

## Completed Phase 2B3-C one-time evaluation and publication

The user granted the dedicated one-time real `internal_test` authorization on 2026-09-07 before
the permit, ledger, pixel, or cache boundary was opened. The metadata-only readiness document was
kept outside Git and bound:

- readiness SHA-256: `78e46a77df4d5229db30511b20dfad17aaa522839384c9177229b828cb9538f6`;
- evaluation ID: `516b0dc182329201d0e472fa15e5d95ed3863ac2e2b83762f40cce4c2c55b80c`;
- implementation revision: `1f9d07a654d10cd155e32a329150383382ce93b6`;
- computation-tree SHA-256: `db52687da91ac56b1852bced50accbdc58d611a10bc82f58b08a2222d31442c7`;
- ordered-membership SHA-256:
  `dc94582cde20facc6b0f4d87d5899d49256497d59627c219a2a3c14b7c6ae354`; and
- environment SHA-256: `80240178079221f3a99c4b56c26b827dd8cb5e96ba11f783681b3934bcb4bb14`.

The canonical permit SHA-256 is
`efec86c1bbced2e2c0d9c8101d51a602e724853e3a99cea518aee0fe5265b464`. The first exact K5 probe
reported `0/600` verified entries and `600/600` missing, then stopped before LDSR construction as
required. After the user separately authorized GPU startup and use, the same permit and ledger
resumed with the existing base `/opt/conda/bin/python`; no environment or dependency was created or
modified. LDSR generated only the missing exact entries. Formal evaluate, immediate reconstruction,
explicit inference-free replay, copied-bundle independent verification, publication, and terminal
ledger advancement all completed successfully.

The preregistered result is `empirically_met_but_inconclusive`:

- aggregate trusted-pixel coverage: `0.5741024335225423`, above the fixed `0.10` minimum;
- empirical mean loss: `0.04198510404780997`, below the fixed `0.05` target; and
- grid-Kelly risk UCB: `0.0779855394819395`, above the fixed `0.05` target.

The sole reason is `finite_sample_upper_bound_exceeds_target`. This is not a confirmation claim and
must not be rescued by changing the threshold, alpha, coverage gate, score, seeds, membership,
weighting, risk, or confidence method.

The only Git-published Phase 2B3-C result files are:

| File | SHA-256 |
|---|---|
| `sen2naipv2-internal-test-evaluation-v1.json` | `9f267e459908a28e6fd254de9cc1d7d2352bf23b672366a9eda9839d53a0e115` |
| `sen2naipv2-internal-test-evaluation-cache-audit-v1.json` | `4bf048b222d4993adb0fd70b0273f60a994ecb7b3539e58ed3b2d57dcf4f43ff` |
| `sen2naipv2-internal-test-evaluation-acceptance-v1.json` | `6fda78e13b13f78201487043fe9f679b432b692216778d172e7f36bf2d9ded21` |

The independent verifier reported `acceptance_authorized=true`,
`cache_computation_verified=true`, `prediction_inference_verified=false`, and ledger state
`accepted`. The published acceptance binds the `bundle_complete` event SHA-256
`45c08caa9d90d69fbaf8a72b66b960ff2930de05ec26b04ef2db4590dcac2c38` and records the terminal
state literal, but its v1 schema does not contain the subsequently appended accepted-event digest.
From the deterministic ledger schema, exact permit binding, canonical storage identity, sequence
`4`, and that predecessor digest, the accepted-event SHA-256 was independently reconstructed
offline as `3217912ad1387e186f0000d5eabe103d84f574419df034fe8ecd1b5f2630a278`.
Do not revise the published schema or rerun protected data to change this limitation.

The publication policy scanner examines Git-index entries. The coordinator therefore staged only
the exact three result names before running the scanner; it returned no violations. A separate
read-only review found no forbidden path, host, credential, timestamp, GPU identity, or per-sample
numerical outcome in the four canonical Phase 2B3-C artifacts.

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

GPU use was separately authorized only after the exact probe reported all 600 entries missing.
The GPU generated those entries once; formal replay and independent verification were
inference-free. All remote commands used the existing base interpreter and made no environment or
dependency changes. Remote compute and verification have ended, so the GPU/server is no longer
required and may remain off.

Do not reconnect to enumerate caches, recover unpublished per-sample values, rerun inference, or
repeat the consumed evaluation. The external cache, bundle, readiness, ledger, model, runtime,
replay, path, endpoint, and credential state remains outside Git and is not a merge source.

Cloud-side code, logs, tensors, caches, models, paths, endpoints, and credentials must never enter
Git or become a merge source.

## Phase 2B3-C terminal state and boundary for new work

Phase 2B3-C is terminal. Do not create another permit or ledger, rerun preflight/evaluate/replay/
verify, reopen protected data, or attempt a second confirmation. Future work may consume only the
three published aggregate result documents under their stated inconclusive interpretation. Any
downstream design requires a new plan and must not reinterpret this run as `confirmed`.

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

The Phase 2B3-C final local-only pytest gate ran once on 2026-09-04 with the exact planned command:

```bash
uv run pytest -q \
  tests/cli/test_phase2b3c.py tests/cli/test_phase2b3c_verify.py \
  tests/evaluation/test_phase2b3c_*.py \
  tests/evaluation/test_internal_test_*.py \
  tests/data/test_internal_test_subset.py tests/data/test_internal_test_pairs.py \
  tests/data/test_local_data_policy.py tests/models/test_ldsr_s2.py tests/risk/test_local.py
```

It reached `100%` with exit code zero. The exact planned Ruff scope passed with `All checks passed!`;
`uv run python -m compileall -q src`, all five Phase 2B3-C CLI help commands, and
`git diff --check` also exited zero. Task 9's expanded verifier/computation/acceptance scope passed
before commit `d7d0c35`, and Task 10's policy plus three-decision synthetic end-to-end scope passed
before commit `b45ca5e`.

Post-publication, the exact three result files were staged before policy inspection. The dedicated
publication scan returned no violations; the tracked-data policy and scoped policy/result/
acceptance tests reached `100%`; all four Phase 2B3-C artifacts were canonical JSON; cross-file
result/cache/acceptance digests, counts, decision, checks, and terminal-state fields matched. An
independent read-only review found no critical artifact issue and confirmed the decision semantics.
Its implementation-level findings—the acceptance v1 accepted-event limitation and the index-only
scanner behavior—are recorded above without changing code or rerunning consumed data.

## Persistent stop conditions

- Preserve the completed Phase 2B3-B evidence and accept only the exact 120 frozen
  `internal_test` records in their canonical post-manifest order for Phase 2B3-C.
- Fail closed on any evidence, revision, membership, asset, tensor, model, policy, cache, replay,
  runtime, permit, ledger, canonical JSON, path, or digest mismatch.
- The dedicated authorization has been consumed and the ledger is terminal `accepted`; never open
  another `internal_test` image/cache, run another real C command, or create a replacement permit.
- Never reset, delete, replace, copy as authority, or manually advance the external ledger history.
- Never use development, calibration, or `internal_test` observations to tune Phase 2B3-C.
- Never lower alpha or minimum coverage to rescue an unfavorable result.
- Never treat a structurally self-consistent receipt or transport-valid bundle as scientific
  authority without independent manifest, Git ancestry, and semantic verification.
- Preserve the six published Phase 2B3-A evidence files as the immutable upstream baseline.
- Publish whichever preregistered C decision is observed; only `confirmed` supports a confirmation
  claim, while all three decisions permanently complete the one-time phase after acceptance.
