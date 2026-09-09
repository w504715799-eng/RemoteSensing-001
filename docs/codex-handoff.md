# Codex handoff: first-paper research plan; Phase 2B3-C remains terminal

Date: 2026-09-09 (Asia/Shanghai)

## ACTIVE RESUME CHECKPOINT — integrated CPU method and calibration implemented

2026-09-09. User requested direct method/calibration development. Implemented new
`research/trustmask/calibration.py`, `pipeline.py`, `integrated_demo.py`; no frozen
source edits, images, models, old prediction caches, server or GPU access.
Report: `docs/reports/integrated-calibration.md`; plan:
`docs/superpowers/plans/2026-09-09-integrated-calibration.md`.

Workflow: immutable precomputed feature/R9 inputs; five-role sample/group separation;
64 fixed equal-ROI samples for scale medians; 36 candidate configurations across
8 named methods. All development choices finish BEFORE any formal calibration
loss access. All thresholds freeze BEFORE test evaluation. Formal calibration
uses bounded monotone group CRC; deployment excludes reference inputs. No feasible
development configuration or no feasible calibration threshold yields reject-all.
Test output includes 8-method risk bounds, full-vs-w3 paired coverage lower bound,
per-group observations and workload proxies. CRC is marginal expectation under
exchangeability; test bounds separately require independent groups. Not unconditional
certification. With preview n=3045 and M=8, test margins are 0.0307763 risk / 0.0492231 gain.

Synthetic demo runs end-to-end and labels results synthetic_not_research_evidence;
small test-group counts correctly do not certify a gain. Tests exercise numerical
curves, weights, no-candidate fallback, stage access order and test/calibration-label
isolation. Independent method and code reviews completed; final scoped gates run
before commit. Full demo output stays under /tmp, not published as scientific data.

NEXT: prepare real-data execution adapter/protocol: authenticated exact assignments,
fixed common support, center/seed/model/R9 definitions, bounded batch extraction
and inference, streamed feature/curve storage and one-time terminal-test ledger.
Current core is an in-memory reference implementation, unsuitable for loading all
520x520 maps at once without an explicit memory plan. Freeze protocol and commands
before notifying user to start GPU. Public provenance is accepted; do not revive
source-map/contact/login requirements. Old C remains terminal; main single writer.

---

## ACTIVE RESUME CHECKPOINT — public provenance accepted; proceed to integrated method

2026-09-09 USER OVERRIDE: actual per-member source mapping is NOT required. A
public paper or open release is sufficient. Official SEN2NAIPv2 release describes
8,000 real crosssensor pairs and is accepted. Original SEN2NAIP paper describes v1;
do not attribute v2's exact member manifest to that earlier paper. See current spec
addendum and `docs/reports/public-source-partition.md` for sources and interpretation.

Cancel the previous contact/login dependency and pending publishing question.
The unsent provenance draft is retired. Do not ask again, contact upstream, repeat
STAC probes or continue timestamp/source-ID investigations. Missing per-scene
provenance is a disclosed limitation, not a blocker. Statistical guarantees remain
conditional on explicitly stated geographic-group assumptions.

Completed offline five-role preview: 6,158 groups / 7,477 members after excluding
523 members in historically touched 5 km + shared-NAIP components. Groups by role:
scale_fit 256; development_calibration 512; development_validation 845;
calibration 1,500; test 3,045. No group/member crosses roles. Full assignment is
ignored at `artifacts/progressive-partition-preview/assignments.json`; published
summary `research/evidence/crosssensor-public-source-partition-preview-v1.json`
binds assignment/member/history digests. This is a preview, not a frozen protocol.

NEXT: implement CPU synthetic integrated scale fitting, development selection,
group-risk calibration and paired evaluation in research/trustmask. Then freeze
exact protocol/method family and prepare cloud extraction/inference commands.
Notify user to start GPU only when those are ready. Public provenance acceptance
now permits that path; no further source-map approval needed. Do not reopen old
A/B/C pixels or caches; old C remains terminal. Main remains single writer.

---

## ACTIVE RESUME CHECKPOINT — writer pinned; upstream contact pending

2026-09-09. This checkpoint supersedes older next-step instructions. The previously
inspected writer snapshot is now pinned to commit ee570c3f8ef75178e5cd1b064904c9205355d9bc
and blob 786b85cfae429077c6afd38564a68b2c1dac9919. Both the Git blob SHA-1 and
SHA-256 match the original 13,954-byte cached source. This identifies the inspected
code version, not the actual dataset build version or timezone.
Evidence: `research/evidence/crosssensor-writer-source-pin-v1.json`.

Read-only follow-up of the official discussion and indexed GitHub searches did
not provide a revision-applicable generation mapping; this is not proof none exists.
No prior STAC probe was repeated and no image/cache/GPU access occurred.

Concrete external request remains `docs/drafts/sen2naipv2-provenance-request.md`.
An async question asks whether the user authorizes publishing it to the official
Hugging Face dataset discussion. No response has arrived at this checkpoint;
no message has been sent. A boolean-only credential check found no HF token.
Do not infer approval from elapsed time or repeat a credential value in output.
If publishing is explicitly authorized, the user must first establish local HF
login or publish the prepared draft themselves. Do not switch to another external
channel without authorization. Meanwhile preserve all completed provenance evidence;
source-aware split freeze and GPU experiments still require the actual source map,
or an explicitly designed source-traceable replacement data route.

---

## ACTIVE RESUME CHECKPOINT — timestamp mechanism and bounded candidates investigated

2026-09-08. This checkpoint supersedes older next-step instructions. No GPU/server
needed. Full report: `docs/reports/timestamp-source-probe.md`; reproducible evidence:
`research/evidence/crosssensor-timestamp-source-probe-v1.json`. Fixed historical
projection and fixed three nested probes are SHA-bound by the new offline runner.

All 8,000 LR and HR times are midnight in the Europe/Madrid hypothesis zone;
all HR local dates equal the NAIP filename suffix. This does not identify the
actual upstream timezone. Public Tortilla STAC writer code directly calls
`.timestamp()` on datetime objects, consistent with local-time serialization of
naive dates. Snapshot SHA is in report; actual dataset build version remains unknown.
Do not rewrite old timestamps or treat them as verified sensing instants.

Three predeclared 48-hour, small-bbox Earth Search L2A queries returned two
candidates each, all on the inferred following date: first two members have
adjacent-tile alternatives; third has same tile 10SEH, baselines 02.14/05.00.
Successful response bytes total 14,485; attempts 1/1/2. No images/assets fetched.
All responses report matched=returned=2 but include a next link, so evidence is
explicitly first-page-only / enumeration not certified. No pagination followed.
Ignored cache `artifacts/progressive-timestamp-source` supports offline replay;
do not repeat these completed bounded probes or silently expand their budget.

Next dependency: actual revision-applicable generation code/source manifest,
including date serialization, tile/version choice and all composite contributors.
The local unsent provenance-request draft now includes concrete examples; no
external message authorized or sent. Broader catalog proximity queries cannot
prove which products were used. If source records are unavailable, explicitly
design a new source-traceable data route before independent grouping/split freeze.
GPU experiments remain premature. Continue on main, single writer; old C terminal.

---

## ACTIVE RESUME CHECKPOINT — historical metadata recovered and audited

2026-09-08. This checkpoint supersedes older next-step instructions. The original
Phase 2B1-A full manifest was found on the existing cloud persistent disk. Its exact
pinned SHA passed before projection; no legacy recovery CLI, assets, predictions,
models or GPU computation were used. Only allowlisted metadata was returned locally.
The server is no longer needed for this stage; user was notified it can be shut down.

Ignored local export: `artifacts/progressive-historical-manifest/metadata.jsonl`
and `receipt.json`. Projection SHA:
`8bf11a4851a97a1af26107f8f91c65946ca931b0156fa85cf8c436d7589bd08a`.
Published summary: `research/evidence/crosssensor-historical-manifest-audit-v1.json`.
Reproducible offline verifier: `research/trustmask/audit_recovered_manifest.py`.
All 8,000 identities, source indices, source identity, centroids and geometry match
the committed progressive shards. All signed HR-minus-LR UTC date deltas pass:
-1: 2,145; 0: 3,972; +1: 1,883. Top-level time equals LR time for every member.

New limitation: LR timestamps occur only at 22:00Z (7,174) or 23:00Z (826); HR
similarly only at these hours. These are verified stored metadata values, not
verified satellite sensing instants. Do not match catalogs by exact instant or
infer S2 product identity from timestamps. The cause of this pattern is unresolved.
Next step is a bounded metadata-only investigation of timestamp semantics and
candidate source products, using the existing predeclared new-member probes and
an explicit date-window convention. Catalog matches would remain candidates until
construction/compositing provenance is established. No split freeze or GPU run yet.
Prior external request draft remains unsent; no contact authorization. Old C terminal.

---

## ACTIVE RESUME CHECKPOINT — recover existing historical full manifest first

2026-09-08. This checkpoint supersedes older next-step instructions. Git commits
`a27e169` and `06e577e` record a full 8,000-member Phase 2B1-A manifest with actual
LR/HR acquisition times, intentionally retained on cloud storage. Its pinned SHA
is `7487b0af2ebef86910e918d5d6b2fb927a6f5e46bac7c2e30be7ffb2ce994482`.
Prioritizing maintainer contact before checking this existing evidence was premature.
The manifest does not contain S2 product IDs; source independence remains unresolved.

Prepared stdlib-only export: `research/trustmask/recover_manifest.py`.
Exact path, command and follow-up checks: `docs/reports/historical-manifest-recovery.md`.
Local synthetic tests cover digest-before-parse, count rejection and exclusion of
observations/assets. Actual recovery and semantic validation have not run.
Next dependency is the server holding the persistent disk being online and its
connection/storage location being available. User requested notification when needed.
This operation needs CPU/disk access only, no GPU computation. Return only the
allowlisted metadata export; never reopen old pixels or prediction caches. Continue
on main, single writer. Phase 2B3-C remains terminal; no new experiment authorized
by this recovery. The external request draft remains unsent.

---

## ACTIVE RESUME CHECKPOINT — provenance probes complete, source manifest missing

2026-09-08. This section supersedes all older next-step instructions. main single writer.
New evidence research/evidence/crosssensor-upstream-provenance-v2.json binds preservedv1.
All3originalZIP probes now complete; do not retry ROI_0001. New311byte range at298677695 has
SHA59a07b49cc0b15bdd1b2ba490f04352605a8965ac923fb28ead1f14bee6931bc; oneURLError then success.
Decoded JSON SHA3194a648a927ff5ae22425e37169ec5407533295d9d1dafede126b450109e281.
The new JSON embedsROI_00004, actualS2 productID andNAIP ID; zero exactv2NAIP matches,
nearestv2center16.938km. All3probes have zero matches, not a catalogwide nonoverlap proof.
ZIP cumulative unique source bytes1167378; all size/CRC/decompression replays passed.

New bounded source lead: tacofoundation/tortilla_demo revision1e5af89d6895000583a6fd6678ece92a99d86fec,
sen2naipv2-crosssensor.taco size9716781901, declaredSHAc3f80d653e7e369f8e72f87969a4062101af5b683fd24948dac0d9a0aeca39d5.
Ignored cache artifacts/progressive-provenance/alternate contains2byteWX magic,16byte pointer,
Parquet directory at9716395212 length380067, IDs and schema/transport receipts.
All8000IDs and order match fixedsource; this does NOT establish pixel/file equivalence.
One existing predeclared new member NA5120_E1183N0757__m_3912321_nw_10_060_20220710
has nested parent2995776789 length1305795,18byte header, directory2997074063 length8521.
NoS2IDs in top or inspected nested schema. Only388624source bytes, no images/full object.

Indexed public code searches and inspected official/source-related files did not supply av2
mapping; see report for exact pinned URLs. This is not proof no public mapping exists.
Concrete next dependency: obtain directv2source manifest with memberID→S2product/GEEimageID,
complete mosaic/composite source sets, NAIPID, times/geometry and applicable revision.
Local unsent draft: docs/drafts/sen2naipv2-provenance-request.md. No external message authorized
or sent yet; posting/contact needs explicit user authorization. Do not loop over completed
probes or bulk request empty schemas. If no manifest can be obtained, design an explicitly
identified alternative data route; do not silently waive source-sharing constraints.
Independent sample-size/split freeze, fitting/calibration andGPU remain pending. OldC terminal.
No production code changed this resume.20ZIP tests, frozen source gate, offline evidence
replay and independent read-only evidence review passed. Final staged-policy/diff gates run
before commit. Local Hugging Face posting credential is unavailable (boolean-only check);
no credential value exposed. User can send draft or establish authenticated/authorized posting.
No full suite or old pixels/caches needed.

---

## ACTIVE RESUME CHECKPOINT — upstream provenance attempt, 2026-09-08

This newest section supersedes all older next-step instructions. Continue on main, single writer.
New bounded ZIP reader: research/trustmask/zip_metadata.py. Plan:
docs/superpowers/plans/2026-09-08-progressive-provenance.md. Evidence:
research/evidence/crosssensor-upstream-provenance-v1.json. Current bounded attempt is published;
all three probes are NOT complete. Do not interpret its partial delivery as a closed source audit.

Fixed upstream ZIP revision79e93461ad93911ebcd5aa0f342376e4c41e8743;2222325537bytes.
EOCD at2222325515 length22 SHA67037df532f56f997cd052e11e27ac2a0c7481d3fe4a252abba06008e7463696.
Central directory at2221159369 length1166146;11405entries,2851metadata.json.
Ignored durable cache artifacts/progressive-provenance contains verified ranges, entries,
selected.json, probes.json, both member-transport attempts, and probe-0.json/probe-2.json.
Only1167067unique source bytes retrieved, no TIFF or full ZIP; full-object SHA not verified.
Use .venv/bin/python; ZIP module needs no PyArrow/new dependencies.

Preselected first3JSONs: ROI_0000/metadata.json andROI_0002/metadata.json succeeded;
ROI_0001/metadata.json remains incomplete: body offset298677695 length311; local header at
298677602 length30 and prefix at298677632 length63 already cached. Reverify receipt SHA before
reuse; central member record/reader authorize reads. Initial per-range two attempts followed by
one missing-range pass with provenance_probe=2; URLError logs preserved. Do not endlessly retry.

Two actual S2 productIDs recovered, but their NAIP suffixes have0exact matches among8000v2
members; nearest v2 centers are8.456km and17.714km away. FolderROI_0000 embedsROI_00002;
folderROI_0002 embedsROI_00006. Never join by numerical ROI indices. Identity projections only;
quality numeric tokens not converted/printed/used. This is2schema observations, not bulk mapping.

Next: complete remaining tiny JSON when transport recovers, and seek a direct v2 construction
manifest mapping v2 members to actual S2 productIDs/mosaic source sets plus NAIP identity.
Checked public project descriptions did not supply this mapping; do not claim no public mapping
exists. Do not bulk request2851JSONs or8000nested directories without a justified mapping strategy.
No source-sharing closure, independence certification, sample-size/split freeze, fitting or GPU.
The earlier conditional Hoeffding sample-size planning remains valid only under its assumptions.
No external messages sent; contacting maintainers would need explicit sending authorization.

Validation:20ZIP tests and106research tests, research Ruff, frozen load_frozen source gate,
cachedSHA and completedJSON CRC/decompression replay passed. Reviewer overlap finding fixed
with regression before source expansion. Frozen src/trustsr and scripts/paper unchanged.
Independent evidence review and final staged data-policy/diff gates passed.

---

## ACTIVE RESUME CHECKPOINT — source/footprint audit delivery, 2026-09-08

This section supersedes the older terminal checkpoint below. Continue on main, single writer.
Completed the bounded source-footprints plan and preserved old evidence receipts.
New evidence: `research/evidence/crosssensor-source-footprint-audit-v1.json`.
All three predeclared nested probes completed: second and third directories succeeded after
one bounded retry; prior URLError attempt retained in receipt. No pixels, models, GPU or SSH.
All three directories lack LR scene IDs. All three top timestamps equal LR timestamps and HR
is one day earlier; this is only three observations, not a catalogwide rule.
Four geometry shards match pinned top metadata exactly; all8000 geometry summaries and7km
groups replayed.7km screen retains7269 members/5279 candidate groups, not independent samples.
Upstream SEN2NAIP revision resolved:79e93461ad93911ebcd5aa0f342376e4c41e8743. Its cross-sensor
folder only lists cross-sensor.zip (2222325537 bytes); archive not downloaded. Original paper
v1 sizes/counts differ from v2; identity mapping is not proven.

Next: design bounded upstream ZIP directory/metadata.json provenance checks and establish v1/v2
mapping or another justified source-identity route. Do not bulk fetch8000 nested directories.
LR-sharing audit, precise spatial separation, sample-size method and five-stage split freeze
remain pending. Conditional Hoeffding feasibility is in docs/reports/progressive-data-audit.md:
6-method risk family needs3045 independent test groups at0.03 empirical risk margin;6851 at0.02.
These are planning calculations, not observed results, independence claims or a frozen procedure.
No fitting/calibration/GPU work until these gates close. Old C remains inconclusive and terminal.

Verification:86 tests via `.venv/bin/python -m pytest -q tests/research`, research Ruff,
geometry/source projection equality, offline7km replay and frozen load_frozen gate passed.
Read-only implementation and evidence reviews approved; exact staged-policy and diff gates passed.
Direct `.venv/bin/pytest` lacked research import path; use python -m pytest. Environment unchanged.
Full research progress is not complete; no new empirical benefit results. The prior checkpoint
below preserves cache identities and historical details; its1-of-3 probe status is superseded.

---

## ACTIVE RESUME CHECKPOINT — terminal handoff, 2026-09-08

**Read this section first. It supersedes older “next” instructions below.** User requested
only saving the handoff before opening another terminal. No new experiments or network requests
were started for this handoff. Continue the approved progressive study when the user resumes.

### Working state and authorization

- Workspace `/home/wanghongxu/code/RemoteSensing001`, branch `main`, HEAD `7040975`
  (`feat(research): audit candidate metadata and isolate historical groups`). Single writer;
  continue on main, no worktree. Current implementation/evidence changes are **uncommitted**.
- User wants evidence fusion → spatial adaptation → risk-constrained calibration, all optimizing
  retained coverage under fixed expected ROI-max R9 loss ≤0.05, with convincing practical-benefit
  quantification and comparison/ablations. Approved design remains
  `docs/superpowers/specs/2026-09-08-progressive-coverage-design.md`.
- Prefer focused tests; no full suite, no cost questions. No GPU/SSH needed now. Ask user to power
  on only before work that actually needs GPU. Do not reopen old A/B/C pixels/caches or reuse
  observed Spain as independent confirmation. Old C remains `empirically_met_but_inconclusive`.
- New code stays in `research/trustmask/`, outside frozen `src/trustsr` and `scripts/paper`.
- Process check at handoff found no matching active trustmask/catalog/probe process. Ignored
  local metadata caches persist across terminals; they are not committed evidence.

### Actual progress since HEAD

1. **Controlled top metadata transport succeeded.** After intermittent TLS EOF failures, the
   final `fetch_http_range` implementation successfully retrieved all three known ranges and
   matched their previously recorded SHA256. Redirect handling stays body-free, HTTPS only,
   exact 206/Content-Range required. Never disable TLS or accept full-object fallback.
   The old v1 evidence and lower checkpoint still say final transport incomplete: preserve that
   historical receipt and publish a new receipt rather than silently rewriting its history.
2. Added `nested_directory_interval` to `research/trustmask/catalog.py`: nested header is exactly
   **18 bytes**, not 42; relative directory offsets are bounded inside the top-authorized parent,
   directory ≤64KiB. Seven new tests passed; catalog scope total **23 passed**.
3. Added `research/trustmask/footprints.py` and eight passing focused tests. Geometry only:
   north-up northern UTM, 520×520 at 2.5m, perimeter transformed with rasterio. All 8,000 top
   records have nominal side 1,300m and area 1,690,000m². Maximum centroid discrepancy
   0.07165529090313844m; sampled perimeter radius range 917.7795405006733–920.5813956013698m.
   Sampled radius is **not a certified enclosure**, and nominal area is not valid/nodata support.
4. Saved four small identity-bound geometry shards under `artifacts/datasets/` (2,000 rows each).
   Hashes below. Full per-row derived geometry stays in ignored cache.
5. A metadata-only **7km centroid-spacing sensitivity** gives 5,622 components; excluding all
   connected historical groups removes 731 members, leaving **7,269 members / 5,279 candidate
   groups**. All 360 historical IDs matched. Original 5km screen remains 7,477 / 6,158.
   Rationale: 5km center separation is not 5km footprint-edge separation; ~0.921km radii imply
   ~3.16km vs ~5.16km approximate edge gaps for 5km vs 7km center screens. This is a conservative
   preliminary screen, **not exact edge-distance certification or statistical independence**.
6. Nested source schema probe is only **1 of 3 complete**. First member:
   `NA5120_E1183N0757__m_3912321_nw_10_060_20220710`.
   Parent offset 2995954125, length 1305867; header 18 bytes; nested footer offset 2997251399,
   length 8593. Only `lr`/`hr`, geometry, timestamps, format and structural fields are present;
   no actual Sentinel-2 scene ID in this directory. LR shape 130² at 10m; HR 520² at 2.5m.
   Do not generalize one probe to all 8,000 members or substitute timestamp for scene ID.
7. **Important correction verified during handoff:** `validate_acquisition_times` in
   `src/trustsr/data/crosssensor_schema.py` validates top time and LR/HR timestamps plus signed
   HR-minus-LR UTC-date difference; it does **NOT** enforce top timestamp = LR timestamp.
   For the single completed probe top time equals LR time 1657490400, HR 1657404000,
   `days_between=-1`. This observation does not establish the relation across the catalog.
   Top `days_between` values are {-1,0,1}; do not bulk infer LR dates without evidence.

### Persistent local evidence and exact source

Fixed source `tacofoundation/SEN2NAIPv2`, revision
`c370504201072fdb1dd388013ab8c0fc7d00a57e`, object `sen2naipv2-crosssensor.taco`,
size 9717583850. Declared full SHA
`c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5`
is not locally full-object verified. No image asset payload was read.

Ignored cache directory: `artifacts/progressive-metadata/`:

| File | SHA256 / meaning |
| --- | --- |
| `0-42.metadata` | `0ae2b7363e73464ffd0c0c64e890277d9c0718caa3786bf7166aa8fc12937c93` |
| `9717576621-7229.metadata` | `ef1505137d5431bfcc1098a91482f1c52e6a99edb8ab58c5e1f45a73ae38af03` |
| `9716971212-605409.metadata` | `9eb6bc8f5e52b3ca9f0aba0a733c98b95e579e6b5c02db515f44cc6d36921ea1` |
| `2995954125-18.metadata` | `7a0c9ced8c58aa4d8292a359db3c6d37d0b5b4b7ae6ca1e77f257af76197b78d` |
| `2997251399-8593.metadata` | `c5e0b909cf81b06bef5b9a4c9136c14c27893ae1488339f3c4c49ee1993866a7` |
| `nested-first-probe.json` | Allowed identity/time/geometry projection and nested receipt |
| `nested-probe.json` | Empty list from failed initial parallel probes; not a success receipt |
| `9483557768-18.metadata` | Additional cached header only; verify its member/bounds before reuse |
| `footprint-summary.json`, `footprint-summary-rows.json` | Aggregate and all-row geometry diagnostics |
| `candidate-groups-7km.json` | Full audit object, eligible groups and excluded member list |

Successful top source payload total: 612,680 bytes, excluding redirect/network overhead.
Network errors were intermittent TLS EOF; successful redirect target was `us.aws.cdn.hf.co`.
Never log signed URL query values. Cache each verified range durably before subsequent requests.
Use `.venv/bin/python`; PyArrow19.0.1 lives separately at `/tmp/trustmask-metadata-libs`:
`PYTHONPATH=/tmp/trustmask-metadata-libs .venv/bin/python ...`. Do not upgrade the main environment.

New geometry shards (untracked; each ~261KB, eligible for small-metadata Git policy):

- `artifacts/datasets/progressive-crosssensor-geometry-1.json`: SHA256 `eecfd7ef9d2fc9e153a44fc2bdf35cca9724d84340bf9e2890a9e833a87bb5ef`.
- `artifacts/datasets/progressive-crosssensor-geometry-2.json`: SHA256 `41f343b33cb05af430e33bc7a8fa4cb99fdeddaeeb16659807eede68896b82e4`.
- `artifacts/datasets/progressive-crosssensor-geometry-3.json`: SHA256 `008d669135a51660234cd592ca7ad5f2334f6acd7cd355d6ea94697130684db9`.
- `artifacts/datasets/progressive-crosssensor-geometry-4.json`: SHA256 `98b152f5b635f8f25593df70e566b81c9ae14efd32c6f18a2f1a3f6e3e24594a`.

### Exact resume order and incomplete checks

1. Read `docs/superpowers/plans/2026-09-08-progressive-source-footprints.md`; its checkboxes
   have not yet been updated. Transport/geometry implementation succeeded, publication is pending.
2. Finish or explicitly record partial status for the remaining two predeclared schema probes:
   `NA5120_E1183N0791__m_4112455_se_10_060_20220620` and
   `NA5120_E1185N0724__m_3812242_ne_10_060_20200602`. These are from the original 5km eligible
   inventory, chosen lexicographically before 7km sensitivity. Only bounded headers/directories;
   no image data and no bulk 8,000-member requests until source-identity strategy is justified.
3. Research actual LR provenance. Fixed collection raw link points to
   `https://huggingface.co/datasets/isp-uv-es/SEN2NAIP`; root tree discovery returned cross-sensor,
   demo, synthetic folders. Resolve immutable revision before relying on that source. Original
   paper https://www.nature.com/articles/s41597-024-04214-y discusses per-ROI metadata.json,
   but v1-to-v2 identity mapping is unproven. https://github.com/ESAOpenSR/opensr-degradation
   is another primary construction reference. Possible public Sentinel-2 STAC source-set grouping
   needs a justified date relation and catalog completeness; it is not implemented or approved
   as an independence claim. Do not treat this speculative idea as the current frozen method.
4. Publish new transport/geometry/nested evidence receipt, update
   `docs/reports/progressive-data-audit.md` and plan status, preserve old receipts. Consider a
   reproducible offline geometry audit entry point; current aggregate execution used local snippets.
5. Finish source-sharing audit, statistical sample-size feasibility and five-stage split design
   before fitting/calibration or GPU work. New empirical benefit results do not exist yet.
6. Review changed code. Existing read-only reviewer `review_external` can be reused under the
   requesting-code-review skill. Current changes have **not** had final review.
7. Run only relevant checks: `pytest -q tests/research` (expected count 86 from prior 71 + 15 new,
   **not yet run as a combined scope**), changed/research Ruff, `git diff --check`, original frozen
   protocol `load_frozen` source gate, then tracked-data policy against the staged exact file set.
   Individual catalog 23 and footprint 8 tests and changed-file Ruff previously passed; no claim
   that all final checks passed. Do not add ignored raw metadata to Git.
8. Commit the reviewed work on main and report actual progress. Do not repeat the former old-scope
   90–95% estimate or claim new experiments are finished. This handoff itself does not commit code.

Uncommitted implementation files at handoff: `research/trustmask/catalog.py`,
`tests/research/test_catalog.py`, new `research/trustmask/footprints.py`,
`tests/research/test_footprints.py`, four geometry shards, source-footprints plan, and this handoff.
Original frozen protocol remains `paper/protocols/spain-external-frozen-v1.json`, expected SHA
`798200c5c8b2cd9102755202810e8161832064de686a81b41c02776ddaf85525`.

---

## Latest checkpoint: new metadata inventory and historical exclusion audited

Metadata-only public crosssensor source audit yielded8000 members. All360 published historical
IDs matched; source-sharing plus5km centroid-connected grouping excludes523 members, leaving
7477 members in6158 candidate groups. These are NOT certified independent test samples.
No quality columns decoded, no pixel/model/cache or SSH/GPU access. Persistent small identity
projection is artifacts/datasets/progressive-crosssensor-members-v1.json (SHA30013c7285c0aa91b1f4de48fd1734db314a81b71a8da3809294d3ee919d9cd6).
See [data audit](reports/progressive-data-audit.md) and research/evidence/crosssensor-catalog-audit-v1.json.

Implemented bounded TACO metadata header/HTTP range checks, body-free redirects, fixed historical
SHA gates, source/spatial grouping and five-stage structural partition checks.71 focused tests
and Ruff passed, read-only review issues fixed. Old frozen implementation remains unchanged.
Initial3metadata ranges succeeded; after redirect hardening,2 attempts hit TLS EOF. Final network
retrieval has NOT completed, but final history/group logic exactly replays the stored projection.
Do not redownload old pixels or call all6158 groups independent. Next: validate hardened metadata
retrieval and audit LR-source sharing/full footprints, then independent sample-size and split
freezing. Current work needs no GPU; notify user before GPU-dependent steps.

## Latest checkpoint: approved progressive study, local foundation delivered

User approved the design and explicitly prioritized comprehensive experiments and measurable
practical benefit. Implemented research/trustmask scoring decomposition/fusion and offline mask
workload accounting, plus a labeled synthetic demo. New code is outside the frozen source tree.
42 focused tests and changed-file Ruff passed; read-only code review issue fixed by regression.
See [foundation report](reports/progressive-coverage-foundation.md), design and implementation
plan dated2026-09-08. This is NOT new empirical evidence or a calibrated deployment package.

Scientific target is equal geographic-group / within-group ROI mean loss and coverage, with
physical workload totals separate. Main comparison is full method versus existing neighborhood;
6-row minimal matrix and additional ablations. Joint risk/benefit delta allocation explicit.
Review tiles/area are proxies; human time is only modeled until actually measured. No GPU/SSH,
model or pixel/cache access. Public fixed HF tree lacks a standalone member metadata catalog.
Next is bounded metadata extraction, old-group overlap exclusion, independent sample design,
then fitting/calibration and full run protocol. Real new experiments remain pending. Notify user
before GPU-dependent work; no repeated cost or routine design approval questions.

## Latest user direction: one optimization goal and three dependent studies

User now wants three genuinely progressive studies and final comparison with 2–3 methods.
See [new research proposal](reports/first-paper-progressive-research-proposal.md): maximize
retained coverage under a fixed ROI risk target, via evidence fusion → spatial adaptation →
calibration/independent evaluation. This is a proposal, not a frozen method or completed result.
Old experiments remain complete and unchanged, but the expanded objective requires new work;
do not repeat the old 90–95% completion estimate as if it covers this new scope.
Do not reopen A/B/C pixels/cache or use observed Spain as independent evidence for a new method.
Next: mechanism/literature, fresh geographic split and sample-size feasibility, then protocol.
No GPU is needed for the current design; notify user before GPU-dependent work.

## Latest writing refinement: connect the three contributions

The manuscript now organizes the questions as score evaluation → spatial adaptation → limits
of threshold use. This is a progression of questions, not a validated serial algorithm: the
neighborhood score reuses K5 predictions, while threshold transfer applies only to original K5.
Do not claim that neighborhood AURC gains imply calibrated coverage or transferred risk control.
Only abstract/introduction/discussion/conclusion wording changed; no new results or experiments.

## Latest checkpoint: manuscript methods and literature review delivered

User authorized the next step and instructed us to pause and ask them to power on if GPU is
needed. This checkpoint is local documentation only: no SSH, raw data, cache or model access.
Reviewed fixed source/protocol and published small evidence; expanded score formulas, units,
deployment calls, sampler settings, calibration threshold rules and pinned OpenSR parameters.
Explicitly distinguish joint neighborhood texture variance from smoothed K5, and compute R9
before masking. Retain unfavorable results and the original inconclusive internal terminal.

See [completion review](reports/first-paper-completion-review.md) and `paper/manuscript.md`.
Six direct references checked against primary/official sources; no dependency upgrades.
Frozen protocol/source gate and science/execution hashes passed unchanged; no scientific/code
artifact diff and no full tests. Prior table/figure reconstruction is already complete; do not rerun.
Task 8 generic draft and method review delivered; target venue/type/format selection remains.
No institutional requirements have been supplied; do not invent them or ask about costs again.
Next work is submission positioning/formatting and author-provided metadata, not another GPU
benchmark. If a later authorized study requires GPU, notify the user before dependent work.

## Latest checkpoint: full Spain study, independent replay and figures completed

User authorized the next study, supplied the server and one-hour availability, and requested
parallel acquisition. Task 7 now completed under source 2c851de and the final protocol below.
377 deployed text/source file hashes and original runtime preflight passed. Direct cloud
download was unreachable; downloaded both fixed official packages locally in parallel and
relayed them to a new durable directory, verifying size/SHA at both ends. Restricted decoding
and all embedded members passed without changing scientific code or the allowlist.

All336 prediction slots stored, zero failures, full28 Crops/20 Urban members; 2 science replays
byte-identical and manifest-last publication succeeded. Main process1234.032s; separate-process
CPU-only replay378.192s passed and all five published result/state/runtime files stayed unchanged.
Science SHA `f09fc42555cb00deac5125e478b61817200de9a796bd668e5abcc15fbfb07fe4`;
execution evidence SHA `8eaa870b5677bbfb856fdf6533f76f054d8053e33820a89d67e62a780f888651`.
See [full results](reports/first-paper-spain-results.md), `paper/tables/spain-science-v1.json`
and `paper/tables/spain-execution-v1.json`. Only small text evidence/scientific figures enter Git.
All cloud work finished, outputs synced/read back; no task/GPU compute processes remained.
User was told the server can stop while retaining the durable disk. Do not rerun this study.

K5 mean R9 AURC is6.1579%/8.6360% lower than LR in Crops/Urban, with28/28 and20/20 valid pairs.
Fixed threshold mean coverage47.4338%/23.5099%, mean ROImax R9 loss .04035626/.04878006.
Neighborhood w3 has lowest mean R1/R9 AURC in both subsets. Preserve unfavorable comparisons:
Crops R1 three-model AURC is lower than K5; both subsets' R1 rho is higher for three-model.
All findings remain descriptive. Structural softmin averages cover only upstream common finite
support, not all230400 cropped pixels and not hard hallucination fractions. Old C stays terminal.

Added postprocessors under `paper/tools/` (outside frozen implementation tree):5 CSVs,2 summary
plots and fixed first-two-ROI panels per subset.3 focused renderer tests pass, no full suite;
independent text/code review passed, figures inspected. Panel guard fixes preserve identical PNG.
`src/trustsr` and `scripts/paper` unchanged; final frozen protocol remains valid.
Updated manuscript working abstract/results/conclusion and reproducibility. Next is Task8 full
paper quality/methodology/literature review and submission preparation, NOT costs or more GPU
diagnostics. No need to ask again about pricing or authorization already supplied in this session.

## Latest override: user manages costs; freeze functionality and final protocol delivered

User explicitly said to disregard their costs and focus on functionality. This overrides ALL
historical hourly-price/billing-budget blockers below. Do not ask for price or cost approval again.
Added text-only `python -m scripts.paper.run_spain freeze --output ...` and `build_frozen`.
User-managed cost mode omits price/currency and binds the existing timing evidence, 2100-second
estimate and 2700-second run timeout. Canonical SHA, implementation, science, runtime, data and
model checks remain; old monetary protocol shape stays supported. Conflicting files cannot be
overwritten. No cloud/raw data/model/cache access was needed to implement or freeze this.

Final `paper/protocols/spain-external-frozen-v1.json` SHA:
`798200c5c8b2cd9102755202810e8161832064de686a81b41c02776ddaf85525`.
Current draft SHA: `ac5ce4302fc1766d162967403961ef7a4366be27f15b04fdf76e080d0cbbb594`.
See [freeze record](reports/first-paper-freeze-record.md) and updated execution runbook.
29 protocol/CLI tests and 5 execution/publication tests passed; seven new cases were red first.
Changed-file Ruff and read-only code review pass. No full-suite or completed diagnostics rerun.
Next: functional integration/deployment with this exact source and protocol, then the planned
external study as authorized by the session; actual Spain compatibility/results remain untested.
Do not reinterpret this as another cost/permission questionnaire. Old C remains terminal.

## Latest checkpoint: freeze review packet prepared; actual price still pending

User requested continuation. Completed the local-only
[freeze review packet](reports/first-paper-freeze-record.md): exact two-package/48-ROI scope,
336 prediction slots, candidate/code/budget identities, normalization and remaining source
limitations, failure/display rules, prior validation and unchanged C acceptance identity.
Updated the design's stale decoder/structure/automatic-retry wording to match verified code.
No Python/protocol/evidence changes; candidate SHA remains the one recorded below. No cloud
connection, raw input/cache/model reads, inference or test-suite rerun this turn.
Verified candidate bytes against current implementation, pinned evidence and linked files.
Hourly rate and billing granularity requested again; await the actual values to finalize cost.
Task 6 review preparation is delivered, but final freeze and Task 7 execution remain incomplete.
Do not repeat completed measurements. Final budget/authorization decisions must use the session
context and concrete completed packet, not assume every continuation requires new permission.

## Latest checkpoint: original-cloud CPU measurement complete; price pending

2026-09-08 user supplied the original server for the previously scoped CPU-only measurement.
Deployed local 07103bf archive to a new durable directory and verified all 372 source/text files.
Existing base environment unchanged; CUDA hidden. One full-grid synthetic benchmark completed:
decode0.583128s, cache-write proxy0.228323s, two identical science replays7.466629/6.708647s.
25 focused cloud tests passed in29.70s. No real inference, Spain or historical-cache access.
Sources were fsynced/read back and only three small path-free JSON records were copied locally.
No task/GPU compute processes remained; user was told server can stop while retaining the disk.

Evidence `paper/tables/cloud-cpu-execution-v1.json` SHA-256:
`4467cfeb57d291d96fc57bd4c35378829727d7fe6cacf9f73650e2fd85a0c6dc`.
See [cloud CPU evidence and budget](reports/first-paper-cloud-cpu-budget.md).
The full measured runtime version map now populates the draft and is mandatory at the freeze gate;
22 local protocol/CLI tests pass. No full-suite rerun. No source changes were merged from the cloud.
Current draft SHA-256 (supersedes historical draft hashes below):
`dfe9239b162136a15af532810674a89fbc4e7252520bad2d90306e9f4ef7897c`.

48-ROI nominal inference+two-replay+cache-write projection is1122.132718s (~18.70min).
Budget draft proposes35min including explicit input/loading/variance allowances, with a45min
run-only limit; these are assumptions, not a billing bound or execution authorization.
Hourly price and billing granularity are still missing and were requested from the user.
`paper/protocols/spain-budget-draft-v1.json` records the arithmetic; execution protocol remains
DRAFT with budget=null. Do not guess price or start Spain. No further diagnostic rerun is needed.
Next: receive price/billing terms, review the concrete budget, finalize the protocol and obtain
remaining dedicated external/GPU execution authorization. Original Phase 2B3-C stays terminal.

## Latest checkpoint: formal execution entry implemented; protocol remains draft

2026-09-08 continuation implements `python -m scripts.paper.run_spain` with draft/preflight/run/
replay commands. Exact package/member/model/science/implementation identities are protocol-bound;
actual imported code must come from the repository. Missing-only predictions persist/fsync an
attempt before inference; resumed failed/interrupted slots are not retried automatically. Complete
caches do not construct models. Two cache-only science reconstructions must agree before the
no-overwrite manifest-last publication. Input normalization receipts and explicit failure reasons
are retained; timing is separate. See [execution runbook](first-paper-spain-execution-runbook.md).

Machine-readable candidate: `paper/protocols/spain-external-draft-v1.json`. It is intentionally
non-runnable: status=draft, unknown cloud runtime versions and budget are null. Do not fill these
with guesses or flip status merely to pass gates. Final freeze and dedicated external access/GPU
authorization have not occurred. No cloud, Spain, old B/C or real model access happened locally.

Verification: 70 scoped execution/replay/score tests passed (18.98 s) plus 7 data-policy tests
(0.97 s); no full suite. Changed-file Ruff and whitespace checks pass; prior artifacts, data/model
modules and old score/replay cores have no diff. Draft SHA-256 is
`d0c6fa3e2672b03fe7af07f90bc6a7c6a625e6afae31d45a4dfbe9a7afa98e14` and matches current source hashes.
Read-only review found no remaining blockers after the fixes below.

Review fixes: state directory fsync before inference (plus new-run parent sync), import-origin
validation, canonical protocol SHA identity, normalized JSON return on completed resume.
Next needed input: availability of the original cloud server and its hourly price, requested from
the user for CPU-only synthetic timing/runtime inventory. Continue without redoing the completed
GPU timing or SEN2SRLite diagnostics. Current implementation uses the existing environment only.

## Latest local checkpoint: restricted package decode and authenticated replay cores

2026-09-08 user requested continued execution with fewer full-suite runs. New local-only
Task 5 work implements restricted hash-gated package decoding, package-SHA source identities,
seven-slot cache-authenticated replay, partial-method/paired/transfer denominators and isolated
OpenSR correctness with actual harmonization registration failure detection. A subset replay
binds expected model-provenance SHA; the final protocol must supply that trusted expected value.
See [integration evidence and remaining work](reports/first-paper-external-local-integration.md).

Validation: 77 focused tests + 7 local data-policy tests passed; no full suite rerun. Full-grid
synthetic package→cache→five scores→real OpenSR→summary ran twice with byte-identical science;
replay times 4.185629/4.037788 seconds on local CPU, NOT the frozen cloud environment or a price
projection. Changed-file Ruff and whitespace checks pass. No cloud/Spain/B/C access occurred.

Next: formal execution CLI binding frozen package/member/model identities, missing-prediction
execution and failure reasons, final source/nodata disclosure, cloud CPU end-to-end budget,
and protocol freeze. The new cores are not a complete authorized external runner. Real package
pickle compatibility is untested; unknown types must stop, never trigger unrestricted fallback.
Do not rerun the completed GPU timing or SEN2SRLite diagnostics. Work sequentially on main.

## Latest override: SEN2SRLite discrepancy resolved on the cloud profile

2026-09-08 CPU-only follow-up supersedes the unresolved-thread hypothesis below.
On the same three fixed development ROIs, 1 thread reproduces timing hashes and 96 threads
reproduce historical A hashes, each repeated twice. One-thread differences are small but two
R1 AURC deltas exceed the stated diagnostic tolerance; do not call them universally negligible.
New `CloudSEN2SRLiteX4` pins 96 threads during loading/predict, restores caller settings even on
failure, and versions cache provenance. Use only serial, single-worker CPU execution; old adapters
and historical results remain unchanged. Actual cloud adapter verification matches 3/3 old hashes.
This profile's three-call mean is 0.888265 s (including first cold call), replacing 0.209878 s
for future SEN2SRLite budget projections; nominal inference-only estimate is now 7.179 minutes.
See [diagnosis and handoff](reports/first-paper-sen2sr-reproducibility.md) for numeric evidence,
scope, limitations and remaining gates. No Spain/B/C access or new GPU inference occurred.
Next: safe decoding, authenticated external runner and per-method failure accounting, OpenSR
failure rules, end-to-end budget and final protocol freeze. Do not rerun completed diagnostics.
Only `main` exists locally and remotely; integrate directly, no artificial merge or cloud-code merge.
Verification: full local suite completed with exit 0 (2492 tests, 43 existing JIT deprecation
warnings); 22 focused model tests and 5 cloud policy tests passed. New-file Ruff, staged data
policy, canonical evidence hashes and cloud adapter source hash all pass; read-only review found
no blocking issues. Both cloud JSON records were synced and read back; no GPU process remained.
The user was notified that the server can be stopped while retaining its data disk.

## Current objective: approved first-paper research direction

The user accepted the recommended research direction on 2026-09-07 and requested a new plan
and Git publication. The first paper now focuses on uncertainty-score cost, error-scale
sensitivity, and frozen calibration transfer for Sentinel-2 RGBN x4 super-resolution.

Read these documents before starting new work:

- [Current first-paper roadmap](research-roadmap.md);
- [Task-by-task research execution plan](superpowers/plans/2026-09-07-first-paper-research.md);
- [Literature review and scope rationale](reports/2026-09-07-research-replan.md).

The user subsequently authorized execution, with a stop to request GPU resources when needed.
Tasks 1–2 have evidence, metric, manuscript, and rendering-design documents. The development
neighborhood comparison has now completed; Spain remains a draft, not a final freeze.
The read-only renderer and synthetic neighborhood module are now implemented. Three CSV tables
and the development R1/R9 PDF are generated. Software checkpoint: 137 scoped local tests;
cloud timing preparation: 27 tests (the earlier neighborhood run had 28 scoped cloud tests).
The local OpenSR contract checks emit 43 existing PyTorch JIT deprecation warnings.
Cloud development reuse verified 120 ROIs, 240 assets and 600 K5 predictions. Windows 3 and 9
were evaluated twice on CPU with byte-identical science output; the preregistered rule selected 3.
See [development results](reports/first-paper-neighborhood-development.md). No inference or B/C
pixel/cache access occurred. The existing base environment was used; the user explicitly approved
installing pytest 8.4.2 and iniconfig 2.3.0. No new environment was created.
New science and runtime records are durable; only the small science JSON was copied locally.
The user was notified that the server may be stopped while retaining durable storage. No job remains.
Next: resolve versioned Spain contracts, build the independent external entry, and measure missing
inference costs before final protocol freeze. Do not repeat the completed development selection.

Latest local follow-up: a pinned text-only metadata projection now records 28 crops and 20 urban
ROIs without quality columns. Before any image access, the proposed image version was changed
from 021 to 100 to match the available CSV version; both package identities remain documented.
This is not a claim that embedded metadata/tensor order has been verified. New input decoding,
nodata/range rules and final compute budget still block the external freeze.
OpenSR 1.3.3 contract tests establish cropped grids, soft class shares and missing-value semantics.
See [OpenSR contract](reports/first-paper-opensr-contract.md). No cloud reconnection or image download
was needed for this follow-up. Keep structural diagnostics image-level pending the remaining checks.

2026-09-08: the active user goal is to continue until GPU is genuinely needed, without stepwise
approval. New `spain_inputs.py` implements strict decoded-array preparation and member-to-row
binding, with 23 new tests; 110 scoped tests pass (43 existing JIT deprecation warnings).
This is NOT a complete package loader: hash-gated safe decoding, five-score external aggregation
and final protocol/budget gates remain. No Spain pixels were accessed, and the goal remains active.
Next local checkpoint: `external_scores.py` now computes all five R1/R9 diagnostics on a shared
center prediction, descriptive fixed-threshold K5 transfer, and the primary equal-ROI paired
summary with exhaustive whole-ROI failure accounting. Its score builder accepts no HR input.
16 new synthetic tests and the scoped regression suite pass: 126 tests, 43 existing warnings.
This is an array-level core, not a complete external runner: prediction authentication, partial
method failure accounting, safe package decoding and GPU timing remain outstanding.
The bounded hardware timing entry is now `python -m scripts.paper.benchmark_inference`;
scope and the exact three development IDs are in the compute-budget report. It retains SEN2SRLite
on CPU (matching frozen A), measures 1 cold + 15 LDSR calls, and never writes predictions or
computes quality metrics. No CUDA causes failure before any data/model load or output creation.
11 additional CPU checks pass; scoped total is now 137 (43 known warnings). Real GPU timing has
NOT run. It is the next resource-dependent step, not evidence that remaining external CPU work
or protocol freeze has finished. Request a short GPU session before running it.
Final local verification for this checkpoint: full suite 2487 passed, 43 known JIT warnings
(778.23 seconds); new-file Ruff, staged data policy and frozen-tree diff checks pass.
Read-only review found no blocking issues. That local checkpoint involved no cloud or Spain access.

2026-09-08 GPU follow-up: the user supplied the server after the resource request. The bounded
timing run at `49a7e39` completed with exit 0: 3 bicubic calls, 1 cold + 3 SEN2SRLite CPU calls,
1 cold + 15 LDSR GPU calls, exactly the fixed first three development IDs. Cloud 27 CPU tests pass;
no dependencies/environment changed. Timing is durable, GPU processes are absent, and the user
was told the server may be stopped while retaining storage. Do NOT repeat this measurement.
LDSR mean 1.615943 seconds; prediction-window allocated peak 3.849831 GiB, reserved 5.078125 GiB.
Nominal 48-ROI inference-only projection is 6.637 minutes, not an end-to-end or price guarantee.
Only the small, path-free timing JSON was synchronized locally; no images/models/predictions.
New-vs-published prediction hashes: LDSR 15/15 and bicubic 3/3 equal; SEN2SRLite 0/3 equal despite
matching inputs, weights and recorded identity fields. Cause and error magnitude are unresolved;
CPU thread settings are a hypothesis, not an established explanation. Investigate before final freeze,
without changing historical results or using Spain for debugging. See
[timing report](reports/first-paper-inference-timing.md) and
`paper/tables/inference-timing-v1.json` for source hashes and exact measurements.
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
