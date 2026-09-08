# Bounded upstream provenance audit

> **For agentic workers:** Use superpowers:executing-plans task-by-task; main single writer is already authorized.

**Goal:** Determine whether pinned upstream ZIP metadata can supply actual v2 LR identities.
**Architecture:** Reuse exact HTTPS ranges; start with final22bytes (no-comment EOCD only), then its bounded central directory. A tested metadata-only reader validates central/local agreement before fetching at most three lexicographically first metadata.json entries. Publish identity projection and negative findings without claiming an unproved version mapping.
**Tech Stack:** existing Python stdlib, existing catalog transport; no installs.
**Spec:** docs/superpowers/specs/2026-09-08-progressive-coverage-design.md; latest handoff next step.

## Constraints and files

- Upstream isp-uv-es/SEN2NAIP revision79e93461ad93911ebcd5aa0f342376e4c41e8743, cross-sensor/cross-sensor.zip,2222325537bytes. Declared SHA4fe9cc4f292320568fea7e39fc86d97fed13ea8cf8f0e37823aa73c4e54019d2 is not full-object verified.
- HTTPS exact206 only; no whole ZIP, no generic remote seek, no images/old pixels/models/GPU.
- EOCD22bytes; reject comment, ZIP64, multi-disk. Directory≤8MiB, exact count/length.
- Selected JSON: unencrypted stored/deflate only; compressed≤64KiB, uncompressed≤256KiB. Read local30byte header then name/extra, require agreement and boundaries before content. Verify size/CRC/decompressor completion. Reject other file names before any fetch.
- Inspect top-level JSON keys; select identity/time/geometry only, skip quality/histogram values without numeric decoding or printing. At most3 schema probes; no bulk source extrapolation.
- Cache verified ranges in ignored artifacts/progressive-provenance, small derived evidence only committed.
- Code research/trustmask/zip_metadata.py; tests/research/test_zip_metadata.py; research/evidence/crosssensor-upstream-provenance-v1.json; docs/reports/progressive-data-audit.md; docs/codex-handoff.md.

## Tasks

- [x] Test real synthetic ZIP with image before JSON; permitted reads exclude image payload. Test malformed EOCD/directory/local-header, disallowed path, CRC and bounded decompression. First run must fail for missing module.
- [x] Implement eocd_interval(raw,total), directory_entries(raw,start,count), read_metadata_member(fetch,entry,directory_start). Focused test/Ruff; review before network expansion.
- [x] Retrieve pinned central directory and three predeclared JSON entries; preserve range receipts and failed attempts. Resolve source keys before selecting values; compare identity naming/geometry with v2 only if justified. Check official construction references if mapping absent.
- [x] Publish evidence and source-sharing consequences; no actual split freeze. Relevant research tests, frozen source gate, reviewed claims, staged policy/diff gates, commit on main.

## Execution status

Central directory complete:11405entries,2851metadata.json;20ZIP and106research tests pass.
Read-only review overlap issue fixed with a failing-then-passing regression.
Two of three selected JSONs complete (ROI_0000 andROI_0002);ROI_0001 body remains unavailable
after bounded requests. Do not mark all probes complete. Both successful JSONs contain actual
s2_id, but neither NAIP ID matches v2; ROI folder number differs from embedded roi_id.
Receipt records exact pending interval and all observed failures. Bulk mapping and independent
source groups remain unproved. Report/evidence publication completes this bounded attempt;
network-dependent third probe remains pending. No pixels or full ZIP downloaded.

Final independent evidence review verified rangeSHA,byte totals,JSON replay,zeroNAIP matches
and nearest-distance arithmetic. Exact staged data-policy and diff gates passed.

## Later resume completion

All3ZIP JSON probes now complete; missing311byte body retrieved and replayed. Newv2receipt
preserves historical partialv1. All3NAIP suffixes have zero exact matches in fixedv2 inventory.
Bounded additional provenance lead: tacofoundation/tortilla_demo revision
1e5af89d6895000583a6fd6678ece92a99d86fec, same basename but different declaredSHA.
Read2byte magic then16directory pointer bytes;380067byte top directory has the same8000IDs/order
but no S2IDs. One previously predeclared new member nested18byte header/8521byte directory
also lacks S2IDs. No other alternative nested records, images or whole objects read.
The audit plan is delivered; source-sharing gate remains open. Local maintainer inquiry draft
is ready but unsent. Do not repeat the completed probes or assume source equivalence.
