# Progressive source and footprint audit

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close metadata transport verification, inspect nominal footprints and determine whether nested source identities support group isolation.

**Architecture:** Reuse pinned top-directory ranges and project geometry only. Authorize nested18-byte headers and their bounded metadata directories for the first three lexicographically sorted eligible members; never read nested image assets. Sampling is a schema probe, not an all-member source audit.

**Tech Stack:** existing NumPy/SciPy/rasterio; isolated PyArrow19.0.1.

**Spec:** `docs/superpowers/specs/2026-09-08-progressive-coverage-design.md`

## Constraints

- No old A/B/C pixel/cache or source asset payload access; no GPU/SSH.
- Known top range SHA must match before geometry or nested-container offsets are used.
- Nested18-byte header magic#y/WX, relative footer offset≥18, length≤64KiB, inside its top-authorized parent; parent must precede top directory, file_format=TORTILLA.
- Only decode identity, timestamps, geometry, sizes and source identifiers; no quality columns.
- Nominal footprint does not establish nodata/valid support or statistical independence.
- Initial nested probe contains three new eligible members only. If actual sourceIDs are missing, record it and research original dataset construction before bulk requests.

## Tasks

- [x] Verify all three top metadata ranges through final controlled redirects and publish receipt; retain previous failures.
- [x] TDD `nested_directory_interval(header,parent_offset,parent_length,directory_start)` in catalog.py: exact18byte header, integerbounds, parent containment, length caps, no pixel fallback; test relative offset vs absolute and overflowing containers.
- [x] TDD `footprint_summary(crs,transform,shape,centroid)` in footprints.py: require finite north-upUTM-N,520²,2.5m; transform perimeter metadata to lonlat using rasterio; return nominal side/area, centroid discrepancy and maximum sampled perimeter radius. Test real synthetic UTM construction, malformedgeometry, no mutation.
- [x] Run geometry projection for all8000 using fixed topfooter, preserve only metadata summary and identity-bound geometry shards under1MiB each. Report centroid mismatch and conservative isolation implications; no source-pixel reads.
- [x] Probe first3 lexicographic eligible nested directories, persist metadata byte receipts and source-field projection. No direct-call path may read asset data.
- [x] Document source completeness and sample-size consequences, update handoff, run focused tests/Ruff, oldprotocol/data-policy gates, commit.

## Delivery status (2026-09-08 resume)

All three predeclared probes completed after one bounded retry; none contains LR scene IDs.
New receipt: research/evidence/crosssensor-source-footprint-audit-v1.json. Prior receipts retained.
All geometry shards match pinned top projection; geometry and7km grouping replayed offline.
86 research tests, Ruff and frozen source gate passed; implementation read-only review approved.
Source completeness remains unresolved; conditional sample-size consequences are documented in
docs/reports/progressive-data-audit.md. This completes this bounded audit plan, not the larger
source-sharing audit or five-stage split freeze. No bulk requests, pixels or GPU authorized here.
