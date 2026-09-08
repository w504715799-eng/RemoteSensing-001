# Progressive Metadata Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Audit fresh candidate membership without reading image payloads, then enforce geographic/source and historical exclusions before any new split.

**Architecture:** A bounded legacy TACO header decoder authorizes only collection JSON and top-level Parquet byte intervals. A separate metadata-only audit projects identity/location fields, excludes seen connected components, and reports candidate counts; it does not authorize experiments or assert independence.

**Tech Stack:** existing Python, NumPy, SciPy; PyArrow 19.0.1 in a separate temporary target for Parquet inspection, not the project environment.

**Spec:** `docs/superpowers/specs/2026-09-08-progressive-coverage-design.md`

## Global Constraints

- No old A/B/C pixels/cache or Spain pixels; no SSH/GPU.
- Pin SEN2NAIPv2 revision c370504201072fdb1dd388013ab8c0fc7d00a57e and crosssensor object size9717583850/SHA c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5.
- Header42 bytes, maximum top-level directory8MiB, collection64KiB; validate206, exactContent-Range and body length; reject whole-file200 before body reading. No automatic fallback to full download.
- Read only identity/centroid/geometry/time/source columns; do not inspect correlation or quality values.
- Metadata-range receipt hashes do not establish the complete object SHA.
- New code outside src/trustsr and scripts/paper; only focused tests.

## Task 1: Bounded metadata intervals

**Files:** research/trustmask/catalog.py, tests/research/test_catalog.py.
**Interface:** `metadata_intervals(header: bytes, total_bytes: int) -> dict[str, tuple[int,int]]` returns offset/length for collection and directory. ValidateWX/#y magic,42byteheader, nonoverlap, positive lengths, bounds/caps. `fetch_metadata(fetch_range, total_bytes)` calls header then JSON collection validation(version0.4.0) then directory, rejects non-PAR1 bytes. Callback accepts(offset,length), no arbitrary assets.

- [x] Test valid hand-built header and exact3calls; corruptmagic/truncation/overflow/overlap/range caps rejected before followup reads; wrongcollectionversion preventsdirectoryfetch; invalidParquetmarkers rejected.
- [x] Observe missing module fail; implement bounded decoder and callback orchestration, run tests.

## Task 2: Source/spatial connected components and history exclusion

**Files:** research/trustmask/grouping.py, tests/research/test_grouping.py.
**Interface:** `audit_groups(rows, seen_ids, *, radius_km=5.0) -> dict`. Row fields id,lon,lat,source_ids list of qualified nonempty source strings. Union rows sharing any source id or within inclusive geodesicradius, then exclude entire component touching seenid. Return deterministic groupIDs/members, counts, missing_seen_ids; caller must resolve missing history before selecting new samples. This is candidate grouping, not proof of independent sources or split authorization.

- [x] Test transitive spatialchains, far sharedsources, seen propagation, duplicate/invalidrows, inputpermutation invariance, dateline proximity, missinghistory not silently dropped.
- [x] Observe missingfeature fail; implement cKDTree spherical candidatepairs with geodesiccheck and sourceunion; run scoped tests.

## Task 3: Execute public metadata-only audit and document limits

**Files:** research/trustmask/audit_catalog.py, docs/reports/progressive-data-audit.md, handoff.
- [x] Verify official tacoreader0.4.5wheelSHA and headerlayout from source only, do not install or import upstream reader.
- [x] Retrieve exact allowedranges, recordstatus/size/hash/offset; parse projectedmetadata using temporaryPyArrow; never output quality columns or read nestedassets.
- [x] Extract oldIDs only from already-published smalltext evidence; run grouping with sourceIDs frommetadata, report unresolvedLRscenes and historicalIDs explicitly.
- [x] Record groups remaining, group-size distribution, source/footprint limits and next statistically viable choices; do not invent freshdata independence or power.
- [x] Focusedtests/Ruff, oldprotocol gate, review and commit; no fullsuite.

## Checkpoint and remaining access validation

初轮源目录审计完成：8000成员，历史360全部匹配，整组排除523，剩7477成员/6158候选组。
加强后源网络复取遇TLS EOF两次，最终网络入口完整实取尚未通过；本地投影复算已通过。
- [ ] 最终受控网络入口成功获取全部三段并匹配已发布范围SHA；不退回旧重定向策略。
- [ ] LR场景和足迹元数据审计、样本量与正式角色冻结（下一依赖阶段）。

持久投影与回执、71项定向测试和范围限制见 `docs/reports/progressive-data-audit.md`。
