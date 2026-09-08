# Historical Phase 2B1-A metadata recovery

Git commits `a27e169` (acquisition times) and `06e577e` (pilot audit) establish
that all 8,000 members had LR/HR timestamps recorded. The authoritative digest is
in `artifacts/datasets/sen2naipv2-phase2b1a-audit-v1.json`:
`7487b0af2ebef86910e918d5d6b2fb927a6f5e46bac7c2e30be7ffb2ce994482`.
The full manifest was intentionally retained on cloud storage. Phase 2B1-B's
360-member post-manifest is not a substitute. Neither manifest schema includes
Sentinel-2 product IDs; timestamps alone do not certify source independence.

## CPU-only recovery on the server holding the persistent disk

Copy the locally authored `research/trustmask/recover_manifest.py` to the server.
Set `STORAGE_ROOT` to the existing mounted storage root; do not regenerate data.
Run with Python 3 on Linux, choosing a new output directory:

```bash
python3 recover_manifest.py \
  --manifest "$STORAGE_ROOT/trustsr/phase2b1a/manifests/7487b0af2ebef86910e918d5d6b2fb927a6f5e46bac7c2e30be7ffb2ce994482/samples.jsonl" \
  --output ./historical-metadata-export
```

This reads one bounded regular file, checks its exact digest before JSON parsing,
and exports only identity, acquisition times, geometry and day offsets. Numeric
tokens remain strings; correlation and asset statistics are not numerically
decoded or exported. No asset paths are followed. No GPU, model, image, prediction
cache, TACO download, or legacy recovery CLI is needed. Digest failures stop the
operation; never override the pinned digest or substitute a newly generated file.

Return only `metadata.jsonl` and `receipt.json` to an ignored local artifact
directory. Do not copy the full historical manifest into local Git. If the file is
missing, inspect the storage mount and historical run location before considering
another source route. No SSH host or storage-root value is assumed here.

## Required local follow-up after recovery

Verify the projection digest against the receipt and the pinned manifest digest.
Check all 8,000 identities/source indices and source identity; compare centroids
and geometries with the committed progressive member/geometry shards. Parse and
validate the three timestamps and signed day offsets; report whether top-level
time actually equals LR time across all members. Only then publish small audited
metadata evidence and decide whether acquisition timestamps support a bounded
catalog lookup. This export is not a completed semantic audit or an independence
certificate. Source grouping, split freeze and GPU experiments remain pending.

## Recovery result (2026-09-08)

The original persistent-disk file was recovered using the prepared standalone
script. The remote SHA check passed. Only the export and receipt were returned;
no image or prediction cache was opened. GPU computation was not used. The full
historical manifest remains remote, and the local projection is ignored by Git.

The offline audit passed all 8,000 identity/source-index/source/centroid/geometry
comparisons against the committed progressive shards. Signed UTC calendar-date
deltas were -1 for 2,145 records, 0 for 3,972, and +1 for 1,883. All top-level times
equal LR times. See `research/evidence/crosssensor-historical-manifest-audit-v1.json`
for digests, timestamp ranges and complete aggregate results.

All LR timestamps are 22:00Z (7,174) or 23:00Z (826); HR timestamps have the same
restricted hours (7,175 / 825). This concentration leaves the interpretation as
actual sensing instants unverified. Its cause has not been established. A future
bounded catalog probe must document its date-window assumptions and retain all
candidate products; it cannot treat exact-time matching or proximity as provenance.
The completed audit establishes consistency of the stored metadata, not S2 product
identity, complete composite membership, or independence of candidate groups.

Replay locally, with a fresh output path:

```bash
.venv/bin/python -m research.trustmask.audit_recovered_manifest \
  --export artifacts/progressive-historical-manifest \
  --output /tmp/historical-manifest-audit-replay.json
```

The verifier pins the recovered projection digest, checks the export receipt, and
records digests of the five reference shards used for comparison. The receipt's
original pending-validation field remains unchanged as a transport record; the
separate audit records the subsequent semantic checks.
