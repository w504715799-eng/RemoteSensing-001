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
