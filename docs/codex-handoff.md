# Codex handoff: Phase 2B3-A saturation-v2 rerun

Date: 2026-09-03 (Asia/Shanghai)

## Repository checkpoint

- Integration branch: `main`; local Git is authoritative. Do not merge cloud-side code.
- Resume handoff checkpoint: `fcaf135749c934426f86ed2629cfb26ad6c53502`.
- Previously reviewed and pushed cross-commit repair:
  `68e13d99553a94ea0f875f0fdd03dcc352e854ec`.
- The final saturation-v2 reviewed integration commit is pending local integration. Do not invent or
  deploy a replacement SHA before the integration gates pass.
- The GPU rerun is paused. Ask the user to restart the GPU only after local integration, focused
  validation, final review, and push of the exact deployment commit are complete.

## Objective and scope

Rerun Phase 2B3-A with the explicit
`uint16_saturate_10000_divide_10000_v2` policy, first producing a fresh four-ROI A1 and then the
exact frozen 120-development-ROI A2. Preserve every ROI and every existing asset-integrity check.
This phase does not inspect calibration or `internal_test` pixels.

Only allowlisted, host-free JSON evidence may enter Git. Pixels, tensors, caches, logs, models,
checkpoint archives, endpoints, credentials, host paths, and host runtime details remain outside
the repository. Cloud checkouts and disposable compute state may be deleted and recreated from the
exact pushed local commit; no cloud branch or cloud-side modification is a merge source.

## Completed diagnosis

The exact A2 development attempt stopped while loading the ninth ROI, before model construction or
inference, because raw reflectance exceeded the former `[0,10000]` assumption. It produced no A2
result, runtime, or replay, and GPU processes returned to zero.

Host-free inspection of the frozen 120-ROI set established:

- Exactly one of 120 ROIs is affected; the other 119 have raw maxima no greater than `9572`.
- LR contains eight values above `10000`, ordered B04/B03/B02/B08 as `[4,0,0,4]`.
- HR contains 117 values above `10000`, ordered B04/B03/B02/B08 as `[56,0,0,61]`.
- The affected aligned-crop raw maximum is `11968`.
- Asset bytes, sidecars, shape, dtype, CRS, nodata, masks, and alignment remain internally
  consistent; this is a policy/data-contract mismatch, not evidence of corrupt input.

The former `uint16_divide_10000_no_clip_v1` assumption came from the smaller smoke sample and is
retained only for historical paths. The frozen metadata does not justify inventing an offset.

## Saturation-v2 decision

For each Phase 2B3-A aligned LR and HR crop:

1. Keep all existing byte, sidecar, geometry, dtype, nodata, and mask checks.
2. Reject the full raw input if any value exceeds `32767`.
3. Count values strictly above `10000`, in total and in B04/B03/B02/B08 order.
4. Saturate only the aligned crop at `10000` without mutating the source.
5. Convert to contiguous CPU float32 and divide by `10000.0`.

Each sample records `radiometric_saturation` for LR and HR. Each result records a derived
`radiometric_policy` aggregate, and prediction provenance records the normalization policy. Replay,
offline bundle verification, and current checkpoint construction fail closed on policy or aggregate
mismatch.

Existing tracked Phase 2B2-A evidence and the accepted A1 v1 publications remain byte-for-byte
historical. They must not be overwritten, relabelled, or represented as v2 evidence.

## Schema contract

Fresh A1 uses:

- result `trustsr.phase2b3a-development-smoke.v2`;
- cache audit `trustsr.phase2b3a-development-smoke-cache-audit.v2`;
- acceptance `trustsr.phase2b3a-development-smoke-acceptance.v2`;
- runtime `trustsr.phase2b3a-a1-runtime.v2`;
- replay `trustsr.phase2b3a-a1-replay.v2`;
- bundle manifest `trustsr.phase2b3a-bundle-manifest.v2`.

A2 remains pre-publication v1 while requiring the v2 policy and radiometric evidence:

- result `trustsr.phase2b3a-development-score-audit.v1`;
- cache audit `trustsr.phase2b3a-development-score-cache-audit.v1`;
- acceptance `trustsr.phase2b3a-development-score-acceptance.v1`;
- runtime `trustsr.phase2b3a-a2-runtime.v1`;
- replay `trustsr.phase2b3a-a2-replay.v1`;
- bundle manifest `trustsr.phase2b3a-bundle-manifest.v1`.

Both result and cache audit require top-level `normalization_policy`. Result samples require exact
LR/HR saturation statistics, and result/runtime require identical `radiometric_policy` objects.
A2 keeps v1 because no A2 evidence has yet been published, not because it permits legacy input.

## Immutable recovery identity and checkpoint asymmetry

The accepted historical A1 recovery identity is exact:

- Manifest:
  `phase2b3a-workspace-a1-623535c33fee50e7d05b83386158b349c4056d1f4aa256efda1189933e9993f8.json`.
- Archive digest: the 64-hex value embedded in that basename.
- Archive size: `933263360` bytes.
- Producer commit: `4df5195e0a28701391c3951659a42409f81a11c2`.

The manifest/archive pair was independently verified before the pause and remains immutable. It may
be restored only under its exact accepted identity and producer lineage, and only to recover frozen
data and verified models. After recovery, the disposable live `trustsr/phase2b3a` must be reset and
A1 must be rebuilt from scratch under v2.

New A1 and A2 checkpoint builds require the current
`uint16_saturate_10000_divide_10000_v2` runtime policy. The exact accepted historical A1 is the sole
legacy restore exception; it cannot be re-checkpointed, accepted as current evidence, or used to
resume A2 directly. A pause after the new A1 checkpoint resumes from that exact new v2 checkpoint.

## Required cloud sequence

Follow [the cloud runbook](phase2b3a-cloud-runbook.md) using runtime-only operator values:

1. Confirm the user has restarted the GPU, then verify mount identities, free bytes and inodes,
   idle GPU, canonical non-symlink base Python, the exact immutable A1 pair, and model inventories.
2. Recreate a clean attached cloud checkout at the exact final reviewed and pushed local commit.
   Require exact HEAD and a clean status; never merge cloud code.
3. Restore the accepted legacy A1 only for data/model recovery, with the explicit historical
   producer and verified copy-mode models.
4. Run `scripts/phase2b3a/reset_live_phase2b3a.sh WORKSPACE_ROOT`. It may delete and recreate only
   disposable live `trustsr/phase2b3a`; it must never target the durable checkpoint or model roots.
5. Run formal preflight, then fresh v2 `single`, `smoke`, and inference-free `replay`.
6. Checkpoint the new A1 and independently verify its exact manifest. Before A2 replaces the live
   bundle manifest, pull the A1-v2 bundle with the existing pull script and verify it offline. Write
   only these three new tracked paths, enabled by the integration `.gitignore` allowlist:
   - `artifacts/phase2b3a/sen2naipv2-development-smoke-v2.json`;
   - `artifacts/phase2b3a/sen2naipv2-development-smoke-cache-audit-v2.json`;
   - `artifacts/phase2b3a/sen2naipv2-development-smoke-acceptance-v2.json`.
7. Run exact A2 `development`, inference-free `development-replay`, checkpoint A2, and independently
   verify its exact manifest.
8. Before the A2 pull, require Git porcelain status to contain exactly the three untracked A1-v2
   files above and no other tracked or untracked change. Pull the A2-v1 bundle into a different new
   local destination, reverify both bundles, and write exactly:
   - `artifacts/phase2b3a/sen2naipv2-development-score-audit-v1.json`;
   - `artifacts/phase2b3a/sen2naipv2-development-score-cache-audit-v1.json`;
   - `artifacts/phase2b3a/sen2naipv2-development-score-acceptance-v1.json`.
9. Require an exact six-file status and staged-filename allowlist. Preserve the old A1 v1
   publications unchanged, review the diff, commit, push, and prove remote and local SHAs match.

No stale instruction to restore the historical A1 and immediately run `development` is valid. The
guarded reset, fresh v2 A1, checkpoint verification, and A1-v2 offline bundle verification are hard
prerequisites for A2.

## Local integration status and gates

The versioned loader, Phase 2B3-A policy selection, cache identity, sample evidence, and aggregate
evidence have focused unit coverage on their implementation branches. Offline-verifier,
checkpoint-boundary, guarded-reset, pull-script, and documentation changes are being integrated in
separate worktrees. Treat none of them as deployable until the coordinator has inspected and
cherry-picked their commits in dependency order.

To keep the delayed cycle moving, use focused tests for each isolated change and one integrated
local acceptance pass before deployment. At minimum, the integrated gate must cover the changed
loader, online stage, replay, offline verifier, checkpoint, reset, and pull boundaries; Ruff, shell
syntax, CLI help, `git diff --check`, a clean status, and a sensitive-string scan must also pass.
An independent final review must resolve every Critical or Important finding. Record the final
reviewed commit only after these gates pass.

## Stop conditions and remaining risks

- The GPU is paused until the user restarts it; connectivity failure before that is expected.
- Never bypass exact commit ancestry, canonical JSON, digest, model inventory, mount, inode,
  capacity, GPU-idleness, lock, path, or output-collision checks.
- Never broaden the guarded reset target or alter the immutable accepted A1 checkpoint pair.
- Model copy mode consumes disposable capacity; recheck bytes and inodes before restore and compute.
- Values above `32767`, malformed or inconsistent radiometric evidence, a legacy current runtime,
  or any policy mismatch are hard failures.
- Do not start A2 unless fresh v2 A1 replay, checkpoint verification, and offline bundle verification
  have all succeeded.
- Do not inspect or evaluate `internal_test` data in Phase 2B3-A.
- Do not commit cloud payloads or operator connection details.
