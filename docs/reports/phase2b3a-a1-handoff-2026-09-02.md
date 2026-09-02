# Phase 2B3-A A1 handoff — 2026-09-02

## Current state

- Branch: `feature/phase2b3a-score-audit-design`
- Reviewed compute commit: `4df5195e0a28701391c3951659a42409f81a11c2`
- Latest evidence commit: `01cbd28562345d2d03720052b40371cd0738ff78`
- The latest evidence commit is pushed to the same GitHub branch.
- Cloud state reached `A1_CHECKPOINTED`, which is a safe pause point.
- Do not start A2 until the user starts the instance and supplies a current SSH endpoint.
- Never write the SSH endpoint, port, host identity, GPU identity, or credentials into tracked files.

## Completed work

1. Recovered the durable JuiceFS inode budget by deleting only the obsolete
   `/root/rivermind-fs/trustsr-phase1b/conda-env` directory. The mount subsequently reported
   8,719 free inodes and accepted new files.
2. Verified the disposable ext4 work root and durable JuiceFS root, frozen input hashes, model
   sources, clean attached Git checkout, base Python 3.12 dependencies, CUDA visibility, and idle
   GPU state.
3. The cloud container denied `mount --bind` because it lacks mount capability. The two model
   trees were copied from their durable sources to the disposable work root, compared by entry
   type/mode and per-file SHA-256, and made non-writable. Model weights were not changed.
4. Fixed checkpoint publication for JuiceFS Trash semantics. Publishing now uses atomic
   `renameat2(RENAME_NOREPLACE)` instead of hard-linking a partial file and unlinking it. The old
   approach left a Trash hard link and caused `st_nlink == 2`. The regression test and the full
   checkpoint-related test suites passed before commit `4df5195e...` was pushed.
5. Re-ran the formal preflight at reviewed compute commit `4df5195e...` and independently verified
   the A0 checkpoint.
6. A1 results:
   - `single`: `repeatability_pass=true`
   - `smoke`: four development ROIs completed
   - `replay`: `byte_identical=true`
   - scientific decision: `include_ldsr_variance_k5=true`
7. Pulled the allowlisted A1 JSON bundle locally, verified it with `trustsr-phase2b3a-verify a1`,
   committed the three evidence files, and pushed evidence commit `01cbd285...`.
8. Created and independently reverified the A1 durable checkpoint. No compute process or stage
   lock remained when the cloud work stopped.

## Durable checkpoint identities

### A0

- Manifest: `phase2b3a-workspace-a0-58a729d33d7e9ff8b59bb519bda92bda8124a486603d2ec7c336e5649c05aaac.json`
- Archive SHA-256: `58a729d33d7e9ff8b59bb519bda92bda8124a486603d2ec7c336e5649c05aaac`
- Archive size: `437637120` bytes

### A1

- Manifest: `phase2b3a-workspace-a1-623535c33fee50e7d05b83386158b349c4056d1f4aa256efda1189933e9993f8.json`
- Archive SHA-256: `623535c33fee50e7d05b83386158b349c4056d1f4aa256efda1189933e9993f8`
- Archive size: `933263360` bytes

Both pairs are under `/root/rivermind-fs/trustsr-phase2b3a-checkpoints` and were verified as
one-link regular files.

## Committed A1 evidence

- `artifacts/phase2b3a/sen2naipv2-development-smoke-v1.json`
- `artifacts/phase2b3a/sen2naipv2-development-smoke-acceptance-v1.json`
- `artifacts/phase2b3a/sen2naipv2-development-smoke-cache-audit-v1.json`

The temporary pulled bundle was stored at `/tmp/trustsr-phase2b3a-a1-bundle-4df5195`; it is not a
durable input and is no longer required because its accepted evidence is committed.

## Important platform constraint before A2

The current `restore_workspace.sh` requires privileged read-only bind mounts. This cloud container
does not permit bind mounts. Do not silently replace this with symlinks: the reviewed path guards
reject symlink components.

Before starting A2, choose and review one of these two routes:

1. Preferred for this platform: add an explicit copy-mode restore that copies each durable model
   tree into the disposable work root, verifies a canonical entry inventory and per-file hashes,
   removes write permission, and records the restoration mode. Keep the durable source unchanged.
2. Use bind mounts only if the provider explicitly enables the required container capability.

If the same instance and data disk resume with the current A1 workspace intact, that saves transfer
time, but the formal A2 state transition still needs a reviewed restoration/preflight decision. Do
not bypass the runbook merely because the files remain present.

## Next-session sequence

1. Ask the user to start the instance and provide the current SSH endpoint. Remember that every
   powered-on minute is billed as GPU time, including CPU-only work.
2. Perform one concise read-only check of both mounts, free bytes/inodes, the reviewed compute
   checkout, exact A1 checkpoint, model sources/copies, GPU availability, and absence of foreign
   compute processes.
3. Resolve the bind-mount constraint using the reviewed route above.
4. Restore the exact A1 checkpoint, then run formal `preflight`.
5. Run `development`, followed by cache-only `development-replay`.
6. While the instance is still on, pull the A2 evidence and create/reverify the A2 checkpoint as
   soon as compute finishes. Then immediately tell the user to pause the instance.
7. Perform local A2 verification, commit only the three allowlisted A2 evidence files, and push.

## Cost rule

This provider bills the GPU whenever the remote instance is powered on, even during nominally
CPU-only commands. Batch all unavoidable remote operations, parallelize independent result pulling
and checkpointing when safe, and move review, JSON validation, documentation, and Git work local.
