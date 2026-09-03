# Codex handoff: Phase 2B3-A A2 resume

Date: 2026-09-03 (Asia/Shanghai)

## Repository checkpoint

- Current branch before this handoff commit: `main`.
- Reviewed implementation HEAD: `68e13d99553a94ea0f875f0fdd03dcc352e854ec`.
- `main` and `origin/main` both pointed to that exact implementation commit before this handoff
  document was created.
- The working tree and index were clean before this file was added. There were no pre-existing user
  modifications or diffs to include.
- The temporary branch `fix/phase2b3a-a2-cross-commit-resume` was fast-forwarded into `main`, then
  deleted together with its owned temporary worktree. Historical feature worktrees were preserved.

## Task objective

Continue Phase 2B3-A from the accepted A1 cloud checkpoint and execute the exact 120-ROI A2
development score audit on a rented remote GPU, while keeping local code review and evidence
management separate from cloud computation. The workflow must preserve publication-grade
provenance: immutable A1 evidence remains tied to the code that produced it, A2 is tied to the
newer reviewed code, only allowlisted host-free JSON evidence enters Git, and no `internal_test`
evaluation is performed during this phase.

The immediate repair objective was to make a valid cross-commit resume possible. A1 was produced
at `4df5195e0a28701391c3951659a42409f81a11c2`, while the reviewed A2 implementation advanced
beyond it. The former implementation incorrectly required both stages to use the same commit and
also restored fixed preflight output names that collided with the required resumed preflight.

## Completed work

### Branch consolidation and evidence preservation

- Consolidated the completed Phase branches into `main`; the reviewed main line was first unified at
  `530486a6b465bfc729f9984acfc7375cd6a7a655`.
- Preserved the tracked A1 publication evidence without modification:
  - `artifacts/phase2b3a/sen2naipv2-development-smoke-v1.json`
  - `artifacts/phase2b3a/sen2naipv2-development-smoke-acceptance-v1.json`
  - `artifacts/phase2b3a/sen2naipv2-development-smoke-cache-audit-v1.json`
- No A2 scientific result exists yet, and no A2 GPU inference has run.

### Cross-commit A2 resume repair

Implementation commit `68e13d99553a94ea0f875f0fdd03dcc352e854ec` is pushed to
`origin/main` and includes the following:

- A1 runtime evidence remains bound to its original producer commit.
- A2 accepts that producer only when it is a canonical lowercase 40-hex Git commit and
  `git merge-base --is-ancestor A1_PRODUCER CURRENT_A2_COMMIT` succeeds in the reviewed checkout.
- Noncanonical, unavailable, future, sibling, or otherwise unrelated producer commits fail closed.
- A2 runtime records the current reviewed commit in `git_commit` and separately records the old
  producer in `a1_producer_commit`.
- Inference-free A2 replay rechecks both the current checkout identity and A1-to-A2 ancestry.
- The offline A2 bundle verifier requires and validates the new producer field. This v1 runtime
  field was finalized before any A2 evidence bundle existed, so there is no published A2 migration.
- A0/A1 checkpoint build now requires a nonempty regular preflight log plus a canonical preflight
  runtime. The runtime producer must exactly equal the checkpoint reviewed commit.
- Staged checkpoint restoration revalidates those preflight files before publishing the live
  `trustsr` tree, including checkpoints created by older code.
- After a valid A0/A1 restore record, the restore wrapper moves fixed preflight outputs to
  stage-and-producer names before the new preflight runs. The move preserves the original bytes.
- Missing, partial, symlinked, ambiguous, multiple-runtime, or archive-collision states fail closed.
- If the verified post-restore archive step fails, the newly restored workspace and model
  publication are rolled back. If the restore program emits an unverifiable/malformed success
  record, the published state is retained for forensic safety, matching the pre-existing contract.
- The cloud runbook now requires the canonical non-symlink base Python. On the current image this is
  `/opt/conda/bin/python3.12`; `/opt/conda/bin/python` is a rejected symlink.

### Files changed by the repair

- `docs/phase2b3a-cloud-runbook.md`
- `scripts/phase2b3a/restore_workspace.sh`
- `src/trustsr/artifacts/workspace_checkpoint.py`
- `src/trustsr/cli/phase2b3a.py`
- `src/trustsr/cli/phase2b3a_verify.py`
- `tests/artifacts/test_workspace_checkpoint.py`
- `tests/cli/test_phase2b3a.py`
- `tests/cli/test_phase2b3a_verify.py`
- `tests/scripts/test_phase2b3a_checkpoint_scripts.py`

## Key decisions

1. Treat A1 producer identity and A2 execution identity as separate provenance fields. Rewriting the
   old A1 runtime to the new commit would falsify experimental provenance and is prohibited.
2. Permit cross-commit reuse only along verified Git ancestry. A digest-shaped string alone is not
   sufficient evidence of compatibility.
3. Keep A2 scientific result and cache-audit code revisions bound to the current reviewed checkout;
   bind the A1 dependency through its replay digest and explicit producer commit.
4. Preserve restored fixed-name preflight evidence by same-filesystem rename rather than overwrite,
   deletion, or silent reuse.
5. Enforce required resumable preflight evidence in the Python checkpoint authority before live
   publication, with an additional Shell boundary for archive ambiguity and collisions.
6. Keep GPU computation paused until local repair, review, merge, push, and full validation finish.
7. Never store the user-provided SSH endpoint, port, credentials, or host-identifying runtime data in
   tracked files. This document intentionally omits them; obtain the endpoint from operator context.

## Validation evidence

- Followed a red-green regression cycle for the original failures:
  - valid ancestor A1 producer was initially rejected;
  - nonancestor rejection lacked the required lineage check;
  - A2 runtime lacked explicit A1 producer provenance;
  - restored preflight files initially remained at collision paths;
  - missing/partial preflight checkpoints and post-restore archive rollback initially failed the new
    fail-closed tests.
- The focused checkpoint, restore-wrapper, online-stage, replay, and offline-verifier test files pass.
- The full repository test suite passed on the repair branch and passed again after fast-forwarding
  the exact commit into `main`.
- `uv run ruff check .` passed.
- `bash -n scripts/phase2b3a/*.sh` passed.
- `git diff --check` passed.
- The full suite emitted only the existing 43 PyTorch JIT deprecation warnings.
- A direct real-Git check confirmed that the A1 producer
  `4df5195e0a28701391c3951659a42409f81a11c2` is an ancestor of the previously reviewed A2 base
  `530486a6b465bfc729f9984acfc7375cd6a7a655`; the final repair commit descends from that base.
- An independent read-only code review initially found two restore-boundary gaps. Both were fixed and
  retested. The second review reported no Critical, Important, or Minor findings and marked the
  implementation ready to merge.
- A sensitive-string scan found no remote endpoint, SSH account, or port in the changed files.

## Remote compute state at pause

The user paused the GPU instance. Do not assume it is reachable until the user explicitly reports
that it has been restarted.

Durable state previously verified on the cloud host:

- Persistent Phase 2B3-A checkpoint directory:
  `/root/rivermind-fs/trustsr-phase2b3a-checkpoints`.
- Exact accepted A1 manifest:
  `phase2b3a-workspace-a1-623535c33fee50e7d05b83386158b349c4056d1f4aa256efda1189933e9993f8.json`.
- Its archive has the same digest in its basename, size `933263360` bytes, and was independently
  verified before the pause.
- A1 checkpoint producer commit:
  `4df5195e0a28701391c3951659a42409f81a11c2`.
- Durable model sources:
  - `/root/rivermind-fs/trustsr-phase1b/models/sen2srlite`
  - `/root/rivermind-fs/trustsr-phase1b/models/ldsr-s2`
- Disposable prior A1 state was preserved by rename as:
  - `/root/rivermind-data/trustsr-a1-preserved`
  - `/root/rivermind-data/model-mounts-a1-preserved`
- A newer disposable checkout was created at
  `/root/rivermind-data/repos/RemoteSensing001-530486a` for the first A2-resume attempt.
- That attempt restored the immutable A1 checkpoint in verified copy mode and completed a new
  preflight at `530486a...`, but `development` stopped before prediction because the old equality
  check rejected the valid A1 producer commit.
- No A2 inference, prediction cache generation, result commit, or GPU compute process started.
- During diagnosis, the restored A1 preflight log/runtime were manually preserved under alternate
  names to get past fixed-name collisions. Treat that disposable tree as diagnostic state, not as
  the next authoritative resume root. The immutable persistent A1 checkpoint remains authoritative.

## Remaining work

1. Ask the user to restart the same GPU instance; do not attempt instance control.
2. Reconnect using operator-provided runtime values without writing the endpoint into files or logs.
3. Confirm the paused host still exposes the expected disposable and persistent mount identities,
   free space/inodes, idle GPU, exact immutable A1 checkpoint, and durable model inventories.
4. Create or update a clean attached cloud checkout to exact pushed commit
   `68e13d99553a94ea0f875f0fdd03dcc352e854ec`. Require a clean status and exact HEAD.
5. Do not reuse or merge the partially diagnosed disposable `trustsr` tree. Preserve it by guarded
   rename if still needed, or choose a new empty disposable workspace root. Never alter the durable
   A1 checkpoint pair.
6. Follow `docs/phase2b3a-cloud-runbook.md` exactly with the real non-symlink base Python and
   `copy` model restore mode:
   - bootstrap and mount/storage checks;
   - restore the exact A1 manifest with explicit A1 checkpoint commit;
   - verify the restore JSON contains the old checkpoint commit and new restore-code commit;
   - assert the old preflight files were archived under stage-and-producer names;
   - run formal `preflight`;
   - run `development` only after preflight succeeds;
   - run inference-free `development-replay`;
   - checkpoint and independently reverify completed stage `a2`.
7. Pull only the allowlisted A2 bundle locally, run the offline verifier, and commit only the three
   final host-free A2 publication JSON files listed by the runbook.
8. Update the progress/publication assessment only after the exact A2 evidence is verified locally.

## Known risks and stop conditions

- The GPU instance is paused; connectivity failure is an expected external blocker, not a reason to
  alter local code or checkpoint data.
- The current disposable diagnostic tree contains manual preflight renames plus a newer preflight.
  Reusing it could mix state from different restore attempts. Start from the immutable A1 checkpoint
  in an empty destination.
- The A1 archive is large and model copy mode consumes disposable capacity. Recheck free bytes and
  inodes before restoration and before A2 execution.
- Never use the symlinked base Python path; the restore boundary will reject it.
- Never bypass a Git ancestry, exact-commit, canonical JSON, digest, mount, inode, free-space, GPU
  idleness, stage-lock, or output-collision failure. Stop and diagnose instead.
- A2 may be costly. Do not start it unless formal preflight succeeds and A1 acceptance/replay plus
  resource gates validate under the repaired code.
- Do not evaluate or inspect `internal_test` data in Phase 2B3-A.
- Do not commit cloud caches, pixels, tensors, model files, logs, checkpoints, endpoints, credentials,
  or host-identifying runtime manifests.

## Suggested independent workflows

1. **Cloud restore and A2 compute**: host checks, clean checkout, exact A1 restore, preflight, A2,
   replay, immutable A2 checkpoint.
2. **Local evidence intake**: allowlisted pull, bundle digest verification, offline verifier, and the
   three final publication JSON commits. This starts only after the cloud workflow produces A2.
3. **Paper-readiness analysis**: convert verified A1/A2 metrics into tables, statistical claims,
   limitations, and reproducibility text without touching compute state.
4. **Infrastructure hygiene**: after final A2 evidence is durable and locally verified, separately
   inventory diagnostic disposable trees and historical worktrees before any approved cleanup.
