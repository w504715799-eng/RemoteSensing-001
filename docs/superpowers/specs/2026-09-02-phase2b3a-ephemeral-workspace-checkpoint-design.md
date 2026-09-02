# Phase 2B3-A Ephemeral Workspace Checkpoint Design

Date: 2026-09-02
Status: proposed for implementation
Parent workflow: `2026-09-01-phase2b3a-development-score-audit-design.md`

## 1. Purpose

Phase 2B3-A must run on `/root/rivermind-data`, whose ext4 inode capacity is suitable for prediction
and score caches but whose contents are not assumed to survive instance replacement. Long-term state
must be checkpointed to `/root/rivermind-fs`, whose JuiceFS bytes persist but whose 10,000-inode quota
is currently exhausted.

The workflow will therefore use:

- `/root/rivermind-data` as the disposable working storage root;
- `/root/rivermind-fs` for existing model bytes and a small number of verified archive files;
- GitHub as the durable source for code and Git history;
- deterministic, digest-bound workspace archives as the durable source for datasets, caches, logs,
  runtime evidence, and scientific results.

No scientific gate may be weakened to accommodate either filesystem.

## 2. Observed constraints

- `/root/rivermind-fs` has ample free bytes but zero free inodes. It cannot currently create even one
  checkpoint file.
- The exact Phase 2B1-B selection and Phase 2B2-A input audit are currently on
  `/root/rivermind-data`.
- The reviewed SEN2SRLite and LDSR model directories are currently on `/root/rivermind-fs`.
- The old detached clean checkout at `/root/rivermind-fs/trustsr-phase1b/repo` is reconstructible
  only if its exact commit and all required history are verified on GitHub first.
- Code need not be stored in a workspace checkpoint because the feature branch and exact reviewed
  commit are stored on GitHub.

## 3. Chosen architecture

### 3.1 Working layout

The session uses these runtime-only locations:

```text
/root/rivermind-data/
  repos/RemoteSensing001/            # clean attached GitHub checkout; not archived
  model-mounts/sen2srlite/           # read-only bind mount; not archived
  model-mounts/ldsr-s2/              # read-only bind mount; not archived
  trustsr/                            # datasets, audits, Phase 2B3 caches/results/logs; archived
```

`/root/rivermind-data` remains the `--storage-root`. The two existing model directories are bind
mounted beneath it and then remounted read-only. Bind targets are ordinary directories, not
symlinks, so the existing component-wise path and model validation remains effective. The archive
never traverses `model-mounts` because only the `trustsr` tree is included.

### 3.2 Persistent layout

After inode recovery, persistent checkpoints use:

```text
/root/rivermind-fs/trustsr-phase2b3a-checkpoints/
  phase2b3a-workspace-<stage>-<archive-sha256>.tar
  phase2b3a-workspace-<stage>-<archive-sha256>.json
```

There is no mutable `LATEST` pointer. A canonical manifest identifies one immutable archive by exact
basename, byte size, SHA-256, reviewed Git commit, completed stage, creation protocol version, and
the three allowlisted archive roots:

```text
trustsr/phase2b1b
trustsr/phase2b2a
trustsr/phase2b3a
```

The manifest also records the frozen selection-manifest and input-audit digests. Hostnames, SSH
targets, credentials, absolute checkout paths, GPU UUIDs, and secrets are prohibited.

## 4. One-time inode recovery

The first retry is read-only until all of these checks pass:

1. The current SSH target and both mounts are freshly verified.
2. The old checkout is a regular, non-symlink directory at the exact approved path.
3. Git status is clean and the checkout contains no untracked files.
4. Its exact detached commit is reachable from a named remote branch or tag on GitHub.
5. An inode count proves that removing only this checkout will free more than 1,024 inodes plus a
   20% safety margin.

Only after those checks may the exact old checkout directory be deleted. No model, dataset, audit,
cache, or evidence directory may be deleted. The operation is recoverable by cloning the verified
GitHub revision; the deletion and before/after inode counts must be reported.

If the commit is not remotely reachable, the checkout has local changes, or it cannot free enough
inodes, deletion stops. The fallback is an inode-quota increase from the storage provider; no other
existing file is repurposed or overwritten.

## 5. Checkpoint creation

Checkpoint creation is allowed only at a stage boundary with no Phase 2B3-A process running and no
active stage or publication lock.

1. Revalidate the exact working storage root, frozen input digests, reviewed Git commit, cache
   integrity, and completed stage evidence.
2. Build an uncompressed deterministic tar archive on `/root/rivermind-data`. GeoTIFF,
   SafeTensors, and model-derived tensors are already poorly compressible, so compression would add
   GPU-instance time without materially improving the storage contract.
3. Permit only regular files and directories beneath the three allowlisted relative roots. Reject
   symlinks, hard links, devices, FIFOs, sockets, absolute names, `..`, and unexpected roots.
4. Normalize tar ordering, numeric ownership, and timestamps so identical working content produces
   identical archive bytes.
5. Fsync the local archive, compute its SHA-256 and size, and construct the canonical JSON manifest.
6. Copy the archive to a same-directory `.part` file on `/root/rivermind-fs`, fsync it, verify the
   copied SHA and size, and atomically rename it to its digest-qualified immutable basename.
7. Publish the canonical manifest last with the same no-overwrite and fsync discipline.
8. Re-open and verify the persistent archive and manifest before declaring the checkpoint accepted.

Existing identical archive/manifest bytes are idempotent. Any existing different bytes, partial
entry, symlink, or unexpected file fails closed. A failed copy never causes the source working tree
to be removed.

## 6. Restore

Every new session starts from an explicit checkpoint manifest, never from “the newest file”.

1. Verify the persistent manifest's canonical schema and allowed basename.
2. Verify the archive size and SHA-256 before copying.
3. Copy the archive to a private staging directory on `/root/rivermind-data` and verify it again.
4. Inspect the complete tar member list before extraction using the same root/type/path allowlist.
5. Extract into an empty private staging tree, not into a live workspace.
6. Revalidate the exact frozen selection and input-audit digests, Phase 2B3 result markers, and cache
   metadata before atomically publishing the restored `trustsr` tree.
7. Clone/fetch code separately from GitHub and require a clean attached checkout at the reviewed
   commit recorded by the checkpoint.
8. Establish the two read-only model bind mounts and rerun the normal Phase 2B3-A preflight. A
   checkpoint never substitutes for mount, environment, model, repeatability, or resource gates.

Restore refuses a non-empty destination, unsafe tar member, missing required root, digest mismatch,
wrong Git commit, or partial checkpoint. It never merges uncertain old and restored caches.

## 7. Session shutdown protocol

The GPU/server may be paused only after one of these terminal states:

- the session made no durable change and all stages stopped before mutation; or
- a new checkpoint archive and manifest were both verified from `/root/rivermind-fs`; or
- a failure occurred and the user explicitly accepts losing the disposable working state.

For a successful A1 or A2 session, the order is:

1. finish and locally verify the scientific stage;
2. pull and commit the small allowlisted evidence to GitHub;
3. push the exact branch and verify the remote SHA;
4. checkpoint the full working `trustsr` tree;
5. verify the persistent checkpoint;
6. tell the user the GPU/server can be paused.

## 8. Implementation boundaries

Add two narrowly scoped, test-driven shell entry points:

```text
scripts/phase2b3a/checkpoint_workspace.sh
scripts/phase2b3a/restore_workspace.sh
```

They will use the existing Phase 2B3-A path, lock, collision, canonical-JSON, and hostile-input test
patterns. Tests use small synthetic trees and fake mounts; they never archive real data locally.
The cloud runbook will document one-time inode recovery, read-only model bind mounts, checkpoint
creation, restore, and shutdown gates. Existing scientific Python schemas and algorithms remain
unchanged.

## 9. Acceptance criteria

- One-time cleanup deletes only the verified clean old checkout and frees the required inode margin.
- No model or scientific data is deleted from persistent storage.
- Hostile archive members and path/symlink/collision races fail closed.
- Interrupted checkpoint/restore leaves no accepted partial destination.
- A synthetic checkpoint round trip is byte- and digest-identical.
- Existing identical checkpoints are idempotent; different collisions never overwrite.
- Code restoration comes from the exact GitHub commit, not the archive.
- Model bind mounts are read-only and remain outside the archive.
- The server is never declared safe to pause until persistent verification completes.
- Task 13 restarts from its read-only resource gate using the new reviewed A0 commit after the local
  implementation and acceptance suite are rerun and pushed.
