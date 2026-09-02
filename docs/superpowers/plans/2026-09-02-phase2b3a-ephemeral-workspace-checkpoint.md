# Phase 2B3-A Ephemeral Workspace Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-closed, deterministic checkpoint and restore workflow that lets Phase 2B3-A compute on disposable `/root/rivermind-data` while preserving scientific state as immutable verified files on `/root/rivermind-fs`.

**Architecture:** A dependency-light Python module owns archive membership, canonical manifests, deterministic tar bytes, digest verification, atomic publication, and staged restore. Two narrow Bash entry points own cloud boundary checks: mounted roots, clean reviewed Git state, stage reservations, one checkpoint reservation, and read-only model bind mounts. The existing Phase 2B3-A runner and scientific schemas remain unchanged; the cloud runbook composes the new commands around A1 and A2.

**Tech Stack:** Python 3.12 standard library (`dataclasses`, `hashlib`, `json`, `os`, `pathlib`, `shutil`, `stat`, `tarfile`, `tempfile`), Bash 5, Linux `mount`/`findmnt`, pytest 8, Ruff, Git/GitHub.

**Spec:** `docs/superpowers/specs/2026-09-02-phase2b3a-ephemeral-workspace-checkpoint-design.md`

## Global Constraints

- `/root/rivermind-data` is disposable working storage; `/root/rivermind-fs` is durable storage with a 10,000-inode quota.
- GitHub, not a workspace archive, is the durable source for code and Git history.
- Existing SEN2SRLite and LDSR-S2 model directories remain on `/root/rivermind-fs`; they are mounted read-only beneath the working root and never archived.
- Archive membership is exactly `trustsr/phase2b1b`, `trustsr/phase2b2a`, and `trustsr/phase2b3a`; all three roots must exist.
- Archives are uncompressed deterministic tar files; identical content, completed stage, and reviewed commit must produce identical bytes.
- Only regular files and directories are accepted. Symlinks, hard links, devices, FIFOs, sockets, absolute paths, `..`, and unexpected roots fail closed.
- Frozen selection-manifest SHA-256 is `c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a`.
- Frozen input-audit SHA-256 is `fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b`.
- Durable names are `phase2b3a-workspace-<stage>-<archive-sha256>.tar` and the matching `.json`; there is no mutable latest pointer.
- Existing identical durable bytes are idempotent; any different collision, symlink, or partial entry fails without replacement.
- Restore uses an explicit manifest basename, an empty live destination, and a private staging directory. It never merges trees or caches.
- The server is safe to pause only after no durable mutation occurred or the new persistent archive and manifest have both been reverified.
- No task may install dependencies, create a remote Conda environment, weaken a scientific gate, or archive real cloud data on the local CPU machine.

---

## File and responsibility map

- Create `src/trustsr/artifacts/workspace_checkpoint.py`: the sole checkpoint format authority. It validates source trees and manifests, emits deterministic tar bytes, publishes immutable pairs, validates stage evidence, and restores through private staging.
- Create `tests/artifacts/test_workspace_checkpoint.py`: small synthetic-tree tests for deterministic bytes, hostile inputs, canonical schemas, collisions, interrupted publication, and staged restore.
- Create `scripts/phase2b3a/checkpoint_workspace.sh`: validates the two mountpoints, reviewed checkout, frozen inputs, inactive stage reservations, and then invokes the Python build/publish operations under one reservation.
- Create `scripts/phase2b3a/restore_workspace.sh`: validates the explicit persistent pair and reviewed checkout, establishes verified read-only model bind mounts, and invokes staged restore into an empty working destination.
- Create `tests/scripts/test_phase2b3a_checkpoint_scripts.py`: fake-mount executable-boundary tests. These tests create only tiny temporary files and never require root or a real mount.
- Modify `docs/phase2b3a-cloud-runbook.md`: replace the single-persistent-root assumption with transient work, durable checkpoint, restore, and shutdown sequences.
- Modify `README.md`: link the checkpoint runbook and state that large data remains cloud-only.
- Modify `.superpowers/sdd/2026-09-01-phase2b3a-development-score-audit/progress.md`: record the storage sub-plan and the new A0 handoff SHA after acceptance.

### Task 1: Deterministic checkpoint format and source-tree safety

**Files:**
- Create: `src/trustsr/artifacts/workspace_checkpoint.py`
- Create: `tests/artifacts/test_workspace_checkpoint.py`

**Interfaces:**
- Consumes: `trustsr.jsonio.canonical_json(value: Any) -> bytes`.
- Produces: `CheckpointError`, `CheckpointManifest`, `BuiltCheckpoint`, `build_checkpoint(workspace_root: Path, output_directory: Path, *, completed_stage: str, reviewed_commit: str) -> BuiltCheckpoint`, `load_manifest(path: Path) -> CheckpointManifest`, and `verify_checkpoint(archive_path: Path, manifest_path: Path) -> CheckpointManifest`.
- Constants: `SCHEMA = "trustsr.phase2b3a-workspace-checkpoint.v1"`, `PROTOCOL_VERSION = 1`, `ARCHIVE_ROOTS = ("trustsr/phase2b1b", "trustsr/phase2b2a", "trustsr/phase2b3a")`, `COMPLETED_STAGES = frozenset({"a0", "a1", "a2"})`.

- [ ] **Step 1: Write the manifest and deterministic-round-trip tests**

Create a fixture with all three roots, the exact frozen input files, and hand-written payloads. Build twice after changing source mtimes and creation order, then assert identical bytes and exact members:

```python
def test_build_checkpoint_is_deterministic_and_has_exact_members(tmp_path: Path) -> None:
    workspace = _valid_workspace(tmp_path / "workspace")
    first = build_checkpoint(
        workspace,
        tmp_path / "first",
        completed_stage="a0",
        reviewed_commit="a" * 40,
    )
    os.utime(
        workspace / "trustsr/phase2b1b/selections" / POST_SHA256 / "samples.jsonl",
        ns=(1_800_000_000_000_000_000, 1_800_000_000_000_000_000),
    )
    second = build_checkpoint(
        workspace,
        tmp_path / "second",
        completed_stage="a0",
        reviewed_commit="a" * 40,
    )

    assert first.archive_path.read_bytes() == second.archive_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.manifest.archive_sha256 == hashlib.sha256(
        first.archive_path.read_bytes()
    ).hexdigest()
    with tarfile.open(first.archive_path, mode="r:") as archive:
        assert [member.name for member in archive.getmembers()] == EXPECTED_MEMBER_NAMES
        assert all(member.uid == member.gid == member.mtime == 0 for member in archive.getmembers())
```

Assert the canonical manifest has exactly these keys and no host path:

```python
assert json.loads(first.manifest_path.read_bytes()) == {
    "archive_basename": first.archive_path.name,
    "archive_sha256": first.manifest.archive_sha256,
    "archive_size_bytes": first.archive_path.stat().st_size,
    "completed_stage": "a0",
    "input_audit_sha256": INPUT_AUDIT_SHA256,
    "protocol_version": 1,
    "reviewed_commit": "a" * 40,
    "roots": list(ARCHIVE_ROOTS),
    "schema": SCHEMA,
    "selection_manifest_sha256": POST_SHA256,
}
assert first.manifest_path.read_bytes() == canonical_json(
    json.loads(first.manifest_path.read_bytes())
)
```

- [ ] **Step 2: Run the focused tests and confirm the missing module failure**

Run: `uv run pytest -q tests/artifacts/test_workspace_checkpoint.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'trustsr.artifacts.workspace_checkpoint'`.

- [ ] **Step 3: Implement the frozen types and strict manifest parser**

Use frozen dataclasses and exact-key validation. Do not accept booleans where integers are expected, abbreviated hashes, uppercase hashes, extra keys, noncanonical JSON, or a basename that does not bind its stage and archive digest:

```python
@dataclass(frozen=True)
class CheckpointManifest:
    schema: str
    protocol_version: int
    completed_stage: str
    reviewed_commit: str
    roots: tuple[str, ...]
    selection_manifest_sha256: str
    input_audit_sha256: str
    archive_basename: str
    archive_size_bytes: int
    archive_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "archive_basename": self.archive_basename,
            "archive_sha256": self.archive_sha256,
            "archive_size_bytes": self.archive_size_bytes,
            "completed_stage": self.completed_stage,
            "input_audit_sha256": self.input_audit_sha256,
            "protocol_version": self.protocol_version,
            "reviewed_commit": self.reviewed_commit,
            "roots": list(self.roots),
            "schema": self.schema,
            "selection_manifest_sha256": self.selection_manifest_sha256,
        }
```

`load_manifest()` must read no more than 1 MiB, compare raw bytes with `canonical_json(value)`, require the exact ten-key set, and require:

```python
expected_basename = (
    f"phase2b3a-workspace-{manifest.completed_stage}-{manifest.archive_sha256}.tar"
)
if manifest.archive_basename != expected_basename:
    raise CheckpointError("archive basename is not digest-bound")
```

- [ ] **Step 4: Implement race-resistant source scanning and deterministic tar creation**

Walk each root with `os.scandir()` in bytewise UTF-8 name order. For every entry use `follow_symlinks=False`; accept directories and regular files only, reject regular files with `st_nlink != 1`, and reject any name whose POSIX representation is absolute, contains `..`, or falls outside `ARCHIVE_ROOTS`.

For each file, open with `os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)`, compare `os.fstat(fd)` with the earlier `lstat` device/inode/size, stream it to `tarfile.TarFile.addfile()`, then compare size and `st_mtime_ns` again. Normalize every header:

```python
info.uid = 0
info.gid = 0
info.uname = ""
info.gname = ""
info.mtime = 0
info.mode = 0o755 if info.isdir() else 0o644
info.pax_headers = {}
```

Open with `tarfile.open(..., mode="w:", format=tarfile.USTAR_FORMAT, dereference=False)`. Fsync the completed local archive, derive its SHA-256/size, rename it to the digest-qualified basename with no replacement, write the matching canonical JSON through an exclusive temporary file plus fsync, and fsync the output directory.

- [ ] **Step 5: Add hostile-source and corrupt-manifest cases**

Parameterize explicit failures for a missing archive root, symlink file, symlink directory, FIFO, Unix socket, source hard link, uppercase commit, changed frozen input bytes, noncanonical manifest, extra manifest key, wrong size, wrong digest, wrong stage, and an archive basename containing `/` or `..`.

Representative hard-link assertion:

```python
def test_build_checkpoint_rejects_source_hard_link(tmp_path: Path) -> None:
    workspace = _valid_workspace(tmp_path / "workspace")
    source = workspace / "trustsr/phase2b3a/cache.bin"
    source.write_bytes(b"cache")
    os.link(source, workspace / "trustsr/phase2b3a/cache-copy.bin")
    with pytest.raises(CheckpointError, match="hard link"):
        build_checkpoint(
            workspace,
            tmp_path / "out",
            completed_stage="a0",
            reviewed_commit="a" * 40,
        )
```

- [ ] **Step 6: Run Task 1 tests and lint**

Run:

```bash
uv run pytest -q tests/artifacts/test_workspace_checkpoint.py
uv run ruff check src/trustsr/artifacts/workspace_checkpoint.py tests/artifacts/test_workspace_checkpoint.py
```

Expected: all focused tests pass and Ruff prints `All checks passed!`.

- [ ] **Step 7: Commit Task 1**

```bash
git add src/trustsr/artifacts/workspace_checkpoint.py tests/artifacts/test_workspace_checkpoint.py
git commit -m "feat: add deterministic phase2b3a checkpoints"
```

### Task 2: Immutable publication and safe staged restore

**Files:**
- Modify: `src/trustsr/artifacts/workspace_checkpoint.py`
- Modify: `tests/artifacts/test_workspace_checkpoint.py`

**Interfaces:**
- Consumes: Task 1 `CheckpointManifest`, `BuiltCheckpoint`, `load_manifest()`, and `verify_checkpoint()`.
- Produces: `publish_checkpoint(built: BuiltCheckpoint, persistent_directory: Path) -> tuple[Path, Path]`, `restore_checkpoint(persistent_directory: Path, manifest_basename: str, workspace_root: Path, *, expected_reviewed_commit: str) -> Path`, and module CLI subcommands `build`, `publish`, `verify`, and `restore`.
- CLI output: one canonical JSON line containing only basenames, digests, sizes, stage, and status; no hostname, credential, or absolute path.

- [ ] **Step 1: Write publication tests before implementation**

Cover successful archive-first/manifest-last publication, idempotent identical publication, different-byte collision, symlink collision, stale `.part` collision, an unrelated unexpected directory entry, absence of any mutable latest pointer, and a simulated interrupted copy where no accepted manifest exists.

```python
def test_publish_checkpoint_is_idempotent_but_never_overwrites(tmp_path: Path) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    first = publish_checkpoint(built, persistent)
    second = publish_checkpoint(built, persistent)
    assert first == second
    assert first[0].read_bytes() == built.archive_path.read_bytes()
    first[1].write_bytes(b"different")
    with pytest.raises(CheckpointError, match="collision"):
        publish_checkpoint(built, persistent)
    assert first[1].read_bytes() == b"different"
```

Use an injected `copy_file: Callable[[BinaryIO, BinaryIO], None]` defaulting to `shutil.copyfileobj` so a test can raise after one write and prove that only a hidden `.part` exists and no manifest was accepted.

- [ ] **Step 2: Run the publication tests and confirm the missing symbol failure**

Run: `uv run pytest -q tests/artifacts/test_workspace_checkpoint.py -k publish`

Expected: tests fail because `publish_checkpoint` is not defined.

- [ ] **Step 3: Implement immutable cross-filesystem publication**

Require the persistent directory to be a real, non-symlink directory. Create archive and manifest partials with `os.open(..., O_CREAT | O_EXCL | O_NOFOLLOW, 0o600)`. Copy and fsync the archive partial, verify its size/digest, then use `os.link(part, final, follow_symlinks=False)` as the no-replace publication operation and unlink the partial. Publish the manifest only after reopening and verifying the final archive. Fsync the directory after each final link.

If a final exists, accept it only when `lstat` says one-link regular file and its bytes match exactly. Never use `os.replace()` for an immutable final name.

- [ ] **Step 4: Write full-member archive inspection and restore tests**

Add tests that independently create malicious tar files containing an absolute member, parent traversal, symlink, hard-link record, FIFO, unexpected root, duplicate member name, missing required root, file/child conflict, and oversized manifest declaration. Also prove a valid restore:

```python
def test_restore_checkpoint_publishes_only_after_full_validation(tmp_path: Path) -> None:
    built = _build_valid_checkpoint(tmp_path)
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    _, manifest_path = publish_checkpoint(built, persistent)
    live = tmp_path / "new-session"
    live.mkdir()
    restored = restore_checkpoint(
        persistent,
        manifest_path.name,
        live,
        expected_reviewed_commit="a" * 40,
    )
    assert restored == live / "trustsr"
    assert _tree_file_digests(restored) == _tree_file_digests(
        tmp_path / "workspace/trustsr"
    )
    assert not any(path.name.startswith(".phase2b3a-restore.") for path in live.iterdir())
```

Assert restore rejects a non-empty or symlink `workspace_root/trustsr`, a mismatched reviewed commit, and any invalid archive before publishing `trustsr`.

- [ ] **Step 5: Implement verification, evidence binding, and staged extraction**

`verify_checkpoint()` must compare manifest basename, archive basename, size, SHA-256, and the full member inventory before extraction. Inventory validation uses `PurePosixPath`, rejects repeated names, requires directory parents before children, and requires each of the three root directory records. `restore_checkpoint()` must first copy the already verified persistent archive into its private work-root staging directory, fsync it, and independently reverify size, SHA-256, and full membership from that copied file before extraction.

Before live publication, validate the restored frozen files at these exact relative paths:

```python
SELECTION_RELATIVE = Path(
    "trustsr/phase2b1b/selections/"
    f"{SELECTION_MANIFEST_SHA256}/samples.jsonl"
)
INPUT_AUDIT_RELATIVE = Path(
    "trustsr/phase2b2a/input-audits/"
    f"{SELECTION_MANIFEST_SHA256}/phase2b2a-input-audit.json"
)
```

For `a1` and `a2`, require the canonical bundle manifest beneath `trustsr/phase2b3a/results/<selection-sha>/`, require its declared phase to equal the checkpoint stage, require exactly the four expected evidence basenames, and verify each declared size and SHA-256. This binds the completed result, cache-audit metadata, runtime evidence, and byte-identical replay marker before publication.

Put this logic in one `_validate_workspace_evidence(trustsr_root: Path, completed_stage: str) -> None` helper. Call it both from `build_checkpoint()` before opening the output archive and from `restore_checkpoint()` against the extracted staging tree, so neither a source checkpoint nor a restored checkpoint can claim an unproven completed stage.

Extract regular files with exclusive `O_CREAT | O_EXCL | O_NOFOLLOW`; create directories with mode `0o700` during staging; fsync files and staging directories. After all checks, change data file modes to `0o600` and directories to `0o700`, then publish `staging/trustsr` with a Linux `renameat2(..., RENAME_NOREPLACE)` wrapper implemented through `ctypes.CDLL(None, use_errno=True)`. Treat `EEXIST` as a collision and treat `ENOSYS` as an unsupported fail-closed platform; never fall back to replacement-capable `os.rename()`. On any exception, remove only the private staging directory.

- [ ] **Step 6: Add the dependency-light module CLI**

Implement `argparse` subcommands with exact signatures:

```text
python -m trustsr.artifacts.workspace_checkpoint build \
  WORKSPACE_ROOT LOCAL_SCRATCH COMPLETED_STAGE REVIEWED_COMMIT
python -m trustsr.artifacts.workspace_checkpoint publish \
  LOCAL_SCRATCH MANIFEST_BASENAME PERSISTENT_DIRECTORY
python -m trustsr.artifacts.workspace_checkpoint verify \
  PERSISTENT_DIRECTORY MANIFEST_BASENAME
python -m trustsr.artifacts.workspace_checkpoint restore \
  PERSISTENT_DIRECTORY MANIFEST_BASENAME WORKSPACE_ROOT EXPECTED_REVIEWED_COMMIT
```

Every command returns exit 2 for a contract error and prints a single safe canonical JSON line on success. The success payload is:

```python
{
    "archive_basename": manifest.archive_basename,
    "archive_sha256": manifest.archive_sha256,
    "archive_size_bytes": manifest.archive_size_bytes,
    "completed_stage": manifest.completed_stage,
    "manifest_basename": manifest.archive_basename.removesuffix(".tar") + ".json",
    "reviewed_commit": manifest.reviewed_commit,
    "status": command_name,
}
```

- [ ] **Step 7: Run Task 2 tests and lint**

Run:

```bash
uv run pytest -q tests/artifacts/test_workspace_checkpoint.py
uv run ruff check src/trustsr/artifacts/workspace_checkpoint.py tests/artifacts/test_workspace_checkpoint.py
```

Expected: all checkpoint tests pass and Ruff prints `All checks passed!`.

- [ ] **Step 8: Commit Task 2**

```bash
git add src/trustsr/artifacts/workspace_checkpoint.py tests/artifacts/test_workspace_checkpoint.py
git commit -m "feat: publish and restore phase2b3a checkpoints"
```

### Task 3: Cloud checkpoint entry point

**Files:**
- Create: `scripts/phase2b3a/checkpoint_workspace.sh`
- Create: `tests/scripts/test_phase2b3a_checkpoint_scripts.py`

**Interfaces:**
- Consumes: Task 2 `build`, `publish`, and `verify` module commands.
- Produces: `checkpoint_workspace.sh BASE_PYTHON WORKSPACE_ROOT PERSISTENT_ROOT REPOSITORY COMPLETED_STAGE REVIEWED_COMMIT`.
- Success: emits exactly the final `verify` canonical JSON line. Contract failures return 2.

- [ ] **Step 1: Write executable-boundary tests with fake mounts and Git**

Create fake `mountpoint`, `df`, and `git` executables using the established pattern in `tests/scripts/test_phase2b3a_scripts.py`. Use the real current Python through a small executable launcher and a synthetic three-root workspace. For `a1`/`a2`, the fixture must write the exact phase bundle manifest and four digest-bound evidence files required by Task 2. Define `_invoke_checkpoint(tmp_path: Path, *, stage: str = "a0", **boundary_state: object) -> tuple[subprocess.CompletedProcess[str], Path, Path]`, returning the completed process, persistent directory, and prohibited-command marker.

The success test is:

```python
@pytest.mark.parametrize("stage", ["a0", "a1", "a2"])
def test_checkpoint_script_builds_publishes_and_reverifies(
    stage: str, tmp_path: Path
) -> None:
    completed, persistent, prohibited = _invoke_checkpoint(tmp_path, stage=stage)
    assert completed.returncode == 0, completed.stderr
    record = json.loads(completed.stdout)
    assert record["completed_stage"] == stage
    assert record["status"] == "verify"
    checkpoint = persistent / "trustsr-phase2b3a-checkpoints"
    assert len(list(checkpoint.glob("*.tar"))) == 1
    assert len(list(checkpoint.glob("*.json"))) == 1
    assert not list(checkpoint.glob("*.part"))
    assert not (checkpoint / ".checkpoint.lock").exists()
    assert not prohibited.exists()
```

Add named failure tests for: unmounted work root; unmounted persistent root; equal roots; a symlink component in every positional path; dirty, detached, and wrong Git `HEAD`; an active `*.jsonl.lock` stage reservation; a pre-existing `.checkpoint.lock`; either frozen digest mismatch; insufficient work bytes; insufficient persistent bytes after build; fewer than four persistent inodes; and attempted `conda`, `pip`, `curl`, or `wget`. Every case must assert exit 2, no final `.json`, and no prohibited command call.

For the success case, assert that the persistent directory contains exactly one `.tar` and one `.json`, the JSON line reports `status == "verify"`, and the script leaves no `.part`, scratch, or reservation entry.

- [ ] **Step 2: Run the script tests and confirm the missing-file failure**

Run: `uv run pytest -q tests/scripts/test_phase2b3a_checkpoint_scripts.py -k checkpoint_script`

Expected: tests fail because `scripts/phase2b3a/checkpoint_workspace.sh` does not exist.

- [ ] **Step 3: Implement strict argument, path, mount, and Git validation**

Start with `set -euo pipefail`, seven positional arguments, the same normalized absolute-path and component-wise symlink rejection used by `run_cloud.sh`, and exact `a0|a1|a2`/lowercase-hash validation.

Require:

```bash
mountpoint -q -- "$workspace_root"
mountpoint -q -- "$persistent_root"
[[ "$workspace_root" != "$persistent_root" ]]
[[ "$repository" == "$workspace_root"/* ]]
[[ "$base_python" != "$repository"/* && "$base_python" != "$workspace_root"/* ]]
[[ "$(git -C "$repository" rev-parse HEAD)" == "$reviewed_commit" ]]
[[ -n "$(git -C "$repository" symbolic-ref --short HEAD)" ]]
[[ -z "$(git -C "$repository" status --porcelain)" ]]
```

Require at least 10 GiB free on the work root. After `build` returns the canonical archive size, require the persistent root to have at least that many free bytes plus 1 GiB and at least four free inodes before invoking `publish`.

- [ ] **Step 4: Implement stage-boundary and checkpoint reservations**

Reject if `trustsr/phase2b3a/logs` contains any `*.jsonl.lock` directory. Atomically reserve one durable inode with:

```bash
checkpoint_directory="$persistent_root/trustsr-phase2b3a-checkpoints"
mkdir -p -- "$checkpoint_directory"
reservation="$checkpoint_directory/.checkpoint.lock"
mkdir -- "$reservation" 2>/dev/null || die 'checkpoint publication is already reserved'
trap 'rmdir -- "$reservation" 2>/dev/null || true' EXIT
```

Create scratch only under the work root with `mktemp -d`, run `build` with `PYTHONPATH="$repository/src"`, parse its exact seven-key canonical JSON, perform the persistent byte/inode gate, invoke `publish`, and finally run `verify` against the same manifest basename. Require all three JSON records to agree on basenames, digest, size, stage, and reviewed commit; reject any absolute path or extra output. Remove scratch and reservation only after verification succeeds.

- [ ] **Step 5: Run syntax, focused tests, and existing runner regressions**

Run:

```bash
bash -n scripts/phase2b3a/checkpoint_workspace.sh
uv run pytest -q tests/scripts/test_phase2b3a_checkpoint_scripts.py -k checkpoint_script
uv run pytest -q tests/scripts/test_phase2b3a_scripts.py
```

Expected: syntax succeeds; new tests pass; the existing Phase 2B3-A script suite remains green.

- [ ] **Step 6: Commit Task 3**

```bash
git add scripts/phase2b3a/checkpoint_workspace.sh tests/scripts/test_phase2b3a_checkpoint_scripts.py
git commit -m "feat: add phase2b3a checkpoint entry point"
```

### Task 4: Cloud restore and read-only model mounts

**Files:**
- Create: `scripts/phase2b3a/restore_workspace.sh`
- Modify: `tests/scripts/test_phase2b3a_checkpoint_scripts.py`

**Interfaces:**
- Consumes: Task 2 `verify` and `restore` module commands.
- Produces: `restore_workspace.sh BASE_PYTHON WORKSPACE_ROOT PERSISTENT_ROOT REPOSITORY MANIFEST_BASENAME SEN2SRLITE_SOURCE LDSR_SOURCE`.
- Publishes: `WORKSPACE_ROOT/trustsr` and read-only bind targets `WORKSPACE_ROOT/model-mounts/sen2srlite` and `WORKSPACE_ROOT/model-mounts/ldsr-s2`.

- [ ] **Step 1: Write restore-script and fake-mount tests**

Extend the fake command directory with `mount`, `umount`, and `findmnt` executables that record argv as JSON lines. Write exact cases for successful explicit restore, unsafe manifest basename, mismatched Git commit, non-empty live `trustsr`, missing model source, source outside the durable root, symlink model source/target, bind failure cleanup, remount-read-only failure cleanup, and reported mount options without `ro`.

Success assertions:

```python
assert mount_calls == [
    ["--bind", str(sen2_source), str(work / "model-mounts/sen2srlite")],
    ["-o", "remount,bind,ro", str(work / "model-mounts/sen2srlite")],
    ["--bind", str(ldsr_source), str(work / "model-mounts/ldsr-s2")],
    ["-o", "remount,bind,ro", str(work / "model-mounts/ldsr-s2")],
]
assert (work / "trustsr/phase2b1b").is_dir()
assert (work / "trustsr/phase2b2a").is_dir()
assert (work / "trustsr/phase2b3a").is_dir()
assert json.loads(completed.stdout)["status"] == "restore"
```

- [ ] **Step 2: Run restore tests and confirm the missing-file failure**

Run: `uv run pytest -q tests/scripts/test_phase2b3a_checkpoint_scripts.py -k restore_script`

Expected: tests fail because `scripts/phase2b3a/restore_workspace.sh` does not exist.

- [ ] **Step 3: Implement explicit-pair and exact-checkout validation**

Reuse the strict absolute-path and symlink-component checks. Restrict `MANIFEST_BASENAME` with:

```bash
[[ "$manifest_basename" =~ ^phase2b3a-workspace-(a0|a1|a2)-[0-9a-f]{64}\.json$ ]] ||
  die 'checkpoint manifest basename'
```

Require both roots mounted and distinct, repository beneath the work root, clean attached Git `HEAD`, model sources as real directories beneath the persistent root, and an absent `WORKSPACE_ROOT/trustsr`. Run module `verify`, parse the `reviewed_commit` from its exact seven-key canonical output, and compare it with full `git rev-parse HEAD` before any mount or extraction.

- [ ] **Step 4: Implement read-only bind mount transaction**

Create the two empty targets with mode `0700`. Mount each source with `mount --bind`, immediately remount it using `mount -o remount,bind,ro`, then require `findmnt -n -o OPTIONS --target "$target"` to contain a comma-delimited `ro` option. Track mounted targets in an array and unmount them in reverse order on any failure before live restore publication.

Do not compare GPU model names or hardware models. The contract is exact source path, ordinary directory, durable-root containment, bind identity, and read-only options.

- [ ] **Step 5: Invoke staged restore only after mounts and Git pass**

Run:

```bash
PYTHONPATH="$repository/src" "$base_python" -m trustsr.artifacts.workspace_checkpoint restore \
  "$persistent_root/trustsr-phase2b3a-checkpoints" \
  "$manifest_basename" \
  "$workspace_root" \
  "$git_head"
```

On success, clear the cleanup trap without unmounting: the read-only model mounts are the intended live runtime state. Emit only the module's canonical success JSON. The operator must next run the ordinary Phase 2B3-A preflight; restore never marks a scientific stage accepted.

- [ ] **Step 6: Run syntax and all checkpoint boundary tests**

Run:

```bash
bash -n scripts/phase2b3a/checkpoint_workspace.sh scripts/phase2b3a/restore_workspace.sh
uv run pytest -q tests/scripts/test_phase2b3a_checkpoint_scripts.py
uv run pytest -q tests/scripts/test_phase2b3a_scripts.py
```

Expected: both scripts pass Bash syntax; all new and existing Phase 2B3-A boundary tests pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add scripts/phase2b3a/restore_workspace.sh tests/scripts/test_phase2b3a_checkpoint_scripts.py
git commit -m "feat: restore phase2b3a cloud workspaces"
```

### Task 5: Runbook integration and local A0 acceptance

**Files:**
- Modify: `docs/phase2b3a-cloud-runbook.md`
- Modify: `README.md`
- Modify: `.superpowers/sdd/2026-09-01-phase2b3a-development-score-audit/progress.md`

**Interfaces:**
- Consumes: Task 3 checkpoint command and Task 4 restore command.
- Produces: one operator sequence for first-session inode recovery, A0 checkout, model mounts, A1/A2 checkpointing, later restore, and safe pause.

- [ ] **Step 1: Add the two-root runtime variables and session state machine to the runbook**

Document these runtime-only variables without real hostnames, ports, credentials, or GPU identity:

```bash
: "${PHASE2B3A_WORK_ROOT:?set disposable mounted work root}"
: "${PHASE2B3A_PERSISTENT_ROOT:?set durable mounted checkpoint/model root}"
: "${PHASE2B3A_REPOSITORY:?set clean attached checkout below work root}"
: "${PHASE2B3A_BASE_PYTHON:?set existing cloud base Python}"
: "${PHASE2B3A_REVIEWED_COMMIT:?set exact reviewed and pushed 40-hex SHA}"
: "${PHASE2B3A_SEN2SRLITE_SOURCE:?set durable SEN2SRLite model directory}"
: "${PHASE2B3A_LDSR_SOURCE:?set durable LDSR-S2 model directory}"
PHASE2B3A_STORAGE_ROOT="$PHASE2B3A_WORK_ROOT"
PHASE2B3A_SEN2SRLITE_DIR="$PHASE2B3A_WORK_ROOT/model-mounts/sen2srlite"
PHASE2B3A_LDSR_DIR="$PHASE2B3A_WORK_ROOT/model-mounts/ldsr-s2"
```

State the allowed session transitions exactly:

```text
UNVERIFIED -> RESTORED -> PREFLIGHT_OK -> A1_OK -> A1_CHECKPOINTED
A1_CHECKPOINTED -> RESTORED -> PREFLIGHT_OK -> A2_OK -> A2_CHECKPOINTED
```

Only `UNVERIFIED` with no mutation, `A1_CHECKPOINTED`, and `A2_CHECKPOINTED` are safe pause points.

- [ ] **Step 2: Document the one-time inode recovery without embedding a destructive wildcard**

Use the exact old checkout path `/root/rivermind-fs/trustsr-phase1b/repo`. Require `realpath -e`, non-symlink directory checks, `git status --porcelain=v1 --untracked-files=all`, full `HEAD`, `git fetch --prune --tags origin`, and a non-empty named result from `git branch -r --contains "$old_head"` or `git tag --contains "$old_head"`.

Measure old checkout inodes with `find "$old_repo" -xdev -printf . | wc -c`; require the count to be at least `ceil(1024 * 1.20) = 1229`. Record `df -Pi` before and after. The runbook's deletion line must name only the validated variable whose exact value was asserted:

```bash
[[ "$old_repo" == /root/rivermind-fs/trustsr-phase1b/repo ]]
[[ "$(realpath -e -- "$old_repo")" == "$old_repo" ]]
rm -rf --one-file-system -- "$old_repo"
```

Explain that this removes only a clean GitHub-reachable checkout, is recoverable by cloning the verified revision, and never targets a model, dataset, audit, cache, or evidence directory. Stop if any prerequisite fails.

- [ ] **Step 3: Document first checkpoint, later restore, and post-stage shutdown order**

First session: create an empty ordinary `trustsr/phase2b3a`, clone/fetch exact A0 code under the work root, establish model binds with the restore script only when restoring an existing checkpoint, or use the same four explicit bind commands for the initial uncheckpointed input tree, run normal preflight, and create the `a0` baseline checkpoint before GPU compute.

After A1/A2: verify the scientific stage, pull/verify/commit the small evidence, push Git and verify remote SHA, then run:

```bash
scripts/phase2b3a/checkpoint_workspace.sh \
  "$PHASE2B3A_BASE_PYTHON" \
  "$PHASE2B3A_WORK_ROOT" \
  "$PHASE2B3A_PERSISTENT_ROOT" \
  "$PHASE2B3A_REPOSITORY" \
  a1 \
  "$PHASE2B3A_REVIEWED_COMMIT"
```

Use `a2` after A2. On a new session require an explicitly copied manifest basename and run `restore_workspace.sh`; never glob for the newest file.

- [ ] **Step 4: Update README and SDD ledger**

Link `docs/phase2b3a-cloud-runbook.md` from the Phase 2B3 section and state: cloud pixels, tensors, caches, and checkpoint tar files are never downloaded locally or committed; only allowlisted JSON evidence enters Git.

In the SDD ledger, add a storage-recovery sub-plan row with this plan/spec path, Tasks 1–5 status, and the reviewed A0 SHA field. Do not mark parent Task 13 complete; record it as ready to restart only after the remote baseline checkpoint succeeds.

- [ ] **Step 5: Run documentation leakage and command checks**

Run:

```bash
sensitive_pattern='g''hp_[A-Za-z0-9]+|gpu''home\.cc|ssh root''@|GPU''-[0-9A-Fa-f-]+'
rg -n "$sensitive_pattern" \
  README.md docs scripts src tests .superpowers
bash -n scripts/phase2b3a/*.sh
uv run pytest -q tests/artifacts/test_workspace_checkpoint.py \
  tests/scripts/test_phase2b3a_checkpoint_scripts.py \
  tests/scripts/test_phase2b3a_scripts.py
```

Expected: the leakage search has no matches, Bash syntax succeeds, and all focused tests pass.

- [ ] **Step 6: Commit Task 5**

```bash
git add README.md docs/phase2b3a-cloud-runbook.md \
  .superpowers/sdd/2026-09-01-phase2b3a-development-score-audit/progress.md
git commit -m "docs: integrate phase2b3a cloud checkpoints"
```

- [ ] **Step 7: Run the complete A0 acceptance gate**

Run:

```bash
uv run pytest -q
uv run ruff check .
bash -n scripts/phase2b3a/*.sh
uv run trustsr-phase2b3a --help >/dev/null
PYTHONPATH=src python -m trustsr.artifacts.workspace_checkpoint --help >/dev/null
git diff --check
git status --short --branch
```

Expected: the complete suite passes, Ruff and Bash syntax pass, both help commands exit 0, `git diff --check` is silent, and only the expected branch tracking status remains.

- [ ] **Step 8: Review and push the new immutable A0**

Use `superpowers:requesting-code-review`, resolve every verified issue with TDD, rerun Step 7, then:

```bash
git push origin feature/phase2b3a-score-audit-design
local_sha="$(git rev-parse HEAD)"
remote_sha="$(git ls-remote --heads origin feature/phase2b3a-score-audit-design | awk '{print $1}')"
test "$local_sha" = "$remote_sha"
```

Expected: local and remote full SHAs match. That reviewed code SHA is the A0 pin. Record it in the SDD ledger in a later documentation-only evidence commit; do not attempt to make a commit contain its own SHA. On the cloud, create an attached local branch at the recorded A0 pin even if the remote feature branch has subsequently advanced with evidence documentation.

### Task 6: One-time cloud migration, baseline checkpoint, and parent Task 13 handoff

**Files:**
- Modify only if factual evidence changes: `.superpowers/sdd/2026-09-01-phase2b3a-development-score-audit/task-13-report.md`
- Modify: `.superpowers/sdd/2026-09-01-phase2b3a-development-score-audit/progress.md`
- No cloud archive, model, cache, dataset, log, or SSH material enters Git.

**Interfaces:**
- Consumes: the exact pushed A0 from Task 5, user-provided current SSH endpoint, the two cloud mount roots, and the existing frozen inputs/models.
- Produces: verified inode recovery, one immutable `a0` checkpoint pair, a clean attached A0 checkout below `/root/rivermind-data`, read-only model mounts, and a read-only Phase 2B3-A preflight result ready for the parent plan's Task 13 A1 sequence.

- [ ] **Step 1: Obtain only the current SSH endpoint when the instance is running**

Tell the user GPU compute is not yet needed, but the cloud instance must be reachable for storage migration. Use public-key SSH only. Do not request, echo, persist, or commit a password.

- [ ] **Step 2: Re-run the read-only resource gate before deletion**

Verify both exact mountpoints, filesystem types, free bytes/inodes, frozen input paths/digests, model directory identities, and the old checkout. Capture command output into the local task report with hostnames, ports, usernames, GPU UUIDs, and credentials removed.

Require all one-time cleanup predicates from Task 5, including at least 1,229 inodes beneath only `/root/rivermind-fs/trustsr-phase1b/repo` and a named GitHub branch/tag containing its full detached commit.

- [ ] **Step 3: Delete only the verified old checkout and prove the inode delta**

Execute the exact runbook assertion and deletion. Immediately record before/after free inode counts and require:

```text
after_free_inodes - before_free_inodes >= 1229
after_free_inodes >= 1229
```

If either assertion fails, stop without deleting anything else and report that the storage provider must raise the inode quota. If it succeeds, report that the removed checkout is recoverable from its verified GitHub revision.

- [ ] **Step 4: Check out the reviewed A0 on disposable storage and prepare runtime mounts**

Clone or fetch the repository into `/root/rivermind-data/repos/RemoteSensing001`, check out the pushed feature branch at the exact reviewed A0, require clean attached `HEAD`, and use the existing cloud base Python directly.

Create the two ordinary bind targets and establish/remount the persistent model sources read-only using the runbook commands. Verify `findmnt` source identity and `ro` options. Do not compare or restrict the current GPU model name.

- [ ] **Step 5: Run preflight, then build and verify the baseline `a0` checkpoint**

Ensure `trustsr/phase2b3a` starts as an empty ordinary directory, rerun exact frozen input digests, and run the existing `run_cloud.sh ... preflight` with work-root input paths, read-only model mount targets, and exact reviewed A0. After preflight exits and releases its stage reservation, invoke `checkpoint_workspace.sh` with completed stage `a0` and the reviewed SHA. Independently invoke the module `verify` command against the explicit resulting manifest basename and record archive basename, byte size, SHA-256, reviewed commit, and stage in the redacted report.

Expected persistent membership: one immutable `.tar`, one matching `.json`, and no `.part` or `.checkpoint.lock` entry from this transaction.

- [ ] **Step 6: Resume the parent plan from the accepted preflight**

Update the ledger to mark the storage sub-plan complete and parent Task 13 ready for `single -> smoke -> replay` under the same accepted preflight and A0 pin.

If the instance is still funded and healthy, continue directly into the parent Task 13 A1 sequence to avoid restart cost. Otherwise, because the verified `a0` checkpoint exists and no stage mutation is in flight, tell the user the server can be paused.

- [ ] **Step 7: Commit and push the redacted migration evidence**

Run the leakage search from Task 5, inspect the exact diff, then:

```bash
git add .superpowers/sdd/2026-09-01-phase2b3a-development-score-audit/progress.md \
  .superpowers/sdd/2026-09-01-phase2b3a-development-score-audit/task-13-report.md
git commit -m "docs: record phase2b3a checkpoint migration"
git push origin feature/phase2b3a-score-audit-design
```

Expected: only redacted textual evidence enters Git. The full cloud `trustsr` tree remains represented only by the verified persistent checkpoint pair.
