# Phase 2B0 SEN2NAIPv2 Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, offline provenance audit for the pinned SEN2NAIPv2 source without downloading any real pixel data locally.

**Architecture:** A strict standard-library loader converts one checked-in JSON inventory into immutable dataclasses. A thin CLI renders a canonical audit summary and has no downloader or networking dependency. Repository-policy tests enforce that the local checkout contains metadata only; geographic splitting and real TACO access remain a separate Phase 2B1 project.

**Tech Stack:** Python 3.12, standard-library `dataclasses`/`json`/`pathlib`/`hashlib`, pytest, Ruff, uv.

**Spec:** `docs/superpowers/specs/2026-08-27-phase2b0-sen2naipv2-provenance.md`

## Global Constraints

- Do not install `tacoreader` or change the PyTorch dependency set.
- Do not run Git LFS smudge, `git lfs pull`, Hugging Face download commands, remote range reads, or any command that downloads real SEN2NAIPv2 pixels locally.
- Local files are limited to source, documentation, metadata/LFS pointer text below 1 MiB, and synthetic test fixtures.
- Real TACO objects, decoded samples, and model caches belong only on user-designated cloud storage in a separately approved Phase 2B1.
- Never hard-code a GPU model, SSH endpoint, credential, or cloud storage root.
- OpenSR-Test final test regions must not be used for calibration or threshold selection.
- All application and test commands run through `uv run`.

---

### Task 1: Strict offline provenance types and loader

**Files:**
- Create: `src/trustsr/data/provenance.py`
- Create: `tests/data/test_provenance.py`

**Interfaces:**
- Consumes: a local `pathlib.Path` containing schema `trustsr.sen2naipv2-source.v1` JSON.
- Produces: `LfsObject`, `DatasetSource`, and `load_dataset_source(path: Path) -> DatasetSource`.

- [ ] **Step 1: Write failing happy-path and offline tests**

Create `tests/data/test_provenance.py` with a small `source_payload()` helper and tests that monkeypatch `socket.create_connection` and `urllib.request.urlopen` to raise, write JSON through `tmp_path`, load it, and assert:

```python
source.repository == "tacofoundation/SEN2NAIPv2"
source.revision == "c370504201072fdb1dd388013ab8c0fc7d00a57e"
source.bands == ("B04", "B03", "B02", "B08")
source.lr_shape == (130, 130)
source.hr_shape == (520, 520)
source.total_bytes == 12
source.objects[0].path == "sen2naipv2-crosssensor.taco"
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
uv run pytest tests/data/test_provenance.py -v
```

Expected: collection fails because `trustsr.data.provenance` does not exist.

- [ ] **Step 3: Implement immutable types and exact JSON decoding**

Create `src/trustsr/data/provenance.py` with frozen dataclasses, an `object_pairs_hook` that rejects duplicate JSON keys, and explicit key-set checks. Use these signatures:

```python
@dataclass(frozen=True)
class LfsObject:
    path: str
    sha256: str
    size_bytes: int

@dataclass(frozen=True)
class DatasetSource:
    schema: str
    repository: str
    revision: str
    license_claim: str
    card_sha256: str
    bands: tuple[str, ...]
    scale: int
    lr_shape: tuple[int, int]
    hr_shape: tuple[int, int]
    declared_total_bytes: int
    objects: tuple[LfsObject, ...]

    @property
    def total_bytes(self) -> int:
        return sum(item.size_bytes for item in self.objects)

def load_dataset_source(path: Path) -> DatasetSource:
    ...
```

The loader must use only `Path.read_text(encoding="utf-8")` and `json.loads`; it must not import HTTP, Hugging Face, Git, LFS, TACO, or raster libraries.

- [ ] **Step 4: Write failing validation tests**

Parameterize mutations that must each raise `ValueError` with a specific message fragment:

```python
(
    ("schema", "wrong"),
    ("revision", "ABC"),
    ("card_sha256", "xyz"),
    ("bands", ["B02", "B03", "B04", "B08"]),
    ("scale", 2),
    ("declared_total_bytes", 99),
)
```

Add separate cases for unknown/missing/duplicate JSON keys, duplicate object paths, absolute paths, `..` path components, invalid object SHA-256, Boolean sizes, zero sizes, empty objects, and HR shape not equal to LR shape times scale.

- [ ] **Step 5: Implement all validation and run focused/full checks**

Validate exact lowercase hexadecimal lengths with `re.fullmatch`, reject `bool` where integer values are required, and raise before constructing the final dataclass.

Run:

```bash
uv run pytest tests/data/test_provenance.py -v
uv run pytest
uv run ruff check .
```

Expected: all commands pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/trustsr/data/provenance.py tests/data/test_provenance.py
git commit -m "feat: add offline dataset provenance contract"
```

---

### Task 2: Pinned SEN2NAIPv2 source inventory

**Files:**
- Create: `artifacts/datasets/sen2naipv2-source-v1.json`
- Create: `tests/data/test_sen2naipv2_inventory.py`

**Interfaces:**
- Consumes: `load_dataset_source()` from Task 1 and the exact upstream facts frozen in the spec.
- Produces: one reviewable source-of-truth inventory for Phase 2B1.

- [ ] **Step 1: Write a failing exact-inventory test**

Create `tests/data/test_sen2naipv2_inventory.py`. Load the repository-relative artifact and assert the exact repository, revision, license claim, card SHA-256, band order, shapes, scale, object count, object path order, each object SHA-256/size pair, and:

```python
assert source.total_bytes == 149_356_128_592
assert source.declared_total_bytes == source.total_bytes
```

The expected nine tuples must be copied verbatim from Section 2 of the spec, not derived from the artifact under test.

- [ ] **Step 2: Run the focused test and verify the missing artifact failure**

Run:

```bash
uv run pytest tests/data/test_sen2naipv2_inventory.py -v
```

Expected: FAIL because `artifacts/datasets/sen2naipv2-source-v1.json` does not exist.

- [ ] **Step 3: Add the canonical inventory JSON**

Create the artifact with these top-level keys in this order:

```json
{
  "schema": "trustsr.sen2naipv2-source.v1",
  "repository": "tacofoundation/SEN2NAIPv2",
  "revision": "c370504201072fdb1dd388013ab8c0fc7d00a57e",
  "license_claim": "cc0-1.0",
  "card_sha256": "5897aed9410fef305953ff5b34e83697b466901583b880158af2902a8267a58d",
  "bands": ["B04", "B03", "B02", "B08"],
  "scale": 4,
  "lr_shape": [130, 130],
  "hr_shape": [520, 520],
  "declared_total_bytes": 149356128592,
  "objects": []
}
```

Populate `objects` with all nine exact `path`, `sha256`, and `size_bytes` entries from the spec, sorted lexicographically by path. Do not include URLs, absolute paths, tokens, timestamps, or image content.

- [ ] **Step 4: Verify the inventory and commit**

Run:

```bash
uv run pytest tests/data/test_sen2naipv2_inventory.py -v
uv run pytest
uv run ruff check .
git ls-files artifacts/datasets | xargs -r stat --printf='%s %n\n'
```

Expected: all tests and lint pass; the JSON is far below 1 MiB.

Commit:

```bash
git add artifacts/datasets/sen2naipv2-source-v1.json tests/data/test_sen2naipv2_inventory.py
git commit -m "data: pin sen2naipv2 source inventory"
```

---

### Task 3: Deterministic metadata-only audit CLI

**Files:**
- Create: `src/trustsr/cli/dataset_audit.py`
- Create: `tests/cli/test_dataset_audit.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: `load_dataset_source(path: Path) -> DatasetSource`.
- Produces: `build_payload(source: DatasetSource) -> dict[str, object]` and the `trustsr-dataset-audit` console script.

- [ ] **Step 1: Write failing payload and CLI tests**

Create tests that load the pinned inventory and assert:

```python
payload["schema"] == "trustsr.dataset-audit.v1"
payload["metadata_only"] is True
payload["network_accessed"] is False
payload["pixel_data_downloaded"] is False
payload["local_real_pixel_policy"] == "forbidden"
payload["ready_for_phase2b1_schema_probe"] is True
payload["object_count"] == 9
payload["total_bytes"] == 149_356_128_592
payload["variant_counts"] == {"crosssensor": 1, "histmatch": 4, "unet": 4}
```

Invoke `main()` twice with monkeypatched `sys.argv`, capture stdout, and assert byte-identical single-line strict JSON plus a trailing newline. Monkeypatch network entry points to raise so accidental access fails the test.

- [ ] **Step 2: Run tests and verify the missing module failure**

Run:

```bash
uv run pytest tests/cli/test_dataset_audit.py -v
```

Expected: collection fails because `trustsr.cli.dataset_audit` does not exist.

- [ ] **Step 3: Implement the CLI without downloader options**

Use only `argparse`, `json`, and `pathlib`. Accept exactly one option:

```python
parser.add_argument(
    "--source",
    type=Path,
    default=Path("artifacts/datasets/sen2naipv2-source-v1.json"),
)
```

Classify variants from the fixed filename prefixes, sort all mapping keys via
`json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)`, and print once. Do not add URL, network, token, SSH, output-directory, or download flags.

- [ ] **Step 4: Register the console script and update the lock deterministically**

Add to `[project.scripts]`:

```toml
trustsr-dataset-audit = "trustsr.cli.dataset_audit:main"
```

Run `uv lock` only to refresh the editable project metadata; no new dependency may appear.

- [ ] **Step 5: Verify focused tests, repeatability, and all checks**

Run:

```bash
uv run pytest tests/cli/test_dataset_audit.py -v
uv run trustsr-dataset-audit > /tmp/trustsr-phase2b0-audit-1.json
uv run trustsr-dataset-audit > /tmp/trustsr-phase2b0-audit-2.json
cmp /tmp/trustsr-phase2b0-audit-1.json /tmp/trustsr-phase2b0-audit-2.json
uv run pytest
uv run ruff check .
```

Expected: all commands pass and `cmp` is silent.

- [ ] **Step 6: Commit Task 3**

```bash
git add pyproject.toml uv.lock src/trustsr/cli/dataset_audit.py tests/cli/test_dataset_audit.py
git commit -m "feat: add deterministic dataset provenance audit"
```

---

### Task 4: Local no-pixel policy and Phase 2B0 acceptance report

**Files:**
- Create: `src/trustsr/data/local_policy.py`
- Create: `tests/data/test_local_data_policy.py`
- Modify: `.gitignore`
- Modify: `README.md`
- Create: `docs/reports/phase2b0-sen2naipv2-provenance.md`

**Interfaces:**
- Consumes: all Task 1–3 deliverables.
- Produces: a repository-level guard against tracked real TACO data and a reproducible acceptance record.

- [ ] **Step 1: Write the failing repository-policy tests**

Create tests against the wished-for interface:

```python
def tracked_data_policy_violations(repo_root: Path) -> tuple[str, ...]: ...
```

One test creates a temporary Git repository, force-adds a tiny `.taco`, and asserts the returned violations
name it. A second force-adds a file larger than 1 MiB below `artifacts/datasets/` and asserts it is rejected.
A third calls the function on this repository and expects no violations. The policy must reject:

- any tracked path ending in `.taco`;
- any newly tracked file under `artifacts/datasets/` other than `.json`/`.md`;
- any tracked file under `artifacts/datasets/` larger than `1_048_576` bytes.

The test must inspect only Git-tracked files, so an unrelated user cache outside the repository cannot cause a false failure.

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
uv run pytest tests/data/test_local_data_policy.py -v
```

Expected: collection fails because `trustsr.data.local_policy` does not exist.

- [ ] **Step 3: Implement tracked-file policy checks**

Create `src/trustsr/data/local_policy.py`. Run `git -C <repo_root> ls-files -z`, fail closed on a nonzero
return code, resolve every returned relative path inside `repo_root`, and return sorted human-readable
violations. Reject every `.taco`; below `artifacts/datasets/`, reject non-`.json`/`.md` files and files larger
than `1_048_576` bytes. Do not scan untracked files or paths outside the repository.

- [ ] **Step 4: Add explicit ignore rules and document local/cloud commands**

Append narrowly scoped rules to `.gitignore`:

```gitignore
*.taco
artifacts/datasets/cache/
artifacts/datasets/pixels/
```

Update `README.md` with:

- the offline `uv run trustsr-dataset-audit` command;
- an explicit statement that local SEN2NAIPv2 pixel downloads are forbidden;
- a statement that cloud access and real sample extraction begin only in Phase 2B1;
- no `git lfs pull` or Hugging Face download example.

- [ ] **Step 5: Produce the acceptance report from fresh evidence**

Create `docs/reports/phase2b0-sen2naipv2-provenance.md` containing the frozen revision, card hash, object count/total bytes, local/cloud boundary, exact verification commands, test count, Ruff result, repeatability SHA-256, and the statement `GPU required: no`.

Generate the repeatability evidence with:

```bash
uv run trustsr-dataset-audit > /tmp/trustsr-phase2b0-accepted.json
sha256sum /tmp/trustsr-phase2b0-accepted.json
```

- [ ] **Step 6: Run final scope and repository checks**

Run:

```bash
uv run pytest
uv run ruff check .
uv run trustsr-dataset-audit
uv run python -c 'from pathlib import Path; from trustsr.data.local_policy import tracked_data_policy_violations; assert not tracked_data_policy_violations(Path.cwd())'
git diff --check
git ls-files '*.taco'
git status --short
```

Expected: tests and Ruff pass; audit emits one JSON line; diff check is silent; `git ls-files '*.taco'` emits nothing; status contains only intended Phase 2B0 files before commit.

- [ ] **Step 7: Commit, push, and open the stacked pull request**

```bash
git add .gitignore README.md src/trustsr/data/local_policy.py tests/data/test_local_data_policy.py docs/reports/phase2b0-sen2naipv2-provenance.md docs/superpowers/specs/2026-08-27-phase2b0-sen2naipv2-provenance.md docs/superpowers/plans/2026-08-27-phase2b0-sen2naipv2-provenance.md
git commit -m "docs: accept phase2b0 provenance checkpoint"
git push -u origin feature/phase2b0-data-provenance
gh pr create --base feature/phase2a-conformal-core --head feature/phase2b0-data-provenance --title "Phase 2B0: freeze SEN2NAIPv2 provenance" --body-file <prepared-body-file>
```

Expected: a stacked PR targeting `feature/phase2a-conformal-core`, with no real data files and no credentials in its diff.

---

## Self-review

- Spec coverage: Tasks 1–4 cover strict source validation, exact pinned inventory, offline deterministic audit, local no-pixel policy, documentation, and acceptance evidence.
- Deferred by design: TACO dependency selection, real metadata schema discovery, geographic grouping, subset extraction, model caches, and GPU sampling belong to Phase 2B1 and are not hidden placeholders in this plan.
- Type consistency: Task 1's `DatasetSource`/`LfsObject` and `load_dataset_source()` signatures are consumed unchanged by Tasks 2–3.
- Dependency boundary: every implementation uses the standard library; no new runtime package is permitted.
