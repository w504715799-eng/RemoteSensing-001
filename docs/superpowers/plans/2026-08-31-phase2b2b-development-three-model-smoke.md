# Phase 2B2-B Development Three-Model Smoke Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Use superpowers:test-driven-development for every product-code change and superpowers:verification-before-completion before claiming success.

**Goal:** Run Bicubic, SEN2SRLite, and LDSR-S2 on exactly four frozen crosssensor development samples, persist 12 identity-bound prediction caches, compute the fixed OpenSR metric suite, and prove prediction-free byte-identical replay.

**Architecture:** Extend the frozen crosssensor selector with a development-only view, add one small evaluation module that owns deterministic result/cache-audit construction, and expose four restartable CLI stages. The execution stages receive models through factories; replay receives only frozen provenance and cache identities, so it cannot construct or invoke models. All pixels and prediction caches remain on persistent cloud storage, while only deterministic host-free JSON evidence may be copied into Git.

**Tech Stack:** Python 3.12, PyTorch, safetensors, rasterio, opensr-test 1.3.3, pytest, Ruff, uv, Bash.

**Approved spec:** `docs/superpowers/specs/2026-08-31-phase2b2b-development-three-model-smoke.md`

---

## Frozen values and interfaces

Use these constants exactly once in product code and import them elsewhere:

```python
EXPERIMENT_SCHEMA = "trustsr.phase2b2b-development-smoke.v1"
CACHE_AUDIT_SCHEMA = "trustsr.phase2b2b-cache-audit.v1"
INPUT_AUDIT_SHA256 = "fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b"
MODEL_NAMES = ("bicubic-x4", "sen2srlite-x4", "ldsr-s2-x4")
```

The existing `POST_MANIFEST_SHA256`, `PHASE2B1B_AUDIT_SHA256`, `BASE_MANIFEST_SHA256`, model adapters, `PredictionCache`, and `METRIC_KEYS` remain authoritative. Do not duplicate their digest values.

The deterministic smoke result has this top-level contract:

```python
{
    "schema": EXPERIMENT_SCHEMA,
    "dataset_role": "development_engineering_smoke_only",
    "upstream": {...},
    "bands": ["B04", "B03", "B02", "B08"],
    "scale": 4,
    "sample_count": 4,
    "model_count": 3,
    "prediction_count": 12,
    "samples": [...],
    "models": [...],
}
```

`models` is a list in `MODEL_NAMES` order. Every model record contains `name`, `model_provenance`, `cache_provenance`, four `predictions`, and `mean_metrics`. Every prediction contains `sample_id`, `correlation_bin`, `cache_key`, `prediction_sha256`, and all seven metrics under `metrics`.

The cache audit has this top-level contract:

```python
{
    "schema": CACHE_AUDIT_SCHEMA,
    "experiment_schema": EXPERIMENT_SCHEMA,
    "post_manifest_sha256": POST_MANIFEST_SHA256,
    "input_audit_sha256": INPUT_AUDIT_SHA256,
    "prediction_count": 12,
    "entries": [...],
}
```

Entries are sorted by `(MODEL_NAMES index, correlation_bin, sample_id)` and contain the two cache files' basename, byte size, and SHA-256. Runtime files are outside both contracts.

### Task 1: Add the development-only selector

**Files:**

- Modify: `src/trustsr/data/crosssensor_pairs.py`
- Modify: `tests/data/test_crosssensor_pairs.py`

**Step 1: Write the failing selector tests**

Add tests after the existing 12-cell selector tests:

```python
def test_select_development_smoke_records_filters_after_canonical_selection() -> None:
    selected = select_development_smoke_records(_complete_smoke_records())
    assert len(selected) == 4
    assert [record["split"] for record in selected] == ["development"] * 4
    assert [record["correlation_bin"] for record in selected] == [0, 1, 2, 3]


def test_select_development_smoke_records_is_input_order_independent() -> None:
    records = list(_complete_smoke_records())
    assert select_development_smoke_records(records[::-1]) == (
        select_development_smoke_records(records)
    )
```

Also add a test that monkeypatches or damages the full 12-cell set and proves the function does not accept a development-only four-row input. This preserves the rule “validate canonical 12 first, filter second.”

**Step 2: Run the tests to verify RED**

Run:

```bash
uv run pytest tests/data/test_crosssensor_pairs.py -k development_smoke -q
```

Expected: collection/import failure because `select_development_smoke_records` does not exist.

**Step 3: Implement the minimal selector**

Add:

```python
def select_development_smoke_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    selected = tuple(
        record for record in select_input_smoke_records(records)
        if record["split"] == "development"
    )
    if len(selected) != 4 or [record["correlation_bin"] for record in selected] != list(SMOKE_BINS):
        raise ValueError("development smoke selection must contain the four canonical bins")
    _require_unique_strings(selected, "sample_id")
    _require_unique_strings(selected, "spatial_group_id")
    return selected
```

Format the long condition to satisfy Ruff.

**Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/data/test_crosssensor_pairs.py -q
uv run ruff check src/trustsr/data/crosssensor_pairs.py tests/data/test_crosssensor_pairs.py
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/trustsr/data/crosssensor_pairs.py tests/data/test_crosssensor_pairs.py
git commit -m "feat: select phase2b2b development smoke records"
```

### Task 2: Verify the frozen Phase 2B2-A input audit

**Files:**

- Modify: `src/trustsr/data/input_audit.py`
- Modify: `tests/data/test_input_audit.py`

**Step 1: Write failing frozen-audit tests**

Add tests for:

```python
def test_load_frozen_input_audit_accepts_exact_canonical_bytes(tmp_path: Path) -> None: ...
def test_load_frozen_input_audit_rejects_wrong_digest(tmp_path: Path) -> None: ...
def test_load_frozen_input_audit_rejects_noncanonical_or_symlink(tmp_path: Path) -> None: ...
def test_load_frozen_input_audit_rejects_changed_schema_or_upstream(tmp_path: Path) -> None: ...
```

Build a valid payload with the existing `build_input_audit` helper, serialize it with `canonical_json`, and pass its actual test digest into a private validator. Keep the production public wrapper locked to `INPUT_AUDIT_SHA256` in Task 3 so synthetic tests do not pretend to match the real frozen digest.

**Step 2: Run to verify RED**

```bash
uv run pytest tests/data/test_input_audit.py -k load_frozen -q
```

Expected: import/attribute failure.

**Step 3: Implement strict loading**

Add:

```python
def load_input_audit(path: Path, *, expected_sha256: str) -> dict[str, object]: ...
```

Requirements:

- `path` is an existing non-symlink regular file;
- expected digest is lowercase SHA-256;
- hash raw bytes before parsing;
- parsed root is a dict;
- `canonical_json(parsed) == raw_bytes`;
- schema and upstream constants equal current frozen constants;
- `smoke_pair_count == 12`, `repeated_load_equal is True`, and both inference/GPU flags are false;
- any mismatch raises `ValueError` without writing files.

**Step 4: Verify GREEN**

```bash
uv run pytest tests/data/test_input_audit.py -q
uv run ruff check src/trustsr/data/input_audit.py tests/data/test_input_audit.py
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/trustsr/data/input_audit.py tests/data/test_input_audit.py
git commit -m "feat: verify frozen phase2b2a input audit"
```

### Task 3: Build context-bound cache identities and file evidence

**Files:**

- Create: `src/trustsr/evaluation/crosssensor_smoke.py`
- Create: `tests/evaluation/test_crosssensor_smoke.py`

**Step 1: Write failing cache-context tests**

Create synthetic four-sample `LoadedCrosssensorPair` fixtures and a `FakeModel`. Add tests for:

```python
def test_build_cache_provenance_binds_experiment_and_upstream() -> None: ...
def test_context_change_changes_prediction_identity() -> None: ...
def test_cache_entry_evidence_hashes_both_named_files(tmp_path: Path) -> None: ...
def test_snapshot_cache_files_detects_filename_size_mtime_or_digest_change(tmp_path: Path) -> None: ...
```

Assert the cache provenance preserves every model provenance scalar and adds exactly:

```python
{
    "experiment_schema": EXPERIMENT_SCHEMA,
    "post_manifest_sha256": POST_MANIFEST_SHA256,
    "input_audit_sha256": INPUT_AUDIT_SHA256,
}
```

Reject collisions with those reserved names rather than overwrite model data.

**Step 2: Run to verify RED**

```bash
uv run pytest tests/evaluation/test_crosssensor_smoke.py -k 'cache_provenance or evidence or snapshot' -q
```

Expected: module/import failure.

**Step 3: Implement the identity/evidence helpers**

Implement:

```python
def build_cache_provenance(model_provenance: Mapping[str, JsonScalar]) -> dict[str, JsonScalar]: ...
def cache_entry_evidence(cache_root: Path, identity: PredictionIdentity) -> dict[str, object]: ...
def snapshot_cache_files(cache_root: Path, identities: Sequence[PredictionIdentity]) -> tuple[tuple[object, ...], ...]: ...
```

`cache_entry_evidence` must call `PredictionCache.get(identity)` before hashing. Hash files in streaming chunks. `snapshot_cache_files` includes basename, `st_size`, `st_mtime_ns`, and SHA-256 for exactly two files per identity and rejects symlinks, duplicates, missing files, and extra suffixes for named keys.

**Step 4: Verify GREEN**

```bash
uv run pytest tests/evaluation/test_crosssensor_smoke.py -k 'cache_provenance or evidence or snapshot' -q
uv run ruff check src/trustsr/evaluation/crosssensor_smoke.py tests/evaluation/test_crosssensor_smoke.py
```

Expected: pass.

**Step 5: Commit**

```bash
git add src/trustsr/evaluation/crosssensor_smoke.py tests/evaluation/test_crosssensor_smoke.py
git commit -m "feat: bind phase2b2b prediction cache identities"
```

### Task 4: Implement deterministic model-grid evaluation

**Files:**

- Modify: `src/trustsr/evaluation/crosssensor_smoke.py`
- Modify: `tests/evaluation/test_crosssensor_smoke.py`

**Step 1: Write failing evaluation tests**

Add tests that assert:

- exactly four validated development pairs and models in `MODEL_NAMES` order are required;
- a cold run invokes every model four times and creates 12 cache pairs;
- a warm run invokes no model and returns byte-identical result/audit objects;
- metrics receive the exact tensor returned by `PredictionCache.get`, not the model's temporary tensor;
- model provenance mismatch, duplicate names, scale != 4, bad predictions, or non-finite/missing metrics fail;
- result contains exactly 84 sample-level values and 21 finite means;
- sample and model order are deterministic.

Use a metric stub whose values depend on model/sample index, so mean calculations can be checked exactly.

**Step 2: Run to verify RED**

```bash
uv run pytest tests/evaluation/test_crosssensor_smoke.py -k 'model_grid or warm or metrics' -q
```

Expected: missing runner failures.

**Step 3: Implement evaluation**

Add:

```python
def evaluate_development_smoke(
    loaded_pairs: Sequence[LoadedCrosssensorPair],
    models: Sequence[SRModel],
    cache_root: Path,
    *,
    metric_fn: Callable[[SRPair, torch.Tensor], Mapping[str, float]] = compute_opensr_metrics,
) -> tuple[dict[str, object], dict[str, object]]: ...
```

For each model/sample:

1. validate the model and pair metadata;
2. build cache provenance and identity;
3. call `cache.get`;
4. on miss call `model.predict`, validate, `cache.put`, then call `cache.get` again;
5. compute finite metrics from the verified cached tensor;
6. record prediction tensor digest and cache evidence.

Construct sample records from metadata and `tensor_sha256`; include no paths or runtime. Build arithmetic means in `METRIC_KEYS` order. Run `canonical_json` on both objects before returning them.

**Step 4: Add prediction-free rebuilding**

Write failing tests, then implement:

```python
def replay_development_smoke(
    loaded_pairs: Sequence[LoadedCrosssensorPair],
    committed_result: Mapping[str, object],
    committed_audit: Mapping[str, object],
    cache_root: Path,
    *,
    metric_fn: Callable[[SRPair, torch.Tensor], Mapping[str, float]] = compute_opensr_metrics,
) -> tuple[dict[str, object], dict[str, object]]: ...
```

It reconstructs model/cache provenance only from the validated committed result, verifies every audit identity against recomputed `PredictionIdentity.key`, snapshots cache files before/after, reads all predictions with `PredictionCache.get`, recomputes metrics, and returns rebuilt objects. It must not accept a model or factory argument and must not import model adapter modules.

Tests must mutate result/audit schema, cache key, LR digest, prediction digest, ordering,
entry count, file bytes, and add an extra audit entry; every mutation fails. Add a separate
test whose metric stub changes a cache file mtime during replay; the before/after snapshot
must detect that in-flight mutation.

**Step 5: Verify GREEN**

```bash
uv run pytest tests/evaluation/test_crosssensor_smoke.py -q
uv run ruff check src/trustsr/evaluation/crosssensor_smoke.py tests/evaluation/test_crosssensor_smoke.py
```

Expected: pass.

**Step 6: Commit**

```bash
git add src/trustsr/evaluation/crosssensor_smoke.py tests/evaluation/test_crosssensor_smoke.py
git commit -m "feat: evaluate and replay phase2b2b model grid"
```

### Task 5: Add staged CLI orchestration

**Files:**

- Create: `src/trustsr/cli/phase2b2b.py`
- Create: `tests/cli/test_phase2b2b.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Step 1: Write failing parser and path tests**

Test exactly four commands and no scientific override flags:

```python
assert set(subcommands) == {"preflight", "single", "smoke", "replay"}
for forbidden in ("--split", "--sample-id", "--limit", "--seed", "--metric"):
    with pytest.raises(SystemExit):
        parser.parse_args(["smoke", forbidden, "x"])
```

Test all commands require explicit storage confirmation and frozen manifest/input-audit digest arguments. Test invalid digest/audit fails before the Phase 2B2-B output directory is created.

**Step 2: Run to verify RED**

```bash
uv run pytest tests/cli/test_phase2b2b.py -q
```

Expected: module/import failure.

**Step 3: Implement shared loading and paths**

In `phase2b2b.py`, add:

```python
def _validate_upstream(args: argparse.Namespace) -> Path: ...
def _load_development_pairs(args: argparse.Namespace, *, limit: int | None = None) -> tuple[LoadedCrosssensorPair, ...]: ...
def _result_path(root: Path, filename: str) -> Path: ...
def _commit_identical_or_new(path: Path, payload: bytes) -> bool: ...
```

`_validate_upstream` calls `require_cloud_confirmation`, `load_crosssensor_records`, and `load_input_audit`. `_load_development_pairs` selects records before loading pixels and only calls `load_crosssensor_pair` for four development rows (or the first one). Validate every derived path for symlink components.

**Step 4: Implement `preflight` with lazy model imports**

Write tests proving hardware/input validation happens before model construction and no pair pixels are loaded. Then implement:

- capture one `GPUHardwareSnapshot`;
- lazily import and construct SEN2SRLite/LDSR from verified model directories plus Bicubic;
- verify exact names, scales, and provenance;
- write a separate non-deterministic `preflight-runtime.json` with environment/model provenance;
- do not create a prediction cache or deterministic result.

**Step 5: Implement `single` and `smoke`**

Inject a private `_model_factory(args)` in tests. `single` loads only bin 0 and calls the same evaluator with an explicit `expected_sample_count=1` option added under TDD; it commits `single-result.json` only. `smoke` loads four, evaluates all three models, and atomically commits the result and audit only when both validate. If either deterministic file already exists, require byte equality.

Measure duration and CUDA peak memory outside the evaluator and write separate `single-runtime.json` / `smoke-runtime.json`; those files are mutable operational evidence and never compared during replay.

**Step 6: Implement `replay` without model imports**

Tests monkeypatch Python import to fail for `trustsr.models.bicubic`, `sen2srlite`, and `ldsr_s2` during the handler. `replay` validates upstream, loads four development pairs, reads the two committed deterministic JSON files, calls `replay_development_smoke`, and requires:

```python
canonical_json(rebuilt_result) == committed_result_bytes
canonical_json(rebuilt_audit) == committed_audit_bytes
```

It emits a small stdout summary but writes no file.

**Step 7: Register the console script and update lock**

Add:

```toml
trustsr-phase2b2b = "trustsr.cli.phase2b2b:main"
```

Run `uv lock` only if `uv lock --check` says the project metadata changed the lock; do not update unrelated dependencies.

**Step 8: Verify GREEN**

```bash
uv run pytest tests/cli/test_phase2b2b.py tests/evaluation/test_crosssensor_smoke.py -q
uv run ruff check src/trustsr/cli/phase2b2b.py tests/cli/test_phase2b2b.py
uv lock --check
```

Expected: pass.

**Step 9: Commit**

```bash
git add src/trustsr/cli/phase2b2b.py tests/cli/test_phase2b2b.py pyproject.toml uv.lock
git commit -m "feat: orchestrate staged phase2b2b smoke"
```

### Task 6: Add the base-environment cloud runner

**Files:**

- Create: `scripts/phase2b2b/run_cloud.sh`
- Create: `tests/scripts/test_phase2b2b_scripts.py`

**Step 1: Write failing shell-runner tests**

Adapt the Phase 2B2-A fake-command harness. Test that the runner:

- uses `/opt/conda/bin/python` by default and never calls `conda create`, `pip`, `wget`, or `curl`;
- accepts only `preflight|single|smoke|replay`;
- passes storage root once and rejects stage attempts to override it;
- requires `--confirm-cloud-storage`, fixed manifest and input-audit arguments;
- rejects unsafe root/repository, missing mount, low capacity/inodes, symlink components, and colon paths;
- keeps stage logs under `<storage-root>/trustsr/phase2b2b/logs/`;
- does not mention or test a specific GPU model.

**Step 2: Run to verify RED**

```bash
uv run pytest tests/scripts/test_phase2b2b_scripts.py -q
```

Expected: runner missing.

**Step 3: Implement the minimal runner**

Source the same fail-closed path/mount patterns as Phase 2B2-A, but require more than 8 GiB free and 1024 inodes because predictions are retained. Invoke:

```bash
PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$base_python" -m trustsr.cli.phase2b2b "$stage" \
  --storage-root "$storage_root" "$@"
```

Do not create an environment or install packages. Use one JSONL log per stage and quote every path.

**Step 4: Verify GREEN and syntax**

```bash
uv run pytest tests/scripts/test_phase2b2b_scripts.py -q
bash -n scripts/phase2b2b/run_cloud.sh
uv run ruff check tests/scripts/test_phase2b2b_scripts.py
```

Expected: pass.

**Step 5: Commit**

```bash
git add scripts/phase2b2b/run_cloud.sh tests/scripts/test_phase2b2b_scripts.py
git commit -m "feat: add phase2b2b cloud runner"
```

### Task 7: Document the staged runbook and paper claim boundary

**Files:**

- Create: `docs/phase2b2b-cloud-runbook.md`
- Modify: `README.md`

**Step 1: Write the runbook**

Document exact commands in order:

1. check mount, repo commit, base Python imports, model directories, and no foreign GPU process;
2. run `preflight`;
3. inspect only the preflight summary/runtime;
4. run `single` and stop on any contract/runtime failure;
5. run `smoke`;
6. stop GPU processes and run `replay` on CPU;
7. verify zero remote compute processes, then tell the user the GPU can be paused;
8. copy only the two host-free JSON files selected for Git after a separate sensitive-data scan.

State prominently that only development metrics may be inspected and that this is not a paper result.

**Step 2: Update README status**

Add Phase 2B2-B as “local implementation / cloud acceptance pending” until real GPU evidence exists. Do not mark it complete during local tests.

**Step 3: Validate docs**

```bash
rg -n 'calibration|internal_test|development|GPU|replay' docs/phase2b2b-cloud-runbook.md README.md
rg -n 'ghp_|密码|password|ssh root@|/root/rivermind' docs/phase2b2b-cloud-runbook.md README.md
git diff --check
```

Expected: the first command shows explicit leakage/resource rules; the second has no matches.

**Step 4: Commit**

```bash
git add docs/phase2b2b-cloud-runbook.md README.md
git commit -m "docs: add phase2b2b staged cloud runbook"
```

### Task 8: Run local acceptance and stop at the GPU gate

**Files:**

- Modify only files required by failures attributable to Phase 2B2-B

**Step 1: Run focused tests**

```bash
uv run pytest \
  tests/data/test_crosssensor_pairs.py \
  tests/data/test_input_audit.py \
  tests/evaluation/test_crosssensor_smoke.py \
  tests/cli/test_phase2b2b.py \
  tests/scripts/test_phase2b2b_scripts.py -q
```

Expected: pass.

**Step 2: Run complete quality gates**

```bash
uv run pytest -q
uv run ruff check .
uv lock --check
git diff --check
```

Expected: all pass; existing Torch JIT deprecation warnings may remain but no new warnings are accepted without review.

**Step 3: Audit tracked artifacts and secrets**

```bash
git ls-files -z | xargs -0 -r du -b | sort -nr | head -20
git ls-files | rg '\.(tif|tiff|taco|safetensors|ckpt|pth|pt)$' || true
git grep -nE 'ghp_[A-Za-z0-9]{20,}|ssh[[:space:]]+[^[:space:]]+@|password[[:space:]]*[:=]|密码[：:]' -- . ':(exclude)uv.lock' || true
git status --short --branch
```

Expected: no new binary/pixel/model files or credentials; worktree contains only intentional commits.

**Step 4: Review against the approved spec**

Use `superpowers:requesting-code-review` and address findings with `superpowers:receiving-code-review`. Re-run all gates after any fix.

**Step 5: Commit any acceptance-only corrections**

Stage only the explicit files changed to address reviewed findings, then run:

```bash
git commit -m "fix: close phase2b2b local acceptance gaps"
```

Skip this commit if no corrections are needed.

**Step 6: Stop and request the current GPU endpoint**

Report the exact local test evidence and ask the user to start a GPU instance and provide only its current SSH endpoint. Do not ask for a password if the configured public key works. Do not start Phase 2B3 or full-360 inference.

### Task 9: Execute real cloud acceptance after GPU access is supplied

**Files:**

- Create after verified run: `artifacts/phase2b2b/sen2naipv2-development-three-model-smoke-v1.json`
- Create after verified run: `artifacts/phase2b2b/sen2naipv2-development-cache-audit-v1.json`
- Modify: `README.md`

**Step 1: Read-only remote checks**

Verify endpoint identity, mount/inode/capacity, repo commit, base interpreter imports, model asset hashes, CUDA visibility, and no foreign compute process. Never record SSH endpoint, hostname, credential, absolute storage path, or GPU model in deterministic Git artifacts.

**Step 2: Run staged commands**

Run `preflight`, then `single`. Inspect shape/range/cache and runtime. Only if single passes, run `smoke`. After smoke commits all 12 predictions, terminate model processes and run `replay` without model construction.

**Step 3: Verify real evidence**

Require:

- four development samples in bins 0–3;
- three frozen models and 12 unique cache keys;
- 84 finite sample metrics and 21 finite means;
- exact upstream/input/model/tensor binding;
- byte-identical result/audit replay;
- unchanged cache file set, size, mtime_ns, and digest;
- zero remaining compute processes.

**Step 4: Stage only host-free artifacts**

Copy the deterministic smoke result and cache audit into the two Git paths above. Recompute SHA-256 locally, compare with remote, scan for secrets/paths/host data, and ensure each file is small JSON. Do not copy predictions, weights, runtime files, or logs.

**Step 5: Update status and verify**

Mark Phase 2B2-B Checkpoint A complete in README only after real evidence passes. Run the complete Task 8 gates again.

**Step 6: Commit and publish for review**

```bash
git add artifacts/phase2b2b README.md
git commit -m "data: record phase2b2b development smoke"
git push -u origin feature/phase2b2b-development-smoke
```

Create a pull request with exact test counts, artifact SHA-256 values, paper claim limitations, and confirmation that calibration/internal_test metrics were never computed.

**Step 7: Tell the user to pause GPU**

Only after verifying no remote processes are running, explicitly tell the user the GPU server can be paused. Then reassess Phase 2B3 using the measured per-sample runtime and storage cost; do not automatically launch the 360-pair experiment.
