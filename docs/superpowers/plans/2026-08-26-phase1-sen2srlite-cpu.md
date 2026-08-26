# Phase 1A SEN2SRLite CPU Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task with a fresh implementer and reviewer for every task.

**Goal:** Add a supply-chain-verified SEN2SRLite RGBN ×4 adapter, safe prediction cache, and deterministic CPU comparison against bicubic interpolation on all nine OpenSR-Test SPOT v3 samples.

**Architecture:** Keep dataset, model, cache, and benchmark concerns separate. Both baselines implement a small structural protocol; the pretrained adapter validates assets before dynamic loading; the cache keys predictions by complete model/data identity; the CLI evaluates both models on one frozen in-memory sample manifest and emits stable JSON.

**Tech Stack:** Python 3.12, PyTorch, mlstac 0.4.9, sen2sr 0.8.5, safetensors, OpenSR-Test, pytest, Ruff, uv.

**Spec:** `docs/superpowers/specs/2026-08-26-phase1-sen2srlite-cpu.md`

## Global constraints

- Work only on `feature/phase1-pretrained-baselines` in its dedicated worktree.
- Use strict test-driven development: first add one focused failing test, observe the expected failure, implement the smallest behavior, then rerun the focused and full relevant tests.
- Unit tests must not access the network, download datasets/models, or require CUDA.
- Never load the model's dynamic `load.py` before all pinned asset hashes pass.
- Preserve phase 0 behavior and its CLI.
- Commit after each task using the exact commit message listed below.
- Do not begin LDSR-S2 or any GPU work in this plan.

### Task 1: Define the common SR model protocol

**Files:**

- Create: `src/trustsr/models/protocols.py`
- Modify: `src/trustsr/models/bicubic.py`
- Modify: `src/trustsr/models/__init__.py`
- Create: `tests/models/test_protocols.py`
- Modify: `tests/models/test_bicubic.py`

**Step 1: Write the failing protocol and provenance tests**

Add tests that require `BicubicX4` to satisfy a runtime-checkable `SRModel` protocol and return exactly JSON-scalar provenance fields describing `name`, `scale`, interpolation implementation/mode, `align_corners`, antialias choice, and `[0,1]` output policy. Add rejection cases for non-finite and out-of-range input so both baselines will share the same physical input assumptions.

**Step 2: Run the focused tests and observe failure**

Run:

```bash
uv run pytest tests/models/test_protocols.py tests/models/test_bicubic.py -q
```

Expected: collection or assertions fail because `protocols.py`, `provenance()`, and stricter validation do not yet exist.

**Step 3: Implement the smallest protocol and bicubic changes**

In `protocols.py`, define:

```python
JsonScalar = str | int | float | bool | None

@runtime_checkable
class SRModel(Protocol):
    name: str
    scale: int

    def predict(self, lr: torch.Tensor) -> torch.Tensor: ...
    def provenance(self) -> dict[str, JsonScalar]: ...
```

Set `BicubicX4.name = "bicubic_x4"` and `scale = 4`. Centralize validation only if doing so reduces duplication without changing the public contract. Ensure prediction validates float32, `(4,H,W)`, finite `[0,1]` input and returns detached contiguous CPU float32 clipped to `[0,1]`. Implement stable provenance with JSON scalar values only.

**Step 4: Verify task behavior**

Run:

```bash
uv run pytest tests/models/test_protocols.py tests/models/test_bicubic.py -q
uv run pytest -q
uv run ruff check .
```

Expected: all pass.

**Step 5: Commit**

```bash
git add src/trustsr/models tests/models
git commit -m "feat: define super-resolution model protocol"
```

### Task 2: Add the verified SEN2SRLite adapter

**Files:**

- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/trustsr/models/sen2srlite.py`
- Modify: `src/trustsr/models/__init__.py`
- Create: `tests/models/test_sen2srlite.py`

**Step 1: Add failing offline tests using a fake backend**

Tests must cover:

- rejecting wrong dtype, shape other than `(4,128,128)`, non-finite values, and values outside `[0,1]`;
- passing `(1,4,128,128)` to an injected backend under inference mode;
- accepting only backend output `(1,4,512,512)` and rejecting non-finite/wrong-shaped output;
- returning detached contiguous CPU float32 `(4,512,512)` clipped to `[0,1]`;
- reporting manifest, model ID, dependency versions, device, output policy, and asset hashes in provenance;
- detecting a missing asset and a deliberately modified byte before calling any loader;
- ensuring `from_pretrained()` calls asset verification before `mlstac.load()` through monkeypatched functions.

**Step 2: Observe the tests fail**

Run:

```bash
uv run pytest tests/models/test_sen2srlite.py -q
```

Expected: tests fail because the module and dependencies are absent.

**Step 3: Pin dependencies and implement integrity verification**

Add exact direct dependencies `mlstac==0.4.9` and `sen2sr==0.8.5`; add a compatible direct safetensors dependency if it is not already guaranteed as a runtime dependency. Run `uv lock` and `uv sync --dev`.

In `sen2srlite.py`, declare constants for the exact manifest URL, model ID, and every SHA-256 listed in the phase specification. Implement `verify_model_assets(root, expected=MODEL_ASSET_SHA256)` that resolves only named files under `root`, streams SHA-256, and raises a descriptive exception on a missing/mismatched asset.

Implement `download_verified_model(cache_dir)` with this strict order:

1. `mlstac.download(MODEL_MANIFEST_URL, cache_dir)`;
2. determine the returned model root without globbing outside `cache_dir`;
3. `verify_model_assets(model_root)`;
4. `mlstac.load(model_root / "mlm.json")`.

No fallback to unverified files is allowed.

**Step 4: Implement `SEN2SRLiteX4`**

Allow construction with an injected callable/backend for offline testing. Add `from_pretrained(cache_dir, device="cpu")`, which obtains the verified MLM and calls its compiled model API for the requested device. Apply the input/output contract from the spec, use `torch.inference_mode()`, and encode clipping as provenance. Do not silently move a non-CPU requested device or pretend CUDA is available.

**Step 5: Verify task behavior**

Run:

```bash
uv run pytest tests/models/test_sen2srlite.py tests/models/test_protocols.py -q
uv run pytest -q
uv run ruff check .
```

Expected: all pass without network access.

**Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/trustsr/models tests/models
git commit -m "feat: add verified SEN2SRLite adapter"
```

### Task 3: Add identity-bound safe prediction caching

**Files:**

- Create: `src/trustsr/artifacts/__init__.py`
- Create: `src/trustsr/artifacts/predictions.py`
- Create: `tests/artifacts/test_predictions.py`

**Step 1: Write failing cache tests**

Require tests for:

- canonical identity and cache key remaining stable across dictionary insertion order;
- key changes for model provenance, source, sample ID, tensor content, shape, or dtype changes;
- a stored prediction round trip preserving exact float32 values;
- cache miss before storage and cache hit afterward;
- rejection of wrong `(4,4H,4W)` shape, non-float32, non-finite, or out-of-range prediction;
- rejection/miss for modified metadata, mismatched identity, truncated safetensors, or a tensor digest mismatch;
- no `.pt`, pickle, absolute path, timestamp, or temporary file left after a successful write.

**Step 2: Observe failure**

Run:

```bash
uv run pytest tests/artifacts/test_predictions.py -q
```

Expected: module import fails.

**Step 3: Implement canonical identity**

Create functions that hash contiguous CPU tensor bytes and construct an identity object from model provenance, source, sample ID, and LR tensor metadata/digest. Serialize with UTF-8, sorted keys, compact separators, and `allow_nan=False`; use the resulting SHA-256 as the cache key.

**Step 4: Implement safe atomic storage**

Store the prediction with safetensors and a canonical JSON sidecar. Sidecar fields must include schema version, cache key, complete identity, prediction shape/dtype/digest, and tensor filename. Validate the output contract against the LR size before writing. Write unique temporary siblings, fsync/close through normal library behavior, then `Path.replace()` into final names; clean temporary files on exceptions.

On load, require both files, validate metadata and identity before returning, use safetensors only, validate tensor digest and output contract, and return `None` for an ordinary miss. Corruption or a present-but-invalid entry must raise a dedicated cache integrity exception so it cannot be confused with a miss.

**Step 5: Verify task behavior**

Run:

```bash
uv run pytest tests/artifacts/test_predictions.py -q
uv run pytest -q
uv run ruff check .
```

Expected: all pass.

**Step 6: Commit**

```bash
git add src/trustsr/artifacts tests/artifacts
git commit -m "feat: add verified prediction cache"
```

### Task 4: Build and run the deterministic two-model SPOT benchmark

**Files:**

- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/trustsr/cli/benchmark_baselines.py`
- Create: `tests/cli/test_benchmark_baselines.py`
- Modify: `README.md`

**Step 1: Write failing CLI tests with injected data and models**

Refactor the benchmark entry point around a testable function while keeping production defaults explicit. Tests require:

- exactly nine unique SPOT samples; eight or ten must fail rather than silently truncate;
- both models receive the same ordered sample objects and share one manifest hash;
- cache miss performs `predict()` once per model/sample and cache replay performs zero predictions;
- per-sample and mean metrics contain only finite JSON numbers;
- the result schema contains `run` and `models`, complete provenance, ordered samples, and environment/dataset versions;
- the same fake run written twice is byte-identical and omits timestamps, durations, absolute paths, and cache-hit state;
- a prediction that violates the model output contract fails before metrics are computed.

**Step 2: Observe failure**

Run:

```bash
uv run pytest tests/cli/test_benchmark_baselines.py -q
```

Expected: module import fails.

**Step 3: Implement the benchmark orchestration**

Create a pure orchestration function that accepts pairs, models, cache root, result path, and environment metadata. Freeze the nine-sample manifest from ordered `source`/`sample_id` values and input/target tensor digests. For every model/sample, construct the cache identity, load or predict/store, validate `(4,4H,4W)` finite float32 `[0,1]`, then call the existing finite OpenSR metric wrapper. Calculate means in deterministic sample/metric order and reject every non-finite result.

Write canonical JSON atomically with `sort_keys=True`, fixed indentation/newline, and `allow_nan=False`. The result must not depend on whether predictions came from cache.

**Step 4: Add production CLI defaults and documentation**

Register:

```toml
trustsr-benchmark = "trustsr.cli.benchmark_baselines:main"
```

Production `main()` must load `load_opensr_pairs(dataset="spot", limit=9)`, assert the returned set is exactly the expected nine SPOT v3 samples, instantiate `BicubicX4` and verified `SEN2SRLiteX4` on CPU, and use the paths from the specification. Add optional command-line path overrides without weakening dataset/model defaults.

Update README with:

- `uv sync --dev` and `uv run trustsr-benchmark` commands;
- first-run download/caching expectations and approximate CPU suitability;
- exact output/cache/model locations;
- the statement that SPOT is a development reproducibility check, not final scientific evidence;
- the future GPU phase is intentionally separate.

**Step 5: Verify offline behavior**

Run:

```bash
uv run pytest tests/cli/test_benchmark_baselines.py -q
uv run pytest -q
uv run ruff check .
```

Expected: all pass without network access.

**Step 6: Commit the implementation**

```bash
git add pyproject.toml uv.lock src/trustsr/cli tests/cli README.md
git commit -m "feat: benchmark pretrained SR baselines"
```

**Step 7: Run the real CPU acceptance test twice**

Run from a clean working tree:

```bash
uv run trustsr-benchmark
sha256sum artifacts/phase1/spot-v3-baselines.json
uv run trustsr-benchmark
sha256sum artifacts/phase1/spot-v3-baselines.json
```

Expected: the first run downloads and verifies assets/data and fills prediction caches; the second uses caches; both hashes are identical. Inspect the JSON to confirm nine samples per model, matching manifest hashes, finite metrics, and complete provenance. Do not commit downloaded models, data, caches, or result artifacts.

### Task 5: Final integration review and handoff

**Files:**

- Review all files changed since the phase 0 base commit.

**Step 1: Run final gates**

```bash
uv sync --dev
uv run pytest -q
uv run ruff check .
git status --short
```

Expected: tests and lint pass; only ignored runtime data/artifacts may exist.

**Step 2: Perform two independent reviews**

First review the implementation against every statement in the phase 1A specification and this plan. Then perform a code-quality/security/reproducibility review, paying special attention to dynamic model loading order, cache identity, physical value validation, deterministic output, and preservation of phase 0 behavior. Fix findings with tests and commit fixes separately.

**Step 3: Push the stacked feature branch**

Push `feature/phase1-pretrained-baselines` to the configured GitHub remote. Because phase 0 is still under review, keep phase 1A based on `feature/phase0-foundation`; do not merge either branch automatically.

**Step 4: Stop at the GPU boundary**

Report the real benchmark outcome, tests, commits, and remote branch. Ask the user for the rented GPU SSH connection only when they are ready to begin the separately planned phase 1B LDSR-S2 work.

