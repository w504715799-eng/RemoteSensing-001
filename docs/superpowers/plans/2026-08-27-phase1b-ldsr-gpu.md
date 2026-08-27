# Phase 1B LDSR-S2 GPU Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a supply-chain-verified, deterministic LDSR-S2 RGBN ×4 adapter and run a staged one-sample then nine-sample benchmark on a single cloud GPU that satisfies the approved capability gate.

**Architecture:** Keep GPU-only imports behind lazy factories so the default CPU environment remains unchanged. Separate immutable asset verification, safe upstream backend construction, RNG-isolated model adaptation, repeatability evaluation, staged CLI orchestration, and remote operations. Local tasks and reviews finish first; the cloud instance stays off until the explicit GPU acceptance task.

**Tech Stack:** Python 3.12, PyTorch 2.x/CUDA 13, opensr-model 1.1.1, OmegaConf, safetensors, OpenSR-Test, Conda, uv 0.12.5, pytest, Ruff, Bash, SSH/rsync.

**Spec:** `docs/superpowers/specs/2026-08-27-phase1b-ldsr-gpu.md`

## Global Constraints

- Work only on `feature/phase1b-ldsr-gpu` in `/home/wanghongxu/code/RemoteSensing001/.worktrees/phase1b-ldsr-gpu`.
- Use strict TDD for every Python behavior: add the focused failing test, run it and retain RED output, implement minimally, then run focused tests, the full suite once, and Ruff.
- Unit tests must remain offline, CPU-only, and must not import `opensr_model` unless a fake module is injected.
- The default `uv sync --dev` environment must not install `opensr-model`; GPU installation uses the optional `gpu` extra.
- Pin `opensr-model==1.1.1`, uv `0.12.5`, upstream tag commit `10f4c01cc8172586841ea9e78c6de9939da47337`, package/config/checkpoint hashes, and the exact checkpoint size from the specification.
- Never call upstream `load_pretrained()`. Verify the checkpoint and packaged config before `torch.load(..., weights_only=True)` or backend construction consumes them.
- Preserve Phase 0 and Phase 1A CLIs, identifiers, deterministic JSON, caches, tests, and CPU behavior.
- `LDSRS2X4` uses CUDA only in production, fixed seed `3407`, 100 steps, eta `0.95`, temperature `1.0`, histogram matching enabled, and output clipping to `[0,1]`.
- Do not add uncertainty-map batches, training, `opensr-utils`, whole-tile inference, external test datasets, or CUDA fallbacks.
- Do not store SSH host, port, password, private key, GitHub credentials, or cloud-console credentials in Git, scripts, manifests, reports, shell history, or artifacts.
- No GPU task begins until all local tasks and reviews pass and the user is explicitly asked to restart the instance.
- Each implementation task commits with the exact message listed. Remote runtime data, model files, prediction caches, and result JSON stay ignored.

---

### Task 1: Add optional GPU dependencies and verified asset download

**Files:**

- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/trustsr/models/ldsr_assets.py`
- Create: `tests/models/test_ldsr_assets.py`

**Interfaces:**

- Consumes: standard library `hashlib`, `os`, `tempfile`, `urllib.request`, and `Path`.
- Produces:

```python
@dataclass(frozen=True)
class VerifiedAsset:
    path: Path
    size: int
    sha256: str

def file_sha256(path: Path | str) -> str: ...
def verify_asset(
    path: Path | str,
    *,
    expected_size: int,
    expected_sha256: str,
) -> VerifiedAsset: ...
def download_verified_checkpoint(
    model_dir: Path | str,
    *,
    opener: Callable[..., ContextManager[BinaryIO]] = urlopen,
    url: str = CHECKPOINT_URL,
    expected_size: int = CHECKPOINT_SIZE,
    expected_sha256: str = CHECKPOINT_SHA256,
) -> VerifiedAsset: ...
def verify_packaged_config(package_root: Path | str) -> VerifiedAsset: ...
```

- Constants are exact: `CHECKPOINT_NAME`, `CHECKPOINT_URL`, `CHECKPOINT_SIZE = 1_130_715_795`, `CHECKPOINT_SHA256`, `CONFIG_RELATIVE_PATH = Path("configs/config_10m.yaml")`, `CONFIG_SIZE = 1_487`, and `CONFIG_SHA256` from the spec.

- [ ] **Step 1: Add failing asset tests**

Create tests with a context-manager fake response whose `read(1024 * 1024)` returns deterministic chunks. Require:

```python
def test_download_verifies_then_atomically_commits(tmp_path):
    payload = b"verified checkpoint"
    asset = download_verified_checkpoint(
        tmp_path,
        opener=FakeOpener(payload),
        expected_size=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert asset.path == tmp_path / CHECKPOINT_NAME
    assert asset.path.read_bytes() == payload
    assert not list(tmp_path.glob(f".{CHECKPOINT_NAME}.*.tmp"))

def test_download_hash_failure_never_commits_final_file(tmp_path):
    with pytest.raises(AssetIntegrityError, match="SHA-256"):
        download_verified_checkpoint(
            tmp_path,
            opener=FakeOpener(b"wrong"),
            expected_size=5,
            expected_sha256="0" * 64,
        )
    assert not (tmp_path / CHECKPOINT_NAME).exists()
```

Also test missing file, exact size mismatch, existing valid-file reuse without invoking the opener, existing invalid-file failure without silently redownloading, config path confinement, config hash mismatch, fsync/temporary cleanup on raised read errors, and constants matching the specification.

- [ ] **Step 2: Run RED**

Run:

```bash
uv run pytest tests/models/test_ldsr_assets.py -q
```

Expected: collection fails with `ModuleNotFoundError: trustsr.models.ldsr_assets`.

- [ ] **Step 3: Add the optional dependency and lock it**

Add:

```toml
[project.optional-dependencies]
gpu = [
  "opensr-model==1.1.1",
]
```

Run:

```bash
uv lock
```

Inspect `uv.lock` and require the `opensr-model` wheel entry to include SHA-256 `6168336d800d24976751bba46dd6cb129906109608b8c6003354c89a7a5b72e0`. Do not replace the project-wide PyTorch range with an unreviewed CUDA-specific URL.

- [ ] **Step 4: Implement fail-closed asset handling**

`verify_asset()` streams 1 MiB blocks and validates both byte count and SHA-256. `download_verified_checkpoint()` must:

```python
model_root = Path(model_dir).resolve()
model_root.mkdir(parents=True, exist_ok=True)
final_path = (model_root / CHECKPOINT_NAME).resolve()
if final_path.parent != model_root:
    raise ValueError("checkpoint path escapes model directory")
```

If `final_path` exists, verify and return it without network access. Otherwise write a unique temporary sibling, flush and fsync it, verify the temporary file, `os.replace()` it into the final path, and fsync the directory. Delete the temporary file on every exception. Define `AssetIntegrityError(RuntimeError)` for a present or downloaded invalid asset.

`verify_packaged_config(package_root)` resolves exactly `package_root / CONFIG_RELATIVE_PATH`, rejects path escape, and calls `verify_asset()` with `CONFIG_SIZE = 1_487` and `CONFIG_SHA256`. No runtime network lookup or size discovery is allowed.

- [ ] **Step 5: Verify default and GPU dependency modes**

Run:

```bash
uv sync --dev
uv run python -c 'import importlib.util; assert importlib.util.find_spec("opensr_model") is None'
uv run pytest tests/models/test_ldsr_assets.py -q
uv sync --dev --extra gpu
uv run python -c 'import importlib.metadata as m; assert m.version("opensr-model") == "1.1.1"'
uv sync --dev
uv run pytest -q
uv run ruff check .
```

Expected: all assertions/tests pass; the final sync returns to the CPU-only environment.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/trustsr/models/ldsr_assets.py tests/models/test_ldsr_assets.py
git commit -m "build: add verified LDSR-S2 GPU assets"
```

### Task 2: Construct the upstream backend without unsafe auto-loading

**Files:**

- Create: `src/trustsr/models/ldsr_backend.py`
- Create: `tests/models/test_ldsr_backend.py`

**Interfaces:**

- Consumes: `VerifiedAsset`, `download_verified_checkpoint()`, `verify_packaged_config()` from Task 1.
- Produces:

```python
def load_verified_state_dict(
    checkpoint: VerifiedAsset,
    *,
    map_location: str | torch.device,
    torch_load: Callable[..., Any] = torch.load,
) -> dict[str, torch.Tensor]: ...

def build_verified_backend(
    model_dir: Path | str,
    *,
    device: str,
    package_module: ModuleType | None = None,
    omega_conf: Any | None = None,
) -> Any: ...
```

- [ ] **Step 1: Add failing safe-load tests**

Use fake package modules and fake OmegaConf objects. Test this exact event order:

```python
assert events == [
    "download_checkpoint",
    "verify_config",
    "parse_config",
    "construct_backend",
    "torch_load_weights_only",
    "strict_state_load",
]
```

The fake `torch_load` asserts:

```python
assert kwargs == {"map_location": torch.device("cuda:0"), "weights_only": True}
```

Require rejection before backend construction for invalid checkpoint/config; rejection of a checkpoint object without a mapping `state_dict`; rejection of non-string state keys or non-tensor values; removal only of keys containing `loss`; and `backend.model.load_state_dict(filtered, strict=True)`.

- [ ] **Step 2: Run RED**

```bash
uv run pytest tests/models/test_ldsr_backend.py -q
```

Expected: collection fails because `ldsr_backend.py` does not exist.

- [ ] **Step 3: Implement lazy imports and safe loading**

Production lazy imports occur inside `build_verified_backend()`:

```python
if package_module is None:
    import opensr_model as package_module
if omega_conf is None:
    from omegaconf import OmegaConf as omega_conf
```

Resolve `Path(package_module.__file__).parent`, verify the packaged config before parsing it, download/verify the checkpoint, build `package_module.SRLatentDiffusion(config, device=device)`, then call `load_verified_state_dict()`. Never call `backend.load_pretrained()`.

Validate the loaded object is a mapping with a mapping under `state_dict`; copy only string-to-tensor entries whose key does not contain `loss`. Let strict state-loading errors propagate as `BackendLoadError` with the original exception chained. After successful load require `backend.training is False` and `backend.model.training is False`.

- [ ] **Step 4: Verify**

```bash
uv run pytest tests/models/test_ldsr_backend.py tests/models/test_ldsr_assets.py -q
uv run pytest -q
uv run ruff check .
```

- [ ] **Step 5: Commit**

```bash
git add src/trustsr/models/ldsr_backend.py tests/models/test_ldsr_backend.py
git commit -m "feat: load verified LDSR-S2 backend"
```

### Task 3: Add the RNG-isolated `LDSRS2X4` adapter

**Files:**

- Create: `src/trustsr/models/ldsr_s2.py`
- Modify: `src/trustsr/models/__init__.py`
- Create: `tests/models/test_ldsr_s2.py`

**Interfaces:**

- Consumes: `build_verified_backend(model_dir, device=...)` from Task 2 and `JsonScalar`/`SRModel` from the existing protocol.
- Produces:

```python
class LDSRS2X4:
    name = "ldsr-s2-x4"
    scale = 4

    def __init__(
        self,
        backend: Callable[..., torch.Tensor] | Any,
        *,
        device: str = "cuda:0",
        seed: int = 3407,
        sampling_steps: int = 100,
        sampling_eta: float = 0.95,
        sampling_temperature: float = 1.0,
        histogram_matching: bool = True,
    ) -> None: ...

    @classmethod
    def from_pretrained(cls, model_dir: Path | str, *, device: str = "cuda:0") -> "LDSRS2X4": ...
    def provenance(self) -> dict[str, JsonScalar]: ...
    def predict(self, lr: torch.Tensor) -> torch.Tensor: ...
```

- [ ] **Step 1: Add failing adapter tests with a fake backend**

The fake backend records input, keyword arguments, inference mode, and current Python/NumPy/Torch random draws. Tests require:

- exact float32 `(4,128,128)` finite `[0,1]` input;
- exact backend batch `(1,4,128,128)` on configured device;
- keyword arguments exactly `sampling_steps=100`, `sampling_eta=0.95`, `sampling_temperature=1.0`, `histogram_matching=True`, `save_iterations=False`, `verbose=False`;
- inference mode enabled;
- wrong-shaped/non-finite backend output rejected;
- public output detached, contiguous CPU float32 `(4,512,512)` clipped to `[0,1]`;
- two calls with the same configured seed show the same fake-backend RNG observations;
- Python, NumPy and Torch RNG states observed after `predict()` equal the states expected if `predict()` had never run;
- cuDNN deterministic/benchmark flags are restored after success and exception;
- provenance contains every scalar field from the specification and cache keys change for seed, steps, eta, temperature, histogram matching, checkpoint/config hash, package version, PyTorch version or CUDA runtime changes;
- `from_pretrained()` rejects CPU/unavailable CUDA before calling `build_verified_backend()`.

- [ ] **Step 2: Run RED**

```bash
uv run pytest tests/models/test_ldsr_s2.py -q
```

Expected: collection fails because `LDSRS2X4` is absent.

- [ ] **Step 3: Implement isolated randomness**

Implement a private context manager that snapshots/restores `random.getstate()`, `numpy.random.get_state()`, cuDNN flags, and uses:

```python
torch_device = torch.device(device)
cuda_devices = [torch_device.index or 0] if torch_device.type == "cuda" else []
with torch.random.fork_rng(devices=cuda_devices):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    yield
```

Restoration belongs in `finally`. Reject bool-as-int and invalid values: seed must be a non-negative exact int; steps a positive exact int; eta and temperature finite non-negative floats; histogram matching an exact bool.

- [ ] **Step 4: Implement prediction and provenance**

Call the backend under `torch.inference_mode()` and the RNG context. Validate output before moving it to CPU. Provenance must use only JSON scalars and include:

```python
{
    "name": "ldsr-s2-x4",
    "scale": 4,
    "implementation_schema_version": 1,
    "opensr_model_version": "1.1.1",
    "torch_version": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "checkpoint_name": CHECKPOINT_NAME,
    "checkpoint_url": CHECKPOINT_URL,
    "checkpoint_size": CHECKPOINT_SIZE,
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "config_sha256": CONFIG_SHA256,
    "device": "cuda",
    "seed": 3407,
    "sampling_steps": 100,
    "sampling_eta": 0.95,
    "sampling_temperature": 1.0,
    "histogram_matching": True,
    "output_policy": "clip_to_[0,1]",
}
```

Normalize `cuda:0` to scalar provenance value `cuda` so the physical GPU index/name does not alter the algorithm cache identity.

- [ ] **Step 5: Verify**

```bash
uv run pytest tests/models/test_ldsr_s2.py tests/models/test_protocols.py -q
uv run pytest -q
uv run ruff check .
```

- [ ] **Step 6: Commit**

```bash
git add src/trustsr/models/ldsr_s2.py src/trustsr/models/__init__.py tests/models/test_ldsr_s2.py
git commit -m "feat: add deterministic LDSR-S2 adapter"
```

### Task 4: Add repeatability evaluation and three-model benchmark support

**Files:**

- Create: `src/trustsr/evaluation/repeatability.py`
- Create: `tests/evaluation/test_repeatability.py`
- Modify: `src/trustsr/cli/benchmark_baselines.py`
- Modify: `tests/cli/test_benchmark_baselines.py`

**Interfaces:**

- Consumes: any `SRModel`, `tensor_sha256()`, and existing `run_benchmark()`.
- Produces:

```python
@dataclass(frozen=True)
class RepeatabilitySummary:
    first_sha256: str
    second_sha256: str
    bitwise_equal: bool
    max_abs_diff: float
    tolerance: float

    def as_dict(self) -> dict[str, str | bool | float]: ...

class RepeatabilityError(RuntimeError): ...

def run_repeatability(
    model: SRModel,
    lr: torch.Tensor,
    *,
    tolerance: float = 1e-6,
) -> tuple[torch.Tensor, RepeatabilitySummary]: ...

def run_benchmark(..., expected_model_count: int = 2) -> dict[str, object]: ...
```

- [ ] **Step 1: Add failing repeatability tests**

Test exact equality, non-bitwise outputs within `1e-6`, outputs above tolerance, wrong shape/dtype/non-finite/range, and exactly two calls regardless of equality. Require `RepeatabilityError` for failure and require `first` to be a detached contiguous CPU float32 tensor.

Add benchmark tests:

```python
result = run_benchmark(
    pairs=PAIRS,
    models=[FakeModel("a"), FakeModel("b"), FakeModel("c")],
    expected_model_count=3,
    ...,
)
assert list(result["models"]) == ["a", "b", "c"]
```

Also require default calls still reject one or three models and Phase 1A's two-model result remains byte-identical.

- [ ] **Step 2: Run RED**

```bash
uv run pytest tests/evaluation/test_repeatability.py tests/cli/test_benchmark_baselines.py -q
```

- [ ] **Step 3: Implement repeatability**

Call `model.predict(lr)` twice without consulting `PredictionCache`. Validate both outputs share the same shape/dtype and finite `[0,1]` values. Compute:

```python
bitwise_equal = torch.equal(first, second)
max_abs_diff = float((first - second).abs().max().item())
```

Reject non-finite/negative tolerance and raise if `not bitwise_equal and max_abs_diff > tolerance`. Hash contiguous outputs with the existing tensor helper.

- [ ] **Step 4: Generalize benchmark count without changing Phase 1A defaults**

Add the keyword-only `expected_model_count: int = 2`, validate it is an exact positive int, and replace the hard-coded `len(models) != 2` check. Do not change result schema, ordering, cache behavior, environment keys, or Phase 1A `main()`.

- [ ] **Step 5: Verify**

```bash
uv run pytest tests/evaluation/test_repeatability.py tests/cli/test_benchmark_baselines.py -q
uv run pytest -q
uv run ruff check .
```

- [ ] **Step 6: Commit**

```bash
git add src/trustsr/evaluation/repeatability.py tests/evaluation/test_repeatability.py src/trustsr/cli/benchmark_baselines.py tests/cli/test_benchmark_baselines.py
git commit -m "feat: add LDSR repeatability gate"
```

### Task 5: Add staged GPU manifests and CLI orchestration

**Files:**

- Create: `src/trustsr/artifacts/gpu_run.py`
- Modify: `src/trustsr/artifacts/__init__.py`
- Create: `src/trustsr/cli/ldsr_gpu.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `tests/artifacts/test_gpu_run.py`
- Create: `tests/cli/test_ldsr_gpu.py`

**Interfaces:**

- Consumes: verified LDSR factory, repeatability gate, prediction cache, OpenSR loader, metric computation, and generalized benchmark.
- Produces commands:

```text
trustsr-ldsr-gpu preflight
trustsr-ldsr-gpu single
trustsr-ldsr-gpu benchmark
trustsr-ldsr-gpu manifest
```

- Produces artifact helpers:

```python
def collect_gpu_environment(
    *,
    command_runner: Callable[..., CompletedProcess[str]] = subprocess.run,
) -> dict[str, JsonScalar]: ...

def write_artifact_manifest(root: Path, relative_paths: Sequence[Path]) -> Path: ...
def verify_artifact_manifest(root: Path, manifest_path: Path) -> None: ...

def run_preflight(args: argparse.Namespace) -> dict[str, object]: ...
def run_single(args: argparse.Namespace) -> dict[str, object]: ...
def run_three_model_benchmark(args: argparse.Namespace) -> dict[str, object]: ...
def run_manifest(args: argparse.Namespace) -> dict[str, object]: ...
```

The environment JSON has top-level `schema_version`, `run_started_utc`, `git_commit`, `runtime`, `gpu`, `limits`, `determinism`, `dependency_lock_sha256`, and `model_provenance`. The artifact manifest is exactly:

```python
{
    "schema_version": 1,
    "files": [
        {"path": "phase1b/environment.json", "size": 123, "sha256": "a" * 64},
    ],
}
```

Production uses real values; the example fixes the field names and scalar types.

- [ ] **Step 1: Add failing manifest tests**

Use a fake command runner for `nvidia-smi` and `nvcc`. Require exact scalar fields for schema version, Git commit, `run_started_utc` as an ISO-8601 `Z` string, GPU name/UUID/driver/memory/compute capability, cgroup CPU/memory limit, Python/Conda/uv/PyTorch/CUDA toolkit/runtime/opensr-model/OpenSR-Test versions, deterministic flags, and dependency-lock SHA-256. This environment manifest is intentionally non-deterministic; timestamps remain forbidden only in the repeatability and benchmark result JSON.

Assert serialized manifest text contains none of: `ssh`, `password`, `private`, `github`, the test temporary absolute path, hostname, or port. Environment manifests may contain GPU UUID but no host identity.

Artifact-manifest tests require sorted relative POSIX paths, file size and SHA-256; reject absolute paths, `..`, symlink escape, missing files, modified files, and malformed digest entries.

- [ ] **Step 2: Add failing CLI tests**

Monkeypatch all model/data/GPU factories. Require:

- `preflight` writes environment JSON and constructs exactly one LDSR model but never loads SPOT;
- `single` loads exactly nine upstream samples, selects only `spot-0000`, performs two fresh predictions, writes deterministic summary and runtime report separately, stores/reads one verified prediction cache entry, and refuses to continue when repeatability fails;
- `benchmark` uses the exact nine identities and ordered model names `bicubic-x4`, `sen2srlite-x4`, `ldsr-s2-x4`, calls `run_benchmark(... expected_model_count=3)`, and writes `artifacts/phase1b/spot-v3-three-models.json`;
- `manifest` hashes only the allowlisted Phase 1B outputs and fails if any are absent;
- CLI path overrides work, but seed/steps/eta/temperature/histogram settings are not user-overridable in this phase;
- runtime JSON alone may contain durations/peak memory; deterministic repeatability and benchmark JSON contain no timestamp, duration, path or cache-hit fields.

- [ ] **Step 3: Run RED**

```bash
uv run pytest tests/artifacts/test_gpu_run.py tests/cli/test_ldsr_gpu.py -q
```

- [ ] **Step 4: Implement canonical manifests**

Reuse canonical sorted JSON and atomic fsync writing patterns. `collect_gpu_environment()` runs only fixed argument arrays—never shell strings—and parses:

```text
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.free,compute_cap --format=csv,noheader,nounits
nvcc --version
conda --version
uv --version
git rev-parse HEAD
```

Read `/sys/fs/cgroup/memory.max` and `/sys/fs/cgroup/cpu.max` when present. Record package versions through `importlib.metadata`. Do not record hostname, user, network interface, SSH variables, current directory, or environment-variable dumps.

- [ ] **Step 5: Implement the staged CLI**

Register:

```toml
trustsr-ldsr-gpu = "trustsr.cli.ldsr_gpu:main"
```

Use `argparse` subparsers with defaults rooted at `data/`, `models/`, and `artifacts/`. Before model construction, every production subcommand requires CUDA, exactly one visible GPU, a numeric `major.minor` compute capability of at least `8.0`, at least 18 GiB free VRAM, and no unexpected active compute process; the active-process check permits only the current process after model construction and fails before construction if another PID is reported. Record the actual name, UUID, driver, memory, and capability without making a product-name requirement.

`single` records `torch.cuda.reset_peak_memory_stats()`, monotonic duration and `torch.cuda.max_memory_allocated()` only in `single-runtime.json`. Compare duration only within the hardware recorded in the environment manifest; deterministic hashes, repeatability, and quality metrics remain comparable. Its deterministic summary contains sample identity/input digest, model provenance, two output hashes, equality/difference/tolerance, cache key and finite metrics.

The deterministic single-result schema is exactly:

```python
{
    "schema_version": 1,
    "source": "opensr-test/spot/v3",
    "sample_id": "spot-0000",
    "lr_sha256": "a" * 64,
    "model_provenance": {"name": "ldsr-s2-x4", "scale": 4},
    "repeatability": {
        "first_sha256": "b" * 64,
        "second_sha256": "b" * 64,
        "bitwise_equal": True,
        "max_abs_diff": 0.0,
        "tolerance": 1e-6,
    },
    "cache_key": "c" * 64,
    "metrics": {"reflectance": 0.0, "spectral": 0.0, "spatial": 0.0,
                "synthesis": 0.0, "ha_metric": 0.0, "om_metric": 0.0,
                "im_metric": 0.0},
}
```

The repeated letters only illustrate exact string type/length; production tests must assert the produced values match `tensor_sha256()` and the cache identity exactly.

`benchmark` uses SEN2SRLite on CPU and LDSR on `cuda:0`; it writes the existing deterministic result schema with environment `device="cuda:0"` and then exits. `manifest` allowlists environment, single deterministic/runtime, three-model result, and current LDSR cache sidecars/tensors discovered from keys named in deterministic/model results—not every old cache file.

- [ ] **Step 6: Verify**

```bash
uv run pytest tests/artifacts/test_gpu_run.py tests/cli/test_ldsr_gpu.py -q
uv run pytest -q
uv run ruff check .
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/trustsr/artifacts src/trustsr/cli/ldsr_gpu.py tests/artifacts/test_gpu_run.py tests/cli/test_ldsr_gpu.py
git commit -m "feat: add staged LDSR-S2 GPU workflow"
```

### Task 6: Add reproducible remote bootstrap and operator runbook

**Files:**

- Create: `scripts/phase1b/bootstrap_remote.sh`
- Create: `scripts/phase1b/run_remote.sh`
- Create: `scripts/phase1b/pull_artifacts.sh`
- Create: `tests/scripts/test_phase1b_scripts.py`
- Modify: `README.md`

**Interfaces:**

- Consumes the `trustsr-ldsr-gpu` subcommands from Task 5.
- Produces shell entry points with no embedded host/authentication data:

```text
scripts/phase1b/bootstrap_remote.sh REMOTE_ROOT REPO_DIR
scripts/phase1b/run_remote.sh REMOTE_ROOT preflight|single|benchmark|manifest
scripts/phase1b/pull_artifacts.sh SSH_ALIAS REMOTE_ROOT LOCAL_OUTPUT_DIR
```

- [ ] **Step 1: Add failing script-contract tests**

Tests run `bash -n` and inspect behavior with fake executables placed first in `PATH`. Require:

- `set -euo pipefail` in every script;
- exact argument count and refusal of empty, `/`, `/root`, `~`, glob-containing, or newline-containing paths;
- remote root must resolve under `/root/rivermind-fs/`;
- bootstrap creates Conda prefix `${REMOTE_ROOT}/conda-env` with `--override-channels --channel conda-forge`, explicitly sourcing remote Python packages from conda-forge to avoid ambient default-channel policy/configuration, installs `uv==0.12.5`, and executes frozen `uv sync --extra gpu` against `REPO_DIR` without touching base Conda or accepting channel terms;
- run script accepts only four named stages and invokes one matching CLI command;
- pull script accepts an SSH config alias rather than raw user/host/port/password, uses `rsync --protect-args`, pulls the manifest first, pulls only paths listed in the manifest, and runs local verification;
- no script contains the cloud hostname, port, username, password, `StrictHostKeyChecking=no`, GitHub token, `shutdown`, `poweroff`, `rm -rf`, or `git reset`.

- [ ] **Step 2: Run RED**

```bash
uv run pytest tests/scripts/test_phase1b_scripts.py -q
```

- [ ] **Step 3: Implement bootstrap**

The core commands are:

```bash
conda create --yes --override-channels --channel conda-forge \
  --prefix "${remote_root}/conda-env" python=3.12 pip
"${remote_root}/conda-env/bin/python" -m pip install "uv==0.12.5"
UV_PROJECT_ENVIRONMENT="${remote_root}/conda-env" \
  "${remote_root}/conda-env/bin/uv" sync \
  --directory "${repo_dir}" --frozen --no-dev --extra gpu
```

Before creating anything, assert the data disk has at least 15 GiB free. If the prefix exists, require its `python --version`, uv version and project lock digest to match; otherwise fail with instructions instead of mutating it in place.

- [ ] **Step 4: Implement stage runner and puller**

The runner exports only project paths:

```bash
export TRUSTSR_DATA_CACHE_DIR="${remote_root}/data/opensr"
export TRUSTSR_SEN2SR_MODEL_DIR="${remote_root}/models/sen2srlite"
export TRUSTSR_LDSR_MODEL_DIR="${remote_root}/models/ldsr-s2"
export TRUSTSR_ARTIFACT_ROOT="${remote_root}/artifacts"
```

It calls the requested subcommand with explicit path flags and never enables shell tracing. The puller first retrieves `artifacts/phase1b/artifact-manifest.json`, validates its relative paths locally, transfers exactly them, then executes a small Python one-liner importing `verify_artifact_manifest()` from the local checked-out commit.

- [ ] **Step 5: Document the local/GPU boundary**

README must state:

- server remains off through Tasks 1–6;
- exact command sequence for preflight, single, benchmark and manifest;
- 1.13 GB checkpoint and approximate disk budget;
- deterministic seed/100-step parameters and SPOT development-only limitation;
- artifacts are pulled before the user is told to stop the instance;
- SSH password/key never belongs in commands or the repository;
- cloud shutdown is performed in the provider console only after local digest verification.

- [ ] **Step 6: Verify**

```bash
uv run pytest tests/scripts/test_phase1b_scripts.py -q
uv run pytest -q
uv run ruff check .
bash -n scripts/phase1b/bootstrap_remote.sh
bash -n scripts/phase1b/run_remote.sh
bash -n scripts/phase1b/pull_artifacts.sh
git diff --check
```

- [ ] **Step 7: Commit**

```bash
git add scripts/phase1b tests/scripts/test_phase1b_scripts.py README.md
git commit -m "ops: add reproducible phase1b remote workflow"
```

### Task 7: Local integration review and GPU handoff gate

**Files:**

- Review all files changed from `4532b77f708189d461db5a559dbc0a4a544c1479` to current HEAD.

**Interfaces:**

- Consumes all local tasks.
- Produces a reviewed, pushed branch ready for an exact remote checkout.

- [ ] **Step 1: Verify both dependency modes**

```bash
uv sync --dev
uv run pytest -q
uv run ruff check .
uv run python -c 'import importlib.util; assert importlib.util.find_spec("opensr_model") is None'
uv sync --dev --extra gpu
uv run python -c 'import importlib.metadata as m; assert m.version("opensr-model") == "1.1.1"'
uv run pytest -q
uv sync --dev
git status --short
```

Expected: both suites pass; final default environment excludes the GPU package; worktree is clean.

- [ ] **Step 2: Perform independent reviews**

Run task-scoped reviews after every task, then a whole-branch review against the specification. The final review must inspect checkpoint verification before pickle loading, lazy GPU imports, RNG restoration, cache identity sensitivity, Phase 1A compatibility, no-secret scripts/manifests, raw nine-sample enforcement, and remote path safety. Fix Critical/Important findings with tests before continuing.

- [ ] **Step 3: Stop for the external-side-effect gate**

Report local tests and reviews. Ask the user to:

1. authorize pushing `feature/phase1-pretrained-baselines` and `feature/phase1b-ldsr-gpu` plus creation of a stacked PR;
2. restart the GPU instance;
3. confirm the current SSH host, port and authentication method.

Do not connect, push, publish, configure SSH keys or start paid GPU work before this response.

### Task 8: Execute staged cloud-GPU acceptance

**Files:**

- No tracked code changes are expected. Runtime outputs are ignored under local and remote `artifacts/`, `data/`, and `models/`.

**Interfaces:**

- Consumes the reviewed/pushed branch and user-confirmed SSH configuration.
- Produces locally verified Phase 1B artifacts and the explicit safe-to-stop notice.

- [ ] **Step 1: Establish authenticated SSH without recording secrets**

Confirm the presented ED25519 host fingerprint matches the previously accepted host or ask the user about any change. Generate a host-specific Ed25519 key outside Git only if it does not already exist, install only its public key with the user-authorized password session, verify key login, and instruct the user to rotate the exposed password. Create a local SSH config alias with `IdentityFile`, host, port and user; never include a password.

- [ ] **Step 2: Push exact branches and clone the reviewed commit**

After the Task 7 authorization:

```bash
git push -u origin feature/phase1-pretrained-baselines
git push -u origin feature/phase1b-ldsr-gpu
```

Create the stacked PR against `feature/phase1-pretrained-baselines`. On the remote data disk, clone/fetch the public repository and checkout the exact reviewed commit. Verify `git rev-parse HEAD` locally, on GitHub and remotely are identical before bootstrap.

- [ ] **Step 3: Run preflight only**

First confirm `/root/rivermind-fs` is an actual mountpoint; otherwise wait and do
not run any command.

```bash
scripts/phase1b/bootstrap_remote.sh /root/rivermind-fs/trustsr-phase1b /root/rivermind-fs/trustsr-phase1b/repo
scripts/phase1b/run_remote.sh /root/rivermind-fs/trustsr-phase1b preflight
```

Inspect the manifest for CUDA availability, exactly one visible GPU, a numeric `major.minor` compute capability of at least `8.0`, at least 18 GiB free VRAM before construction, the recorded actual name/UUID/driver/memory, CUDA/Python/PyTorch/package versions, config/checkpoint hashes, exact Git commit, cgroup limits and absence of credentials. Stop on any mismatch.

- [ ] **Step 4: Run the single-sample gate**

```bash
scripts/phase1b/run_remote.sh /root/rivermind-fs/trustsr-phase1b single
```

Require `spot-0000`, two fresh 100-step predictions, exact hashes or `max_abs_diff <= 1e-6`, finite clipped `(4,512,512)` output, finite metrics, recorded duration/peak VRAM, and verified prediction-cache replay. Compare duration only within the recorded same hardware; quality and repeatability outputs remain comparable. Stop on failure.

- [ ] **Step 5: Run the full benchmark twice**

```bash
scripts/phase1b/run_remote.sh /root/rivermind-fs/trustsr-phase1b benchmark
sha256sum /root/rivermind-fs/trustsr-phase1b/artifacts/phase1b/spot-v3-three-models.json
scripts/phase1b/run_remote.sh /root/rivermind-fs/trustsr-phase1b benchmark
sha256sum /root/rivermind-fs/trustsr-phase1b/artifacts/phase1b/spot-v3-three-models.json
```

Hashes must match. Inspect exactly nine samples/model, one shared manifest hash, finite metrics and complete provenance. Snapshot cache filenames, sizes, hashes and modification times before the second run; require no current entry changes.

- [ ] **Step 6: Build and pull the artifact manifest**

```bash
scripts/phase1b/run_remote.sh /root/rivermind-fs/trustsr-phase1b manifest
scripts/phase1b/pull_artifacts.sh trustsr-phase1b /root/rivermind-fs/trustsr-phase1b artifacts/remote/phase1b
```

Re-run local `verify_artifact_manifest()` and inspect every JSON for finite values and prohibited credential/path fields. Do not delete remote artifacts.

- [ ] **Step 7: Final acceptance review**

Review the runtime manifests/results against every Stage 2–5 specification gate. If code changes are required, commit/review/push them and repeat all affected GPU stages. If no code changes are needed, record exact commands, hashes, GPU resource usage and metrics in the handoff response without committing runtime artifacts.

- [ ] **Step 8: Notify the user**

Only after local artifact hash verification, explicitly say the GPU instance can be stopped in the provider console. Report that stopping via SSH was intentionally not attempted and that data under `/root/rivermind-fs/trustsr-phase1b` must be retained until the user decides otherwise.
