# Phase 0 Foundation and Bicubic Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, CPU-only command that loads a small public OpenSR-Test SPOT development sample, performs RGBN ×4 bicubic interpolation, and writes reproducible OpenSR metrics to JSON.

**Architecture:** A small `trustsr` Python package separates the data contract, OpenSR-Test loading, model inference, evaluation, and CLI orchestration. This phase deliberately contains no learned model and no conformal code; it proves the data and metric path before adding expensive components.

**Tech Stack:** Python 3.12, uv, PyTorch, NumPy, opensr-test 1.3.3, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-26-trustworthy-sentinel2-sr-roadmap.md`

## Global Constraints

- Use Python `>=3.12,<3.13`; the available interpreter is Python 3.12.3.
- Use public data only.
- Use Sentinel-2 L2A bands in exact order `(B04, B03, B02, B08)` from L2A indices `(3, 2, 1, 7)`.
- Represent reflectance as `torch.float32` in `[0, 1]`.
- Keep the task fixed at RGBN ×4; do not add ×2 or 12-band behavior.
- Treat OpenSR-Test SPOT as development smoke data only, never as a paper headline result.
- Do not download NAIP, Spain Urban, or Spain Crops during this phase.
- Do not add SEN2SR, LDSR-S2, conformal calibration, training, notebooks, Hydra, Lightning, databases, or web dashboards.
- Do not commit `data/`, `artifacts/`, model weights, caches, virtual environments, or generated figures.
- Run all test and application commands through `uv run`.

---

## File Structure

```text
RemoteSensing001/
├── .gitignore                         # Excludes data, artifacts, caches, weights
├── README.md                          # Phase-0 setup and one reproducible command
├── pyproject.toml                     # Python package and pinned direct dependencies
├── src/trustsr/
│   ├── __init__.py                    # Package version
│   ├── contracts.py                   # Validated SRPair data contract
│   ├── cli/
│   │   ├── __init__.py
│   │   └── smoke_baseline.py          # Phase-0 CLI and JSON provenance
│   ├── data/
│   │   ├── __init__.py
│   │   └── opensr.py                  # OpenSR-Test RGBN loader
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── opensr_metrics.py          # opensr-test metric adapter
│   └── models/
│       ├── __init__.py
│       └── bicubic.py                 # Zero-parameter ×4 baseline
└── tests/
    ├── test_contracts.py
    ├── test_opensr_data.py
    ├── test_opensr_metrics.py
    ├── test_package.py
    ├── test_smoke_cli.py
    └── models/test_bicubic.py
```

### Task 1: Bootstrap the repository and reproducible Python environment

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `pyproject.toml`
- Create: `src/trustsr/__init__.py`
- Create: `tests/test_package.py`

**Interfaces:**
- Consumes: Python 3.12.3 and uv 0.12.5 installed on the host.
- Produces: importable package `trustsr` with `trustsr.__version__ == "0.1.0"`.

- [ ] **Step 1: Initialize version control**

Run:

```bash
git init
```

Expected: an empty Git repository is initialized in `/home/wanghongxu/code/RemoteSensing001`.

- [ ] **Step 2: Create the package metadata and ignore rules**

Create `pyproject.toml`:

```toml
[project]
name = "trustsr"
version = "0.1.0"
description = "Risk-controlled Sentinel-2 super-resolution experiments"
requires-python = ">=3.12,<3.13"
dependencies = [
  "numpy>=2.0,<3",
  "opensr-test==1.3.3",
  "pandas>=2.2,<3",
  "torch>=2.4,<3",
]

[project.scripts]
trustsr-smoke = "trustsr.cli.smoke_baseline:main"

[dependency-groups]
dev = [
  "pytest>=8.3,<9",
  "pytest-cov>=5,<7",
  "ruff>=0.9,<1",
]

[build-system]
requires = ["hatchling>=1.27,<2"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/trustsr"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

Create `.gitignore`:

```gitignore
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
*.py[cod]
data/
artifacts/
models/
checkpoints/
*.ckpt
*.pth
*.pt
```

Create `README.md`:

````markdown
# TrustSR

Incremental experiments for trustworthy Sentinel-2 RGBN ×4 super-resolution.

## Phase 0

```bash
uv sync --dev
uv run pytest
uv run trustsr-smoke --dataset spot --limit 2
```

The SPOT run is a development smoke test, not a final scientific result.
````

Create the package directory and a buildable scaffold in `src/trustsr/__init__.py`:

```python
"""Trustworthy Sentinel-2 super-resolution experiments."""
```

- [ ] **Step 3: Write the failing package test**

Create `tests/test_package.py`:

```python
def test_package_version() -> None:
    import trustsr

    assert trustsr.__version__ == "0.1.0"
```

- [ ] **Step 4: Install dependencies and verify the test fails**

Run:

```bash
uv sync --dev
uv run pytest tests/test_package.py -v
```

Expected: FAIL with `AttributeError: module 'trustsr' has no attribute '__version__'`.

- [ ] **Step 5: Add the minimal package**

Create `src/trustsr/__init__.py`:

```python
"""Trustworthy Sentinel-2 super-resolution experiments."""

__version__ = "0.1.0"
```

- [ ] **Step 6: Verify package import and linting**

Run:

```bash
uv run pytest tests/test_package.py -v
uv run ruff check .
```

Expected: both commands PASS.

- [ ] **Step 7: Commit the bootstrap**

```bash
git add .gitignore README.md pyproject.toml uv.lock src/trustsr/__init__.py tests/test_package.py docs/superpowers
git commit -m "chore: bootstrap trustsr experiment package"
```

### Task 2: Define and validate the SR pair contract

**Files:**
- Create: `src/trustsr/contracts.py`
- Create: `tests/test_contracts.py`

**Interfaces:**
- Consumes: `torch.Tensor` images in channel-first layout.
- Produces: `SRPair(sample_id: str, source: str, lr: Tensor, hr: Tensor, scale: int)` and `SRPair.validate() -> None`.

- [ ] **Step 1: Write tests for a valid RGBN ×4 pair and invalid shapes**

Create `tests/test_contracts.py`:

```python
import pytest
import torch

from trustsr.contracts import SRPair


def test_valid_rgbn_x4_pair() -> None:
    pair = SRPair(
        sample_id="spot-0",
        source="opensr-test/spot/v3",
        lr=torch.rand(4, 16, 16),
        hr=torch.rand(4, 64, 64),
        scale=4,
    )

    pair.validate()


@pytest.mark.parametrize(
    ("lr", "hr", "message"),
    [
        (torch.rand(3, 16, 16), torch.rand(4, 64, 64), "four RGBN channels"),
        (torch.rand(4, 16, 16), torch.rand(4, 63, 64), "exactly scale"),
        (torch.full((4, 16, 16), 1.1), torch.rand(4, 64, 64), "reflectance"),
    ],
)
def test_invalid_pair_is_rejected(
    lr: torch.Tensor, hr: torch.Tensor, message: str
) -> None:
    pair = SRPair("bad", "fixture", lr, hr, 4)

    with pytest.raises(ValueError, match=message):
        pair.validate()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
uv run pytest tests/test_contracts.py -v
```

Expected: FAIL because `trustsr.contracts` does not exist.

- [ ] **Step 3: Implement the immutable data contract**

Create `src/trustsr/contracts.py`:

```python
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SRPair:
    sample_id: str
    source: str
    lr: torch.Tensor
    hr: torch.Tensor
    scale: int

    def validate(self) -> None:
        if self.scale != 4:
            raise ValueError("Phase 0 supports scale=4 only")
        if self.lr.ndim != 3 or self.hr.ndim != 3:
            raise ValueError("lr and hr must use channel-first CHW layout")
        if self.lr.shape[0] != 4 or self.hr.shape[0] != 4:
            raise ValueError("lr and hr must have four RGBN channels")
        expected = (self.lr.shape[1] * self.scale, self.lr.shape[2] * self.scale)
        if self.hr.shape[1:] != expected:
            raise ValueError("hr height and width must be exactly scale times lr")
        for name, tensor in (("lr", self.lr), ("hr", self.hr)):
            if tensor.dtype != torch.float32:
                raise ValueError(f"{name} must use torch.float32")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite reflectance")
            if tensor.min().item() < 0.0 or tensor.max().item() > 1.0:
                raise ValueError(f"{name} reflectance must be in [0, 1]")
```

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
uv run pytest tests/test_contracts.py -v
uv run pytest
```

Expected: all tests PASS.

- [ ] **Step 5: Commit the contract**

```bash
git add src/trustsr/contracts.py tests/test_contracts.py
git commit -m "feat: define validated RGBN super-resolution pair"
```

### Task 3: Add the deterministic bicubic ×4 baseline

**Files:**
- Create: `src/trustsr/models/__init__.py`
- Create: `src/trustsr/models/bicubic.py`
- Create: `tests/models/test_bicubic.py`

**Interfaces:**
- Consumes: `BicubicX4.predict(lr: torch.Tensor)`, where `lr` has shape `(4, H, W)`.
- Produces: a detached CPU `torch.float32` tensor with shape `(4, 4H, 4W)` and values in `[0, 1]`.

- [ ] **Step 1: Write model behavior tests**

Create `tests/models/test_bicubic.py`:

```python
import torch

from trustsr.models.bicubic import BicubicX4


def test_bicubic_predicts_rgbn_x4_deterministically() -> None:
    lr = torch.linspace(0, 1, 4 * 8 * 9, dtype=torch.float32).reshape(4, 8, 9)
    model = BicubicX4()

    first = model.predict(lr)
    second = model.predict(lr)

    assert first.shape == (4, 32, 36)
    assert first.dtype == torch.float32
    assert first.min() >= 0 and first.max() <= 1
    assert torch.equal(first, second)
    assert not first.requires_grad
```

- [ ] **Step 2: Verify the test fails**

Run:

```bash
uv run pytest tests/models/test_bicubic.py -v
```

Expected: FAIL because `trustsr.models.bicubic` does not exist.

- [ ] **Step 3: Implement the baseline**

Create an empty `src/trustsr/models/__init__.py` and create `src/trustsr/models/bicubic.py`:

```python
import torch
import torch.nn.functional as functional


class BicubicX4:
    name = "bicubic-x4"
    scale = 4

    @torch.inference_mode()
    def predict(self, lr: torch.Tensor) -> torch.Tensor:
        if lr.ndim != 3 or lr.shape[0] != 4:
            raise ValueError("expected RGBN tensor with shape (4, H, W)")
        sr = functional.interpolate(
            lr.unsqueeze(0),
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        return sr.to(dtype=torch.float32, device="cpu").clamp_(0.0, 1.0)
```

- [ ] **Step 4: Run tests and lint**

Run:

```bash
uv run pytest tests/models/test_bicubic.py -v
uv run ruff check src/trustsr/models tests/models
```

Expected: both commands PASS.

- [ ] **Step 5: Commit the baseline**

```bash
git add src/trustsr/models tests/models
git commit -m "feat: add deterministic bicubic x4 baseline"
```

### Task 4: Load OpenSR-Test RGBN pairs without band-order ambiguity

**Files:**
- Create: `src/trustsr/data/__init__.py`
- Create: `src/trustsr/data/opensr.py`
- Create: `tests/test_opensr_data.py`

**Interfaces:**
- Consumes: `load_opensr_pairs(dataset_name: str, cache_dir: Path, version: str, limit: int) -> list[SRPair]`.
- Produces: validated `SRPair` objects with L2A indices `(3, 2, 1, 7)` and HRharm channels already ordered RGBN.

- [ ] **Step 1: Write a loader test using an in-memory upstream fixture**

Create `tests/test_opensr_data.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from trustsr.data.opensr import L2A_RGBN_INDICES, load_opensr_pairs


def test_loader_selects_b04_b03_b02_b08(monkeypatch, tmp_path: Path) -> None:
    l2a = np.zeros((2, 12, 3, 5), dtype=np.uint16)
    for band in range(12):
        l2a[:, band] = (band + 1) * 100
    hr = np.full((2, 4, 12, 20), 500, dtype=np.uint16)

    monkeypatch.setattr(
        "opensr_test.load",
        lambda dataset, model_dir, version: {"L2A": l2a, "HRharm": hr},
    )

    pairs = load_opensr_pairs("spot", tmp_path, "v3", limit=1)

    assert L2A_RGBN_INDICES == (3, 2, 1, 7)
    assert len(pairs) == 1
    assert pairs[0].sample_id == "spot-0000"
    assert pairs[0].lr[:, 0, 0].tolist() == pytest.approx([0.04, 0.03, 0.02, 0.08])
    assert pairs[0].hr.shape == (4, 12, 20)
```

- [ ] **Step 2: Verify the loader test fails**

Run:

```bash
uv run pytest tests/test_opensr_data.py -v
```

Expected: FAIL because `trustsr.data.opensr` does not exist.

- [ ] **Step 3: Implement the loader and explicit band mapping**

Create an empty `src/trustsr/data/__init__.py` and create `src/trustsr/data/opensr.py`:

```python
from pathlib import Path

import numpy as np
import opensr_test
import torch

from trustsr.contracts import SRPair

L2A_RGBN_INDICES = (3, 2, 1, 7)
REFLECTANCE_SCALE = 10_000.0


def _to_reflectance(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array).copy()).to(torch.float32).div_(
        REFLECTANCE_SCALE
    ).clamp_(0.0, 1.0)


def load_opensr_pairs(
    dataset_name: str,
    cache_dir: Path,
    version: str = "v3",
    limit: int = 2,
) -> list[SRPair]:
    if dataset_name != "spot":
        raise ValueError("Phase 0 allows the SPOT development dataset only")
    if limit < 1:
        raise ValueError("limit must be positive")

    cache_dir.mkdir(parents=True, exist_ok=True)
    raw = opensr_test.load(dataset_name, model_dir=str(cache_dir), version=version)
    count = min(limit, len(raw["L2A"]))
    pairs: list[SRPair] = []
    for index in range(count):
        pair = SRPair(
            sample_id=f"{dataset_name}-{index:04d}",
            source=f"opensr-test/{dataset_name}/{version}",
            lr=_to_reflectance(
                np.take(raw["L2A"][index], L2A_RGBN_INDICES, axis=0)
            ),
            hr=_to_reflectance(raw["HRharm"][index]),
            scale=4,
        )
        pair.validate()
        pairs.append(pair)
    return pairs
```

- [ ] **Step 4: Run loader and full tests**

Run:

```bash
uv run pytest tests/test_opensr_data.py -v
uv run pytest
```

Expected: all tests PASS without downloading external data because the test replaces `opensr_test.load`.

- [ ] **Step 5: Commit the loader**

```bash
git add src/trustsr/data tests/test_opensr_data.py
git commit -m "feat: load explicit RGBN pairs from opensr test"
```

### Task 5: Wrap OpenSR evaluation behind a stable local interface

**Files:**
- Create: `src/trustsr/evaluation/__init__.py`
- Create: `src/trustsr/evaluation/opensr_metrics.py`
- Create: `tests/test_opensr_metrics.py`

**Interfaces:**
- Consumes: `compute_opensr_metrics(pair: SRPair, sr: Tensor) -> dict[str, float]`.
- Produces: exactly seven scalar keys: `reflectance`, `spectral`, `spatial`, `synthesis`, `ha_metric`, `om_metric`, `im_metric`.

- [ ] **Step 1: Write a fast adapter test with an upstream test double**

Create `tests/test_opensr_metrics.py`:

```python
import torch

from trustsr.contracts import SRPair
from trustsr.evaluation.opensr_metrics import METRIC_KEYS, compute_opensr_metrics


def test_metric_adapter_returns_plain_floats(monkeypatch) -> None:
    expected = {name: index / 10 for index, name in enumerate(METRIC_KEYS)}

    class FakeMetrics:
        def __init__(self, **kwargs) -> None:
            assert kwargs == {"device": "cpu"}

        def compute(self, *, lr, sr, hr, gradient_threshold):
            assert lr.shape == (4, 4, 4)
            assert sr.shape == hr.shape == (4, 16, 16)
            assert gradient_threshold == "auto"
            return expected

    monkeypatch.setattr("opensr_test.Metrics", FakeMetrics)
    pair = SRPair("fixture", "unit", torch.rand(4, 4, 4), torch.rand(4, 16, 16), 4)

    result = compute_opensr_metrics(pair, torch.rand(4, 16, 16))

    assert result == expected
    assert all(type(value) is float for value in result.values())
```

- [ ] **Step 2: Verify the adapter test fails**

Run:

```bash
uv run pytest tests/test_opensr_metrics.py -v
```

Expected: FAIL because `trustsr.evaluation.opensr_metrics` does not exist.

- [ ] **Step 3: Implement the stable adapter**

Create an empty `src/trustsr/evaluation/__init__.py` and create `src/trustsr/evaluation/opensr_metrics.py`:

```python
import opensr_test
import torch

from trustsr.contracts import SRPair

METRIC_KEYS = (
    "reflectance",
    "spectral",
    "spatial",
    "synthesis",
    "ha_metric",
    "om_metric",
    "im_metric",
)


def compute_opensr_metrics(pair: SRPair, sr: torch.Tensor) -> dict[str, float]:
    pair.validate()
    if sr.shape != pair.hr.shape:
        raise ValueError("sr shape must match hr shape")
    if not torch.isfinite(sr).all():
        raise ValueError("sr contains non-finite values")

    evaluator = opensr_test.Metrics(device="cpu")
    raw = evaluator.compute(
        lr=pair.lr.cpu(),
        sr=sr.detach().to(torch.float32).cpu(),
        hr=pair.hr.cpu(),
        gradient_threshold="auto",
    )
    return {key: float(raw[key]) for key in METRIC_KEYS}
```

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
uv run pytest tests/test_opensr_metrics.py -v
uv run pytest
uv run ruff check .
```

Expected: all commands PASS.

- [ ] **Step 5: Commit the evaluation adapter**

```bash
git add src/trustsr/evaluation tests/test_opensr_metrics.py
git commit -m "feat: add stable opensr metric adapter"
```

### Task 6: Add the end-to-end smoke command and provenance JSON

**Files:**
- Create: `src/trustsr/cli/__init__.py`
- Create: `src/trustsr/cli/smoke_baseline.py`
- Create: `tests/test_smoke_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: CLI options `--dataset`, `--version`, `--limit`, `--cache-dir`, and `--output`.
- Produces: JSON schema with `run`, `samples`, and `mean_metrics`; `run.dataset_role` must equal `development_smoke_only`.

- [ ] **Step 1: Write an orchestration test that performs no network access**

Create `tests/test_smoke_cli.py`:

```python
import json
from pathlib import Path

import torch

from trustsr.cli.smoke_baseline import run
from trustsr.contracts import SRPair


def test_run_writes_reproducible_json(monkeypatch, tmp_path: Path) -> None:
    pair = SRPair("spot-0000", "fixture", torch.rand(4, 4, 4), torch.rand(4, 16, 16), 4)
    monkeypatch.setattr(
        "trustsr.cli.smoke_baseline.load_opensr_pairs",
        lambda dataset_name, cache_dir, version, limit: [pair],
    )
    monkeypatch.setattr(
        "trustsr.cli.smoke_baseline.compute_opensr_metrics",
        lambda pair, sr: {
            "reflectance": 0.1,
            "spectral": 0.2,
            "spatial": 0.3,
            "synthesis": 0.4,
            "ha_metric": 0.5,
            "om_metric": 0.6,
            "im_metric": 0.7,
        },
    )
    output = tmp_path / "result.json"

    result = run("spot", "v3", 1, tmp_path / "cache", output)

    assert json.loads(output.read_text()) == result
    assert result["run"]["dataset_role"] == "development_smoke_only"
    assert result["run"]["bands"] == ["B04", "B03", "B02", "B08"]
    assert len(result["run"]["sample_manifest_sha256"]) == 64
    assert result["samples"][0]["sample_id"] == "spot-0000"
    assert result["mean_metrics"]["ha_metric"] == 0.5
```

- [ ] **Step 2: Verify the orchestration test fails**

Run:

```bash
uv run pytest tests/test_smoke_cli.py -v
```

Expected: FAIL because `trustsr.cli.smoke_baseline` does not exist.

- [ ] **Step 3: Implement the CLI and JSON provenance**

Create an empty `src/trustsr/cli/__init__.py` and create `src/trustsr/cli/smoke_baseline.py`:

```python
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import opensr_test
import torch

from trustsr.data.opensr import load_opensr_pairs
from trustsr.evaluation.opensr_metrics import METRIC_KEYS, compute_opensr_metrics
from trustsr.models.bicubic import BicubicX4


def _git_commit() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else "unavailable"


def run(
    dataset: str,
    version: str,
    limit: int,
    cache_dir: Path,
    output: Path,
) -> dict:
    pairs = load_opensr_pairs(dataset, cache_dir, version, limit)
    model = BicubicX4()
    samples = []
    for pair in pairs:
        sr = model.predict(pair.lr)
        samples.append(
            {"sample_id": pair.sample_id, "metrics": compute_opensr_metrics(pair, sr)}
        )

    manifest = "\n".join(f"{pair.source}:{pair.sample_id}" for pair in pairs)
    means = {
        key: sum(item["metrics"][key] for item in samples) / len(samples)
        for key in METRIC_KEYS
    }
    result = {
        "run": {
            "dataset": dataset,
            "dataset_version": version,
            "dataset_role": "development_smoke_only",
            "sample_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
            "model": model.name,
            "scale": 4,
            "bands": ["B04", "B03", "B02", "B08"],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "opensr_test": opensr_test.__version__,
            "device": "cpu",
            "git_commit": _git_commit(),
        },
        "samples": samples,
        "mean_metrics": means,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase-0 bicubic smoke test")
    parser.add_argument("--dataset", default="spot", choices=["spot"])
    parser.add_argument("--version", default="v3", choices=["v3"])
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/opensr-test"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/phase0/bicubic-spot-v3.json")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args.dataset, args.version, args.limit, args.cache_dir, args.output)
    print(json.dumps(result["mean_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests and static checks before downloading data**

Run:

```bash
uv run pytest
uv run ruff check .
```

Expected: all tests and checks PASS.

- [ ] **Step 5: Run the real two-sample SPOT smoke test**

Run:

```bash
uv run trustsr-smoke --dataset spot --version v3 --limit 2
```

Expected:

- One public SPOT pickle is downloaded to `data/cache/opensr-test/spot.pkl` on first run.
- The process exits with code 0 on CPU.
- `artifacts/phase0/bicubic-spot-v3.json` contains two samples and seven finite mean metrics.
- No NAIP or Spain dataset is downloaded.

- [ ] **Step 6: Verify deterministic replay**

Run twice:

```bash
cp artifacts/phase0/bicubic-spot-v3.json /tmp/trustsr-first-run.json
uv run trustsr-smoke --dataset spot --version v3 --limit 2
diff -u /tmp/trustsr-first-run.json artifacts/phase0/bicubic-spot-v3.json
```

Expected: `diff` prints no differences.

- [ ] **Step 7: Document the output and phase boundary**

Append to `README.md`:

```markdown
The command writes `artifacts/phase0/bicubic-spot-v3.json`. Generated artifacts and
downloaded data are intentionally untracked. Phase 0 is complete only when tests,
linting, the two-sample run, and deterministic replay all pass. Learned models are
introduced in a separate Phase-1 plan after this checkpoint is reviewed.
```

- [ ] **Step 8: Commit the working Phase-0 vertical slice**

```bash
git add README.md src/trustsr/cli tests/test_smoke_cli.py
git commit -m "feat: add reproducible phase zero smoke pipeline"
```

## Phase-0 Acceptance Gate

Run the final gate:

```bash
uv run pytest
uv run ruff check .
uv run trustsr-smoke --dataset spot --version v3 --limit 2
git status --short
```

Accept Phase 0 only when:

- tests and Ruff pass;
- the command succeeds without CUDA;
- both samples have shape-valid RGBN inputs and outputs;
- all seven metrics in the JSON are finite numbers;
- replay is byte-for-byte deterministic;
- Git status contains no downloaded data or generated artifact;
- no learned model or final test dataset has entered the implementation.

After acceptance, write a separate Phase-1 plan for the SEN2SRLite adapter, cached inference artifacts, and a fair comparison against this frozen bicubic baseline.
