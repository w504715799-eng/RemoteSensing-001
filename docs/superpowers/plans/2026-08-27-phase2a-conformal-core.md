# Phase 2A Conformal Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the deterministic CPU-only conformal fidelity-mask core used as the non-novel baseline for later multispectral Sentinel-2 risk research.

**Architecture:** Keep local risk/score construction, calibration, selective evaluation, and CLI orchestration in separate modules. Calibration treats each complete ROI as one exchangeable item, searches only observed finite score thresholds, and returns `-inf` for valid all-abstain outcomes. A deterministic synthetic CLI proves the end-to-end contract without downloading data or invoking an SR model.

**Tech Stack:** Python 3.12, PyTorch, standard-library dataclasses/JSON, pytest, Ruff, uv.

**Spec:** `docs/superpowers/specs/2026-08-27-phase2a-conformal-core.md`

## Global Constraints

- Use Python `>=3.12,<3.13`; do not add a runtime dependency.
- CPU only; do not import `opensr_model`, require CUDA, access SSH, or download a dataset.
- Inputs are four-band reflectance tensors in `[0, 1]`; fail closed on invalid shape, dtype values, NaN, Inf, or out-of-range values.
- A calibration item is an ROI tensor, never an individual pixel.
- Use `torch.float64` for risk, score, threshold-search, and evaluation arithmetic.
- Do not claim novelty or cross-sensor coverage; the CLI must emit `"synthetic_smoke": true`.
- Preserve all existing Phase 0/1/1B behavior and the default CPU dependency boundary.
- Use TDD and commit after every independently reviewable task.

---

### Task 1: Add local fidelity risk and ensemble-variance score

**Files:**
- Create: `src/trustsr/risk/__init__.py`
- Create: `src/trustsr/risk/local.py`
- Create: `tests/risk/test_local.py`

**Interfaces:**
- Consumes: `torch.Tensor` values only.
- Produces: `local_l1_risk(sr, hr, *, window) -> torch.Tensor` and `ensemble_variance_score(samples) -> torch.Tensor`, both returning finite `torch.float64` `(H, W)` tensors.

- [ ] **Step 1: Write failing happy-path tests**

```python
import torch

from trustsr.risk.local import ensemble_variance_score, local_l1_risk


def test_local_l1_risk_window_one_is_band_mean_absolute_error() -> None:
    sr = torch.tensor(
        [
            [[0.0, 0.2], [0.4, 0.6]],
            [[0.1, 0.3], [0.5, 0.7]],
            [[0.2, 0.4], [0.6, 0.8]],
            [[0.3, 0.5], [0.7, 0.9]],
        ]
    )
    hr = torch.zeros_like(sr)

    result = local_l1_risk(sr, hr, window=1)

    expected = torch.tensor([[0.15, 0.35], [0.55, 0.75]], dtype=torch.float64)
    torch.testing.assert_close(result, expected)


def test_ensemble_variance_score_uses_population_variance_and_band_mean() -> None:
    first = torch.zeros((4, 2, 2))
    second = torch.ones((4, 2, 2))

    result = ensemble_variance_score(torch.stack((first, second)))

    torch.testing.assert_close(result, torch.full((2, 2), 0.25, dtype=torch.float64))
```

- [ ] **Step 2: Run the happy-path tests and verify import failure**

Run: `uv run pytest tests/risk/test_local.py -v`
Expected: FAIL during collection with `ModuleNotFoundError: No module named 'trustsr.risk'`.

- [ ] **Step 3: Write failing validation and reflection-padding tests**

Add parameterized cases that assert `ValueError` for non-`(4,H,W)` SR/HR tensors, mismatched shapes, non-finite values, values outside `[0,1]`, even/zero/oversized windows, fewer than two ensemble members, and non-`(K,4,H,W)` samples. Add this exact reflection example:

```python
def test_local_l1_risk_uses_reflection_padding() -> None:
    sr = torch.zeros((4, 3, 3))
    sr[:, 0, 0] = 1.0
    result = local_l1_risk(sr, torch.zeros_like(sr), window=3)
    expected = torch.tensor(
        [[1 / 9, 1 / 9, 0.0], [1 / 9, 1 / 9, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result, expected)
```

- [ ] **Step 4: Implement the minimal risk module**

In `local.py`, add private validators for `(4,H,W)` and `(K,4,H,W)` reflectance tensors. Convert validated inputs to `float64`; compute `abs(sr-hr).mean(dim=0)`. For `window > 1`, call `torch.nn.functional.pad` with `mode="reflect"`, then `avg_pool2d` with stride 1. Compute variance with `samples.to(torch.float64).var(dim=0, correction=0).mean(dim=0)`. Export both functions from `risk/__init__.py`.

- [ ] **Step 5: Run focused and full validation**

Run: `uv run pytest tests/risk/test_local.py -v`
Expected: PASS.
Run: `uv run pytest && uv run ruff check .`
Expected: all tests and Ruff PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/trustsr/risk tests/risk
git commit -m "feat: add conformal risk score primitives"
```

---

### Task 2: Add ROI-level conformal calibration and trusted masks

**Files:**
- Create: `src/trustsr/calibration/__init__.py`
- Create: `src/trustsr/calibration/conformal.py`
- Create: `tests/calibration/test_conformal.py`

**Interfaces:**
- Consumes: equal-length sequences of finite, same-shaped, two-dimensional score/risk tensors; `alpha`; risk upper bound.
- Produces: immutable `ConformalCalibration`, `calibrate_fidelity_mask(...)`, and `trusted_mask(...)` exactly as specified.

- [ ] **Step 1: Write a failing hand-calculated threshold test**

```python
import pytest
import torch

from trustsr.calibration.conformal import calibrate_fidelity_mask


def test_calibration_uses_roi_maxima_and_finite_sample_correction() -> None:
    scores = (
        torch.tensor([[0.1, 0.2]], dtype=torch.float64),
        torch.tensor([[0.1, 0.3]], dtype=torch.float64),
    )
    risks = (
        torch.tensor([[0.1, 0.4]], dtype=torch.float64),
        torch.tensor([[0.2, 0.9]], dtype=torch.float64),
    )

    result = calibrate_fidelity_mask(scores, risks, alpha=0.55)

    assert result.threshold == pytest.approx(0.2)
    assert result.risk_bound == pytest.approx((0.4 + 0.2 + 1.0) / 3.0)
    assert result.calibration_size == 2
    assert result.trusted_pixels == 3
    assert result.total_pixels == 4
```

- [ ] **Step 2: Run the hand-calculated test and verify import failure**

Run: `uv run pytest tests/calibration/test_conformal.py::test_calibration_uses_roi_maxima_and_finite_sample_correction -v`
Expected: FAIL during collection because `trustsr.calibration` is absent.

- [ ] **Step 3: Write failing all-abstain, ROI-unit, and mask tests**

Add tests proving:

```python
def test_alpha_below_finite_sample_correction_abstains_everywhere() -> None:
    result = calibrate_fidelity_mask(
        (torch.tensor([[0.1]]),),
        (torch.tensor([[0.0]]),),
        alpha=0.49,
    )
    assert result.threshold == float("-inf")
    assert result.trusted_pixels == 0
    assert result.risk_bound == pytest.approx(0.5)


def test_pixels_are_not_treated_as_independent_calibration_items() -> None:
    result = calibrate_fidelity_mask(
        (torch.tensor([[0.1, 0.2]]),),
        (torch.tensor([[0.1, 0.9]]),),
        alpha=0.6,
    )
    assert result.calibration_size == 1
    assert result.threshold == pytest.approx(0.1)
```

Also assert that `trusted_mask` returns `torch.bool`, uses `<=`, rejects non-finite score maps, and returns all false for `-inf`.

- [ ] **Step 4: Write failing input-contract tests**

Parameterize errors for empty/mismatched sequences, non-2D maps, unequal map shapes within an ROI, NaN/Inf, negative scores, risks outside `[0,risk_upper_bound]`, `alpha` outside `(0,risk_upper_bound]`, and a non-positive/non-finite upper bound. Assert exact stable error fragments such as `"scores and risks must be non-empty"`, `"ROI shapes must match"`, and `"risk exceeds risk_upper_bound"`.

- [ ] **Step 5: Implement deterministic threshold search**

Implement the frozen dataclass. Validate without mutating caller tensors, convert maps to CPU `float64`, sort the union of unique observed scores, and for every candidate compute each ROI's maximum risk among `score <= candidate`, using zero for an empty selection. Compute `(sum(worst) + risk_upper_bound)/(n+1)` and retain the largest passing threshold. If no candidate passes, return `threshold=-inf`, `trusted_pixels=0`, and `risk_bound=risk_upper_bound/(n+1)`. Recompute trusted/total pixels and the selected bound for the returned object. Export the public names from `calibration/__init__.py`.

- [ ] **Step 6: Run focused and full validation**

Run: `uv run pytest tests/calibration/test_conformal.py -v`
Expected: PASS.
Run: `uv run pytest && uv run ruff check .`
Expected: all tests and Ruff PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add src/trustsr/calibration tests/calibration
git commit -m "feat: add ROI-level conformal calibration"
```

---

### Task 3: Add empirical selective-risk evaluation

**Files:**
- Create: `src/trustsr/evaluation/selective.py`
- Modify: `src/trustsr/evaluation/__init__.py`
- Create: `tests/evaluation/test_selective.py`

**Interfaces:**
- Consumes: validated score/risk ROI sequences and a finite threshold or `-inf`.
- Produces: immutable `SelectivePoint` and `evaluate_selective_point(...)` with pixel coverage and mean ROI-maximum risk.

- [ ] **Step 1: Write failing hand-calculated evaluation tests**

```python
import pytest
import torch

from trustsr.evaluation.selective import evaluate_selective_point


def test_selective_point_reports_pixel_coverage_and_mean_roi_maximum() -> None:
    scores = (
        torch.tensor([[0.1, 0.4]]),
        torch.tensor([[0.2, 0.3]]),
    )
    risks = (
        torch.tensor([[0.2, 0.8]]),
        torch.tensor([[0.3, 0.7]]),
    )

    result = evaluate_selective_point(scores, risks, threshold=0.2)

    assert result.coverage == pytest.approx(0.5)
    assert result.roi_max_risk == pytest.approx(0.25)
```

Add an all-abstain test expecting coverage and risk `0.0`, and a monotone construction whose lower threshold has no larger risk than its higher threshold.

- [ ] **Step 2: Run tests and verify import failure**

Run: `uv run pytest tests/evaluation/test_selective.py -v`
Expected: FAIL because `trustsr.evaluation.selective` is absent.

- [ ] **Step 3: Add validation tests**

Reuse the public behavior—not private functions—to reject empty/mismatched sequences, invalid shapes, non-finite/negative scores, risks outside `[0,1]`, and `threshold=NaN` or `+inf`. Explicitly accept `threshold=-inf` as the all-abstain sentinel.

- [ ] **Step 4: Implement selective evaluation**

Implement the frozen dataclass and local validation. For each ROI use `mask = score <= threshold`; accumulate trusted pixel count; append `risk[mask].max()` or zero for an empty mask. Return total trusted pixels divided by total pixels and the arithmetic mean of per-ROI maxima, both as Python floats. Export the two public names from `evaluation/__init__.py` without removing current exports.

- [ ] **Step 5: Run focused and full validation**

Run: `uv run pytest tests/evaluation/test_selective.py -v`
Expected: PASS.
Run: `uv run pytest && uv run ruff check .`
Expected: all tests and Ruff PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/trustsr/evaluation tests/evaluation
git commit -m "feat: evaluate conformal risk coverage"
```

---

### Task 4: Add deterministic synthetic conformal smoke CLI

**Files:**
- Create: `src/trustsr/cli/conformal_smoke.py`
- Create: `tests/cli/test_conformal_smoke.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

**Interfaces:**
- Consumes: CLI flags `--alpha` (default `0.27`) and `--window` (default `1`).
- Produces: one canonical JSON document on stdout with schema `trustsr.conformal-smoke.v1` and no external files.

- [ ] **Step 1: Write failing parser and payload tests**

```python
import json

from trustsr.cli.conformal_smoke import main, run


def test_run_marks_result_as_synthetic_and_uses_roi_calibration() -> None:
    result = run(alpha=0.27, window=1)
    assert result["schema"] == "trustsr.conformal-smoke.v1"
    assert result["synthetic_smoke"] is True
    assert result["config"] == {"alpha": 0.27, "channels": 4, "scale": 4, "window": 1}
    assert result["calibration"]["calibration_size"] == 3
    assert 0.0 <= result["test"]["coverage"] <= 1.0


def test_main_prints_canonical_json(capsys) -> None:
    assert main(["--alpha", "0.27", "--window", "1"]) == 0
    output = capsys.readouterr().out
    assert output.endswith("\n")
    assert output == json.dumps(json.loads(output), sort_keys=True, separators=(",", ":")) + "\n"
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `uv run pytest tests/cli/test_conformal_smoke.py -v`
Expected: FAIL because `trustsr.cli.conformal_smoke` is absent.

- [ ] **Step 3: Write failing determinism and validation tests**

Call `main(...)` twice and assert byte-identical stdout. Assert invalid `alpha` and even/zero window values cause `argparse` exit code 2. Monkeypatch `torch.cuda.is_available` to return true and prove the module still constructs only CPU tensors. Assert the payload has exactly top-level keys `schema`, `synthetic_smoke`, `config`, `calibration`, and `test`.

- [ ] **Step 4: Implement the fixed synthetic experiment**

Use this exact deterministic construction; it makes ensemble variance increase with reconstruction error without calling a random-number API:

```python
base = torch.linspace(0.1, 0.7, 64, dtype=torch.float64).reshape(4, 4, 4)
pattern = torch.linspace(0.0, 1.0, 16, dtype=torch.float64).reshape(1, 4, 4)
pattern = pattern.expand(4, -1, -1)

hrs: list[torch.Tensor] = []
sample_sets: list[torch.Tensor] = []
predictions: list[torch.Tensor] = []
for index in range(5):
    hr = torch.roll(base, shifts=(index % 3, (2 * index) % 3), dims=(-2, -1))
    hr = hr + 0.01 * index
    spread = 0.005 * (index + 1) * (1.0 + pattern)
    center = hr + 2.0 * spread
    samples = torch.stack((center - spread, center + spread))
    hrs.append(hr)
    sample_sets.append(samples)
    predictions.append(samples.mean(dim=0))
```

For every ROI compute risk from `predictions[index]` and `hrs[index]`, and score from
`sample_sets[index]`. Use the first three ROIs for calibration and the last two for test. Compute
one `SelectivePoint` on the calibration ROIs and one on the test ROIs using the fitted threshold.
Build this exact JSON shape from JSON-native scalars:

```python
payload = {
    "schema": "trustsr.conformal-smoke.v1",
    "synthetic_smoke": True,
    "config": {"alpha": alpha, "channels": 4, "scale": 4, "window": window},
    "calibration": {
        "calibration_size": calibration.calibration_size,
        "coverage": calibration_point.coverage,
        "risk_bound": calibration.risk_bound,
        "roi_max_risk": calibration_point.roi_max_risk,
        "threshold": calibration.threshold,
    },
    "test": {
        "coverage": test_point.coverage,
        "roi_count": 2,
        "roi_max_risk": test_point.roi_max_risk,
    },
}
```

Serialize it with
`json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"`.

- [ ] **Step 5: Register and document the CLI**

Add this exact entry point to `pyproject.toml`:

```toml
trustsr-conformal-smoke = "trustsr.cli.conformal_smoke:main"
```

Add a README section titled `Phase 2A synthetic conformal smoke` with:

```bash
uv run trustsr-conformal-smoke --alpha 0.27 --window 1
```

State directly that this command uses no real satellite data and provides no paper evidence or cross-sensor guarantee.

- [ ] **Step 6: Run focused tests and prove byte repeatability**

Run: `uv run pytest tests/cli/test_conformal_smoke.py -v`
Expected: PASS.
Run:

```bash
uv run trustsr-conformal-smoke > /tmp/trustsr-phase2a-first.json
uv run trustsr-conformal-smoke > /tmp/trustsr-phase2a-second.json
cmp /tmp/trustsr-phase2a-first.json /tmp/trustsr-phase2a-second.json
```

Expected: `cmp` exits 0.

- [ ] **Step 7: Run full acceptance**

Run: `uv run pytest`
Expected: all existing and new tests PASS.
Run: `uv run ruff check .`
Expected: PASS.
Run: `git diff --check`
Expected: no output and exit 0.

- [ ] **Step 8: Commit Task 4**

```bash
git add pyproject.toml README.md src/trustsr/cli/conformal_smoke.py tests/cli/test_conformal_smoke.py
git commit -m "feat: add synthetic conformal smoke workflow"
```

---

### Task 5: Perform Phase 2A acceptance and freeze the Phase 2B gate

**Files:**
- Create: `docs/reports/phase2a-conformal-core.md`

**Interfaces:**
- Consumes: the four completed implementation commits and their command outputs.
- Produces: a concise acceptance report with commit, environment, test counts, deterministic output SHA-256, observed calibration/test values, and explicit non-claims.

- [ ] **Step 1: Run final verification from a clean shell**

Run:

```bash
uv sync --dev
uv run pytest
uv run ruff check .
git diff --check
uv run trustsr-conformal-smoke > /tmp/trustsr-phase2a-accepted.json
sha256sum /tmp/trustsr-phase2a-accepted.json
git status --short
```

Expected: sync succeeds; tests/Ruff/diff check pass; SHA-256 is recorded; only the not-yet-committed report is absent from status at this point.

- [ ] **Step 2: Write the acceptance report with observed values**

The report must include the exact commands and outputs from Step 1, the current Git commit, the JSON payload, and these boundaries verbatim:

```text
Phase 2A verifies only the deterministic CPU implementation of a published generic
conformal SR baseline. It is not evidence of novelty, real Sentinel-2 performance,
cross-sensor validity, or downstream utility. Phase 2B must use geographically
separated ROI-level data and must treat external sensor shift as an empirical audit,
not as an automatic conformal guarantee.
```

- [ ] **Step 3: Commit the acceptance report**

```bash
git add docs/reports/phase2a-conformal-core.md
git commit -m "docs: record phase2a conformal acceptance"
```

- [ ] **Step 4: Verify the final branch state**

Run: `git status --short --branch`
Expected: clean `feature/phase2a-conformal-core` branch.
Run: `git log --oneline --decorate -5`
Expected: the Task 1–5 commits are visible in order.
