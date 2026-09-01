# Phase 2B3-A Development Score Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a leak-safe, development-only audit that compares LDSR `K=5` variance, LR reprojection residual, and three-model disagreement against frozen local L1 risk, then deterministically freezes one eligible score.

**Architecture:** Keep tensor scoring, ROI diagnostics, candidate selection, artifact caching, prediction orchestration, and CLI staging in separate modules. A0 develops every contract with synthetic CPU fixtures; A1 runs a four-ROI `K=25` stability gate; only a passing A1 permits the 120-ROI A2 audit. Calibration and internal-test pixels remain unreachable from this workflow.

**Tech Stack:** Python 3.12, PyTorch, NumPy, SciPy, safetensors, rasterio, pytest 8, Ruff, uv, Bash, CUDA-enabled LDSR-S2 on the cloud base environment.

**Spec:** `docs/superpowers/specs/2026-09-01-phase2b3a-development-score-audit-design.md`

## Global Constraints

- Base all work on commit `2455c1e07686076561b181c753cc61a5ed440222` plus the approved spec commit; do not implement on `main` or alter Phase 2B2-B evidence.
- Use only manifest `c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a` and Phase 2B2-A's frozen input contract.
- A1 uses exactly the four canonical development ROI. A2 uses exactly all 120 development ROI and all 12 strata; no arbitrary ID or limit flag is allowed.
- Never load calibration or internal-test pixels in this phase. Never fit a conformal threshold or emit a trusted mask.
- Evaluate only LDSR seed 3407 against `local_l1_risk(window=9)`; `window=1` is descriptive and cannot affect selection.
- Freeze seeds `S5A=3407..3411`, `S5B=3412..3416`, and `S25=3407..3431`. Never scale `K=25` beyond A1 under this plan.
- Compute all score maps and statistics on CPU in `torch.float64`; GPU output remains cached `torch.float32`.
- Use 10,000 ROI bootstrap resamples from `numpy.random.Generator(PCG64(23031))` and fixed coverages `0.1..1.0`.
- Keep all TIFFs, models, predictions, and score tensors on the mounted cloud storage. Only verified small canonical JSON may enter Git.
- Use the cloud image's base Python/PyTorch. Do not create a conda environment and do not replace the PyTorch/CUDA stack.
- Do not pin a GPU product name. Require one visible CUDA device, compute capability at least 8.0, at least 18 GiB initial free VRAM, and no foreign compute process.
- Do not write an SSH host, password, token, private key, username, absolute local path, or cloud hostname into committed files.
- Every product-code task follows RED → GREEN → focused tests → Ruff → commit. Do not combine commits listed below.

## File Structure

- `src/trustsr/data/crosssensor_pairs.py`: exact 120-development-record selection and validation.
- `src/trustsr/risk/proxies.py`: the two deterministic score maps; LDSR variance continues to use `risk.local.ensemble_variance_score`.
- `src/trustsr/evaluation/score_diagnostics.py`: one-ROI rank, risk-coverage, AURC, and high-risk miss diagnostics.
- `src/trustsr/evaluation/score_selection.py`: ROI bootstrap, eligibility, strata checks, paired comparisons, and cost-aware freezing.
- `src/trustsr/artifacts/scores.py`: identity-bound, atomic score-map storage and integrity verification.
- `src/trustsr/models/ldsr_s2.py`: immutable seeded views that share one verified LDSR backend.
- `src/trustsr/evaluation/development_predictions.py`: prediction-grid generation and cache reuse for fixed models/seeds.
- `src/trustsr/evaluation/development_score_audit.py`: A1/A2 result construction and inference-free replay.
- `src/trustsr/cli/phase2b3a.py`: strict preflight, single, smoke, replay, development, and development-replay stages.
- `src/trustsr/cli/phase2b3a_verify.py`: local-only validation of pulled A1/A2 evidence bundles.
- `scripts/phase2b3a/run_cloud.sh`: mounted-storage/base-Python execution wrapper.
- `scripts/phase2b3a/pull_results.sh`: allowlisted small-result transfer and local digest verification.
- `docs/phase2b3a-cloud-runbook.md`: operator checkpoints, cost projections, commands, and shutdown handoff.

---

### Task 1: Select Exactly 120 Development ROI

**Files:**

- Modify: `src/trustsr/data/crosssensor_pairs.py`
- Modify: `tests/data/test_crosssensor_pairs.py`

**Interfaces:**

- Consumes: `Sequence[Mapping[str, object]]` returned by `load_crosssensor_records`.
- Produces: `select_development_records(records) -> tuple[Mapping[str, object], ...]` in manifest order.

- [ ] **Step 1: Write failing selection tests**

Add a 360-record synthetic fixture and these assertions:

```python
def _complete_research_records() -> tuple[dict[str, object], ...]:
    rows = []
    for split in ("development", "calibration", "internal_test"):
        for days_between in (-1, 0, 1):
            for bin_index in range(4):
                for selection_round in range(1, 11):
                    sample_id = f"{split}-{days_between}-{bin_index}-{selection_round}"
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "split": split,
                            "spatial_group_id": f"group-{sample_id}",
                            "days_between": days_between,
                            "correlation_bin": bin_index,
                            "selection_round": selection_round,
                        }
                    )
    return tuple(rows)


def test_select_development_records_preserves_manifest_order_and_strata() -> None:
    records = _complete_research_records()
    selected = select_development_records(records)
    assert len(selected) == 120
    assert selected == tuple(row for row in records if row["split"] == "development")
    assert len({row["spatial_group_id"] for row in selected}) == 120


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate_sample", "duplicate_group", "bad_day", "bad_bin", "bad_round"],
)
def test_select_development_records_rejects_broken_frozen_cells(mutation: str) -> None:
    records = [dict(row) for row in _complete_research_records()]
    development_index = next(i for i, row in enumerate(records) if row["split"] == "development")
    if mutation == "missing":
        records.pop(development_index)
    elif mutation == "duplicate_sample":
        records[development_index + 1]["sample_id"] = records[development_index]["sample_id"]
    elif mutation == "duplicate_group":
        records[development_index + 1]["spatial_group_id"] = records[development_index]["spatial_group_id"]
    elif mutation == "bad_day":
        records[development_index]["days_between"] = 2
    elif mutation == "bad_bin":
        records[development_index]["correlation_bin"] = 4
    else:
        records[development_index]["selection_round"] = 11
    with pytest.raises(ValueError, match="development"):
        select_development_records(records)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run pytest tests/data/test_crosssensor_pairs.py -k select_development_records -q`  
Expected: import/collection failure because `select_development_records` is absent.

- [ ] **Step 3: Implement exact-cell validation**

Add the function below, using `_require_unique_strings` for both identifiers:

```python
DEVELOPMENT_DAYS = (-1, 0, 1)
DEVELOPMENT_BINS = (0, 1, 2, 3)
DEVELOPMENT_ROUNDS = tuple(range(1, 11))


def select_development_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    selected = tuple(record for record in records if record.get("split") == "development")
    if len(selected) != 120:
        raise ValueError("development selection must contain exactly 120 records")
    _require_unique_strings(selected, "sample_id")
    _require_unique_strings(selected, "spatial_group_id")
    cells: dict[tuple[int, int], list[int]] = {
        (day, bin_index): []
        for day in DEVELOPMENT_DAYS
        for bin_index in DEVELOPMENT_BINS
    }
    for record in selected:
        day = _require_integer(record, "days_between")
        bin_index = _require_integer(record, "correlation_bin")
        selection_round = _require_integer(record, "selection_round")
        if (day, bin_index) not in cells:
            raise ValueError("development record has an invalid stratum")
        cells[(day, bin_index)].append(selection_round)
    if any(tuple(sorted(rounds)) != DEVELOPMENT_ROUNDS for rounds in cells.values()):
        raise ValueError("development strata must each contain selection rounds 1 through 10")
    return selected
```

- [ ] **Step 4: Verify GREEN and lint**

Run: `uv run pytest tests/data/test_crosssensor_pairs.py -q`  
Expected: all tests pass.

Run: `uv run ruff check src/trustsr/data/crosssensor_pairs.py tests/data/test_crosssensor_pairs.py`  
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/trustsr/data/crosssensor_pairs.py tests/data/test_crosssensor_pairs.py
git commit -m "feat: select phase2b3a development records"
```

### Task 2: Implement Deterministic Score Proxies

**Files:**

- Create: `src/trustsr/risk/proxies.py`
- Create: `tests/risk/test_proxies.py`
- Modify: `src/trustsr/risk/__init__.py`

**Interfaces:**

- Consumes: validated cached `torch.float32` RGBN tensors.
- Produces:
  - `lr_reprojection_l1_score(prediction, lr, *, scale=4) -> torch.Tensor`
  - `three_model_disagreement_score(predictions) -> torch.Tensor`

- [ ] **Step 1: Write failing hand-calculated tests**

```python
def test_lr_reprojection_l1_score_uses_area_and_constant_4x4_blocks() -> None:
    prediction = torch.zeros((4, 8, 8), dtype=torch.float32)
    prediction[:, :4, :4] = 0.4
    lr = torch.zeros((4, 2, 2), dtype=torch.float32)
    expected = torch.zeros((8, 8), dtype=torch.float64)
    expected[:4, :4] = 0.4
    actual = lr_reprojection_l1_score(prediction, lr, scale=4)
    assert actual.dtype == torch.float64
    assert actual.device.type == "cpu"
    assert torch.equal(actual, expected)


def test_three_model_disagreement_is_population_variance_over_models_and_bands() -> None:
    predictions = tuple(
        torch.full((4, 2, 3), value, dtype=torch.float32)
        for value in (0.0, 0.5, 1.0)
    )
    actual = three_model_disagreement_score(predictions)
    assert torch.allclose(actual, torch.full((2, 3), 1.0 / 6.0, dtype=torch.float64))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_score_proxies_reject_invalid_reflectance(bad: float) -> None:
    prediction = torch.full((4, 8, 8), bad, dtype=torch.float32)
    lr = torch.zeros((4, 2, 2), dtype=torch.float32)
    with pytest.raises(ValueError):
        lr_reprojection_l1_score(prediction, lr, scale=4)
```

Also test non-tensors, wrong channel count, non-float32 input, mismatched shapes, scale other than 4,
fewer/more than three model predictions, input mutation, and returned contiguity.

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/risk/test_proxies.py -q`  
Expected: module import failure.

- [ ] **Step 3: Implement the score functions**

```python
from collections.abc import Sequence

import torch
from torch.nn import functional as F

from trustsr.risk.local import ensemble_variance_score


def _require_rgbn(value: torch.Tensor, *, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        raise ValueError(f"{name} must be a torch.float32 tensor")
    if value.ndim != 3 or value.shape[0] != 4 or min(value.shape[1:]) <= 0:
        raise ValueError(f"{name} must have shape (4, H, W)")
    if not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError(f"{name} must contain finite reflectance in [0, 1]")


def lr_reprojection_l1_score(
    prediction: torch.Tensor,
    lr: torch.Tensor,
    *,
    scale: int = 4,
) -> torch.Tensor:
    _require_rgbn(prediction, name="prediction")
    _require_rgbn(lr, name="lr")
    if type(scale) is not int or scale != 4:
        raise ValueError("scale must equal 4")
    if prediction.shape[1:] != (lr.shape[1] * scale, lr.shape[2] * scale):
        raise ValueError("prediction spatial shape must be four times lr")
    prediction64 = prediction.detach().to(device="cpu", dtype=torch.float64)
    lr64 = lr.detach().to(device="cpu", dtype=torch.float64)
    projected = F.interpolate(
        prediction64.unsqueeze(0), size=lr.shape[1:], mode="area"
    ).squeeze(0)
    residual = (projected - lr64).abs().mean(dim=0)
    return residual.repeat_interleave(scale, 0).repeat_interleave(scale, 1).contiguous()


def three_model_disagreement_score(
    predictions: Sequence[torch.Tensor],
) -> torch.Tensor:
    if len(predictions) != 3:
        raise ValueError("three_model_disagreement requires exactly three predictions")
    for index, prediction in enumerate(predictions):
        _require_rgbn(prediction, name=f"predictions[{index}]")
    if len({tuple(prediction.shape) for prediction in predictions}) != 1:
        raise ValueError("all model predictions must have matching shapes")
    stacked = torch.stack(
        [item.detach().to(device="cpu") for item in predictions], dim=0
    )
    return ensemble_variance_score(stacked).contiguous()
```

- [ ] **Step 4: Verify GREEN and lint**

Run: `uv run pytest tests/risk/test_local.py tests/risk/test_proxies.py -q`  
Expected: pass.

Run: `uv run ruff check src/trustsr/risk tests/risk`  
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/trustsr/risk tests/risk/test_proxies.py
git commit -m "feat: add phase2b3a uncertainty proxies"
```

### Task 3: Compute One-ROI Score Diagnostics

**Files:**

- Create: `src/trustsr/evaluation/score_diagnostics.py`
- Create: `tests/evaluation/test_score_diagnostics.py`

**Interfaces:**

- Consumes: one finite nonnegative score map and one `[0,1]` risk map with identical shape.
- Produces: immutable `RoiScoreDiagnostics` and `score_map_spearman`/`top_fraction_jaccard` helpers.

- [ ] **Step 1: Write failing exact-statistic tests**

```python
def test_evaluate_roi_score_uses_stable_low_score_coverage() -> None:
    score = torch.tensor([[0.0, 0.0], [2.0, 3.0]], dtype=torch.float64)
    risk = torch.tensor([[0.1, 0.2], [0.8, 0.9]], dtype=torch.float64)
    result = evaluate_roi_score(score, risk, coverages=(0.5, 1.0))
    assert result.constant_score is False
    assert result.coverages == (0.5, 1.0)
    assert result.selective_mean_risks == pytest.approx((0.15, 0.5))
    assert result.aurc == pytest.approx(0.325)
    assert result.random_aurc == pytest.approx(0.5)
    assert result.aurc_gain == pytest.approx(0.175)


def test_constant_score_has_zero_spearman_and_row_major_tie_break() -> None:
    score = torch.zeros((2, 2), dtype=torch.float64)
    risk = torch.tensor([[0.4, 0.3], [0.2, 0.1]], dtype=torch.float64)
    result = evaluate_roi_score(score, risk, coverages=(0.5, 1.0))
    assert result.rho == 0.0
    assert result.constant_score is True
    assert result.selective_mean_risks[0] == pytest.approx(0.35)


def test_top_fraction_jaccard_uses_exact_count_and_row_major_ties() -> None:
    first = torch.tensor([[4.0, 3.0], [2.0, 1.0]], dtype=torch.float64)
    second = torch.tensor([[4.0, 1.0], [3.0, 2.0]], dtype=torch.float64)
    assert top_fraction_jaccard(first, second, fraction=0.5) == pytest.approx(1.0 / 3.0)
```

Also hand-check average-rank Spearman with ties, 80%-coverage high-risk miss rate, invalid coverages,
NaN/Inf, negative score, out-of-range risk, unequal shapes, and input mutation.

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/evaluation/test_score_diagnostics.py -q`  
Expected: module import failure.

- [ ] **Step 3: Implement deterministic diagnostics**

Define this public record:

```python
@dataclass(frozen=True)
class RoiScoreDiagnostics:
    rho: float
    constant_score: bool
    coverages: tuple[float, ...]
    selective_mean_risks: tuple[float, ...]
    aurc: float
    random_aurc: float
    aurc_gain: float
    high_risk_miss_rate_at_80: float
```

Implement average ranks with `scipy.stats.rankdata(method="average")`, correlation with NumPy after
ranking, and stable pixel selection with `np.lexsort((np.arange(size), score_flat))`. Use
`ceil(coverage * size)` pixels. Define the highest-risk 10% and lowest-score 80% using exact counts
and row-major tie breaking. A constant score returns `rho=0.0`; a constant risk also returns
`rho=0.0`. Convert every returned scalar through `float` and reject non-finite results.

The central function must have this signature and default:

```python
DEFAULT_COVERAGES = tuple(index / 10 for index in range(1, 11))


def evaluate_roi_score(
    score: torch.Tensor,
    risk: torch.Tensor,
    *,
    coverages: tuple[float, ...] = DEFAULT_COVERAGES,
) -> RoiScoreDiagnostics:
    score_values, risk_values = _validated_flat_arrays(score, risk)
    order = np.lexsort((np.arange(score_values.size), score_values))
    selected_risks = tuple(
        float(risk_values[order[: math.ceil(coverage * order.size)]].mean())
        for coverage in coverages
    )
    random_aurc = float(risk_values.mean())
    aurc = float(np.mean(selected_risks))
    return RoiScoreDiagnostics(
        rho=_spearman(score_values, risk_values),
        constant_score=bool(np.ptp(score_values) == 0.0),
        coverages=coverages,
        selective_mean_risks=selected_risks,
        aurc=aurc,
        random_aurc=random_aurc,
        aurc_gain=random_aurc - aurc,
        high_risk_miss_rate_at_80=_high_risk_miss(score_values, risk_values),
    )
```

- [ ] **Step 4: Verify GREEN and lint**

Run: `uv run pytest tests/evaluation/test_score_diagnostics.py -q`  
Expected: pass.

Run: `uv run ruff check src/trustsr/evaluation/score_diagnostics.py tests/evaluation/test_score_diagnostics.py`  
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/trustsr/evaluation/score_diagnostics.py tests/evaluation/test_score_diagnostics.py
git commit -m "feat: add roi score diagnostics"
```

### Task 4: Bootstrap, Qualify, and Freeze Candidates

**Files:**

- Create: `src/trustsr/evaluation/score_selection.py`
- Create: `tests/evaluation/test_score_selection.py`

**Interfaces:**

- Consumes: exactly 120 `DevelopmentRoiResult` records per candidate and fixed stratum fields.
- Produces: `CandidateSummary`, `FrozenScore`, and deterministic bootstrap indices.

- [ ] **Step 1: Write failing bootstrap and eligibility tests**

Use compact scalar fixtures, not pixel maps:

```python
def _candidate(rho: float, gain: float) -> list[DevelopmentRoiResult]:
    return [
        DevelopmentRoiResult(
            sample_id=f"sample-{day}-{bin_index}-{round_index}",
            spatial_group_id=f"group-{day}-{bin_index}-{round_index}",
            days_between=day,
            correlation_bin=bin_index,
            selection_round=round_index,
            rho=rho,
            constant_score=False,
            aurc_gain=gain,
            high_risk_miss_rate_at_80=0.5,
        )
        for day in (-1, 0, 1)
        for bin_index in range(4)
        for round_index in range(1, 11)
    ]


def test_bootstrap_indices_are_fixed_roi_resamples() -> None:
    first = build_bootstrap_indices()
    second = build_bootstrap_indices()
    assert first.shape == (10_000, 120)
    assert first.dtype == np.int64
    assert np.array_equal(first, second)
    assert int(first.min()) >= 0 and int(first.max()) < 120


def test_candidate_must_pass_all_five_eligibility_rules() -> None:
    summary = summarize_candidate("lr_reprojection_l1", _candidate(0.2, 0.03))
    assert summary.eligible is True
    assert summary.failure_reasons == ()


def test_freeze_prefers_cheapest_statistically_indistinguishable_score() -> None:
    candidates = {
        "lr_reprojection_l1": _candidate(0.20, 0.03),
        "three_model_disagreement": _candidate(0.20, 0.03),
        "ldsr_variance_k5": _candidate(0.20, 0.03),
    }
    frozen = freeze_score(candidates)
    assert frozen.name == "lr_reprojection_l1"
    assert frozen.cost_rank == 0
```

Add independent tests for 113 nonconstant ROI, lower CI equal to zero, only 8 positive strata, one
stratum below `-0.10`, a missing ROI, duplicated sample/group, mismatched candidate membership, a
clearly superior expensive candidate, and zero eligible candidates.

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/evaluation/test_score_selection.py -q`  
Expected: module import failure.

- [ ] **Step 3: Define immutable records and fixed constants**

```python
BOOTSTRAP_SEED = 23031
BOOTSTRAP_RESAMPLES = 10_000
COST_ORDER = (
    "lr_reprojection_l1",
    "three_model_disagreement",
    "ldsr_variance_k5",
)


@dataclass(frozen=True)
class DevelopmentRoiResult:
    sample_id: str
    spatial_group_id: str
    days_between: int
    correlation_bin: int
    selection_round: int
    rho: float
    constant_score: bool
    aurc_gain: float
    high_risk_miss_rate_at_80: float


@dataclass(frozen=True)
class CandidateSummary:
    name: str
    eligible: bool
    failure_reasons: tuple[str, ...]
    nonconstant_count: int
    mean_rho: float
    mean_rho_ci95: tuple[float, float]
    mean_aurc_gain: float
    mean_aurc_gain_ci95: tuple[float, float]
    positive_strata: int
    minimum_stratum_mean_rho: float


@dataclass(frozen=True)
class FrozenScore:
    name: str
    cost_rank: int
    statistical_leader: str
    indistinguishable_candidates: tuple[str, ...]
    candidate_summaries: tuple[CandidateSummary, ...]
```

- [ ] **Step 4: Implement paired bootstrap and exact selection**

`build_bootstrap_indices()` creates one `(10000,120)` array and all summaries/comparisons reuse it.
Use `np.percentile(values, (2.5, 97.5))`. Validate exact 12×10 cells before computing statistics.
Eligibility implements all spec rules verbatim. For freezing, find the eligible maximum `mean_rho`,
bootstrap `leader_rho - candidate_rho` with paired rows, remove a candidate only when the 2.5th
percentile is strictly greater than zero, then choose the lowest `COST_ORDER.index(name)`.

Use these public signatures:

```python
def build_bootstrap_indices() -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    return rng.integers(0, 120, size=(BOOTSTRAP_RESAMPLES, 120), dtype=np.int64)


def summarize_candidate(
    name: str,
    results: Sequence[DevelopmentRoiResult],
    *,
    bootstrap_indices: np.ndarray | None = None,
) -> CandidateSummary:
    indices = build_bootstrap_indices() if bootstrap_indices is None else bootstrap_indices
    return _summarize_validated_candidate(name, tuple(results), indices)


def freeze_score(
    candidates: Mapping[str, Sequence[DevelopmentRoiResult]],
) -> FrozenScore:
    indices = build_bootstrap_indices()
    validated = _validate_matching_candidate_membership(candidates)
    summaries = tuple(
        summarize_candidate(name, validated[name], bootstrap_indices=indices)
        for name in COST_ORDER
        if name in validated
    )
    eligible = tuple(summary for summary in summaries if summary.eligible)
    if not eligible:
        raise ValueError("no development score candidate is eligible")
    leader = max(eligible, key=lambda item: item.mean_rho)
    indistinguishable = _paired_indistinguishable(leader, eligible, validated, indices)
    chosen = min(indistinguishable, key=lambda name: COST_ORDER.index(name))
    return FrozenScore(
        name=chosen,
        cost_rank=COST_ORDER.index(chosen),
        statistical_leader=leader.name,
        indistinguishable_candidates=tuple(
            name for name in COST_ORDER if name in indistinguishable
        ),
        candidate_summaries=summaries,
    )
```

- [ ] **Step 5: Verify GREEN and lint**

Run: `uv run pytest tests/evaluation/test_score_selection.py -q`  
Expected: pass.

Run: `uv run ruff check src/trustsr/evaluation/score_selection.py tests/evaluation/test_score_selection.py`  
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/trustsr/evaluation/score_selection.py tests/evaluation/test_score_selection.py
git commit -m "feat: freeze eligible development score"
```

### Task 5: Add an Identity-Bound Score Cache

**Files:**

- Create: `src/trustsr/artifacts/scores.py`
- Create: `tests/artifacts/test_scores.py`

**Interfaces:**

- Consumes: score name/version, sample ID, ordered input tensor SHA-256 values, scalar operator parameters, and a `float64 (H,W)` score.
- Produces: immutable `ScoreIdentity`, atomic `ScoreCache.put/get`, and file evidence.

- [ ] **Step 1: Write failing identity and integrity tests**

```python
def _identity() -> ScoreIdentity:
    return ScoreIdentity(
        score_name="lr_reprojection_l1",
        score_schema_version=1,
        sample_id="development-0",
        input_sha256s=("a" * 64, "b" * 64),
        operator_parameters={"scale": 4},
    )


def test_score_identity_changes_for_input_order_or_operator() -> None:
    original = _identity()
    reversed_inputs = replace(original, input_sha256s=tuple(reversed(original.input_sha256s)))
    changed_scale = replace(original, operator_parameters={"scale": 2})
    assert len({original.key, reversed_inputs.key, changed_scale.key}) == 3


def test_score_cache_round_trip_is_detached_float64(tmp_path: Path) -> None:
    score = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    cache = ScoreCache(tmp_path)
    cache.put(_identity(), score)
    loaded = cache.get(_identity())
    assert loaded is not None
    assert loaded.dtype == torch.float64
    assert loaded.device.type == "cpu"
    assert torch.equal(loaded, score)
    assert loaded.data_ptr() != score.data_ptr()
```

Also test mutable nested parameter rejection, invalid digest/name/version/sample, float32, negative,
NaN/Inf, 1-D/3-D/empty scores, symlinks, one-sided entries, altered JSON, altered tensor bytes,
wrong tensor SHA, unexpected tensor key, atomic temporary cleanup, and no overwrite of valid bytes.

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/artifacts/test_scores.py -q`  
Expected: module import failure.

- [ ] **Step 3: Implement identity and atomic cache**

Use `canonical_json`, `safetensors.torch.save_file/load_file`, `NamedTemporaryFile`, `fsync`, and
`os.replace`. The committed JSON sidecar is the final commit marker. Define:

```python
SCORE_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ScoreIdentity:
    score_name: str
    score_schema_version: int
    sample_id: str
    input_sha256s: tuple[str, ...]
    operator_parameters: Mapping[str, JsonScalar]

    def __post_init__(self) -> None:
        parameters = _validated_scalar_mapping(self.operator_parameters)
        object.__setattr__(self, "operator_parameters", MappingProxyType(parameters))
        _validate_score_identity_fields(self)

    def as_dict(self) -> dict[str, object]:
        return {
            "score_name": self.score_name,
            "score_schema_version": self.score_schema_version,
            "sample_id": self.sample_id,
            "input_sha256s": list(self.input_sha256s),
            "operator_parameters": dict(self.operator_parameters),
        }

    @property
    def key(self) -> str:
        return hashlib.sha256(canonical_json(self.as_dict())).hexdigest()


class ScoreCache:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, identity: ScoreIdentity, score: torch.Tensor) -> str:
        validated = _validate_score(score)
        return self._atomic_commit(identity, validated)

    def get(self, identity: ScoreIdentity) -> torch.Tensor | None:
        return self._verified_load(identity)
```

Metadata must contain the cache schema, key, exact identity, tensor filename, shape, dtype, and
logical tensor SHA-256. `_verified_load` rejects any deviation before returning a detached contiguous
CPU tensor. Add `score_entry_evidence(root, identity)` that first calls `get`, then streams SHA-256 for
the exact JSON and safetensors basenames.

- [ ] **Step 4: Verify GREEN and lint**

Run: `uv run pytest tests/artifacts/test_scores.py -q`  
Expected: pass.

Run: `uv run ruff check src/trustsr/artifacts/scores.py tests/artifacts/test_scores.py`  
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/trustsr/artifacts/scores.py tests/artifacts/test_scores.py
git commit -m "feat: cache phase2b3a score maps"
```

### Task 6: Reuse One LDSR Backend Across Immutable Seeds

**Files:**

- Modify: `src/trustsr/models/ldsr_s2.py`
- Modify: `tests/models/test_ldsr_s2.py`

**Interfaces:**

- Consumes: a verified `LDSRS2X4` instance and nonnegative integer seed.
- Produces: `LDSRS2X4.for_seed(seed) -> LDSRS2X4` sharing the same backend and frozen settings.

- [ ] **Step 1: Write failing seeded-view tests**

```python
def test_for_seed_shares_backend_and_changes_only_seed_provenance() -> None:
    backend = FakeBackend()
    original = LDSRS2X4(backend, device="cpu", seed=3407)
    seeded = original.for_seed(3412)
    assert seeded is not original
    assert seeded._backend is backend
    assert seeded.seed == 3412
    expected = original.provenance() | {"seed": 3412}
    assert seeded.provenance() == expected


def test_for_seed_predictions_are_repeatable_and_seed_distinct() -> None:
    backend = SeedSensitiveFakeBackend()
    model = LDSRS2X4(backend, device="cpu")
    first = model.for_seed(3408).predict(_input())
    replay = model.for_seed(3408).predict(_input())
    other = model.for_seed(3409).predict(_input())
    assert torch.equal(first, replay)
    assert not torch.equal(first, other)
```

Add invalid bool, negative, float, and oversized-for-NumPy seeds. Assert the original provenance and
prediction remain unchanged after creating views.

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/models/test_ldsr_s2.py -k for_seed -q`  
Expected: `AttributeError` for `for_seed`.

- [ ] **Step 3: Implement immutable seeded views**

Reuse `_validate_configuration` and preserve every current parameter:

```python
def for_seed(self, seed: int) -> LDSRS2X4:
    return type(self)(
        self._backend,
        device=self.device,
        seed=seed,
        sampling_steps=self.sampling_steps,
        sampling_eta=self.sampling_eta,
        sampling_temperature=self.sampling_temperature,
        histogram_matching=self.histogram_matching,
    )
```

Tighten seed validation to `0 <= seed <= np.iinfo(np.uint32).max` because NumPy seeding is part of
the adapter. Do not reload the checkpoint and do not mutate `self.seed`.

- [ ] **Step 4: Verify GREEN and all adapter regressions**

Run: `uv run pytest tests/models/test_ldsr_s2.py tests/models/test_ldsr_backend.py -q`  
Expected: pass.

Run: `uv run ruff check src/trustsr/models/ldsr_s2.py tests/models/test_ldsr_s2.py`  
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/trustsr/models/ldsr_s2.py tests/models/test_ldsr_s2.py
git commit -m "feat: add immutable ldsr seed views"
```

### Task 7: Generate and Reuse the Fixed Prediction Grid

**Files:**

- Create: `src/trustsr/evaluation/development_predictions.py`
- Create: `tests/evaluation/test_development_predictions.py`

**Interfaces:**

- Consumes: one validated development pair, fixed Bicubic/SEN2SRLite/LDSR models, a seed tuple, and `PredictionCache`.
- Produces: immutable `DevelopmentPredictionBundle` with verified cached tensors and identities.

- [ ] **Step 1: Write failing cold/warm grid tests**

```python
def test_prediction_grid_computes_central_models_once_and_all_requested_seeds(
    tmp_path: Path,
) -> None:
    pair = _loaded_development_pair()
    bicubic, sen2srlite, ldsr = _fake_models()
    bundle = load_or_generate_prediction_bundle(
        pair,
        bicubic=bicubic,
        sen2srlite=sen2srlite,
        ldsr=ldsr,
        ldsr_seeds=(3407, 3408, 3409),
        cache=PredictionCache(tmp_path),
    )
    assert tuple(item.seed for item in bundle.ldsr) == (3407, 3408, 3409)
    assert bicubic.calls == 1
    assert sen2srlite.calls == 1
    assert ldsr.calls_by_seed == {3407: 1, 3408: 1, 3409: 1}
    assert bundle.ldsr[0].prediction_sha256 == tensor_sha256(bundle.ldsr[0].tensor)


def test_prediction_grid_warm_run_invokes_no_model(tmp_path: Path) -> None:
    pair = _loaded_development_pair()
    cold = _fake_models()
    first = load_or_generate_prediction_bundle(
        pair,
        bicubic=cold[0],
        sen2srlite=cold[1],
        ldsr=cold[2],
        ldsr_seeds=(3407, 3408),
        cache=PredictionCache(tmp_path),
    )
    warm = _fake_models(fail_on_predict=True)
    second = load_or_generate_prediction_bundle(
        pair,
        bicubic=warm[0],
        sen2srlite=warm[1],
        ldsr=warm[2],
        ldsr_seeds=(3407, 3408),
        cache=PredictionCache(tmp_path),
    )
    assert second == first
```

Also test wrong model order/name/scale, missing seed 3407, duplicate/unsorted seeds, non-development
metadata, context collisions, invalid cached prediction, and post-commit reload differing from model
output.

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/evaluation/test_development_predictions.py -q`  
Expected: module import failure.

- [ ] **Step 3: Define immutable prediction records**

```python
A1_SEEDS = tuple(range(3407, 3432))
K5A_SEEDS = tuple(range(3407, 3412))
K5B_SEEDS = tuple(range(3412, 3417))


@dataclass(frozen=True)
class CachedDevelopmentPrediction:
    model_name: str
    seed: int | None
    identity: PredictionIdentity
    prediction_sha256: str
    tensor: torch.Tensor = field(compare=False, repr=False)


@dataclass(frozen=True)
class DevelopmentPredictionBundle:
    sample_id: str
    bicubic: CachedDevelopmentPrediction
    sen2srlite: CachedDevelopmentPrediction
    ldsr: tuple[CachedDevelopmentPrediction, ...]

    def ldsr_for_seed(self, seed: int) -> CachedDevelopmentPrediction:
        matches = tuple(item for item in self.ldsr if item.seed == seed)
        if len(matches) != 1:
            raise ValueError("prediction bundle does not contain exactly one requested LDSR seed")
        return matches[0]
```

- [ ] **Step 4: Implement cache-bound generation**

Create a Phase 2B3-A cache provenance function that imports the authoritative manifest and input
audit constants and adds `experiment_schema="trustsr.phase2b3a-predictions.v1"`. Reject reserved-key
collisions. For each identity, run this exact sequence:

```python
prediction = cache.get(identity)
if prediction is None:
    produced = model.predict(pair.pair.lr)
    produced_sha256 = tensor_sha256(produced)
    cache.put(identity, produced)
    prediction = cache.get(identity)
    if prediction is None or tensor_sha256(prediction) != produced_sha256:
        raise RuntimeError("prediction differs after cache commit")
return CachedDevelopmentPrediction(
    model_name=model.name,
    seed=seed,
    identity=identity,
    prediction_sha256=tensor_sha256(prediction),
    tensor=prediction,
)
```

Construct Bicubic and SEN2SRLite once. For LDSR call `ldsr.for_seed(seed)` without reconstructing the
backend. The reusable tensor function accepts a strictly increasing tuple containing 3407 so small
unit fixtures remain possible; the CLI is the sole public workflow boundary and accepts only
`A1_SEEDS` or `K5A_SEEDS`.

- [ ] **Step 5: Verify GREEN and lint**

Run: `uv run pytest tests/evaluation/test_development_predictions.py tests/artifacts/test_predictions.py -q`  
Expected: pass.

Run: `uv run ruff check src/trustsr/evaluation/development_predictions.py tests/evaluation/test_development_predictions.py`  
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/trustsr/evaluation/development_predictions.py tests/evaluation/test_development_predictions.py
git commit -m "feat: cache phase2b3a prediction grids"
```

### Task 8: Build the Four-ROI A1 Stability Result and Replay

**Files:**

- Create: `src/trustsr/evaluation/development_score_audit.py`
- Create: `tests/evaluation/test_development_score_audit.py`

**Interfaces:**

- Consumes: exactly four canonical A1 pairs, their 25-seed prediction bundles, and prediction/score caches.
- Produces: `trustsr.phase2b3a-development-smoke.v1`, cache audit, and inference-free replay.

- [ ] **Step 1: Write failing score-construction tests**

```python
def test_build_score_maps_uses_frozen_central_prediction_and_seed_sets(tmp_path: Path) -> None:
    pair = _small_loaded_pair(correlation_bin=0)
    bundle = _synthetic_prediction_bundle(pair, seeds=A1_SEEDS)
    scores = build_a1_score_maps(pair, bundle, ScoreCache(tmp_path))
    assert tuple(item.name for item in scores) == (
        "ldsr_variance_k5a",
        "ldsr_variance_k5b",
        "ldsr_variance_k25",
        "lr_reprojection_l1",
        "three_model_disagreement",
    )
    assert all(item.tensor.dtype == torch.float64 for item in scores)
    assert scores[0].identity.input_sha256s == tuple(
        bundle.ldsr_for_seed(seed).prediction_sha256 for seed in K5A_SEEDS
    )
```

Use small deterministic fake tensors whose K5A/K5B/K25 correlations and top-decile Jaccards are
hand-controlled. Test every threshold boundary (`0.60`, `0.40`, `0.80`, `0.60`, `0.50`), a constant
map, wrong four-ROI order, missing bin, non-development split, score cache warm replay, and R9/R1
remaining distinct.

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/evaluation/test_development_score_audit.py -k 'a1 or score_maps' -q`  
Expected: module import failure.

- [ ] **Step 3: Implement frozen score-map construction**

Define constants once:

```python
A1_RESULT_SCHEMA = "trustsr.phase2b3a-development-smoke.v1"
A1_CACHE_AUDIT_SCHEMA = "trustsr.phase2b3a-development-smoke-cache-audit.v1"
PRIMARY_RISK_WINDOW = 9
SENSITIVITY_RISK_WINDOW = 1


@dataclass(frozen=True)
class CachedScoreMap:
    name: str
    identity: ScoreIdentity
    score_sha256: str
    tensor: torch.Tensor = field(compare=False, repr=False)
```

`build_a1_score_maps` computes LDSR variance with `ensemble_variance_score`, the two deterministic
proxies from Task 2, stores each via `ScoreCache`, reloads it, and verifies the logical SHA before use.
Build `R9` and `R1` from the seed-3407 tensor with `local_l1_risk`.

- [ ] **Step 4: Construct the deterministic A1 result**

Implement:

```python
def evaluate_a1_smoke(
    pairs: Sequence[LoadedCrosssensorPair],
    bundles: Sequence[DevelopmentPredictionBundle],
    score_cache: ScoreCache,
) -> tuple[dict[str, object], dict[str, object]]:
    validated = _validate_a1_inputs(pairs, bundles)
    sample_records = tuple(_evaluate_a1_sample(pair, bundle, score_cache) for pair, bundle in validated)
    k5a_k5b = tuple(record["stability"]["k5a_k5b_spearman"] for record in sample_records)
    k5a_k25 = tuple(record["stability"]["k5a_k25_spearman"] for record in sample_records)
    jaccards = tuple(record["stability"]["k5a_k25_top10_jaccard"] for record in sample_records)
    stable = (
        statistics.median(k5a_k5b) >= 0.60
        and min(k5a_k5b) >= 0.40
        and statistics.median(k5a_k25) >= 0.80
        and min(k5a_k25) >= 0.60
        and statistics.median(jaccards) >= 0.50
    )
    result = _a1_result_payload(sample_records, k5_statistically_stable=stable)
    audit = _score_and_prediction_evidence_payload(validated, score_cache)
    canonical_json(result)
    canonical_json(audit)
    return result, audit
```

The result contains no host, path, runtime, timestamp, or GPU data. It records `K=5` statistical
stability only; resource eligibility is verified separately from the runtime manifest in Task 10.

- [ ] **Step 5: Add inference-free A1 replay**

```python
def replay_a1_smoke(
    pairs: Sequence[LoadedCrosssensorPair],
    committed_result: Mapping[str, object],
    committed_audit: Mapping[str, object],
    prediction_cache: PredictionCache,
    score_cache: ScoreCache,
) -> tuple[dict[str, object], dict[str, object]]:
    identities = _validate_committed_a1_and_rebuild_identities(
        pairs, committed_result, committed_audit
    )
    before = _snapshot_all_cache_files(identities, prediction_cache, score_cache)
    rebuilt = _rebuild_a1_from_caches(pairs, identities, prediction_cache, score_cache)
    after = _snapshot_all_cache_files(identities, prediction_cache, score_cache)
    if before != after:
        raise RuntimeError("cache files changed during A1 replay")
    return rebuilt
```

Tests must pass factories that raise on construction and monkeypatch LDSR imports to raise. Mutate
schemas, sample order, bins, seeds, identity keys, logical tensor SHA, entry counts, and cache mtimes;
every mutation must fail.

- [ ] **Step 6: Verify GREEN and lint**

Run: `uv run pytest tests/evaluation/test_development_score_audit.py -k a1 -q`  
Expected: pass.

Run: `uv run ruff check src/trustsr/evaluation/development_score_audit.py tests/evaluation/test_development_score_audit.py`  
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add src/trustsr/evaluation/development_score_audit.py tests/evaluation/test_development_score_audit.py
git commit -m "feat: evaluate phase2b3a a1 stability"
```

### Task 9: Build the 120-ROI A2 Audit and Freeze Result

**Files:**

- Modify: `src/trustsr/evaluation/development_score_audit.py`
- Modify: `tests/evaluation/test_development_score_audit.py`

**Interfaces:**

- Consumes: 120 ordered development pairs, one bundle per pair, A1 acceptance, and eligible candidate names.
- Produces: `trustsr.phase2b3a-development-score-audit.v1`, cache audit, and zero-inference replay.

- [ ] **Step 1: Write failing complete-development tests**

Create 120 small synthetic ROI spanning exact 12×10 cells. Use score maps with known positive rank,
random-equivalent rank, and one failed stratum. Assert:

```python
def test_a2_freezes_only_from_exact_complete_development_set(tmp_path: Path) -> None:
    result, audit = evaluate_a2_development(
        _complete_small_pairs(),
        _complete_small_bundles(),
        prediction_cache=PredictionCache(tmp_path / "predictions"),
        score_cache=ScoreCache(tmp_path / "scores"),
        include_ldsr_variance_k5=True,
        code_revision="d" * 40,
    )
    assert result["schema"] == "trustsr.phase2b3a-development-score-audit.v1"
    assert result["sample_count"] == 120
    assert result["statistical_unit"] == "roi"
    assert result["bootstrap"] == {
        "algorithm": "numpy.PCG64",
        "seed": 23031,
        "resamples": 10_000,
        "ci_percentiles": [2.5, 97.5],
    }
    assert result["frozen_score"]["name"] in COST_ORDER
    assert audit["sample_count"] == 120
```

Also reject 119/121 ROI, any duplicate identity, reordered bundle membership, a calibration pair,
wrong A1 flag, requested variance after A1 removed it, and incomplete diagnostics. For a complete run
with no eligible score, assert `frozen_score is None`, `phase_decision == "stop_no_eligible_score"`,
and retention of every failed candidate summary.

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/evaluation/test_development_score_audit.py -k a2 -q`  
Expected: missing `evaluate_a2_development` failure.

- [ ] **Step 3: Implement streaming A2 evaluation**

Define:

```python
A2_RESULT_SCHEMA = "trustsr.phase2b3a-development-score-audit.v1"
A2_CACHE_AUDIT_SCHEMA = "trustsr.phase2b3a-development-score-cache-audit.v1"
A2_SCORE_NAMES = (
    "lr_reprojection_l1",
    "three_model_disagreement",
    "ldsr_variance_k5",
)
```

Give `evaluate_a2_development` a required keyword-only `code_revision: str` and reject anything except
a 40-character lowercase Git object ID. Process one ROI at a time: load verified cached predictions, compute/load candidate score maps, build
R9/R1, produce `RoiScoreDiagnostics`, append only scalar records, and release tensor references before
the next ROI. Pass primary R9 scalar records to `freeze_score`. Include R1 summaries under
`sensitivity_window_1` but never pass them to `freeze_score`.

Catch only `ValueError("no development score candidate is eligible")` from `freeze_score` and convert
it to the explicit scientific stop payload above. Re-raise every other validation/integrity error.

`frozen_score` must include the selected name, exact operator parameters, applicable seed tuple,
manifest SHA, source code commit supplied by the CLI, cost rank, statistical leader, indistinguishable
candidates, and all candidate eligibility evidence.

- [ ] **Step 4: Add exact A2 replay**

Implement `replay_a2_development` with the same no-model signature pattern as A1. It must rebuild all
score identities from committed candidate configuration, verify 120 prediction/score cache groups,
snapshot before/after, recompute R9/R1 diagnostics and bootstrap selection, and require
`canonical_json(rebuilt_result) == canonical_json(committed_result)` plus the same audit equality.

Add a test where `build_bootstrap_indices` is monkeypatched to return altered rows after the committed
result is created; replay must fail, proving selection is recomputed rather than copied.

- [ ] **Step 5: Verify GREEN and lint**

Run: `uv run pytest tests/evaluation/test_development_score_audit.py -q`  
Expected: pass.

Run: `uv run ruff check src/trustsr/evaluation/development_score_audit.py tests/evaluation/test_development_score_audit.py`  
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/trustsr/evaluation/development_score_audit.py tests/evaluation/test_development_score_audit.py
git commit -m "feat: freeze phase2b3a development score"
```

### Task 10: Add Strict Staged CLI Orchestration

**Files:**

- Create: `src/trustsr/cli/phase2b3a.py`
- Create: `src/trustsr/cli/phase2b3a_verify.py`
- Create: `tests/cli/test_phase2b3a.py`
- Create: `tests/cli/test_phase2b3a_verify.py`
- Modify: `pyproject.toml`

**Interfaces:**

- Consumes: digest-addressed cloud storage, reviewed project root, model directories, and one fixed stage.
- Produces: six cloud stages through `trustsr-phase2b3a` and two local-only evidence checks through
  `trustsr-phase2b3a-verify`.

- [ ] **Step 1: Write failing parser, upstream, and local-verifier tests**

```python
def test_parser_exposes_only_frozen_stages_and_no_limit_or_sample_ids() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "preflight" in help_text
    assert "development-replay" in help_text
    assert "--limit" not in help_text
    assert "--sample-id" not in help_text


@pytest.mark.parametrize("stage", ["smoke", "replay"])
def test_a1_stages_load_only_four_development_pairs(monkeypatch, stage: str) -> None:
    loaded_splits = []
    monkeypatch.setattr(phase2b3a, "load_crosssensor_pair", _recording_loader(loaded_splits))
    _invoke_stage(stage, monkeypatch)
    assert loaded_splits == ["development"] * 4


@pytest.mark.parametrize("stage", ["development", "development-replay"])
def test_a2_stages_load_only_120_development_pairs(monkeypatch, stage: str) -> None:
    loaded_splits = []
    monkeypatch.setattr(phase2b3a, "load_crosssensor_pair", _recording_loader(loaded_splits))
    _invoke_stage(stage, monkeypatch)
    assert loaded_splits == ["development"] * 120
```

Also test all digest/path/symlink failures, exact result destinations, preflight-before-model ordering,
wrong A1 acceptance, output collision with different bytes, canonical stdout, replay model factory that
raises, git dirty/detached/mismatched commit, and runtime fields absent from scientific JSON.

In `tests/cli/test_phase2b3a_verify.py`, require `--bundle` and `--output`, reject an output outside
`artifacts/phase2b3a`, and assert a valid synthetic bundle writes exact `canonical_json(acceptance)`
bytes rather than formatted stdout.

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/cli/test_phase2b3a.py tests/cli/test_phase2b3a_verify.py -q`  
Expected: module import failure.

- [ ] **Step 3: Implement parser, roots, and factories**

Add to `pyproject.toml`:

```toml
trustsr-phase2b3a = "trustsr.cli.phase2b3a:main"
trustsr-phase2b3a-verify = "trustsr.cli.phase2b3a_verify:main"
```

Use the Phase 2B2-B strict root/model/input-audit validators rather than weakening them. Keep model
imports inside `_model_factory`. Build LDSR once and use `for_seed`. Set directories under:

```python
def _phase_root(root: Path) -> Path:
    return root / "trustsr" / "phase2b3a"


def _prediction_cache_directory(root: Path) -> Path:
    return _phase_root(root) / "predictions" / POST_MANIFEST_SHA256


def _score_cache_directory(root: Path) -> Path:
    return _phase_root(root) / "scores" / POST_MANIFEST_SHA256


def _result_directory(root: Path) -> Path:
    return _phase_root(root) / "results" / POST_MANIFEST_SHA256
```

- [ ] **Step 4: Implement six fail-closed stages**

Stage contracts:

```python
STAGES = (
    "preflight",
    "single",
    "smoke",
    "replay",
    "development",
    "development-replay",
)
```

- `preflight`: capture hardware before model construction, validate reviewed clean commit/base packages,
  load manifests without pixels, validate 10 GiB free disk, then construct the three models.
- `single`: load the first canonical A1 pair, make two direct uncached seed-3407 LDSR calls, require
  identical logical SHA, record duration and peak allocation in runtime JSON, and do not commit either
  direct result to scientific caches.
- `smoke`: generate exactly 4×25 LDSR plus 4×2 deterministic predictions, evaluate A1 core, write the
  separate runtime manifest, add only its SHA-256 to the scientific result, then atomically commit the
  result/audit.
- `replay`: reject model arguments after validation, verify the referenced runtime-manifest SHA,
  rebuild A1 from caches with the same SHA field, require byte identity, and write a replay receipt
  outside scientific JSON.
- `development`: first verify A1 result/audit/replay receipt and runtime resource gates; choose either
  seed 3407 only or `K5A_SEEDS`; load exactly 120 development pairs, evaluate A2, and commit result/audit.
- `development-replay`: construct no model, rebuild A2 from caches, require byte identity, and write a
  replay receipt.

Resource acceptance must implement:

```python
resource_gate_pass = (
    single_peak_memory_bytes <= int(0.80 * gpu_total_memory_bytes)
    and persistent_free_bytes >= 10 * 1024**3
    and 1.5 * projected_a2_uncached_seconds <= 2 * 60 * 60
)
```

The projected A2 time uses the A1 median uncached LDSR prediction duration and the exact missing-seed
count after checking existing caches. Each completed compute/replay pair writes a
`phase2b3a-bundle-manifest.json` containing only allowlisted basenames, byte sizes, and SHA-256 values.

- [ ] **Step 5: Implement the local-only bundle verifier**

`phase2b3a_verify.py` exposes subcommands `a1` and `a2`, each requiring `--bundle Path` and
`--output Path`. It never
loads GeoTIFFs or imports model modules. It validates the bundle manifest, canonical JSON, schemas,
all internal SHA references, no path-like/secret-like values, the 5 MiB limit, and:

```python
def verify_a1_bundle(bundle: Path) -> dict[str, object]:
    files = _verify_allowlisted_bundle(bundle, expected_phase="a1")
    return _verify_a1_result_audit_runtime_replay(files)


def verify_a2_bundle(bundle: Path) -> dict[str, object]:
    files = _verify_allowlisted_bundle(bundle, expected_phase="a2")
    return _verify_a2_result_audit_runtime_replay(files)
```

The returned acceptance payload is atomically written as canonical JSON and contains only digests, pass/fail fields,
`include_ldsr_variance_k5` for A1, and `frozen_score` or `no_eligible_score` for A2. Add tests for every
missing/extra file, digest mutation, noncanonical JSON, oversized file, secret key, wrong schema,
wrong ROI/strata count, replay mismatch, and calibration/internal-test evidence.
Before creating the output parent, resolve the repository root and require the output to be one of the
two declared acceptance basenames directly under `artifacts/phase2b3a`; reject symlink components.

- [ ] **Step 6: Verify GREEN and lint**

Run: `uv run pytest tests/cli/test_phase2b3a.py tests/cli/test_phase2b3a_verify.py -q`  
Expected: pass.

Run: `uv run ruff check src/trustsr/cli/phase2b3a.py src/trustsr/cli/phase2b3a_verify.py tests/cli/test_phase2b3a.py tests/cli/test_phase2b3a_verify.py pyproject.toml`  
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/trustsr/cli/phase2b3a.py src/trustsr/cli/phase2b3a_verify.py tests/cli/test_phase2b3a.py tests/cli/test_phase2b3a_verify.py
git commit -m "feat: orchestrate phase2b3a audit stages"
```

### Task 11: Add the Base-Environment Cloud Runner and Safe Puller

**Files:**

- Create: `scripts/phase2b3a/run_cloud.sh`
- Create: `scripts/phase2b3a/pull_results.sh`
- Create: `tests/scripts/test_phase2b3a_scripts.py`
- Create: `docs/phase2b3a-cloud-runbook.md`
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**

- Consumes: explicit base-Python path, mounted storage root, reviewed repository, stage, and runtime-only
  SSH host/port values.
- Produces: one stage invocation/log or an allowlisted local JSON evidence bundle.

- [ ] **Step 1: Write failing shell-contract tests**

Adapt the existing subprocess fixture style and assert exact argv:

```python
@pytest.mark.parametrize(
    "stage",
    ["preflight", "single", "smoke", "replay", "development", "development-replay"],
)
def test_runner_uses_base_python_and_one_exact_stage(tmp_path: Path, stage: str) -> None:
    completed, calls, prohibited = _invoke(tmp_path, stage=stage)
    assert completed.returncode == 0, completed.stderr
    assert calls[0][:3] == ["-m", "trustsr.cli.phase2b3a", stage]
    assert len(calls) == 1
    assert not prohibited.exists()
```

Test unsafe `/`, `/root`, relative, glob, colon and symlink paths; unmounted storage; less than 10 GiB;
low inode count; unknown stage; dirty/mismatched repository; a non-base interpreter; attempted
conda/pip/curl/wget; secret-like environment names; log collisions; puller extra filenames, symlinks,
oversized files, digest mismatch, and path traversal.

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/scripts/test_phase2b3a_scripts.py -q`  
Expected: missing script failures.

- [ ] **Step 3: Implement `run_cloud.sh`**

Use `set -euo pipefail`, quoted argv, `realpath -e`, component-wise symlink rejection, `mountpoint -q
--`, `df -Pk`, `df -Pi`, and an allowlist case statement for all six stages. The runner must execute
exactly:

```bash
PYTHONPATH="$repository/src" "$base_python" -m trustsr.cli.phase2b3a "$stage" \
  --storage-root "$storage_root" \
  --project-root "$repository" \
  "${stage_arguments[@]}"
```

It must not run package installation or network download commands. Write stdout once to
`$storage_root/trustsr/phase2b3a/logs/$stage.jsonl` while preserving canonical stdout.

- [ ] **Step 4: Implement the allowlisted puller and runbook**

The puller accepts four positional values: SSH host, numeric SSH port, remote storage root, and local
destination. It validates the port with `[[ "$ssh_port" =~ ^[0-9]+$ ]]` and range `1..65535`, then
uses `ssh -p "$ssh_port" -- "$ssh_host"` and `scp -P "$ssh_port" --`. It first retrieves a
digest-addressed bundle manifest and validates that every relative path is one of:

```text
phase2b3a-bundle-manifest.json
phase2b3a-a1-result.json
phase2b3a-a1-cache-audit.json
phase2b3a-a1-runtime.json
phase2b3a-a1-replay.json
phase2b3a-a2-result.json
phase2b3a-a2-cache-audit.json
phase2b3a-a2-runtime.json
phase2b3a-a2-replay.json
```

Then transfer only listed files with protected arguments, reject files above 5 MiB, and verify SHA-256
locally. The runbook must define shell variables without real endpoints:

```bash
: "${PHASE2B3A_STORAGE_ROOT:?set mounted persistent root}"
: "${PHASE2B3A_REPOSITORY:?set reviewed checkout}"
: "${PHASE2B3A_BASE_PYTHON:?set cloud base Python}"
: "${PHASE2B3A_SSH_HOST:?set current user-provided SSH user@host only when pulling}"
: "${PHASE2B3A_SSH_PORT:?set current user-provided numeric SSH port only when pulling}"
```

Document A0, A1, A2 as separate checkpoints, the exact commands, no local pixel transfer, failure
branches, and when to tell the user the GPU can be paused.

Add only these Phase 2B3-A evidence exceptions to `.gitignore`:

```gitignore
!/artifacts/phase2b3a/
/artifacts/phase2b3a/*
!/artifacts/phase2b3a/sen2naipv2-development-smoke-v1.json
!/artifacts/phase2b3a/sen2naipv2-development-smoke-cache-audit-v1.json
!/artifacts/phase2b3a/sen2naipv2-development-smoke-acceptance-v1.json
!/artifacts/phase2b3a/sen2naipv2-development-score-audit-v1.json
!/artifacts/phase2b3a/sen2naipv2-development-score-cache-audit-v1.json
!/artifacts/phase2b3a/sen2naipv2-development-score-acceptance-v1.json
```

- [ ] **Step 5: Verify GREEN, syntax, and lint**

Run: `uv run pytest tests/scripts/test_phase2b3a_scripts.py -q`  
Expected: pass.

Run: `bash -n scripts/phase2b3a/run_cloud.sh scripts/phase2b3a/pull_results.sh`  
Expected: no output and exit 0.

Run: `uv run ruff check tests/scripts/test_phase2b3a_scripts.py`  
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add .gitignore README.md docs/phase2b3a-cloud-runbook.md scripts/phase2b3a tests/scripts/test_phase2b3a_scripts.py
git commit -m "docs: add phase2b3a staged cloud workflow"
```

### Task 12: Pass the Complete A0 Local Acceptance Gate

**Files:**

- Modify only if verification exposes a defect: files owned by Tasks 1–11 and their tests.
- Do not create real-data artifacts in this task.

**Interfaces:**

- Consumes: the complete A0 implementation.
- Produces: a reviewed, pushed A0 commit that is safe to run remotely.

- [ ] **Step 1: Synchronize and verify the lock**

Run:

```bash
uv sync --dev --locked
git diff --exit-code -- uv.lock
```

Expected: sync succeeds and `uv.lock` has no unexpected diff. If adding no dependency, it remains
byte-identical.

- [ ] **Step 2: Run focused Phase 2B3-A tests**

```bash
uv run pytest \
  tests/data/test_crosssensor_pairs.py \
  tests/risk/test_proxies.py \
  tests/evaluation/test_score_diagnostics.py \
  tests/evaluation/test_score_selection.py \
  tests/artifacts/test_scores.py \
  tests/models/test_ldsr_s2.py \
  tests/evaluation/test_development_predictions.py \
  tests/evaluation/test_development_score_audit.py \
  tests/cli/test_phase2b3a.py \
  tests/cli/test_phase2b3a_verify.py \
  tests/scripts/test_phase2b3a_scripts.py -q
```

Expected: all focused tests pass.

- [ ] **Step 3: Run full repository verification**

```bash
uv run pytest
uv run ruff check .
bash -n scripts/phase2b3a/run_cloud.sh scripts/phase2b3a/pull_results.sh
git diff --check
git status --short
```

Expected: all tests and Ruff pass, shell syntax passes, no whitespace errors, and the worktree is clean.

- [ ] **Step 4: Review security and leakage strings**

```bash
rg -n -i 'password|passwd|token|private.key|ghp_|ssh root@|calibration.*\.tif|internal_test.*\.tif' \
  src/trustsr/cli/phase2b3a.py scripts/phase2b3a docs/phase2b3a-cloud-runbook.md
```

Expected: only explanatory prohibitions or test fixtures; no credential, endpoint, or forbidden pixel
path. Inspect every match.

- [ ] **Step 5: Push the exact A0 branch**

```bash
git push -u origin feature/phase2b3a-score-audit-design
git rev-parse HEAD
git status --short --branch
```

Expected: remote branch points to the printed clean commit. Record that SHA for the cloud checkout.

### Task 13: Run and Accept the A1 Four-ROI GPU Gate

**Files:**

- Create after verified run: `artifacts/phase2b3a/sen2naipv2-development-smoke-v1.json`
- Create after verified run: `artifacts/phase2b3a/sen2naipv2-development-smoke-cache-audit-v1.json`
- Create after verified run: `artifacts/phase2b3a/sen2naipv2-development-smoke-acceptance-v1.json`

**Interfaces:**

- Consumes: user-provided current SSH target, the exact pushed A0 commit, mounted long-term storage, existing data/models, and cloud base Python.
- Produces: verified A1 stability/resource decision and a `K=5` include/exclude flag for A2.

- [ ] **Step 1: Stop locally and request the current GPU connection**

Tell the user: Phase 2B3-A A0 is complete; A1 now needs a GPU instance and the current SSH target. Do
not reuse an old host/port from conversation history. Do not place the received target into Git.

- [ ] **Step 2: Verify remote mount, checkout, base environment, and commit**

After the user supplies the target, export it only in the active shell and run read-only checks first:

```bash
: "${PHASE2B3A_SSH_HOST:?set user@host from the user's current connection}"
: "${PHASE2B3A_SSH_PORT:?set numeric port from the user's current connection}"
: "${PHASE2B3A_REMOTE_REPOSITORY:?set verified persistent checkout path}"
ssh -p "$PHASE2B3A_SSH_PORT" -- "$PHASE2B3A_SSH_HOST" mountpoint -q /root/rivermind-fs
ssh -p "$PHASE2B3A_SSH_PORT" -- "$PHASE2B3A_SSH_HOST" \
  git -C "$PHASE2B3A_REMOTE_REPOSITORY" status --short
ssh -p "$PHASE2B3A_SSH_PORT" -- "$PHASE2B3A_SSH_HOST" \
  git -C "$PHASE2B3A_REMOTE_REPOSITORY" rev-parse HEAD
ssh -p "$PHASE2B3A_SSH_PORT" -- "$PHASE2B3A_SSH_HOST" /opt/conda/bin/python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Expected: mount succeeds, checkout is clean and equals the pushed A0 SHA, and the base interpreter has
CUDA-enabled PyTorch. If the persistent checkout uses a different reviewed path, set
`PHASE2B3A_REPOSITORY` to that verified absolute path for this shell; do not edit committed scripts.

- [ ] **Step 3: Run A1 stages separately**

On the verified remote shell:

```bash
export PHASE2B3A_STORAGE_ROOT=/root/rivermind-fs
export PHASE2B3A_REPOSITORY="$PHASE2B3A_REMOTE_REPOSITORY"
export PHASE2B3A_BASE_PYTHON=/opt/conda/bin/python
: "${PHASE2B3A_SELECTION_MANIFEST:?set frozen Phase 2B1-B samples.jsonl path}"
: "${PHASE2B3A_INPUT_AUDIT:?set frozen Phase 2B2-A input audit path}"
: "${PHASE2B3A_SEN2SRLITE_MODEL_DIR:?set verified SEN2SRLite model directory}"
: "${PHASE2B3A_LDSR_MODEL_DIR:?set verified LDSR model directory}"
PHASE2B3A_COMMON_ARGS=(
  --selection-manifest "$PHASE2B3A_SELECTION_MANIFEST"
  --selection-manifest-sha256 c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a
  --input-audit "$PHASE2B3A_INPUT_AUDIT"
  --input-audit-sha256 fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b
  --sen2srlite-model-dir "$PHASE2B3A_SEN2SRLITE_MODEL_DIR"
  --ldsr-model-dir "$PHASE2B3A_LDSR_MODEL_DIR"
  --confirm-cloud-storage
)
cd "$PHASE2B3A_REPOSITORY"

scripts/phase2b3a/run_cloud.sh \
  "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_STORAGE_ROOT" "$PHASE2B3A_REPOSITORY" preflight \
  "${PHASE2B3A_COMMON_ARGS[@]}"
scripts/phase2b3a/run_cloud.sh \
  "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_STORAGE_ROOT" "$PHASE2B3A_REPOSITORY" single \
  "${PHASE2B3A_COMMON_ARGS[@]}"
scripts/phase2b3a/run_cloud.sh \
  "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_STORAGE_ROOT" "$PHASE2B3A_REPOSITORY" smoke \
  "${PHASE2B3A_COMMON_ARGS[@]}"
scripts/phase2b3a/run_cloud.sh \
  "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_STORAGE_ROOT" "$PHASE2B3A_REPOSITORY" replay \
  "${PHASE2B3A_COMMON_ARGS[@]}"
```

Expected: every stage exits 0; replay reports byte-identical science output and zero inference.

- [ ] **Step 4: Pull only the A1 allowlist and verify locally**

```bash
scripts/phase2b3a/pull_results.sh \
  "$PHASE2B3A_SSH_HOST" "$PHASE2B3A_SSH_PORT" /root/rivermind-fs \
  artifacts/remote-phase2b3a-a1
uv run python -m trustsr.cli.phase2b3a_verify a1 \
  --bundle artifacts/remote-phase2b3a-a1 \
  --output artifacts/phase2b3a/sen2naipv2-development-smoke-acceptance-v1.json
```

Expected: manifest/file SHA values pass; result/replay are byte-identical; four bins exist; all caches
are accounted for; runtime gate is explicit; no file exceeds 5 MiB.

- [ ] **Step 5: Apply the frozen A1 decision**

- If statistical stability and resource gates pass, set the acceptance flag
  `include_ldsr_variance_k5=true` and proceed immediately to Task 14 while the same GPU remains useful.
- If only statistical stability fails, set `include_ldsr_variance_k5=false`; deterministic candidates
  may proceed to Task 14.
- If repeatability, cache integrity, mount, environment, or resource safety fails, stop, preserve
  evidence, tell the user the GPU can be paused, and invoke systematic debugging before any fix.

- [ ] **Step 6: Commit verified small A1 evidence**

Copy only the verified canonical result and cache audit summary beside the acceptance record:

```bash
cp -- artifacts/remote-phase2b3a-a1/phase2b3a-a1-result.json \
  artifacts/phase2b3a/sen2naipv2-development-smoke-v1.json
cp -- artifacts/remote-phase2b3a-a1/phase2b3a-a1-cache-audit.json \
  artifacts/phase2b3a/sen2naipv2-development-smoke-cache-audit-v1.json
```

Then run:

```bash
test "$(find artifacts/phase2b3a -type f -size +5M -print -quit)" = ""
git diff --check
git add artifacts/phase2b3a
git commit -m "data: record phase2b3a a1 acceptance"
git push
```

Expected: only small JSON is committed; no runtime path/host/credential is present.

### Task 14: Run, Accept, and Publish the A2 Development Audit

**Files:**

- Create after verified run: `artifacts/phase2b3a/sen2naipv2-development-score-audit-v1.json`
- Create after verified run: `artifacts/phase2b3a/sen2naipv2-development-score-cache-audit-v1.json`
- Create after verified run: `artifacts/phase2b3a/sen2naipv2-development-score-acceptance-v1.json`
- Create: `docs/phase2b3a-pr-body.md`
- Modify: `README.md`

**Interfaces:**

- Consumes: accepted A1 flag, same exact code commit on cloud, all 120 development ROI, and verified caches.
- Produces: one frozen eligible score or an explicit preregistered stop result; no calibration/test claim.

- [ ] **Step 1: Synchronize the remote checkout to the A1 evidence commit**

Push the A1 commit first, then update the verified persistent checkout non-interactively and require a
clean exact SHA. Do not merge unrelated branches or change dependencies on the running instance.

```bash
git push
ssh -p "$PHASE2B3A_SSH_PORT" -- "$PHASE2B3A_SSH_HOST" \
  git -C "$PHASE2B3A_REMOTE_REPOSITORY" fetch origin feature/phase2b3a-score-audit-design
ssh -p "$PHASE2B3A_SSH_PORT" -- "$PHASE2B3A_SSH_HOST" \
  git -C "$PHASE2B3A_REMOTE_REPOSITORY" switch feature/phase2b3a-score-audit-design
ssh -p "$PHASE2B3A_SSH_PORT" -- "$PHASE2B3A_SSH_HOST" \
  git -C "$PHASE2B3A_REMOTE_REPOSITORY" pull --ff-only
```

Expected: remote HEAD equals local HEAD and status is clean.

- [ ] **Step 2: Run A2 compute and replay**

```bash
scripts/phase2b3a/run_cloud.sh \
  "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_STORAGE_ROOT" "$PHASE2B3A_REPOSITORY" development \
  "${PHASE2B3A_COMMON_ARGS[@]}"
scripts/phase2b3a/run_cloud.sh \
  "$PHASE2B3A_BASE_PYTHON" "$PHASE2B3A_STORAGE_ROOT" "$PHASE2B3A_REPOSITORY" development-replay \
  "${PHASE2B3A_COMMON_ARGS[@]}"
```

Expected: exactly 120 ROI and 12×10 strata; replay byte identity; zero inference during replay; no
calibration/internal-test pixel load. If the process is interrupted, rerun the same stage and rely on
verified caches rather than deleting anything.

- [ ] **Step 3: Pull only the A2 allowlist and verify locally**

```bash
scripts/phase2b3a/pull_results.sh \
  "$PHASE2B3A_SSH_HOST" "$PHASE2B3A_SSH_PORT" /root/rivermind-fs \
  artifacts/remote-phase2b3a-a2
uv run python -m trustsr.cli.phase2b3a_verify a2 \
  --bundle artifacts/remote-phase2b3a-a2 \
  --output artifacts/phase2b3a/sen2naipv2-development-score-acceptance-v1.json
```

Expected: exact schemas/digests; 120 unique sample/group IDs; 12 strata of 10; bootstrap seed/count;
all candidate reasons; and either one deterministic `frozen_score` or the exact no-eligible stop error.

- [ ] **Step 4: Tell the user the GPU can be paused**

After local bundle verification succeeds and no retry remains, explicitly tell the user that Phase
2B3-A no longer needs GPU and the instance can be paused. Do not execute a cloud shutdown command.

- [ ] **Step 5: Commit the verified small A2 evidence**

Copy only the verified result and audit beside the generated acceptance JSON:

```bash
cp -- artifacts/remote-phase2b3a-a2/phase2b3a-a2-result.json \
  artifacts/phase2b3a/sen2naipv2-development-score-audit-v1.json
cp -- artifacts/remote-phase2b3a-a2/phase2b3a-a2-cache-audit.json \
  artifacts/phase2b3a/sen2naipv2-development-score-cache-audit-v1.json
```

Create
`docs/phase2b3a-pr-body.md` with headings `Summary`, `A0 verification`, `A1 evidence`, `A2 evidence`,
`Leakage controls`, `GPU resources`, and `Scientific decision`; populate every heading from the
verified JSON, without paths or credentials. Update README with:
the development-only scope, frozen score name or explicit failure, and a statement that no conformal
or internal-test conclusion has been made.

```bash
test "$(find artifacts/phase2b3a -type f -size +5M -print -quit)" = ""
git diff --check
git add artifacts/phase2b3a README.md docs/phase2b3a-pr-body.md
git commit -m "data: record phase2b3a development score audit"
```

- [ ] **Step 6: Run final verification**

```bash
uv sync --dev --locked
uv run pytest
uv run ruff check .
bash -n scripts/phase2b3a/run_cloud.sh scripts/phase2b3a/pull_results.sh
git diff --check
git status --short --branch
```

Expected: all commands pass and the branch is clean.

- [ ] **Step 7: Push and open/update the stacked pull request**

```bash
git push
gh pr create \
  --base feature/phase2b2b-development-smoke \
  --head feature/phase2b3a-score-audit-design \
  --title "Phase 2B3-A: audit development uncertainty scores" \
  --body-file docs/phase2b3a-pr-body.md
```

The body file contains the exact A0/A1/A2 test evidence, scientific stop/pass decision, frozen score,
data-leakage statement, cache/replay hashes, GPU resource summary, and statement that
calibration/internal-test remain unused. If a PR already exists, use
`gh pr edit` with the same body file rather than creating a duplicate.

The phase ends here. Do not begin Phase 2B3-B until this evidence is reviewed and a separate
calibration-stage design is approved.
