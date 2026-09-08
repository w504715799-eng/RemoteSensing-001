# Progressive Coverage Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deliver tested local scoring and actual-benefit measurement primitives for the approved progressive study.

**Architecture:** Pure CPU functions live outside the old frozen implementation tree. Deploy-time scoring does not accept HR; offline benefit evaluation consumes precomputed full-grid R9 and masks. This foundation is independently useful, while data acquisition/calibration/terminal comparison are a later dependent phase.

**Tech Stack:** Python 3.12, NumPy, PyTorch, pytest, Ruff; existing environment only.

**Spec:** `docs/superpowers/specs/2026-09-08-progressive-coverage-design.md`

## Global Constraints

- No old A/B/C pixels/cache, Spain raw data, SSH or model access.
- Do not modify src/trustsr or scripts/paper; preserve old protocol/source identity.
- Main-branch single-writer workflow follows the established user preference.
- Only focused tests; no full-suite runs. Synthetic examples are not research findings.
- Scores cannot accept HR; offline R9 is computed on the complete grid before masking.

## Task 1: Quantify automatic area and review workload

**Files:** Create `research/trustmask/benefits.py`; test `tests/research/test_benefits.py`.
**Interfaces:** `evaluate_mask(risk, mask, *, high_error_threshold=0.05, review_tile_size=32, pixel_area_m2=None) -> dict`; NumPy real floating 2D risk in [0,1], same-grid bool mask. Return counts, coverage, retained mean/max, high-error diagnostics, tile review workload, optional physical area. All-reject mean is null, max is zero, high-error recall null when none exist.

- [x] Write hand-derived tests: 2×2 risks [[.01,.2],[.03,.4]], mask [[True,False],[True,False]], tile size1, area6.25 -> coverage .5, max .03, mean .02, review2, accepted12.5m². Check all-reject, all-accept, high-error-free nulls, ragged tile counts, invalid dtype/shape/range, immutability.
- [x] Run `.venv/bin/python -m pytest tests/research/test_benefits.py -q`; confirm missing feature fails.
- [x] Implement counts from `risk[mask]`, never masked neighborhood recomputation. Reject nonbool mask and nonfinite risk. Tile count is one review per block containing any rejection.
- [x] Add `compare_workload(baseline, candidate, *, baseline_compute_seconds, candidate_compute_seconds, seconds_per_review_tile=None) -> dict`: coverage delta in pp, delta review units, extra wall time, break-even seconds/review unit when fewer units, optional explicitly modeled net seconds. Require matching ROI workload grids/area/error threshold and validated counters; negative benefits remain negative.
- [x] Test 2 fewer review units +4s extra compute -> break-even2s; assumed3s/review -> modeled2s saving. No assumed time returns null modeled saving; no saved units returns null break-even; mismatched support rejected.
- [x] Run focused tests and `.venv/bin/ruff check research/trustmask/benefits.py tests/research/test_benefits.py`.

## Task 2: Decompose spatial scores and fuse evidence

**Files:** Create `research/trustmask/scores.py`; test `tests/research/test_progressive_scores.py`.
**Interfaces:** `spatial_components(samples) -> tuple[Tensor,Tensor]` returns G(B3(V)) and G(T) with 5×4×H×W floating [0,1] input. `fuse_scores(lr, uncertainty, *, lr_scale, uncertainty_scale, lr_weight) -> Tensor` implements fixed positive-scale x/(x+a) transforms then convex fusion on same 2D grid; no HR argument or in-place changes.

- [x] Test five spatial constants 0,0,0,1,1 -> uncertainty .24 and texture zero. Identical spatially varying seeds -> uncertainty zero, positive texture. Sum must equal the existing frozen neighborhood w3 implementation within1e-12 on a fixed synthetic random tensor.
- [x] Test fusion LR=[0,1], variance=[1,0], both scales1, weight.25 -> [.375,.125]; changing one ROI cannot change another ROI normalization. Reject invalid grids, scales, weights, nonfinite/range values.
- [x] Run `.venv/bin/python -m pytest tests/research/test_progressive_scores.py -q` and observe missing feature failure.
- [x] Implement CPU float64 reflected pooling/Gaussian and validated frozen-scale fusion. No fitted values or selections are implied by this primitive.
- [x] Run focused scores tests and changed-file Ruff.

## Task 3: Demonstrate interfaces and audit the boundary

**Files:** Create `research/trustmask/demo.py`, `research/trustmask/README.md`, `docs/reports/progressive-coverage-foundation.md`; update handoff and this plan.
**Interfaces:** `.venv/bin/python -m research.trustmask.demo` prints small synthetic JSON using both modules; no file/network/model access.

- [x] Create a synthetic LR/variance example with a fixed illustrative threshold, full-grid illustrative risk, all-accept baseline, and candidate mask. Label output `synthetic_not_research_evidence`; never label threshold calibrated or risk certified.
- [x] Run demo and parse JSON; independently verify counts against the tiny mask. Include explicit workload/time assumptions and missing actual human-time evidence.
- [x] Document deployed inputs/outputs, optional offline HR-derived risk, unsupported claims and next data/calibration phase. Report tested functions separately from pending integrated software and real experiments.
- [x] Read-only review per requesting-code-review; address important findings with focused regression tests.
- [x] Load old frozen protocol with its published SHA, check published science/execution hashes, run changed-file tests/Ruff and staged data policy; commit exact changed files.

## Following dependent phase

Source metadata audit must identify new nonoverlapping geographic groups and support the target statistical power before pixel access. Next plan will specify exact data/splits, development scale-fit and selection, group-loss calibration, simultaneous test risk bounds and paired coverage comparisons. No experimental benefit is claimed by foundation completion; do not mark the overall study complete.

本地基础阶段完成，证据见 `docs/reports/progressive-coverage-foundation.md`。后续依赖阶段尚未执行，不代表整体研究完成。
