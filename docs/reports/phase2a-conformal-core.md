# Phase 2A conformal core acceptance

Acceptance was run on 2026-08-27 from a clean shell in the
`feature/phase2a-conformal-core` worktree, before this report was created.

## Environment and revision

```text
platform: Linux 6.6.87.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Thu Jun  5 18:30:46 UTC 2025 x86_64 GNU/Linux
uv: uv 0.12.5 (x86_64-unknown-linux-gnu)
python: Python 3.12.3
git: git version 2.43.0
commit: 3c0a9eaebe909e68510ed85f5a792e10fdc1447a
```

## Verification commands and observed output

```bash
uv sync --dev
```

```text
Resolved 134 packages in 0.78ms
Checked 120 packages in 0.85ms
```

```bash
uv run pytest
```

```text
392 passed, 43 warnings in 14.99s
```

The 43 warnings were `torch.jit.script` deprecation warnings from
`torch/jit/_script.py:1488`.

```bash
uv run ruff check .
```

```text
All checks passed!
```

```bash
git diff --check
```

```text
(no output; exit 0)
```

```bash
uv run trustsr-conformal-smoke > /tmp/trustsr-phase2a-accepted.json
sha256sum /tmp/trustsr-phase2a-accepted.json
```

```text
e1fecf77881f8be8cc8bb207ea49af5595af82cf32ea3cf307bdb0e5885414a5  /tmp/trustsr-phase2a-accepted.json
```

```bash
git status --short
```

```text
(no output; the report did not yet exist)
```

## Deterministic smoke payload and observed values

```json
{"calibration":{"calibration_size":3,"coverage":0.5208333333333334,"risk_bound":0.2698333333333333,"roi_max_risk":0.026444444444444448,"threshold":0.0002250000000000004},"config":{"alpha":0.27,"channels":4,"scale":4,"window":1},"schema":"trustsr.conformal-smoke.v1","synthetic_smoke":true,"test":{"coverage":0.0,"roi_count":2,"roi_max_risk":0.0}}
```

Observed calibration values: size 3; coverage 0.5208333333333334; risk bound
0.2698333333333333; ROI maximum risk 0.026444444444444448; threshold
0.0002250000000000004. Observed test values: coverage 0.0; 2 ROIs; ROI maximum
risk 0.0.

## Boundary of this acceptance

Phase 2A verifies only the deterministic CPU implementation of a published generic
conformal SR baseline. It is not evidence of novelty, real Sentinel-2 performance,
cross-sensor validity, or downstream utility. Phase 2B must use geographically
separated ROI-level data and must treat external sensor shift as an empirical audit,
not as an automatic conformal guarantee.
