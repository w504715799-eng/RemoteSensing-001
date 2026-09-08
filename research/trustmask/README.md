# Progressive trust-mask research primitives

This repository-local module implements reference-free scoring and offline workload accounting.
It is not yet a calibrated application, wheel distribution, or evidence of real performance gains.
No new dependency, model, data download, or GPU is needed for these functions.

```sh
.venv/bin/python -m research.trustmask.demo
.venv/bin/python -m pytest tests/research -q
```

The demo outputs `synthetic_not_research_evidence`: a hand-constructed risk map, assumed timings,
and an uncalibrated threshold. It is an interface example, not a baseline experiment.

`spatial_components(samples)` accepts finite floating `(5,4,H,W)` predictions in [0,1]
and returns separately Gaussian-smoothed seed variance and ensemble-mean spatial texture.
Sum equals the frozen w3 joint neighborhood score up to floating-point rounding. This identity
is not a claim of novelty. Texture weight and positive normalization scales must be chosen on
new development groups and frozen before calibration/test; current primitives do not fit them.

`fuse_scores(lr, uncertainty, lr_scale=..., uncertainty_scale=..., lr_weight=...)` uses fixed
x/(x+a) transforms and convex fusion; it does not use reference imagery or per-image ranks.
Current output is a score, not a calibrated probability. Independent calibration, authenticated
configuration serialization and deployable mask generation remain in the next phase.

`evaluate_mask(risk, mask, ...)` is an **offline** evaluator: `risk` is a full-grid R9 map computed
before masking. It counts automatic pixels, rejected pixels, physical area if supplied,
high-error leakage/capture, and fixed tiles requiring review. A tile requires review if any
pixel is rejected; partial edge tiles count once. No-risk pixels have undefined capture recall;
all-reject has zero ROI-max loss but undefined retained mean and zero coverage.

`compare_workload(...)` requires summaries of matching reference grids and measurement settings.
Checksums detect accidental local summary edits, not adversarial changes or authenticated ROI
identity. The future runner must bind source/ROI identity and membership separately. This helper
returns per-ROI differences; it does not pool groups, compute confidence intervals, certify risk,
or establish independence. Never treat `risk_qualification=not_evaluated` as passed.

Review-unit savings are a workload **proxy**. Optional per-unit seconds and compute durations
produce an explicitly modeled net saving and break-even review time. No person-time is measured
by this module. A reference error is not a downstream task label or error-free ground truth.

Scientific design and dependent data/statistics phase:
[design](../../docs/superpowers/specs/2026-09-08-progressive-coverage-design.md),
[implementation plan](../../docs/superpowers/plans/2026-09-08-progressive-coverage-foundation.md).
