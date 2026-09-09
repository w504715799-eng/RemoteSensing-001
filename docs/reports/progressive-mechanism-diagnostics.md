# CPU mechanism diagnostics for the progressive manuscript

2026-09-09. These are deterministic synthetic diagnostics, not measured real-data
accuracy/coverage or a new theoretical contribution. No dataset, old image/cache,
model inference or GPU was used. The frozen implementation is unchanged.

Executable: `python -m paper.tools.progressive_mechanism_demo --output NEW_DIRECTORY`.
Published [numeric output](../../paper/tables/progressive-synthetic-mechanisms-v1.json)
and [figure](../../paper/figures/progressive-mechanism-counterexample.png) are labeled
synthetic and generated using the actual local `extract_features` and R9 functions.

## Accurate texture and indistinguishable invented detail

The five generated32x32RGBN samples are identical checkerboards alternating .25/.75.
Area downsampling by4gives the same uniform.5LR input for two possible references:
the checkerboard itself, or a uniform.5image. Model/sample input is exactly the same
in the two cases, so every deployed feature is identical. This construction checks
both references downsample to the supplied LR rather than merely declaring that fact.

| Quantity | Reference A: checkerboard | Reference B: flat |
| --- | ---: | ---: |
| LR reprojection residual | 0 | 0 |
| Smoothed sampling variance | 0 | 0 |
| Mean smoothed texture score | .0617283951 | .0617283951 |
| Full-grid R9 loss | 0 | .25 |

The texture component penalizes genuinely accurate texture in A, but can also help
flag invented texture in B. Removing texture is therefore not universally beneficial.
No method depending only on these identical inputs can distinguish A from B. A
risk guarantee averaged over an exchangeable population does not remove this
per-instance ambiguity. Avoid claims of detecting every semantic hallucination or
identifying true reconstruction error from sampling variance alone.

## Joint versus average within-sample spatial variance

Five spatially constant samples have values .25,.375,.5,.625,.75. The joint
seed/spatial score is.03125, while each member's local spatial variance is zero.
This demonstrates the specific algebraic distinction found in the pinned
[official baseline audit](progressive-official-baseline-audit.md). It does not execute
or validate an entire third-party pipeline and does not imply published results
are wrong. Numerical identities are not a novel contribution.

## What remains for a paper

The current frozen beta0/beta1 and other component comparisons can show whether
texture weighting changes real coverage/risk, but cannot alone establish a causal
mechanism or its prevalence. Spatial stratification needs actual corresponding
features/references, which current compact receipts do not retain. It therefore
requires a separately specified artifact/data plan; do not promise GPU-free replay
or reopen the consumed test set after choosing favorable subgroups.

Two diagnostic tests and37related score/pipeline/calibration tests pass (39total).
The figure was visually inspected. This evidence supports limitations and mechanism
hypotheses only; all final real-data results and submission conclusions remain pending.
