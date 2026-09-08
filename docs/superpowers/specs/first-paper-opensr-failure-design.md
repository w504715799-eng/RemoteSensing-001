# OpenSR image-level failure contract

2026-09-08, pre-access Task 5 contract. Keep exploratory image-level softmin shares; no pixel-level
hallucination claim. Call pinned OpenSR-Test 1.3.3 correctness only, with explicit CPU, pixel
aggregation, border=16, spectral/spatial harmonization, PCC max translation=5, correctness distance
nd, softmin temperature=.25, scores=.05, RGB indices [0,1,2], gradient_threshold=auto.

Observe the actual SR-to-HR harmonization alignment return value: upstream sr_harm_setup discards
its error scalar, whereas SpatialMetricAlign returns NaN when satalign.warning_status is set.
Intercept this boundary with a checked wrapper and mark registration_failed before producing
correctness values. Do not infer this from the separate LR consistency spatial scalar.

Run correctness in a separate CPU-only process: upstream seed_everything also changes CUDA RNG
and cuDNN flags. A process avoids perturbing frozen model inference or caller RNG/backend state.
Timeout 120 seconds per ROI, no retry/fallback/change of aligner. No model download. Return only
JSON scalars: cropped pixel count, jointly finite softmin pixel count, ha/om/im means, status and
static failure reason. Unknown/failed registration, no finite pixels, inconsistent masks, nonfinite
or out-of-range shares and runtime/timeout failures retain null shares, not zero. Registration
failures have unknown valid-pixel count (null); zero valid pixels is a distinct observed outcome.

A successful row requires finite identical support across three maps, values within [0,1], and
per-pixel softmin sums within 1e-5 of one. Preserve upstream equal-valid-pixel means; never apply a
trusted-score mask to re-evaluate OpenSR. These are secondary diagnostics on the center prediction;
method-score ranking and primary paired comparisons remain unchanged on diagnostic failure.
