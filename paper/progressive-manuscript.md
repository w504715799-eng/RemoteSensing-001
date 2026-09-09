# Risk-Controlled Trusted Coverage for Sentinel-2 Super-Resolution: Disentangling Sampling Variation and Spatial Texture

Working research manuscript scaffold, 2026-09-09. The title is provisional and
states the question, not a demonstrated improvement. This document concerns the
new progressive study; `manuscript.md` records historical experiments. No new
terminal results have been inspected to write this scaffold. No acceptance or
publication-readiness claim is made.

## Research question and contribution boundary

Can separating sampling variation and local texture, together with LR consistency,
increase risk-qualified trusted coverage at a fixed five-sample inference budget?

Inherited: conformal risk control, confidence masks, neighborhood second moments,
Gaussian smoothing and sampling variance. The variance decomposition is an identity,
not a new theorem. Candidate contributions are an explicitly audited multispectral
adaptation, a controlled test of the texture-weighting hypothesis, and reproducible
risk/coverage/workload evidence. Their novelty and practical importance are not yet
established by implementation alone.

See [novelty audit](../docs/reports/progressive-novelty-audit.md) and
[official baseline audit](../docs/reports/progressive-official-baseline-audit.md).
The most direct baseline is a **paper-formula adaptation** of Adame et al., rather
than an asserted faithful reproduction of every released script. Official code and
paper algebra differ in the position of the squared mean; disclose this distinction.

## Related work and task matching

Adame et al., NeurIPS2025, directly address trusted SR masks with expected fidelity
risk control. Their RGB/Lab example, loss bound and released workflow differ from
ours. Interval prediction methods such as Conffusion answer a related but different
question. New satellite self-supervised uncertainty work prevents unsupported claims
of first reference-free EO uncertainty estimation. Cite primary sources and describe
training/backbone/modality differences, not just publication names.

## Data and fixed protocol

7477eligible public-release SEN2NAIPv2members in6158geographic groups, after excluding
523members in components touching historical data. Roles: scale-fit318ROIs,
development-calibration614, development-validation1041, calibration1814, test3690.
Source revision, member assignments, support/model/seed definitions and checksums
are fixed in `research/protocols/trustmask-real-batches-v1.json`.

Public availability is accepted provenance, not proof of scene independence or
absence of pretraining overlap. HR is a harmonized reference with sensor/time/
registration limitations. Historical A/B/C is terminal; old Spain outputs are
seen descriptive context, not independent evidence for this new fusion method.

## Methods and hypotheses

Fixed center seed3407 and samples3407–3411; RGBN x4; fixed128/512crop and full512²
R9 risk. Define U=B3(mean-band population variance) and
T=mean-band[B3(mu²)-B3(mu)²]. Candidate Q_beta=G(U+beta*T), beta∈{0,.5,1}; combine
its fixed-scale normalized score with normalized LR consistency using the declared
lambda grid. All scales and method selection use the separated development roles.

H1: a positive texture contribution can reject accurate but textured areas. H2:
reducing texture weight may improve coverage in some cases, but can also fail to
flag stable invented texture. H3: LR residual adds information about input
inconsistency; it cannot identify detail in the downsampling nullspace on its own.
These are falsifiable hypotheses, not completed experimental findings.

[CPU synthetic diagnostics](../docs/reports/progressive-mechanism-diagnostics.md)
explicitly include identical LR/predictions with different
HR references, and the distinction between joint variance and mean per-sample
spatial variance. They explain limitations, not real-data improvement or prevalence.

## Evaluation and decision rules

Primary comparison remains full vs w3 with fixed8-method risk-family accounting.
All thresholds calibrated once, all methods frozen before terminal access. Expected
group loss is equal-group/equal-ROI mean of retained-ROI maximum R9; it is not the
percentage of bad pixels. Risk target .05; joint risk/gain delta .05. Report paired
coverage difference and its lower bound, distinguishing positive gain from the
prespecified5percentage-point engineering target. No threshold tuning on test.

At3045testgroups, existing8-method Hoeffding risk margin is about.03078; a method's
empirical mean must be at most about.01922 to meet the .05 test upper-bound criterion.
CRC calibration at .05 does not automatically ensure this stronger evidence gate.
If it fails, report inconclusive independent-test certification; do not lower the
risk target, relax the interval or replace the test after observing results.

Secondary descriptive outputs: fixed32x32whole-tile acceptance, rejected workload,
high-error acceptance and all-rejected cases. These count automated output and
remaining processing workload, not human hours saved. Main coverage uses group
weighting; pooled pixel/tile totals are explicitly separate operational summaries.

## Results slots (pending complete authorized evaluation)

Supplemental experiments are authorized in the
[implementation plan](../docs/superpowers/plans/2026-09-09-supplemental-baselines.md).
Pending additional evidence: tuned paper-formula and released-helper adaptations,
an independent im2im-UQ learned comparator, independent geographic/sensor validation,
and standalone compute/workload measurements. Separate native interval metrics from
fixed-center mask adaptation; do not label the latter an exact official pipeline.
These experiments are not implemented or completed merely by being listed here.
They have separate data/protocol gates and do not change the eight-method E0 table.

- Table1: all8methods, calibration threshold, test risk mean/upper bound, group coverage.
- Table2: prespecified paired full-vs-w3 gain and risk/gain qualification decisions.
- Table3: all8methods' whole-tile pass rate, workload and high-error acceptance.
- Figure1: method and role flow; Figure2: complete method comparison; Figure3:
  synthetic mechanism counterexamples, conspicuously labeled synthetic.
- Supplement: full candidate/selection audit, source/membership/execution manifests,
  unfavorable comparisons and failure accounting.

No primary result values are filled in before the frozen run completes. Do not
select only favorable configurations or relabel exploratory analyses confirmatory.

## Compute and limitations

All scores share the expensive five-seed predictions in this study. LR-only
standalone deployment uses fewer calls; shared research wall time cannot establish
per-method deployment efficiency. Two-process scheduling has operational throughput
evidence and an execution amendment, not a new scientific algorithm contribution.

Full feature/prediction maps are not retained. Additional spatial analyses or a
released-code baseline may require new artifacts/inference and an independent
supplemental protocol. Cross-domain and cross-backbone performance remain open.
Without real task labels, do not equate low R9 with absence of semantic hallucinations.

## Submission gate

Write abstract/results/conclusion after evidence verification. Assess contribution,
baseline fairness, effect strength, mechanism evidence, generality and reproducibility
before selecting the final venue/scope. More computation alone does not establish
publishability. Author metadata and submission authorization remain separate steps.
