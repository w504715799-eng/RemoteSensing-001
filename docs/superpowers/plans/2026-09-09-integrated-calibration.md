# Integrated CPU calibration workflow

User explicitly requested direct implementation. Continue on main; only new
research code and synthetic tests. Public release provenance is accepted. No
images, models, remote server, historical prediction caches or external messages.

1. Implement `research/trustmask/calibration.py`: validated nondecreasing bounded
   group loss/coverage curves on a fixed threshold grid; CRC threshold selection
   `(sum(group_losses)+1)/(n_groups+1) <= alpha`, with deterministic reject-all
   fallback; independent test Hoeffding bounds and paired coverage comparison.
   CRC controls expectation over calibration and a fresh exchangeable group,
   not a high-probability conditional risk after seeing calibration data.
2. Implement `research/trustmask/pipeline.py`: immutable feature/observation inputs;
   LR reprojection and five-seed/spatial feature adapter; equal-ROI fixed 64-position
   median scale fit with floor 1e-8; score families and independent five-role
   membership checks. Candidate scales never use HR or non-scale-fit observations.
3. Fixed families: full (5 lambda x 3 beta), LR, K5, w3, fusion (5 lambda), spatial
   (3 beta), beta0 (5 lambda), beta1 (5 lambda). Lambda 0,.25,.5,.75,1; beta 0,.5,1.
   Development CRC uses only development-calibration. Development-validation selects
   by empirical risk<=alpha, highest equal-group coverage, lower risk, fixed order.
   If none qualifies choose a deterministic reject-only method. Formal calibration
   never chooses a configuration. Deploy methods accept features, never HR/R9.
4. Fixed 101-point grid [0,1] plus explicit reject sentinel -1. Group losses are
   means of per-ROI retained maximum R9, empty=0; coverage uses the same hierarchy.
   Compare eight methods with test risk delta=.025/8; full-vs-w3 paired lower bound
   uses delta=.025 and range [-1,1]. Other comparisons descriptive only. Eight rather
   than six methods slightly increase the prior test planning risk margin; report
   actual count-dependent margins and do not reuse six-method planning constants.
5. Synthetic runnable demo and tests: hand-calculated unequal-size groups, finite-grid
   monotonicity and rejection, CRC correction, role overlap, scale/test isolation,
   deterministic selection, deploy interface, paired interval and full run. Integrate
   existing mask/workload measurement, without invented physical area or human time.
6. Independent review, focused tests/lint, frozen gate, staged-data policy, documentation
   and handoff. No real empirical gain or completed cloud experiment claim.

The pipeline consumes precomputed full-grid R9 offline. The real-data runner must
still bind member IDs/groups to the exact authenticated assignment, SR/HR/R9
generation and common support. This module's separation checks are structural,
not source authentication or one-time test-access enforcement. GPU notification
comes only after the real execution protocol and extraction/inference runner exist.


Implementation status: all six local CPU tasks completed. Source/report files and
synthetic tests are present; the runnable example produces eight-method results.
Formal real-data execution is deliberately the next dependent task, not completed
by this implementation. All configuration selection now precedes any formal
calibration loss access, with a regression test enforcing the stage boundary.
