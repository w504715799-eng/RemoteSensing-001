# Parallel GPU execution amendment

2026-09-09. User explicitly requested using available GPU capacity to shorten the
ongoing7477-ROI study, with no further batch/role pauses. Serial throughput measured
before the concurrency probe was9.9084seconds/ROI over20recent receipts; sampled
GPU utilization67%, memory5685MiB, power281.26W.

## Scientific and operational boundary

The frozen scientific implementation and protocol remain byte-identical. New
`research/operations/parallel_batches.py` is an explicit operational amendment:
two spawn-isolated processes call the unchanged LocalTacoProvider, LDSR backend,
five seeds, feature extraction and R9 calculation. Tensor batches, precision,
thresholds, method family and source membership do not change. No worker shares
random state. The coordinator alone runs the frozen batch engine and writes its
receipts in the original order. Prefetch is bounded to two outstanding jobs within
the role currently requested by that engine. The test ledger and all frozen methods
must exist and match before any test job submission.

This does amend the original **one resident ROI** operational policy: two ROI jobs
can be outstanding, plus the coordinator's current statistics. That difference is
recorded explicitly, rather than represented as an unchanged execution policy.
The original scientific/runtime journal binding and completed receipts are retained.
`execution_amendments/` in the existing study binds wrapper SHA, worker count,
budget, original binding/protocol and every pre-cutover receipt SHA. The wrapper
only resumes existing verified studies, and refuses another active writer.

A crash may discard uncommitted same-role prefetch and retry it under the same
scientific binding. Already committed ROI inference is not repeated. SIGTERM to
the coordinator unwinds the process pool before fallback, allowing its at-most-two
outstanding jobs to finish without promoting uncommitted results to receipts.

## GPU equivalence and measurement

`research/operations/gpu_parallel_probe.py` uses two fixed synthetic float32 RGBN
128² inputs, never study images. A serial worker and two parallel workers each
compute all five seed outputs. All10prediction SHA256s, shapes and dtypes matched.
After separate initialization/warmup, serial wall24.0305s versus concurrent17.5932s,
a speedup1.3659x. The serial study remained active during this synthetic probe, so
these timings include contention and are **not** the real-study speedup estimate.
Exact agreement on these inputs is regression evidence, not a proof for all inputs.
Evidence: `research/evidence/trustmask-parallel-synthetic-probe-v1.json`.

## Cutover and continuous execution

The original writer PID470 was intentionally terminated for immediate replacement,
after local tests, review and synthetic GPU parity passed. Its wrapper's exit-15
measurement is expected cutover evidence, not an unexplained study failure. Cutover
preserved1055committedROIs; an in-flight uncommitted ROI can be retried.

Deployed checkout remains `/root/rivermind-fs/trustmask-runner-673810f` with the new
operations module added. Scientific files and environment were not altered.
Operational supervisor `parallel-full.py` (observed PID743) launches
`python -m research.operations.parallel_batches --workers 2 --max-rois 7477` with
the same source/model/protocol/assignments/study paths and OMP/MKL threads4.
It runs through the final study result. On nonzero parallel exit it automatically
tries the unchanged serial CLI from the same committed prefix, without resetting
the study or asking for batch approval. A serial fallback failure still requires
investigation; do not claim automatic recovery from arbitrary data/protocol errors.

Current log: `parallel-full.log`; final operational measurement:
`parallel-full-measurement.json`, both under the deployed checkout. Final scientific
report remains `/root/rivermind-fs/trustmask-public-crosssensor-v1/result.json`.
Do not monitor the obsolete full-study.log as the live task, or launch a second writer.
Keep the GPU powered on until all inference and final report validation complete.

## Local validation

28 scoped tests passed (new scheduler, batch, pipeline and CLI), including ordered
and bounded prefetch, role barriers, wrong-request rejection, exact remaining budget,
terminal guard mismatch and resumed end-to-end parity with the reference pipeline.
Ruff and both old/new frozen-protocol gates passed. Independent code review found
no blockers. A writer race between cutover inspection and batch lock acquisition
fails closed through lock/order validation; it does not silently process other IDs.

GPU memory usage is not a throughput measure. Two workers can saturate this GPU
while leaving memory free; allocating the remaining memory does not itself shorten
execution. Real throughput should be judged from fresh receipt intervals, and
remaining time estimates must allow different stage postprocessing costs.

## Verified real throughput checkpoint

At1084totalcommittedROIs,29were produced by the two-process execution. Excluding
the first two outputs, observed mean receipt interval6.94135seconds/ROI versus the
pre-probe serial9.90840seconds/ROI, throughput1.42745x (about30% less time per ROI).
All1055pre-cutover receipt SHA256s verified unchanged. The latest receipt was less
than one second old; GPU sampled100%utilization,11353MiB memory and333.55W.
The coordinator remains in development_validation; terminal test has not started.
There was no scientific criterion used to select this scheduling change.

The remaining6393ROIs project to12.33hours at this short-window observed rate,
compared with17.60hours at the earlier serial rate. This is a workload scheduling
estimate, not a fixed completion deadline; formalcal and terminal postprocessing
can have different costs. The continuing run has not completed. Verified metadata:
`research/evidence/trustmask-parallel-cutover-v1.json`.
