# External authenticated replay and partial-method accounting

2026-09-08. Task 5 implementation, no external access authorization or final protocol.

`external_replay.replay_roi(pair, cache, provenances)` reads seven identity-bound cache slots:
LDSR seeds 3407..3411, bicubic, and the cloud 96-thread SEN2SRLite profile. Build keys from the
actual normalized LR, source, ROI and caller's protocol-bound model provenance. Require exact
slots, consistent LDSR provenance except seed and correct integer seed per slot. The eventual
frozen CLI must authenticate the provenance manifest; this core does not independently establish
model lineage or infer that self-consistent cache bytes were produced by a model.

Integrity errors stop the batch and are never converted to missing predictions. Absent caches
are explicit per-slot missing records. No inference, writes, retries, downloads or fallback.
Available center enables LR and random; center+bicubic+SEN2SRLite enables three-model; all K5
seeds enable K5, neighborhood and fixed-threshold transfer. Missing SEN2SRLite must not suppress
K5–LR. Missing a non-center seed must preserve LR/random/three-model. Missing center invalidates
all methods. Reuse existing formulas, risk grids, coverage points and field-name conversion.

`summary(members, rows)` requires exactly one row for every planned ROI, and for each R1/R9
method exactly one diagnostic or nonempty failure. Report each method's planned/valid/failure
counts and complete reasons, paired primary differences only where LR and K5 both exist, and
null for an absent mean. Preserve deterministic ROI order and equal-ROI weighting. Transfer has
its own valid/planned denominator. Whole input/identity contract failures are not silently dropped.

Synthetic tests compare a complete replay to the existing array core and assert partial dependency
behavior, missing denominators, corrupted cache abort, LR changes preventing old cache reuse,
seed swaps and old cloud-policy rejection. No new tests on unrelated historical modules.
