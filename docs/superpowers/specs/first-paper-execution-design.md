# First-paper external execution entry

2026-09-08. Authorized continuation of Task 5; implementation is not authorization to read Spain.
Work sequentially on main; scoped tests only. Do not repeat completed GPU/development diagnostics.

## Protocol and entry

`external_protocol.py` derives an exact scientific payload from pinned local text metadata, timing
and cloud-policy JSON: both 100 packages, all 28/20 projected members, seven complete model
provenances, normalization, score/risk/threshold/structure policies, descriptive statistics,
failure and display rules. Bind all repository Python implementation files under src/trustsr and
scripts/paper/run_spain.py by hashes. The draft contains unresolved runtime/budget fields and is
not runnable. No science/seed/model/subset overrides on the execution CLI.

`python -m scripts.paper.run_spain draft --output ...` writes a no-overwrite draft without image,
model, network or cache access. `run` / `replay` require an externally reviewed protocol SHA,
status frozen, complete runtime version inventory and finite budget, and explicit external-access
acknowledgement. A user-supplied SHA is the reviewed trust anchor, not a cryptographic signature.
The final freeze/authorization remains a separate research gate; never change draft status merely
to make preflight pass. All code/model/member identities are checked before image access.

Only local already-acquired exact package filenames are accepted. No download fallback for data
or model assets. Run output must be an independent directory outside repository/package/model
roots, not an existing unrelated directory or symlink. A marker binds it to the protocol SHA;
cache is always output/cache, not an arbitrary existing historical cache. flock serializes work.

## Prediction and recovery

Authenticate all inputs and existing cache slots before model construction. Bind normalized input
hashes (both LR and HR), sources, members, model identity and ordered slots to resumable state.
Only construct a model for an absent, never-attempted slot; require exact provenance before predict.
Models receive LR only. Load LDSR once and use for_seed; use CloudSEN2SRLiteX4 on CPU.
Missing LDSR requires separately supplied GPU acknowledgement and available CUDA before any model
is loaded. Existing complete caches can be processed without CUDA or model directories.

Persist a started record before every call, then stored/hash/timing or a static failure category.
RuntimeError, MemoryError and output contract failures are per-prediction failures. Asset/provenance,
input/cache identity, storage and protocol errors stop the run. No implicit repair or overwrite.
Resume never retries a started/failed attempt with no cache: record interrupted/failed as missing;
if a started attempt left a complete valid cache, accept recovery without inference. Stored/cache-hit
outputs missing later are integrity errors. No cache mutation after science publication.

## Evaluate and publish

Process both subsets with existing replay code and independent denominators. Include exact
prediction failure reasons beside science (no exception messages/paths). Perform a second
cache-only reconstruction and require canonical scientific equality before publication. Use
new run directory science.json + verification.json, a small manifest written last as the commit
marker. All retain protocol binding; no automatic Git copy. Existing different outputs fail;
identical partial publication may be finished after interruption. Completed run is immutable.
Separate replay only authenticates/recomputes/compares and never loads a model or writes results.
Scientific files omit durations; runtime timings remain in state.json and are not source guarantees.

## Implementation sequence and verification

1. Protocol derivation, frozen gate and draft CLI; synthetic/text-only negative gate tests.
2. Missing-only executor + restart/failure/integrity tests using tiny synthetic pairs and real cache.
3. Formal run/replay orchestration + no-overwrite publication; synthetic integration and tripwires.
4. Generate draft, update runbook/handoff, focused regression and changed-file lint; read-only review.
5. Cloud CPU timing/runtime/price inputs and final protocol freeze remain pending; no Spain access.
