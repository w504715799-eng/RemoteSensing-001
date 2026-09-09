# Parallel execution amendment (scientific definition unchanged)

User explicitly requests reducing elapsed GPU time through parallel computation.
Keep the current serial run active during preparation. Do not modify any file bound
by the scientific protocol, discard receipts, change seeds/dtypes, or use test data
for the performance decision.

Implement an operational wrapper under research/operations, outside the frozen
scientific implementation. Two spawn-isolated CUDA processes each call the existing
LocalTacoProvider unchanged. One coordinator calls existing run_batch and commits
in its original order. Only the currently requested role may be prefetched, with
at most two outstanding ROI jobs; test scheduling requires the existing frozen
methods and test-start ledger. No thread-shared random state and no tensor batching.

The original journal binding remains intact. Record an explicit, durable execution
amendment binding wrapper hash, base journal/protocol hash, worker count and receipt
prefix at cutover. This changes the operational one-resident-ROI policy to bounded
concurrent ROIs, not the scientific calculation or historical evidence.

Tests: ordered delivery, bounded pending jobs, role barriers, exact budget, no
completed-ROI rereads, identity failure, terminal ledger guard and serial reference
parity using synthetic providers. On cloud, compare deterministic synthetic LR
predictions serial vs concurrent using exact hashes, never reopening existing data.
Switch only after local validation. Record serial throughput, concurrent throughput
and resource observations. If no material speedup or any error, resume serial with
all committed receipts preserved. When valid and faster, run continuously to the
existing final result, with no batch/role approval pause.
