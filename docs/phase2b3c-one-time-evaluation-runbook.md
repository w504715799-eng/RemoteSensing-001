# Phase 2B3-C one-time internal-test evaluation runbook

Phase 2B3-C evaluates the one frozen Phase 2B3-B rule exactly once on the 120 frozen
`internal_test` ROIs. It does not tune or compare thresholds. The immutable operating point is:

- threshold `7.970395366024563e-06`, inclusive;
- score `ldsr_variance_k5`, seeds `3407..3411`, population variance and RGBN mean;
- risk `local_l1_risk(window=9)`, upper bound `1.0`;
- risk target `alpha=0.05`, minimum aggregate pixel coverage `0.10`;
- grid-Kelly confidence error `0.05`, grid size `20`, and 128 bisections; and
- exactly 120 `internal_test` records in frozen post-manifest order.

None of these values is a CLI option. Do not edit, round, refit, relax, or replace them after any
holdout pixel, test cache, prediction, score, risk, metric, aggregate, or decision is observed.

## 1. Local-only readiness gate

Complete Tasks 1–11 on local CPU before any real Phase 2B3-C command. The final gate is:

```bash
uv run pytest -q \
  tests/cli/test_phase2b3c.py tests/cli/test_phase2b3c_verify.py \
  tests/evaluation/test_phase2b3c_*.py \
  tests/evaluation/test_internal_test_*.py \
  tests/data/test_internal_test_subset.py tests/data/test_internal_test_pairs.py \
  tests/data/test_local_data_policy.py tests/models/test_ldsr_s2.py tests/risk/test_local.py
uv run ruff check \
  src/trustsr/cli/phase2b3c.py src/trustsr/cli/phase2b3c_verify.py \
  src/trustsr/data/internal_test_subset.py src/trustsr/data/internal_test_pairs.py \
  src/trustsr/evaluation/phase2b3c_*.py src/trustsr/evaluation/internal_test_*.py \
  tests/cli/test_phase2b3c.py tests/cli/test_phase2b3c_verify.py \
  tests/data/test_internal_test_*.py tests/evaluation/test_phase2b3c_*.py \
  tests/evaluation/test_internal_test_*.py
uv run python -m compileall -q src
uv run trustsr-phase2b3c --help >/dev/null
uv run trustsr-phase2b3c preflight --help >/dev/null
uv run trustsr-phase2b3c evaluate --help >/dev/null
uv run trustsr-phase2b3c evaluation-replay --help >/dev/null
uv run trustsr-phase2b3c-verify --help >/dev/null
git diff --check
git status --short --branch
git rev-parse HEAD
```

The checkout must be clean, attached to `main`, and committed. These tests use generated metadata,
tiny CPU tensors, test-only permit capabilities, and temporary ledgers. They neither authorize nor
perform real `internal_test` access, test-cache inspection, LDSR construction, CUDA inspection, or
real ledger advancement.

## 2. Mandatory readiness and authorization stop

After the local gate, the coordinator may run one metadata-only real preflight to produce the
readiness document outside Git. Preflight may validate the complete manifest and exact metadata
membership; it must not open image assets, enumerate test caches, construct LDSR, query CUDA,
create a ledger, publish results, or create a permit.

Record the readiness SHA-256, evaluation ID, implementation revision, computation-tree digest,
and local gate results. Then stop and ask the user for explicit authorization dedicated to the
one-time Phase 2B3-C real-data evaluation. A routine instruction to continue with recommended
steps is not authorization. No permit or ledger may exist at this point.

Only after that dedicated authorization may the reviewed canonical permit be written at:

```text
artifacts/phase2b3c/sen2naipv2-internal-test-access-authorization-v1.json
```

Review its exact readiness, evaluation, code-tree, upstream evidence, manifest, policy, and
authorization bindings; commit only that permit. A normal command never creates or approves it.

## 3. Fixed remote paths and interpreter

On the machine holding persistent data, use the already installed base interpreter only:

```bash
PHASE2B3C_PROJECT_ROOT=/absolute/path/to/clean/RemoteSensing001
PHASE2B3C_STORAGE_ROOT=/absolute/path/to/persistent/storage
PHASE2B3C_PYTHON=/opt/conda/bin/python
PHASE2B3C_EVIDENCE_DIR="$PHASE2B3C_PROJECT_ROOT/artifacts/phase2b3b"
PHASE2B3C_POST_MANIFEST_SHA256=c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a
PHASE2B3C_MANIFEST="$PHASE2B3C_STORAGE_ROOT/trustsr/phase2b1b/selections/$PHASE2B3C_POST_MANIFEST_SHA256/samples.jsonl"
PHASE2B3C_PERMIT="$PHASE2B3C_PROJECT_ROOT/artifacts/phase2b3c/sen2naipv2-internal-test-access-authorization-v1.json"
PHASE2B3C_BUNDLE="$PHASE2B3C_STORAGE_ROOT/trustsr/phase2b3c/bundles/$PHASE2B3C_POST_MANIFEST_SHA256"
PHASE2B3C_COMMON_ARGS=(
  --project-root "$PHASE2B3C_PROJECT_ROOT"
  --evidence-dir "$PHASE2B3C_EVIDENCE_DIR"
  --storage-root "$PHASE2B3C_STORAGE_ROOT"
  --manifest "$PHASE2B3C_MANIFEST"
  --access-permit "$PHASE2B3C_PERMIT"
  --confirm-persistent-storage
)
```

Do not run remote `uv`, use or create `.venv`, create an environment, install packages, or modify
dependencies. The storage root must be absolute, canonical, non-symlinked, persistent, and have
more than 10 GiB free. All formal stages use one exclusive non-blocking lock.

## 4. Access ledger and consumption boundary

The immutable digest-chained ledger is stored below the persistent root and follows:

```text
reserved -> pixels_opened -> caches_complete -> bundle_complete -> accepted
                                      \-> invalidated
```

The workflow writes `pixels_opened` immediately before the first test image or test-cache read.
Once that event exists, the one-time access is consumed even if the process crashes. Never delete,
reset, replace, truncate, copy to another storage root, or manually advance ledger events.

Before `pixels_opened`, an exact-identity restart is allowed. Afterwards, resume only with the same
permit, evaluation ID, code tree, dependencies, scientific configuration, membership, and storage
root. Any incompatible code/science change, second evaluation, alternate storage root, or output
mismatch invalidates the evaluation. A defect discovered after access may be documented but not
fixed and rerun as the same confirmation.

After `pixels_opened`, logs may contain only opaque digests, progress counts, and state transitions.
Do not print per-ROI or per-stratum values, partial coverage, partial loss, running confidence
bounds, intermediate metrics, or intermediate decisions.

## 5. Authorized CPU cache probe and evaluation

Only after the reviewed permit commit, run `evaluate` without a model path:

```bash
cd "$PHASE2B3C_PROJECT_ROOT"
PYTHONPATH=src "$PHASE2B3C_PYTHON" -m trustsr.cli.phase2b3c \
  evaluate "${PHASE2B3C_COMMON_ARGS[@]}"
```

The command loads only the frozen 120 test pairs, records input authority, and probes exactly 600
K5 prediction identities before any model construction. If all 600 entries verify, it completes on
CPU: fixed maps, ROI metrics, grid-Kelly statistics, diagnostics, result, cache audit, runtime,
immediate independent cache reconstruction, replay receipt, ledger snapshot, and external bundle.
Do not start or inspect a GPU.

If the exact probe reports any missing entries, stop before LDSR construction. Report the precise
present/missing counts and request separate GPU startup/usage authorization. Only after that second
authorization may a reviewed model directory be supplied:

```bash
PHASE2B3C_LDSR_MODEL_DIR=/absolute/path/to/verified/ldsr/model
PYTHONPATH=src "$PHASE2B3C_PYTHON" -m trustsr.cli.phase2b3c \
  evaluate "${PHASE2B3C_COMMON_ARGS[@]}" \
  --ldsr-model-dir "$PHASE2B3C_LDSR_MODEL_DIR"
```

This is the sole surface that may construct LDSR or generate only missing exact entries. No device
or worker override exists. Individually atomic cache entries may be reused only after complete
identity and tensor-digest verification.

## 6. Explicit inference-free replay

After `bundle_complete`, replay with no model path:

```bash
PYTHONPATH=src "$PHASE2B3C_PYTHON" -m trustsr.cli.phase2b3c \
  evaluation-replay "${PHASE2B3C_COMMON_ARGS[@]}"
```

Replay reloads the same authorized pairs and complete K5 cache, independently recomputes every
downstream map, ROI statistic, aggregate, diagnostic, audit, result, and runtime projection, and
requires byte-identical bundle evidence. It cannot construct LDSR or run inference.

## 7. Copied-bundle independent verification and publication

Copy the completed external bundle to a separate canonical directory, then run the verifier:

```bash
PHASE2B3C_VERIFY_PARENT=/absolute/path/to/independent/verification
PHASE2B3C_COPIED_BUNDLE="$PHASE2B3C_VERIFY_PARENT/phase2b3c-bundle"
mkdir -p "$PHASE2B3C_VERIFY_PARENT"
test ! -e "$PHASE2B3C_COPIED_BUNDLE"
cp -a -- "$PHASE2B3C_BUNDLE" "$PHASE2B3C_COPIED_BUNDLE"
PYTHONPATH=src "$PHASE2B3C_PYTHON" -m trustsr.cli.phase2b3c_verify \
  --bundle "$PHASE2B3C_COPIED_BUNDLE" \
  "${PHASE2B3C_COMMON_ARGS[@]}"
```

The verifier reacquires the formal lock, rejects the producer bundle before authority or pixel
reads, requires the exact `bundle_complete` ledger, independently reloads inputs, requires the
complete 600-entry cache, and recomputes without inference. It publishes before appending the
terminal `accepted` event; a publication failure must not advance the ledger.

The permit already occupies `artifacts/phase2b3c/`, so POSIX cannot add the three result names with
one directory rename. Publication therefore provides serialized, descriptor-hardened all-or-clean
failure semantics under the formal and permit locks, not a single external visibility point for
uncooperative readers. Each no-replace destination is inode-bound for rollback. Partial, extra,
symlinked, raced, or different-byte state fails closed, and the permit is never rolled back.

The only three result files are:

```text
sen2naipv2-internal-test-evaluation-v1.json
sen2naipv2-internal-test-evaluation-cache-audit-v1.json
sen2naipv2-internal-test-evaluation-acceptance-v1.json
```

The runtime, replay, ledger snapshot, bundle manifest, tensors, caches, models, paths, hosts,
endpoints, credentials, raw timestamps, and per-sample numerical metrics stay outside Git.

## 8. Review, decision, and shutdown

Run the tracked-data and Phase 2B3-C publication policy scans, review canonical bytes and digests,
and commit the permit plus three result files only according to the reviewed publication procedure.
The acceptance contains exactly one final preregistered decision:

- `confirmed`: coverage is at least `0.10` and the grid-Kelly UCB is at most `0.05`;
- `empirically_met_but_inconclusive`: coverage is at least `0.10` and mean loss is at most `0.05`,
  but the finite-sample UCB exceeds `0.05`; or
- `failed`: coverage is below `0.10` or empirical mean loss exceeds `0.05`, with exact reasons.

All three decisions permanently complete the one-time phase after acceptance. Only `confirmed`
supports a confirmation claim. Never rescue another outcome by changing the threshold, target,
coverage gate, confidence method, seeds, score, risk, sample membership, or weighting.

On any evidence, Git, permit, ledger, membership, asset, tensor, cache, model, computation, replay,
runtime, publication, canonical JSON, path, or digest mismatch, stop. If access was consumed and the
state cannot validly resume, append `invalidated` through the supported workflow and document the
reason without exposing protected values. Shut down the GPU/server when no longer required; if the
600-entry cache was complete, no GPU should have been started.
