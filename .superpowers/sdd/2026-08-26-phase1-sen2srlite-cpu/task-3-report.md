# Task 3 report: identity-bound safe prediction caching

## RED/GREEN evidence

The requested initial test run was expected to fail because the artifacts module did not exist (the test file and implementation were introduced together in this worktree). After implementation, `uv run pytest tests/artifacts/test_predictions.py -q` passes (8 tests), the full suite passes (42 tests), and `uv run ruff check .` passes.

## API

- `build_identity(model_provenance, source, sample_id, lr)` returns a `PredictionIdentity`; its `key`/`cache_key` is SHA-256 over canonical JSON containing scalar provenance, source, sample ID, LR shape, dtype, and contiguous CPU-byte digest.
- `PredictionCache(root).put(identity, prediction)` (aliases `store`/`save`) validates and atomically writes `<key>.safetensors` followed by `<key>.json`.
- `PredictionCache(root).get(identity)` (alias `load`) returns an exact CPU float32 prediction, `None` for a miss, and raises `CacheIntegrityError` for a committed but invalid entry.

Canonical JSON is UTF-8, sorted-key, compact, and `allow_nan=False`; only safetensors and JSON are used. Temporary siblings are cleaned on success and failure.

## Changed files

- `src/trustsr/artifacts/__init__.py`
- `src/trustsr/artifacts/predictions.py`
- `tests/artifacts/test_predictions.py`

## Self-review

The JSON sidecar is the commit marker: tensor replacement precedes metadata replacement. Tensor-only orphans are misses, while sidecar-present corruption, identity mismatch, malformed safetensors, shape/dtype/range failures, and digest mismatches raise `CacheIntegrityError`. Metadata contains no paths, timestamps, pickle, or `.pt` artifacts.

## Concerns

No known concerns for the specified contract. Cache directory creation is intentionally limited to the configured root.

## Fix round (review findings)

Implemented explicit close/flush/fsync for tensor and metadata temporary files, directory fsync after each replacement, and retained JSON-last commit ordering. Identity provenance is now defensively copied into an immutable mapping. Added coverage in `tests/artifacts/test_predictions.py`:

- `test_mismatched_identity_is_integrity_error`
- `test_truncated_tensor_is_integrity_error`
- `test_tensor_digest_mismatch_is_integrity_error`
- `test_identity_defensively_copies_and_rejects_non_scalar_provenance`
- `test_metadata_has_no_paths_timestamps_pickle_or_temporary_files`

Commands and output:

```text
$ uv run pytest tests/artifacts/test_predictions.py -q
.............                                                            [100%]

$ uv run pytest -q
...............................................                          [100%]
43 warnings (torch.jit deprecation)

$ uv run ruff check .
All checks passed!
```

## Fix round 2

`PredictionIdentity.__post_init__` now routes every public construction through provenance scalar validation and a defensive immutable copy. Added `test_public_identity_constructor_enforces_immutable_scalar_provenance`, covering direct mutable and nested provenance construction, stable keys, and defensive copying.

Exact verification:

```text
$ uv run pytest tests/artifacts/test_predictions.py -q
..............                                                           [100%]

$ uv run pytest -q
................................................                         [100%]
43 warnings (torch.jit deprecation)

$ uv run ruff check .
All checks passed!
```
