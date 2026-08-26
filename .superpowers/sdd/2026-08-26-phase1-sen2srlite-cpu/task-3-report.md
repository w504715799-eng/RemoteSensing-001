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
