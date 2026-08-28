# TrustSR

Incremental experiments for trustworthy Sentinel-2 RGBN ×4 super-resolution.

## Phase 0

Set up the CPU development environment, then run the reproducible bicubic baseline:

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run trustsr-smoke --dataset spot --version v3 --limit 2
```

This downloads only the public OpenSR-test SPOT v3 development dataset and evaluates
two RGBN samples on CPU. It does not download NAIP or Spain, and it does not use a GPU.
The command writes `artifacts/phase0/bicubic-spot-v3.json`, including the sample
manifest hash, code revision, runtime versions, device, per-sample metrics, and mean
metrics.

Generated artifacts and downloaded data are intentionally untracked. Phase 0 is
complete only when tests, linting, the two-sample run, and deterministic replay all
pass. Learned models are introduced in a separate Phase-1 plan after this checkpoint
is reviewed.

The SPOT run is a development smoke test, not a final scientific result. Conda is
reserved for a later cloud-GPU phase; use `uv run` for this checkpoint.
