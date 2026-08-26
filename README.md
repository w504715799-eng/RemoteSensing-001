# TrustSR

Incremental experiments for trustworthy Sentinel-2 RGBN ×4 super-resolution.

## Phase 0

Initialize the environment and run the package tests:

```bash
uv sync --dev
uv run pytest
```

The Phase 0 smoke-test target is planned for Task 6. After Task 6 is
complete, run it with:

```bash
uv run trustsr-smoke --dataset spot --limit 2
```

The SPOT run is a development smoke test, not a final scientific result.
