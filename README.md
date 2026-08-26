# TrustSR

Incremental experiments for trustworthy Sentinel-2 RGBN ×4 super-resolution.

## Phase 0

```bash
uv sync --dev
uv run pytest
uv run trustsr-smoke --dataset spot --limit 2
```

The SPOT run is a development smoke test, not a final scientific result.
