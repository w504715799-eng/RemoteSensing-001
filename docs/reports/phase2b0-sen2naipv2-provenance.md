# Phase 2B0 SEN2NAIPv2 provenance acceptance

## Frozen source

- Repository: `tacofoundation/SEN2NAIPv2`
- Frozen revision: `c370504201072fdb1dd388013ab8c0fc7d00a57e`
- License claim: `cc0-1.0`
- Data-card SHA-256: `5897aed9410fef305953ff5b34e83697b466901583b880158af2902a8267a58d`
- Git LFS object count: 9
- Total declared bytes: 149,356,128,592

## Local/cloud boundary

This checkpoint stores only validated local provenance metadata. Local SEN2NAIPv2 pixel
downloads and TACO files are forbidden; the tracked-file policy rejects `.taco` files,
non-JSON/Markdown files below `artifacts/datasets/`, and metadata files larger than
1,048,576 bytes. Cloud access and real sample extraction are deferred to Phase 2B1.

GPU required: no.

## Fresh verification evidence

The following commands were run from the repository root:

```bash
uv run pytest tests/data/test_local_data_policy.py -v
uv run pytest
uv run ruff check .
uv run trustsr-dataset-audit > /tmp/trustsr-phase2b0-audit-1.json
uv run trustsr-dataset-audit > /tmp/trustsr-phase2b0-audit-2.json
cmp /tmp/trustsr-phase2b0-audit-1.json /tmp/trustsr-phase2b0-audit-2.json
uv run trustsr-dataset-audit > /tmp/trustsr-phase2b0-accepted.json
sha256sum /tmp/trustsr-phase2b0-accepted.json
uv run trustsr-dataset-audit
uv run python -c 'from pathlib import Path; from trustsr.data.local_policy import tracked_data_policy_violations; assert not tracked_data_policy_violations(Path.cwd())'
git diff --check
git ls-files '*.taco'
```

- Local policy tests: 3 passed.
- Full test suite: 434 passed; 43 third-party PyTorch deprecation warnings.
- Ruff: all checks passed.
- Audit repeatability: the two audit files were byte-identical (`cmp` exit status 0).
- Accepted audit SHA-256: `2cb8aebf21942d3a9875c49439eebc8726ef1c62bd1118597ad35e1772bc1418`.
- Audit output states `metadata_only: true`, `network_accessed: false`, and
  `pixel_data_downloaded: false`.

## Final post-fix evidence

The final policy/configuration fix is commit `04c7b4b` (`fix: enforce dataset index
size policy`). The policy now derives tracked paths and staged blob sizes from the Git
index, checks a present working-tree file as well, and fails closed when index entries
or blob inspection are malformed. The ignore rules preserve generated artifact ignores
while explicitly allowing `artifacts/datasets/` metadata subject to the targeted
TACO/cache/pixel exclusions.

Fresh final commands and results:

- `uv run pytest tests/data/test_local_data_policy.py -v`: 6 passed.
- `uv run pytest`: 437 passed; 43 third-party PyTorch deprecation warnings.
- `uv run ruff check .`: all checks passed.
- Two `uv run trustsr-dataset-audit` outputs were byte-identical (`cmp` exit 0); their
  SHA-256 is `2cb8aebf21942d3a9875c49439eebc8726ef1c62bd1118597ad35e1772bc1418`.
- The repository-root policy assertion, `git diff --check`, and `git ls-files '*.taco'`
  all exited successfully with no output.
