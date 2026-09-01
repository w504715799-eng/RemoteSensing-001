# Task 1 Report: Select Exactly 120 Development ROI

## Implementation

- Added `select_development_records(records)` to `src/trustsr/data/crosssensor_pairs.py`.
- The selector preserves input/manifest order, filters to `split == "development"`, requires exactly 120 records, validates unique `sample_id` and `spatial_group_id`, and validates all 12 `(days_between, correlation_bin)` cells contain selection rounds 1 through 10.
- Added frozen development constants and synthetic 360-record coverage plus broken-cell parametrized tests in `tests/data/test_crosssensor_pairs.py`.

## RED evidence

Command:

```text
uv run pytest tests/data/test_crosssensor_pairs.py -k select_development_records -q
```

Expected failure observed during collection:

```text
ImportError: cannot import name 'select_development_records' from 'trustsr.data.crosssensor_pairs'
```

## GREEN evidence

Focused selector tests: `7 passed`.

Module tests: `34 passed`.

Ruff:

```text
uv run ruff check src/trustsr/data/crosssensor_pairs.py tests/data/test_crosssensor_pairs.py
All checks passed!
```

Full suite:

```text
uv run pytest -q
completed at 100% with exit code 0; no test failures (43 warnings from torch.jit deprecation)
```

## Self-review

The implementation uses tuple output and does not reorder records. It rejects missing/extra development rows, duplicate identities, invalid day/bin strata, duplicate rounds, and out-of-range rounds. Calibration and internal-test records are never selected or validated as development cells.

## Concerns

No known functional concerns. Existing `_require_unique_strings` messages refer to smoke records; development identity failures are wrapped with development context to satisfy the development-selection contract.
