# Task 8g: Runtime version report

## Change

- `_command_version()` now returns the dotted numeric version token following
  the allowlisted command name, rather than the final whitespace token.
- `collect_gpu_environment()` regression coverage records the real outputs:
  `uv 0.12.5 (x86_64-unknown-linux-gnu)` as `0.12.5`, and `conda 26.5.3` as
  `26.5.3`.
- Malformed version tokens remain `unavailable`.

## TDD evidence

- RED: `uv run pytest tests/artifacts/test_gpu_run.py -q` failed before the
  implementation change: uv was recorded as `(x86_64-unknown-linux-gnu)`;
  malformed conda and uv tokens were also returned unchanged.
- GREEN: the same focused command passed after the parser change (36 tests).

## Final verification

- `uv run pytest tests/artifacts/test_gpu_run.py -q` — passed (36 tests).
- `uv run pytest -q` — passed; only existing Torch deprecation warnings were
  emitted.
- `uv run ruff check .` — passed.
- `git diff --check` — passed.

## Self-review

The manifest schema, GPU capability/provenance data, and fixed command argv
behavior are unchanged. The parser is still used only for the existing
allowlisted conda and active-prefix uv commands.

## Fix round 1

Review identified that the parser validated the version token but not the
command-name token. It now receives the expected allowlisted name from each
existing caller and returns `unavailable` unless the output begins with that
name and has a dotted numeric version token.

### TDD evidence

- RED: `uv run pytest tests/artifacts/test_gpu_run.py -q` failed for
  `unexpected 1.2.3` and `unavailable 1.2.3`, which were incorrectly recorded
  as `1.2.3`.
- GREEN: the focused command passed after the expected-name check was added
  (42 tests).

### Added boundary coverage

The malformed-output matrix now covers wrong command names with valid version
tokens, empty output, literal `unavailable` output, and non-version tokens
through `collect_gpu_environment()`.

### Verification

- `uv run pytest tests/artifacts/test_gpu_run.py -q` — passed (42 tests).
- `uv run pytest -q` — passed; only existing Torch deprecation warnings were
  emitted.
- `uv run ruff check .` — passed.
- `git diff --check` — passed.
