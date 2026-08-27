# Task 8d: conda-forge remote bootstrap report

## Outcome

Remote bootstrap now creates its isolated Python prefix with the exact channel
selection `conda create --yes --override-channels --channel conda-forge --prefix PREFIX python=3.12 pip`.
It does not mutate Conda configuration or accept channel terms. Existing storage,
prefix compatibility, uv, frozen GPU sync, lock stamp, and remote GPU path behavior
remain unchanged.

## TDD evidence

### RED

Updated the fake-Conda executable contract in
`tests/scripts/test_phase1b_scripts.py` to reject any creation argv other than the
required exact array. Before changing production code:

```text
$ uv run --no-sync pytest -q tests/scripts/test_phase1b_scripts.py -k test_bootstrap_creates_only_a_prefix_and_runs_frozen_gpu_sync
F
AssertionError: unexpected conda create argv: create --yes --prefix .../conda-env python=3.12 pip
assert 99 == 0
```

The failure was caused by the old bootstrap command reaching the fake executable,
which is the intended policy regression.

### GREEN

Changed only the production Conda invocation to add the required
`--override-channels --channel conda-forge` arguments in the required order. The
same focused test then passed:

```text
$ uv run --no-sync pytest -q tests/scripts/test_phase1b_scripts.py -k test_bootstrap_creates_only_a_prefix_and_runs_frozen_gpu_sync
.
```

## Verification

- `uv run --no-sync pytest -q tests/scripts/test_phase1b_scripts.py` — 62 passed.
- `uv run --no-sync pytest -q` — 287 passed, 43 existing Torch deprecation warnings.
- `uv run --no-sync ruff check .` — passed.
- `bash -n scripts/phase1b/bootstrap_remote.sh` — passed.
- `bash -n scripts/phase1b/run_remote.sh` — passed.
- `bash -n scripts/phase1b/pull_artifacts.sh` — passed.
- `git diff --check` — passed.

## Changed files

- `scripts/phase1b/bootstrap_remote.sh`: pin prefix creation to conda-forge with
  `--override-channels`.
- `tests/scripts/test_phase1b_scripts.py`: fake Conda now enforces exact creation
  argv; bootstrap assertion records that contract.
- `README.md`: documents explicit conda-forge sourcing and no term acceptance.
- `docs/superpowers/specs/2026-08-27-phase1b-ldsr-gpu.md`: documents the same
  channel and ambient-policy boundary in the remote layout.
- `docs/superpowers/plans/2026-08-27-phase1b-ldsr-gpu.md`: updates the contract
  and core command.

## Self-review

- No remote connection, download, GPU execution, endpoint/auth/secret handling,
  or legal-term acceptance was added.
- The free-space check still precedes prefix creation and all writes.
- Existing-prefix compatibility remains fail-closed and does not recreate or
  mutate an incompatible prefix.
- `uv==0.12.5`, frozen `--no-dev --extra gpu` synchronization, and lock stamping
  are unchanged.
- `/root/rivermind-fs` confinement and the GPU capability gate are unchanged.
- No `conda config` or `conda tos accept` invocation exists in the bootstrap.
