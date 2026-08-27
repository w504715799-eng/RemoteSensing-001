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

## Phase 1A

Run the deterministic CPU comparison of bicubic interpolation and the verified
SEN2SRLite RGBN ×4 pretrained model with:

```bash
uv sync --dev
uv run trustsr-benchmark
```

The first run downloads OpenSR-Test SPOT v3 and the pinned SEN2SRLite assets, verifies
their hashes before loading, and fills the prediction cache. This is intended for a
local CPU development check; the nine-sample run can take several minutes depending on
network and CPU. Subsequent runs reuse predictions from
`artifacts/cache/predictions/`. The model download is stored in
`models/SEN2SRLite_RGBN/`, and the deterministic result is
`artifacts/phase1/spot-v3-baselines.json`. All three locations are intentionally
untracked. Path overrides are available through `trustsr-benchmark --help`.

SPOT is a development reproducibility check, not final scientific evidence. The future
LDSR-S2 GPU phase is intentionally separate from this CPU checkpoint.

## Phase 1B remote GPU runbook

The server remains off through Tasks 1–6. These scripts are an operator runbook
only; this repository does not connect to a server, contain SSH authentication
material, or start GPU work by itself. Configure an SSH alias in your local SSH
configuration outside this repository. Do not put SSH authentication material in
commands or committed files: an SSH password or key never belongs in a command
or the repository.

When the approved GPU instance is available, clone this exact checked-out commit
onto the verified persistent data disk and run the following commands *on that
instance*. The only accepted production root is
`/root/rivermind-fs/trustsr-phase1b`; bootstrap refuses any other resolved
location and requires at least 15 GiB free before it changes anything. Do not
run a stage until `/root/rivermind-fs` is an actual mountpoint. Before model
construction, the workflow requires CUDA, exactly one visible GPU, a numeric
`major.minor` compute capability of at least `8.0`, at least 18 GiB free VRAM,
and no foreign CUDA compute process. The manifest preserves the actual GPU name,
UUID, driver, memory, and capability. Compare durations only among runs on the
recorded same hardware; deterministic hashes, repeatability, and quality metrics
remain comparable scientific gates.

```bash
scripts/phase1b/bootstrap_remote.sh /root/rivermind-fs/trustsr-phase1b "$PWD"
scripts/phase1b/run_remote.sh /root/rivermind-fs/trustsr-phase1b preflight
scripts/phase1b/run_remote.sh /root/rivermind-fs/trustsr-phase1b single
scripts/phase1b/run_remote.sh /root/rivermind-fs/trustsr-phase1b benchmark
scripts/phase1b/run_remote.sh /root/rivermind-fs/trustsr-phase1b manifest
```

The approved cloud image already supplies the fixed `/opt/conda/bin/python` with
Python 3.12 and its CUDA-enabled PyTorch stack. Bootstrap deliberately reuses that
base interpreter: it first verifies Python, PyTorch, torchvision, and CUDA; obtains
a structured pip dry-run report and refuses any proposed PyTorch/CUDA-stack change;
then installs only `uv==0.12.5` and this editable project with its `gpu` extra using
`only-if-needed`. It records the `uv.lock` digest and actual verified package/CUDA
fingerprint in a non-secret provenance stamp under the approved persistent root.
After installation it requires the exact same PyTorch/torchvision/CUDA fingerprint,
CUDA availability, `opensr-model==1.1.1`, `uv==0.12.5`, a TrustSR import, and a
clean `pip check`. This is not a fully frozen isolated environment: the cloud image
is the base dependency source, so actual package and hardware provenance is required
for every run. Any existing partial `conda-env` directory is intentionally ignored
and left untouched.
Budget at least 15 GiB free disk space before bootstrap, including the verified
approximately 1.13 GB LDSR-S2 checkpoint and downloaded/developed outputs.

The scientific settings are immutable: LDSR-S2 uses seed 3407 and 100 sampling
steps (with eta 0.95, temperature 1.0, and histogram matching). SPOT v3 is a
development-only reproducibility dataset, not final scientific evidence. The
stages are intentionally separate: preflight verifies the CUDA model, single
runs only `spot-0000` with the repeatability gate, benchmark uses all fixed nine
SPOT samples and the fixed three-model order, and manifest allowlists their
artifacts.

Before telling anyone to stop the instance, pull and verify the artifacts from a
local checkout using the SSH config alias:

```bash
scripts/phase1b/pull_artifacts.sh phase1b-gpu /root/rivermind-fs/trustsr-phase1b ./artifacts/remote-phase1b
```

The puller retrieves the manifest first, transfers only its listed paths with
protected arguments, then verifies local file digests from the local checked-out
code. Only after that verification succeeds should the operator stop the
instance, using the cloud provider console. No shutdown action belongs in these
scripts.
