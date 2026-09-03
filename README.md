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

## Phase 2A synthetic conformal smoke

Run the deterministic CPU-only conformal smoke command with:

```bash
uv run trustsr-conformal-smoke --alpha 0.27 --window 1
```

This command uses no real satellite data and provides no paper evidence or cross-sensor
guarantee. In schema `trustsr.conformal-smoke.v1`, a JSON `null` calibration threshold
means the internal threshold is `-inf`, the valid all-abstain sentinel.

## Phase 2B0 SEN2NAIPv2 provenance

Inspect the frozen, offline SEN2NAIPv2 provenance record with:

```bash
uv run trustsr-dataset-audit
```

Local SEN2NAIPv2 pixel downloads are forbidden. This checkpoint records metadata and
Git LFS object identities only; it does not download TACO files or extract real
samples. Cloud access and real sample extraction begin only in Phase 2B1.

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

If network access is unavailable, operators may stage the five pinned SEN2SRLite files
into the model directory by an available transport. Production loads a staged cache
only after every pinned SHA-256 digest verifies successfully.

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
run a stage until `/root/rivermind-fs` is an actual mountpoint; both remote
entry points enforce that mount before writing or installing. Before model
construction, the workflow requires CUDA, exactly one CUDA-visible device, a numeric
`major.minor` compute capability of at least `8.0`, at least 18 GiB free VRAM,
and no foreign CUDA compute process. The manifest preserves the actual GPU name,
UUID, driver, memory, capability, and validated visible-device count. Compare durations only among runs on the
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

## Phase 2B1A cloud crosssensor pilot

Phase 2B1A uses a cloud instance only for the legacy-reader installation and the
real crosssensor object. It does not use GPU computation. Start an instance with a
network connection and a persistent filesystem, set the two required values in the
shell, and run every command from a checked-out repository on that instance:

```bash
: "${PHASE2B1A_STORAGE_ROOT:?set this to the persistent filesystem mountpoint}"
: "${PHASE2B1A_TRANSPORT_URL:?set this to the explicit HTTPS transport URL}"
PHASE2B1A_SOURCE="$PWD/artifacts/datasets/sen2naipv2-source-v1.json"

phase2b1a_json_digest() {
  /opt/conda/bin/python -c 'import json, string, sys
value = json.load(sys.stdin)["digests"][sys.argv[1]]
if not isinstance(value, str) or len(value) != 64 or any(c not in string.hexdigits.lower() for c in value):
    raise SystemExit("stage output did not contain a lowercase SHA-256 digest")
print(value)' "$1"
}

scripts/phase2b1a/bootstrap_reader.sh "$PHASE2B1A_STORAGE_ROOT" "$PWD"
scripts/phase2b1a/run_cloud.sh "$PHASE2B1A_STORAGE_ROOT" "$PWD" download \
  --confirm-cloud-storage --source "$PHASE2B1A_SOURCE" \
  --transport-url "$PHASE2B1A_TRANSPORT_URL"

PHASE2B1A_MANIFEST_JSON="$(
  scripts/phase2b1a/run_cloud.sh "$PHASE2B1A_STORAGE_ROOT" "$PWD" manifest \
    --confirm-cloud-storage --source "$PHASE2B1A_SOURCE"
)"
PHASE2B1A_PRE_MANIFEST_SHA256="$(
  printf '%s\n' "$PHASE2B1A_MANIFEST_JSON" | phase2b1a_json_digest manifest_sha256
)"
PHASE2B1A_PRE_MANIFEST_PATH="${PHASE2B1A_STORAGE_ROOT%/}/trustsr/phase2b1a/manifests/${PHASE2B1A_PRE_MANIFEST_SHA256}/samples.jsonl"

PHASE2B1A_PILOT_JSON="$(
  scripts/phase2b1a/run_cloud.sh "$PHASE2B1A_STORAGE_ROOT" "$PWD" pilot \
    --confirm-cloud-storage --source "$PHASE2B1A_SOURCE" \
    --manifest "$PHASE2B1A_PRE_MANIFEST_PATH"
)"
PHASE2B1A_POST_MANIFEST_SHA256="$(
  printf '%s\n' "$PHASE2B1A_PILOT_JSON" | phase2b1a_json_digest manifest_sha256
)"
PHASE2B1A_POST_MANIFEST_PATH="${PHASE2B1A_STORAGE_ROOT%/}/trustsr/phase2b1a/manifests/${PHASE2B1A_POST_MANIFEST_SHA256}/samples.jsonl"

PHASE2B1A_AUDIT_JSON="$(
  scripts/phase2b1a/run_cloud.sh "$PHASE2B1A_STORAGE_ROOT" "$PWD" audit \
    --confirm-cloud-storage --source "$PHASE2B1A_SOURCE" \
    --manifest "$PHASE2B1A_POST_MANIFEST_PATH"
)"
PHASE2B1A_AUDIT_PATH="${PHASE2B1A_STORAGE_ROOT%/}/trustsr/phase2b1a/audits/${PHASE2B1A_POST_MANIFEST_SHA256}/phase2b1a-audit.json"
printf 'Pre-extraction manifest: %s\nPost-extraction manifest: %s\nAudit: %s\n' \
  "$PHASE2B1A_PRE_MANIFEST_PATH" "$PHASE2B1A_POST_MANIFEST_PATH" \
  "$PHASE2B1A_AUDIT_PATH"
```

`PHASE2B1A_STORAGE_ROOT` must be the mountpoint itself rather than a directory
inside it; both scripts reject home, root, symlink, relative, wildcard, and
newline paths and require strictly more than 15 GiB free. The runner deliberately
requires `--confirm-cloud-storage` for every stage. The bootstrap reuses only
Python 3.12 at `/opt/conda/bin/python`. It performs one combined dry-run over the
six legacy-reader pins and four direct CLI-runtime pins, refuses any transaction
that would alter the PyTorch/CUDA stack, and only then installs that same combined
snapshot. It neither installs the TrustSR project nor creates a Conda environment.
Its postcheck verifies all ten direct versions, imports the required scientific
stack, and proves that the Phase 2B1A CLI comes from the supplied checkout. The
pre-extraction and post-extraction manifest paths are intentionally distinct; the
audit always consumes the latter. Do not place cloud credentials or a transport
URL in repository files.

## Phase 2B1B cloud crosssensor research subset

Phase 2B1B reuses the verified Phase 2B1A source object and base environment. It
does not download the TACO object again, create a Conda environment, run a model,
or use GPU computation. Run the following commands on the cloud instance from the
exact reviewed repository checkout:

```bash
: "${PHASE2B1B_STORAGE_ROOT:?set this to the persistent filesystem mountpoint}"
PHASE2B1B_SOURCE="$PWD/artifacts/datasets/sen2naipv2-source-v1.json"
PHASE2B1B_BASE_MANIFEST_SHA256=7487b0af2ebef86910e918d5d6b2fb927a6f5e46bac7c2e30be7ffb2ce994482
PHASE2B1B_BASE_MANIFEST="${PHASE2B1B_STORAGE_ROOT%/}/trustsr/phase2b1a/manifests/${PHASE2B1B_BASE_MANIFEST_SHA256}/samples.jsonl"

phase2b1b_json_digest() {
  /opt/conda/bin/python -c 'import json, string, sys
value = json.load(sys.stdin)["digests"][sys.argv[1]]
if not isinstance(value, str) or len(value) != 64 or any(c not in string.hexdigits.lower() for c in value):
    raise SystemExit("stage output did not contain a lowercase SHA-256 digest")
print(value)' "$1"
}

PHASE2B1B_SELECT_JSON="$(
  scripts/phase2b1b/run_cloud.sh "$PHASE2B1B_STORAGE_ROOT" "$PWD" select \
    --confirm-cloud-storage --source "$PHASE2B1B_SOURCE" \
    --base-manifest "$PHASE2B1B_BASE_MANIFEST"
)"
PHASE2B1B_PRE_MANIFEST_SHA256="$(
  printf '%s\n' "$PHASE2B1B_SELECT_JSON" | \
    phase2b1b_json_digest selection_manifest_sha256
)"
PHASE2B1B_PRE_MANIFEST="${PHASE2B1B_STORAGE_ROOT%/}/trustsr/phase2b1b/selections/${PHASE2B1B_PRE_MANIFEST_SHA256}/samples.jsonl"

PHASE2B1B_EXTRACT_JSON="$(
  scripts/phase2b1b/run_cloud.sh "$PHASE2B1B_STORAGE_ROOT" "$PWD" extract \
    --confirm-cloud-storage --source "$PHASE2B1B_SOURCE" \
    --selection-manifest "$PHASE2B1B_PRE_MANIFEST"
)"
PHASE2B1B_POST_MANIFEST_SHA256="$(
  printf '%s\n' "$PHASE2B1B_EXTRACT_JSON" | \
    phase2b1b_json_digest selection_manifest_sha256
)"
PHASE2B1B_POST_MANIFEST="${PHASE2B1B_STORAGE_ROOT%/}/trustsr/phase2b1b/selections/${PHASE2B1B_POST_MANIFEST_SHA256}/samples.jsonl"

PHASE2B1B_AUDIT_JSON="$(
  scripts/phase2b1b/run_cloud.sh "$PHASE2B1B_STORAGE_ROOT" "$PWD" audit \
    --confirm-cloud-storage --source "$PHASE2B1B_SOURCE" \
    --selection-manifest "$PHASE2B1B_POST_MANIFEST"
)"
PHASE2B1B_AUDIT="${PHASE2B1B_STORAGE_ROOT%/}/trustsr/phase2b1b/audits/${PHASE2B1B_POST_MANIFEST_SHA256}/phase2b1b-audit.json"
printf 'Pre-extraction sidecar: %s\nPost-extraction sidecar: %s\nAudit: %s\n' \
  "$PHASE2B1B_PRE_MANIFEST" "$PHASE2B1B_POST_MANIFEST" "$PHASE2B1B_AUDIT"
```

`PHASE2B1B_STORAGE_ROOT` must be the selected data-filesystem mountpoint itself. The runner
rejects root, home, relative, wildcard, newline, symlink, and non-mounted paths;
requires explicit `--confirm-cloud-storage` and strictly more than 5 GiB free;
and invokes only `/opt/conda/bin/python` from the existing cloud base environment.
Each successful canonical result is appended to the explicit
`trustsr/phase2b1b/logs/<stage>.jsonl` path below that mount. Check both `df -h`
and `df -ih`: a filesystem can have ample byte capacity while its inode quota is
exhausted. Extraction computes its inode requirement from unfinished work as
three inodes per missing pair plus 16 inodes of directory/commit headroom, so a
valid partial run remains restartable. If the persistent filesystem is
inode-constrained, use a separate mounted scratch filesystem after staging the
exact digest-qualified TACO object, frozen base manifest, and pre-extraction
sidecar beneath the same layout. The TIFF tree is reconstructible, but copy and
verify the canonical audit before stopping an ephemeral instance. The three
stages are intentionally restartable. Never copy the TACO object, 360-row
sidecar, or GeoTIFF tree into Git.

## Phase 2B2-A CPU crosssensor input audit

Phase 2B2-A reads 12 deterministic pairs from the frozen Phase 2B1B pixel tree,
checks every selected GeoTIFF again, and proves that two independent loads produce
the same aligned reflectance tensors. It does not run a super-resolution model,
download weights, or use GPU computation. Run from the exact reviewed checkout:

```bash
: "${PHASE2B2A_STORAGE_ROOT:?set this to the persistent filesystem mountpoint}"
PHASE2B2A_POST_MANIFEST_SHA256=c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a
PHASE2B2A_POST_MANIFEST="${PHASE2B2A_STORAGE_ROOT%/}/trustsr/phase2b1b/selections/${PHASE2B2A_POST_MANIFEST_SHA256}/samples.jsonl"

scripts/phase2b2a/run_cloud.sh \
  "$PHASE2B2A_STORAGE_ROOT" "$PWD" \
  --selection-manifest "$PHASE2B2A_POST_MANIFEST" \
  --selection-manifest-sha256 "$PHASE2B2A_POST_MANIFEST_SHA256" \
  --confirm-cloud-storage
```

The storage root must be the mounted data filesystem itself. The runner rejects
root, home, relative, wildcard, newline, symlink, non-mounted, and storage-root
override inputs; it requires more than 1 GiB free and more than 1024 free inodes.
It invokes only `/opt/conda/bin/python` from the existing base environment and
appends canonical stdout to `trustsr/phase2b2a/logs/audit-inputs.jsonl`.

The immutable audit is written to
`trustsr/phase2b2a/input-audits/<post-manifest-sha256>/phase2b2a-input-audit.json`.
After its bytes and SHA-256 have been copied and verified as the Git-safe audit,
and no audit process remains, the cloud instance can be paused. Never copy the
360-row sidecar, GeoTIFF files, normalized tensors, or logs into Git.

## Phase 2B2-B development three-model smoke

Status: Checkpoint A is complete. Real cloud acceptance ran bicubic, SEN2SRLite,
and LDSR-S2 on exactly four frozen `development` samples (correlation bins 0--3),
persisted 12 identity-bound predictions, and rebuilt the deterministic result by
cache-only replay. The replay was byte-identical and no compute process remained
after acceptance. It never computed calibration or `internal_test` metrics and is
an engineering smoke checkpoint, not a paper-result checkpoint.

The committed host-free evidence is:

- `artifacts/phase2b2b/sen2naipv2-development-three-model-smoke-v1.json`
  (`sha256:864f312a1c409f718de0b2fbcd827f6fdabbbb2828a4fd0aed7069989ba1ffcb`)
- `artifacts/phase2b2b/sen2naipv2-development-cache-audit-v1.json`
  (`sha256:9e4d51bb80386576c985316dbab62d98b232771d81be7570809711e652ce1d6e`)

The reproducible staged procedure remains in
[`docs/phase2b2b-cloud-runbook.md`](docs/phase2b2b-cloud-runbook.md).

## Phase 2B3-A development score audit

Phase 2B3-A adds six fail-closed stages for a four-ROI A1 stability/resource gate and an exact
120-ROI A2 development score audit. Every invocation takes an explicit cloud base Python, mounted
storage root, clean attached reviewed checkout, stage, and pinned reviewed commit. Compute stages
require the two verified model directories; replay stages reject them and remain model-free.

The safe pull workflow retrieves only a digest-addressed completed bundle manifest and its four
allowlisted JSON files, verifies remote and local sizes/SHA-256 values, and publishes the bundle only
after complete verification. Cloud pixels, tensors, caches, and checkpoint tar files are never
downloaded locally or committed; only allowlisted JSON evidence enters Git. Models, logs, remote
commit markers, lock files, instance endpoints, and credentials also stay out of Git. The disposable
work/durable checkpoint workflow, A0/A1/A2 checkpoints, exact command arrays, failure branches,
local verification, and pause conditions are in
[`docs/phase2b3a-cloud-runbook.md`](docs/phase2b3a-cloud-runbook.md). Development and review happen
locally before renting GPU time; the current unprivileged cloud container restores models through
verified, non-writable disposable copies rather than privileged bind mounts.
