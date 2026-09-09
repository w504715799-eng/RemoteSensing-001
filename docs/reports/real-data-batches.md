# Real-data batch execution

2026-09-09. Local implementation and synthetic validation complete. No real pixels,
old A/B/C prediction caches, SSH, model downloads or GPU inference were used.

The new runner connects selected TACO members to verified five-seed LDSR inference,
reference-free features, development selection, formal group-risk calibration and
terminal evaluation. One ROI's imagery/predictions is resident at a time; only compact
per-ROI statistics and provenance receipts persist. Compact statistics accumulate
with sample count; this is bounded image memory, not constant total process memory.

## Frozen definition

`research/protocols/trustmask-real-batches-v1.json` binds the exact accepted assignment,
implementation file hashes, source metadata hashes, model checkpoint/config, seeds
3407–3411, center3407, sampling100/eta0.95/temperature1/histogram matching, RGBN,
x4, fixed raw130/520 → center128/512 crop, and full512² R9 risk support. uint16
nodata65535 and entirely valid raw raster masks are required. Values above10000
(up to32767) are saturated before normalization, with per-band clipping counts.
All scoring and risk calculations use the entire fixed512² support.

Protocol SHA256: `a65df39f8eb937e046e48c4dc24f740bf53cf1be1b8fe9bc53ec70f93ba411bd`.
Assignments SHA256: `460eb73bea4cab104cbcd92f5ac8d6538da28dc62fabb0f3767365a90454354b`.
The ignored assignment file remains
`artifacts/progressive-partition-preview/assignments.json`; transfer it with the
checkout. If absent, reproduce it metadata-only using
`python -m research.trustmask.partition_preview --output NEW_DIRECTORY`, then pass
that directory's `assignments.json` explicitly. The CLI verifies its exact digest.
The original preview remains unchanged and explicitly labeled a preview.

| Stage | Groups | ROIs |
| --- | ---: | ---: |
| scale_fit | 256 | 318 |
| development_calibration | 512 | 614 |
| development_validation | 845 | 1041 |
| calibration | 1500 | 1814 |
| test | 3045 | 3690 |

Public SEN2NAIPv2 release provenance is accepted; per-member S2 source mapping is
not required. Runtime verifies fixed top metadata, selected nested ranges and
raster geometry; receipts hash the actual selected payloads. The published full
9.7GB object SHA is declared provenance, **not** claimed runtime full-object
verification. No excluded image bytes are opened merely to hash the container.

## Cloud runbook

Use a new checkout/environment and persistent study directory; do not run legacy
A/B/C commands or reuse their caches/ledgers. The known SSH endpoint is
`ssh root@fj01-ssh.gpuhome.cc -p 30370`; availability and the actual local TACO path
still need checking after the GPU is powered on. Keep the source file unchanged
throughout the study. There is no automatic download of the dataset.

From the new checkout, install the existing locked GPU environment with
`uv sync --frozen --extra gpu`. PyArrow19.0.1 is additionally required; it was tested
locally as an isolated dependency. For example:

```bash
uv pip install --python .venv/bin/python --target /root/rivermind-fs/trustmask-deps pyarrow==19.0.1
export PYTHONPATH=/root/rivermind-fs/trustmask-deps:src:.
```

Set these paths to the actual source file and new persistent locations. The source
path below is a placeholder, not an assertion about the server filesystem.

```bash
export TRUSTMASK_TACO=/absolute/path/to/sen2naipv2-crosssensor.taco
export TRUSTMASK_MODELS=/root/rivermind-fs/trustmask-models
export TRUSTMASK_STUDY=/root/rivermind-fs/trustmask-public-crosssensor-v1
export TRUSTMASK_PROTOCOL=research/protocols/trustmask-real-batches-v1.json
export TRUSTMASK_ASSIGNMENTS=artifacts/progressive-partition-preview/assignments.json

.venv/bin/python -m research.trustmask.batch_cli preflight \
  --protocol "$TRUSTMASK_PROTOCOL" --assignments "$TRUSTMASK_ASSIGNMENTS" \
  --taco "$TRUSTMASK_TACO" --model-dir "$TRUSTMASK_MODELS"

/usr/bin/time -v .venv/bin/python -m research.trustmask.batch_cli run \
  --protocol "$TRUSTMASK_PROTOCOL" --assignments "$TRUSTMASK_ASSIGNMENTS" \
  --taco "$TRUSTMASK_TACO" --model-dir "$TRUSTMASK_MODELS" \
  --study-dir "$TRUSTMASK_STUDY" --max-rois 1
```

Preflight checks only the three pinned top metadata intervals and assignment
membership, not model availability or selected images. The first one-ROI run is
part of scale fitting, not an extra test inspection. It verifies actual model
execution and gives initial runtime/resource measurements. The verified model
loader may download its pinned checkpoint into the new model directory.

After assessing that first run, repeat the same `run` command with `--max-rois 16`
(or another positive budget). Each invocation processes at most that many **new**
ROIs across ordered stages. Stop repeating when stdout reports `status=complete`.
There is no uncontrolled full-corpus run by default. No manual method/threshold
selection is needed at stage boundaries.

## Resume and outputs

`journal.json` binds protocol, assignments, runtime package versions, device and
source file stat fingerprint. Incompatible software/source/binding changes fail
before inference. An exclusive file lock prevents concurrent writers. Atomic
`receipts/NNNNNNNN.json` contain statistics, member/group/stage, payload digests,
nested metadata digest, clipping counts and model provenance; no prediction arrays.

All development selections precede formal calibration. `frozen_methods.json` is
written before any test ROI; `terminal_test_started.json` is durable before first
test provider access. Same-binding crash continuation can retry an uncommitted ROI,
while committed receipts never trigger inference again. Missing pretest receipts
after freezing, gaps, changed digests and incomplete final reports fail closed.
Do not delete a consumed study directory or regenerate the protocol to rerun test.
The ledger prevents operational accidents, not deliberate filesystem-owner bypass.
A crash before the very first journal rename can leave an initialization-only
`journal.tmp`; no ROI was accessed, and a new empty directory can initialize safely.

`result.json` uses a `{sha256,payload}` envelope. The payload contains frozen methods,
all development candidates, per-group terminal observations, risk/coverage bounds,
workload proxies and real-input evidence/binding identifiers. A completed invocation
checks receipts and returns the same report without further inference. It still
requires source metadata/runtime preflight through the CLI.

## Validation and limits

83 scoped tests passed: batch, CLI, adapter, reference pipeline, calibration, scores
and benefits. They include exact reference-result parity, varied batch boundaries,
crash retry, corrupt/missing receipts, locking, pre-test freezing, source-header
redirection rejection, nodata/internal masks, geometry mismatch and lazy seed reuse.
Ruff passed; historical frozen-protocol gate passed. Independent code and method
reviews found two adapter and one terminal-resume defect; all were corrected with
regression coverage. CLI prepare/verify checked the actual metadata-only assignment.

Real TACO extraction and CUDA execution remain untested until the cloud run.
Expected group CRC and independent-test bounds retain their exchangeability and
independence assumptions; geographic grouping and public availability do not prove
those assumptions. No coverage gain or real-data risk qualification is claimed yet.
