# First real-data cloud batches

2026-09-09. The RTX4090 server was made available at
`root@fj01-ssh.gpuhome.cc:30370`. A separate checkout of commit673810f is deployed
at `/root/rivermind-fs/trustmask-runner-673810f`; the new study is
`/root/rivermind-fs/trustmask-public-crosssensor-v1`. Do not recreate that study.
The exact accepted assignment is copied to the checkout's `assignments.json`.

The independent venv was created with `uv venv --python /opt/conda/bin/python
--system-site-packages .venv`, then PyArrow19.0.1 installed into this venv with
`uv pip install --python .venv/bin/python --no-deps pyarrow==19.0.1`. The base
GPU stack already provides opensr-model1.1.1 and Torch2.12.1+cu130. It was not
modified. Package versions are recorded in the study's immutable runtime binding.

Source file (read selected assets only):
`/root/rivermind-fs/trustsr/phase2b1a/source/c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5/sen2naipv2-crosssensor.taco`.
Verified model weights are reused read-only from
`/root/rivermind-data/model-mounts/ldsr-s2`; no historical prediction cache is reused.

Metadata-only preflight passed for7477members and read no pixel payloads.
The original timing command failed because `/usr/bin/time` is absent; that attempt
exited127 before Python/ROI access. The replacement cloud `timed-run.py` uses
`time.time`, `subprocess.run` and `resource.getrusage(RUSAGE_CHILDREN)`. It sets
OMP_NUM_THREADS=4 and MKL_NUM_THREADS=4 and executes the unchanged frozen CLI.
`first-roi-python.log` and `first-roi-measurement.json` retain its output.

The first run committed one scale_fit ROI with exit0 in23.7245seconds, peak child
RSS2108988KiB. This includes interpreter/model initialization, checkpoint verification,
selected image loading, five-seed inference, score extraction and receipt writing;
it is not isolated inference latency. No threshold, formalcal or test decision was
made from this measurement.

A follow-up invocation `batch16.py` uses the same command with `--max-rois16`.
It resumes the same bound study; `batch16.log` and `batch16-measurement.json` retain
its output. The independent `audit-live.py` checks all committed envelope digests,
contiguous indices, exact role/group/member identities, bound checkpoint and 64
samples for each of the five scale components. It exports metadata only, not maps.

## Exact continuation command

From the deployed checkout, keeping the same source, environment and study path:

```bash
export PYTHONPATH=src:.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
.venv/bin/python -m research.trustmask.batch_cli run \
  --protocol research/protocols/trustmask-real-batches-v1.json \
  --assignments assignments.json \
  --taco /root/rivermind-fs/trustsr/phase2b1a/source/c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5/sen2naipv2-crosssensor.taco \
  --model-dir /root/rivermind-data/model-mounts/ldsr-s2 \
  --study-dir /root/rivermind-fs/trustmask-public-crosssensor-v1 \
  --max-rois 16
```

Use a distinct operational log filename for each subsequent invocation. Do not
rerun `batch16.py` merely to inspect its output: that would process another16ROIs
and overwrite its measurement file. Read the existing files instead.

The new protocol remains unchanged. Old A/B/C image/cache access and old terminal
ledger reopening did not occur. Scale-fit success does not establish calibrated
risk or coverage benefit; those require completing all later frozen stages.

## Completed checkpoint

Both invocations exited0;17ROIs are committed, all in scale_fit (17/318).
The16-ROI continuation took159.4228seconds (9.9639seconds/ROI including startup),
peak child RSS2113836KiB (~2.02GiB). GPU memory sampled during inference was5685MiB;
that is an observation, not a measured peak. All17receipt digests and identities
passed the independent metadata-only audit; the first receipt digest stayed unchanged.
No frozen-method artifact or test-start ledger exists yet. GPU returned to idle.
Published metadata evidence:
`research/evidence/trustmask-first-cloud-batches-v1.json`.

There are301scale-fitROIs remaining. At this small-batch observed rate this is
approximately50minutes, excluding variation and later-stage curve computation.
This is a planning estimate, not a promised total study duration. No unattended
job remains running at this checkpoint. New study artifacts and code are persistent;
resume from the same directory after checking environment/source fingerprint.
