"""One correctness call in a disposable CPU process; only scalar JSON leaves it."""

import contextlib
import io
import json
import sys
import warnings
from importlib.metadata import version

import numpy as np
import torch

from trustsr.evaluation.external_opensr import (
    CheckedAligner,
    RegistrationFailure,
    _failed,
    summarize_correctness_maps,
)


def main():
    if version('opensr-test') != '1.3.3':
        raise ValueError('unreviewed OpenSR version')
    import opensr_test

    torch.set_num_threads(1)
    raw = sys.stdin.buffer.read(10_000_001)
    if len(raw) > 10_000_000:
        raise ValueError('oversized input')
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        lr, sr, hr = [torch.from_numpy(data[key]) for key in ('lr', 'sr', 'hr')]
    evaluator = opensr_test.Metrics(
        device='cpu', agg_method='pixel', patch_size=None, border_mask=16,
        rgb_bands=[0, 1, 2], harm_apply_spectral=True, harm_apply_spatial=True,
        spatial_method='pcc', spatial_threshold_distance=5, spatial_max_num_keypoints=500,
        correctness_distance='nd', correctness_norm='softmin',
        correctness_temperature=0.25, im_score=0.05, om_score=0.05, ha_score=0.05,
    )
    evaluator.spatial_aligner = CheckedAligner(evaluator.spatial_aligner)
    try:
        # Upstream progress/registration prints are not research output.
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True):
            evaluator.correctness(lr, sr, hr, gradient_threshold='auto')
        maps = torch.stack([evaluator.improvement, evaluator.omission, evaluator.hallucination])
        result = summarize_correctness_maps(maps)
    except RegistrationFailure:
        result = _failed('registration_failed', (hr.shape[1] - 32) * (hr.shape[2] - 32))
    except (ValueError, RuntimeError, FloatingPointError):
        result = _failed('metric_failed', (hr.shape[1] - 32) * (hr.shape[2] - 32))
    sys.stdout.write(json.dumps(result, allow_nan=False, sort_keys=True, separators=(',', ':')))


if __name__ == '__main__':
    main()
