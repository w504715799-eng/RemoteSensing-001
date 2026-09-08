"""Isolated CPU-only image-level correctness, with explicit failure/support semantics."""

import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


class RegistrationFailure(ValueError):
    """Actual SR-to-HR harmonization reported a failed registration."""


class CheckedAligner:
    def __init__(self, aligner):
        self.aligner = aligner

    def get_metric(self, x, y):
        aligned, distance = self.aligner.get_metric(x, y)
        if not math.isfinite(float(distance)) or not torch.isfinite(aligned).all():
            raise RegistrationFailure('registration_failed')
        return aligned, distance


def _failed(reason, grid_pixels=None, valid_pixels=None):
    return {'status': 'failed', 'reason': reason, 'shares': None,
            'grid_pixels': grid_pixels, 'valid_pixels': valid_pixels}


def summarize_correctness_maps(maps: torch.Tensor) -> dict:
    """Preserve joint softmin support; do not turn missing correctness into zero."""
    if maps.ndim != 3 or maps.shape[0] != 3 or maps.numel() == 0:
        raise ValueError('expected three nonempty image-level softmin maps')
    grid = maps.shape[1] * maps.shape[2]
    if torch.isinf(maps).any():
        return _failed('invalid_softmin', grid)
    finite = torch.isfinite(maps)
    if not torch.equal(finite[0], finite[1]) or not torch.equal(finite[0], finite[2]):
        return _failed('inconsistent_support', grid)
    valid = int(finite[0].sum())
    if valid == 0:
        return _failed('no_valid_pixels', grid, 0)
    values = maps[:, finite[0]].to(torch.float64)
    if ((values < 0).any() or (values > 1).any()
            or not torch.allclose(values.sum(0), torch.ones(valid, dtype=torch.float64),
                                  atol=1e-5, rtol=0)):
        return _failed('invalid_softmin', grid, valid)
    return {'status': 'valid', 'reason': None, 'grid_pixels': grid, 'valid_pixels': valid,
            'shares': dict(zip(('im', 'om', 'ha'), values.mean(1).tolist(), strict=True))}


def compute_correctness(lr: np.ndarray, sr: np.ndarray, hr: np.ndarray) -> dict:
    """Evaluate validated arrays in a fresh process; no parent RNG/backend mutation."""
    if (any(type(a) is not np.ndarray or a.dtype != np.float32 or a.ndim != 3
            or a.shape[0] != 4 or not np.isfinite(a).all()
            or (a < 0).any() or (a > 1).any() for a in (lr, sr, hr))
            or sr.shape != hr.shape or hr.shape[1:] != tuple(d * 4 for d in lr.shape[1:])
            or min(hr.shape[1:]) <= 32 or max(hr.shape[1:]) > 512):
        raise ValueError('invalid full-grid correctness input')
    stream = io.BytesIO()
    np.savez(stream, lr=lr, sr=sr, hr=hr)
    # Pin source location, including non-installed cloud checkouts, without passing secrets.
    root = str(Path(__file__).resolve().parents[2])
    environment = {'PYTHONPATH': root, 'CUDA_VISIBLE_DEVICES': '',
                   'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                   'MKL_NUM_THREADS': '1', 'PYTHONHASHSEED': '0'}
    # Windows is unsupported; subprocess inherits no operational credentials or HOME.
    if 'PATH' in os.environ:
        environment['PATH'] = os.environ['PATH']
    try:
        process = subprocess.run(
            [sys.executable, '-m', 'trustsr.evaluation._external_opensr_worker'],
            input=stream.getvalue(), capture_output=True, timeout=120, check=False,
            env=environment, cwd=root,
        )
    except subprocess.TimeoutExpired:
        return _failed('timeout')
    if process.returncode != 0:
        return _failed('worker_failed')
    try:
        result = json.loads(process.stdout)
    except (ValueError, UnicodeError):
        return _failed('invalid_worker_output')
    return result
