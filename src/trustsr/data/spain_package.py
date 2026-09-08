"""Hash-gated local Spain decoding. No download, model access, or study authorization."""

import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from trustsr.data.spain_inputs import bind_member_rows, prepare_spain_pair


def decode_package(
    path: Path, *, expected_sha256: str, expected_size: int, members: list[dict],
):
    """Authenticate exact package bytes and return pairs in pinned member order.

    Identity arguments must come from the future frozen runner. This primitive is
    not an external access permit. The restricted worker is defense in depth for
    a pinned trusted source, not a general-purpose hostile-pickle OS sandbox.
    """
    if (not isinstance(expected_sha256, str)
            or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None
            or type(expected_size) is not int or not 0 < expected_size <= 150_000_000
            or not isinstance(members, list) or not 0 < len(members) <= 28):
        raise ValueError('invalid package identity or member count')
    # No final symlink traversal; authenticate the bytes read from this descriptor.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
            raise ValueError('package size/type mismatch')
        raw = handle.read(expected_size + 1)
    if len(raw) != expected_size:
        raise ValueError('package size mismatch')
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('package SHA-256 mismatch')
    worker = Path(__file__).with_name('_spain_pickle_worker.py')
    try:
        process = subprocess.run(
            [sys.executable, '-I', str(worker), str(len(members))], input=raw,
            capture_output=True, timeout=40, check=False,
            env={'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                 'MKL_NUM_THREADS': '1', 'PYTHONHASHSEED': '0'},
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError('restricted package decoder timed out') from exc
    if process.returncode != 0:
        raise ValueError('restricted package decoding failed')
    with np.load(io.BytesIO(process.stdout), allow_pickle=False) as decoded:
        if sorted(decoded.files) != ['HRharm', 'L2A', 'metadata']:
            raise ValueError('invalid decoder output')
        frame = pd.DataFrame(json.loads(decoded['metadata'].tobytes()))
        positions = bind_member_rows(frame, members)
        lr, hr = decoded['L2A'], decoded['HRharm']
    pairs, normalization = [], []
    for member, position in zip(members, positions, strict=True):
        pair, counts = prepare_spain_pair(member['roi'], lr[position], hr[position])
        pair = replace(pair, source=f'opensr-test/100/{expected_sha256}')
        pairs.append(pair)
        normalization.append({'roi': member['roi'], **counts})
    return pairs, {
        'package_sha256': expected_sha256, 'package_size': expected_size,
        'decoder': 'restricted-numpy-pandas-v1', 'positions': list(positions),
        'normalization': normalization,
    }
