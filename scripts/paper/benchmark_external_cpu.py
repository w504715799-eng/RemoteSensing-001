"""Synthetic full-grid CPU integration timing; no files, models, or network inputs."""

import hashlib
import json
import pickle
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trustsr.artifacts.predictions import PredictionCache, build_identity
from trustsr.data.spain_package import decode_package
from trustsr.evaluation.external_replay import replay_subset
from trustsr.jsonio import canonical_json
from trustsr.models.bicubic import BicubicX4


def main():
    torch.set_num_threads(1)
    generator = np.random.default_rng(20260908)
    lr = generator.integers(1000, 8000, (1, 12, 128, 128), dtype=np.int16)
    hr = lr[:, [3, 2, 1, 7]].repeat(4, axis=2).repeat(4, axis=3)
    members = [dict(roi='synthetic_0', lr_file='synthetic_lr', hr_file='synthetic_hr',
                    lr_gee_id='synthetic_scene', crs='EPSG:32630',
                    affine=[2.5, 0, 0, 0, -2.5, 0])]
    payload = pickle.dumps({'L2A': lr, 'HRharm': hr, 'metadata': pd.DataFrame(members)},
                           protocol=4)
    with tempfile.TemporaryDirectory(prefix='trustsr-synthetic-cpu-') as temporary:
        root = Path(temporary)
        path = root / 'synthetic.pkl'
        path.write_bytes(payload)
        start = time.perf_counter()
        pairs, _ = decode_package(path, expected_sha256=hashlib.sha256(payload).hexdigest(),
                                  expected_size=len(payload), members=members)
        decode_seconds = time.perf_counter() - start
        pair = pairs[0]
        provenance = {f'ldsr_{seed}': {'name': 'synthetic', 'seed': seed}
                      for seed in range(3407, 3412)}
        provenance.update(bicubic={'name': 'synthetic-bicubic'}, sen2sr={
            'name': 'synthetic-sen2sr', 'cpu_execution_policy': 'cloud-sen2srlite-96-v1',
            'cpu_intraop_threads': 96})
        cache = PredictionCache(root / 'cache')
        start = time.perf_counter()
        for i, (slot, model) in enumerate(provenance.items()):
            prediction = ((pair.hr + i * 0.001).clamp(0, 1) if slot != 'bicubic'
                          else BicubicX4().predict(pair.lr))
            cache.put(build_identity(model, pair.source, pair.sample_id, pair.lr), prediction)
        cache_write_seconds = time.perf_counter() - start
        digest = hashlib.sha256(canonical_json(provenance)).hexdigest()
        timings, results = [], []
        for _ in range(2):
            start = time.perf_counter()
            result = replay_subset(pairs, ['synthetic_0'], cache, provenance,
                                   expected_provenances_sha256=digest)
            timings.append(time.perf_counter() - start)
            results.append(canonical_json(result))
        if results[0] != results[1] or result['summary']['primary']['paired_valid'] != 1:
            raise ValueError('synthetic replay failed exact reproducibility or membership')
        if result['structure']['valid'] != 1:
            raise ValueError('synthetic structural diagnostic unexpectedly failed')
        print(json.dumps({
            'scope': 'synthetic-local-cpu-only; not Spain results or a cloud budget',
            'machine_architecture': platform.machine(), 'torch_version': str(torch.__version__),
            'numpy_version': np.__version__, 'threads': 1, 'rois': 1, 'hr_grid': [512, 512],
            'package_decode_seconds': decode_seconds,
            'seven_prediction_cache_write_seconds': cache_write_seconds,
            'cache_score_structure_replay_seconds': timings,
            'science_replay_byte_identical': True, 'structure_valid': True,
        }, sort_keys=True))


if __name__ == '__main__':
    main()
