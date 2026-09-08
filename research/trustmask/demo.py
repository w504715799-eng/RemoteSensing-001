"""Run only a tiny synthetic illustration; no calibrated thresholds or research results."""

import json

import numpy as np
import torch

from research.trustmask.benefits import compare_workload, evaluate_mask
from research.trustmask.scores import fuse_scores, spatial_components


def main():
    samples = (torch.arange(5, dtype=torch.float64) / 10)[:, None, None, None]
    samples = samples.expand(5, 4, 8, 8)
    seed_term, texture_term = spatial_components(samples)
    lr = torch.full((8, 8), .01, dtype=torch.float64)
    lr[:, 4:] = .3
    score = fuse_scores(lr, seed_term + .5 * texture_term, lr_scale=.1,
                        uncertainty_scale=.02, lr_weight=.5)
    mask = score.numpy() <= .5  # Illustrative, not fitted/calibrated.
    risk = np.full((8, 8), .02)
    risk[:, 4:] = .12
    candidate = evaluate_mask(risk, mask, review_tile_size=2, pixel_area_m2=6.25)
    baseline = evaluate_mask(risk, np.zeros((8, 8), dtype=bool),
                             review_tile_size=2, pixel_area_m2=6.25)
    result = dict(
        evidence_type='synthetic_not_research_evidence',
        threshold_status='illustrative_uncalibrated',
        baseline_description='review_everything', baseline=baseline, candidate=candidate,
        time_inputs='assumed_not_measured',
        workload=compare_workload(baseline, candidate, baseline_compute_seconds=0,
                                  candidate_compute_seconds=.5, seconds_per_review_tile=2),
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
