"""Offline mask/workload measurements, separate from risk certification or human timing."""

import hashlib
import json
import math
from numbers import Real

import numpy as np


def _number(value, name, *, minimum=0, maximum=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or value < minimum
            or (maximum is not None and value > maximum)):
        raise ValueError(f'{name} must be a finite real in the required range')
    return float(value)


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode()).hexdigest()


def evaluate_mask(risk, mask, *, high_error_threshold=0.05, review_tile_size=32,
                  pixel_area_m2=None):
    """Measure one full-grid R9 map and bool retention mask; no inference or calibration.

    A reviewed tile contains at least one rejected pixel. Its count is a workload
    proxy, not measured person-time. Caller supplies a fixed physical support.
    """
    if (not isinstance(risk, np.ndarray) or risk.ndim != 2 or not risk.size
            or risk.dtype.kind != 'f' or not np.isfinite(risk).all()
            or (risk < 0).any() or (risk > 1).any()):
        raise ValueError('risk must be a nonempty finite floating 2D array in [0,1]')
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != risk.shape:
        raise ValueError('mask must be bool on the exact risk grid')
    tau = _number(high_error_threshold, 'high_error_threshold', maximum=1)
    if type(review_tile_size) is not int or review_tile_size < 1:
        raise ValueError('review_tile_size must be a positive integer')
    area = None if pixel_area_m2 is None else _number(pixel_area_m2, 'pixel_area_m2')
    if area == 0:
        raise ValueError('pixel_area_m2 must be positive')
    values = risk.astype(np.float64, copy=False)
    kept = values[mask]
    total, accepted = int(mask.size), int(mask.sum())
    high = values > tau
    high_count, accepted_high = int(high.sum()), int((high & mask).sum())
    tiles = [bool((~mask[y:y + review_tile_size, x:x + review_tile_size]).any())
             for y in range(0, mask.shape[0], review_tile_size)
             for x in range(0, mask.shape[1], review_tile_size)]
    identity = {
        'shape': list(mask.shape), 'pixel_area_m2': area, 'review_tile_size': review_tile_size,
        'high_error_threshold': tau,
        'risk_sha256': hashlib.sha256(values.astype('<f8').tobytes()).hexdigest(),
    }
    result = dict(
        schema='trustmask.workload.v1', support=identity,
        total_pixels=total, accepted_pixels=accepted, review_pixels=total - accepted,
        coverage=accepted / total, all_rejected=accepted == 0,
        retained_mean_risk=float(kept.mean()) if accepted else None,
        roi_loss=float(kept.max()) if accepted else 0.,
        high_error_pixels=high_count, accepted_high_error_pixels=accepted_high,
        accepted_high_error_fraction=accepted_high / accepted if accepted else None,
        high_error_capture_fraction=(
            (high_count - accepted_high) / high_count if high_count else None),
        total_tiles=len(tiles), review_tiles=sum(tiles),
        review_tile_fraction=sum(tiles) / len(tiles),
        accepted_area_m2=accepted * area if area is not None else None,
        review_area_m2=(total - accepted) * area if area is not None else None,
    )
    # Detect accidental edits during local handoff; this is not provenance authentication.
    result['summary_sha256'] = _digest(result)
    return result


def compare_workload(baseline, candidate, *, baseline_compute_seconds,
                     candidate_compute_seconds, seconds_per_review_tile=None):
    """Compare the same ROI/support; does not declare either method risk-qualified.

    Summary checksums detect accidental mutation, not adversarial provenance. The
    experiment runner must separately authenticate ROI identity and fixed membership.
    """
    for summary in (baseline, candidate):
        try:
            payload = {k: v for k, v in summary.items() if k != 'summary_sha256'}
            valid = (summary['schema'] == 'trustmask.workload.v1'
                     and summary['summary_sha256'] == _digest(payload))
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ValueError('invalid workload summary') from error
        if not valid:
            raise ValueError('workload summary changed or has unsupported schema')
    if baseline['support'] != candidate['support']:
        raise ValueError('comparison requires identical reference grid and workload definition')
    base_time = _number(baseline_compute_seconds, 'baseline_compute_seconds')
    new_time = _number(candidate_compute_seconds, 'candidate_compute_seconds')
    review_time = (None if seconds_per_review_tile is None
                   else _number(seconds_per_review_tile, 'seconds_per_review_tile'))
    saved = baseline['review_tiles'] - candidate['review_tiles']
    extra = new_time - base_time
    delta = candidate['coverage'] - baseline['coverage']
    modeled = saved * review_time - extra if review_time is not None else None
    if modeled is not None and not math.isfinite(modeled):
        raise ValueError('modeled time overflows finite seconds')
    return dict(
        coverage_delta_pp=100 * delta,
        relative_coverage_gain=delta / baseline['coverage'] if baseline['coverage'] else None,
        review_tiles_saved=saved, extra_compute_seconds=extra,
        break_even_seconds_per_review_tile=max(0., extra / saved) if saved > 0 else None,
        assumed_seconds_per_review_tile=review_time,
        modeled_net_seconds_saved=modeled,
        human_time_measured=False, risk_qualification='not_evaluated',
    )
