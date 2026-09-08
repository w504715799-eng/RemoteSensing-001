"""Hand-derived workload fixtures catch wrong denominators and optimistic time claims."""

import numpy as np
import pytest

from research.trustmask.benefits import compare_workload, evaluate_mask


def example(mask):
    return evaluate_mask(
        np.array([[.01, .2], [.03, .4]]), np.array(mask, dtype=bool),
        review_tile_size=1, pixel_area_m2=6.25,
    )


def test_counts_risk_and_area_use_the_same_retained_support():
    r = example([[True, False], [True, False]])
    assert r['coverage'] == .5
    assert r['retained_mean_risk'] == pytest.approx(.02)
    assert r['roi_loss'] == .03
    assert r['review_pixels'] == 2
    assert r['accepted_area_m2'] == 12.5
    assert r['review_tiles'] == 2
    assert r['high_error_capture_fraction'] == 1
    assert r['accepted_high_error_fraction'] == 0


def test_all_reject_is_zero_loss_but_no_accepted_mean():
    r = example([[False, False], [False, False]])
    assert r['all_rejected'] is True
    assert r['roi_loss'] == 0
    assert r['retained_mean_risk'] is None
    assert r['accepted_high_error_fraction'] is None
    assert r['review_tiles'] == 4


def test_all_accept_reports_errors_instead_of_claiming_trust():
    r = example([[True, True], [True, True]])
    assert r['review_tiles'] == 0
    assert r['roi_loss'] == .4
    assert r['accepted_high_error_pixels'] == 2
    assert r['accepted_high_error_fraction'] == .5
    assert r['high_error_capture_fraction'] == 0


def test_no_high_error_pixels_has_undefined_capture_and_no_inferred_area():
    r = evaluate_mask(np.zeros((2, 2)), np.ones((2, 2), dtype=bool))
    assert r['high_error_capture_fraction'] is None
    assert r['accepted_area_m2'] is None


def test_fragmentation_and_partial_tiles_count_actual_review_units():
    mask = np.ones((3, 5), dtype=bool)
    mask[0, 0] = mask[2, 4] = False
    risk = np.zeros((3, 5))
    r = evaluate_mask(risk, mask, review_tile_size=2)
    assert r['total_tiles'] == 6
    assert r['review_tiles'] == 2
    assert r['review_pixels'] == 2
    assert r['review_tile_fraction'] == pytest.approx(1 / 3)
    assert risk.sum() == 0 and mask.sum() == 13


@pytest.mark.parametrize('risk,mask,kwargs', [
    (np.zeros((2, 2)), np.ones((2, 2)), {}),
    (np.zeros((2, 2)), np.ones((1, 2), dtype=bool), {}),
    (np.array([[np.nan]]), np.ones((1, 1), dtype=bool), {}),
    (np.array([[1.01]]), np.ones((1, 1), dtype=bool), {}),
    (np.array([[-.01]]), np.ones((1, 1), dtype=bool), {}),
    (np.empty((0, 2)), np.empty((0, 2), dtype=bool), {}),
    (np.zeros((2, 2)), np.ones((2, 2), dtype=bool), {'review_tile_size': True}),
    (np.zeros((2, 2)), np.ones((2, 2), dtype=bool), {'pixel_area_m2': 0}),
    (np.zeros((2, 2)), np.ones((2, 2), dtype=bool), {'high_error_threshold': float('inf')}),
])
def test_invalid_support_is_not_silently_cast_or_dropped(risk, mask, kwargs):
    with pytest.raises(ValueError):
        evaluate_mask(risk, mask, **kwargs)


def test_workload_time_is_an_explicit_model_with_break_even():
    baseline = example([[False, False], [False, False]])
    candidate = example([[True, False], [True, False]])
    r = compare_workload(baseline, candidate, baseline_compute_seconds=1,
                         candidate_compute_seconds=5, seconds_per_review_tile=3)
    assert r['coverage_delta_pp'] == 50
    assert r['relative_coverage_gain'] is None
    assert r['review_tiles_saved'] == 2
    assert r['break_even_seconds_per_review_tile'] == 2
    assert r['modeled_net_seconds_saved'] == 2
    assert r['human_time_measured'] is False
    assert r['risk_qualification'] == 'not_evaluated'


def test_no_assumed_time_does_not_invent_seconds_saved():
    r = compare_workload(example([[True, True], [True, True]]),
                         example([[True, False], [True, False]]),
                         baseline_compute_seconds=1, candidate_compute_seconds=2)
    assert r['review_tiles_saved'] == -2
    assert r['break_even_seconds_per_review_tile'] is None
    assert r['modeled_net_seconds_saved'] is None
    assert r['coverage_delta_pp'] == -50


def test_comparison_rejects_mismatched_reference_even_if_counts_match():
    a = example([[True, False], [True, False]])
    b = evaluate_mask(np.array([[.02, .2], [.03, .4]]),
                      np.array([[True, False], [True, False]]),
                      review_tile_size=1, pixel_area_m2=6.25)
    with pytest.raises(ValueError):
        compare_workload(a, b, baseline_compute_seconds=1, candidate_compute_seconds=1)


def test_negative_or_nonfinite_time_and_tampered_summary_rejected():
    a = example([[True, False], [True, False]])
    for time in (-1, float('nan'), True):
        with pytest.raises(ValueError):
            compare_workload(a, a, baseline_compute_seconds=time, candidate_compute_seconds=1)
    b = dict(a, review_tiles=-1)
    with pytest.raises(ValueError):
        compare_workload(a, b, baseline_compute_seconds=1, candidate_compute_seconds=1)


def test_finite_time_inputs_cannot_export_infinite_modeled_benefit():
    baseline = example([[False, False], [False, False]])
    candidate = example([[True, False], [True, False]])
    with pytest.raises(ValueError):
        compare_workload(baseline, candidate, baseline_compute_seconds=0,
                         candidate_compute_seconds=0, seconds_per_review_tile=1e308)
