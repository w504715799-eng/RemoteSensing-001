"""Constant and spatial fixtures distinguish seed uncertainty from texture."""

import pytest
import torch

from research.trustmask.scores import fuse_scores, spatial_components
from trustsr.risk.neighborhood import neighborhood_variance_score


def test_spatial_constants_have_seed_variance_but_no_texture():
    samples = torch.zeros((5, 4, 8, 8), dtype=torch.float64)
    samples[3:] = 1
    original = samples.clone()
    u, t = spatial_components(samples)
    torch.testing.assert_close(u, torch.full((8, 8), .24, dtype=torch.float64))
    torch.testing.assert_close(t, torch.zeros((8, 8), dtype=torch.float64), atol=1e-12, rtol=0)
    assert torch.equal(samples, original)


def test_identical_spatially_varying_samples_are_texture_not_seed_uncertainty():
    pattern = (torch.arange(8, dtype=torch.float64) % 2).repeat(8, 1)
    samples = pattern.expand(5, 4, 8, 8).clone()
    u, t = spatial_components(samples)
    assert u.abs().max() == 0
    assert t.min() > .1


def test_decomposition_sum_matches_frozen_joint_neighborhood_operator():
    generator = torch.Generator().manual_seed(47)
    samples = torch.rand((5, 4, 9, 11), generator=generator, dtype=torch.float64)
    u, t = spatial_components(samples)
    expected = neighborhood_variance_score(samples, window=3)
    torch.testing.assert_close(u + t, expected, atol=1e-12, rtol=0)


def test_fixed_scales_fuse_complementary_maps_without_per_roi_ranking():
    lr = torch.tensor([[0., 1.]], dtype=torch.float64)
    variance = torch.tensor([[1., 0.]], dtype=torch.float64)
    output = fuse_scores(lr, variance, lr_scale=1, uncertainty_scale=1, lr_weight=.25)
    torch.testing.assert_close(output, torch.tensor([[.375, .125]], dtype=torch.float64))
    assert torch.equal(lr, torch.tensor([[0., 1.]], dtype=torch.float64))
    singleton = fuse_scores(lr[:, 1:], variance[:, 1:], lr_scale=1,
                            uncertainty_scale=1, lr_weight=.25)
    assert singleton.item() == .125


@pytest.mark.parametrize('weight,expected', [(0, .5), (1, 0)])
def test_fusion_endpoints_represent_component_ablations(weight, expected):
    result = fuse_scores(torch.zeros((1, 1)), torch.ones((1, 1)),
                         lr_scale=1, uncertainty_scale=1, lr_weight=weight)
    assert result.item() == expected


@pytest.mark.parametrize('kwargs', [
    {'lr_scale': 0}, {'uncertainty_scale': -1}, {'lr_weight': 1.01},
    {'lr_weight': float('nan')}, {'lr_scale': True}, {'uncertainty_scale': float('inf')},
])
def test_bad_frozen_parameters_cannot_produce_a_score(kwargs):
    params = dict(lr_scale=1, uncertainty_scale=1, lr_weight=.5)
    params.update(kwargs)
    with pytest.raises(ValueError):
        fuse_scores(torch.zeros((2, 2)), torch.ones((2, 2)), **params)


@pytest.mark.parametrize('samples', [
    torch.zeros((4, 4, 8, 8)), torch.zeros((5, 3, 8, 8)),
    torch.zeros((5, 4, 3, 3)), torch.ones((5, 4, 8, 8), dtype=torch.int32),
    torch.full((5, 4, 8, 8), float('nan')), torch.full((5, 4, 8, 8), -1.),
])
def test_invalid_samples_rejected(samples):
    with pytest.raises(ValueError):
        spatial_components(samples)


@pytest.mark.parametrize('lr,variance', [
    (torch.zeros((2, 2)), torch.zeros((2, 3))),
    (torch.zeros(2), torch.zeros(2)),
    (torch.full((2, 2), -1.), torch.zeros((2, 2))),
    (torch.zeros((2, 2)), torch.full((2, 2), float('nan'))),
])
def test_bad_score_maps_rejected(lr, variance):
    with pytest.raises(ValueError):
        fuse_scores(lr, variance, lr_scale=1, uncertainty_scale=1, lr_weight=.5)


def test_positive_subnormal_and_extreme_scales_keep_correct_ratios():
    tiny = torch.tensor([[1e-320]], dtype=torch.float64)
    result = fuse_scores(tiny, tiny, lr_scale=1e-320,
                         uncertainty_scale=1e-320, lr_weight=.5)
    assert result.item() == .5
    huge = torch.tensor([[1e308]], dtype=torch.float64)
    result = fuse_scores(huge, huge, lr_scale=1e308,
                         uncertainty_scale=1e308, lr_weight=.5)
    assert result.item() == .5
