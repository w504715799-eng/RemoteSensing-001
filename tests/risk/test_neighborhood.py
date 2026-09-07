import math

import pytest
import torch

from trustsr.risk import neighborhood


@pytest.mark.parametrize("window", [3, 9])
def test_spatial_constants_preserve_between_seed_variance(window):
    samples = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]).reshape(5, 1, 1, 1)
    samples = samples.expand(5, 4, 9, 10).clone()
    before = samples.clone()
    actual = neighborhood.neighborhood_variance_score(samples, window=window)
    torch.testing.assert_close(
        actual, torch.full((9, 10), 0.24, dtype=torch.float64), atol=1e-12, rtol=0
    )
    assert actual.device.type == "cpu"
    assert torch.equal(samples, before)


def test_constant_ensemble_has_zero_score():
    actual = neighborhood.neighborhood_variance_score(torch.full((5, 4, 4, 4), 0.5), window=3)
    assert torch.count_nonzero(actual) == 0


def test_spatial_variation_and_reflected_edges_match_direct_population_oracle():
    # A single varying band catches accidental summation instead of band averaging.
    samples = torch.zeros((5, 4, 4, 5), dtype=torch.float64)
    samples[:, 0, :, 0] = 1
    before = samples.clone()

    def reflect(i, n):
        while i < 0 or i >= n:
            i = -i if i < 0 else 2 * n - 2 - i
        return i

    variance = torch.zeros((4, 5), dtype=torch.float64)
    for y in range(4):
        for x in range(5):
            # Direct population variance, not the implementation's moment formula.
            values = [
                float(samples[k, 0, reflect(y + dy, 4), reflect(x + dx, 5)])
                for k in range(5)
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            ]
            mean = sum(values) / len(values)
            variance[y, x] = sum((v - mean) ** 2 for v in values) / len(values) / 4
    weights = [math.exp(-i * i / 2) for i in range(-3, 4)]
    expected = torch.zeros((4, 5), dtype=torch.float64)
    for y in range(4):
        for x in range(5):
            expected[y, x] = (
                sum(
                    weights[dy + 3]
                    * weights[dx + 3]
                    * float(variance[reflect(y + dy, 4), reflect(x + dx, 5)])
                    for dy in range(-3, 4)
                    for dx in range(-3, 4)
                )
                / sum(weights) ** 2
            )
    actual = neighborhood.neighborhood_variance_score(samples, window=3)
    assert actual.min() > 0
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    assert torch.equal(samples, before)


@pytest.mark.parametrize("shape", [(4, 4, 9, 9), (5, 3, 9, 9), (5, 4, 3, 9), (5, 4, 9)])
def test_invalid_ensemble_shape_is_rejected(shape):
    with pytest.raises(ValueError):
        neighborhood.neighborhood_variance_score(torch.zeros(shape), window=3)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_reflectance_is_rejected(value):
    with pytest.raises(ValueError):
        neighborhood.neighborhood_variance_score(torch.full((5, 4, 9, 9), value), window=3)


@pytest.mark.parametrize("window", [True, 1, 5, 3.0, 0])
def test_unregistered_window_is_rejected(window):
    with pytest.raises(ValueError):
        neighborhood.neighborhood_variance_score(torch.zeros((5, 4, 9, 9)), window=window)


@pytest.mark.parametrize("dtype", [torch.int64, torch.complex64])
def test_nonfloating_input_is_rejected(dtype):
    with pytest.raises(ValueError):
        neighborhood.neighborhood_variance_score(torch.zeros((5, 4, 9, 9), dtype=dtype), window=3)
