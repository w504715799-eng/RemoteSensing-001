import pytest
import torch

from trustsr.risk.local import ensemble_variance_score, local_l1_risk


def test_local_l1_risk_window_one_is_band_mean_absolute_error() -> None:
    sr = torch.tensor(
        [
            [[0.0, 0.2], [0.4, 0.6]],
            [[0.1, 0.3], [0.5, 0.7]],
            [[0.2, 0.4], [0.6, 0.8]],
            [[0.3, 0.5], [0.7, 0.9]],
        ]
    )
    hr = torch.zeros_like(sr)

    result = local_l1_risk(sr, hr, window=1)

    expected = torch.tensor([[0.15, 0.35], [0.55, 0.75]], dtype=torch.float64)
    torch.testing.assert_close(result, expected)


def test_ensemble_variance_score_uses_population_variance_and_band_mean() -> None:
    first = torch.zeros((4, 2, 2))
    second = torch.ones((4, 2, 2))

    result = ensemble_variance_score(torch.stack((first, second)))

    torch.testing.assert_close(result, torch.full((2, 2), 0.25, dtype=torch.float64))


def test_local_l1_risk_uses_reflection_padding() -> None:
    sr = torch.zeros((4, 3, 3))
    sr[:, 0, 0] = 1.0
    result = local_l1_risk(sr, torch.zeros_like(sr), window=3)
    expected = torch.tensor(
        [[1 / 9, 1 / 9, 0.0], [1 / 9, 1 / 9, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    "sr, hr, window",
    [
        (torch.zeros((3, 2, 2)), torch.zeros((3, 2, 2)), 1),
        (torch.zeros((4, 2, 2, 1)), torch.zeros((4, 2, 2, 1)), 1),
        (torch.zeros((4, 2, 2)), torch.zeros((4, 2, 3)), 1),
        (torch.full((4, 2, 2), float("nan")), torch.zeros((4, 2, 2)), 1),
        (torch.zeros((4, 2, 2)), torch.full((4, 2, 2), 2.0), 1),
        (torch.zeros((4, 2, 2)), torch.zeros((4, 2, 2)), 0),
        (torch.zeros((4, 2, 2)), torch.zeros((4, 2, 2)), 2),
        (torch.zeros((4, 2, 2)), torch.zeros((4, 2, 2)), 3),
    ],
)
def test_local_l1_risk_rejects_invalid_inputs(sr, hr, window) -> None:
    with pytest.raises(ValueError):
        local_l1_risk(sr, hr, window=window)


@pytest.mark.parametrize(
    "samples",
    [
        torch.zeros((1, 4, 2, 2)),
        torch.zeros((2, 3, 2, 2)),
        torch.zeros((2, 4, 2)),
        torch.full((2, 4, 2, 2), float("inf")),
        torch.full((2, 4, 2, 2), -0.1),
    ],
)
def test_ensemble_variance_score_rejects_invalid_inputs(samples) -> None:
    with pytest.raises(ValueError):
        ensemble_variance_score(samples)


def test_local_l1_risk_rejects_complex_tensors_explicitly() -> None:
    sr = torch.zeros((4, 2, 2), dtype=torch.complex64)

    with pytest.raises(ValueError, match="sr must be real-valued"):
        local_l1_risk(sr, sr, window=1)


def test_ensemble_variance_score_rejects_complex_tensors_explicitly() -> None:
    samples = torch.zeros((2, 4, 2, 2), dtype=torch.complex64)

    with pytest.raises(ValueError, match="samples must be real-valued"):
        ensemble_variance_score(samples)
