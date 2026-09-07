import opensr_test
import torch
from opensr_test.correctness import get_correctness_stats


def test_metric_grid_is_cropped_not_original_pixel_coordinates():
    lr = torch.arange(4 * 16 * 16, dtype=torch.float32).reshape(4, 16, 16) / 1024
    hr = torch.zeros((4, 64, 64), dtype=torch.float32)
    evaluator = opensr_test.Metrics(device="cpu")
    evaluator.setup(lr, hr, hr)
    assert evaluator.hr.shape == (4, 32, 32)
    torch.testing.assert_close(evaluator.lr, lr[:, 4:-4, 4:-4])


def test_soft_class_share_is_not_hard_pixel_fraction():
    zeros = torch.zeros((2, 2))
    mask = torch.ones((2, 2))
    soft = get_correctness_stats(
        zeros, zeros, zeros, mask, correctness_norm="softmin", temperature=0.25
    )
    hard = get_correctness_stats(
        zeros, zeros, zeros, mask, correctness_norm="percent", temperature=0.25
    )
    torch.testing.assert_close(torch.stack(soft["stats"]), torch.tensor([1 / 3, 1 / 3, 1 / 3]))
    torch.testing.assert_close(torch.stack(hard["stats"]), torch.tensor([1.0, 0.0, 0.0]))


def test_no_valid_pixels_remain_missing_not_perfect_zero_risk():
    missing = torch.full((2, 2), float("nan"))
    result = get_correctness_stats(
        missing, missing, missing, missing, correctness_norm="softmin", temperature=0.25
    )
    assert all(torch.isnan(value) for value in result["stats"])
