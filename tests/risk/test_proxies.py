import pytest
import torch

from trustsr.risk.proxies import (
    lr_reprojection_l1_score,
    three_model_disagreement_score,
)


def test_lr_reprojection_l1_score_uses_area_and_constant_4x4_blocks() -> None:
    prediction = torch.zeros((4, 8, 8), dtype=torch.float32)
    prediction[:, :4, :4] = 0.4
    lr = torch.zeros((4, 2, 2), dtype=torch.float32)
    expected = torch.zeros((8, 8), dtype=torch.float64)
    expected[:4, :4] = 0.4

    actual = lr_reprojection_l1_score(prediction, lr, scale=4)

    assert actual.dtype == torch.float64
    assert actual.device.type == "cpu"
    assert actual.is_contiguous()
    # Float32 cached inputs represent 0.4 as 0.4000000059604645 after
    # conversion to float64; compare the hand-derived map within float32 precision.
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-7)


def test_three_model_disagreement_is_population_variance_over_models_and_bands() -> None:
    predictions = tuple(
        torch.full((4, 2, 3), value, dtype=torch.float32)
        for value in (0.0, 0.5, 1.0)
    )

    actual = three_model_disagreement_score(predictions)

    assert actual.dtype == torch.float64
    assert actual.device.type == "cpu"
    assert actual.is_contiguous()
    assert torch.allclose(actual, torch.full((2, 3), 1.0 / 6.0, dtype=torch.float64))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_score_proxies_reject_invalid_reflectance(bad: float) -> None:
    prediction = torch.full((4, 8, 8), bad, dtype=torch.float32)
    lr = torch.zeros((4, 2, 2), dtype=torch.float32)

    with pytest.raises(ValueError):
        lr_reprojection_l1_score(prediction, lr, scale=4)


@pytest.mark.parametrize(
    "prediction, lr",
    [
        ("not a tensor", torch.zeros((4, 2, 2), dtype=torch.float32)),
        (torch.zeros((3, 8, 8), dtype=torch.float32), torch.zeros((4, 2, 2), dtype=torch.float32)),
        (torch.zeros((4, 8, 8), dtype=torch.float64), torch.zeros((4, 2, 2), dtype=torch.float32)),
        (torch.zeros((4, 8, 8), dtype=torch.float32), torch.zeros((3, 2, 2), dtype=torch.float32)),
        (torch.zeros((4, 8, 8), dtype=torch.float32), torch.zeros((4, 3, 2), dtype=torch.float32)),
    ],
)
def test_lr_reprojection_l1_score_rejects_invalid_inputs(prediction, lr) -> None:
    with pytest.raises(ValueError):
        lr_reprojection_l1_score(prediction, lr)


def test_lr_reprojection_l1_score_rejects_non_four_scale() -> None:
    prediction = torch.zeros((4, 8, 8), dtype=torch.float32)
    lr = torch.zeros((4, 2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="scale must equal 4"):
        lr_reprojection_l1_score(prediction, lr, scale=2)


def test_lr_reprojection_l1_score_rejects_mismatched_spatial_shape() -> None:
    prediction = torch.zeros((4, 8, 7), dtype=torch.float32)
    lr = torch.zeros((4, 2, 2), dtype=torch.float32)

    with pytest.raises(ValueError):
        lr_reprojection_l1_score(prediction, lr)


def test_score_proxies_do_not_mutate_inputs() -> None:
    prediction = torch.rand((4, 8, 8), dtype=torch.float32)
    lr = torch.rand((4, 2, 2), dtype=torch.float32)
    prediction_before, lr_before = prediction.clone(), lr.clone()

    lr_reprojection_l1_score(prediction, lr)
    three_model_disagreement_score((prediction, prediction.clone(), prediction.clone()))

    assert torch.equal(prediction, prediction_before)
    assert torch.equal(lr, lr_before)


@pytest.mark.parametrize(
    "predictions",
    [(), (torch.zeros((4, 2, 2)),) * 2, (torch.zeros((4, 2, 2)),) * 4],
)
def test_three_model_disagreement_requires_exactly_three_predictions(predictions) -> None:
    with pytest.raises(ValueError):
        three_model_disagreement_score(predictions)


def test_three_model_disagreement_rejects_mismatched_shapes_and_invalid_member() -> None:
    valid = torch.zeros((4, 2, 2), dtype=torch.float32)

    with pytest.raises(ValueError):
        three_model_disagreement_score((valid, torch.zeros((4, 3, 2)), valid))
    with pytest.raises(ValueError):
        three_model_disagreement_score((valid, torch.zeros((4, 2, 2), dtype=torch.float64), valid))
