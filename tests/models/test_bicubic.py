import torch

from trustsr.models.bicubic import BicubicX4


def test_bicubic_predicts_rgbn_x4_deterministically() -> None:
    lr = torch.linspace(0, 1, 4 * 8 * 9, dtype=torch.float32).reshape(4, 8, 9)
    model = BicubicX4()

    first = model.predict(lr)
    second = model.predict(lr)

    assert first.shape == (4, 32, 36)
    assert first.dtype == torch.float32
    assert first.device.type == "cpu"
    assert first.is_contiguous()
    assert first.min() >= 0 and first.max() <= 1
    assert torch.equal(first, second)
    assert not first.requires_grad
