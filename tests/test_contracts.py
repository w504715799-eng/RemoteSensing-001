import pytest
import torch

from trustsr.contracts import SRPair


def test_valid_rgbn_x4_pair() -> None:
    pair = SRPair(
        sample_id="spot-0",
        source="opensr-test/spot/v3",
        lr=torch.rand(4, 16, 16),
        hr=torch.rand(4, 64, 64),
        scale=4,
    )

    pair.validate()


@pytest.mark.parametrize(
    ("lr", "hr", "message"),
    [
        (torch.rand(3, 16, 16), torch.rand(4, 64, 64), "four RGBN channels"),
        (torch.rand(4, 16, 16), torch.rand(4, 63, 64), "exactly scale"),
        (torch.full((4, 16, 16), 1.1), torch.rand(4, 64, 64), "reflectance"),
    ],
)
def test_invalid_pair_is_rejected(
    lr: torch.Tensor, hr: torch.Tensor, message: str
) -> None:
    pair = SRPair("bad", "fixture", lr, hr, 4)

    with pytest.raises(ValueError, match=message):
        pair.validate()
