import torch

from trustsr.contracts import SRPair
from trustsr.evaluation.opensr_metrics import METRIC_KEYS, compute_opensr_metrics


def test_metric_adapter_returns_plain_floats(monkeypatch) -> None:
    expected = {name: index / 10 for index, name in enumerate(METRIC_KEYS)}

    class FakeMetrics:
        def __init__(self, **kwargs) -> None:
            assert kwargs == {"device": "cpu"}

        def compute(self, *, lr, sr, hr, gradient_threshold):
            assert lr.shape == (4, 4, 4)
            assert sr.shape == hr.shape == (4, 16, 16)
            assert gradient_threshold == "auto"
            return expected

    monkeypatch.setattr("opensr_test.Metrics", FakeMetrics)
    pair = SRPair("fixture", "unit", torch.rand(4, 4, 4), torch.rand(4, 16, 16), 4)

    result = compute_opensr_metrics(pair, torch.rand(4, 16, 16))

    assert result == expected
    assert all(type(value) is float for value in result.values())
