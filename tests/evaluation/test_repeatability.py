import pytest
import torch

from trustsr.evaluation.repeatability import (
    RepeatabilityError,
    RepeatabilitySummary,
    run_repeatability,
)

LR = torch.zeros((4, 2, 3), dtype=torch.float32)


class SequenceModel:
    name = "sequence"
    scale = 4

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def predict(self, lr):
        self.calls += 1
        return next(self.outputs)


def output(value=0.5):
    return torch.full((4, 8, 12), value, dtype=torch.float32)


def test_exactly_equal_outputs_return_summary_and_cpu_float32_first():
    first, summary = run_repeatability(SequenceModel([output(), output()]), LR)

    assert first.dtype is torch.float32
    assert first.device.type == "cpu"
    assert first.is_contiguous()
    assert not first.requires_grad
    assert summary.bitwise_equal
    assert summary.max_abs_diff == 0.0
    assert summary.tolerance == 1e-6
    assert summary.as_dict()["first_sha256"] == summary.first_sha256


def test_non_bitwise_outputs_within_tolerance_pass():
    first, summary = run_repeatability(
        SequenceModel([output(), output(0.5000005)]), LR
    )

    assert not summary.bitwise_equal
    assert summary.max_abs_diff <= 1e-6
    assert first.equal(output())


def test_outputs_above_tolerance_fail_after_two_calls():
    model = SequenceModel([output(), output(0.6)])

    with pytest.raises(RepeatabilityError, match="tolerance"):
        run_repeatability(model, LR)
    assert model.calls == 2


@pytest.mark.parametrize(
    "outputs",
    [
        [torch.zeros((4, 7, 12)), output()],
        [torch.zeros((4, 8, 12), dtype=torch.float64), output()],
        [torch.full((4, 8, 12), float("nan")), output()],
        [torch.full((4, 8, 12), -0.1), output()],
        [output(), torch.full((4, 8, 12), 1.1)],
    ],
)
def test_invalid_outputs_fail(outputs):
    with pytest.raises(RepeatabilityError):
        run_repeatability(SequenceModel(outputs), LR)


def test_tolerance_must_be_finite_nonnegative():
    for tolerance in (-1.0, float("nan"), float("inf")):
        with pytest.raises(RepeatabilityError, match="tolerance"):
            run_repeatability(SequenceModel([output(), output()]), LR, tolerance=tolerance)


def test_model_is_called_exactly_twice_even_when_first_call_fails_validation():
    model = SequenceModel([torch.zeros((4, 7, 12)), output()])
    with pytest.raises(RepeatabilityError):
        run_repeatability(model, LR)
    assert model.calls == 2


def test_summary_is_frozen():
    summary = RepeatabilitySummary("a", "b", False, 0.1, 1e-6)
    with pytest.raises(AttributeError):
        summary.tolerance = 0.2
