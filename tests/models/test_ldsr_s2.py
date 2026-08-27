import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

import trustsr.models.ldsr_s2 as ldsr_s2
from trustsr.models.ldsr_assets import (
    CHECKPOINT_NAME,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    CHECKPOINT_URL,
    CONFIG_SHA256,
)
from trustsr.models.ldsr_s2 import LDSRS2X4
from trustsr.models.protocols import SRModel


class FakeBackend:
    def __init__(self, output: torch.Tensor | None = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.output = output
        self.error = error

    def __call__(self, value: torch.Tensor, **kwargs: object) -> torch.Tensor:
        self.calls.append(
            {
                "input": value,
                "kwargs": kwargs,
                "inference_mode": torch.is_inference_mode_enabled(),
                "python_random": random.random(),
                "numpy_random": float(np.random.random()),
                "torch_random": torch.rand((), device=value.device).cpu(),
            }
        )
        if self.error is not None:
            raise self.error
        if self.output is None:
            return torch.linspace(-1, 2, 4 * 512 * 512, device=value.device).reshape(
                1, 4, 512, 512
            )
        return self.output.to(value.device)


def _input() -> torch.Tensor:
    return torch.full((4, 128, 128), 0.5, dtype=torch.float32)


def _numpy_state_equal(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_contract_forwards_exact_batch_keywords_and_inference_mode() -> None:
    backend = FakeBackend()
    model = LDSRS2X4(backend, device="cpu")

    result = model.predict(_input())

    assert isinstance(model, SRModel)
    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert isinstance(call["input"], torch.Tensor)
    assert call["input"].shape == (1, 4, 128, 128)
    assert call["input"].dtype == torch.float32
    assert call["input"].device == torch.device("cpu")
    assert call["kwargs"] == {
        "sampling_steps": 100,
        "sampling_eta": 0.95,
        "sampling_temperature": 1.0,
        "histogram_matching": True,
        "save_iterations": False,
        "verbose": False,
    }
    assert call["inference_mode"] is True
    assert result.shape == (4, 512, 512)
    assert result.dtype == torch.float32 and result.device.type == "cpu"
    assert result.is_contiguous() and not result.requires_grad and result.grad_fn is None
    assert result.min() == 0 and result.max() == 1


@pytest.mark.parametrize(
    "value",
    [
        torch.ones(3, 128, 128),
        torch.ones(4, 127, 128),
        torch.ones(4, 128, 128, dtype=torch.float64),
        torch.full((4, 128, 128), float("nan")),
        torch.full((4, 128, 128), float("inf")),
        torch.full((4, 128, 128), -0.01),
        torch.full((4, 128, 128), 1.01),
    ],
)
def test_rejects_invalid_input(value: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        LDSRS2X4(FakeBackend(), device="cpu").predict(value)


@pytest.mark.parametrize(
    "output",
    [
        torch.ones(4, 512, 512),
        torch.ones(1, 4, 511, 512),
        torch.full((1, 4, 512, 512), float("nan")),
        torch.full((1, 4, 512, 512), float("inf")),
    ],
)
def test_rejects_invalid_backend_output_before_cpu_transfer(output: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        LDSRS2X4(FakeBackend(output), device="cpu").predict(_input())


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("seed", True),
        ("seed", -1),
        ("seed", 1.0),
        ("sampling_steps", True),
        ("sampling_steps", 0),
        ("sampling_steps", 1.0),
        ("sampling_eta", True),
        ("sampling_eta", 1),
        ("sampling_eta", -0.1),
        ("sampling_eta", float("nan")),
        ("sampling_temperature", True),
        ("sampling_temperature", 1),
        ("sampling_temperature", -0.1),
        ("sampling_temperature", float("inf")),
        ("histogram_matching", 1),
        ("histogram_matching", "true"),
    ],
)
def test_rejects_invalid_configuration_values(argument: str, value: object) -> None:
    kwargs = {argument: value}
    with pytest.raises(ValueError):
        LDSRS2X4(FakeBackend(), device="cpu", **kwargs)


def test_predict_isolates_repeatable_backend_randomness() -> None:
    backend = FakeBackend()
    model = LDSRS2X4(backend, device="cpu", seed=23)

    model.predict(_input())
    model.predict(_input())

    first, second = backend.calls
    assert first["python_random"] == second["python_random"]
    assert first["numpy_random"] == second["numpy_random"]
    assert torch.equal(first["torch_random"], second["torch_random"])


def test_predict_restores_python_numpy_and_torch_rng_state() -> None:
    random.seed(111)
    np.random.seed(222)
    torch.manual_seed(333)
    expected_python = random.getstate()
    expected_numpy = np.random.get_state()
    expected_torch = torch.random.get_rng_state()

    LDSRS2X4(FakeBackend(), device="cpu").predict(_input())

    assert random.getstate() == expected_python
    assert _numpy_state_equal(np.random.get_state(), expected_numpy)
    assert torch.equal(torch.random.get_rng_state(), expected_torch)


@pytest.mark.parametrize("raises", [False, True])
def test_predict_restores_cudnn_flags_after_success_and_exception(raises: bool) -> None:
    original_benchmark = torch.backends.cudnn.benchmark
    original_deterministic = torch.backends.cudnn.deterministic
    torch.backends.cudnn.benchmark = not original_benchmark
    torch.backends.cudnn.deterministic = not original_deterministic
    expected = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    try:
        backend = FakeBackend(error=RuntimeError("backend failure") if raises else None)
        if raises:
            with pytest.raises(RuntimeError, match="backend failure"):
                LDSRS2X4(backend, device="cpu").predict(_input())
        else:
            LDSRS2X4(backend, device="cpu").predict(_input())
        assert (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) == expected
    finally:
        torch.backends.cudnn.benchmark = original_benchmark
        torch.backends.cudnn.deterministic = original_deterministic


def test_provenance_is_complete_scalar_and_normalizes_cuda_device() -> None:
    provenance = LDSRS2X4(FakeBackend(), device="cuda:0").provenance()

    assert provenance == {
        "name": "ldsr-s2-x4",
        "scale": 4,
        "implementation_schema_version": 1,
        "opensr_model_version": "1.1.1",
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "checkpoint_name": CHECKPOINT_NAME,
        "checkpoint_url": CHECKPOINT_URL,
        "checkpoint_size": CHECKPOINT_SIZE,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "device": "cuda",
        "seed": 3407,
        "sampling_steps": 100,
        "sampling_eta": 0.95,
        "sampling_temperature": 1.0,
        "histogram_matching": True,
        "output_policy": "clip_to_[0,1]",
    }
    assert json.loads(json.dumps(provenance)) == provenance
    assert all(
        value is None or isinstance(value, (str, int, float, bool))
        for value in provenance.values()
    )


@pytest.mark.parametrize(
    "field",
    [
        "seed",
        "sampling_steps",
        "sampling_eta",
        "sampling_temperature",
        "histogram_matching",
        "checkpoint_sha256",
        "config_sha256",
        "opensr_model_version",
        "torch_version",
        "cuda_runtime",
    ],
)
def test_provenance_fields_change_cache_identity(field: str) -> None:
    from trustsr.artifacts.predictions import build_identity

    provenance = LDSRS2X4(FakeBackend(), device="cuda:0").provenance()
    changed = dict(provenance)
    value = changed[field]
    changed[field] = "different" if value is None or isinstance(value, str) else not value
    lr = torch.zeros((4, 128, 128), dtype=torch.float32)

    assert build_identity(provenance, "source", "id", lr).key != build_identity(
        changed, "source", "id", lr
    ).key


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_from_pretrained_rejects_nonproduction_devices_before_backend_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, device: str
) -> None:
    constructed = False

    def fail_build(*args: object, **kwargs: object) -> None:
        nonlocal constructed
        constructed = True

    monkeypatch.setattr(ldsr_s2.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ldsr_s2, "build_verified_backend", fail_build)

    with pytest.raises(ValueError, match="requested CUDA device is unavailable"):
        LDSRS2X4.from_pretrained(tmp_path, device=device)
    assert constructed is False


def test_from_pretrained_constructs_only_after_available_cuda_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = FakeBackend()
    calls: list[tuple[Path | str, str]] = []
    monkeypatch.setattr(ldsr_s2.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        ldsr_s2,
        "build_verified_backend",
        lambda model_dir, *, device: calls.append((model_dir, device)) or backend,
    )

    model = LDSRS2X4.from_pretrained(tmp_path, device="cuda:0")

    assert model._backend is backend
    assert calls == [(tmp_path, "cuda:0")]
