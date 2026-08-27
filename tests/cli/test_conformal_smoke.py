import json

import pytest
import torch

import trustsr.cli.conformal_smoke as conformal_smoke
from trustsr.cli.conformal_smoke import main, run


def test_run_marks_result_as_synthetic_and_uses_roi_calibration() -> None:
    """Catch an incomplete payload or a calibration split that is not ROI-level."""
    result = run(alpha=0.27, window=1)

    assert set(result) == {"schema", "synthetic_smoke", "config", "calibration", "test"}
    assert result["schema"] == "trustsr.conformal-smoke.v1"
    assert result["synthetic_smoke"] is True
    assert result["config"] == {"alpha": 0.27, "channels": 4, "scale": 4, "window": 1}
    assert result["calibration"]["calibration_size"] == 3
    assert 0.0 <= result["test"]["coverage"] <= 1.0


def test_main_prints_canonical_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Catch stdout that is not the single canonical JSON document contract."""
    assert main(["--alpha", "0.27", "--window", "1"]) == 0

    output = capsys.readouterr().out
    assert output.endswith("\n")
    assert output == json.dumps(json.loads(output), sort_keys=True, separators=(",", ":")) + "\n"


def test_main_is_byte_repeatable(capsys: pytest.CaptureFixture[str]) -> None:
    """Catch synthetic execution that adds nondeterministic stdout content."""
    assert main([]) == 0
    first = capsys.readouterr().out
    assert main([]) == 0
    second = capsys.readouterr().out

    assert second == first


@pytest.mark.parametrize(
    "arguments",
    [
        ["--alpha", "0"],
        ["--alpha", "1.01"],
        ["--alpha", "nan"],
        ["--window", "0"],
        ["--window", "2"],
    ],
)
def test_main_rejects_invalid_conformal_parameters(arguments: list[str]) -> None:
    """Catch parameters accepted outside conformal alpha and local-risk window domains."""
    with pytest.raises(SystemExit) as error:
        main(arguments)

    assert error.value.code == 2


def test_run_uses_cpu_tensors_when_cuda_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catch a device-selection regression that changes the CPU-only smoke contract."""
    constructed_devices: list[str] = []
    real_linspace = torch.linspace

    def cpu_observing_linspace(*args: object, **kwargs: object) -> torch.Tensor:
        tensor = real_linspace(*args, **kwargs)
        constructed_devices.append(tensor.device.type)
        return tensor

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(conformal_smoke.torch, "linspace", cpu_observing_linspace)

    run(alpha=0.27, window=1)

    assert constructed_devices == ["cpu", "cpu"]
