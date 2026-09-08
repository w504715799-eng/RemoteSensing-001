"""The new cloud policy must isolate CPU threads without changing old adapters."""

import importlib

import pytest
import torch

from trustsr.artifacts.predictions import build_identity
from trustsr.models.sen2srlite import SEN2SRLiteX4


def policy():
    return importlib.import_module("trustsr.models.cloud_sen2srlite").CloudSEN2SRLiteX4


@pytest.mark.parametrize("fail", [False, True])
def test_predict_runs_with_historical_threads_and_restores_caller(fail):
    original = torch.get_num_threads()
    observed = []

    def backend(lr):
        observed.append(torch.get_num_threads())
        if fail:
            raise RuntimeError("backend failure")
        return torch.zeros(1, 4, 512, 512)

    try:
        torch.set_num_threads(1)
        model = policy()(backend, device="cpu")
        if fail:
            with pytest.raises(RuntimeError, match="backend failure"):
                model.predict(torch.zeros(4, 128, 128))
        else:
            result = model.predict(torch.zeros(4, 128, 128))
            assert result.shape == (4, 512, 512)
            assert torch.count_nonzero(result) == 0
        assert observed == [96]
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(original)


def test_execution_policy_changes_cache_identity_without_changing_weights():
    model = policy()(lambda lr: lr, device="cpu")
    old = SEN2SRLiteX4(lambda lr: lr, device="cpu")
    provenance = model.provenance()
    assert all(provenance[k] == v for k, v in old.provenance().items())
    assert provenance["cpu_intraop_threads"] == 96
    assert provenance["cpu_interop_threads"] == torch.get_num_interop_threads()
    lr = torch.zeros(4, 128, 128)
    assert build_identity(provenance, "synthetic", "roi", lr).key != build_identity(
        old.provenance(), "synthetic", "roi", lr
    ).key


def test_gpu_use_is_rejected_before_backend_call():
    def backend(lr):
        pytest.fail("CPU policy must not call a GPU backend")

    model = policy()(backend, device="cuda:0")
    with pytest.raises(ValueError, match="CPU"):
        model.predict(torch.zeros(4, 128, 128))


def test_missing_assets_fail_without_download_or_directory_creation(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        policy().from_pretrained(missing)
    assert list(tmp_path.iterdir()) == []
