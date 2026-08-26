import hashlib

import pytest
import torch

import trustsr.models.sen2srlite as sen2srlite
from trustsr.models.sen2srlite import (
    MODEL_ASSET_SHA256,
    SEN2SRLiteX4,
    verify_model_assets,
)


class FakeBackend:
    def __init__(self, output=None):
        self.seen = None
        self.inference_mode = False
        self.output = output

    def __call__(self, value):
        self.seen = value
        self.inference_mode = torch.is_inference_mode_enabled()
        return self.output if self.output is not None else torch.zeros(1, 4, 512, 512)


def test_contract_and_backend_batch():
    backend = FakeBackend(torch.linspace(-1, 2, 4 * 512 * 512).reshape(1, 4, 512, 512))
    model = SEN2SRLiteX4(backend)
    result = model.predict(torch.ones(4, 128, 128))
    assert backend.seen.shape == (1, 4, 128, 128)
    assert backend.inference_mode
    assert result.shape == (4, 512, 512)
    assert result.dtype == torch.float32 and result.device.type == "cpu"
    assert result.is_contiguous() and result.min() == 0 and result.max() == 1
    assert not result.requires_grad and result.grad_fn is None


def test_rejects_wrong_input_dtype():
    with pytest.raises(ValueError, match="float32"):
        SEN2SRLiteX4(FakeBackend()).predict(torch.zeros(4, 128, 128, dtype=torch.float64))


@pytest.mark.parametrize(
    "value",
    [
        torch.ones(3, 128, 128),
        torch.ones(4, 127, 128),
        torch.full((4, 128, 128), 2.0),
        torch.full((4, 128, 128), float("nan")),
    ],
)
def test_rejects_invalid_input(value):
    with pytest.raises(ValueError):
        SEN2SRLiteX4(FakeBackend()).predict(value)


@pytest.mark.parametrize(
    "output",
    [torch.ones(4, 512, 512), torch.full((1, 4, 512, 512), float("nan"))],
)
def test_rejects_invalid_output(output):
    with pytest.raises(ValueError):
        SEN2SRLiteX4(FakeBackend(output)).predict(torch.zeros(4, 128, 128))


def test_verifies_named_assets(tmp_path):
    expected = {}
    for name in MODEL_ASSET_SHA256:
        path = tmp_path / name
        path.write_bytes(name.encode())
        expected[name] = hashlib.sha256(name.encode()).hexdigest()
    verify_model_assets(tmp_path, expected)
    (tmp_path / "load.py").write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_model_assets(tmp_path, expected)


def test_missing_asset_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing model asset"):
        verify_model_assets(tmp_path, {"model.safetensor": "0"})


def test_provenance_contains_required_scalar_values():
    provenance = SEN2SRLiteX4(FakeBackend(), device="cpu").provenance()
    assert provenance["model_id"] == "SEN2SRLite_NonReference_RGBN_x4"
    assert provenance["manifest_url"].endswith("SEN2SRLite/NonReference_RGBN_x4/mlm.json")
    assert provenance["mlstac_version"] == "0.4.9"
    assert provenance["sen2sr_version"] == "0.8.5"
    assert provenance["device"] == "cpu"
    assert provenance["output_policy"] == "clip_to_[0,1]"
    for name, digest in MODEL_ASSET_SHA256.items():
        assert provenance[f"asset_sha256:{name}"] == digest


def test_verification_failure_prevents_mlm_load(monkeypatch, tmp_path):
    load_called = False

    class Downloaded:
        source = str(tmp_path)

    def fail_load(*_args):
        nonlocal load_called
        load_called = True
        raise AssertionError("mlstac.load must not run")

    monkeypatch.setattr(sen2srlite.mlstac, "download", lambda *_: Downloaded())
    monkeypatch.setattr(sen2srlite.mlstac, "load", fail_load)
    with pytest.raises(FileNotFoundError, match="missing model asset"):
        sen2srlite.download_verified_model(tmp_path)
    assert not load_called


def test_pretrained_verifies_before_loader(monkeypatch, tmp_path):
    events = []

    class Loader:
        source = str(tmp_path)

        def compiled_model(self, **kwargs):
            events.append("compiled")
            return FakeBackend()

    monkeypatch.setattr(
        sen2srlite.mlstac, "download", lambda *_: events.append("download") or Loader()
    )
    monkeypatch.setattr(sen2srlite, "verify_model_assets", lambda root: events.append("verify"))
    monkeypatch.setattr(sen2srlite.mlstac, "load", lambda *_: events.append("load") or Loader())
    SEN2SRLiteX4.from_pretrained(tmp_path)
    assert events == ["download", "verify", "load", "compiled"]
