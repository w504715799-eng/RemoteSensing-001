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
    assert provenance["torch_version"] == torch.__version__
    assert provenance["implementation_schema_version"] == 1
    for name, digest in MODEL_ASSET_SHA256.items():
        assert provenance[f"asset_sha256:{name}"] == digest


def test_sen2srlite_runtime_and_adapter_versions_affect_cache_identity():
    from trustsr.artifacts.predictions import build_identity

    lr = torch.zeros((4, 128, 128), dtype=torch.float32)
    provenance = SEN2SRLiteX4(FakeBackend()).provenance()
    assert build_identity(provenance, "source", "id", lr).key != build_identity(
        dict(provenance, torch_version="different"), "source", "id", lr
    ).key
    assert build_identity(provenance, "source", "id", lr).key != build_identity(
        dict(provenance, implementation_schema_version=2), "source", "id", lr
    ).key


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


def test_complete_valid_cache_skips_download_and_verifies_before_load(monkeypatch, tmp_path):
    events = []
    assets = {name: f"verified {name}".encode() for name in MODEL_ASSET_SHA256}
    for name, content in assets.items():
        (tmp_path / name).write_bytes(content)
        monkeypatch.setitem(MODEL_ASSET_SHA256, name, hashlib.sha256(content).hexdigest())

    original_verify = sen2srlite.verify_model_assets

    def record_verify(root):
        events.append(("verify", root))
        original_verify(root)

    loader = object()
    monkeypatch.setattr(
        sen2srlite.mlstac,
        "download",
        lambda *_: (_ for _ in ()).throw(AssertionError("download must not run")),
    )
    monkeypatch.setattr(sen2srlite, "verify_model_assets", record_verify)
    monkeypatch.setattr(
        sen2srlite.mlstac,
        "load",
        lambda manifest: events.append(("load", manifest)) or loader,
    )

    assert sen2srlite.download_verified_model(tmp_path) is loader
    assert events == [
        ("verify", tmp_path),
        ("load", tmp_path / "mlm.json"),
    ]


def test_complete_invalid_cache_skips_download_and_load(monkeypatch, tmp_path):
    calls = []
    for name in MODEL_ASSET_SHA256:
        (tmp_path / name).write_bytes(b"invalid")

    monkeypatch.setattr(
        sen2srlite.mlstac,
        "download",
        lambda *_: calls.append("download"),
    )
    monkeypatch.setattr(
        sen2srlite.mlstac,
        "load",
        lambda *_: calls.append("load"),
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        sen2srlite.download_verified_model(tmp_path)
    assert calls == []


def test_partial_cache_uses_official_download_path(monkeypatch, tmp_path):
    events = []
    assets = {name: f"downloaded {name}".encode() for name in MODEL_ASSET_SHA256}
    for name, content in assets.items():
        monkeypatch.setitem(MODEL_ASSET_SHA256, name, hashlib.sha256(content).hexdigest())
    (tmp_path / "mlm.json").write_bytes(assets["mlm.json"])

    class Downloaded:
        source = str(tmp_path)

    loader = object()

    original_verify = sen2srlite.verify_model_assets

    def download(url, cache_dir):
        events.append(("download", url, cache_dir))
        for name, content in assets.items():
            (cache_dir / name).write_bytes(content)
        return Downloaded()

    def record_verify(root):
        events.append(("verify", root))
        original_verify(root)

    monkeypatch.setattr(
        sen2srlite.mlstac,
        "download",
        download,
    )
    monkeypatch.setattr(sen2srlite, "verify_model_assets", record_verify)
    monkeypatch.setattr(
        sen2srlite.mlstac,
        "load",
        lambda manifest: events.append(("load", manifest)) or loader,
    )

    assert sen2srlite.download_verified_model(tmp_path) is loader
    assert events == [
        ("download", sen2srlite.MODEL_MANIFEST_URL, tmp_path),
        ("verify", tmp_path),
        ("load", tmp_path / "mlm.json"),
    ]


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
