from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from trustsr.models.ldsr_assets import VerifiedAsset


def _asset(path: Path) -> VerifiedAsset:
    return VerifiedAsset(path, 1, "a" * 64)


def _fake_package(config_path: Path, events: list[str], backend_box: dict):
    package = ModuleType("fake_opensr_model")
    package.__file__ = str(config_path.parents[1] / "__init__.py")

    class Backend:
        def __init__(self, config, device):
            events.append("construct_backend")
            self.config, self.device = config, device
            self.training = True
            self.model = SimpleNamespace(training=True, load_state_dict=self._load)

        def load_pretrained(self):
            raise AssertionError("unsafe auto-loading must not be called")

        def _load(self, state, strict):
            events.append("strict_state_load")
            assert strict is True
            self.loaded = state
            self.training = False
            self.model.training = False

    package.SRLatentDiffusion = Backend
    backend_box["type"] = Backend
    return package


def test_builds_verified_backend_in_safe_event_order(tmp_path, monkeypatch):
    from trustsr.models import ldsr_backend

    events: list[str] = []
    backend_box: dict = {}
    package_root = tmp_path / "package"
    config_path = package_root / "configs" / "config_10m.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(b"config")
    checkpoint = _asset(tmp_path / "checkpoint")
    package = _fake_package(config_path, events, backend_box)

    monkeypatch.setattr(
        ldsr_backend,
        "download_verified_checkpoint",
        lambda _: events.append("download_checkpoint") or checkpoint,
    )
    monkeypatch.setattr(
        ldsr_backend,
        "verify_packaged_config",
        lambda _: events.append("verify_config") or _asset(config_path),
    )

    class OmegaConf:
        @staticmethod
        def load(path):
            events.append("parse_config")
            assert path == config_path
            return {"model": "fake"}

    weights = {
        "state_dict": {"layer.weight": torch.tensor([1.0]), "loss.total": torch.tensor([2.0])}
    }

    def fake_torch_load(path, **kwargs):
        events.append("torch_load_weights_only")
        assert path == checkpoint.path
        assert kwargs == {"map_location": torch.device("cuda:0"), "weights_only": True}
        return weights

    monkeypatch.setattr(ldsr_backend.torch, "load", fake_torch_load)
    backend = ldsr_backend.build_verified_backend(
        tmp_path, device="cuda:0", package_module=package, omega_conf=OmegaConf
    )
    assert events == [
        "download_checkpoint",
        "verify_config",
        "parse_config",
        "construct_backend",
        "torch_load_weights_only",
        "strict_state_load",
    ]
    assert backend.loaded == {"layer.weight": weights["state_dict"]["layer.weight"]}


def test_rejects_invalid_assets_before_backend_construction(tmp_path, monkeypatch):
    from trustsr.models import ldsr_backend

    events: list[str] = []
    package = ModuleType("fake")
    package.__file__ = str(tmp_path / "__init__.py")
    package.SRLatentDiffusion = lambda *_args, **_kwargs: events.append("construct_backend")
    monkeypatch.setattr(
        ldsr_backend,
        "download_verified_checkpoint",
        lambda _: events.append("download_checkpoint") or _asset(tmp_path / "checkpoint"),
    )
    monkeypatch.setattr(
        ldsr_backend,
        "verify_packaged_config",
        lambda _: (_ for _ in ()).throw(ldsr_backend.AssetIntegrityError("bad config")),
    )

    with pytest.raises(ldsr_backend.AssetIntegrityError):
        ldsr_backend.build_verified_backend(
            tmp_path, device="cuda:0", package_module=package, omega_conf=object()
        )
    assert events == ["download_checkpoint"]


@pytest.mark.parametrize(
    "loaded",
    [{}, {"state_dict": []}, {"state_dict": {1: torch.tensor(1)}}, {"state_dict": {"x": 1}}],
)
def test_load_verified_state_dict_rejects_malformed_checkpoint(tmp_path, loaded):
    from trustsr.models.ldsr_backend import BackendLoadError, load_verified_state_dict

    with pytest.raises(BackendLoadError):
        load_verified_state_dict(
            _asset(tmp_path / "checkpoint"),
            map_location="cpu",
            torch_load=lambda *_args, **_kwargs: loaded,
        )


def test_load_verified_state_dict_filters_only_loss_keys(tmp_path):
    from trustsr.models.ldsr_backend import load_verified_state_dict

    tensor = torch.tensor(1.0)
    result = load_verified_state_dict(
        _asset(tmp_path / "checkpoint"),
        map_location="cpu",
        torch_load=lambda *_args, **_kwargs: {
            "state_dict": {"a": tensor, "loss.foo": tensor, "lossy": tensor}
        },
    )
    assert set(result) == {"a"}
    assert result["a"] is tensor


def test_strict_load_error_is_wrapped_and_training_disabled(tmp_path, monkeypatch):
    from trustsr.models import ldsr_backend

    package = ModuleType("fake")
    package.__file__ = str(tmp_path / "__init__.py")
    config = tmp_path / "config.yaml"
    config.write_text("x")
    checkpoint = _asset(tmp_path / "checkpoint")
    monkeypatch.setattr(ldsr_backend, "download_verified_checkpoint", lambda _: checkpoint)
    monkeypatch.setattr(ldsr_backend, "verify_packaged_config", lambda _: _asset(config))
    package.SRLatentDiffusion = lambda *_args, **_kwargs: SimpleNamespace(
        training=True,
        model=SimpleNamespace(
            training=True,
            load_state_dict=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("missing")),
        ),
    )
    monkeypatch.setattr(
        ldsr_backend.torch, "load", lambda *_a, **_k: {"state_dict": {"x": torch.tensor(1)}}
    )
    with pytest.raises(ldsr_backend.BackendLoadError, match="strict state load") as exc:
        ldsr_backend.build_verified_backend(
            tmp_path,
            device="cpu",
            package_module=package,
            omega_conf=SimpleNamespace(load=lambda _: {}),
        )
    assert isinstance(exc.value.__cause__, RuntimeError)
