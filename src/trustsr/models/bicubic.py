import torch
import torch.nn.functional as functional

from trustsr.models.protocols import JsonScalar


class BicubicX4:
    name = "bicubic-x4"
    scale = 4

    def provenance(self) -> dict[str, JsonScalar]:
        return {
            "name": self.name,
            "scale": self.scale,
            "implementation": "torch.nn.functional.interpolate",
            "mode": "bicubic",
            "align_corners": False,
            "antialias": True,
            "output_policy": "clip_to_[0,1]",
            "torch_version": torch.__version__,
            "implementation_schema_version": 1,
        }

    @torch.inference_mode()
    def predict(self, lr: torch.Tensor) -> torch.Tensor:
        if not isinstance(lr, torch.Tensor) or lr.dtype != torch.float32:
            raise ValueError("expected float32 tensor")
        if lr.ndim != 3 or lr.shape[0] != 4:
            raise ValueError("expected RGBN tensor with shape (4, H, W)")
        if not torch.isfinite(lr).all():
            raise ValueError("input must contain only finite values")
        if (lr < 0).any() or (lr > 1).any():
            raise ValueError("input values must be in [0, 1]")
        sr = functional.interpolate(
            lr.unsqueeze(0),
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        return sr.to(dtype=torch.float32, device="cpu").clamp_(0.0, 1.0).detach().contiguous()
