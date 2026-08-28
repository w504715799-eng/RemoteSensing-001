import torch
import torch.nn.functional as functional


class BicubicX4:
    name = "bicubic-x4"
    scale = 4

    @torch.inference_mode()
    def predict(self, lr: torch.Tensor) -> torch.Tensor:
        if lr.ndim != 3 or lr.shape[0] != 4:
            raise ValueError("expected RGBN tensor with shape (4, H, W)")
        sr = functional.interpolate(
            lr.unsqueeze(0),
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        return sr.to(dtype=torch.float32, device="cpu").clamp_(0.0, 1.0)
