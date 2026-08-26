from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SRPair:
    sample_id: str
    source: str
    lr: torch.Tensor
    hr: torch.Tensor
    scale: int

    def validate(self) -> None:
        if self.scale != 4:
            raise ValueError("Phase 0 supports scale=4 only")
        if self.lr.ndim != 3 or self.hr.ndim != 3:
            raise ValueError("lr and hr must use channel-first CHW layout")
        if self.lr.shape[0] != 4 or self.hr.shape[0] != 4:
            raise ValueError("lr and hr must have four RGBN channels")
        expected = (self.lr.shape[1] * self.scale, self.lr.shape[2] * self.scale)
        if self.hr.shape[1:] != expected:
            raise ValueError("hr height and width must be exactly scale times lr")
        for name, tensor in (("lr", self.lr), ("hr", self.hr)):
            if tensor.dtype != torch.float32:
                raise ValueError(f"{name} must use torch.float32")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite reflectance")
            if tensor.min().item() < 0.0 or tensor.max().item() > 1.0:
                raise ValueError(f"{name} reflectance must be in [0, 1]")
