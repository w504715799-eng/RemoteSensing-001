from typing import Protocol, runtime_checkable

import torch

JsonScalar = str | int | float | bool | None


@runtime_checkable
class SRModel(Protocol):
    name: str
    scale: int

    def predict(self, lr: torch.Tensor) -> torch.Tensor: ...

    def provenance(self) -> dict[str, JsonScalar]: ...
