import opensr_test
import torch

from trustsr.contracts import SRPair

METRIC_KEYS = (
    "reflectance",
    "spectral",
    "spatial",
    "synthesis",
    "ha_metric",
    "om_metric",
    "im_metric",
)


def compute_opensr_metrics(pair: SRPair, sr: torch.Tensor) -> dict[str, float]:
    pair.validate()
    if sr.shape != pair.hr.shape:
        raise ValueError("sr shape must match hr shape")
    if not torch.isfinite(sr).all():
        raise ValueError("sr contains non-finite values")

    evaluator = opensr_test.Metrics(device="cpu")
    raw = evaluator.compute(
        lr=pair.lr.cpu(),
        sr=sr.detach().to(torch.float32).cpu(),
        hr=pair.hr.cpu(),
        gradient_threshold="auto",
    )
    return {key: float(raw[key]) for key in METRIC_KEYS}
