from trustsr.models.protocols import JsonScalar, SRModel

__all__ = ["BicubicX4", "JsonScalar", "LDSRS2X4", "SRModel", "SEN2SRLiteX4"]


def __getattr__(name: str):
    """Keep convenience exports without importing heavyweight adapters eagerly."""

    if name == "BicubicX4":
        from trustsr.models.bicubic import BicubicX4

        return BicubicX4
    if name == "LDSRS2X4":
        from trustsr.models.ldsr_s2 import LDSRS2X4

        return LDSRS2X4
    if name == "SEN2SRLiteX4":
        from trustsr.models.sen2srlite import SEN2SRLiteX4

        return SEN2SRLiteX4
    raise AttributeError(name)
