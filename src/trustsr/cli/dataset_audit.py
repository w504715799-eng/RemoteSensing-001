"""Emit a deterministic, metadata-only SEN2NAIPv2 provenance audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trustsr.data.provenance import DatasetSource, load_dataset_source

_VARIANT_PREFIXES = {
    "crosssensor": "sen2naipv2-crosssensor",
    "histmatch": "sen2naipv2-histmatch",
    "unet": "sen2naipv2-unet",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the local metadata audit parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/datasets/sen2naipv2-source-v1.json"),
    )
    return parser


def _variant_counts(source: DatasetSource) -> dict[str, int]:
    counts = {variant: 0 for variant in _VARIANT_PREFIXES}
    for item in source.objects:
        for variant, prefix in _VARIANT_PREFIXES.items():
            if item.path.startswith(prefix):
                counts[variant] += 1
                break
        else:
            raise ValueError(f"unrecognized SEN2NAIPv2 object path: {item.path}")
    return counts


def build_payload(source: DatasetSource) -> dict[str, object]:
    """Build JSON-native, local-only audit evidence from validated source metadata."""
    return {
        "schema": "trustsr.dataset-audit.v1",
        "source_schema": source.schema,
        "repository": source.repository,
        "revision": source.revision,
        "license_claim": source.license_claim,
        "card_sha256": source.card_sha256,
        "bands": list(source.bands),
        "scale": source.scale,
        "lr_shape": list(source.lr_shape),
        "hr_shape": list(source.hr_shape),
        "object_count": len(source.objects),
        "total_bytes": source.total_bytes,
        "variant_counts": _variant_counts(source),
        "metadata_only": True,
        "network_accessed": False,
        "pixel_data_downloaded": False,
        "local_real_pixel_policy": "forbidden",
        "ready_for_phase2b1_schema_probe": True,
    }


def main(argv: list[str] | None = None) -> int:
    """Print the canonical local metadata audit payload."""
    args = build_parser().parse_args(argv)
    payload = build_payload(load_dataset_source(args.source))
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
