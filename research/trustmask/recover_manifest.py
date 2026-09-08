"""Standalone, stdlib-only historical metadata export; never opens referenced assets.

Numeric tokens deliberately remain strings. The export is transport evidence, not
a validated scientific manifest; local membership/geometry/time checks must follow.
"""

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

MANIFEST_SHA256 = "7487b0af2ebef86910e918d5d6b2fb927a6f5e46bac7c2e30be7ffb2ce994482"
MAX_BYTES = 64 * 1024 * 1024
FIELDS = (
    "source",
    "source_index",
    "sample_id",
    "centroid",
    "crs",
    "geotransform",
    "raster_shape",
    "time_start",
    "lr_time_start",
    "hr_time_start",
    "days_between",
)


def project(raw: bytes, expected_sha256: str, expected_count: int) -> bytes:
    """Verify bytes before parsing; test parameters are not exposed by the CLI."""
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("manifest digest mismatch")
    lines = raw.splitlines()
    if len(lines) != expected_count:
        raise ValueError("manifest count mismatch")
    rows = []
    seen = set()
    for line in lines:
        record = json.loads(line, parse_int=str, parse_float=str)
        row = {key: record[key] for key in FIELDS}
        identity = row["sample_id"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("invalid or duplicate member identity")
        seen.add(identity)
        rows.append(json.dumps(row, sort_keys=True, separators=(",", ":")))
    return ("\n".join(rows) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new export directory")
    args = parser.parse_args()
    # O_NONBLOCK avoids blocking on a FIFO; O_NOFOLLOW rejects a symlink leaf.
    fd = os.open(args.manifest, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
            raise ValueError("manifest must be a bounded regular file")
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("manifest exceeds size limit")
    exported = project(raw, MANIFEST_SHA256, 8000)
    receipt = {
        "schema": "trustmask-historical-metadata-export-v1",
        "manifest_sha256": MANIFEST_SHA256,
        "projection_sha256": hashlib.sha256(exported).hexdigest(),
        "member_count": 8000,
        "numeric_encoding": "JSON numeric tokens exported as strings",
        "scientific_validation": "pending_local_identity_geometry_time_checks",
        "s2_product_ids_available": False,
    }
    args.output.mkdir(parents=False, exist_ok=False)
    (args.output / "metadata.jsonl").write_bytes(exported)
    (args.output / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print("Verified historical digest; exported 8000 metadata records. Local audit pending.")


if __name__ == "__main__":
    main()
