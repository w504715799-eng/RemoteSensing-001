"""Pinned text-only Spain metadata projection; no image or pickle access."""

import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

REVISION = "e4600b9c74a621adeec047e5f6cc7a2d70a58134"
SOURCES = {
    "spain_crops": (28, "cd8c1f32ad0fde7f1ab6b4c80931897e9a280e598fd4d160269312d511e12316"),
    "spain_urban": (20, "98eb3da89578fa4d54360d0b894c7284f3a366d3d46b55e334cb763479a3641c"),
}
FIELDS = ("roi", "lr_file", "hr_file", "lr_gee_id", "crs", "affine")


def project_metadata(raw, *, expected_sha, expected_count):
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("metadata SHA-256 mismatch")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    if reader.fieldnames is None or not set(FIELDS).issubset(reader.fieldnames):
        raise ValueError("missing required metadata column")
    members = []
    for row in reader:
        member = {key: row[key] for key in FIELDS}
        if any(not value or not value.strip() for value in member.values()):
            raise ValueError("empty member metadata")
        affine = [float(value) for value in member["affine"].split(",")]
        if len(affine) != 6 or not all(math.isfinite(v) for v in affine):
            raise ValueError("invalid affine")
        if member["crs"] not in ("EPSG:32630", "EPSG:32631"):
            raise ValueError("unreviewed CRS")
        member["affine"] = affine
        members.append(member)
    if len(members) != expected_count or len({m["roi"] for m in members}) != expected_count:
        raise ValueError("member count or uniqueness mismatch")
    return dict(
        members=sorted(members, key=lambda m: m["roi"]),
        lr_scene_count=len({m["lr_gee_id"] for m in members}),
        hr_source_counts=dict(
            sorted(Counter(m["hr_file"].split("__")[-1] for m in members).items())
        ),
    )


def main():
    subsets = {}
    for subset, (count, digest) in SOURCES.items():
        relative = f"100/{subset}/{subset}_metadata.csv"
        url = f"https://huggingface.co/datasets/isp-uv-es/opensr-test/resolve/{REVISION}/{relative}"
        with urlopen(url, timeout=30) as response:
            raw = response.read()
        subsets[subset] = dict(
            source_path=relative,
            source_sha256=digest,
            **project_metadata(raw, expected_sha=digest, expected_count=count),
        )
    output = dict(
        schema="trustsr.spain-text-metadata.v1",
        revision=REVISION,
        source_version="100",
        proposed_image_version="100",
        image_membership_correspondence="same_version_metadata_payload_check_pending",
        subsets=subsets,
    )
    path = Path(__file__).resolve().parents[2] / "artifacts/datasets/spain-metadata-v1.json"
    raw = json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("refusing to overwrite different published metadata")
    else:
        with path.open("xb") as handle:
            handle.write(raw)
    print(
        json.dumps(
            {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "counts": {s: len(v["members"]) for s, v in subsets.items()},
            }
        )
    )


if __name__ == "__main__":
    main()
