"""Offline semantic checks of the pinned historical metadata export."""

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from research.trustmask.recover_manifest import MANIFEST_SHA256

PROJECTION_SHA256 = "8bf11a4851a97a1af26107f8f91c65946ca931b0156fa85cf8c436d7589bd08a"
SOURCE = {
    "revision": "c370504201072fdb1dd388013ab8c0fc7d00a57e",
    "object_sha256": "c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.utcoffset() is not None, "timestamp lacks timezone")
    return parsed.astimezone(UTC)


def audit_rows(rows, members, geometry):
    require(len(rows) == len(members) == len(geometry), "member count mismatch")
    member_map = {r[0]: (i, r[1], r[2]) for i, r in enumerate(members)}
    geom_map = {r[0]: r[1:] for r in geometry}
    require(len(member_map) == len(geom_map) == len(rows), "duplicate reference identity")
    seen = set()
    days = Counter()
    hours = {f: Counter() for f in ("time_start", "lr_time_start", "hr_time_start")}
    ranges = {f: [] for f in hours}
    equal = 0
    for row in rows:
        identity = row["sample_id"]
        require(
            identity not in seen and identity in member_map and identity in geom_map,
            "duplicate or unknown member",
        )
        seen.add(identity)
        index, lon, lat = member_map[identity]
        require(row["source"] == SOURCE, "source identity mismatch")
        require(int(row["source_index"]) == index, "source index mismatch")
        actual = [float(row["centroid"][k]) for k in ("longitude", "latitude")]
        require(all(math.isfinite(v) for v in actual) and actual == [lon, lat], "centroid mismatch")
        crs, transform, shape = geom_map[identity]
        require(
            row["crs"] == crs
            and [float(v) for v in row["geotransform"]] == transform
            and [int(v) for v in row["raster_shape"]] == shape,
            "geometry mismatch",
        )
        times = {f: timestamp(row[f]) for f in hours}
        signed = (times["hr_time_start"].date() - times["lr_time_start"].date()).days
        require(
            signed == int(row["days_between"]) and signed in {-1, 0, 1}, "signed UTC date mismatch"
        )
        days[str(signed)] += 1
        equal += times["time_start"] == times["lr_time_start"]
        for f, t in times.items():
            hours[f][t.strftime("%H:%M:%SZ")] += 1
            ranges[f].append(t.isoformat().replace("+00:00", "Z"))
    return {
        "member_count": len(rows),
        "identity_source_index_centroid_geometry_checks": "all_passed",
        "signed_utc_date_checks": "all_passed",
        "top_equals_lr_count": equal,
        "signed_day_counts": dict(sorted(days.items())),
        "time_of_day_counts": {f: dict(sorted(c.items())) for f, c in hours.items()},
        "timestamp_ranges": {f: [min(v), max(v)] for f, v in ranges.items()},
        "s2_product_identity": "unresolved",
        "timestamp_semantics": "stored_metadata_not_verified_satellite_sensing_instants",
        "source_independence": "not_certified",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    raw = (args.export / "metadata.jsonl").read_bytes()
    receipt = json.loads((args.export / "receipt.json").read_bytes())
    require(hashlib.sha256(raw).hexdigest() == PROJECTION_SHA256, "projection digest mismatch")
    require(
        receipt["manifest_sha256"] == MANIFEST_SHA256
        and receipt["projection_sha256"] == PROJECTION_SHA256
        and receipt["member_count"] == 8000,
        "receipt mismatch",
    )
    paths = [root / "artifacts/datasets/progressive-crosssensor-members-v1.json"]
    paths += [
        root / f"artifacts/datasets/progressive-crosssensor-geometry-{i}.json" for i in range(1, 5)
    ]
    blobs = [p.read_bytes() for p in paths]
    tables = [json.loads(b)["rows"] for b in blobs]
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) == 8000, "expected 8000 members")
    report = audit_rows(rows, tables[0], [r for table in tables[1:] for r in table])
    report.update(
        {
            "schema": "trustmask-historical-manifest-audit-v1",
            "manifest_sha256": MANIFEST_SHA256,
            "projection_sha256": PROJECTION_SHA256,
            "reference_sha256": {
                str(p.relative_to(root)): hashlib.sha256(b).hexdigest()
                for p, b in zip(paths, blobs, strict=True)
            },
            "recovery": "remote_pinned_digest_verified_before_projection",
            "pixels_opened": False,
            "gpu_used": False,
        }
    )
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
