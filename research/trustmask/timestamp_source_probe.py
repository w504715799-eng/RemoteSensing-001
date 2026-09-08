"""Three predeclared metadata-only catalog probes; offline replay by default."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo

from research.trustmask.audit_recovered_manifest import PROJECTION_SHA256, require, timestamp

ENDPOINT = "https://earth-search.aws.element84.com/v1/search"
PROBE_SELECTION_SHA256 = "16e9abbb51239d71fd34d6fc861d4059d019e0c54f8d879630f7358850fbe863"
MAX_BYTES = 1024 * 1024
PROPERTIES = (
    "datetime",
    "platform",
    "s2:product_uri",
    "s2:granule_id",
    "s2:processing_baseline",
    "sat:relative_orbit",
    "mgrs:utm_zone",
    "mgrs:latitude_band",
    "mgrs:grid_square",
)


def time_audit(rows):
    zone = ZoneInfo("Europe/Madrid")
    midnight = Counter()
    hr_match = 0
    offsets = Counter()
    for row in rows:
        date = datetime.strptime(row["sample_id"][-8:], "%Y%m%d").date()
        local = {f: timestamp(row[f]).astimezone(zone) for f in ("lr_time_start", "hr_time_start")}
        for f, t in local.items():
            midnight[f] += t.hour == t.minute == t.second == t.microsecond == 0
        hr_match += local["hr_time_start"].date() == date
        offsets[str((local["lr_time_start"].date() - date).days)] += 1
    return {
        "member_count": len(rows),
        "hypothesis_zone": zone.key,
        "local_midnight_counts": dict(midnight),
        "hr_filename_date_match_in_hypothesis_zone": hr_match,
        "lr_local_date_minus_filename_date_counts": dict(sorted(offsets.items())),
        "interpretation": "consistent_with_local_midnight_serialization_not_build_proof",
    }


def query_for(row):
    start = timestamp(row["lr_time_start"]).replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=2) - timedelta(microseconds=1)
    lon, lat = (float(row["centroid"][f]) for f in ("longitude", "latitude"))
    return {
        "collections": "sentinel-2-l2a",
        "limit": "50",
        "bbox": ",".join(
            str(round(v, 6)) for v in (lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02)
        ),
        "datetime": "/".join(t.isoformat().replace("+00:00", "Z") for t in (start, end)),
        "fields": ",".join(
            ["id", "collection", "geometry", "bbox"]
            + ["properties." + f for f in PROPERTIES]
            + ["-assets"]
        ),
    }


def candidate_projection(body, *, require_complete=True):
    require(body.get("type") == "FeatureCollection", "not a feature collection")
    features = body["features"]
    require(len(features) <= 50, "result limit exceeded")
    if require_complete:
        require(
            not any(link.get("rel") == "next" for link in body.get("links", [])),
            "pagination incomplete",
        )
    for key in ("numberMatched", "numberReturned"):
        if key in body and (require_complete or key == "numberReturned"):
            require(int(body[key]) == len(features), "catalog count incomplete")
    result = []
    seen = set()
    for feature in features:
        identity = feature["id"]
        require(isinstance(identity, str) and identity not in seen, "duplicate/invalid item")
        require(feature["collection"] == "sentinel-2-l2a", "unexpected collection")
        seen.add(identity)
        properties = feature["properties"]
        timestamp(properties["datetime"])
        result.append(
            {
                "id": identity,
                "collection": feature["collection"],
                "properties": {k: properties[k] for k in PROPERTIES if k in properties},
                **{k: feature[k] for k in ("geometry", "bbox") if k in feature},
            }
        )
    return sorted(result, key=lambda r: r["id"])


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch(query):
    request = Request(
        ENDPOINT + "?" + urlencode(query),
        headers={
            "Accept": "application/geo+json, application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "trustmask-metadata-probe/1",
        },
    )
    with build_opener(NoRedirect()).open(request, timeout=20) as response:
        require(response.status == 200, "unexpected HTTP status")
        require(
            response.headers.get("Content-Encoding", "identity") == "identity",
            "unexpected compression",
        )
        require(
            response.headers.get_content_type() in {"application/geo+json", "application/json"},
            "unexpected content type",
        )
        raw = response.read(MAX_BYTES + 1)
    require(len(raw) <= MAX_BYTES, "response byte limit exceeded")
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists")
    root = Path(__file__).resolve().parents[2]
    raw = (root / "artifacts/progressive-historical-manifest/metadata.jsonl").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PROJECTION_SHA256, "historical digest mismatch")
    rows = [json.loads(line) for line in raw.splitlines()]
    probe_raw = (root / "research/evidence/crosssensor-source-footprint-audit-v1.json").read_bytes()
    require(
        hashlib.sha256(probe_raw).hexdigest() == PROBE_SELECTION_SHA256,
        "probe selection digest mismatch",
    )
    ids = [p["id"] for p in json.loads(probe_raw)["nested_probes"]]
    require(len(ids) == len(set(ids)) == 3, "expected three fixed probes")
    by_id = {row["sample_id"]: row for row in rows}
    args.cache.mkdir(exist_ok=True)
    results = []
    for i, identity in enumerate(ids):
        query = query_for(by_id[identity])
        path = args.cache / f"catalog-{i}.json"
        receipt_path = args.cache / f"catalog-{i}-receipt.json"
        if not path.exists() and args.fetch:
            for attempt in range(1, 3):
                try:
                    payload = fetch(query)
                except (URLError, TimeoutError) as exc:
                    print(f"probe {i} attempt {attempt}: {type(exc).__name__}", flush=True)
                    if attempt == 2:
                        raise
                else:
                    path.write_bytes(payload)
                    receipt_path.write_text(
                        json.dumps(
                            {
                                "query": query,
                                "response_sha256": hashlib.sha256(payload).hexdigest(),
                                "retrieved_at": datetime.now(UTC).isoformat(),
                                "attempts": attempt,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    break
        payload = path.read_bytes()
        receipt = json.loads(receipt_path.read_bytes())
        require(
            len(payload) <= MAX_BYTES
            and receipt["query"] == query
            and receipt["response_sha256"] == hashlib.sha256(payload).hexdigest(),
            "cached response/receipt mismatch",
        )
        body = json.loads(payload, parse_int=str, parse_float=str)
        candidates = candidate_projection(body, require_complete=False)
        complete = not any(link.get("rel") == "next" for link in body.get("links", [])) and int(
            body.get("numberMatched", -1)
        ) == len(candidates)
        start, end = [timestamp(t) for t in query["datetime"].split("/")]
        local_date = (
            timestamp(by_id[identity]["lr_time_start"]).astimezone(ZoneInfo("Europe/Madrid")).date()
        )
        require(
            all(start <= timestamp(c["properties"]["datetime"]) <= end for c in candidates),
            "candidate outside date window",
        )
        results.append(
            {
                "member_id": identity,
                "request": query,
                "receipt": receipt,
                "response_bytes": len(payload),
                "candidate_count": len(candidates),
                "candidate_raw_utc_date_count": sum(
                    timestamp(c["properties"]["datetime"]).date() == start.date()
                    for c in candidates
                ),
                "candidate_hypothesis_date_count": sum(
                    timestamp(c["properties"]["datetime"]).date() == local_date for c in candidates
                ),
                "candidates": candidates,
                "catalog_number_matched": body.get("numberMatched"),
                "next_link_present": any(
                    link.get("rel") == "next" for link in body.get("links", [])
                ),
                "status": (
                    "bounded_query_complete"
                    if complete
                    else "first_page_only_enumeration_not_certified"
                ),
            }
        )
        print(f"probe {i}: {len(candidates)} candidates", flush=True)
    report = {
        "schema": "trustmask-timestamp-source-probe-v1",
        "endpoint": ENDPOINT,
        "projection_sha256": PROJECTION_SHA256,
        "probe_selection_sha256": hashlib.sha256(probe_raw).hexdigest(),
        "time_audit": time_audit(rows),
        "catalog_probes": results,
        "source_identity": "candidates_only_not_verified_mapping",
        "candidate_numeric_encoding": "numeric_tokens_preserved_as_strings",
        "source_independence": "not_certified",
        "gpu_used": False,
        "assets_opened": False,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
