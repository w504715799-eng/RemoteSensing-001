"""Explicit metadata-only public source audit. Output directory must be new."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from research.trustmask.catalog import fetch_http_range, fetch_metadata
from research.trustmask.grouping import audit_groups

REVISION = 'c370504201072fdb1dd388013ab8c0fc7d00a57e'
OBJECT = 'sen2naipv2-crosssensor.taco'
SIZE = 9717583850
SOURCE_SHA = 'c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5'
URL = f'https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2/resolve/{REVISION}/{OBJECT}'
HISTORY = (
    ('artifacts/phase2b3a/sen2naipv2-development-score-cache-audit-v1.json',
     'd61c36e2180a2dc3468d4d9aba083ac0925d163ac2bb910e0227138e9fa249f1'),
    ('artifacts/phase2b3b/sen2naipv2-calibration-conformal-cache-audit-v1.json',
     '40aff8aaccb27719d958d42026d68d8efc0c7f79479e3bd99777053ee1d6399f'),
    ('artifacts/phase2b3c/sen2naipv2-internal-test-evaluation-cache-audit-v1.json',
     '4bf048b222d4993adb0fd70b0273f60a994ecb7b3539e58ed3b2d57dcf4f43ff'),
)


def _seen_ids(obj):
    found = set()
    if isinstance(obj, dict):
        if 'sample_id' in obj:
            found.add(obj['sample_id'])
        for value in obj.values():
            found.update(_seen_ids(value))
    elif isinstance(obj, list):
        for value in obj:
            found.update(_seen_ids(value))
    return found


def historical_ids(root):
    seen, history = set(), []
    for name, expected in HISTORY:
        raw = (root / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected:
            raise ValueError('historical identity artifact SHA mismatch')
        ids = _seen_ids(json.loads(raw))
        if len(ids) != 120:
            raise ValueError('historical identity count differs')
        seen.update(ids)
        history.append(dict(path=name, sha256=digest, members=len(ids)))
    if len(seen) != 360:
        raise ValueError('expected 360 distinct historical identities')
    return seen, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    if pa.__version__ != '19.0.1':
        raise ValueError('use isolated pyarrow 19.0.1 for this audit')
    root = Path(__file__).resolve().parents[2]
    seen, history = historical_ids(root)
    args.output.mkdir(parents=True, exist_ok=False)
    receipts = []

    def fetch(offset, length):
        # Per-range query prevents a shared CDN cache from replaying another range.
        raw = fetch_http_range(URL + f'?metadata_range={offset}-{offset + length - 1}',
                               offset, length, SIZE)
        receipts.append(dict(offset=offset, length=length, sha256=hashlib.sha256(raw).hexdigest()))
        (args.output / 'range-receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
        return raw

    parts = fetch_metadata(fetch, SIZE)
    for key in ('header', 'collection', 'directory'):
        (args.output / f'{key}.metadata').write_bytes(parts[key])
    # Decode only identity/geometry columns. Do not print any quality-value columns.
    parquet = pq.ParquetFile(pa.BufferReader(parts['directory']))
    required = ['tortilla:id', 'stac:centroid']
    if not set(required).issubset(parquet.schema_arrow.names):
        raise ValueError('required top-level identity/centroid fields absent')
    table = parquet.read(columns=required)
    rows = []
    for entry in table.to_pylist():
        name, point = entry['tortilla:id'], entry['stac:centroid']
        if not isinstance(name, str) or '__m_' not in name:
            raise ValueError('unreviewed member/source naming')
        match = re.fullmatch(r'POINT\s*\(\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s*\)', point)
        if match is None:
            raise ValueError('invalid centroid WKT')
        rows.append(dict(id=name, lon=float(match[1]), lat=float(match[2]),
                         source_ids=['naip:' + name.split('__', 1)[1]]))
    grouping = audit_groups(rows, seen)
    source_counts = Counter(r['source_ids'][0] for r in rows)
    sizes = Counter(len(g['members']) for g in grouping['eligible_groups'])
    result = dict(
        schema='trustmask.catalog-audit.v1', source_revision=REVISION, source_object=OBJECT,
        source_declared_sha256=SOURCE_SHA, full_source_sha_verified=False, source_size=SIZE,
        range_receipts=receipts, metadata_bytes_received=sum(r['length'] for r in receipts),
        history=history, history_members=len(seen), member_count=len(rows),
        source_count=len(source_counts), grouping_component_count=grouping['component_count'],
        eligible_group_count=len(grouping['eligible_groups']),
        eligible_group_size_distribution=dict(sorted(sizes.items())),
        eligible_member_count=sum(len(g['members']) for g in grouping['eligible_groups']),
        excluded_member_count=len(grouping['excluded_members']),
        missing_seen_ids=grouping['missing_seen_ids'],
        history_complete=grouping['history_complete'],
        grouping_radius_km=5., source_rule='naip_id_suffix_plus_centroid_5km_connected_components',
        lr_scene_identity_audited=False, footprint_separation_audited=False,
        independence_certified=False, split_frozen=False, pixel_payload_read=False,
        quality_columns_decoded=False, directory_columns=parquet.schema_arrow.names,
    )
    for name, value in [('projected-members.json', rows), ('candidate-groups.json', grouping),
                        ('summary.json', result)]:
        (args.output / name).write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
