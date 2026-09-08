"""Metadata-only source/spatial connected groups; grouping does not prove independence."""

import hashlib
import math
from numbers import Real

import numpy as np
from scipy.spatial import cKDTree


def _real(value, bound):
    if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            or abs(value) > bound):
        raise ValueError('invalid geographic coordinate or radius')
    return float(value)


def audit_groups(rows, seen_ids, *, radius_km=5.0):
    radius = _real(radius_km, 10_000)
    if radius < 0:
        raise ValueError('radius must be nonnegative')
    normalized = []
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {'id', 'lon', 'lat', 'source_ids'}
                or type(row['id']) is not str or not row['id'].strip()
                or not isinstance(row['source_ids'], (list, tuple)) or not row['source_ids']
                or any(type(s) is not str or not s.strip() for s in row['source_ids'])):
            raise ValueError('id,coordinates and nonempty qualified sources required')
        normalized.append(dict(id=row['id'], lon=_real(row['lon'], 180),
                               lat=_real(row['lat'], 90), source_ids=tuple(row['source_ids'])))
    normalized.sort(key=lambda r: r['id'])
    ids = [r['id'] for r in normalized]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate member identity')
    if not isinstance(seen_ids, (set, frozenset, list, tuple)) or any(
            type(x) is not str or not x.strip() for x in seen_ids):
        raise ValueError('explicit historical identities required')
    seen = set(seen_ids)
    parents = list(range(len(ids)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        parents[find(b)] = find(a)

    sources = {}
    for i, row in enumerate(normalized):
        for source in row['source_ids']:
            if source in sources:
                union(i, sources[source])
            else:
                sources[source] = i
    if normalized:
        lat = np.radians([r['lat'] for r in normalized])
        lon = np.radians([r['lon'] for r in normalized])
        vectors = np.column_stack((np.cos(lat) * np.cos(lon),
                                   np.cos(lat) * np.sin(lon), np.sin(lat)))
        chord = 2 * math.sin(radius / (2 * 6371.0088))
        for a, b in cKDTree(vectors).query_pairs(chord + 64 * np.finfo(float).eps):
            h = (math.sin((lat[a] - lat[b]) / 2) ** 2
                 + math.cos(lat[a]) * math.cos(lat[b]) * math.sin((lon[a] - lon[b]) / 2) ** 2)
            distance = 2 * 6371.0088 * math.asin(math.sqrt(min(1., max(0., h))))
            if distance <= radius:
                union(a, b)
    components = {}
    for i, name in enumerate(ids):
        components.setdefault(find(i), []).append(name)
    eligible, excluded = [], []
    for members in components.values():
        if seen.intersection(members):
            excluded.extend(members)
        else:
            identity = hashlib.sha256('\n'.join(members).encode()).hexdigest()
            eligible.append({'group_id': identity, 'members': members})
    return dict(
        member_count=len(ids), component_count=len(components),
        eligible_groups=sorted(eligible, key=lambda r: r['group_id']),
        excluded_members=sorted(excluded), missing_seen_ids=sorted(seen.difference(ids)),
        history_complete=seen.issubset(ids), independence_certified=False, radius_km=radius,
    )


def check_partition(audit, assignments):
    """Check structural separation of an audit's groups, not authenticate or freeze a split.

    The future protocol must bind this audit to authenticated source metadata. Each
    group gets one role; unused groups are explicit and no stage may share a member.
    """
    roles = ('scale_fit', 'development_calibration', 'development_validation',
             'calibration', 'test', 'unused')
    if audit.get('history_complete') is not True or audit.get('missing_seen_ids') != []:
        raise ValueError('complete historical overlap audit required before partitioning')
    counts = dict.fromkeys(roles, 0)
    ids, members = set(), set()
    excluded = set(audit['excluded_members'])
    for group in audit['eligible_groups']:
        group_id, names = group['group_id'], group['members']
        if (not names or names != sorted(set(names)) or group_id in ids
                or members.intersection(names) or excluded.intersection(names)
                or hashlib.sha256('\n'.join(names).encode()).hexdigest() != group_id):
            raise ValueError('invalid, overlapping or excluded partition members')
        ids.add(group_id)
        members.update(names)
    if not isinstance(assignments, dict) or set(assignments) != ids:
        raise ValueError('assign every eligible group exactly once, including unused')
    for role in assignments.values():
        if type(role) is not str or role not in roles:
            raise ValueError('unsupported partition role')
        counts[role] += 1
    if any(counts[role] == 0 for role in roles[:-1]):
        raise ValueError('all five independent fitting/selection/calibration/test roles required')
    return counts
