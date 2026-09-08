import pytest

from research.trustmask.grouping import audit_groups


def row(name, lon, lat=0., source=None):
    return {'id': name, 'lon': lon, 'lat': lat, 'source_ids': [source or 'hr:' + name]}


def test_transitive_spatial_chain_excludes_entire_seen_component():
    rows = [row('a', 0), row('b', .03), row('c', .06), row('d', 1)]
    r = audit_groups(rows, {'a'})
    assert r['excluded_members'] == ['a', 'b', 'c']
    assert [x['members'] for x in r['eligible_groups']] == [['d']]


def test_far_shared_source_and_spatial_edge_both_propagate_exclusion():
    rows = [row('a', 0, source='hr:shared'), row('b', 10, source='hr:shared'), row('c', 10.03)]
    r = audit_groups(rows, {'a'})
    assert r['excluded_members'] == ['a', 'b', 'c']
    assert not r['eligible_groups']


def test_permutation_invariant_and_missing_history_is_explicit():
    rows = [row('a', 0), row('b', 10)]
    a = audit_groups(rows, {'missing'})
    assert a == audit_groups(list(reversed(rows)), {'missing'})
    assert a['missing_seen_ids'] == ['missing']
    assert a['history_complete'] is False
    assert a['independence_certified'] is False


def test_dateline_neighbors_are_not_split_by_longitude_sign():
    r = audit_groups([row('a', 179.99), row('b', -179.99)], {'a'})
    assert r['excluded_members'] == ['a', 'b']


@pytest.mark.parametrize('rows,radius', [
    ([row('a', 0), row('a', 10)], 5), ([row('a', float('nan'))], 5),
    ([row('a', 181)], 5), ([row('a', 0, 91)], 5), ([row('a', 0)], True),
    ([dict(row('a', 0), source_ids=[])], 5), ([row('a', 0)], -1),
])
def test_invalid_metadata_is_not_silently_dropped(rows, radius):
    with pytest.raises(ValueError):
        audit_groups(rows, set(), radius_km=radius)


def test_partition_checks_complete_membership_and_all_five_roles():
    from research.trustmask.grouping import check_partition
    audit = audit_groups([row(str(i), i * 10) for i in range(6)], set())
    roles = ['scale_fit', 'development_calibration', 'development_validation',
             'calibration', 'test', 'unused']
    mapping = {g['group_id']: role for g, role in zip(audit['eligible_groups'], roles, strict=True)}
    counts = check_partition(audit, mapping)
    assert counts['test'] == 1 and counts['unused'] == 1
    with pytest.raises(ValueError):
        check_partition(audit, dict(mapping, invented='test'))
    missing = dict(mapping)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError):
        check_partition(audit, missing)
    wrong = dict(mapping)
    wrong[next(iter(wrong))] = 'test'
    with pytest.raises(ValueError):
        check_partition(audit, wrong)


def test_incomplete_history_blocks_any_partition():
    from research.trustmask.grouping import check_partition
    audit = audit_groups([row(str(i), i * 10) for i in range(5)], {'unknown-old'})
    with pytest.raises(ValueError):
        check_partition(audit, {})
