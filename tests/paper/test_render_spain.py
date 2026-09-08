import importlib.util
from pathlib import Path

import pytest


def api():
    path = Path(__file__).resolve().parents[2] / 'paper/tools/render_spain.py'
    spec = importlib.util.spec_from_file_location('render_spain', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tables_keep_partial_method_denominators_and_equal_roi_weighting():
    rows = []
    for roi, risk in [('a', 0.2), ('b', 0.4)]:
        metric = {'aurc': risk, 'rho': None, 'coverages': [0.1, 1.0],
                  'selective_mean_risks': [risk, risk]}
        methods = {'lr': metric, 'random': metric}
        if roi == 'a':
            methods['k5'] = dict(metric, aurc=0.1)
        rows.append({'roi': roi, 'R1': methods, 'R9': methods,
                     'transfer': {'coverage': 0.0, 'roi_max_r9': 0.0, 'all_rejected': True}
                     if roi == 'a' else None,
                     'structure': {'status': 'failed', 'reason': 'registration_failed',
                                   'shares': None}})
    tables = api().tables({'subsets': {'crops': {'rows': rows}}})
    lr = next(r for r in tables['scores'] if r['method'] == 'lr' and r['window'] == 'R9')
    k5 = next(r for r in tables['scores'] if r['method'] == 'k5' and r['window'] == 'R9')
    assert lr['mean_aurc'] == pytest.approx(0.3)
    assert (k5['planned'], k5['valid'], k5['missing']) == (2, 1, 1)
    assert tables['paired'][0]['difference'] == pytest.approx(-0.1)
    assert tables['paired'][1]['difference'] is None
    assert tables['transfer'][0]['all_rejected'] is True
    assert tables['transfer'][1]['coverage'] is None
    assert tables['structure'][0]['ha'] is None
    assert tables['structure'][0]['valid_pixels'] is None


def test_wrong_science_digest_cannot_create_outputs(tmp_path):
    source = tmp_path / 'science.json'
    source.write_text('{}')
    with pytest.raises(ValueError, match='SHA'):
        api().render(source, '0' * 64, tmp_path / 'missing-protocol', '0' * 64,
                     tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_all_failed_subset_still_exports_empty_curves_and_failure_rows(tmp_path):
    import hashlib
    import json

    protocol = {'status': 'frozen', 'science': {'subsets': {'crops': {'members': [{'roi': 'a'}]}}}}
    protocol_raw = json.dumps(protocol).encode()
    protocol_sha = hashlib.sha256(protocol_raw).hexdigest()
    science = {'schema': 'trustsr.spain-study-science.v1', 'protocol_sha256': protocol_sha,
               'subsets': {'crops': {'rows': [{'roi': 'a', 'R1': {}, 'R9': {}, 'transfer': None,
                   'structure': {'status': 'failed', 'reason': 'missing_center',
                                 'shares': None}}]}}}
    raw = json.dumps(science).encode()
    source, spec = tmp_path / 'science.json', tmp_path / 'protocol.json'
    source.write_bytes(raw)
    spec.write_bytes(protocol_raw)
    api().render(source, hashlib.sha256(raw).hexdigest(), spec, protocol_sha, tmp_path / 'out')
    assert len((tmp_path / 'out/spain_curves.csv').read_text().splitlines()) == 1
    assert 'missing_center' in (tmp_path / 'out/spain_structure.csv').read_text()
    conflict = tmp_path / 'out/spain_scores.csv'
    conflict.write_text('prior publication')
    with pytest.raises(ValueError):
        api().render(source, hashlib.sha256(raw).hexdigest(), spec, protocol_sha, tmp_path / 'out')
    assert conflict.read_text() == 'prior publication'
