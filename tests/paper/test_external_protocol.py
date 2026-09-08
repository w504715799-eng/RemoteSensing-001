import copy
import hashlib
import importlib
from pathlib import Path

import pytest

from trustsr.jsonio import canonical_json

ROOT = Path(__file__).resolve().parents[2]


def api():
    return importlib.import_module('trustsr.evaluation.external_protocol')


def test_draft_binds_both_full_subsets_and_verified_cloud_provenance():
    draft = api().build_draft(ROOT)
    assert draft['status'] == 'draft'
    assert [len(draft['science']['subsets'][s]['members']) for s in
            ('spain_crops', 'spain_urban')] == [28, 20]
    models = draft['science']['provenances']
    assert models['sen2sr']['cpu_intraop_threads'] == 96
    assert models['ldsr_3411']['seed'] == 3411
    assert draft['budget'] is None
    assert draft['science']['statistics'] == 'descriptive_equal_roi_no_bootstrap'


def frozen_candidate():
    result = api().build_draft(ROOT)
    result['status'] = 'frozen'
    result['budget'] = {'maximum_wall_seconds': 3600, 'estimated_wall_seconds': 1200.0,
                        'hourly_price': 1.0, 'currency': 'CNY',
                        'evidence_sha256': 'a' * 64}
    return result


def test_protocol_sha_and_draft_gates_reject_before_any_external_access():
    draft = api().build_draft(ROOT)
    raw = canonical_json(draft)
    with pytest.raises(ValueError, match='frozen'):
        api().load_frozen(raw, hashlib.sha256(raw).hexdigest(), ROOT)
    with pytest.raises(ValueError, match='SHA'):
        api().load_frozen(raw, '0' * 64, ROOT)


@pytest.mark.parametrize('fault', ['threshold', 'members', 'model', 'budget', 'runtime', 'code'])
def test_rehashed_unreviewed_science_or_incomplete_gate_is_rejected(fault):
    result = copy.deepcopy(frozen_candidate())
    if fault == 'threshold':
        result['science']['transfer_threshold'] = 0.5
    if fault == 'members':
        result['science']['subsets']['spain_crops']['members'].pop()
    if fault == 'model':
        result['science']['provenances']['ldsr_3407']['sampling_steps'] = 1
    if fault == 'budget':
        result['budget']['estimated_wall_seconds'] = float('inf')
    if fault == 'runtime':
        result['runtime_versions'].pop('numpy')
    if fault == 'code':
        key = next(iter(result['implementation']))
        result['implementation'][key] = '0' * 64
    import json
    raw = json.dumps(result, sort_keys=True, separators=(',', ':')).encode()
    with pytest.raises(ValueError):
        api().load_frozen(raw, hashlib.sha256(raw).hexdigest(), ROOT)


def test_complete_frozen_contract_is_readable_without_runtime_or_pixel_access():
    candidate = frozen_candidate()
    raw = canonical_json(candidate)
    assert api().load_frozen(raw, hashlib.sha256(raw).hexdigest(), ROOT) == candidate


def test_frozen_protocol_bytes_must_be_canonical_to_keep_one_digest_identity():
    candidate = frozen_candidate()
    raw = canonical_json(candidate) + b'\n'
    with pytest.raises(ValueError, match='canonical'):
        api().load_frozen(raw, hashlib.sha256(raw).hexdigest(), ROOT)


def test_draft_runtime_inventory_comes_from_completed_cloud_measurement():
    draft = api().build_draft(ROOT)
    assert draft['runtime_versions']['pandas'] == '2.3.3'
    assert draft['runtime_versions']['satalign'] == '0.1.17'
    assert all(isinstance(value, str) for value in draft['runtime_versions'].values())
    assert draft['budget'] is None and draft['status'] == 'draft'


def test_rehashed_runtime_change_cannot_reuse_a_cloud_measurement():
    candidate = api().build_draft(ROOT)
    candidate['status'] = 'frozen'
    candidate['runtime_versions'] = {name: 'test-version' for name in api().RUNTIME_PACKAGES}
    candidate['runtime_versions'].update(api().KNOWN_VERSIONS)
    candidate['budget'] = {'maximum_wall_seconds': 3600, 'estimated_wall_seconds': 1200.0,
                           'hourly_price': 1.0, 'currency': 'CNY', 'evidence_sha256': 'a' * 64}
    raw = canonical_json(candidate)
    with pytest.raises(ValueError, match='runtime'):
        api().load_frozen(raw, hashlib.sha256(raw).hexdigest(), ROOT)
