import copy
import importlib

import pytest
import torch

from trustsr.artifacts.predictions import CacheIntegrityError, PredictionCache, build_identity
from trustsr.contracts import SRPair
from trustsr.evaluation.external_scores import evaluate_external_roi


def api():
    return importlib.import_module('trustsr.evaluation.external_replay')


def setup_cache(tmp_path, missing=()):
    pair = SRPair('roi', 'synthetic', torch.zeros(4, 3, 3),
                  torch.full((4, 12, 12), 0.25), 4)
    cache = PredictionCache(tmp_path / 'cache')
    provenance = {f'ldsr_{s}': {'name': 'ldsr-s2-x4', 'seed': s} for s in range(3407, 3412)}
    provenance['bicubic'] = {'name': 'bicubic-x4'}
    provenance['sen2sr'] = {'name': 'sen2sr-lite-x4',
                           'cpu_execution_policy': 'cloud-sen2srlite-96-v1',
                           'cpu_intraop_threads': 96}
    predictions = {}
    for index, (slot, model) in enumerate(provenance.items()):
        predictions[slot] = torch.full((4, 12, 12), index / 10)
        if slot not in missing:
            identity = build_identity(model, pair.source, pair.sample_id, pair.lr)
            cache.put(identity, predictions[slot])
    return pair, cache, provenance, predictions


def test_authenticated_complete_replay_matches_existing_array_core(tmp_path):
    pair, cache, provenance, predictions = setup_cache(tmp_path)
    result = api().replay_roi(pair, cache, provenance)
    expected = evaluate_external_roi(pair, torch.stack([predictions[f'ldsr_{s}']
                                     for s in range(3407, 3412)]),
                                     predictions['bicubic'], predictions['sen2sr'])
    for field in ('roi', 'R1', 'R9', 'transfer'):
        assert result[field] == expected[field]
    assert len(result['predictions']) == 7
    assert result['failures'] == {'R1': {}, 'R9': {}}


@pytest.mark.parametrize('missing,valid', [
    (['sen2sr'], {'lr', 'random', 'k5', 'neighborhood'}),
    (['ldsr_3411'], {'lr', 'random', 'three_model'}),
    (['ldsr_3407'], set()),
])
def test_missing_predictions_only_disable_dependent_methods(tmp_path, missing, valid):
    pair, cache, provenance, _ = setup_cache(tmp_path, missing)
    result = api().replay_roi(pair, cache, provenance)
    for window in ('R1', 'R9'):
        assert set(result[window]) == valid
        assert set(result['failures'][window]) == {
            'lr', 'random', 'k5', 'neighborhood', 'three_model'} - valid
    assert (result['transfer'] is None) == ('k5' not in valid)
    summary = api().summarize_replay(['roi'], [result])
    assert summary['primary']['paired_valid'] == int('k5' in valid)
    assert summary['methods']['R9']['three_model']['valid'] == int('three_model' in valid)
    assert summary['methods']['R9']['three_model']['planned'] == 1


def test_corrupt_cache_is_fatal_not_a_missing_method(tmp_path):
    pair, cache, provenance, _ = setup_cache(tmp_path)
    identity = build_identity(provenance['sen2sr'], pair.source, pair.sample_id, pair.lr)
    (cache.root / f'{identity.key}.safetensors').write_bytes(b'corrupt')
    with pytest.raises(CacheIntegrityError):
        api().replay_roi(pair, cache, provenance)


def test_changed_lr_cannot_reuse_other_input_predictions(tmp_path):
    pair, cache, provenance, _ = setup_cache(tmp_path)
    pair.lr.add_(0.1)
    result = api().replay_roi(pair, cache, provenance)
    assert result['R9'] == {}
    assert all(p['status'] == 'missing' for p in result['predictions'].values())


@pytest.mark.parametrize('fault', ['seed_swap', 'old_policy', 'extra_slot', 'different_sampler'])
def test_invalid_prediction_protocol_rejected_before_cache_read(tmp_path, fault, monkeypatch):
    pair, cache, provenance, _ = setup_cache(tmp_path)
    if fault == 'seed_swap':
        provenance['ldsr_3407']['seed'] = 3408
    if fault == 'old_policy':
        del provenance['sen2sr']['cpu_execution_policy']
    if fault == 'extra_slot':
        provenance['unexpected'] = {}
    if fault == 'different_sampler':
        provenance['ldsr_3411']['steps'] = 999

    def forbidden(*args):
        pytest.fail('invalid protocol reached cache')

    monkeypatch.setattr(cache, 'get', forbidden)
    with pytest.raises(ValueError):
        api().replay_roi(pair, cache, provenance)


def test_partial_summary_retains_denominators_and_null_not_zero(tmp_path):
    pair, cache, provenance, _ = setup_cache(tmp_path, ['ldsr_3411'])
    row = api().replay_roi(pair, cache, provenance)
    result = api().summarize_replay(['roi'], [row])
    assert result['primary']['mean_paired_difference'] is None
    assert result['methods']['R9']['k5']['failures'] == {'roi': 'missing:ldsr_3411'}
    assert result['transfer'] == {'planned': 1, 'valid': 0, 'all_rejected': 0}


@pytest.mark.parametrize('fault', ['missing_roi', 'duplicate_roi', 'missing_method',
                                  'result_and_failure', 'nan'])
def test_summary_rejects_silent_loss_and_invalid_diagnostics(tmp_path, fault):
    pair, cache, provenance, _ = setup_cache(tmp_path)
    row = copy.deepcopy(api().replay_roi(pair, cache, provenance))
    rows = [row]
    if fault == 'missing_roi':
        rows = []
    if fault == 'duplicate_roi':
        rows = [row, row]
    if fault == 'missing_method':
        del row['R1']['lr']
    if fault == 'result_and_failure':
        row['failures']['R9']['lr'] = 'failed'
    if fault == 'nan':
        row['R1']['lr']['aurc'] = float('nan')
    with pytest.raises(ValueError):
        api().summarize_replay(['roi'], rows)


def test_subset_checks_protocol_digest_before_cache_access(tmp_path, monkeypatch):
    pair, cache, provenance, _ = setup_cache(tmp_path)

    def forbidden(*args):
        pytest.fail('unbound model configuration reached cache')

    monkeypatch.setattr(cache, 'get', forbidden)
    with pytest.raises(ValueError, match='SHA'):
        api().replay_subset([pair], ['roi'], cache, provenance,
                            expected_provenances_sha256='0' * 64)


def test_subset_replay_is_deterministic_and_retains_missing_center_diagnostics(tmp_path):
    import hashlib

    from trustsr.jsonio import canonical_json

    pair, cache, provenance, _ = setup_cache(tmp_path, ['ldsr_3407'])
    digest = hashlib.sha256(canonical_json(provenance)).hexdigest()
    first = api().replay_subset([pair], ['roi'], cache, provenance,
                                expected_provenances_sha256=digest)
    second = api().replay_subset([pair], ['roi'], cache, provenance,
                                 expected_provenances_sha256=digest)
    assert canonical_json(first) == canonical_json(second)
    assert first['structure']['valid'] == 0 and first['structure']['planned'] == 1
    assert first['rows'][0]['structure']['reason'] == 'missing_center'
    assert first['summary']['primary']['mean_paired_difference'] is None
