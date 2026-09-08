import copy
import importlib

import pytest
import torch

from trustsr.artifacts.predictions import CacheIntegrityError, PredictionCache, build_identity
from trustsr.contracts import SRPair


def api():
    return importlib.import_module('trustsr.evaluation.external_execution')


def fixture(tmp_path):
    pair = SRPair('roi', 'synthetic', torch.zeros(4, 3, 3), torch.ones(4, 12, 12), 4)
    models = {f'ldsr_{s}': {'name': 'synthetic', 'seed': s} for s in range(3407, 3412)}
    models.update(bicubic={'name': 'synthetic-bicubic'}, sen2sr={
        'name': 'synthetic-sen2sr', 'cpu_execution_policy': 'cloud-sen2srlite-96-v1',
        'cpu_intraop_threads': 96})
    return {'subset': [pair]}, PredictionCache(tmp_path / 'cache'), models


class Model:
    def __init__(self, provenance, calls, slot, fail=False, bad=False):
        self.identity, self.calls, self.slot = provenance, calls, slot
        self.fail, self.bad = fail, bad

    def provenance(self):
        return self.identity

    def predict(self, lr):
        self.calls.append(self.slot)
        assert lr.shape == (4, 3, 3)
        if self.fail:
            raise RuntimeError('do not publish this secret exception path')
        return torch.full((4, 12, 12), float('nan') if self.bad else 0.5)


def execute(tmp_path, *, failed=(), bad=(), existing=()):
    subsets, cache, provenances = fixture(tmp_path)
    pair = subsets['subset'][0]
    for slot in existing:
        cache.put(build_identity(provenances[slot], pair.source, pair.sample_id, pair.lr),
                  torch.zeros(4, 12, 12))
    calls, saved, state = [], [], {}

    def factory(slot):
        return Model(provenances[slot], calls, slot, slot in failed, slot in bad)

    api().populate_predictions(subsets, cache, provenances, factory, state,
                               lambda: saved.append(copy.deepcopy(state)), allow_ldsr=True)
    return subsets, cache, provenances, calls, saved, state, factory


def test_only_missing_predictions_are_called_and_started_is_durable(tmp_path):
    _, _, _, calls, saved, state, _ = execute(tmp_path, existing=['ldsr_3407'])
    assert len(calls) == 6 and 'ldsr_3407' not in calls
    assert any(r['status'] == 'started' for snapshot in saved
               for r in snapshot['predictions'].values())
    assert len(state['predictions']) == 7
    assert sum(r['status'] == 'cached' for r in state['predictions'].values()) == 1


def test_runtime_failures_do_not_retry_or_erase_other_predictions(tmp_path):
    subsets, cache, provenance, calls, _, state, factory = execute(tmp_path, failed=['sen2sr'])
    assert len(calls) == 7
    failure = [r for r in state['predictions'].values() if r['status'] == 'failed'][0]
    assert failure['reason'] == 'prediction_runtime_failed'
    assert 'secret' not in str(state)
    api().populate_predictions(subsets, cache, provenance, factory, state, lambda: None,
                               allow_ldsr=True)
    assert len(calls) == 7


def test_invalid_model_output_is_recorded_without_cache(tmp_path):
    _, _, _, calls, _, state, _ = execute(tmp_path, bad=['sen2sr'])
    assert len(calls) == 7
    assert any(r.get('reason') == 'prediction_contract_failed'
               for r in state['predictions'].values())


def test_missing_gpu_authorization_stops_before_model_loading(tmp_path):
    subsets, cache, provenances = fixture(tmp_path)

    def forbidden(slot):
        pytest.fail('model loaded without GPU scope')

    with pytest.raises(RuntimeError, match='GPU'):
        api().populate_predictions(subsets, cache, provenances, forbidden, {}, lambda: None,
                                   allow_ldsr=False)


def test_corruption_anywhere_stops_before_first_model_load(tmp_path):
    subsets, cache, provenances = fixture(tmp_path)
    pair = subsets['subset'][0]
    identity = build_identity(provenances['sen2sr'], pair.source, pair.sample_id, pair.lr)
    cache.put(identity, torch.zeros(4, 12, 12))
    (cache.root / f'{identity.key}.safetensors').write_bytes(b'corrupt')

    def forbidden(slot):
        pytest.fail('model loaded before complete cache integrity probe')

    with pytest.raises(CacheIntegrityError):
        api().populate_predictions(subsets, cache, provenances, forbidden, {}, lambda: None,
                                   allow_ldsr=True)


def test_wrong_model_provenance_is_fatal_before_predict(tmp_path):
    subsets, cache, provenances = fixture(tmp_path)
    calls = []
    with pytest.raises(ValueError, match='provenance'):
        api().populate_predictions(subsets, cache, provenances,
                                   lambda slot: Model({'wrong': True}, calls, slot), {},
                                   lambda: None, allow_ldsr=True)
    assert calls == []


@pytest.mark.parametrize('fault', ['input', 'deleted_cache', 'extra_record'])
def test_resume_refuses_changed_inputs_or_lost_successful_outputs(tmp_path, fault):
    subsets, cache, provenances, _, _, state, factory = execute(tmp_path)
    if fault == 'input':
        subsets['subset'][0].hr.zero_()
    if fault == 'deleted_cache':
        for path in cache.root.glob('*'):
            path.unlink()
    if fault == 'extra_record':
        state['predictions']['unplanned'] = {'status': 'failed', 'reason': 'bad'}
    with pytest.raises((ValueError, CacheIntegrityError)):
        api().populate_predictions(subsets, cache, provenances, factory, state, lambda: None,
                                   allow_ldsr=True)


@pytest.mark.parametrize('cache_present', [False, True])
def test_interrupted_attempt_is_never_automatically_repeated(tmp_path, cache_present):
    subsets, cache, provenances, calls, _, state, factory = execute(tmp_path)
    key = next(iter(state['predictions']))
    state['predictions'][key] = {'status': 'started'}
    if not cache_present:
        for suffix in ('.json', '.safetensors'):
            (cache.root / (key + suffix)).unlink()
    api().populate_predictions(subsets, cache, provenances, factory, state, lambda: None,
                               allow_ldsr=True)
    assert len(calls) == 7
    assert state['predictions'][key]['status'] == ('recovered' if cache_present else 'failed')
