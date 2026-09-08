import hashlib
import importlib
import json

import pytest
import torch

from trustsr.contracts import SRPair
from trustsr.jsonio import canonical_json


def api():
    return importlib.import_module('trustsr.evaluation.external_run')


def study():
    lr = torch.rand((4, 16, 16), generator=torch.Generator().manual_seed(19)) * 0.5 + 0.1
    hr = lr.repeat_interleave(4, 1).repeat_interleave(4, 2)
    pair = SRPair('roi', 'synthetic', lr, hr, 4)
    provenances = {f'ldsr_{s}': {'name': 'synthetic', 'seed': s} for s in range(3407, 3412)}
    provenances.update(bicubic={'name': 'synthetic-bicubic'}, sen2sr={
        'name': 'synthetic-sen2sr', 'cpu_execution_policy': 'cloud-sen2srlite-96-v1',
        'cpu_intraop_threads': 96})
    return {'subset': [pair]}, {'science': {'provenances': provenances,
                                           'subsets': {'subset': {'members': [{'roi': 'roi'}]}}}}


def test_run_publishes_verified_whole_study_and_replay_never_constructs_model(tmp_path):
    subsets, protocol = study()
    calls = []

    class Model:
        def __init__(self, slot):
            self.slot = slot

        def provenance(self):
            return protocol['science']['provenances'][self.slot]

        def predict(self, lr):
            calls.append(self.slot)
            if self.slot == 'sen2sr':
                raise RuntimeError('unpublished detailed cause')
            return lr.repeat_interleave(4, 1).repeat_interleave(4, 2)

    output = tmp_path / 'run'
    result = api().run_study(subsets, protocol, output, Model, allow_ldsr=True)
    assert len(calls) == 7
    assert result['subsets']['subset']['summary']['primary']['paired_valid'] == 1
    assert result['prediction_failures']['subset']['roi']['sen2sr'] == 'prediction_runtime_failed'
    assert (output / 'manifest.json').is_file()
    manifest = json.loads((output / 'manifest.json').read_bytes())
    assert manifest['science_sha256'] == hashlib.sha256(canonical_json(result)).hexdigest()

    def forbidden(*args):
        pytest.fail('replay/complete resume tried constructing a model')

    before = {p.name: p.read_bytes() for p in output.glob('*.json')}
    replay = api().run_study(subsets, protocol, output, forbidden,
                             allow_ldsr=False, replay_only=True)
    assert canonical_json(replay) == canonical_json(result)
    assert before == {p.name: p.read_bytes() for p in output.glob('*.json')}
    assert api().run_study(subsets, protocol, output, forbidden,
                           allow_ldsr=False) == result


def test_unrelated_run_directory_is_not_adopted(tmp_path):
    subsets, protocol = study()
    output = tmp_path / 'other'
    output.mkdir()
    sentinel = output / 'old-cache'
    sentinel.write_text('keep')
    with pytest.raises(ValueError):
        api().run_study(subsets, protocol, output, None, allow_ldsr=False)
    assert sentinel.read_text() == 'keep'


def test_atomic_publication_refuses_different_bytes_and_accepts_identical(tmp_path):
    path = tmp_path / 'science.json'
    api().write_once(path, b'{"value":1}')
    api().write_once(path, b'{"value":1}')
    with pytest.raises(ValueError):
        api().write_once(path, b'{"value":2}')
    assert path.read_bytes() == b'{"value":1}'


def test_replay_cannot_create_a_run_or_cache(tmp_path):
    subsets, protocol = study()
    output = tmp_path / 'missing'
    with pytest.raises(ValueError):
        api().run_study(subsets, protocol, output, None, allow_ldsr=False, replay_only=True)
    assert not output.exists()


def test_attempt_state_directory_is_synced_before_model_predict(tmp_path, monkeypatch):
    import os
    import stat

    module = api()
    subsets, protocol = study()
    output = tmp_path / 'run'
    original_fsync = os.fsync
    durable_started = set()

    def fsync(descriptor):
        original_fsync(descriptor)
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and (output / 'state.json').exists():
            state = json.loads((output / 'state.json').read_bytes())
            durable_started.update(key for key, value in state['predictions'].items()
                                   if value['status'] == 'started')

    monkeypatch.setattr(module.os, 'fsync', fsync)

    class Model:
        def __init__(self, slot):
            self.slot = slot

        def provenance(self):
            return protocol['science']['provenances'][self.slot]

        def predict(self, lr):
            from trustsr.artifacts.predictions import build_identity
            pair = subsets['subset'][0]
            key = build_identity(self.provenance(), pair.source, pair.sample_id, lr).key
            assert key in durable_started
            raise RuntimeError('all predictions deliberately fail; no expensive metrics')

    result = module.run_study(subsets, protocol, output, Model, allow_ldsr=True)
    assert len(durable_started) == 7
    assert result['subsets']['subset']['summary']['primary']['paired_valid'] == 0
