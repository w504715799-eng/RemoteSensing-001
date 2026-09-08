import hashlib
import pickle

import numpy as np
import pandas as pd
import pytest


def fixture_payload():
    members = [dict(roi=f'ROI_{i}', lr_file=f'lr{i}', hr_file=f'hr{i}',
                    lr_gee_id='scene', crs='EPSG:32630', affine=[2.5, 0, i, 0, -2.5, 0])
               for i in (1, 2)]
    lr = np.zeros((2, 12, 128, 128), dtype=np.int16)
    lr[0] = 2000
    lr[1] = 1000
    return members, {'L2A': lr, 'HRharm': np.zeros((2, 4, 512, 512), dtype=np.int16),
                     'metadata': pd.DataFrame(members[::-1], index=[9, 9]).assign(quality=123.456)}


def decode(tmp_path, raw, members, **kwargs):
    from trustsr.data import spain_package
    path = tmp_path / 'synthetic.pkl'
    path.write_bytes(raw)
    return spain_package.decode_package(
        path, expected_sha256=kwargs.get('digest', hashlib.sha256(raw).hexdigest()),
        expected_size=kwargs.get('size', len(raw)), members=members)


@pytest.mark.parametrize('protocol', [4, 5])
def test_package_authenticates_and_binds_reversed_rows_without_quality(tmp_path, protocol):
    members, payload = fixture_payload()
    pairs, receipt = decode(tmp_path, pickle.dumps(payload, protocol=protocol), members)
    assert [p.sample_id for p in pairs] == ['ROI_1', 'ROI_2']
    assert float(pairs[0].lr[0, 0, 0]) == pytest.approx(0.1)
    assert float(pairs[1].lr[0, 0, 0]) == pytest.approx(0.2)
    assert receipt['positions'] == [1, 0]
    assert pairs[0].source == 'opensr-test/100/' + receipt['package_sha256']
    assert 'quality' not in str(receipt) and '123.456' not in str(receipt)
    assert len(receipt['normalization']) == 2


@pytest.mark.parametrize('fault', ['digest', 'size', 'members', 'shape', 'object', 'trailing'])
def test_invalid_package_stops_before_returning_pairs(tmp_path, fault):
    members, payload = fixture_payload()
    kwargs = {}
    if fault == 'digest':
        kwargs['digest'] = '0' * 64
    if fault == 'size':
        kwargs['size'] = 1
    if fault == 'members':
        members[0] = dict(members[0], hr_file='wrong')
    if fault == 'shape':
        payload['L2A'] = payload['L2A'][:, :4]
    if fault == 'object':
        payload['L2A'] = np.array(['bad'], dtype=object)
    raw = pickle.dumps(payload)
    if fault == 'trailing':
        raw += pickle.dumps(None)
    with pytest.raises(ValueError):
        decode(tmp_path, raw, members, **kwargs)


def test_unknown_reduce_cannot_execute_in_parent_or_child(tmp_path):
    marker = tmp_path / 'executed'

    class Malicious:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).touch()",)

    with pytest.raises(ValueError):
        decode(tmp_path, pickle.dumps(Malicious()), fixture_payload()[0])
    assert not marker.exists()


def test_identity_failure_does_not_start_decoder(tmp_path, monkeypatch):
    from trustsr.data import spain_package

    def forbidden(*args, **kwargs):
        pytest.fail('unverified bytes reached a subprocess')

    monkeypatch.setattr(spain_package.subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='SHA'):
        decode(tmp_path, b'not a pickle', fixture_payload()[0], digest='0' * 64)


def test_package_uses_existing_input_range_rules(tmp_path):
    members, payload = fixture_payload()
    payload['HRharm'][1, 0, 0, 0] = -1
    with pytest.raises(ValueError):
        decode(tmp_path, pickle.dumps(payload), members)


@pytest.mark.parametrize('raw', [b'\x80\x04\x82\x01.', b'Poutside\n.'])
def test_extension_and_persistent_references_are_rejected(tmp_path, raw):
    with pytest.raises(ValueError):
        decode(tmp_path, raw, fixture_payload()[0])
