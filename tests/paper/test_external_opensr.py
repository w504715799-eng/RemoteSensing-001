import importlib

import pytest
import torch


def api():
    return importlib.import_module('trustsr.evaluation.external_opensr')


def test_soft_shares_report_actual_valid_support_and_not_hard_counts():
    maps = torch.full((3, 2, 2), 1 / 3)
    maps[:, 0, 0] = float('nan')
    result = api().summarize_correctness_maps(maps)
    assert result['status'] == 'valid'
    assert result['valid_pixels'] == 3
    assert result['grid_pixels'] == 4
    assert result['shares'] == pytest.approx({'im': 1 / 3, 'om': 1 / 3, 'ha': 1 / 3})


@pytest.mark.parametrize('fault,reason', [('missing', 'no_valid_pixels'),
                                         ('support', 'inconsistent_support'),
                                         ('sum', 'invalid_softmin'),
                                         ('infinity', 'invalid_softmin')])
def test_invalid_diagnostics_are_missing_not_zero(fault, reason):
    maps = torch.full((3, 2, 2), 1 / 3)
    if fault == 'missing':
        maps[:] = float('nan')
    if fault == 'support':
        maps[0, 0, 0] = float('nan')
    if fault == 'sum':
        maps[0] = 0.9
    if fault == 'infinity':
        maps[0, 0, 0] = float('inf')
    result = api().summarize_correctness_maps(maps)
    assert result['status'] == 'failed' and result['reason'] == reason
    assert result['shares'] is None


def test_registration_warning_nan_is_not_accepted_as_a_valid_harmonization():
    class Aligner:
        def get_metric(self, x, y):
            return x, torch.tensor(float('nan'))

    with pytest.raises(api().RegistrationFailure):
        api().CheckedAligner(Aligner()).get_metric(torch.zeros(4, 64, 64), None)


def test_real_cpu_worker_leaves_caller_rng_and_backend_unchanged():
    import numpy as np

    lr = np.random.default_rng(19).random((4, 16, 16), dtype=np.float32)
    hr = np.repeat(np.repeat(lr, 4, axis=1), 4, axis=2)
    before = torch.get_rng_state().clone()
    flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    result = api().compute_correctness(lr, hr, hr)
    assert result['status'] == 'valid'
    assert result['grid_pixels'] == 32 * 32
    assert result['valid_pixels'] > 0
    assert sum(result['shares'].values()) == pytest.approx(1)
    assert torch.equal(before, torch.get_rng_state())
    assert (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark) == flags
