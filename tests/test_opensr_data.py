from pathlib import Path

import numpy as np
import pytest

from trustsr.data.opensr import L2A_RGBN_INDICES, load_opensr_pairs


def test_loader_selects_b04_b03_b02_b08(monkeypatch, tmp_path: Path) -> None:
    l2a = np.zeros((2, 12, 3, 5), dtype=np.uint16)
    for band in range(12):
        l2a[:, band] = (band + 1) * 100
    hr = np.full((2, 4, 12, 20), 500, dtype=np.uint16)

    monkeypatch.setattr(
        "opensr_test.load",
        lambda dataset, model_dir, version: {"L2A": l2a, "HRharm": hr},
    )

    pairs = load_opensr_pairs("spot", tmp_path, "v3", limit=1)

    assert L2A_RGBN_INDICES == (3, 2, 1, 7)
    assert len(pairs) == 1
    assert pairs[0].sample_id == "spot-0000"
    assert pairs[0].lr[:, 0, 0].tolist() == pytest.approx([0.04, 0.03, 0.02, 0.08])
    assert pairs[0].hr.shape == (4, 12, 20)
