import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "evaluate_neighborhood", ROOT / "scripts/paper/evaluate_neighborhood.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def evidence():
    audit = json.loads(
        (ROOT / "artifacts/phase2b3a/sen2naipv2-development-score-cache-audit-v1.json").read_text()
    )
    result = json.loads(
        (ROOT / "artifacts/phase2b3a/sen2naipv2-development-score-audit-v1.json").read_text()
    )
    records = [{"sample_id": s["sample_id"], "split": "development"} for s in result["samples"]]
    return audit, result, records


def test_exact_k5_selection_excludes_other_models():
    audit, result, records = evidence()
    selected = runner.select_inputs(audit, result, records)
    assert len(selected) == 120
    assert sum(len(entries) for _, entries in selected) == 600
    assert all(
        [e["seed"] for e in entries] == [3407, 3408, 3409, 3410, 3411] for _, entries in selected
    )


@pytest.mark.parametrize("fault", ["split", "duplicate", "missing_seed", "different_sample"])
def test_identity_mismatch_fails_before_pixel_loading(fault):
    audit, result, records = evidence()
    if fault == "split":
        records[0]["split"] = "internal_test"
    elif fault == "duplicate":
        records[0] = copy.deepcopy(records[1])
    elif fault == "missing_seed":
        audit["groups"][0]["prediction_entries"].pop()
    else:
        audit["groups"][0]["prediction_entries"][-1]["sample_id"] = "unapproved"
    with pytest.raises(ValueError):
        runner.select_inputs(audit, result, records)


def test_checksum_failure_and_symlink_are_rejected(tmp_path):
    import hashlib

    path = tmp_path / "file"
    path.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    runner.checked_bytes(path, digest, 3)
    with pytest.raises(ValueError):
        runner.checked_bytes(path, digest, 4)
    path.write_bytes(b"abd")
    with pytest.raises(ValueError):
        runner.checked_bytes(path, digest, 3)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        runner.checked_bytes(link, hashlib.sha256(b"abd").hexdigest(), 3)


@pytest.mark.parametrize(
    "values,want", [({3: 0.2, 9: 0.1}, 9), ({3: 0.1, 9: 0.2}, 3), ({3: 0.1 + 5e-13, 9: 0.1}, 3)]
)
def test_selection_uses_lower_r9_mean_with_preregistered_tie(values, want):
    assert runner.choose_window(values) == want
