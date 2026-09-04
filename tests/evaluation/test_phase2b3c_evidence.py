"""Frozen Phase 2B3-B publication boundary consumed by Phase 2B3-C."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
from pathlib import Path

import pytest

from trustsr.jsonio import canonical_json


def _module():
    return importlib.import_module("trustsr.evaluation.phase2b3c_evidence")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _evidence_dir() -> Path:
    return _project_root() / "artifacts" / "phase2b3b"


def test_loads_only_the_exact_accepted_phase2b3b_publication() -> None:
    module = _module()

    evidence = module.load_frozen_phase2b3b_evidence(_evidence_dir(), _project_root())

    assert evidence.result_sha256 == (
        "5fe01daaaa27443d5289cc366eacbe1fb628492c25bab16fadd164d3a8eb3174"
    )
    assert evidence.cache_audit_sha256 == (
        "40aff8aaccb27719d958d42026d68d8efc0c7f79479e3bd99777053ee1d6399f"
    )
    assert evidence.acceptance_sha256 == (
        "ce7f1a91e9954235e289dfc6eececb67261ff57110caaf15fe6640a0fb3b69ab"
    )
    assert evidence.threshold == 7.970395366024563e-06
    assert evidence.alpha == 0.05
    assert evidence.minimum_coverage == 0.10
    assert evidence.seeds == (3407, 3408, 3409, 3410, 3411)
    assert evidence.post_manifest_sha256 == (
        "c7f8ffa8415575d85daafe284a0796ec3f111442f0ac662f1d01311c4a851d4a"
    )
    assert evidence.input_audit_sha256 == (
        "fceb2ec04680ddf46bf4d0ed5a4a93edd33d58a09fc176d936bdef783114b44b"
    )
    assert evidence.producer_revision == "2d17c141174c1062f8a056b491326487202a6deb"
    first = evidence.as_dict()
    first["target"]["alpha"] = 1.0
    assert evidence.as_dict()["target"]["alpha"] == 0.05


@pytest.mark.parametrize(
    "filename", tuple(sorted(path.name for path in _evidence_dir().glob("*.json")))
)
def test_rejects_any_changed_publication_byte(tmp_path: Path, filename: str) -> None:
    module = _module()
    candidate = tmp_path / "evidence"
    shutil.copytree(_evidence_dir(), candidate)
    path = candidate / filename
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="digest"):
        module.load_frozen_phase2b3b_evidence(candidate, _project_root())


def test_rejects_semantically_wrong_frozen_threshold_even_with_test_pinned_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    candidate = tmp_path / "evidence"
    shutil.copytree(_evidence_dir(), candidate)
    name = "sen2naipv2-calibration-conformal-acceptance-v1.json"
    path = candidate / name
    document = json.loads(path.read_text(encoding="utf-8"))
    document["frozen_calibration"]["threshold"] = 0.1
    payload = canonical_json(document)
    path.write_bytes(payload)
    digests = dict(module.PHASE2B3B_FILE_SHA256S)
    digests[name] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(module, "PHASE2B3B_FILE_SHA256S", digests)

    with pytest.raises(ValueError, match="threshold"):
        module.load_frozen_phase2b3b_evidence(candidate, _project_root())


def test_rejects_extra_missing_and_symlinked_evidence(tmp_path: Path) -> None:
    module = _module()
    extra = tmp_path / "extra"
    shutil.copytree(_evidence_dir(), extra)
    (extra / "extra.json").write_text("{}", encoding="utf-8")
    missing = tmp_path / "missing"
    shutil.copytree(_evidence_dir(), missing)
    next(missing.iterdir()).unlink()
    linked = tmp_path / "linked"
    linked.symlink_to(_evidence_dir(), target_is_directory=True)

    for candidate in (extra, missing, linked):
        with pytest.raises(ValueError, match="evidence"):
            module.load_frozen_phase2b3b_evidence(candidate, _project_root())
