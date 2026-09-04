"""Independent Phase 2B3-C verifier command surface."""

from __future__ import annotations

import importlib
import json
import tomllib
from pathlib import Path

import pytest

from trustsr.jsonio import canonical_json


def _module():
    return importlib.import_module("trustsr.cli.phase2b3c_verify")


def _argv(tmp_path: Path) -> list[str]:
    return [
        "--bundle",
        str(tmp_path / "copied-bundle"),
        "--project-root",
        str(tmp_path / "project"),
        "--evidence-dir",
        str(tmp_path / "evidence"),
        "--storage-root",
        str(tmp_path / "storage"),
        "--manifest",
        str(tmp_path / "manifest.jsonl"),
        "--access-permit",
        str(tmp_path / "permit.json"),
        "--confirm-persistent-storage",
    ]


def _expected() -> dict[str, object]:
    return {
        "schema": "trustsr.phase2b3c-independent-verification-cli.v1",
        "verification_scope": "independent_internal_test_acceptance",
        "cache_computation_verified": True,
        "prediction_inference_verified": False,
        "acceptance_authorized": True,
        "ledger_state": "accepted",
        "publication_sha256": "0" * 64,
        "acceptance_sha256": "1" * 64,
        "result_sha256": "2" * 64,
        "cache_audit_sha256": "3" * 64,
        "phase_decision": "confirmed",
    }


def test_runs_verifier_and_emits_one_canonical_host_free_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    calls: list[dict[str, object]] = []

    class Receipt:
        def as_dict(self) -> dict[str, object]:
            return _expected()

    monkeypatch.setattr(
        module,
        "run_independent_phase2b3c_verification",
        lambda **kwargs: calls.append(kwargs) or Receipt(),
    )

    assert module.main(_argv(tmp_path)) == 0

    captured = capsys.readouterr()
    assert captured.out.encode() == canonical_json(_expected()) + b"\n"
    assert json.loads(captured.out)["ledger_state"] == "accepted"
    assert calls == [
        {
            "bundle_dir": tmp_path / "copied-bundle",
            "project_root": tmp_path / "project",
            "evidence_dir": tmp_path / "evidence",
            "storage_root": tmp_path / "storage",
            "manifest_path": tmp_path / "manifest.jsonl",
            "access_permit_path": tmp_path / "permit.json",
            "confirmed_persistent_storage": True,
        }
    ]


@pytest.mark.parametrize(
    "override",
    (
        ["--model-dir", "/tmp/model"],
        ["--ldsr-model-dir", "/tmp/model"],
        ["--alpha", "0.05"],
        ["--minimum-coverage", "0.10"],
        ["--threshold", "1"],
        ["--seed", "3407"],
        ["--sample-id", "x"],
        ["--device", "cuda"],
    ),
)
def test_parser_rejects_model_scientific_and_sample_overrides(
    tmp_path: Path, override: list[str]
) -> None:
    with pytest.raises(SystemExit):
        _module().build_parser().parse_args([*_argv(tmp_path), *override])


def test_pyproject_registers_phase2b3c_verifier() -> None:
    project = Path(__file__).parents[2]
    configured = tomllib.loads((project / "pyproject.toml").read_text())

    assert configured["project"]["scripts"]["trustsr-phase2b3c-verify"] == (
        "trustsr.cli.phase2b3c_verify:main"
    )
