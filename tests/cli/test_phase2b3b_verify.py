"""Acceptance-authorizing CLI contracts for Phase 2B3-B verification."""

from __future__ import annotations

import importlib
import json
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

from trustsr.jsonio import canonical_json


def _module() -> ModuleType:
    return importlib.import_module("trustsr.cli.phase2b3b_verify")


def _argv(tmp_path: Path) -> list[str]:
    return [
        "--bundle",
        str(tmp_path / "bundle"),
        "--project-root",
        str(tmp_path / "project"),
        "--evidence-dir",
        str(tmp_path / "evidence"),
        "--storage-root",
        str(tmp_path / "storage"),
        "--manifest",
        str(tmp_path / "manifest.jsonl"),
        "--confirm-persistent-storage",
    ]


def _expected() -> dict[str, object]:
    return {
        "schema": "trustsr.phase2b3b-independent-verification-cli.v1",
        "verification_scope": "independent_calibration_acceptance",
        "cache_computation_verified": True,
        "prediction_inference_verified": False,
        "acceptance_authorized": True,
        "publication_sha256": "0" * 64,
        "acceptance_sha256": "1" * 64,
        "result_sha256": "2" * 64,
        "cache_audit_sha256": "3" * 64,
        "phase_decision": "freeze_calibration",
    }


def test_runs_independent_verification_and_emits_one_host_free_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    calls: list[dict[str, object]] = []

    class Receipt:
        def as_dict(self) -> dict[str, object]:
            return _expected()

    def verify(**kwargs: object) -> object:
        print("independent verifier diagnostic")
        calls.append(kwargs)
        return Receipt()

    monkeypatch.setattr(module, "run_independent_phase2b3b_verification", verify)

    assert module.main(_argv(tmp_path)) == 0

    captured = capsys.readouterr()
    assert captured.out.encode() == canonical_json(_expected()) + b"\n"
    assert captured.err == "independent verifier diagnostic\n"
    assert json.loads(captured.out)["acceptance_authorized"] is True
    assert calls == [
        {
            "bundle_dir": tmp_path / "bundle",
            "project_root": tmp_path / "project",
            "evidence_dir": tmp_path / "evidence",
            "storage_root": tmp_path / "storage",
            "manifest_path": tmp_path / "manifest.jsonl",
            "confirmed_persistent_storage": True,
        }
    ]
    for forbidden in (str(tmp_path), "secret", "internal_test"):
        assert forbidden not in captured.out


def test_parser_has_only_fixed_operational_arguments(tmp_path: Path) -> None:
    args = _module().build_parser().parse_args(_argv(tmp_path))

    assert vars(args) == {
        "bundle": tmp_path / "bundle",
        "project_root": tmp_path / "project",
        "evidence_dir": tmp_path / "evidence",
        "storage_root": tmp_path / "storage",
        "manifest": tmp_path / "manifest.jsonl",
        "confirm_persistent_storage": True,
    }


@pytest.mark.parametrize(
    "override",
    (
        ["--alpha", "0.05"],
        ["--coverage", "0.10"],
        ["--minimum-coverage", "0.10"],
        ["--seed", "3407"],
        ["--score", "ldsr_variance_k5"],
        ["--sample", "calibration-1"],
        ["--sample-id", "calibration-1"],
        ["--sample-limit", "1"],
        ["--model-dir", "/tmp/model"],
    ),
)
def test_parser_rejects_scientific_sample_and_model_overrides(
    tmp_path: Path, override: list[str]
) -> None:
    with pytest.raises(SystemExit) as caught:
        _module().build_parser().parse_args([*_argv(tmp_path), *override])
    assert caught.value.code != 0


def test_help_states_independent_acceptance_boundary_and_hides_science_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        _module().main(["--help"])
    assert caught.value.code == 0

    help_text = capsys.readouterr().out.casefold()
    assert "independent" in help_text
    assert "acceptance" in help_text
    for option in (
        "--bundle",
        "--project-root",
        "--evidence-dir",
        "--storage-root",
        "--manifest",
        "--confirm-persistent-storage",
    ):
        assert option in help_text
    for forbidden in (
        "--alpha",
        "--coverage",
        "--seed",
        "--score",
        "--sample",
        "--model",
    ):
        assert forbidden not in help_text


def test_pyproject_registers_independent_verifier_entry_point() -> None:
    project = Path(__file__).resolve().parents[2] / "pyproject.toml"
    configured = tomllib.loads(project.read_text(encoding="utf-8"))

    assert configured["project"]["scripts"]["trustsr-phase2b3b-verify"] == (
        "trustsr.cli.phase2b3b_verify:main"
    )
