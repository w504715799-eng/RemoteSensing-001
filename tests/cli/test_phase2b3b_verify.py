"""Candidate-only CLI contracts for Phase 2B3-B metadata verification."""

from __future__ import annotations

import importlib
import json
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

from trustsr.evaluation.phase2b3b_bundle_verify import VerifiedPhase2B3BBundle
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
    ]


def _receipt() -> VerifiedPhase2B3BBundle:
    return VerifiedPhase2B3BBundle(
        schema="trustsr.phase2b3b-candidate-bundle-metadata-verification.v1",
        verification_scope="metadata_consistency_only",
        cache_computation_verified=False,
        manifest_sha256="0" * 64,
        result_sha256="1" * 64,
        cache_audit_sha256="2" * 64,
        runtime_manifest_sha256="3" * 64,
        replay_sha256="4" * 64,
        producer_revision="5" * 40,
        ordered_sample_ids_sha256="6" * 64,
        ordered_membership_sha256="7" * 64,
        input_receipt_sha256="8" * 64,
        ordered_inputs_sha256="9" * 64,
        map_evidence_sha256="a" * 64,
        radiometry_aggregate_sha256="b" * 64,
        phase_decision="freeze_calibration",
    )


def _expected() -> dict[str, object]:
    receipt = _receipt()
    return {
        "schema": "trustsr.phase2b3b-candidate-metadata-verification-cli.v1",
        "verification_scope": "metadata_consistency_only",
        "cache_computation_verified": False,
        "acceptance_authorized": False,
        "manifest_sha256": receipt.manifest_sha256,
        "result_sha256": receipt.result_sha256,
        "cache_audit_sha256": receipt.cache_audit_sha256,
        "runtime_manifest_sha256": receipt.runtime_manifest_sha256,
        "replay_sha256": receipt.replay_sha256,
        "producer_revision": receipt.producer_revision,
        "ordered_sample_ids_sha256": receipt.ordered_sample_ids_sha256,
        "ordered_membership_sha256": receipt.ordered_membership_sha256,
        "input_receipt_sha256": receipt.input_receipt_sha256,
        "ordered_inputs_sha256": receipt.ordered_inputs_sha256,
        "map_evidence_sha256": receipt.map_evidence_sha256,
        "radiometry_aggregate_sha256": receipt.radiometry_aggregate_sha256,
        "phase_decision": receipt.phase_decision,
    }


def test_calls_bundle_verifier_and_emits_one_canonical_candidate_only_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    calls: list[tuple[Path, dict[str, Path]]] = []

    def verify(bundle: Path, **paths: Path) -> VerifiedPhase2B3BBundle:
        print("verifier diagnostic")
        calls.append((bundle, paths))
        return _receipt()

    monkeypatch.setattr(module, "verify_phase2b3b_bundle", verify)

    assert module.main(_argv(tmp_path)) == 0

    captured = capsys.readouterr()
    assert captured.out.encode() == canonical_json(_expected()) + b"\n"
    assert captured.out.count("\n") == 1
    assert captured.err == "verifier diagnostic\n"
    assert calls == [
        (
            tmp_path / "bundle",
            {
                "project_root": tmp_path / "project",
                "evidence_dir": tmp_path / "evidence",
                "storage_root": tmp_path / "storage",
                "manifest_path": tmp_path / "manifest.jsonl",
            },
        )
    ]
    parsed = json.loads(captured.out)
    assert type(parsed) is dict
    assert parsed["acceptance_authorized"] is False
    assert parsed["cache_computation_verified"] is False
    for forbidden in (str(tmp_path), "secret", "internal_test"):
        assert forbidden not in captured.out


def test_verifier_failure_emits_no_success_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()

    def reject(*args: object, **kwargs: object) -> object:
        print("candidate rejected")
        raise ValueError("metadata mismatch")

    monkeypatch.setattr(module, "verify_phase2b3b_bundle", reject)

    with pytest.raises(ValueError, match="metadata mismatch"):
        module.main(_argv(tmp_path))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "candidate rejected\n"


def test_parser_has_only_five_required_path_arguments(tmp_path: Path) -> None:
    module = _module()

    args = module.build_parser().parse_args(_argv(tmp_path))

    assert vars(args) == {
        "bundle": tmp_path / "bundle",
        "project_root": tmp_path / "project",
        "evidence_dir": tmp_path / "evidence",
        "storage_root": tmp_path / "storage",
        "manifest": tmp_path / "manifest.jsonl",
    }
    argv = _argv(tmp_path)
    for option in (
        "--bundle",
        "--project-root",
        "--evidence-dir",
        "--storage-root",
        "--manifest",
    ):
        index = argv.index(option)
        with pytest.raises(SystemExit) as caught:
            module.build_parser().parse_args(argv[:index] + argv[index + 2 :])
        assert caught.value.code != 0


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
    ),
)
def test_parser_rejects_scientific_and_sample_overrides(
    tmp_path: Path, override: list[str]
) -> None:
    with pytest.raises(SystemExit) as caught:
        _module().build_parser().parse_args([*_argv(tmp_path), *override])
    assert caught.value.code != 0


def test_help_states_metadata_only_boundary_and_hides_science_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        _module().main(["--help"])
    assert caught.value.code == 0

    help_text = capsys.readouterr().out
    assert "candidate metadata only" in help_text.casefold()
    assert "cannot authorize acceptance" in help_text.casefold()
    for option in ("--bundle", "--project-root", "--evidence-dir", "--storage-root", "--manifest"):
        assert option in help_text
    for forbidden in ("--alpha", "--coverage", "--seed", "--score", "--sample"):
        assert forbidden not in help_text


def test_pyproject_registers_candidate_verifier_entry_point() -> None:
    project = Path(__file__).resolve().parents[2] / "pyproject.toml"
    configured = tomllib.loads(project.read_text(encoding="utf-8"))

    assert configured["project"]["scripts"]["trustsr-phase2b3b-verify"] == (
        "trustsr.cli.phase2b3b_verify:main"
    )
