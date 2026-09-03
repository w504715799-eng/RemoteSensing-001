"""Narrow metadata-only tests for the Phase 2B3-B preflight CLI."""

from __future__ import annotations

import importlib
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, ModuleType

import pytest

from trustsr.evaluation.phase2b3b_revision import Phase2B3BRevision
from trustsr.jsonio import canonical_json


def _module() -> ModuleType:
    return importlib.import_module("trustsr.cli.phase2b3b")


def _argv(tmp_path: Path) -> list[str]:
    return [
        "preflight",
        "--project-root",
        str(tmp_path / "project"),
        "--evidence-dir",
        str(tmp_path / "evidence"),
        "--storage-root",
        str(tmp_path / "storage"),
        "--manifest",
        str(tmp_path / "manifest.jsonl"),
    ]


def _revision() -> Phase2B3BRevision:
    return Phase2B3BRevision(
        branch="main",
        head_revision="c" * 40,
        calculation_revision="5" * 40,
        evidence_publication="b" * 40,
    )


def _preflight() -> Mapping[str, object]:
    return MappingProxyType(
        {
            "schema": "trustsr.phase2b3b-preflight.v1",
            "calibration": MappingProxyType(
                {
                    "sample_count": 120,
                    "strata": (
                        MappingProxyType(
                            {"days_between": -1, "correlation_bin": 0, "sample_count": 10}
                        ),
                    ),
                }
            ),
            "score": MappingProxyType(
                {"name": "ldsr_variance_k5", "seeds": (3407, 3408, 3409, 3410, 3411)}
            ),
        }
    )


def test_preflight_calls_revision_before_metadata_and_emits_one_canonical_host_free_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    events: list[tuple[object, ...]] = []

    def verify(project_root: Path) -> Phase2B3BRevision:
        print("revision diagnostic")
        events.append(("revision", project_root))
        return _revision()

    def load(evidence_dir: Path, storage_root: Path, manifest: Path) -> Mapping[str, object]:
        print("metadata diagnostic")
        events.append(("preflight", evidence_dir, storage_root, manifest))
        return _preflight()

    monkeypatch.setattr(module, "verify_phase2b3b_revision", verify)
    monkeypatch.setattr(module, "load_phase2b3b_preflight", load)

    assert module.main(_argv(tmp_path)) == 0

    captured = capsys.readouterr()
    expected = {
        "schema": "trustsr.phase2b3b-cli-preflight.v1",
        "revision": {
            "branch": "main",
            "head_revision": "c" * 40,
            "calculation_revision": "5" * 40,
            "evidence_publication": "b" * 40,
        },
        "preflight": {
            "schema": "trustsr.phase2b3b-preflight.v1",
            "calibration": {
                "sample_count": 120,
                "strata": [
                    {"days_between": -1, "correlation_bin": 0, "sample_count": 10}
                ],
            },
            "score": {
                "name": "ldsr_variance_k5",
                "seeds": [3407, 3408, 3409, 3410, 3411],
            },
        },
    }
    assert captured.out.encode("utf-8") == canonical_json(expected) + b"\n"
    assert captured.out.count("\n") == 1
    assert "diagnostic" not in captured.out
    assert "revision diagnostic\nmetadata diagnostic\n" == captured.err
    assert str(tmp_path) not in captured.out
    assert events == [
        ("revision", tmp_path / "project"),
        (
            "preflight",
            tmp_path / "evidence",
            tmp_path / "storage",
            tmp_path / "manifest.jsonl",
        ),
    ]
    parsed = json.loads(captured.out)
    assert type(parsed) is dict
    assert type(parsed["preflight"]["calibration"]["strata"]) is list


@pytest.mark.parametrize("failing_boundary", ("revision", "preflight"))
def test_boundary_error_fails_closed_without_scientific_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failing_boundary: str,
) -> None:
    module = _module()
    events: list[str] = []

    def verify(project_root: Path) -> Phase2B3BRevision:
        events.append("revision")
        if failing_boundary == "revision":
            raise ValueError("revision rejected")
        return _revision()

    def load(evidence_dir: Path, storage_root: Path, manifest: Path) -> Mapping[str, object]:
        events.append("preflight")
        raise ValueError("preflight rejected")

    monkeypatch.setattr(module, "verify_phase2b3b_revision", verify)
    monkeypatch.setattr(module, "load_phase2b3b_preflight", load)

    with pytest.raises(ValueError, match=failing_boundary):
        module.main(_argv(tmp_path))

    assert capsys.readouterr().out == ""
    assert events == (["revision"] if failing_boundary == "revision" else ["revision", "preflight"])


def test_parser_has_only_required_preflight_paths(tmp_path: Path) -> None:
    module = _module()

    args = module.build_parser().parse_args(_argv(tmp_path))

    assert args.stage == "preflight"
    assert args.project_root == tmp_path / "project"
    assert args.evidence_dir == tmp_path / "evidence"
    assert args.storage_root == tmp_path / "storage"
    assert args.manifest == tmp_path / "manifest.jsonl"
    assert set(vars(args)) == {
        "stage",
        "project_root",
        "evidence_dir",
        "storage_root",
        "manifest",
        "handler",
    }


@pytest.mark.parametrize(
    "arguments",
    (
        ["calibration"],
        ["replay"],
        ["--alpha", "0.05"],
        ["--coverage", "0.10"],
        ["--score", "ldsr_variance_k5"],
        ["--seed", "3407"],
        ["--sample", "calibration-1"],
    ),
)
def test_parser_rejects_other_stages_and_scientific_parameters(
    tmp_path: Path, arguments: list[str]
) -> None:
    module = _module()
    argv = (
        arguments
        if arguments[0] in {"calibration", "replay"}
        else [*_argv(tmp_path), *arguments]
    )

    with pytest.raises(SystemExit) as caught:
        module.build_parser().parse_args(argv)
    assert caught.value.code != 0


def test_preflight_requires_all_four_explicit_paths(tmp_path: Path) -> None:
    module = _module()
    argv = _argv(tmp_path)

    for option in ("--project-root", "--evidence-dir", "--storage-root", "--manifest"):
        index = argv.index(option)
        incomplete = argv[:index] + argv[index + 2 :]
        with pytest.raises(SystemExit) as caught:
            module.build_parser().parse_args(incomplete)
        assert caught.value.code != 0


def test_cli_help_exposes_only_preflight_and_path_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()

    with pytest.raises(SystemExit) as top_help:
        module.main(["--help"])
    assert top_help.value.code == 0
    top = capsys.readouterr().out
    assert "preflight" in top
    assert "calibration" not in top
    assert "replay" not in top

    with pytest.raises(SystemExit) as child_help:
        module.main(["preflight", "--help"])
    assert child_help.value.code == 0
    child = capsys.readouterr().out
    for option in ("--project-root", "--evidence-dir", "--storage-root", "--manifest"):
        assert option in child
    for forbidden in ("--alpha", "--coverage", "--score", "--seed", "--sample"):
        assert forbidden not in child


def test_pyproject_registers_only_the_phase2b3b_preflight_entrypoint() -> None:
    project = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())

    scripts = project["project"]["scripts"]
    assert scripts["trustsr-phase2b3b"] == "trustsr.cli.phase2b3b:main"
