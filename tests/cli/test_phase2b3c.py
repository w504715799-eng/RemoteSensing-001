"""Narrow command surface for the one-time Phase 2B3-C workflow."""

from __future__ import annotations

import importlib
import json
import tomllib
from pathlib import Path

import pytest

from trustsr.jsonio import canonical_json


def _module():
    return importlib.import_module("trustsr.cli.phase2b3c")


def _common(tmp_path: Path, stage: str) -> list[str]:
    values = [
        stage,
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
    if stage != "preflight":
        values.extend(["--access-permit", str(tmp_path / "permit.json")])
    return values


def test_parser_exposes_exact_three_stages_and_model_only_on_evaluate(
    tmp_path: Path,
) -> None:
    module = _module()
    parser = module.build_parser()

    evaluate = parser.parse_args(
        [*_common(tmp_path, "evaluate"), "--ldsr-model-dir", str(tmp_path / "model")]
    )
    assert evaluate.stage == "evaluate"
    assert evaluate.ldsr_model_dir == tmp_path / "model"
    assert parser.parse_args(_common(tmp_path, "preflight")).stage == "preflight"
    assert (
        parser.parse_args(_common(tmp_path, "evaluation-replay")).stage
        == "evaluation-replay"
    )
    with pytest.raises(SystemExit):
        parser.parse_args([*_common(tmp_path, "evaluation-replay"), "--ldsr-model-dir", "x"])
    with pytest.raises(SystemExit):
        parser.parse_args(["calibration"])


def test_preflight_accepts_an_optional_reviewed_access_permit(tmp_path: Path) -> None:
    parser = _module().build_parser()
    permit = tmp_path / "permit.json"

    args = parser.parse_args(
        [*_common(tmp_path, "preflight"), "--access-permit", str(permit)]
    )

    assert args.access_permit == permit


@pytest.mark.parametrize(
    "flag",
    (
        "--alpha",
        "--threshold",
        "--minimum-coverage",
        "--delta",
        "--grid-size",
        "--score",
        "--risk",
        "--seed",
        "--split",
        "--sample-id",
        "--device",
        "--workers",
        "--output-name",
    ),
)
def test_parser_rejects_every_scientific_or_execution_override(
    tmp_path: Path, flag: str
) -> None:
    with pytest.raises(SystemExit):
        _module().build_parser().parse_args([*_common(tmp_path, "evaluate"), flag, "1"])


def test_main_routes_handlers_and_emits_one_canonical_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    events: list[str] = []
    monkeypatch.setattr(
        module,
        "run_metadata_preflight",
        lambda **kwargs: events.append("preflight")
        or {"schema": "trustsr.phase2b3c-cli-preflight.v1"},
    )

    assert module.main(_common(tmp_path, "preflight")) == 0

    captured = capsys.readouterr()
    assert captured.out.encode() == canonical_json(
        {"schema": "trustsr.phase2b3c-cli-preflight.v1"}
    ) + b"\n"
    assert events == ["preflight"]
    assert json.loads(captured.out)["schema"] == "trustsr.phase2b3c-cli-preflight.v1"


def test_pyproject_registers_phase2b3c_command() -> None:
    project = Path(__file__).parents[2]
    configuration = tomllib.loads((project / "pyproject.toml").read_text())

    assert configuration["project"]["scripts"]["trustsr-phase2b3c"] == (
        "trustsr.cli.phase2b3c:main"
    )
