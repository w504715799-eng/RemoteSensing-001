import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "render_evidence", ROOT / "scripts/paper/render_evidence.py"
)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def test_development_tables_average_rois_not_gain():
    samples = []
    for aurc, rho, curve in [(0.2, 0.4, 0.1), (0.4, 0.8, 0.3)]:
        metric = dict(
            aurc=aurc,
            rho=rho,
            aurc_gain=99,
            random_aurc=0.5,
            coverages=[i / 10 for i in range(1, 11)],
            selective_mean_risks=[curve] * 10,
        )
        samples.append(
            dict(
                scores=[
                    dict(name="test_score", primary_window_9=metric, sensitivity_window_1=metric)
                ]
            )
        )
    scores, curves = renderer.development_tables(
        dict(samples=samples, candidate_names=["test_score"])
    )
    assert [r["window"] for r in scores] == [1, 9]
    assert scores[0]["mean_aurc"] == pytest.approx(0.3)
    assert scores[0]["mean_rho"] == pytest.approx(0.6)
    assert scores[0]["roi_count"] == 2
    assert len(curves) == 40  # Two windows, score and analytic random reference.
    row = next(r for r in curves if r["method"] == "test_score")
    assert row["mean_selective_risk"] == pytest.approx(0.2)
    assert all(r["mean_selective_risk"] == 0.5 for r in curves if r["method"] == "random_analytic")


def test_tampered_input_stops_before_any_output(tmp_path):
    for relative, _ in renderer.INPUTS.values():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    relative, _ = renderer.INPUTS["C"]
    with (tmp_path / relative).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        renderer.render(tmp_path)
    assert not (tmp_path / "paper").exists()


def test_pinned_render_is_repeatable_and_keeps_bound_roles_separate(tmp_path):
    import csv

    for relative, _ in renderer.INPUTS.values():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    renderer.render(tmp_path)
    before = {p.name: p.read_bytes() for p in (tmp_path / "paper/tables").glob("*.csv")}
    renderer.render(tmp_path)
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "paper/tables").glob("*.csv")}
    rows = list(csv.DictReader((tmp_path / "paper/tables/calibration_evaluation.csv").open()))
    assert rows[0]["risk_ucb"] == ""
    assert float(rows[0]["calibration_criterion"]) == pytest.approx(0.04999883134510526)
    assert rows[1]["calibration_criterion"] == ""
    assert rows[1]["decision"] == "empirically_met_but_inconclusive"
    assert float(rows[1]["risk_ucb"]) == pytest.approx(0.0779855394819395)
    assert (
        (tmp_path / "paper/figures/development_risk_coverage.pdf").read_bytes().startswith(b"%PDF")
    )
