"""Render only pinned, published first-paper evidence."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean

INPUTS = {
    "A": (
        "artifacts/phase2b3a/sen2naipv2-development-score-audit-v1.json",
        "5bb0e5138d6ed1df6c65744556be02ccd48b77d3288df39630d16fbd9cd2dce9",
    ),
    "B": (
        "artifacts/phase2b3b/sen2naipv2-calibration-conformal-v1.json",
        "5fe01daaaa27443d5289cc366eacbe1fb628492c25bab16fadd164d3a8eb3174",
    ),
    "C": (
        "artifacts/phase2b3c/sen2naipv2-internal-test-evaluation-v1.json",
        "9f267e459908a28e6fd254de9cc1d7d2352bf23b672366a9eda9839d53a0e115",
    ),
}


def development_tables(development):
    """Equal-ROI means of published diagnostics; no pixel or cache access."""
    scores, curves = [], []
    for window, key in [(1, "sensitivity_window_1"), (9, "primary_window_9")]:
        for name in sorted(development["candidate_names"]):
            metrics = [
                next(s[key] for s in sample["scores"] if s["name"] == name)
                for sample in development["samples"]
            ]
            scores.append(
                dict(
                    method=name,
                    window=window,
                    roi_count=len(metrics),
                    mean_aurc=fmean(m["aurc"] for m in metrics),
                    mean_rho=fmean(m["rho"] for m in metrics),
                )
            )
            for index, coverage in enumerate(metrics[0]["coverages"]):
                curves.append(
                    dict(
                        method=name,
                        window=window,
                        coverage=coverage,
                        mean_selective_risk=fmean(
                            m["selective_mean_risks"][index] for m in metrics
                        ),
                    )
                )
        # Random expectation is the same for each score on a shared risk map.
        for coverage in metrics[0]["coverages"]:
            curves.append(
                dict(
                    method="random_analytic",
                    window=window,
                    coverage=coverage,
                    mean_selective_risk=fmean(m["random_aurc"] for m in metrics),
                )
            )
    return scores, sorted(curves, key=lambda r: (r["window"], r["method"], r["coverage"]))


def render(root: Path):
    """Validate every pinned input before creating any output."""
    documents = {}
    for role, (relative, digest) in INPUTS.items():
        raw = (root / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        documents[role] = json.loads(raw)
    scores, curves = development_tables(documents["A"])
    b, c = documents["B"], documents["C"]
    statistics = c["statistics"]
    results = [
        dict(
            role="calibration",
            roi_count=b["counts"]["calibration"],
            threshold=b["threshold"],
            coverage=b["coverage"],
            calibration_criterion=b["risk_bound"],
            mean_roi_loss=None,
            risk_ucb=None,
            decision=b["phase_decision"],
        ),
        dict(
            role="internal_test",
            roi_count=statistics["evaluation_size"],
            threshold=b["threshold"],
            coverage=statistics["coverage"],
            calibration_criterion=None,
            mean_roi_loss=statistics["mean_loss"],
            risk_ucb=statistics["risk_ucb"],
            decision=c["phase_decision"],
        ),
    ]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tables = root / "paper/tables"
    figures = root / "paper/figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("development_scores.csv", scores),
        ("development_curves.csv", curves),
        ("calibration_evaluation.csv", results),
    ]:
        with (tables / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for axis, window in zip(axes, [1, 9], strict=True):
        for name in sorted({r["method"] for r in curves}):
            rows = [r for r in curves if r["window"] == window and r["method"] == name]
            axis.plot(
                [r["coverage"] for r in rows],
                [r["mean_selective_risk"] for r in rows],
                label=name,
                linestyle="--" if name == "random_analytic" else "-",
            )
        axis.set(
            title=f"Development R{window} (120 ROIs)",
            xlabel="Retained coverage",
            ylabel="Equal-ROI mean selective L1 risk",
            xlim=(0.1, 1),
        )
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("Development selection evidence, not independent confirmation")
    fig.tight_layout()
    fig.savefig(figures / "development_risk_coverage.pdf")
    plt.close(fig)


if __name__ == "__main__":
    render(Path(__file__).resolve().parents[2])
