"""CPU-only falsifiable mechanism diagnostics, not real-data performance evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from research.trustmask.pipeline import extract_features
from trustsr.risk.local import local_l1_risk


def examples():
    y, x = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    texture = (0.25 + 0.5 * ((x + y) % 2)).to(torch.float32)
    sr = texture.unsqueeze(0).repeat(4, 1, 1)
    samples = sr.unsqueeze(0).repeat(5, 1, 1, 1)
    lr = functional.interpolate(sr.unsqueeze(0), size=(8, 8), mode="area")[0]
    return lr, samples, sr, torch.full_like(sr, 0.5)


def diagnose():
    lr, samples, faithful, invented = examples()
    feature = extract_features(lr, samples)
    feature2 = extract_features(lr.clone(), samples.clone())
    rows = {}
    for name, reference in [("faithful_texture", faithful), ("invented_texture", invented)]:
        risk = local_l1_risk(samples[0], reference, window=9)
        rows[name] = dict(
            risk_max=float(risk.max()),
            mean_risk=float(risk.mean()),
            mean_texture_score=float(feature.texture.mean()),
            mean_uncertainty_score=float(feature.uncertainty.mean()),
            mean_lr_residual=float(feature.lr.mean()),
        )
    levels = torch.linspace(0.25, 0.75, 5)
    constant_samples = levels[:, None, None, None].expand(5, 4, 32, 32).contiguous()
    constant_lr = functional.interpolate(constant_samples[0:1], size=(8, 8), mode="area")[0]
    constant_feature = extract_features(constant_lr, constant_samples)

    def pool(value):
        return functional.avg_pool2d(
            functional.pad(value, (1, 1, 1, 1), mode="reflect"), 3, stride=1
        )

    values = constant_samples.double()
    spatial = pool(values.square()) - pool(values).square()
    return dict(
        schema="trustmask-synthetic-publication-mechanisms-v1",
        evidence_kind="synthetic_mechanism_diagnostic_not_research_benefit",
        gpu_used=False,
        real_data_used=False,
        same_lr_and_predictions=all(
            torch.equal(lr, functional.interpolate(ref[None], size=(8, 8), mode="area")[0])
            for ref in (faithful, invented)
        ),
        same_features=all(
            np.array_equal(getattr(feature, key), getattr(feature2, key))
            for key in ("lr", "variance", "uncertainty", "texture")
        ),
        **rows,
        spatially_constant_distinct_samples=dict(
            joint_score_mean=float(
                (constant_feature.uncertainty + constant_feature.texture).mean()
            ),
            mean_per_sample_spatial_variance=float(spatial.mean()),
            between_sample_variance=float(values.var(0, correction=0).mean()),
        ),
        conclusions=[
            "Exact, identical samples can still have positive texture score.",
            "Identical observables can have different errors; these features cannot tell.",
            "Joint variance differs from average per-sample spatial variance.",
        ],
        limitations=[
            "No real-data frequency, accuracy, coverage gain, or prevalence is established.",
            "Variance decomposition and non-identifiability are not claimed as new theorems.",
            "Texture penalties can flag invented detail; removal is not always beneficial.",
        ],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    report = diagnose()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lr, samples, faithful, invented = examples()
    feature = extract_features(lr, samples)
    figure, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
    panels = [
        (samples[0, 0].numpy(), "Shared generated image", 0.0, 1.0),
        (faithful[0].numpy(), "Reference A (error = 0)", 0.0, 1.0),
        (invented[0].numpy(), "Reference B (error = 0.25)", 0.0, 1.0),
        (feature.texture, "Shared texture score", 0.0, float(feature.texture.max())),
    ]
    for axis, (values, title, low, high) in zip(axes, panels, strict=True):
        plotted = axis.imshow(
            values, cmap="viridis" if axis is axes[-1] else "gray", vmin=low, vmax=high
        )
        if axis is axes[-1]:
            figure.colorbar(plotted, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title(title, fontsize=10)
        axis.set_axis_off()
    figure.suptitle(
        "Synthetic diagnostic: identical observable inputs, different reference errors", fontsize=11
    )
    figure.savefig(args.output / "mechanism-counterexample.png", dpi=180)
    plt.close(figure)
    report["figure_sha256"] = hashlib.sha256(
        (args.output / "mechanism-counterexample.png").read_bytes()
    ).hexdigest()
    (args.output / "diagnostics.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps({"output": str(args.output), "gpu_used": False, "real_data_used": False}))


if __name__ == "__main__":
    main()
