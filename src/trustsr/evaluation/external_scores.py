"""CPU-only external diagnostics; no IO, model loading, or calibration fitting.

The caller authenticates predictions and binds seed order before entering here.
Invoke the summary separately for each frozen subset, never on pooled pixels.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import torch

from trustsr.contracts import SRPair
from trustsr.evaluation.score_diagnostics import DEFAULT_COVERAGES, evaluate_roi_score
from trustsr.risk.local import ensemble_variance_score, local_l1_risk
from trustsr.risk.neighborhood import neighborhood_variance_score
from trustsr.risk.proxies import lr_reprojection_l1_score, three_model_disagreement_score

TRANSFER_THRESHOLD = 7.970395366024563e-06


def build_score_maps(
    lr: torch.Tensor, samples: torch.Tensor, bicubic: torch.Tensor, sen2sr: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build four scores without accepting HR; samples[0] is seed 3407.

    Random is an analytical risk expectation, not a realized score map.
    Neighborhood window 3 was selected on development, not on these inputs.
    """
    if (not isinstance(samples, torch.Tensor) or samples.dtype != torch.float32
            or samples.ndim != 4 or samples.shape[:2] != (5, 4)):
        raise ValueError("samples must be float32 (5, 4, H, W) in frozen seed order")
    samples = samples.detach().cpu()
    k5 = ensemble_variance_score(samples)
    return {
        "lr": lr_reprojection_l1_score(samples[0], lr),
        "three_model": three_model_disagreement_score([samples[0], bicubic, sen2sr]),
        "k5": k5,
        "neighborhood": neighborhood_variance_score(samples, window=3),
    }


def evaluate_external_roi(
    pair: SRPair, samples: torch.Tensor, bicubic: torch.Tensor, sen2sr: torch.Tensor,
) -> dict:
    """Return JSON-safe R1/R9 diagnostics and descriptive frozen K5 transfer.

    This array-level function supports synthetic grids. The Spain loader is
    responsible for enforcing full 512x512 grids and source identity.
    """
    pair.validate()
    if not isinstance(pair.sample_id, str) or not pair.sample_id.strip():
        raise ValueError("sample_id must be nonempty")
    maps = build_score_maps(pair.lr, samples, bicubic, sen2sr)
    center = samples[0].detach().cpu()
    return evaluate_score_maps(pair, center, maps)


def evaluate_score_maps(pair: SRPair, center: torch.Tensor, maps: dict) -> dict:
    """Evaluate available score maps on one center, retaining analytical random risk."""
    pair.validate()
    SRPair(pair.sample_id, pair.source, pair.lr, center, pair.scale).validate()
    hr = pair.hr.detach().cpu()
    result = {"roi": pair.sample_id, "transfer": None}
    for window in (1, 9):
        risk = local_l1_risk(center, hr, window=window)
        diagnostics = {}
        for name, score in maps.items():
            row = asdict(evaluate_roi_score(score, risk))
            # Old field names must not invert the meaning in new paper output.
            row["high_risk_rejection_at_80"] = row.pop("high_risk_miss_rate_at_80")
            diagnostics[name] = row
        random_risk = float(risk.mean())
        diagnostics["random"] = {
            "rho": None, "coverages": DEFAULT_COVERAGES,
            "selective_mean_risks": [random_risk] * len(DEFAULT_COVERAGES),
            "aurc": random_risk, "random_aurc": random_risk, "aurc_gain": 0.0,
        }
        result[f"R{window}"] = diagnostics
        if window == 9 and "k5" in maps:
            mask = maps["k5"] <= TRANSFER_THRESHOLD
            rejected = not bool(mask.any())
            result["transfer"] = {
                "coverage": float(mask.to(torch.float64).mean()),
                "roi_max_r9": 0.0 if rejected else float(risk[mask].max()),
                "all_rejected": rejected,
            }
    return result


def summarize_subset(members: list[str], rows: list[dict], failures: dict[str, str]) -> dict:
    """Summarize the primary complete-pair comparison with exhaustive accounting.

    Failures here represent whole-ROI failures. Partial-method accounting belongs
    to the runner; this function must not silently discard incomplete records.
    """
    if (not members or any(not isinstance(item, str) or not item.strip() for item in members)
            or len(set(members)) != len(members)):
        raise ValueError("members must be unique nonempty IDs")
    ids = [row["roi"] for row in rows]
    if (len(set(ids)) != len(ids) or set(ids) & set(failures)
            or set(ids) | set(failures) != set(members)):
        raise ValueError("every frozen member must have exactly one result or failure")
    if any(not isinstance(reason, str) or not reason.strip() for reason in failures.values()):
        raise ValueError("failures require explicit reasons")
    differences = []
    for row in sorted(rows, key=lambda item: item["roi"]):
        values = [row["R9"][name]["aurc"] for name in ("lr", "k5")]
        if any(type(value) not in (int, float) or not math.isfinite(value)
               or not 0 <= value <= 1 for value in values):
            raise ValueError("AURCs must be finite numbers in [0, 1]")
        differences.append({"roi": row["roi"], "difference": values[1] - values[0]})
    return {
        "planned": len(members), "paired_valid": len(differences),
        "paired_differences": differences,
        "mean_paired_difference": (
            math.fsum(item["difference"] for item in differences) / len(differences)
            if differences else None
        ),
        "failures": dict(sorted(failures.items())),
    }
