"""CPU integration of fixed-scale scores, development selection and group calibration.

Inputs are precomputed observations, not authenticated dataset assets. Structural
role checks do not implement a one-time test ledger or establish group independence.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from research.trustmask.benefits import evaluate_mask
from research.trustmask.calibration import bounded, calibrate, evaluate_comparison, probability
from research.trustmask.scores import spatial_components
from trustsr.risk.proxies import lr_reprojection_l1_score

ROLES = ("scale_fit", "development_calibration", "development_validation", "calibration", "test")
GRID = tuple(float(x) for x in np.linspace(0, 1, 101))
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
BETAS = (0.0, 0.5, 1.0)


def frozen_map(value):
    if not isinstance(value, np.ndarray) or value.dtype.kind != "f":
        raise ValueError("floating numpy map required")
    array = bounded(value, ndim=2)
    # Immutable bytes backing prevents writes even via setflags(write=True).
    return np.frombuffer(array.astype("<f8").tobytes(), dtype="<f8").reshape(array.shape)


@dataclass(frozen=True)
class Features:
    lr: np.ndarray
    variance: np.ndarray
    uncertainty: np.ndarray
    texture: np.ndarray

    def __post_init__(self):
        for name in ("lr", "variance", "uncertainty", "texture"):
            object.__setattr__(self, name, frozen_map(getattr(self, name)))
        if (
            len(
                {getattr(self, name).shape for name in ("lr", "variance", "uncertainty", "texture")}
            )
            != 1
        ):
            raise ValueError("feature grids must match")


@dataclass(frozen=True)
class Observation:
    sample_id: str
    group_id: str
    features: Features
    risk: np.ndarray

    def __post_init__(self):
        if any(not isinstance(s, str) or not s.strip() for s in (self.sample_id, self.group_id)):
            raise ValueError("nonempty sample and group identities required")
        if not isinstance(self.features, Features):
            raise ValueError("Features required")
        object.__setattr__(self, "risk", frozen_map(self.risk))
        if self.risk.shape != self.features.lr.shape:
            raise ValueError("risk and features must share the full fixed grid")


@dataclass(frozen=True)
class Config:
    kind: str
    weight: float = 0.5
    beta: float = 0.5

    def __post_init__(self):
        if (
            self.kind not in {"lr", "k5", "w3", "fusion", "spatial", "full"}
            or type(self.weight) not in (int, float)
            or self.weight not in WEIGHTS
            or type(self.beta) not in (int, float)
            or self.beta not in BETAS
        ):
            raise ValueError("configuration outside declared finite family")


def extract_features(lr, samples):
    """Reference-free adapter: samples[0] is the declared central prediction (seed 3407).

    Caller supplies five ordered RGBN predictions on the same full support. No HR
    input. This function cannot authenticate their model, seeds or member identity.
    """
    uncertainty, texture = spatial_components(samples)
    if samples.dtype != torch.float32:
        raise ValueError("five prediction inputs must be float32")
    residual = lr_reprojection_l1_score(samples[0], lr, scale=4)
    variance = samples.detach().to(device="cpu", dtype=torch.float64).var(0, correction=0).mean(0)
    return Features(*(v.numpy() for v in (residual, variance, uncertainty, texture)))


def raw_components(features):
    return {
        "lr": features.lr,
        "variance": features.variance,
        **{f"q{b:g}": features.uncertainty + b * features.texture for b in BETAS},
    }


def fit_scales(features):
    """64 fixed midpoint samples per ROI, repeated for tiny synthetic grids; equal ROI weight."""
    features = tuple(features)
    if not features or any(not isinstance(f, Features) for f in features):
        raise ValueError("nonempty scale-fit feature collection required")
    sampled = {key: [] for key in raw_components(features[0])}
    for feature in features:
        for name, values in raw_components(feature).items():
            index = np.floor((np.arange(64) + 0.5) * values.size / 64).astype(int)
            sampled[name].append(values.ravel()[index])
    return tuple(
        (key, max(float(np.median(np.concatenate(values))), 1e-8))
        for key, values in sampled.items()
    )


def score_map(features, config, scales):
    if not isinstance(features, Features) or not isinstance(config, Config):
        raise ValueError("Features and Config required")
    scales = dict(scales)
    raw = raw_components(features)
    if set(scales) != set(raw) or any(
        type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in scales.values()
    ):
        raise ValueError("all fixed positive scales required")
    normalized = {k: v / (v + scales[k]) for k, v in raw.items()}
    if config.kind == "lr":
        return normalized["lr"]
    if config.kind == "k5":
        return normalized["variance"]
    if config.kind == "w3":
        return normalized["q1"]
    right = normalized["variance"] if config.kind == "fusion" else normalized[f"q{config.beta:g}"]
    if config.kind == "spatial":
        return right
    return config.weight * normalized["lr"] + (1 - config.weight) * right


def deploy_mask(features, config, scales, threshold):
    """Deploy-time interface deliberately excludes HR and risk maps."""
    if type(threshold) not in (int, float) or not math.isfinite(threshold):
        raise ValueError("finite threshold required")
    if threshold == -1:
        return np.zeros(features.lr.shape, dtype=bool)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be -1 or in [0,1]")
    return score_map(features, config, scales) <= threshold


def group_curves(observations, config, scales, grid=GRID):
    """Compute loss and coverage using score sorting, without pixel x threshold arrays."""
    grid = np.asarray(grid, dtype=np.float64)
    if (
        grid.ndim != 1
        or not grid.size
        or not np.isfinite(grid).all()
        or (np.diff(grid) <= 0).any()
        or ((grid != -1) & ((grid < 0) | (grid > 1))).any()
    ):
        raise ValueError("valid strictly increasing threshold grid required")
    grouped = {}
    seen = set()
    for row in observations:
        if not isinstance(row, Observation) or row.sample_id in seen:
            raise ValueError("unique observation identities required")
        seen.add(row.sample_id)
        score = score_map(row.features, config, scales).ravel()
        order = np.argsort(score, kind="stable")
        count = np.searchsorted(score[order], grid, side="right")
        maxima = np.r_[0.0, np.maximum.accumulate(row.risk.ravel()[order])]
        loss, coverage = maxima[count], count / score.size
        grouped.setdefault(row.group_id, []).append((loss, coverage))
    if not grouped:
        raise ValueError("nonempty observations required")
    ids = tuple(sorted(grouped))
    curves = [np.mean(grouped[g], axis=0) for g in ids]
    return ids, np.stack([x[0] for x in curves]), np.stack([x[1] for x in curves])


def families():
    return {
        "full": tuple(Config("full", w, b) for w in WEIGHTS for b in BETAS),
        "lr": (Config("lr"),),
        "k5": (Config("k5"),),
        "w3": (Config("w3"),),
        "fusion": tuple(Config("fusion", w) for w in WEIGHTS),
        "spatial": tuple(Config("spatial", beta=b) for b in BETAS),
        "beta0": tuple(Config("full", w, 0.0) for w in WEIGHTS),
        "beta1": tuple(Config("full", w, 1.0) for w in WEIGHTS),
    }


def check_roles(partitions):
    if set(partitions) != set(ROLES) or any(not partitions[r] for r in ROLES):
        raise ValueError("all five nonempty roles required")
    ids, groups = set(), set()
    for role in ROLES:
        rows = partitions[role]
        if any(not isinstance(r, Observation) for r in rows):
            raise ValueError("Observation inputs required")
        local_ids, local_groups = {r.sample_id for r in rows}, {r.group_id for r in rows}
        if len(local_ids) != len(rows) or ids & local_ids or groups & local_groups:
            raise ValueError("duplicate identity or leakage across roles")
        ids.update(local_ids)
        groups.update(local_groups)


def select_configuration(configs, scales, development_calibration, development_validation, alpha):
    candidates = []
    for index, config in enumerate(configs):
        _, loss, _ = group_curves(development_calibration, config, scales)
        calibration = calibrate(loss, np.array(GRID), alpha=alpha)
        _, validation_loss, coverage = group_curves(
            development_validation, config, scales, [calibration["threshold"]]
        )
        risk, retained = float(validation_loss.mean()), float(coverage.mean())
        candidates.append(
            dict(
                index=index,
                config=asdict(config),
                validation_risk=risk,
                validation_coverage=retained,
                development_calibration=calibration,
            )
        )
    feasible = [c for c in candidates if c["validation_risk"] <= alpha]
    if not feasible:
        return configs[0], candidates, True
    best = min(
        feasible, key=lambda c: (-c["validation_coverage"], c["validation_risk"], c["index"])
    )
    return Config(**best["config"]), candidates, False


def run_study(partitions, *, alpha=0.05):
    """Fit on development, recalibrate frozen configs, evaluate once per invocation.

    This reusable CPU function does not enforce real-test single consumption.
    A production wrapper must authenticate support/membership and own that ledger.
    """
    alpha = probability(alpha, "alpha")
    check_roles(partitions)
    scales = fit_scales([r.features for r in partitions["scale_fit"]])
    frozen, development, losses, coverages, workloads = {}, {}, {}, {}, {}
    selected = {}
    test_groups = None
    for name, configs in families().items():
        config, candidates, force_reject = select_configuration(
            configs,
            scales,
            partitions["development_calibration"],
            partitions["development_validation"],
            alpha,
        )
        development[name] = candidates
        selected[name] = (config, force_reject)
    # Complete every development decision before accessing formal calibration losses.
    for name, (config, force_reject) in selected.items():
        _, cal_loss, _ = group_curves(partitions["calibration"], config, scales)
        calibration = calibrate(cal_loss, np.array(GRID), alpha=alpha)
        if force_reject:
            calibration.update(
                threshold=-1.0,
                reject_all=True,
                calibration_mean_loss=0.0,
                corrected_calibration_risk=None,
                rule="structural_rejection_no_development_candidate_qualified",
                guarantee="structural_zero_loss_from_rejecting_every_pixel",
            )
        frozen[name] = dict(config=asdict(config), scales=dict(scales), calibration=calibration)
    # No test loss/coverage access until every configuration and threshold is fixed.
    for name, method in frozen.items():
        config = Config(**method["config"])
        threshold = method["calibration"]["threshold"]
        ids, loss, coverage = group_curves(partitions["test"], config, scales, [threshold])
        if test_groups is not None and ids != test_groups:
            raise ValueError("test group alignment mismatch")
        test_groups = ids
        losses[name], coverages[name] = loss[:, 0], coverage[:, 0]
        summaries = [
            evaluate_mask(r.risk, deploy_mask(r.features, config, scales, threshold))
            for r in partitions["test"]
        ]
        keys = (
            "accepted_pixels",
            "review_pixels",
            "review_tiles",
            "total_tiles",
            "accepted_high_error_pixels",
            "high_error_pixels",
        )
        workloads[name] = {key: sum(s[key] for s in summaries) for key in keys}
        workloads[name].update(
            all_rejected_rois=sum(s["all_rejected"] for s in summaries),
            physical_area_m2=None,
            human_time_measured=False,
        )
    frozen_json = json.dumps(frozen, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return dict(
        schema="trustmask-integrated-study-v1",
        evidence_kind="offline_precomputed_inputs_not_provenance_authenticated",
        frozen_methods=frozen,
        frozen_methods_sha256=hashlib.sha256(frozen_json.encode()).hexdigest(),
        development=development,
        test_group_ids=list(test_groups),
        evaluation=evaluate_comparison(losses, coverages, alpha=alpha),
        workload=workloads,
        stage_counts={
            role: dict(rois=len(rows), groups=len({r.group_id for r in rows}))
            for role, rows in partitions.items()
        },
        weighting="equal_groups_then_equal_rois",
        method_family_fixed_before_calibration=True,
        threshold_grid=list(GRID),
        scale_fit_rule="64_fixed_midpoint_positions_per_roi_median_floor_1e-8",
        test_group_observations={
            name: dict(loss=losses[name].tolist(), coverage=coverages[name].tolist())
            for name in frozen
        },
    )
