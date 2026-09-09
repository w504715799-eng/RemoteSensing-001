"""Group-level CRC and held-out bounded-mean comparisons, under stated assumptions."""

import math

import numpy as np


def probability(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a probability strictly between zero and one")
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be a probability strictly between zero and one")
    return float(value)


def bounded(value, *, ndim):
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != ndim
        or not array.size
        or not np.isfinite(array).all()
        or (array < 0).any()
        or (array > 1).any()
    ):
        raise ValueError("nonempty finite bounded [0,1] array required")
    return array


def calibrate(losses, thresholds, *, alpha=0.05):
    """Largest fixed-grid threshold satisfying CRC's worst-new-group correction.

    Expected-risk statement averages over calibration and a fresh exchangeable
    group. It is NOT a high-probability bound conditional on this calibration.
    Threshold -1 means structurally reject everything, independent of any data.
    """
    alpha = probability(alpha, "alpha")
    losses = bounded(losses, ndim=2)
    grid = bounded(thresholds, ndim=1)
    if losses.shape[1] != len(grid) or (np.diff(grid) <= 0).any():
        raise ValueError("strictly increasing matching threshold grid required")
    if (np.diff(losses, axis=1) < 0).any():
        raise ValueError("loss curves must be monotone nondecreasing")
    n = len(losses)
    corrected = (losses.sum(axis=0) + 1) / (n + 1)
    valid = np.flatnonzero(corrected <= alpha)
    index = int(valid[-1]) if len(valid) else None
    return dict(
        threshold=float(grid[index]) if index is not None else -1.0,
        reject_all=index is None,
        groups=n,
        alpha=alpha,
        calibration_mean_loss=float(losses[:, index].mean()) if index is not None else 0.0,
        corrected_calibration_risk=float(corrected[index]) if index is not None else None,
        rule="crc_bounded_group_loss_worst_new_group",
        guarantee="marginal_expected_risk_under_exchangeable_groups_not_conditional_high_probability",
    )


def evaluate_comparison(losses, coverages, *, alpha=0.05, risk_delta=0.025, gain_delta=0.025):
    """Evaluate aligned group vectors; caller binds their identities and frozen methods."""
    alpha = probability(alpha, "alpha")
    risk_delta = probability(risk_delta, "risk_delta")
    gain_delta = probability(gain_delta, "gain_delta")
    if risk_delta + gain_delta >= 1:
        raise ValueError("joint failure budget must be below one")
    if set(losses) != set(coverages) or not {"full", "w3"}.issubset(losses):
        raise ValueError("matching methods including full and w3 required")
    ls = {k: bounded(v, ndim=1) for k, v in losses.items()}
    cs = {k: bounded(v, ndim=1) for k, v in coverages.items()}
    sizes = {len(v) for v in [*ls.values(), *cs.values()]}
    if len(sizes) != 1:
        raise ValueError("aligned equal-size group vectors required")
    n = sizes.pop()
    margin = math.sqrt(math.log(len(ls) / risk_delta) / (2 * n))
    methods = {}
    for name in ls:
        upper = min(1.0, float(ls[name].mean()) + margin)
        methods[name] = dict(
            mean_loss=float(ls[name].mean()),
            coverage=float(cs[name].mean()),
            risk_upper=upper,
            risk_qualified=upper <= alpha,
        )
    difference = cs["full"] - cs["w3"]
    gain_margin = math.sqrt(2 * math.log(1 / gain_delta) / n)
    lower = max(-1.0, float(difference.mean()) - gain_margin)
    qualified = methods["full"]["risk_qualified"] and methods["w3"]["risk_qualified"]
    return dict(
        groups=n,
        methods=methods,
        method_count=len(ls),
        alpha=alpha,
        risk_family_delta=risk_delta,
        primary_gain_delta=gain_delta,
        risk_margin=margin,
        gain_margin=gain_margin,
        primary=dict(
            baseline="w3",
            candidate="full",
            mean_difference=float(difference.mean()),
            lower_bound=lower,
            both_risk_qualified=qualified,
            qualified_positive_gain=qualified and lower > 0,
            qualified_five_point_gain=qualified and lower >= 0.05,
        ),
        other_comparisons="descriptive_only",
        inferential_basis="conditional_on_independent_test_groups_and_frozen_methods",
    )
