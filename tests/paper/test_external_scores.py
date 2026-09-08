"""Synthetic oracles for the external score pipeline; no image IO."""

import importlib
import json

import pytest
import torch

from trustsr.contracts import SRPair


def api():
    return importlib.import_module("trustsr.evaluation.external_scores")


def fixture():
    pair = SRPair("roi", "synthetic", torch.zeros(4, 3, 3),
                  torch.full((4, 12, 12), 0.25), 4)
    samples = torch.zeros(5, 4, 12, 12)
    samples[3:] = 1
    return pair, samples, torch.zeros(4, 12, 12), torch.ones(4, 12, 12)


def test_scores_use_population_variance_and_shared_center():
    pair, samples, bicubic, sen2sr = fixture()
    maps = api().build_score_maps(pair.lr, samples, bicubic, sen2sr)
    for name, expected in [("lr", 0), ("three_model", 2 / 9),
                           ("k5", 0.24), ("neighborhood", 0.24)]:
        assert maps[name].shape == (12, 12)
        torch.testing.assert_close(maps[name], torch.full((12, 12), expected,
                                                        dtype=torch.float64))


def test_all_methods_evaluate_center_not_ensemble_mean_and_are_json_safe():
    record = api().evaluate_external_roi(*fixture())
    assert record["roi"] == "roi"
    for window in ("R1", "R9"):
        assert set(record[window]) == {"lr", "three_model", "k5", "neighborhood", "random"}
        for method in record[window].values():
            assert method["aurc"] == pytest.approx(0.25)
            assert method["selective_mean_risks"] == pytest.approx([0.25] * 10)
        assert record[window]["random"]["rho"] is None
    assert record["transfer"] == {"coverage": 0.0, "roi_max_r9": 0.0, "all_rejected": True}
    json.dumps(record, allow_nan=False)


def test_random_is_analytic_not_index_order_on_nonconstant_risk():
    pair, samples, bicubic, sen2sr = fixture()
    pair.hr[:, :, 6:] = 0.75
    result = api().evaluate_external_roi(pair, samples, bicubic, sen2sr)
    assert result["R1"]["random"]["selective_mean_risks"] == [0.5] * 10
    assert result["R1"]["lr"]["aurc"] < 0.5


def test_fixed_threshold_reports_max_not_mean_risk():
    pair, samples, bicubic, sen2sr = fixture()
    samples[:] = 0
    pair.hr[:, :, 6:] = 0.75
    result = api().evaluate_external_roi(pair, samples, bicubic, sen2sr)
    assert result["transfer"] == {"coverage": 1.0, "roi_max_r9": 0.75, "all_rejected": False}


@pytest.mark.parametrize("bad", [torch.zeros(4, 4, 12, 12),
                                 torch.zeros(5, 4, 12, 12, dtype=torch.float64),
                                 torch.full((5, 4, 12, 12), float("nan")),
                                 torch.full((5, 4, 12, 12), 1.1)])
def test_invalid_predictions_fail_before_scoring(bad):
    pair, _, bicubic, sen2sr = fixture()
    with pytest.raises(ValueError):
        api().evaluate_external_roi(pair, bad, bicubic, sen2sr)


def summary_row(roi, lr, k5):
    return {"roi": roi, "R9": {"lr": {"aurc": lr}, "k5": {"aurc": k5}}}


def test_subset_summary_equal_roi_pairs_and_explicit_whole_roi_failures():
    result = api().summarize_subset(
        ["a", "b", "c"], [summary_row("b", 0.8, 0.4), summary_row("a", 0.2, 0.1)],
        {"c": "prediction_failed"})
    assert result["planned"] == 3
    assert result["paired_valid"] == 2
    assert result["paired_differences"] == [{"roi": "a", "difference": -0.1},
                                            {"roi": "b", "difference": -0.4}]
    assert result["mean_paired_difference"] == -0.25
    assert result["failures"] == {"c": "prediction_failed"}


def test_no_pairs_is_missing_not_zero():
    result = api().summarize_subset(["a"], [], {"a": "prediction_failed"})
    assert result["mean_paired_difference"] is None
    assert result["paired_valid"] == 0


@pytest.mark.parametrize("members,rows,failures", [
    (["a", "a"], [], {"a": "failed"}),
    (["a"], [summary_row("b", 0.2, 0.1)], {}),
    (["a"], [summary_row("a", 0.2, 0.1)] * 2, {}),
    (["a"], [], {}),
    (["a"], [summary_row("a", 0.2, 0.1)], {"a": "failed"}),
    (["a"], [summary_row("a", float("nan"), 0.1)], {}),
])
def test_summary_rejects_silent_member_loss_duplicates_and_invalid_values(members, rows, failures):
    with pytest.raises(ValueError):
        api().summarize_subset(members, rows, failures)
