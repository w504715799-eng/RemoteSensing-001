import numpy as np
import pytest
import torch

from research.trustmask.pipeline import (
    Config,
    Features,
    Observation,
    deploy_mask,
    fit_scales,
    group_curves,
    run_study,
)


def feature(value=0.2):
    return Features(*(np.full((4, 4), value) for _ in range(4)))


def obs(name, group, risk=0.01, value=0.2):
    return Observation(name, group, feature(value), np.full((4, 4), risk))


def test_group_curve_averages_rois_within_group_then_groups():
    data = [obs("a", "A", 0.2), obs("b", "A", 0.4), obs("c", "B", 0.9)]
    groups, losses, coverage = group_curves(
        data, Config("lr"), fit_scales([feature()]), np.array([0.0, 1.0])
    )
    assert groups == ("A", "B")
    assert losses[:, 1] == pytest.approx([0.3, 0.9])
    assert coverage[:, 0] == pytest.approx([0.0, 0.0])


def test_scales_use_equal_roi_samples_and_reject_mask_is_structural():
    small = Features(*(np.full((1, 1), 0.1) for _ in range(4)))
    big = Features(*(np.full((20, 20), 0.9) for _ in range(4)))
    scales = fit_scales([small, big])
    assert dict(scales)["lr"] == pytest.approx(0.5)
    assert not deploy_mask(big, Config("lr"), scales, -1).any()


def partitions():
    return {
        role: [obs(f"{role}-{i}", f"{role}-{i}", value=0.1 + i / 1000) for i in range(25)]
        for role in (
            "scale_fit",
            "development_calibration",
            "development_validation",
            "calibration",
            "test",
        )
    }


def test_full_run_and_test_labels_cannot_change_selected_or_calibrated_methods():
    first = partitions()
    result = run_study(first)
    assert result["evaluation"]["method_count"] == 8
    assert result["evidence_kind"] == "offline_precomputed_inputs_not_provenance_authenticated"
    second = partitions()
    second["test"] = [obs(r.sample_id, r.group_id, risk=0.9) for r in second["test"]]
    changed = run_study(second)
    assert result["frozen_methods"] == changed["frozen_methods"]
    assert result["evaluation"]["methods"]["full"]["mean_loss"] < 0.05
    assert changed["evaluation"]["methods"]["full"]["mean_loss"] > 0.8


def test_cross_role_group_leak_rejected():
    data = partitions()
    data["test"][0] = obs("different", data["calibration"][0].group_id)
    with pytest.raises(ValueError, match="across roles"):
        run_study(data)


def test_no_development_candidate_qualifies_keeps_reject_only_method():
    data = partitions()
    data["development_validation"] = [
        obs(r.sample_id, r.group_id, risk=0.9) for r in data["development_validation"]
    ]
    result = run_study(data)
    assert all(m["calibration"]["threshold"] == -1 for m in result["frozen_methods"].values())
    assert all(m["coverage"] == 0 for m in result["evaluation"]["methods"].values())


def test_adapter_uses_central_prediction_and_has_no_reference_input():
    from research.trustmask.pipeline import extract_features

    predictions = torch.full((5, 4, 8, 8), 0.5, dtype=torch.float32)
    lr = torch.full((4, 2, 2), 0.25, dtype=torch.float32)
    extracted = extract_features(lr, predictions)
    assert extracted.lr == pytest.approx(np.full((8, 8), 0.25))
    assert not extracted.variance.any()
    assert not extracted.uncertainty.any()
    assert not extracted.texture.any()


def test_fixed_scale_snapshot_cannot_be_mutated_through_input_array():
    source = np.ones((2, 2))
    copied = Features(source, source, source, source)
    source[:] = 0
    assert copied.lr.min() == 1
    with pytest.raises(ValueError):
        copied.lr.setflags(write=True)


def test_calibration_labels_do_not_choose_configuration_or_scales():
    first = partitions()
    before = run_study(first)
    first["calibration"] = [obs(r.sample_id, r.group_id, risk=0.9) for r in first["calibration"]]
    after = run_study(first)
    for name in before["frozen_methods"]:
        a, b = before["frozen_methods"][name], after["frozen_methods"][name]
        assert a["config"] == b["config"]
        assert a["scales"] == b["scales"]
        assert a["calibration"]["threshold"] != b["calibration"]["threshold"]


def test_sorted_curves_match_direct_masks_for_every_candidate_and_unequal_groups():
    from research.trustmask.pipeline import families

    random = np.random.default_rng(47)
    rows = [
        Observation(
            str(i),
            "A" if i < 2 else "B",
            Features(*(random.uniform(0, 1, (3, 4)) for _ in range(4))),
            random.uniform(0, 1, (3, 4)),
        )
        for i in range(3)
    ]
    scales = fit_scales([r.features for r in rows])
    grid = np.array([-1.0, 0.0, 0.2, 0.5, 0.8, 1.0])
    for configs in families().values():
        for config in configs:
            ids, losses, coverage = group_curves(rows, config, scales, grid)
            for j, group in enumerate(ids):
                members = [r for r in rows if r.group_id == group]
                for k, threshold in enumerate(grid):
                    masks = [
                        deploy_mask(r.features, config, scales, float(threshold)) for r in members
                    ]
                    expected = [
                        r.risk[m].max() if m.any() else 0
                        for r, m in zip(members, masks, strict=True)
                    ]
                    assert losses[j, k] == pytest.approx(np.mean(expected))
                    assert coverage[j, k] == pytest.approx(np.mean([m.mean() for m in masks]))


def test_all_configurations_selected_before_any_formal_calibration(monkeypatch):
    from research.trustmask import pipeline

    original = pipeline.group_curves
    validation_calls = 0
    expected = sum(len(c) for c in pipeline.families().values())

    def tracked(rows, *args, **kwargs):
        nonlocal validation_calls
        if rows[0].sample_id.startswith("development_validation-"):
            validation_calls += 1
        if rows[0].sample_id.startswith("calibration-"):
            assert validation_calls == expected
        return original(rows, *args, **kwargs)

    monkeypatch.setattr(pipeline, "group_curves", tracked)
    pipeline.run_study(partitions())
