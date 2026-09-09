import numpy as np

from paper.tools.progressive_mechanism_demo import diagnose


def test_reference_free_ambiguity_is_reported_without_performance_claim():
    result = diagnose()
    assert result["evidence_kind"] == "synthetic_mechanism_diagnostic_not_research_benefit"
    assert result["same_lr_and_predictions"] is True
    assert result["same_features"] is True
    assert result["faithful_texture"]["risk_max"] == 0
    assert result["invented_texture"]["risk_max"] == 0.25
    assert result["faithful_texture"]["mean_texture_score"] > 0
    assert result["faithful_texture"]["mean_uncertainty_score"] == 0
    assert result["faithful_texture"]["mean_lr_residual"] == 0


def test_joint_variance_differs_from_mean_per_sample_spatial_variance():
    result = diagnose()["spatially_constant_distinct_samples"]
    assert result["joint_score_mean"] > 0
    assert abs(result["mean_per_sample_spatial_variance"]) < 1e-12
    np.testing.assert_allclose(
        result["joint_score_mean"], result["between_sample_variance"], atol=1e-12, rtol=0
    )
