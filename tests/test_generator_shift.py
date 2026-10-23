from types import SimpleNamespace

import numpy as np

from train.evaluate_generator_shift import leave_one_domain_out
from train import fit_fusion_correlations
from train.fit_fusion_correlations import fit_correlations


def test_learned_error_correlations_handle_missing_scores():
    rng = np.random.default_rng(21)
    y = rng.integers(0, 2, 400)
    signal = (2 * y - 1) * 0.5 + rng.normal(0, 0.2, len(y))
    scores = {
        "neural": np.clip(0.5 + signal, 0.0, 1.0),
        "temporal": np.clip(0.5 + signal + rng.normal(0, 0.03, len(y)), 0.0, 1.0),
        "audio": np.clip(rng.uniform(0.0, 1.0, len(y)), 0.0, 1.0),
    }
    scores["temporal"][:80] = np.nan

    report = fit_correlations(scores, y)
    corr = np.asarray(report["correlation"])

    assert report["method"] == "pairwise_ledoit_wolf_residual_covariance"
    neural_index = report["experts"].index("neural")
    temporal_index = report["experts"].index("temporal")
    assert report["pair_sample_counts"][neural_index][temporal_index] == len(y) - 80
    assert np.allclose(corr, corr.T)
    assert np.linalg.eigvalsh(corr).min() > 0
    assert corr[neural_index, temporal_index] > corr[neural_index, report["experts"].index("audio")]
    assert min(report["precision_by_expert"]["neural"],
               report["precision_by_expert"]["temporal"]) > report["precision_by_expert"]["audio"]


def test_pipeline_score_collection_keeps_late_experts_aligned(monkeypatch):
    class FakePipeline:
        def __init__(self):
            self.index = 0

        def analyze(self, *args, **kwargs):
            evidence = [{"name": "established",
                         "p_fake_raw": 0.1 + self.index * 0.1}]
            if self.index:
                evidence.append({"name": "late_expert", "p_fake_raw": 0.8})
            self.index += 1
            return SimpleNamespace(evidence=evidence)

    fake_pipe = FakePipeline()
    monkeypatch.setattr("detector.pipeline.reset_pipeline", lambda: None)
    monkeypatch.setattr("detector.pipeline.get_pipeline", lambda: fake_pipe)
    items = [SimpleNamespace(path=f"media_{i}.png", label=i % 2, media_type="image")
             for i in range(6)]

    scores, labels = fit_fusion_correlations.collect_scores_from_pipeline(items, "image")

    assert labels.tolist() == [0, 1, 0, 1, 0, 1]
    assert np.allclose(scores["established"], np.arange(1, 7) / 10.0)
    assert np.isnan(scores["late_expert"][0])
    assert np.allclose(scores["late_expert"][1:], 0.8)


def test_leave_one_domain_out_reports_selective_and_conformal_metrics():
    rows = []
    for domain_index, domain in enumerate(("flux", "sd3", "social_reupload")):
        rng = np.random.default_rng(domain_index)
        for y in (0, 1):
            for _ in range(40):
                rows.append({
                    "domain": domain,
                    "y": y,
                    "logit": (2 * y - 1) * 2.0 + rng.normal(0.0, 0.6),
                })

    report = leave_one_domain_out(
        rows, ("flux", "sd3", "social_reupload"), alpha=0.1)

    assert report["protocol"] == "leave_one_domain_out_calibration"
    assert report["model_weights_retrained_per_fold"] is False
    assert set(report["folds"]) == {"flux", "sd3", "social_reupload"}
    for fold in report["folds"].values():
        assert fold["n"] == 80
        assert fold["calibration_n"] == 160
        assert len(fold["risk_coverage"]["coverage"]) == fold["n"]
        assert 0.0 <= fold["conformal"]["empirical_coverage"] <= 1.0
        assert {"abstain_rate", "accuracy_answered"} <= set(fold["selective_decisions"])
