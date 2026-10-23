"""Smoke tests: the evaluation protocols run end-to-end on synthetic folders
with a stub backbone. They test the machinery (no leakage guard, outputs,
paired CIs), NOT the scientific claims, which need real data."""
import json
import os

import numpy as np
import pytest

from detector import models as M
from tests.conftest import make_img, stub_image_fn
from train import evaluate_ablation, evaluate_adaptive, evaluate_robustness
from train import fit_calibration


def mkset(root, seed0, n=30, size=160, noise_shift=0):
    for lab, (r, b) in {"real": (-14, 14), "fake": (14, -14)}.items():
        (root / lab).mkdir(parents=True)
        for i in range(n):
            make_img(r + noise_shift, 0, b, seed=seed0 + i + (1000 if lab == "fake" else 0),
                     size=size).save(root / lab / f"{lab}_{i}.png")
    return str(root)


@pytest.fixture
def splits(tmp_path):
    return (mkset(tmp_path / "calib", 0), mkset(tmp_path / "test", 5000))


@pytest.fixture(autouse=True)
def stub():
    M.set_backends(image=M.StubBackend(stub_image_fn), face=M.StubFaceDetector())
    yield
    M.reset_backends()


def test_ablation_runs_and_reports_paired_differences(splits, tmp_path):
    calib, test = splits
    out = str(tmp_path / "abl.json")
    res = evaluate_ablation.main(["--calib", calib, "--test", test, "--out", out, "--boot", "50"])
    assert set(res) == set(evaluate_ablation.CONFIGS)
    a = res["A_existing"]
    assert a["n"] == 60 and 0 <= a["auroc"]["lo"] <= a["auroc"]["value"] <= a["auroc"]["hi"] <= 1
    for k in ("brier", "ece", "decisions", "aurc", "mean_expert_calls", "latency_p95_s"):
        assert k in a
    e = res["E_full"]
    assert "vs_A_existing" in e and {"auroc", "aurc_lower_is_better"} <= set(e["vs_A_existing"])
    assert res["F_full_no_controller"]["flags"]["controller"] is False
    assert res["G_full_independent"]["flags"]["independent_experts"] is True
    json.load(open(out))


def test_ablation_refuses_overlapping_splits(splits):
    calib, _ = splits
    with pytest.raises(SystemExit):
        evaluate_ablation.main(["--calib", calib, "--test", calib])


def test_adaptive_compares_policies_at_each_budget(splits, tmp_path):
    calib, test = splits
    res = evaluate_adaptive.main(["--calib", calib, "--test", test, "--budgets", "0.2", "1.0",
                                  "--boot", "50", "--max-items", "20"])
    assert "fixed_cascade" in res
    for b in ("0.2", "1.0"):
        for pol in evaluate_adaptive.POLICIES:
            r = res[f"{pol}@{b}"]
            assert r["flags"]["controller_policy"] == pol and r["flags"]["budget"] == float(b)
            assert "vs_fixed_cascade" in r and "expert_calls_diff" in r["vs_fixed_cascade"]
    # a smaller budget can never use more experts than a larger one
    assert res["voi@0.2"]["mean_expert_calls"] <= res["voi@1.0"]["mean_expert_calls"] + 1e-9


def test_robustness_matrix_and_failure_prediction(splits):
    calib, test = splits
    res = evaluate_robustness.main(["--calib", calib, "--test", test, "--boot", "50",
                                    "--max-items", "15"])
    assert "clean" in res["matrix"] and "jpeg_q35" in res["matrix"]
    assert 0 <= res["matrix"]["social_pipeline"]["verdict_change_rate"] <= 1
    fp = res["failure_prediction_auroc"]
    assert {"fragility", "classifier_uncertainty", "insufficiency"} <= set(fp)
    assert {"aurc_own_confidence", "aurc_combined"} <= set(res["selective_risk"])
    g = res["robustness_gate"]
    assert g["after"]["abstain_rate"] >= g["before"]["abstain_rate"]


def test_fit_calibration_conditional_writes_artifacts(splits, tmp_path, monkeypatch):
    calib, _ = splits
    art = str(tmp_path / "art")
    monkeypatch.setenv("DEEPFAKE_ARTIFACT_DIR", art)
    fit_calibration.main(["--data", calib, "--media", "image", "--conditional"])
    for f in ("calibration.json", "conditional_image.json", "expert_correlation.json"):
        assert os.path.exists(os.path.join(art, f)), f
    rep = json.load(open(os.path.join(art, "calibration_report_image.json")))
    assert "conditional" in rep and rep["conditional"]["experts"]
    # the pipeline picks the fitted artifacts up
    import dataclasses
    from detector.config import get_settings
    from detector.pipeline import Pipeline
    p = Pipeline(settings=dataclasses.replace(get_settings(), artifact_dir=art, conditional=True))
    assert "image" in p.conditional or p.status()["conditional_calibration"] == []
    assert p.expert_corr is not None
