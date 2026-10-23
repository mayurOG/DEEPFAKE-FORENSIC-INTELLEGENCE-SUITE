import numpy as np

from detector import metrics as MT
from detector.calibration import fit_calibrator
from detector.conditional_reliability import (ConditionalCalibrator, ReliabilityModel,
                                              fit_conditional, group_coverage, group_key,
                                              load_conditional, save_conditional)


def _make(n, noise, comp, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    z = (2 * y - 1) * 2.0 + rng.normal(0, noise, n)
    return z, y, [{"compression": comp, "n_used": 3, "min_side": 300}] * n


def _mix(seed):
    z1, y1, c1 = _make(2400, 1.0, "low", seed)       # easy, clean media
    z2, y2, c2 = _make(1200, 3.2, "high", seed + 1)  # hard, compressed media
    return np.r_[z1, z2], np.r_[y1, y2], c1 + c2


def test_conditional_conformal_restores_coverage_in_hard_group():
    z, y, c = _mix(0)
    glob = fit_calibrator(z, y, alpha=0.1)
    cond = fit_conditional(z, y, c, alpha=0.1, min_group_n=100)
    zt, yt, ct = _mix(10)
    g_cov = group_coverage(glob, zt, yt, ct)
    c_cov = group_coverage(cond, zt, yt, ct)
    hard = group_key({"compression": "high", "n_used": 3, "min_side": 300})
    assert hard in cond.groups
    # The global guarantee is a marginal one: it under-covers the hard group
    assert g_cov[hard]["coverage"] < 0.9
    assert c_cov[hard]["coverage"] >= 0.86
    assert c_cov[hard]["coverage"] > g_cov[hard]["coverage"]
    # ... by abstaining more there
    assert c_cov[hard]["abstain"] > g_cov[hard]["abstain"]


def test_small_group_falls_back_to_global():
    z, y, c = _mix(0)
    cond = fit_conditional(z, y, c, alpha=0.1, min_group_n=100)
    odd = {"compression": "medium", "n_used": 1, "min_side": 100}
    assert cond.which_group(odd) == "global"
    p = float(cond.prob_fake(1.0))
    assert cond.label_set(p, odd) == cond.base.label_set(p)


def test_persistence_roundtrip(tmp_path):
    z, y, c = _mix(0)
    cond = fit_conditional(z, y, c, min_group_n=100)
    rm = ReliabilityModel().fit(z, y, c)
    path = str(tmp_path / "cc.json")
    save_conditional(path, cond, {"neural": rm})
    cond2, rel2 = load_conditional(path)
    ctx = {"compression": "high", "n_used": 3, "min_side": 300}
    assert cond2.label_set(0.5, ctx) == cond.label_set(0.5, ctx)
    assert abs(rel2["neural"].reliability(ctx) - rm.reliability(ctx)) < 1e-9
    assert load_conditional(str(tmp_path / "nope.json")) == (None, {})


def test_reliability_model_learns_context_dependence():
    z, y, c = _mix(0)
    rm = ReliabilityModel().fit(z, y, c)
    clean = rm.reliability({"compression": "low", "n_used": 3, "min_side": 300})
    bad = rm.reliability({"compression": "high", "n_used": 3, "min_side": 300})
    assert rm.fitted and clean > bad
    assert 0 <= bad <= clean <= 1
    assert ReliabilityModel().fit([1, 2], [1, 0], [{}, {}]).reliability({}) == 1.0  # too little data


def test_risk_coverage_and_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    good = np.clip(0.5 + (2 * y - 1) * 0.35 + rng.normal(0, 0.2, 600), 0.01, 0.99)
    junk = rng.uniform(0, 1, 600)
    assert MT.aurc(y, good) < MT.aurc(y, junk)
    assert MT.coverage_at_risk(y, good, 0.05) > MT.coverage_at_risk(y, junk, 0.05)
    c = MT.risk_coverage_curve(y, good)
    assert c["coverage"][-1] == 1.0 and (c["risk"] >= 0).all()
    ci = MT.bootstrap_ci(MT.auc, y, good, n_boot=200)
    assert ci["lo"] <= ci["value"] <= ci["hi"]
    d = MT.paired_bootstrap_diff(MT.auc, y, good, junk, n_boot=200)
    assert d["lo"] > 0 and d["p_le_zero"] < 0.05
    assert MT.nll(y, good) < MT.nll(y, junk)
