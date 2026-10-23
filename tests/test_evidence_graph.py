import math

import numpy as np

from detector.evidence_graph import (build_graph, decision_risk, estimate_correlation,
                                     gls_fuse, prior_correlation)
from detector.fusion import Evidence, fuse, skipped


def E(name, z, r=1.0, **kw):
    return Evidence(name, z, r, **kw)


def test_reduces_to_weighted_mean_when_independent():
    ev = [E("neural", 2.0), E("audio", 1.0), E("tamper", -0.5, heuristic=True)]
    g = gls_fuse(ev, use_dependency=False)
    f = fuse(ev, disagreement_penalty=0.0)
    assert abs(g["logit"] - f.logit) < 1e-6


def test_duplicate_evidence_is_not_double_counted():
    one = gls_fuse([E("neural", 3.0)])
    dup = gls_fuse([E("neural", 3.0), E("temporal", 3.0)])   # rho 0.85
    indep = gls_fuse([E("neural", 3.0), E("audio", 3.0)])    # rho 0.1
    assert dup["n_eff"] < indep["n_eff"]
    assert dup["sd"] > indep["sd"]            # correlated agreement is weaker confirmation
    assert indep["sufficiency"] >= dup["sufficiency"]
    assert one["n_eff"] == 1.0


def test_disagreement_inflates_uncertainty():
    agree = gls_fuse([E("neural", 3.0), E("audio", 3.0)])
    clash = gls_fuse([E("neural", 3.0), E("audio", -3.0)])
    assert clash["birge"] > agree["birge"] and clash["sufficiency"] < agree["sufficiency"]


def test_negative_gls_weights_are_dropped():
    # a weak expert highly correlated with a strong one would get a negative weight
    ev = [E("neural", 3.0, 1.0), E("temporal", 0.0, 0.2)]
    g = gls_fuse(ev)
    assert all(w >= 0 for w in g["weights"].values())


def test_empty_and_skipped():
    assert gls_fuse([])["n_used"] == 0
    assert gls_fuse([skipped("sync", "x")])["n_used"] == 0


def test_graph_counterfactuals_and_contradictions():
    ev = [E("neural", 4.0), E("audio", -2.5), E("tamper", 0.2, 0.3, heuristic=True),
          skipped("rppg", "no video", True)]
    g = build_graph(ev, quality={"compression": "high"})
    d = g.to_dict()
    kinds = {n["kind"] for n in d["nodes"]}
    assert {"conclusion", "observation", "detector", "unobserved", "quality"} <= kinds
    assert any(c["experts"] == ["neural", "audio"] for c in d["contradictions"])
    top = d["counterfactuals"][0]
    assert top["removed"] in ("neural", "audio") and top["delta"] is not None
    assert "digraph" in g.to_dot()
    # removing the only supporter of "fake" can flip the call
    lone = build_graph([E("neural", 1.0), E("audio", -0.9)]).to_dict()
    assert any(c["flips_decision"] for c in lone["counterfactuals"])


def test_correlation_estimate_and_prior():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    base = (2 * y - 1) * 1.0 + rng.normal(0, 1, 400)
    Z = np.c_[base + rng.normal(0, 0.2, 400), base + rng.normal(0, 0.2, 400),
              rng.normal(0, 1, 400)]
    c = estimate_correlation(Z, ["neural", "audio", "tamper"], y)
    assert c["neural"]["audio"] > c["neural"]["tamper"]
    assert prior_correlation("neural", "temporal") > prior_correlation("neural", "audio")
    fused = gls_fuse([E("neural", 2.0), E("audio", 2.0)], corr=c)
    assert fused["n_eff"] < 1.6


def test_decision_risk():
    assert decision_risk(0.0, 1.0) == 0.5
    assert decision_risk(5.0, 1.0) < 1e-5
    assert decision_risk(1.0, math.inf) == 0.5
