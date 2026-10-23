import numpy as np

from detector.evidence_graph import gls_fuse
from detector.fusion import Evidence
from detector.investigation_controller import (InvestigationController, TestSpec,
                                               expected_benefit)


def E(name, z, r=1.0, **kw):
    return Evidence(name, z, r, **kw)


def registry():
    return [TestSpec("neural_probe", 0.5, lambda c: 0.9),   # correlated with neural
            TestSpec("audio", 0.2, lambda c: 0.9),           # independent family
            TestSpec("tamper", 0.4, lambda c: 0.5, heuristic=True),
            TestSpec("spectral", 0.05, lambda c: 0.0,
                     available=lambda c: False, unavailable_reason="no trained head")]


def test_expected_benefit_prefers_independent_tests_and_is_nonnegative():
    ev = [E("neural", 0.6)]
    b_corr = expected_benefit(ev, "neural_probe", 0.9)
    b_ind = expected_benefit(ev, "audio", 0.9)
    assert b_ind["d_risk"] > b_corr["d_risk"] >= 0
    assert b_ind["d_entropy"] > b_corr["d_entropy"] >= 0


def test_no_benefit_when_already_certain():
    ev = [E("neural", 9.0)]
    b = expected_benefit(ev, "audio", 0.9)
    assert b["d_risk"] < 1e-6


def test_voi_prefers_cheap_independent_test_first():
    c = InvestigationController(registry(), risk_target=0.001, budget=2.0)
    ranked = c.rank([E("neural", 0.6)], {}, 2.0)
    assert ranked[0]["test"] == "audio"
    assert [r for r in ranked if r["test"] == "spectral"][0]["eligible"] is False


def test_stops_when_risk_target_met_and_records_not_run():
    c = InvestigationController(registry(), risk_target=0.05)
    out = c.run([E("neural", 6.0)], {}, lambda n: E(n, 5.0))
    assert out["steps"] == [] and out["stop_reason"] == "risk_target_met"
    assert {n["test"] for n in out["not_run"]} == {"neural_probe", "audio", "tamper", "spectral"}


def test_runs_until_resolved_within_budget_and_reduces_risk():
    c = InvestigationController(registry(), risk_target=0.02, budget=0.7, min_utility=0.0)
    out = c.run([E("neural", 0.5)], {}, lambda n: E(n, 2.5))
    assert out["steps"] and out["spent"] <= 0.7 + 1e-9
    risks = [s["risk_after"] for s in out["steps"]]
    assert risks[-1] < c._risk_now([E("neural", 0.5)])
    assert all("reason" in s and s["utility"] is not None for s in out["steps"])


def test_budget_exhaustion_reason():
    c = InvestigationController(registry(), risk_target=1e-9, budget=0.1, min_utility=0.0)
    out = c.run([E("neural", 0.1)], {}, lambda n: E(n, 0.1))
    assert [s["test"] for s in out["steps"]] == ["audio"] or out["stop_reason"] in (
        "no_eligible_test", "no_test_worth_its_cost")


def test_baseline_policies_and_determinism():
    kw = dict(risk_target=1e-9, budget=10, min_utility=0.0)
    ex = lambda n: E(n, 1.0)
    f = InvestigationController(registry(), policy="fixed", **kw).run([E("neural", 0.2)], {}, ex)
    assert [s["test"] for s in f["steps"]] == ["neural_probe", "audio", "tamper"]
    r1 = InvestigationController(registry(), policy="random", seed=3, **kw).run([E("neural", 0.2)], {}, ex)
    r2 = InvestigationController(registry(), policy="random", seed=3, **kw).run([E("neural", 0.2)], {}, ex)
    assert [s["test"] for s in r1["steps"]] == [s["test"] for s in r2["steps"]]
    q = InvestigationController(registry(), policy="quality", **kw).run(
        [E("neural", 0.2)], {"compression": "high"}, ex)
    assert q["steps"][0]["test"] == "neural_probe"
    try:
        InvestigationController(registry(), policy="bogus")
        assert False
    except ValueError:
        pass
