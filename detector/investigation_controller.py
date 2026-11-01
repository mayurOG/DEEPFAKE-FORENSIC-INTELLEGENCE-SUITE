"""Value-of-information investigation controller.

Instead of running every expert on every file, choose the next test by

    U(a | G) = ( E[dR | a, G] + lambda * E[dH | a, G] ) / ( C(a) + eps )
    a* = argmax  U(a | G)   subject to  C(a) <= remaining budget

where G is the current evidence graph, dR the expected reduction in decision
risk, dH the expected reduction in uncertainty (entropy), and C the cost.

Model (stated so it can be criticised): the current dependency-aware fusion
gives a Gaussian belief N(mu, v) about the common log-odds. A candidate
expert has a prior precision (weight x expected reliability under the
current media quality) and an error correlation with the evidence already
collected. Adding it changes the variance to v' (generalised least squares,
so a highly correlated test is worth little, which is exactly the "two
visual models share a failure mode" case). The preposterior distribution of
the updated mean is N(mu, v - v'), so

    E[R'] = E_{mu'}[ Phi(-|mu'| / sqrt(v')) ]       (Gauss-Hermite quadrature)
    dR    = Phi(-|mu| / sqrt(v)) - E[R']
    dH    = 0.5 * ln(v / v')

No labels or future outputs are used: the expected benefit depends only on
the *stated precision* and correlation of each test. Whether those stated
precisions are right is an empirical question that `train.evaluate_adaptive`
measures against fixed-order, random and quality-only baselines.

Stopping: the risk target is met, the budget is exhausted, or no available
test has utility above ``min_utility`` (reported as the reason, so an
unresolved case can be an explicit Inconclusive rather than a silent guess).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from detector.evidence_graph import (FAMILY, _rho_matrix, _used, decision_risk,
                                     gls_fuse)
from detector.fusion import DEFAULT_WEIGHTS, Evidence

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(21)  # probabilists' Hermite
_GH_W = _GH_W / _GH_W.sum()


@dataclass
class CostProfile:
    """Hardware-aware cost model: lambda_t * T_e + lambda_m * M_e + lambda_c * C_e."""
    lambda_t: float = 1.0    # weight on inference latency (seconds)
    lambda_m: float = 0.001  # weight on memory footprint (MB)
    lambda_c: float = 0.1    # weight on computational intensity / GPU weight

    def effective_cost(
        self,
        cost: float,
        time_s: Optional[float] = None,
        memory_mb: Optional[float] = None,
        compute: Optional[float] = None,
    ) -> float:
        t = time_s if time_s is not None else cost
        m = memory_mb if memory_mb is not None else 0.0
        c = compute if compute is not None else 1.0
        hw_cost = self.lambda_t * t + self.lambda_m * m + self.lambda_c * c
        return max(hw_cost, 1e-4)


@dataclass
class TestSpec:
    """A runnable analysis the controller may choose."""

    __test__ = False  # not a pytest class
    name: str                      # also the evidence name it produces
    cost: float                    # relative compute cost (seconds, or any unit)
    reliability: Callable[[Dict[str, Any]], float] = lambda ctx: 0.6
    available: Callable[[Dict[str, Any]], bool] = lambda ctx: True
    unavailable_reason: str = "not applicable"
    heuristic: bool = False
    # Hardware-aware cost attributes
    time_s: Optional[float] = None
    memory_mb: Optional[float] = None
    compute_intensity: Optional[float] = None
    modality: str = "image"        # image | video | audio
    instability: float = 0.0       # transformation sensitivity in [0, 1]

    def get_cost(self, profile: Optional[CostProfile] = None) -> float:
        if profile is None:
            return self.cost
        return profile.effective_cost(
            self.cost, self.time_s, self.memory_mb, self.compute_intensity
        )


def _gls_var(names: Sequence[str], prec: Sequence[float],
             corr: Optional[Dict[str, Dict[str, float]]]) -> float:
    prec = np.maximum(np.asarray(prec, dtype=np.float64), 1e-6)
    sigma = 1.0 / np.sqrt(prec)
    R = _rho_matrix(list(names), corr)
    S = R * np.outer(sigma, sigma) + 1e-9 * np.eye(len(names))
    ones = np.ones(len(names))
    return float(1.0 / float(ones @ np.linalg.solve(S, ones)))


def expected_benefit(
    evidences: Sequence[Evidence], cand_name: str, cand_precision: float,
    weights: Optional[Dict[str, float]] = None,
    corr: Optional[Dict[str, Dict[str, float]]] = None,
    lam: float = 0.3,
    instability_penalty: float = 0.0,
) -> Dict[str, float]:
    """E[dR] and E[dH] for adding one more test with the given precision."""
    cur = gls_fuse(evidences, weights, corr)
    used = _used(evidences)
    if not used:
        return {"d_risk": 0.0, "d_entropy": 0.0, "risk_before": 0.5, "risk_after": 0.5}
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    names = [e.name for e in used]
    prec = [max(w.get(e.name, 0.3), 0.0) * e.reliability for e in used]
    # Penalize experts whose predictions are unstable under transformations
    penalty_factor = max(1.0 - 0.5 * float(np.clip(instability_penalty, 0.0, 0.9)), 0.1)
    cand_precision = cand_precision * penalty_factor

    v_old = _gls_var(names, prec, corr)
    v_new = _gls_var(names + [cand_name], prec + [cand_precision], corr)
    v_new = min(v_new, v_old)
    birge = cur["birge"]
    s_old, s_new = math.sqrt(v_old * birge), math.sqrt(max(v_new, 1e-12) * birge)
    mu = cur["logit"]
    r_before = decision_risk(mu, s_old)
    spread = math.sqrt(max(v_old - v_new, 0.0) * birge)
    mu_prime = mu + spread * _GH_X
    r_after = float(sum(wi * decision_risk(m, s_new) for wi, m in zip(_GH_W, mu_prime)))
    return {"d_risk": max(r_before - r_after, 0.0),
            "d_entropy": 0.5 * math.log(v_old / max(v_new, 1e-12)),
            "risk_before": r_before, "risk_after": r_after}


@dataclass
class Step:
    test: str
    reason: str
    utility: Optional[float]
    d_risk: Optional[float]
    d_entropy: Optional[float]
    cost: float
    risk_before: Optional[float] = None
    risk_after: Optional[float] = None
    logit: Optional[float] = None
    realized_d_risk: Optional[float] = None
    actually_improved: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


QUALITY_ORDER = {  # static, quality-only routing baseline
    "low": ["spectral", "tamper", "neural_probe", "forensic_prior"],
    "medium": ["neural_probe", "tamper", "spectral", "forensic_prior"],
    "high": ["neural_probe", "forensic_prior", "tamper", "spectral"],
}


class InvestigationController:
    def __init__(
        self, registry: Sequence[TestSpec], lam: float = 0.3, eps: float = 0.05,
        risk_target: float = 0.05, min_utility: float = 0.005, budget: float = 1.0,
        policy: str = "voi", weights: Optional[Dict[str, float]] = None,
        corr: Optional[Dict[str, Dict[str, float]]] = None, seed: int = 0,
        fixed_order: Optional[Sequence[str]] = None,
        cost_profile: Optional[CostProfile] = None,
        min_evidence_count: int = 1,
        learned_reliability: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        if policy not in ("voi", "fixed", "random", "quality"):
            raise ValueError("policy must be voi | fixed | random | quality")
        self.registry = {t.name: t for t in registry}
        self.lam, self.eps = lam, eps
        self.risk_target, self.min_utility, self.budget = risk_target, min_utility, budget
        self.policy, self.weights, self.corr = policy, weights, corr
        self.fixed_order = list(fixed_order) if fixed_order else [t.name for t in registry]
        self._rng = random.Random(seed)
        self.cost_profile = cost_profile or CostProfile()
        self.min_evidence_count = min_evidence_count
        self.learned_reliability = learned_reliability or self._load_learned_reliability()

    def _load_learned_reliability(self) -> Dict[str, Dict[str, float]]:
        """Load empirical validation reliability if saved in artifacts."""
        try:
            from detector.config import get_settings
            p = os.path.join(get_settings().artifact_dir, "expert_reliability.json")
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    # -- scoring -------------------------------------------------------------
    def rank(self, evidences: Sequence[Evidence], ctx: Dict[str, Any],
             remaining: float, done: Sequence[str] = ()) -> List[Dict[str, Any]]:
        w = dict(DEFAULT_WEIGHTS)
        if self.weights:
            w.update(self.weights)
        instability_map = ctx.get("instability", {})
        rows = []
        for name, t in self.registry.items():
            if name in done:
                continue
            if not t.available(ctx):
                rows.append({"test": name, "eligible": False, "why": t.unavailable_reason})
                continue
            effective_cost = t.get_cost(self.cost_profile)
            if effective_cost > remaining + 1e-12:
                rows.append({"test": name, "eligible": False, "why": "over budget"})
                continue

            inst = float(instability_map.get(name, t.instability))
            base_rel = float(np.clip(t.reliability(ctx), 0.0, 1.0))
            if self.learned_reliability and name in self.learned_reliability.get(t.modality, {}):
                prec = float(self.learned_reliability[t.modality][name])
            else:
                prec = w.get(name, 0.3) * base_rel

            b = expected_benefit(
                evidences, name, prec, self.weights, self.corr, self.lam,
                instability_penalty=inst,
            )
            # Hardware-aware Value of Information
            u = (b["d_risk"] + self.lam * b["d_entropy"]) / (effective_cost + self.eps)
            rows.append({"test": name, "eligible": True, "utility": u, "cost": effective_cost,
                         "precision": prec, "instability": inst, **b})
        elig = sorted([r for r in rows if r["eligible"]], key=lambda r: (-r["utility"], r["test"]))
        return elig + [r for r in rows if not r["eligible"]]

    def _risk_now(self, evidences) -> float:
        f = gls_fuse(evidences, self.weights, self.corr)
        return decision_risk(f["logit"], f["sd"])

    def choose(self, evidences: Sequence[Evidence], ctx: Dict[str, Any],
               remaining: float, done: Sequence[str]) -> Tuple[Optional[str], str, Optional[Dict]]:
        used_count = len(_used(evidences))
        if used_count >= self.min_evidence_count and self._risk_now(evidences) <= self.risk_target:
            return None, "risk_target_met", None
        ranked = self.rank(evidences, ctx, remaining, done)
        elig = [r for r in ranked if r["eligible"]]
        if not elig:
            return None, "no_eligible_test", None
        if self.policy == "voi":
            best = elig[0]
            if best["utility"] < self.min_utility:
                return None, "no_test_worth_its_cost", best
            return best["test"], "highest_expected_value_per_cost", best
        by_name = {r["test"]: r for r in elig}
        if self.policy == "random":
            name = self._rng.choice(sorted(by_name))
            return name, "random_baseline", by_name[name]
        order = (self.fixed_order if self.policy == "fixed"
                 else QUALITY_ORDER.get(ctx.get("compression", "medium"), QUALITY_ORDER["medium"]))
        for name in order:
            if name in by_name:
                return name, f"{self.policy}_order_baseline", by_name[name]
        return None, "no_eligible_test", None

    # -- loop ----------------------------------------------------------------
    def run(self, evidences: Sequence[Evidence], ctx: Dict[str, Any],
            executor: Callable[[str], Evidence]) -> Dict[str, Any]:
        ev = list(evidences)
        done: List[str] = []
        steps: List[Step] = []
        spent, stop = 0.0, "risk_target_met"
        while True:
            name, why, info = self.choose(ev, ctx, self.budget - spent, done)
            if name is None:
                stop = why
                break
            t = self.registry[name]
            cost_step = t.get_cost(self.cost_profile)
            risk_before = self._risk_now(ev)
            result = executor(name)
            done.append(name)
            spent += cost_step
            ev.append(result)
            risk_after = self._risk_now(ev)
            realized_d = max(risk_before - risk_after, 0.0)
            improved = bool(risk_after < risk_before - 1e-6)

            steps.append(Step(
                test=name, reason=why, cost=cost_step,
                utility=None if info is None else info.get("utility"),
                d_risk=None if info is None else info.get("d_risk"),
                d_entropy=None if info is None else info.get("d_entropy"),
                risk_before=risk_before,
                risk_after=risk_after,
                logit=None if result.logit is None else float(result.logit),
                realized_d_risk=realized_d,
                actually_improved=improved,
            ))
            if len(done) >= len(self.registry):
                stop = "all_tests_run"
                break
        not_run = []
        for name, t in self.registry.items():
            if name not in done:
                reason = (t.unavailable_reason if not t.available(ctx)
                          else {"risk_target_met": "not needed: risk target met",
                                "no_test_worth_its_cost": "expected value below its cost",
                                "no_eligible_test": "budget exhausted",
                                }.get(stop, stop))
                not_run.append({"test": name, "reason": reason})
        improved_count = sum(1 for s in steps if s.actually_improved)
        return {"evidence": ev, "steps": [s.to_dict() for s in steps],
                "not_run": not_run, "stop_reason": stop, "spent": spent,
                "final_risk": self._risk_now(ev), "risk_target": self.risk_target,
                "policy": self.policy, "budget": self.budget,
                "improved_count": improved_count,
                "efficiency_ratio": (improved_count / len(steps)) if steps else 1.0,
                "min_evidence_count": self.min_evidence_count}
