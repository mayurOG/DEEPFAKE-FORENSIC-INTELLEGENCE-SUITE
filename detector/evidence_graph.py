"""Counterfactual forensic evidence graph and dependency-aware fusion.

Ordinary fusion averages expert scores as if each were independent. Two
visual models that look at the same pixels, or a temporal head built on the
same frame logits, share failure modes, so agreement between them is weaker
confirmation than agreement between unrelated experts.

This module makes that dependency explicit:

* **Graph.** Observation nodes (one per expert output), detector nodes, a
  quality node, and a conclusion node. Edges: ``supports`` /
  ``contradicts`` (observation -> conclusion), ``depends_on`` (expert pairs
  with a high prior error correlation), ``degrades`` (quality -> expert).
* **Dependency-aware fusion.** Generalised least squares for a common
  log-odds with a correlated error covariance. With zero correlation it
  reduces exactly to the existing precision-weighted mean, so it is a strict
  generalisation, and duplicated evidence is not double counted. Experts
  that would receive a negative weight (a known GLS artefact under very high
  correlation) are dropped and the estimate recomputed.
* **Counterfactuals.** Leave-one-out refits show which observations really
  change the decision, rather than just listing scores.
* **Evidence sufficiency.** A z-statistic for "the combined log-odds
  differs from zero", with a Birge-ratio inflation when experts disagree
  more than their stated precision allows. This is deliberately separate
  from the classifier probability: strong classifier confidence from
  little, correlated evidence is *not* sufficient evidence.

The correlation prior below is a hypothesis, not a measurement; fit it from
held-out expert logits with :func:`estimate_correlation` and pass it in.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from detector.calibration import sigmoid
from detector.fusion import DEFAULT_WEIGHTS, Evidence

FAMILY: Dict[str, str] = {
    "neural": "visual", "neural_probe": "visual", "temporal": "visual",
    "siglip_semantic": "visual",
    "spectral": "lowlevel", "tamper": "lowlevel", "localization": "lowlevel",
    "forensic_prior": "lowlevel",
    "audio": "audio", "sync": "audiovisual", "rppg": "physio", "geometry": "physio",
    "metadata_ai": "provenance",
}
_SAME_FAMILY = 0.6
_PAIR_PRIOR = {frozenset(("neural", "temporal")): 0.85,
               frozenset(("neural", "neural_probe")): 0.55,
               frozenset(("audio", "sync")): 0.3,
               frozenset(("tamper", "forensic_prior")): 0.4,
               frozenset(("tamper", "localization")): 0.95,
               frozenset(("neural", "siglip_semantic")): 0.5}
_CROSS = 0.1


def prior_correlation(a: str, b: str) -> float:
    """Prior correlation of two experts' errors (1 for the same expert)."""
    if a == b:
        return 1.0
    if "provenance" in (FAMILY.get(a), FAMILY.get(b)):
        return 0.0
    pair = frozenset((a, b))
    if pair in _PAIR_PRIOR:
        return _PAIR_PRIOR[pair]
    fa, fb = FAMILY.get(a), FAMILY.get(b)
    return _SAME_FAMILY if fa is not None and fa == fb else _CROSS


def estimate_correlation(Z: np.ndarray, names: Sequence[str], y: np.ndarray,
                         shrink: float = 0.0) -> Dict[str, Dict[str, float]]:
    """Estimate expert-error correlations from calibration logits.

    ``Z`` is (n, k) with NaN where an expert did not run. Pairwise Ledoit-Wolf
    covariance shrinkage stabilizes measured correlations; the hand-set
    prior is used only when fewer than ten jointly observed calibration rows
    are available. ``shrink`` optionally adds shrinkage toward independence.
    """
    from sklearn.covariance import LedoitWolf

    Z = np.asarray(Z, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64).reshape(-1)
    if Z.ndim != 2 or Z.shape != (len(labels), len(names)):
        raise ValueError("Z must have shape (len(y), len(names))")
    if not 0.0 <= shrink <= 1.0:
        raise ValueError("shrink must be between 0 and 1")
    P = sigmoid(Z)
    E = P - labels[:, None]
    out: Dict[str, Dict[str, float]] = {n: {} for n in names}
    measured = np.eye(len(names), dtype=np.float64)
    pair_measured = np.eye(len(names), dtype=bool)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                out[a][b] = 1.0
                continue
            ok = np.isfinite(E[:, i]) & np.isfinite(E[:, j])
            prior = prior_correlation(a, b)
            if ok.sum() < 10:
                out[a][b] = prior
                continue
            covariance = LedoitWolf().fit(E[ok][:, [i, j]]).covariance_
            denom = math.sqrt(max(covariance[0, 0], 0.0)
                              * max(covariance[1, 1], 0.0))
            r = float(covariance[0, 1] / denom) if denom > 1e-12 else 0.0
            r = float(np.clip((1.0 - shrink) * r, -0.98, 0.98))
            measured[i, j] = measured[j, i] = r
            pair_measured[i, j] = pair_measured[j, i] = True

    measured = _nearest_correlation(measured)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if pair_measured[i, j] and i != j:
                out[a][b] = float(measured[i, j])
    return out


def _nearest_correlation(matrix: np.ndarray) -> np.ndarray:
    matrix = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    matrix = (vectors * np.maximum(values, 1e-6)) @ vectors.T
    scale = np.sqrt(np.maximum(np.diag(matrix), 1e-12))
    matrix = matrix / np.outer(scale, scale)
    np.fill_diagonal(matrix, 1.0)
    matrix = np.clip(matrix, -0.98, 0.98)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _rho_matrix(names: Sequence[str], corr: Optional[Dict[str, Dict[str, float]]]):
    k = len(names)
    R = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            r = (corr or {}).get(names[i], {}).get(names[j])
            if r is None:
                r = prior_correlation(names[i], names[j])
            R[i, j] = R[j, i] = float(np.clip(r, -0.98, 0.98))
    return _nearest_correlation(R)


def _used(evidences: Sequence[Evidence]) -> List[Evidence]:
    # Skip neutral (logit≈0) experts so they do not dilute the fused score,
    # and skip localization whose logit is a duplicate of tamper_evidence.
    return [
        e for e in evidences
        if e.status == "used"
        and e.logit is not None
        and e.reliability > 0
        and abs(float(e.logit)) >= 1e-6
        and e.name != "localization"
    ]


def _precisions(used: Sequence[Evidence], weights: Optional[Dict[str, float]]):
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    return np.array([max(w.get(e.name, 0.3), 0.0) * e.reliability for e in used])


def gls_fuse(
    evidences: Sequence[Evidence],
    weights: Optional[Dict[str, float]] = None,
    corr: Optional[Dict[str, Dict[str, float]]] = None,
    max_abs_logit: float = 12.0,
    use_dependency: bool = True,
) -> Dict[str, Any]:
    """Dependency-aware fused log-odds.

    Returns dict(logit, sd, n_used, n_eff, sufficiency, birge, weights).
    ``use_dependency=False`` forces independence (rho=0) for ablations."""
    used = _used(evidences)
    if not used:
        return {"logit": 0.0, "sd": math.inf, "n_used": 0, "n_eff": 0.0,
                "sufficiency": 0.0, "birge": 1.0, "weights": {}}
    names = [e.name for e in used]
    z = np.array([np.clip(e.logit, -max_abs_logit, max_abs_logit) for e in used])
    prec = np.maximum(_precisions(used, weights), 1e-6)
    sigma = 1.0 / np.sqrt(prec)  # sigma_i^2 = 1 / precision_i
    R = _rho_matrix(names, corr) if use_dependency else np.eye(len(used))
    active = np.ones(len(used), dtype=bool)
    for _ in range(len(used)):
        idx = np.where(active)[0]
        S = (R[np.ix_(idx, idx)] * np.outer(sigma[idx], sigma[idx])
             + 1e-9 * np.eye(len(idx)))
        Sinv1 = np.linalg.solve(S, np.ones(len(idx)))
        denom = float(Sinv1.sum())
        w_act = Sinv1 / denom
        if (w_act < -1e-12).any():
            active[idx[int(np.argmin(w_act))]] = False
            continue
        break
    w_full = np.zeros(len(used))
    w_full[idx] = np.clip(w_act, 0, None)
    mean = float(w_full @ z)
    var = 1.0 / denom
    k = int(active.sum())
    birge = 1.0
    if k > 1:
        resid = z[idx] - mean
        Q = float(resid @ np.linalg.solve(S, resid))
        birge = max(1.0, Q / (k - 1))
    sd = math.sqrt(var * birge)
    p = prec[idx]
    n_eff = float(p.sum() ** 2 / (p @ R[np.ix_(idx, idx)] @ p))
    sufficiency = float(np.clip(abs(mean) / (1.645 * sd), 0.0, 1.0)) if sd > 0 else 1.0
    return {"logit": mean, "sd": sd, "n_used": len(used), "n_eff": n_eff,
            "sufficiency": sufficiency, "birge": birge,
            "weights": {n: round(float(w), 4) for n, w in zip(names, w_full)}}


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------
@dataclass
class EvidenceGraph:
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    fusion: Dict[str, Any] = field(default_factory=dict)
    counterfactuals: List[Dict[str, Any]] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    shared_failure_modes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges, "fusion": self.fusion,
                "counterfactuals": self.counterfactuals,
                "contradictions": self.contradictions,
                "shared_failure_modes": self.shared_failure_modes}

    def to_dot(self) -> str:
        lines = ["digraph evidence {", "  rankdir=LR;"]
        for n in self.nodes:
            lines.append(f'  "{n["id"]}" [label="{n["label"]}"];')
        for e in self.edges:
            lines.append(f'  "{e["src"]}" -> "{e["dst"]}" [label="{e["type"]}"];')
        lines.append("}")
        return "\n".join(lines)


def build_graph(
    evidences: Sequence[Evidence],
    weights: Optional[Dict[str, float]] = None,
    corr: Optional[Dict[str, Dict[str, float]]] = None,
    quality: Optional[Dict[str, Any]] = None,
    contradiction_threshold: float = 0.5,
    influential_delta: float = 0.5,
) -> EvidenceGraph:
    used = _used(evidences)
    full = gls_fuse(evidences, weights, corr)
    g = EvidenceGraph(fusion=full)
    verdict = "fake" if full["logit"] > 0 else "real"
    g.nodes.append({"id": "conclusion", "kind": "conclusion",
                    "label": f"{verdict} (logit {full['logit']:+.2f}, "
                             f"sufficiency {full['sufficiency']:.2f})"})
    if quality:
        g.nodes.append({"id": "quality", "kind": "quality",
                        "label": f"quality: {quality.get('compression', 'unknown')} compression"})
    skipped_ev = [e for e in evidences if e.status != "used"]
    for e in evidences:
        det_id = f"detector:{e.name}"
        g.nodes.append({"id": det_id, "kind": "detector",
                        "label": e.name, "family": FAMILY.get(e.name, "other"),
                        "heuristic": bool(e.heuristic)})
        if e.status != "used":
            g.nodes.append({"id": f"obs:{e.name}", "kind": "unobserved",
                            "label": f"{e.name}: not run ({e.reason})"})
            g.edges.append({"src": det_id, "dst": f"obs:{e.name}", "type": "produced_nothing"})
            continue
        side = "supports" if (e.logit > 0) == (full["logit"] > 0) else "contradicts"
        g.nodes.append({"id": f"obs:{e.name}", "kind": "observation",
                        "label": f"{e.name}: logit {e.logit:+.2f} (rel {e.reliability:.2f})",
                        "logit": round(float(e.logit), 3),
                        "reliability": round(float(e.reliability), 3)})
        g.edges.append({"src": det_id, "dst": f"obs:{e.name}", "type": "produced"})
        g.edges.append({"src": f"obs:{e.name}", "dst": "conclusion", "type": side})
        if quality and e.reliability < 0.95:
            g.edges.append({"src": "quality", "dst": det_id, "type": "degrades"})

    names = [e.name for e in used]
    R = _rho_matrix(names, corr)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if R[i, j] >= 0.5:
                g.edges.append({"src": f"detector:{names[i]}", "dst": f"detector:{names[j]}",
                                "type": "depends_on", "rho": round(float(R[i, j]), 2)})
                g.shared_failure_modes.append(
                    {"experts": [names[i], names[j]], "rho": round(float(R[i, j]), 2)})
            both_strong = (abs(used[i].logit) * used[i].reliability > contradiction_threshold
                           and abs(used[j].logit) * used[j].reliability > contradiction_threshold)
            if both_strong and (used[i].logit > 0) != (used[j].logit > 0):
                g.contradictions.append({
                    "experts": [names[i], names[j]],
                    "logits": [round(float(used[i].logit), 2), round(float(used[j].logit), 2)],
                    "strength": round(float(min(abs(used[i].logit) * used[i].reliability,
                                                abs(used[j].logit) * used[j].reliability)), 3)})
    for k, e in enumerate(used):
        rest = [x for x in used if x is not e]
        if not rest:
            g.counterfactuals.append({"removed": e.name, "logit_without": None,
                                      "delta": None, "flips_decision": None,
                                      "pivotal": True, "note": "only evidence available"})
            continue
        f2 = gls_fuse(rest, weights, corr)
        flips = (f2["logit"] > 0) != (full["logit"] > 0)
        delta = full["logit"] - f2["logit"]
        g.counterfactuals.append({
            "removed": e.name, "logit_without": round(f2["logit"], 3),
            "delta": round(delta, 3), "flips_decision": bool(flips),
            "sufficiency_without": round(f2["sufficiency"], 3),
            "pivotal": bool(flips), "influential": bool(abs(delta) >= influential_delta)})
    g.counterfactuals.sort(key=lambda c: -abs(c["delta"] or 0))
    return g


def decision_risk(logit: float, sd: float) -> float:
    """P(true common log-odds has the opposite sign of the decision)."""
    if not math.isfinite(sd) or sd <= 0:
        return 0.5 if not math.isfinite(sd) else 0.0
    return 0.5 * math.erfc(abs(logit) / (sd * math.sqrt(2.0)))
