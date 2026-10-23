"""Evidence-conditional reliability and calibration.

Global calibration assumes the score-to-risk relationship is the same for
every input. It is not: heavily compressed, tiny or partly-covered media
produce errors the clean calibration set never showed, so a global conformal
guarantee can hold on average and still fail exactly where users are most
likely to be misled.

Two learned components, both fitted only on held-out calibration data:

1. :class:`ReliabilityModel` - per expert, P(expert is correct | context),
   turned into a reliability in [0, 1] as ``clip(2P - 1, 0, 1)`` (0 = no
   better than a coin under that context). Context = media quality,
   resolution, modality coverage, disagreement, out-of-domain ratio, view
   spread. This replaces the hand-set quality factor in fusion.
2. :class:`ConditionalCalibrator` - Platt scaling shared across contexts, but
   class-conditional conformal thresholds fitted *per context group*
   (Mondrian conformal prediction). Groups with too few calibration samples
   fall back to the global thresholds, so the method never claims a
   guarantee it cannot support. Marginal coverage is preserved by the usual
   argument; coverage *within each group* is what the global method lacks.

Whether this actually beats global conformal prediction on real shifts is an
empirical claim: measure it with risk-coverage curves and per-group coverage
(``train.evaluate_ablation``).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from detector.calibration import (FAKE, MODE_FALLBACK_MARGIN, REAL, Calibrator,
                                  fit_calibrator, fit_conformal, sigmoid)

_COMP = {"low": 0.0, "medium": 1.0, "high": 2.0, "unknown": 1.0}
FEATURES = ["compression", "log_min_side", "n_used", "modalities", "disagreement",
            "ood_ratio", "view_spread"]


def context_vector(ctx: Dict[str, Any]) -> np.ndarray:
    side = float(ctx.get("min_side") or 224.0)
    return np.array([
        _COMP.get(str(ctx.get("compression", "unknown")), 1.0),
        math.log(max(side, 16.0)) - math.log(224.0),
        float(ctx.get("n_used", 1)),
        float(ctx.get("modalities", 1)),
        min(float(ctx.get("disagreement", 0.0)), 6.0),
        min(float(ctx.get("ood_ratio") or 0.0), 5.0),
        min(float(ctx.get("view_spread", 0.0)), 6.0),
    ])


def group_key(ctx: Dict[str, Any]) -> str:
    """Coarse, interpretable context group for Mondrian conformal."""
    comp = str(ctx.get("compression", "unknown"))
    cov = "multi" if float(ctx.get("n_used", 1)) >= 3 else "few"
    side = float(ctx.get("min_side") or 224.0)
    res = "small" if side < 224 else "ok"
    return f"{comp}|{cov}|{res}"


# --------------------------------------------------------------------------
@dataclass
class ReliabilityModel:
    """P(correct | context) for one expert."""
    coef: List[float] = field(default_factory=list)
    intercept: float = 0.0
    mean: List[float] = field(default_factory=list)
    scale: List[float] = field(default_factory=list)
    n: int = 0
    fitted: bool = False

    def fit(self, logits: Sequence[float], labels: Sequence[int],
            ctxs: Sequence[Dict[str, Any]], min_n: int = 30, C: float = 1.0):
        from sklearn.linear_model import LogisticRegression
        z = np.asarray(logits, dtype=np.float64)
        y = np.asarray(labels).astype(int)
        correct = ((z > 0).astype(int) == y).astype(int)
        if len(z) < min_n or len(np.unique(correct)) < 2:
            self.fitted, self.n = False, len(z)
            return self
        X = np.stack([context_vector(c) for c in ctxs])
        self.mean, self.scale = X.mean(0).tolist(), (X.std(0) + 1e-6).tolist()
        clf = LogisticRegression(C=C, max_iter=1000)
        clf.fit((X - self.mean) / self.scale, correct)
        self.coef, self.intercept = clf.coef_[0].tolist(), float(clf.intercept_[0])
        self.n, self.fitted = len(z), True
        return self

    def p_correct(self, ctx: Dict[str, Any]) -> float:
        x = (context_vector(ctx) - np.array(self.mean)) / np.array(self.scale)
        return float(sigmoid(float(np.dot(self.coef, x) + self.intercept)))

    def reliability(self, ctx: Dict[str, Any], default: float = 1.0) -> float:
        if not self.fitted:
            return default
        return float(np.clip(2.0 * self.p_correct(ctx) - 1.0, 0.0, 1.0))

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "ReliabilityModel":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ConditionalCalibrator:
    base: Calibrator = field(default_factory=Calibrator)
    groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    min_group_n: int = 60
    alpha: float = 0.1

    @property
    def fitted(self) -> bool:
        return self.base.fitted

    def prob_fake(self, raw_logit):
        return self.base.prob_fake(raw_logit)

    def _calibrator_for(self, ctx: Optional[Dict[str, Any]]):
        key = group_key(ctx or {})
        g = self.groups.get(key)
        if g is None:
            return self.base, "global"
        cal = Calibrator(a=self.base.a, b=self.base.b, q_real=g["q_real"],
                         q_fake=g["q_fake"], alpha=self.alpha, fitted=True,
                         n_cal=g["n"], op_thresholds=self.base.op_thresholds)
        return cal, key

    def label_set(self, p_fake: float, ctx: Optional[Dict[str, Any]] = None,
                  mode: str = "balanced"):
        cal, _ = self._calibrator_for(ctx)
        return cal.label_set(p_fake, mode)

    def verdict(self, p_fake: float, ctx=None, mode: str = "balanced") -> str:
        s = self.label_set(p_fake, ctx, mode)
        return FAKE if s == [FAKE] else REAL if s == [REAL] else "Inconclusive"

    def which_group(self, ctx) -> str:
        return self._calibrator_for(ctx)[1]

    def to_dict(self) -> dict:
        def enc(v):
            return "inf" if isinstance(v, float) and math.isinf(v) else v
        return {"base": self.base.to_dict(), "min_group_n": self.min_group_n,
                "alpha": self.alpha,
                "groups": {k: {kk: enc(vv) for kk, vv in g.items()}
                           for k, g in self.groups.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "ConditionalCalibrator":
        def dec(v):
            return math.inf if v == "inf" else v
        return cls(base=Calibrator.from_dict(d["base"]),
                   groups={k: {kk: dec(vv) for kk, vv in g.items()}
                           for k, g in d.get("groups", {}).items()},
                   min_group_n=d.get("min_group_n", 60), alpha=d.get("alpha", 0.1))


def fit_conditional(logits: Sequence[float], labels: Sequence[int],
                    ctxs: Sequence[Dict[str, Any]], alpha: float = 0.1,
                    min_group_n: int = 60, seed: int = 0) -> ConditionalCalibrator:
    """Fit Platt + global conformal on one half, per-group conformal on the
    other half (disjoint, so the guarantee is not reused)."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    base = fit_calibrator(z, y, alpha=alpha, seed=seed)
    rng = np.random.default_rng(seed + 1)
    idx = rng.permutation(len(z))
    c_idx = idx[len(idx) // 2:] if len(idx) >= 4 else idx
    p = base.prob_fake(z)
    keys = np.array([group_key(c) for c in ctxs])
    groups: Dict[str, Dict[str, Any]] = {}
    for k in np.unique(keys):
        m = np.zeros(len(z), dtype=bool)
        m[c_idx] = True
        m &= keys == k
        yy = y[m]
        if m.sum() < min_group_n or len(np.unique(yy)) < 2:
            continue
        q_real, q_fake = fit_conformal(p[m], yy, alpha)
        if math.isinf(q_real) and math.isinf(q_fake):
            continue
        groups[str(k)] = {"q_real": float(q_real), "q_fake": float(q_fake), "n": int(m.sum())}
    return ConditionalCalibrator(base=base, groups=groups, min_group_n=min_group_n, alpha=alpha)


def save_conditional(path: str, cal: ConditionalCalibrator,
                     reliability: Optional[Dict[str, ReliabilityModel]] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"calibrator": cal.to_dict(),
                   "reliability": {k: v.to_dict() for k, v in (reliability or {}).items()}},
                  f, indent=2)


def load_conditional(path: str):
    """(ConditionalCalibrator | None, {expert: ReliabilityModel})"""
    if not os.path.exists(path):
        return None, {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return (ConditionalCalibrator.from_dict(d["calibrator"]),
                {k: ReliabilityModel.from_dict(v) for k, v in d.get("reliability", {}).items()})
    except Exception:
        return None, {}


def group_coverage(cal, logits, labels, ctxs, mode="balanced") -> Dict[str, Dict[str, float]]:
    """Empirical per-group coverage / abstention of any calibrator exposing
    prob_fake and label_set(p[, ctx]) - the quantity that matters here."""
    z, y = np.asarray(logits), np.asarray(labels).astype(int)
    out: Dict[str, Dict[str, float]] = {}
    keys = [group_key(c) for c in ctxs]
    p = cal.prob_fake(z)
    for k in sorted(set(keys)):
        idx = [i for i, kk in enumerate(keys) if kk == k]
        cov, ab = [], []
        for i in idx:
            s = (cal.label_set(float(p[i]), ctxs[i], mode)
                 if isinstance(cal, ConditionalCalibrator) else cal.label_set(float(p[i]), mode))
            cov.append(("Fake" if y[i] else "Real") in s)
            ab.append(len(s) != 1)
        out[k] = {"n": len(idx), "coverage": float(np.mean(cov)), "abstain": float(np.mean(ab))}
    return out
