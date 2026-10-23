"""Evidence fusion.

Each expert returns an :class:`Evidence` (a fake log-odds score plus a
reliability in [0, 1]). Fusion is a *reliability-weighted average of
log-odds* rather than a sum: summing assumes the experts are independent,
which they are not (they all look at the same pixels), and would make the
system over-confident.

Disagreement between experts shrinks the fused score towards 0 (maximum
uncertainty), which the conformal stage turns into "Inconclusive".
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from detector.calibration import sigmoid

# Hand-set priors. They are replaced by learned weights when
# artifacts/fusion_weights.json exists (see train/fit_fusion.py).
DEFAULT_WEIGHTS: Dict[str, float] = {
    "neural": 1.0,
    "neural_probe": 1.0,
    "siglip_semantic": 0.6,
    "temporal": 0.8,
    "audio": 1.0,
    "sync": 0.35,
    "rppg": 0.25,
    "geometry": 0.20,
    "spectral": 0.50,
    "tamper": 0.30,
    "localization": 0.0,  # heatmap detail only; logit is a duplicate of tamper
    "forensic_prior": 0.15,
    "metadata_ai": 1.0,
}


@dataclass
class Evidence:
    name: str
    logit: Optional[float] = None
    reliability: float = 1.0
    status: str = "used"  # used | skipped
    reason: str = ""
    heuristic: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "logit": None if self.logit is None else round(float(self.logit), 4),
            "p_fake_raw": None if self.logit is None
            else round(float(sigmoid(self.logit)), 4),
            "reliability": round(float(self.reliability), 3),
            "heuristic": self.heuristic,
            "reason": self.reason,
            "detail": _jsonable(self.detail),
        }


def skipped(name: str, reason: str, heuristic: bool = False) -> Evidence:
    return Evidence(name=name, logit=None, reliability=0.0, status="skipped",
                    reason=reason, heuristic=heuristic)


@dataclass
class FusionResult:
    logit: float
    disagreement: float
    n_used: int
    contributions: Dict[str, float]

    @property
    def p_fake(self) -> float:
        return float(sigmoid(self.logit))


def fuse(
    evidences: Sequence[Evidence],
    weights: Optional[Dict[str, float]] = None,
    disagreement_penalty: float = 0.5,
    max_abs_logit: float = 12.0,
) -> FusionResult:
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    # Neutral experts (logit ≈ 0) must not dilute real evidence. Localization
    # is also excluded from the numeric fuse: its logit is a re-run of tamper.
    used = [
        e for e in evidences
        if e.status == "used"
        and e.logit is not None
        and e.reliability > 0
        and abs(float(e.logit)) >= 1e-6
        and e.name != "localization"
    ]
    if not used:
        return FusionResult(0.0, 0.0, 0, {})
    z = np.array([np.clip(e.logit, -max_abs_logit, max_abs_logit) for e in used])
    wi = np.array([max(w.get(e.name, 0.3), 0.0) * e.reliability for e in used])
    if wi.sum() <= 0:
        return FusionResult(0.0, 0.0, 0, {})
    mean = float((wi * z).sum() / wi.sum())
    var = float((wi * (z - mean) ** 2).sum() / wi.sum())
    disagreement = float(np.sqrt(var))
    shrink = 1.0 / (1.0 + disagreement_penalty * disagreement / 4.0)
    contrib = {e.name: float(x / wi.sum()) for e, x in zip(used, wi)}
    return FusionResult(mean * shrink, disagreement, len(used), contrib)


def load_weights(artifact_dir: str) -> Optional[Dict[str, float]]:
    path = os.path.join(artifact_dir, "fusion_weights.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {k: float(v) for k, v in json.load(f).items()}
    except Exception:
        return None


def fit_fusion_weights(
    Z: np.ndarray, R: np.ndarray, y: np.ndarray, names: Sequence[str],
    C: float = 1.0,
) -> Dict[str, float]:
    """Learn non-negative-ish expert weights by logistic stacking.

    Z: (n, k) expert logits (0 where missing); R: (n, k) reliabilities
    (0 where missing); y: (n,) labels. Features are Z*R so a missing expert
    contributes nothing. Weights are normalised so the strongest expert is
    ~1.0 (the scale of DEFAULT_WEIGHTS) and clipped to [0, 2].
    """
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(Z) * np.asarray(R)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
    clf.fit(X, y)
    coef = np.clip(clf.coef_[0], 0.0, None)
    if coef.max() <= 0:
        return {n: DEFAULT_WEIGHTS.get(n, 0.3) for n in names}
    coef = coef / coef.max()
    return {n: float(np.clip(c, 0.02, 2.0)) for n, c in zip(names, coef)}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    return obj
