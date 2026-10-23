"""Calibration, conformal prediction and out-of-domain detection.

Why this exists
---------------
A raw softmax score is not a probability. For forensic use we want

1. **calibrated probabilities** (Platt / temperature scaling),
2. **a statistical guarantee** (split conformal prediction): the true label is
   inside the predicted label set with probability >= 1 - alpha, and
3. **an honest "I don't know"**: if both labels stay in the set the verdict is
   *Inconclusive*; if the input looks unlike the calibration data (OOD) the
   verdict is never reported as confident.

Everything is numpy/scipy only so it can be fitted on any machine.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

REAL, FAKE = "Real", "Fake"
EPS = 1e-12

# Target false-alarm rates for the operating modes shown in the UI.
MODE_FPR = {"strict": 0.01, "balanced": 0.05, "sensitive": 0.10}
# Fallback p(fake) margins used while no calibration data has been fitted.
MODE_FALLBACK_MARGIN = {"strict": 0.40, "balanced": 0.35, "sensitive": 0.25}


def sigmoid(z):
    z = np.clip(np.asarray(z, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


# --------------------------------------------------------------------------
# Platt / temperature scaling
# --------------------------------------------------------------------------
def fit_platt(
    logits: Sequence[float], labels: Sequence[int], l2: float = 1e-3
) -> Tuple[float, float]:
    """Fit p(fake) = sigmoid(a * logit + b) by minimising the log loss.

    Class weights make the fit insensitive to a real/fake imbalance. Returns
    (a, b). With fewer than 2 samples of a class it returns the identity.
    """
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if len(z) < 4 or len(np.unique(y)) < 2:
        return 1.0, 0.0
    n1, n0 = max(y.sum(), 1.0), max((1 - y).sum(), 1.0)
    w = np.where(y == 1, 0.5 / n1, 0.5 / n0) * len(y)

    def nll(params):
        a, b = params
        p = np.clip(sigmoid(a * z + b), EPS, 1 - EPS)
        loss = -(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).mean()
        return loss + l2 * (a - 1.0) ** 2

    res = minimize(nll, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
    a, b = float(res.x[0]), float(res.x[1])
    return a, b


# --------------------------------------------------------------------------
# Split conformal prediction (class-conditional / Mondrian)
# --------------------------------------------------------------------------
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1 - alpha) quantile of nonconformity scores.
    Returns +inf when there are too few calibration samples for the
    requested risk level (the safe, always-include answer)."""
    n = len(scores)
    if n == 0:
        return math.inf
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        return math.inf
    return float(np.quantile(scores, level, method="higher"))


def fit_conformal(
    p_fake: Sequence[float], labels: Sequence[int], alpha: float
) -> Tuple[float, float]:
    """Return (q_real, q_fake): nonconformity quantiles per true class.

    Nonconformity of a calibration sample with true class y is
    ``1 - p(y | x)``. A label is kept in the prediction set when its
    nonconformity is <= the quantile of its class.
    """
    p = np.asarray(p_fake, dtype=np.float64).reshape(-1)
    y = np.asarray(labels).astype(int).reshape(-1)
    s_real = 1.0 - (1.0 - p[y == 0])  # = p_fake for true-real samples
    s_fake = 1.0 - p[y == 1]
    return conformal_quantile(s_real, alpha), conformal_quantile(s_fake, alpha)


@dataclass
class Calibrator:
    """Maps a raw fake log-odds score to a calibrated probability and a
    conformal label set. ``fitted`` is False until fitted on real data."""

    a: float = 1.0
    b: float = 0.0
    q_real: Optional[float] = None
    q_fake: Optional[float] = None
    alpha: float = 0.1
    fitted: bool = False
    n_cal: int = 0
    # p(fake) thresholds achieving target false-alarm rates, by mode.
    op_thresholds: Dict[str, float] = field(default_factory=dict)
    # Mahalanobis OOD threshold on the embedding (None = no OOD model).
    note: str = ""

    # -- inference -----------------------------------------------------
    def prob_fake(self, raw_logit) -> np.ndarray:
        return sigmoid(self.a * np.asarray(raw_logit, dtype=np.float64) + self.b)

    def label_set(self, p_fake: float, mode: str = "balanced") -> List[str]:
        """Conformal label set for one sample. Empty set is possible and is
        treated as Inconclusive by :meth:`verdict`."""
        p = float(p_fake)
        if not self.fitted or self.q_real is None or self.q_fake is None:
            m = MODE_FALLBACK_MARGIN.get(mode, 0.35)
            if p >= 0.5 + m:
                return [FAKE]
            if p <= 0.5 - m:
                return [REAL]
            return [REAL, FAKE]
        out = []
        if p <= self.q_real:  # nonconformity of "Real" is p_fake
            out.append(REAL)
        if 1.0 - p <= self.q_fake:  # nonconformity of "Fake" is 1 - p_fake
            out.append(FAKE)
        # Operating mode: demand more evidence before calling Fake.
        if out == [FAKE]:
            thr = self.op_thresholds.get(mode)
            if thr is not None and p < thr:
                out = [REAL, FAKE]
        return out

    def verdict(self, p_fake: float, mode: str = "balanced") -> str:
        s = self.label_set(p_fake, mode)
        if s == [FAKE]:
            return FAKE
        if s == [REAL]:
            return REAL
        return "Inconclusive"

    # -- persistence ---------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("q_real", "q_fake"):
            if d[k] is not None and math.isinf(d[k]):
                d[k] = "inf"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        d = dict(d)
        for k in ("q_real", "q_fake"):
            if d.get(k) == "inf":
                d[k] = math.inf
        known = {f for f in cls.__dataclass_fields__}  # type: ignore
        return cls(**{k: v for k, v in d.items() if k in known})


def fit_calibrator(
    raw_logits: Sequence[float],
    labels: Sequence[int],
    alpha: float = 0.1,
    note: str = "",
    seed: int = 0,
) -> Calibrator:
    """Fit Platt scaling on one half of the calibration data and conformal
    quantiles on the other, so the conformal guarantee is not broken by
    reusing the same points for both.
    """
    z = np.asarray(raw_logits, dtype=np.float64).reshape(-1)
    y = np.asarray(labels).astype(int).reshape(-1)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(z))
    half = len(idx) // 2
    a_idx, c_idx = idx[:half], idx[half:]
    # Make sure both halves keep both classes when possible.
    if len(np.unique(y[a_idx])) < 2 or len(np.unique(y[c_idx])) < 2:
        a_idx = c_idx = idx  # tiny dataset: fall back (guarantee is weaker)
    a, b = fit_platt(z[a_idx], y[a_idx])
    p_c = sigmoid(a * z[c_idx] + b)
    q_real, q_fake = fit_conformal(p_c, y[c_idx], alpha)

    p_all = sigmoid(a * z + b)
    ops: Dict[str, float] = {}
    real_scores = np.sort(p_all[y == 0])
    for mode, fpr in MODE_FPR.items():
        if len(real_scores) >= 1:
            # smallest threshold t such that share(real >= t) <= fpr
            k = int(math.floor(len(real_scores) * (1.0 - fpr)))
            k = min(max(k, 0), len(real_scores) - 1)
            ops[mode] = float(real_scores[k]) + 1e-9
    return Calibrator(
        a=a, b=b, q_real=q_real, q_fake=q_fake, alpha=alpha, fitted=True,
        n_cal=int(len(z)), op_thresholds=ops, note=note,
    )


# --------------------------------------------------------------------------
# Calibration bundle (one Calibrator per media type / stage)
# --------------------------------------------------------------------------
class CalibrationBundle:
    """Named calibrators, e.g. ``image.tier1``, ``image.final``,
    ``video.final``, ``audio.final``. Missing names fall back to an unfitted
    Calibrator, which keeps the app usable but clearly flagged."""

    def __init__(self, items: Optional[Dict[str, Calibrator]] = None):
        self.items: Dict[str, Calibrator] = items or {}

    def get(self, name: str) -> Calibrator:
        return self.items.get(name) or Calibrator()

    def set(self, name: str, cal: Calibrator) -> None:
        self.items[name] = cal

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in self.items.items()}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CalibrationBundle":
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls({k: Calibrator.from_dict(v) for k, v in raw.items()})


# --------------------------------------------------------------------------
# Out-of-domain detection (Mahalanobis distance on embeddings)
# --------------------------------------------------------------------------
class OODModel:
    """Flags inputs whose embedding is far from the calibration data.

    Fitted on embeddings of the calibration set (real + fake together): it
    answers "is this the kind of media the detector was calibrated on?", not
    "is it fake?".
    """

    def __init__(self, mean=None, precision=None, threshold: float = math.inf,
                 quantile: float = 0.99):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.precision = (
            None if precision is None else np.asarray(precision, dtype=np.float64)
        )
        self.threshold = float(threshold)
        self.quantile = quantile

    @property
    def fitted(self) -> bool:
        return self.mean is not None and self.precision is not None

    def fit(self, emb: np.ndarray, shrinkage: float = 0.1) -> "OODModel":
        x = np.asarray(emb, dtype=np.float64)
        if x.ndim != 2 or len(x) < 8:
            return self
        self.mean = x.mean(axis=0)
        xc = x - self.mean
        cov = (xc.T @ xc) / max(len(x) - 1, 1)
        d = cov.shape[0]
        target = np.trace(cov) / d * np.eye(d)
        cov = (1 - shrinkage) * cov + shrinkage * target + 1e-6 * np.eye(d)
        self.precision = np.linalg.pinv(cov)
        dist = self.distance(x)
        self.threshold = float(np.quantile(dist, self.quantile))
        return self

    def distance(self, emb: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(emb, dtype=np.float64))
        if not self.fitted:
            return np.zeros(len(x))
        xc = x - self.mean
        d2 = np.einsum("ij,jk,ik->i", xc, self.precision, xc)
        return np.sqrt(np.maximum(d2, 0.0))

    def is_ood(self, emb: np.ndarray) -> Tuple[Optional[bool], Optional[float]]:
        """Returns (is_ood, score_ratio). ``(None, None)`` when unfitted."""
        if not self.fitted or emb is None:
            return None, None
        d = float(self.distance(emb)[0])
        ratio = d / self.threshold if self.threshold > 0 else 0.0
        return bool(d > self.threshold * 1.15), float(ratio)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, mean=self.mean, precision=self.precision,
                 threshold=self.threshold)

    @classmethod
    def load(cls, path: str) -> "OODModel":
        if not os.path.exists(path):
            return cls()
        z = np.load(path)
        return cls(z["mean"], z["precision"], float(z["threshold"]))
