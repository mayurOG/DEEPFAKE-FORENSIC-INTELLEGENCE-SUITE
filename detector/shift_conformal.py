"""Shift-aware conformal prediction.

Split conformal prediction guarantees coverage only if test data look like the
calibration data. Deepfake detection breaks that assumption: new generators and
new compression chains are the norm. Two complementary repairs:

* **Weighted conformal (covariate shift).** A domain classifier estimates the
  likelihood ratio w(x) = p_test(x) / p_cal(x) from embeddings. Calibration
  scores are re-weighted by w, and the test point carries its own weight at
  +inf (Tibshirani et al., 2019). Coverage then holds under covariate shift if
  the ratio is well estimated.
* **Adaptive conformal inference (online drift).** The working miscoverage
  level alpha_t is nudged after every labelled outcome, so long-run coverage
  tracks the target even when the shift is unknown (Gibbs & Candes, 2021).

Both are numpy/scikit-learn only. They are *methods to validate*: guarantees
depend on assumptions (ratio estimation quality, labels arriving online).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def estimate_density_ratio(cal_feats: np.ndarray, target_feats: np.ndarray,
                           clip: float = 20.0, C: float = 1.0) -> "DensityRatio":
    """Fit a logistic domain classifier (cal=0, target=1)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    A = np.asarray(cal_feats, dtype=np.float64)
    B = np.asarray(target_feats, dtype=np.float64)
    sc = StandardScaler().fit(np.vstack([A, B]))
    X = sc.transform(np.vstack([A, B]))
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    clf = LogisticRegression(C=C, max_iter=500, class_weight=None).fit(X, y)
    return DensityRatio(sc, clf, len(A), len(B), clip)


@dataclass
class DensityRatio:
    scaler: object
    clf: object
    n_cal: int
    n_target: int
    clip: float = 20.0

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        X = self.scaler.transform(np.atleast_2d(np.asarray(feats, dtype=np.float64)))
        p = np.clip(self.clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        w = (p / (1 - p)) * (self.n_cal / self.n_target)
        return np.clip(w, 1.0 / self.clip, self.clip)


def weighted_conformal_quantile(scores: Sequence[float], weights: Sequence[float],
                                w_test: float, alpha: float) -> float:
    """(1 - alpha) quantile of the weighted score distribution with the test
    point's mass at +inf. Returns +inf when the mass is insufficient (the safe
    always-include answer)."""
    s = np.asarray(scores, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if len(s) == 0:
        return math.inf
    order = np.argsort(s)
    s, w = s[order], w[order]
    total = w.sum() + float(w_test)
    cum = np.cumsum(w) / total
    k = int(np.searchsorted(cum, 1.0 - alpha, side="left"))
    return math.inf if k >= len(s) else float(s[k])


def weighted_class_quantiles(p_fake, labels, weights, w_test, alpha):
    """Per-class (q_real, q_fake) mirroring ``calibration.fit_conformal``."""
    p = np.asarray(p_fake, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    w = np.asarray(weights, dtype=np.float64)
    q_real = weighted_conformal_quantile(p[y == 0], w[y == 0], w_test, alpha)
    q_fake = weighted_conformal_quantile(1.0 - p[y == 1], w[y == 1], w_test, alpha)
    return q_real, q_fake


def prediction_set(p_fake: float, q_real: float, q_fake: float):
    """Label set under per-class nonconformity thresholds."""
    out = []
    if p_fake <= q_real:
        out.append("Real")
    if 1.0 - p_fake <= q_fake:
        out.append("Fake")
    return out


class AdaptiveConformal:
    """Online alpha_t tracking: alpha_{t+1} = alpha_t + gamma (alpha - err_t)."""

    def __init__(self, alpha: float = 0.1, gamma: float = 0.01):
        self.target, self.gamma = float(alpha), float(gamma)
        self.alpha_t = float(alpha)
        self.n = 0
        self.errors = 0

    def effective_alpha(self) -> float:
        return float(np.clip(self.alpha_t, 1e-3, 0.999))

    def update(self, covered: bool) -> float:
        err = 0.0 if covered else 1.0
        self.alpha_t += self.gamma * (self.target - err)
        self.n += 1
        self.errors += int(err)
        return self.effective_alpha()

    @property
    def running_miscoverage(self) -> Optional[float]:
        return None if self.n == 0 else self.errors / self.n
