"""Evaluation metrics: AUC, EER, TPR at fixed FPR, calibration and selective
(abstain-aware) metrics. Pure numpy/sklearn so it runs anywhere."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def auc(y_true: Sequence[int], score: Sequence[float]) -> float:
    y = _arr(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, _arr(score)))


def eer(y_true: Sequence[int], score: Sequence[float]) -> float:
    """Equal error rate (lower is better)."""
    y = _arr(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, _arr(score))
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def tpr_at_fpr(
    y_true: Sequence[int], score: Sequence[float], target_fpr: float = 0.01
) -> float:
    """Fraction of fakes caught while keeping false alarms on real media
    at or below ``target_fpr``."""
    y = _arr(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, _arr(score))
    ok = np.where(fpr <= target_fpr)[0]
    return float(tpr[ok].max()) if len(ok) else 0.0


def expected_calibration_error(
    y_true: Sequence[int], p_fake: Sequence[float], bins: int = 15
) -> float:
    y = _arr(y_true)
    p = _arr(p_fake)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def brier(y_true: Sequence[int], p_fake: Sequence[float]) -> float:
    return float(np.mean((_arr(p_fake) - _arr(y_true)) ** 2))


def selective_metrics(
    y_true: Sequence[int], verdicts: Sequence[str]
) -> Dict[str, Optional[float]]:
    """Metrics for a system that may answer 'Inconclusive'.

    * abstain_rate        - share of inputs left undecided
    * accuracy_answered   - accuracy on the inputs it did decide
    * false_alarm_rate    - real media called Fake (among answered reals)
    * miss_rate           - fakes called Real (among answered fakes)
    """
    y = np.asarray(y_true).astype(int).reshape(-1)
    v = np.asarray(list(verdicts)).reshape(-1)
    answered = v != "Inconclusive"
    out: Dict[str, Optional[float]] = {
        "n": int(len(y)),
        "abstain_rate": float(1.0 - answered.mean()) if len(y) else None,
    }
    if answered.any():
        pred = (v[answered] == "Fake").astype(int)
        out["accuracy_answered"] = float((pred == y[answered]).mean())
    else:
        out["accuracy_answered"] = None
    real_ans = answered & (y == 0)
    fake_ans = answered & (y == 1)
    out["false_alarm_rate"] = (
        float((v[real_ans] == "Fake").mean()) if real_ans.any() else None
    )
    out["miss_rate"] = (
        float((v[fake_ans] == "Real").mean()) if fake_ans.any() else None
    )
    return out


# --------------------------------------------------------------------------
# selective prediction, likelihood, and uncertainty on the estimates
# --------------------------------------------------------------------------
def nll(y_true: Sequence[int], p_fake: Sequence[float], eps: float = 1e-6) -> float:
    y, p = _arr(y_true), np.clip(_arr(p_fake), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def risk_coverage_curve(
    y_true: Sequence[int], p_fake: Sequence[float],
    confidence: Optional[Sequence[float]] = None,
) -> Dict[str, np.ndarray]:
    """Selective-classification curve: answer only the ``coverage`` most
    confident inputs; ``risk`` is the error rate on those. ``confidence``
    defaults to max(p, 1-p); pass the system's own confidence (e.g. evidence
    sufficiency) to evaluate it as a selector."""
    y, p = _arr(y_true).astype(int), _arr(p_fake)
    conf = np.maximum(p, 1 - p) if confidence is None else _arr(confidence)
    order = np.argsort(-conf, kind="stable")
    err = ((p[order] >= 0.5).astype(int) != y[order]).astype(float)
    k = np.arange(1, len(y) + 1)
    return {"coverage": k / len(y), "risk": np.cumsum(err) / k}


def aurc(y_true, p_fake, confidence=None) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    c = risk_coverage_curve(y_true, p_fake, confidence)
    return float(np.mean(c["risk"]))


def coverage_at_risk(y_true, p_fake, target_risk: float, confidence=None) -> float:
    """Largest coverage whose selective risk is <= target (0 if none)."""
    c = risk_coverage_curve(y_true, p_fake, confidence)
    ok = np.where(c["risk"] <= target_risk)[0]
    return float(c["coverage"][ok.max()]) if len(ok) else 0.0


def bootstrap_ci(metric, *arrays, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap over rows. ``metric(*arrays) -> float``."""
    arrs = [np.asarray(a) for a in arrays]
    n = len(arrs[0])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = metric(*[a[idx] for a in arrs])
        except Exception:
            continue
        if v == v:
            vals.append(v)
    point = metric(*arrs)
    if not vals:
        return {"value": float(point), "lo": float("nan"), "hi": float("nan")}
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"value": float(point), "lo": float(lo), "hi": float(hi)}


def paired_bootstrap_diff(metric, y, score_a, score_b, n_boot: int = 1000,
                          alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap of metric(a) - metric(b) on the same rows. A CI that
    excludes 0 is evidence of a real difference; ``p_le_zero`` is the share of
    resamples where A was not better (for metrics where higher is better)."""
    y, a, b = np.asarray(y), np.asarray(score_a), np.asarray(score_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        try:
            d = metric(y[idx], a[idx]) - metric(y[idx], b[idx])
        except Exception:
            continue
        if d == d:
            diffs.append(d)
    point = metric(y, a) - metric(y, b)
    if not diffs:
        return {"diff": float(point), "lo": float("nan"), "hi": float("nan"), "p_le_zero": float("nan")}
    d = np.asarray(diffs)
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(point), "lo": float(lo), "hi": float(hi),
            "p_le_zero": float((d <= 0).mean())}
