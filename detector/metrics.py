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


def adaptive_ece(
    y_true: Sequence[int], p_fake: Sequence[float], bins: int = 15
) -> float:
    """Adaptive Expected Calibration Error with equal-mass binning."""
    y = _arr(y_true)
    p = _arr(p_fake)
    n = len(p)
    if n == 0:
        return 0.0
    order = np.argsort(p)
    y, p = y[order], p[order]
    chunks = np.array_split(np.arange(n), min(bins, n))
    aece = 0.0
    for chunk in chunks:
        if len(chunk) > 0:
            w = len(chunk) / n
            aece += w * abs(p[chunk].mean() - y[chunk].mean())
    return float(aece)


def class_wise_ece(
    y_true: Sequence[int], p_fake: Sequence[float], bins: int = 15
) -> Dict[str, float]:
    """Class-wise calibration error evaluated separately for real and fake."""
    y = _arr(y_true)
    p = _arr(p_fake)
    m0 = (y == 0)
    m1 = (y == 1)
    ece_real = expected_calibration_error(np.zeros(int(m0.sum())), 1.0 - p[m0], bins=bins) if m0.any() else 0.0
    ece_fake = expected_calibration_error(np.ones(int(m1.sum())), p[m1], bins=bins) if m1.any() else 0.0
    return {
        "ece_real": float(ece_real),
        "ece_fake": float(ece_fake),
        "class_wise_mean": float(0.5 * (ece_real + ece_fake)),
    }


def reliability_diagram_data(
    y_true: Sequence[int], p_fake: Sequence[float], bins: int = 10
) -> Dict[str, Any]:
    """Generates calibration curve coordinates (confidence vs empirical accuracy)."""
    y = _arr(y_true)
    p = _arr(p_fake)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_centers = []
    bin_confs = []
    bin_accs = []
    bin_counts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        count = int(m.sum())
        bin_centers.append(float(0.5 * (lo + hi)))
        bin_counts.append(count)
        if count > 0:
            bin_confs.append(float(p[m].mean()))
            bin_accs.append(float(y[m].mean()))
        else:
            bin_confs.append(float(0.5 * (lo + hi)))
            bin_accs.append(float("nan"))
    return {
        "bin_edges": edges.tolist(),
        "bin_centers": bin_centers,
        "confidence": bin_confs,
        "accuracy": bin_accs,
        "counts": bin_counts,
        "ece": expected_calibration_error(y, p, bins=bins),
        "adaptive_ece": adaptive_ece(y, p, bins=bins),
        "brier": brier(y, p),
        "nll": nll(y, p),
        "class_wise": class_wise_ece(y, p, bins=bins),
    }


def conformal_evaluation_metrics(
    y_true: Sequence[int],
    prediction_sets: Sequence[Sequence[str]],
    condition_tags: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Evaluates empirical coverage, set size, and class-conditional coverage.

    Each prediction set is e.g. ["Real"], ["Fake"], ["Real", "Fake"], or [].
    """
    y = _arr(y_true).astype(int)
    label_map = {0: "Real", 1: "Fake"}
    covered = []
    sizes = []
    singletons = 0
    empties = 0
    cov_real = []
    cov_fake = []

    for yi, s in zip(y, prediction_sets):
        true_name = label_map[yi]
        is_cov = true_name in s
        covered.append(is_cov)
        sizes.append(len(s))
        if len(s) == 1:
            singletons += 1
        elif len(s) == 0:
            empties += 1
        if yi == 0:
            cov_real.append(is_cov)
        else:
            cov_fake.append(is_cov)

    n = max(len(y), 1)
    out: Dict[str, Any] = {
        "n": int(len(y)),
        "empirical_coverage": float(np.mean(covered)) if covered else 0.0,
        "coverage_real": float(np.mean(cov_real)) if cov_real else 0.0,
        "coverage_fake": float(np.mean(cov_fake)) if cov_fake else 0.0,
        "mean_set_size": float(np.mean(sizes)) if sizes else 0.0,
        "fraction_singleton": float(singletons / n),
        "fraction_empty": float(empties / n),
    }

    if condition_tags is not None and len(condition_tags) == len(y):
        tags = np.asarray(condition_tags)
        cond_cov = {}
        for tag in np.unique(tags):
            mask = (tags == tag)
            cond_cov[str(tag)] = {
                "n": int(mask.sum()),
                "coverage": float(np.mean([covered[i] for i in np.where(mask)[0]])),
                "mean_set_size": float(np.mean([sizes[i] for i in np.where(mask)[0]])),
            }
        out["conditional_by_tag"] = cond_cov

    return out


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
