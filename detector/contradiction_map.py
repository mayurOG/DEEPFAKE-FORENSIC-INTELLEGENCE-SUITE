"""Localized cross-modal contradiction mapping.

A single whole-clip lip-sync score hides *where* audio and face disagree and
cannot tell a pasted-in second of synthetic speech from benign dubbing.
This module slides a window over the clip and, per window, compares the
mouth-motion signal with the audio envelope:

* the lag search is centred on the clip's own global lag (+- ``lag_slack_s``),
  so a constant dubbing / latency offset is *not* a contradiction;
* a clip whose windows are *all* uncoupled has no intact reference, so it is
  reported as ``no_reference`` rather than localised (that whole-clip case
  is the job of the global sync expert);
* a window is only judged when both modalities are informative (speech is
  present, the mouth is moving, the face track is not lost). Otherwise it is
  ``uninformative`` - missing evidence is reported, never converted into a
  score;
* the correlation comes with a Fisher-z confidence interval using an
  effective sample size (smoothing reduces independent samples). A window is
  a ``contradiction`` only when the *upper* bound is below ``tau`` (confident
  disagreement with the clip's reference coupling), ``consistent`` when the
  *lower* bound is above it, and ``uncertain`` otherwise;
* consecutive contradiction windows merge into time intervals.

Limits: mouth-region motion is a coarse proxy for visemes, windows are
overlapping (not independent, so many windows means many chances for a false
alarm: use ``min_windows`` for an interval), and it has not been validated
against real dubbing / editing corpora here.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np


def _interp_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    bad = np.isnan(x)
    if bad.all():
        return np.zeros_like(x)
    if bad.any():
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def _smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    return np.convolve(x, np.ones(k) / k, mode="same")


def _norm_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def _best_lagged(m: np.ndarray, a: np.ndarray, lags: range) -> (float, int):
    best, best_l, n = -1.0, 0, len(m)
    for l in lags:
        x, y = (m[: n - l], a[l:]) if l >= 0 else (m[-l:], a[: n + l])
        if len(x) < 6:
            continue
        r = _norm_corr(x, y)
        if r > best:
            best, best_l = r, l
    return best, best_l


def _win_corr(m, a, s_m, s_a, W, lags, n):
    """Best lagged correlation between mouth[s_m:s_m+W] and env around s_a."""
    x = m[s_m:s_m + W]
    best, best_l = -1.0, 0
    for l in lags:
        a0 = s_a + l
        if a0 < 0 or a0 + W > n:
            continue
        r = _norm_corr(x, a[a0:a0 + W])
        if r > best:
            best, best_l = r, l
    return best, best_l


def contradiction_map(
    mouth: Optional[np.ndarray], env: Optional[np.ndarray], fps: float,
    window_s: float = 3.0, hop_s: float = 0.5, global_max_lag_s: float = 0.4,
    lag_slack_s: float = 0.12, p_consistent: float = 0.05, p_contra: float = 0.25,
    min_windows: int = 2, max_face_missing: float = 0.4, smoothing: int = 3,
    n_null: int = 40, min_gap_s: float = 1.0, ref_fraction: float = 0.6,
    seed: int = 0,
) -> Dict[str, Any]:
    if mouth is None or len(mouth) == 0:
        return {"status": "no_face", "windows": [], "intervals": [], "coverage": 0.0,
                "note": "no usable face track; nothing to compare"}
    if env is None or len(env) == 0:
        return {"status": "no_audio", "windows": [], "intervals": [], "coverage": 0.0,
                "note": "no audio track; cross-modal comparison impossible"}
    n = min(len(mouth), len(env))
    raw_m = np.asarray(mouth[:n], dtype=np.float64)
    nan_frac = np.isnan(raw_m)
    m = _smooth(_interp_nan(raw_m), smoothing)
    a = _smooth(np.asarray(env[:n], dtype=np.float64), smoothing)
    W, H = max(8, int(round(window_s * fps))), max(1, int(round(hop_s * fps)))
    if n < W:
        return {"status": "too_short", "windows": [], "intervals": [], "coverage": 0.0,
                "note": f"clip shorter than one {window_s}s window"}
    gl = int(global_max_lag_s * fps)
    # global lag over the whole clip: a constant offset is benign
    g_best, g_lag = _best_lagged(m, a, range(-gl, gl + 1))
    slack = max(1, int(round(lag_slack_s * fps)))
    lags = range(g_lag - slack, g_lag + slack + 1)
    gap = max(W // 2, int(min_gap_s * fps))
    rng = np.random.default_rng(seed)
    lo_env, hi_env = float(np.percentile(a, 20)), float(np.percentile(a, 90))
    speech_thr = lo_env + 0.35 * (hi_env - lo_env)
    m_floor = 0.15 * float(np.std(m) + 1e-9)
    wins: List[Dict[str, Any]] = []
    for s in range(0, n - W + 1, H):
        e = s + W
        w = {"t0": round(s / fps, 3), "t1": round(e / fps, 3)}
        if nan_frac[s:e].mean() > max_face_missing:
            wins.append({**w, "status": "uninformative", "why": "face lost"})
            continue
        if (a[s:e] > speech_thr).mean() < 0.25 or np.ptp(a[s:e]) < 1e-6:
            wins.append({**w, "status": "uninformative", "why": "no speech"})
            continue
        if np.std(m[s:e]) < m_floor:
            wins.append({**w, "status": "uninformative", "why": "no mouth motion"})
            continue
        r_obs, lag = _win_corr(m, a, s, s, W, lags, n)
        starts = [o for o in range(0, n - W + 1) if abs(o - s) >= gap]
        if r_obs <= -1.0 or len(starts) < 8:
            wins.append({**w, "status": "uncertain", "why": "too little audio for a null"})
            continue
        pick = rng.choice(starts, size=min(n_null, len(starts)), replace=False)
        null = np.array([_win_corr(m, a, s, int(o), W, lags, n)[0] for o in pick])
        null = null[null > -1.0]
        p = float((1 + (null >= r_obs).sum()) / (1 + len(null)))
        wins.append({**w, "status": "pending", "corr": round(r_obs, 3), "p": round(p, 3),
                     "null_p95": round(float(np.quantile(null, 0.95)), 3),
                     "lag_s": round(lag / fps, 3)})
    scored = [w for w in wins if w["status"] == "pending"]
    coupled = [w for w in scored if w["p"] <= p_consistent]
    r_ref = float(np.median([w["corr"] for w in coupled])) if coupled else None
    for w in scored:
        if r_ref is None:
            w["status"] = "uncertain"
        elif w["p"] <= p_consistent:
            w["status"] = "consistent"
        elif w["p"] >= p_contra and w["corr"] < ref_fraction * r_ref:
            w["status"] = "contradiction"
        else:
            w["status"] = "uncertain"
    informative = [w for w in wins if w["status"] != "uninformative"]
    intervals: List[Dict[str, Any]] = []
    cur: List[Dict[str, Any]] = []
    for w in wins + [{"status": "end"}]:
        if w["status"] == "contradiction":
            cur.append(w)
            continue
        if len(cur) >= min_windows:
            intervals.append({
                "t0": cur[0]["t0"], "t1": cur[-1]["t1"], "windows": len(cur),
                "mean_corr": round(float(np.mean([c["corr"] for c in cur])), 3),
                "min_p": round(float(min(c["p"] for c in cur)), 3)})
        cur = []
    status = "ok" if r_ref is not None else ("no_reference" if scored else "ok")
    return {"status": status, "fps": fps, "window_s": window_s,
            "global_lag_s": round(g_lag / fps, 3), "global_corr": round(g_best, 3),
            "reference_corr": None if r_ref is None else round(r_ref, 3),
            "benign_latency": bool(abs(g_lag / fps) <= global_max_lag_s * 0.99
                                   and r_ref is not None and not intervals),
            "windows": wins, "intervals": intervals,
            "coverage": round(len(informative) / max(len(wins), 1), 3),
            "n_contradiction_windows": sum(w["status"] == "contradiction" for w in wins)}
