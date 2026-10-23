"""Spectral fingerprint expert.

Generative models (GAN / diffusion / VAE decoders) upsample with learned or
fixed kernels, which leaves characteristic traces in the 2-D Fourier
spectrum: a modified radial decay (the "spectral slope"), excess or missing
high-frequency energy, and periodic off-centre peaks from stride/transpose
convolutions. Camera images follow a smoother ~1/f power law.

Features (all computed from a centre crop, Hann-windowed, in float64):
  * azimuthally averaged log-power in ``n_bins`` radial bins, detrended by
    the image's own power-law fit (so overall contrast cancels)
  * spectral slope and its fit residual
  * high-frequency energy ratio
  * periodic-peak score: strongest off-axis peak in the high band relative
    to the local median (upsampling grids)

A logistic head on these features is trained with ``train.train_spectral``
(CPU, seconds). Without a trained head the expert is *skipped*, never
guessed. It complements, not replaces, the neural backbone: it is cheap,
backbone-independent, and fails differently (e.g. it is weak after heavy
JPEG or resizing, which the reliability term reflects).
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
from PIL import Image

from detector.fusion import Evidence, skipped

N_BINS = 48
CROP = 256


def _prep(img: Image.Image, crop: int = CROP) -> np.ndarray:
    g = np.asarray(img.convert("L"), dtype=np.float64)
    h, w = g.shape
    if min(h, w) < crop:
        k = min(h, w)
        crop = k - (k % 2)
    top, left = (h - crop) // 2, (w - crop) // 2
    g = g[top:top + crop, left:left + crop]
    return g - g.mean()


def radial_profile(power: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    n = power.shape[0]
    yy, xx = np.indices(power.shape)
    r = np.hypot(yy - n / 2, xx - n / 2)
    rmax = n / 2
    idx = np.minimum((r / rmax * n_bins).astype(int), n_bins)
    keep = idx < n_bins
    sums = np.bincount(idx[keep], weights=power[keep], minlength=n_bins)
    cnt = np.bincount(idx[keep], minlength=n_bins)
    return sums / np.maximum(cnt, 1)


def spectral_features(img: Image.Image) -> np.ndarray:
    g = _prep(img)
    n = g.shape[0]
    if n < 64:
        return np.zeros(N_BINS + 5)
    win = np.outer(np.hanning(n), np.hanning(n))
    f = np.fft.fftshift(np.fft.fft2(g * win))
    power = np.abs(f) ** 2 + 1e-12
    prof = np.log(radial_profile(power) + 1e-12)
    x = np.log(np.arange(1, N_BINS + 1, dtype=np.float64))
    lo, hi = 2, N_BINS  # skip DC-dominated first bins
    A = np.vstack([x[lo:hi], np.ones(hi - lo)]).T
    (slope, icpt), res, *_ = np.linalg.lstsq(A, prof[lo:hi], rcond=None)
    fit = A @ np.array([slope, icpt])
    resid = prof[lo:hi] - fit
    detr = np.zeros(N_BINS)
    detr[lo:hi] = resid
    total = np.exp(prof).sum()
    hf_ratio = float(np.exp(prof[int(N_BINS * 0.66):]).sum() / total)
    # periodic peaks: high band, ignore the axes (window leakage) and DC
    yy, xx = np.indices(power.shape)
    r = np.hypot(yy - n / 2, xx - n / 2) / (n / 2)
    band = (r > 0.45) & (r < 0.98)
    band &= (np.abs(yy - n / 2) > 3) & (np.abs(xx - n / 2) > 3)
    lp = np.log(power)
    local = lp[band]
    peak = float((local.max() - np.median(local)) / (local.std() + 1e-9))
    return np.concatenate([detr, [slope, float(resid.std()),
                                  np.log(hf_ratio + 1e-9), peak,
                                  float(np.mean(np.abs(resid)))]])


# -- head ------------------------------------------------------------------
def fit_head(X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None,
             Cs=(0.01, 0.1, 1.0, 10.0), seed: int = 0) -> Dict:
    from sklearn.linear_model import LogisticRegression
    from detector import metrics as MT

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    g = np.asarray(groups) if groups is not None else np.arange(len(y))
    rng = np.random.default_rng(seed)
    va = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):  # group-aware split, stratified so both classes stay in train
        ug = np.unique(g[y == c])
        rng.shuffle(ug)
        k = min(max(1, int(len(ug) * 0.25)), max(len(ug) - 1, 0))
        va |= np.isin(g, ug[:k]) & (y == c)
    leaky = False
    if len(np.unique(y[va])) < 2 or len(np.unique(y[~va])) < 2:
        # too few groups: fall back to a stratified per-sample split
        leaky = True
        va = np.zeros(len(y), dtype=bool)
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            va[idx[: max(1, len(idx) // 4)]] = True
    tr = ~va
    mean, scale = X[tr].mean(0), X[tr].std(0) + 1e-6
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
        clf.fit((X[tr] - mean) / scale, y[tr])
        a = MT.auc(y[va], clf.decision_function((X[va] - mean) / scale)) \
            if len(np.unique(y[va])) > 1 else float("nan")
        s = a if a == a else -1
        if best is None or s > best[0]:
            best = (s, C, a)
    _, C, auc = best
    mean, scale = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
    clf.fit((X - mean) / scale, y)
    return {"w": clf.coef_[0], "b": float(clf.intercept_[0]), "mean": mean,
            "scale": scale, "C": float(C),
            "val_auc": float(auc) if auc == auc else 0.5, "group_split": not leaky}


def save_head(head: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in head.items()})


def load_head(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path)
        return {k: d[k] for k in d.files}
    except Exception:
        return None


def head_logit(head: Dict, feats: np.ndarray) -> float:
    z = (np.asarray(feats) - head["mean"]) / head["scale"]
    return float(z @ head["w"] + float(head["b"]))


def spectral_evidence(img: Image.Image, head: Optional[Dict],
                      compression: str = "low") -> Evidence:
    if head is None:
        return skipped("spectral", "no trained spectral head "
                       "(run python -m train.train_spectral)", False)
    if min(img.size) < 96:
        return skipped("spectral", "image too small for spectral analysis", False)
    f = spectral_features(img)
    z = float(np.clip(head_logit(head, f), -8, 8))
    # JPEG re-compression and downscaling erase upsampling traces
    rel = {"low": 0.9, "medium": 0.6, "high": 0.3}.get(compression, 0.6)
    va = float(head.get("val_auc", 0.5))
    va = va if va == va else 0.5
    rel *= float(np.clip((va - 0.5) * 2.0, 0.0, 1.0))  # a useless head gets no vote
    return Evidence("spectral", z, rel, heuristic=False,
                    detail={"slope": round(float(f[N_BINS]), 3),
                            "periodic_peak_score": round(float(f[N_BINS + 3]), 2),
                            "head_val_auc": round(va, 3)})
