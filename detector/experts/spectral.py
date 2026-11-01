"""NPR (Nearest-Pixel Relation) frequency expert — v4.0.

Replaces the FrequencyStubBackend that returned literal zeros.

This implements the Nearest-Pixel Relation (NPR) algorithm from:
  "Towards Universal Fake Image Detection by Training on an Artificial Dataset"
  (CVPR 2023). NPR captures systematic upsampling artifacts left by:
    * GAN decoder transpose convolutions (checkerboard patterns)
    * Diffusion model UNet decoder learned upsampling
    * VAE decoder bilinear / nearest-neighbour upsampling

Algorithm:
  1. Compute 8-directional nearest-pixel differences in the resized image.
  2. Concatenate with extended spectral features (azimuthal radial profile,
     spectral slope, periodic peak score from the existing spectral expert).
  3. Pass through a calibrated logistic head trained with train.train_spectral.

Why this works:
  Real camera images have noise that is spatially correlated in a 1/f fashion.
  Generator upsampling leaves *periodic* or *structured* residuals — the NPR
  differences pick these up in the spatial domain, while the spectral features
  catch them in the frequency domain. Together they cover a much wider range
  of generator architectures than either alone.

Usage:
  The head is trained with ``python -m train.train_spectral`` and saved to
  ``artifacts/spectral_head.npz``. The existing spectral_evidence() function
  is extended to call npr_features() and concatenate them. No API change.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from detector.fusion import Evidence, skipped

# ── Spectral parameters (unchanged from v3) ──────────────────────────────────
N_BINS = 48
CROP = 256

# ── NPR parameters ───────────────────────────────────────────────────────────
NPR_SIZE = 128       # resize target before computing NPR (fast)
NPR_DIRS = 8         # 8 directional differences (N, NE, E, SE, S, SW, W, NW)
NPR_STATS = 5        # mean, std, skew, kurtosis, max-abs per direction

# Total feature dimension = N_BINS + 5 (spectral) + NPR_DIRS * NPR_STATS (NPR)
FEATURE_DIM = N_BINS + 5 + NPR_DIRS * NPR_STATS


# ── Spectral helpers (from v3 spectral.py) ────────────────────────────────────
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


def spectral_features_only(img: Image.Image) -> np.ndarray:
    """48+5 = 53 spectral features (unchanged from v3)."""
    g = _prep(img)
    n = g.shape[0]
    if n < 64:
        return np.zeros(N_BINS + 5)
    win = np.outer(np.hanning(n), np.hanning(n))
    f = np.fft.fftshift(np.fft.fft2(g * win))
    power = np.abs(f) ** 2 + 1e-12
    prof = np.log(radial_profile(power) + 1e-12)
    x = np.log(np.arange(1, N_BINS + 1, dtype=np.float64))
    lo, hi = 2, N_BINS
    A = np.vstack([x[lo:hi], np.ones(hi - lo)]).T
    (slope, icpt), res, *_ = np.linalg.lstsq(A, prof[lo:hi], rcond=None)
    fit = A @ np.array([slope, icpt])
    resid = prof[lo:hi] - fit
    detr = np.zeros(N_BINS)
    detr[lo:hi] = resid
    total = np.exp(prof).sum()
    hf_ratio = float(np.exp(prof[int(N_BINS * 0.66):]).sum() / total)
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


# ── NPR (Nearest-Pixel Relation) ─────────────────────────────────────────────
_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0),  (1, -1), (0, -1), (-1, -1),
)  # N, NE, E, SE, S, SW, W, NW


def _skewness(x: np.ndarray) -> float:
    mu, s = x.mean(), x.std()
    if s < 1e-9:
        return 0.0
    return float(((x - mu) ** 3).mean() / s ** 3)


def _kurtosis(x: np.ndarray) -> float:
    mu, s = x.mean(), x.std()
    if s < 1e-9:
        return 0.0
    return float(((x - mu) ** 4).mean() / s ** 4)


def npr_features(img: Image.Image, size: int = NPR_SIZE) -> np.ndarray:
    """NPR: 8 directional nearest-pixel difference statistics.

    Each direction d gives 5 statistics (mean, std, skew, kurtosis, max_abs),
    yielding 8 × 5 = 40 features. These are the core NPR fingerprint.

    A camera image has nearly-isotropic noise statistics; GAN / diffusion
    upsampling breaks isotropy in structured ways (often anisotropic, higher
    kurtosis, non-zero mean due to grid aliasing).
    """
    # Resize to a fixed size — makes features resolution-invariant.
    img_r = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img_r, dtype=np.float64)  # (H, W, 3) in [0, 255]
    arr = arr / 255.0  # normalise

    feats = []
    for dy, dx in _OFFSETS:
        # Shift by (dy, dx) with wrap-around (circular boundary).
        shifted = np.roll(np.roll(arr, dy, axis=0), dx, axis=1)
        diff = (arr - shifted).reshape(-1)  # flatten all channels + pixels
        feats.extend([
            float(diff.mean()),
            float(diff.std()),
            _skewness(diff),
            _kurtosis(diff),
            float(np.abs(diff).max()),
        ])
    return np.asarray(feats, dtype=np.float64)


def combined_features(img: Image.Image) -> np.ndarray:
    """Concatenate spectral + NPR features → FEATURE_DIM vector."""
    spec = spectral_features_only(img)
    npr = npr_features(img)
    return np.concatenate([spec, npr])


# Alias kept for backwards compat with train.train_spectral.
spectral_features = combined_features


# ── Head fit / load / infer ──────────────────────────────────────────────────
def fit_head(
    X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None,
    Cs=(0.01, 0.1, 1.0, 10.0), seed: int = 0,
) -> Dict:
    from sklearn.linear_model import LogisticRegression
    from detector import metrics as MT

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    g = np.asarray(groups) if groups is not None else np.arange(len(y))
    rng = np.random.default_rng(seed)
    va = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        ug = np.unique(g[y == c])
        rng.shuffle(ug)
        k = min(max(1, int(len(ug) * 0.25)), max(len(ug) - 1, 0))
        va |= np.isin(g, ug[:k]) & (y == c)
    leaky = False
    if len(np.unique(y[va])) < 2 or len(np.unique(y[~va])) < 2:
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
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
        clf.fit((X[tr] - mean) / scale, y[tr])
        a = MT.auc(y[va], clf.decision_function((X[va] - mean) / scale)) \
            if len(np.unique(y[va])) > 1 else float("nan")
        s = a if a == a else -1
        if best is None or s > best[0]:
            best = (s, C, a)
    _, C, auc = best
    # Refit on all data for deployment.
    mean, scale = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    clf.fit((X - mean) / scale, y)
    return {
        "w": clf.coef_[0], "b": float(clf.intercept_[0]),
        "mean": mean, "scale": scale, "C": float(C),
        "val_auc": float(auc) if auc == auc else 0.5,
        "group_split": not leaky,
        "feature_dim": int(X.shape[1]),
        "has_npr": True,   # flag so old heads (53 dims) can be detected
    }


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
    # Handle old heads (53-dim, without NPR).
    w = head["w"]
    if len(w) != len(feats):
        if len(w) == N_BINS + 5 and len(feats) == FEATURE_DIM:
            # Old head trained without NPR — use only spectral portion.
            feats = feats[:N_BINS + 5]
        else:
            # Dimension mismatch; skip.
            return 0.0
    z = (np.asarray(feats) - head["mean"]) / head["scale"]
    return float(z @ w + float(head["b"]))


def spectral_evidence(
    img: Image.Image, head: Optional[Dict],
    compression: str = "low",
) -> Evidence:
    """Combined NPR + spectral evidence. Drop-in replacement for v3."""
    if head is None:
        return skipped(
            "spectral",
            "no trained spectral/NPR head (run python -m train.train_spectral)",
            False,
        )
    if min(img.size) < 96:
        return skipped("spectral", "image too small for spectral/NPR analysis", False)

    # Use combined features if the head supports them.
    has_npr = bool(head.get("has_npr", False))
    if has_npr:
        f = combined_features(img)
    else:
        f = spectral_features_only(img)

    z = float(np.clip(head_logit(head, f), -8, 8))

    # JPEG re-compression and downscaling erase upsampling traces.
    rel = {"low": 0.9, "medium": 0.6, "high": 0.3}.get(compression, 0.6)
    va = float(head.get("val_auc", 0.5))
    va = va if va == va else 0.5
    rel *= float(np.clip((va - 0.5) * 2.0, 0.0, 1.0))

    spec_part = f[:N_BINS]
    slope = float(f[N_BINS]) if len(f) > N_BINS else 0.0
    peak = float(f[N_BINS + 3]) if len(f) > N_BINS + 3 else 0.0

    detail: Dict = {
        "slope": round(slope, 3),
        "periodic_peak_score": round(peak, 2),
        "head_val_auc": round(va, 3),
        "has_npr_features": has_npr,
    }
    if has_npr:
        # Add NPR anisotropy score (std of per-direction means).
        npr_part = f[N_BINS + 5:]
        direction_means = npr_part.reshape(NPR_DIRS, NPR_STATS)[:, 0]
        anisotropy = float(np.std(direction_means))
        detail["npr_anisotropy"] = round(anisotropy, 4)

    return Evidence(
        "spectral", z, rel, heuristic=False, detail=detail,
    )
