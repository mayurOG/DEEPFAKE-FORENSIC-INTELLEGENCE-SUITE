"""Train the NPR + Spectral fingerprint head — v4.0.

    python -m train.train_spectral --data data/ffpp --out artifacts

v4.0 upgrade over v3:
  * Uses ``combined_features()`` = spectral (53 dims) + NPR (40 dims) = 93 dims.
  * The saved head includes ``has_npr=True`` so the pipeline knows which
    feature extractor to use at inference.
  * Reports NPR anisotropy on the validation set (diagnostic).
  * Backward compatible: old 53-dim heads still work (pipeline checks dim).

The head is CPU-only (no GPU required). It is a logistic regression over
the combined spectral+NPR feature vector. Training on ~2000 images per class
takes ~30 seconds on a laptop.

Data layout expected by ``scan_folder``:
    data/ffpp/
        real/   ← real images or video frames
        fake/   ← deepfake images or video frames
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np

from detector.experts.spectral import (
    combined_features,
    fit_head,
    save_head,
    FEATURE_DIM,
)
from train.data import load_images, scan_folder

log = logging.getLogger("deepfake.train")


def build(items, frames: int = 4):
    """Extract combined NPR+spectral features from all items.

    Returns:
        X: (n, FEATURE_DIM) float64 feature matrix
        y: (n,) int labels (1=fake, 0=real)
        g: (n,) group IDs (for group-aware validation split)
    """
    X, y, g = [], [], []
    for it in items:
        try:
            for im in load_images(it, frames):
                feats = combined_features(im)   # 93-dim NPR+spectral vector
                X.append(feats)
                y.append(it.label)
                g.append(it.group)
        except Exception as e:
            log.warning("skip %s: %s", it.path, e)
    return np.array(X, dtype=np.float64), np.array(y, dtype=int), np.array(g)


def _anisotropy_diagnostic(X: np.ndarray, y: np.ndarray):
    """Report average NPR anisotropy by class (diagnostic only)."""
    from detector.experts.spectral import N_BINS, NPR_DIRS, NPR_STATS
    spec_dim = N_BINS + 5
    npr_start = spec_dim
    if X.shape[1] <= npr_start:
        return  # old-style features without NPR
    npr = X[:, npr_start:].reshape(len(X), NPR_DIRS, NPR_STATS)
    direction_means = npr[:, :, 0]  # (n, 8) per-direction means
    anisotropy = np.std(direction_means, axis=1)  # (n,) per-image
    for label, name in [(0, "real"), (1, "fake")]:
        mask = y == label
        if mask.any():
            log.info("NPR anisotropy [%s]: mean=%.4f ± %.4f",
                     name, anisotropy[mask].mean(), anisotropy[mask].std())


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", required=True, action="append",
                    help="Root folder(s) containing real/ and fake/ subdirectories.")
    ap.add_argument("--out", default="artifacts",
                    help="Output directory for the trained head.")
    ap.add_argument("--frames", type=int, default=4,
                    help="Number of frames to sample per video clip.")
    ap.add_argument("--max-items", type=int, default=None,
                    help="Maximum number of items per class (for quick experiments).")
    ap.add_argument("--no-npr", action="store_true",
                    help="Disable NPR features (use spectral-only, for comparison).")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    items = []
    for d in a.data:
        items += scan_folder(d, max_items=a.max_items)
    log.info("Loaded %d image/clip items from %d data roots.", len(items), len(a.data))

    if a.no_npr:
        # Spectral-only mode (for ablation studies).
        from detector.experts.spectral import spectral_features_only
        X, y, g = [], [], []
        for it in items:
            try:
                for im in load_images(it, a.frames):
                    X.append(spectral_features_only(im))
                    y.append(it.label)
                    g.append(it.group)
            except Exception as e:
                log.warning("skip %s: %s", it.path, e)
        X, y, g = np.array(X, dtype=np.float64), np.array(y, dtype=int), np.array(g)
        log.info("Spectral-only features: %d samples, dim=%d", len(y), X.shape[1] if len(X) else 0)
    else:
        X, y, g = build(items, a.frames)
        log.info("NPR+Spectral features: %d samples, dim=%d", len(y), X.shape[1] if len(X) else 0)
        if len(y):
            _anisotropy_diagnostic(X, y)

    if len(y) == 0:
        raise SystemExit("No features extracted. Check that --data points to a valid folder.")
    if len(np.unique(y)) < 2:
        raise SystemExit("Need both real/ and fake/ examples in --data.")

    log.info("Training head (logistic regression, group-aware CV)...")
    head = fit_head(X, y, g)
    head["has_npr"] = not a.no_npr  # flag for backward compatibility

    path = os.path.join(a.out, "spectral_head.npz")
    save_head(head, path)
    log.info("Saved head to %s", path)

    result = {
        "n_samples": int(len(y)),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "feature_dim": int(X.shape[1]),
        "has_npr": head["has_npr"],
        "val_auc": round(head["val_auc"], 4),
        "best_C": head["C"],
        "saved": path,
    }
    print(json.dumps(result, indent=2))
    return head


if __name__ == "__main__":
    main()
