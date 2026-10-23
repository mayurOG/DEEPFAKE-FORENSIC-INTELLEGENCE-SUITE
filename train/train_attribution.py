"""Train the open-set generator-family attributor on spectral fingerprints.

Layout: ``data/<family>/*.png`` where each folder name is a family label,
e.g. ``real``, ``gan``, ``diffusion``, ``faceswap``. CPU, seconds.

    python -m train.train_attribution --data data/attribution --out artifacts
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

from detector.attribution import fit_attributor, predict_family, save_attributor
from detector.experts.spectral import spectral_features

EXT = ("png", "jpg", "jpeg", "webp", "bmp")


def collect(root: str):
    X, fam = [], []
    for d in sorted(os.listdir(root)):
        folder = os.path.join(root, d)
        if not os.path.isdir(folder):
            continue
        for ext in EXT:
            for f in glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True):
                try:
                    v = spectral_features(Image.open(f).convert("RGB"))
                except Exception:
                    continue
                if np.any(v):
                    X.append(v)
                    fam.append(d)
    return np.array(X), fam


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--unknown-below", type=float, default=0.5)
    ap.add_argument("--holdout", type=float, default=0.2)
    a = ap.parse_args(argv)
    X, fam = collect(a.data)
    fam = np.array(fam)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * a.holdout)
    va, tr = idx[:n_val], idx[n_val:]
    head = fit_attributor(X[tr], fam[tr], a.unknown_below)
    acc = float(np.mean([predict_family(head, X[i])["top"] == fam[i] for i in va])) if n_val else None
    os.makedirs(a.out, exist_ok=True)
    save_attributor(head, os.path.join(a.out, "attribution_head.npz"))
    summ = {"classes": head["classes"], "n_train": int(len(tr)), "n_val": int(n_val),
            "val_top1": acc,
            "note": "Random (not group-aware) holdout: use a different generator "
                    "checkpoint for the real test."}
    json.dump(summ, open(os.path.join(a.out, "attribution_summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
