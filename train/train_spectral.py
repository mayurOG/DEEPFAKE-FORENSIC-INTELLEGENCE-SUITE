"""Train the spectral-fingerprint head (CPU, no GPU or pretrained model).

    python -m train.train_spectral --data data/ffpp --out artifacts

Uses the whole image (not just the face): generator upsampling traces live in
the global spectrum. Splits are group-aware. Run it on every dataset mix you
calibrate on; it is cheap.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np

from detector.experts.spectral import fit_head, save_head, spectral_features
from train.data import load_images, scan_folder

log = logging.getLogger("deepfake.train")


def build(items, frames: int = 4):
    X, y, g = [], [], []
    for it in items:
        try:
            for im in load_images(it, frames):
                X.append(spectral_features(im))
                y.append(it.label)
                g.append(it.group)
        except Exception as e:
            log.warning("skip %s: %s", it.path, e)
    return np.array(X), np.array(y), np.array(g)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, action="append")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--max-items", type=int, default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    items = []
    for d in a.data:
        items += scan_folder(d, max_items=a.max_items)
    X, y, g = build(items, a.frames)
    if len(np.unique(y)) < 2:
        raise SystemExit("Need both real/ and fake/ examples.")
    head = fit_head(X, y, g)
    path = os.path.join(a.out, "spectral_head.npz")
    save_head(head, path)
    print(json.dumps({"n": int(len(y)), "val_auc": head["val_auc"], "saved": path}))
    return head


if __name__ == "__main__":
    main()
