"""Train the Generator Attribution head — v4.0 (NEW).

Identifies WHICH generator created a suspect image or audio clip.
Supporting context only — never changes the real/fake verdict.

Usage:
    # Image attribution (GAN vs Diffusion vs Face_Swap vs ...):
    python -m train.train_attribution \\
        --data data/ffpp/fake \\
        --label GAN \\
        --data data/sd_outputs \\
        --label Diffusion \\
        --data data/deepfacelab \\
        --label Face_Swap \\
        --out artifacts

Output: artifacts/attribution_image_head.npz

The head is loaded by detector.attribution.load_attributor() and called
from pipeline._post() when the verdict is Fake.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List

import numpy as np

from detector.attribution import (
    IMAGE_FAMILIES,
    fit_attribution_head,
    save_attributor,
)
from detector.experts.spectral import combined_features
from train.data import load_images, scan_folder

log = logging.getLogger("deepfake.train_attribution")


def build_image_features(data_roots: List[str], labels: List[str],
                         frames: int = 4) -> tuple:
    family_to_idx = {f: i for i, f in enumerate(IMAGE_FAMILIES)}
    X, y = [], []
    for root, lbl in zip(data_roots, labels):
        if lbl not in family_to_idx:
            log.warning("Label %s not in IMAGE_FAMILIES %s; skipping.", lbl, IMAGE_FAMILIES)
            continue
        idx = family_to_idx[lbl]
        items = scan_folder(root, max_items=None)
        count = 0
        for it in items:
            try:
                for im in load_images(it, frames):
                    feats = combined_features(im)
                    X.append(feats)
                    y.append(idx)
                    count += 1
            except Exception as e:
                log.warning("skip %s: %s", it.path, e)
        log.info("Generator '%s': %d feature vectors extracted.", lbl, count)

    if not X:
        raise SystemExit("No features extracted. Check that --data paths are valid.")

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=int)
    used = sorted(set(y.tolist()))
    families_present = [IMAGE_FAMILIES[i] for i in used]
    remap = {old: new for new, old in enumerate(used)}
    y = np.array([remap[v] for v in y], dtype=int)
    return X, y, families_present


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, action="append")
    ap.add_argument("--label", required=True, action="append",
                    choices=list(IMAGE_FAMILIES))
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--C", type=float, default=1.0)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if len(a.data) != len(a.label):
        raise SystemExit("--data and --label must be provided the same number of times.")

    log.info("Building image attribution features...")
    X, y, families = build_image_features(a.data, a.label, a.frames)
    log.info("Total: %d samples, %d classes: %s", len(y), len(families), families)

    if len(np.unique(y)) < 2:
        raise SystemExit("Need at least 2 generator classes.")

    log.info("Training multi-class attribution head (C=%.3f)...", a.C)
    head = fit_attribution_head(X, y, families, media="image", C=a.C)

    path = os.path.join(a.out, "attribution_image_head.npz")
    save_attributor(head, path)
    log.info("Saved attribution head to %s", path)

    result = {
        "n_samples": int(len(y)),
        "families": families,
        "C": float(a.C),
        "saved": path,
    }
    print(json.dumps(result, indent=2))
    return head


if __name__ == "__main__":
    main()
