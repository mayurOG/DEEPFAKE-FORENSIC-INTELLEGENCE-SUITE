"""Test-time augmentation with an explicit uncertainty readout.

Scoring a handful of label-preserving views of the same crop (original,
mirror, tight centre crop) does two things:

1. averaging the logits is a free accuracy / stability gain for frozen
   backbones, and
2. the *spread* across views is a cheap epistemic-uncertainty signal: a
   detector that flips its call under a mirror or a 10% crop is not looking
   at robust evidence. The spread lowers the evidence reliability and blocks
   the Tier-1 early exit, so unstable cases go to the full cascade.

Views never change colour statistics or add noise, to avoid shifting the
distribution the calibrator was fitted on.
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

VIEW_NAMES = ("original", "mirror", "centre_crop")


def make_views(img: Image.Image, n_views: int = 3) -> List[Image.Image]:
    img = img.convert("RGB")
    views = [img]
    if n_views >= 2:
        views.append(ImageOps.mirror(img))
    if n_views >= 3:
        w, h = img.size
        dx, dy = int(w * 0.05), int(h * 0.05)
        if w - 2 * dx >= 32 and h - 2 * dy >= 32:
            views.append(img.crop((dx, dy, w - dx, h - dy)).resize((w, h), Image.BICUBIC))
        else:
            views.append(img)
    return views[: max(1, n_views)]


def tta_logit(
    predict: Callable[[Sequence[Image.Image]], Tuple[np.ndarray, np.ndarray]],
    crops: Sequence[Image.Image],
    aggregate: Callable[[np.ndarray], float],
    n_views: int = 3,
) -> dict:
    """Score every crop under every view.

    Returns dict(logit, view_logits, spread, per_face (original view),
    embedding (original view)).
    """
    per_view = [make_views(c, n_views) for c in crops]
    n = min(len(v) for v in per_view)
    view_logits, per_face, emb = [], None, None
    for k in range(n):
        z, e = predict([v[k] for v in per_view])
        z = np.asarray(z, dtype=np.float64)
        view_logits.append(float(aggregate(z)))
        if k == 0:
            per_face, emb = z, e
    vl = np.array(view_logits)
    return {
        "logit": float(vl.mean()),
        "view_logits": [round(float(x), 3) for x in vl],
        "spread": float(vl.std()) if len(vl) > 1 else 0.0,
        "n_views": int(n),
        "per_face": per_face,
        "embedding": emb,
        "sign_flip": bool(len(vl) > 1 and (vl.max() > 0 > vl.min())),
    }


def reliability_from_spread(spread: float, scale: float = 2.0) -> float:
    """1.0 when views agree, ->0 as the spread grows past ``scale`` logits."""
    return float(1.0 / (1.0 + (spread / scale) ** 2))
