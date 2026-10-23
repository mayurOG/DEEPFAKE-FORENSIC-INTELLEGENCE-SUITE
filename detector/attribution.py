"""Generator-family attribution (open-set).

Says *which kind* of generator likely produced a fake (e.g. GAN, diffusion,
face-swap, real camera), using the backbone-independent spectral fingerprint.
Trained with ``train.train_attribution`` on folders ``data/<family>/*.png``.

Open-set: if the top class probability is below ``unknown_below`` or the
sample is far from every class centroid, the answer is "unknown" rather than a
forced label. Without a trained head the module returns ``None`` (skipped,
never guessed). Attribution is supporting context in the evidence graph; it
never changes the real/fake verdict by itself.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from detector.experts.spectral import spectral_features


def fit_attributor(X: np.ndarray, families: Sequence[str],
                   unknown_below: float = 0.5, C: float = 1.0) -> Dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float64)
    fam = np.asarray(list(families))
    classes = sorted(set(fam.tolist()))
    if len(classes) < 2:
        raise ValueError("need at least two families")
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    clf = LogisticRegression(C=C, max_iter=2000).fit(Z, fam)
    cent = np.stack([Z[fam == c].mean(0) for c in classes])
    d = np.array([np.linalg.norm(Z[i] - cent[classes.index(fam[i])]) for i in range(len(Z))])
    return {"classes": classes, "mean": sc.mean_, "scale": sc.scale_,
            "coef": clf.coef_, "intercept": clf.intercept_, "centroids": cent,
            "dist_limit": float(np.quantile(d, 0.99) * 1.25),
            "unknown_below": float(unknown_below)}


def predict_family(head: Dict, feats: np.ndarray) -> Dict:
    z = (np.asarray(feats, dtype=np.float64) - head["mean"]) / head["scale"]
    logits = head["coef"] @ z + head["intercept"]
    if len(head["classes"]) == 2:      # sklearn stores one row for binary
        logits = np.array([-logits[0], logits[0]]) if logits.shape[0] == 1 else logits
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    dist = float(np.min(np.linalg.norm(head["centroids"] - z, axis=1)))
    i = int(np.argmax(p))
    unknown = bool(p[i] < head["unknown_below"] or dist > head["dist_limit"])
    return {"family": "unknown" if unknown else head["classes"][i],
            "top": head["classes"][i], "probability": float(p[i]),
            "distance": dist, "scores": {c: float(v) for c, v in zip(head["classes"], p)}}


def attribute_image(img: Image.Image, head: Optional[Dict]) -> Optional[Dict]:
    if head is None:
        return None
    f = spectral_features(img)
    if not np.any(f):
        return None
    return predict_family(head, f)


def save_attributor(head: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, classes=np.array(head["classes"]), mean=head["mean"],
             scale=head["scale"], coef=head["coef"], intercept=head["intercept"],
             centroids=head["centroids"], dist_limit=head["dist_limit"],
             unknown_below=head["unknown_below"])


def load_attributor(path: str) -> Optional[Dict]:
    if not os.path.isfile(path):
        return None
    z = np.load(path, allow_pickle=False)
    return {"classes": [str(c) for c in z["classes"]], "mean": z["mean"],
            "scale": z["scale"], "coef": z["coef"], "intercept": z["intercept"],
            "centroids": z["centroids"], "dist_limit": float(z["dist_limit"]),
            "unknown_below": float(z["unknown_below"])}
