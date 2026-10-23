"""Train a linear probe on cached frozen-encoder embeddings.

Recent work finds that frozen foundation features + a linear head
generalise better to unseen generators than fully fine-tuned or LoRA-tuned
detectors, and training takes seconds on CPU once features are cached.

    python -m train.features --data data/ffpp --encoder <model> --out feats_ffpp.npz
    python -m train.train_probe --train feats_ffpp.npz --out artifacts/probe.npz
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from detector import metrics as MT
from detector.models import l2_normalize


def _group_val_split(groups: np.ndarray, frac: float = 0.2, seed: int = 0):
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(ug)
    val_g = set(ug[: max(1, int(len(ug) * frac))])
    mask = np.array([g in val_g for g in groups])
    return ~mask, mask


def fit_probe(X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              Cs=(0.01, 0.1, 1.0, 10.0), seed: int = 0) -> Dict:
    """Returns dict(w, b, mean, scale, C, val_auc). Embeddings are L2
    normalised then standardised; class weights balance real/fake."""
    Xn = l2_normalize(np.asarray(X, dtype=np.float64))
    if X_val is None:
        g = groups if groups is not None else np.arange(len(y))
        tr, va = _group_val_split(g, 0.2, seed)
        Xt, yt, Xv, yv = Xn[tr], y[tr], Xn[va], y[va]
    else:
        Xt, yt = Xn, y
        Xv, yv = l2_normalize(np.asarray(X_val, dtype=np.float64)), y_val
    mean = Xt.mean(axis=0)
    scale = Xt.std(axis=0) + 1e-6
    Xt_s, Xv_s = (Xt - mean) / scale, (Xv - mean) / scale
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
        clf.fit(Xt_s, yt)
        auc = MT.auc(yv, clf.decision_function(Xv_s)) if len(np.unique(yv)) > 1 else float("nan")
        score = auc if auc == auc else -1
        if best is None or score > best[0]:
            best = (score, C, clf, auc)
    _, C, clf, auc = best
    # refit on train+val for the final model
    Xa = np.concatenate([Xt, Xv])
    ya = np.concatenate([yt, yv])
    mean, scale = Xa.mean(axis=0), Xa.std(axis=0) + 1e-6
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    clf.fit((Xa - mean) / scale, ya)
    return {"w": clf.coef_[0], "b": float(clf.intercept_[0]), "mean": mean,
            "scale": scale, "C": float(C), "val_auc": float(auc)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="features .npz from extract_features")
    ap.add_argument("--val", default=None, help="optional separate validation .npz")
    ap.add_argument("--out", default="artifacts/probe.npz")
    a = ap.parse_args(argv)
    tr = np.load(a.train, allow_pickle=True)
    va = np.load(a.val, allow_pickle=True) if a.val else None
    res = fit_probe(tr["emb"], tr["y"], tr["groups"],
                    None if va is None else va["emb"], None if va is None else va["y"])
    import os
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, w=res["w"], b=res["b"], mean=res["mean"], scale=res["scale"],
             encoder=str(tr["encoder"]))
    print(json.dumps({k: res[k] for k in ("C", "val_auc")}))
    print(f"saved probe to {a.out}. Set EMBED_MODEL_ID={tr['encoder']} to use it.")


if __name__ == "__main__":
    main()
