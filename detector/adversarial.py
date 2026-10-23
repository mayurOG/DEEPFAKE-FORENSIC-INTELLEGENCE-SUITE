"""Black-box evasion stress test.

Measures how easily an attacker who can only query the detector can flip a
verdict with a small L_inf perturbation: random-sign search with greedy
acceptance, the weakest meaningful query attack. A detector that flips under
it is fragile; passing it does NOT prove robustness against adaptive or
white-box attacks. Report it as a lower bound on fragility.
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np
from PIL import Image


def _apply(base: np.ndarray, delta: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(base + delta, 0, 255).astype(np.uint8))


def evasion_attack(score_fn: Callable[[Image.Image], float], img: Image.Image,
                   eps: float = 4.0, queries: int = 60, block: int = 16,
                   seed: int = 0) -> Dict:
    """``score_fn`` returns the fake log-odds. The attack tries to push the
    score across zero (towards the opposite call) within ``eps`` (0-255)."""
    rng = np.random.default_rng(seed)
    base = np.asarray(img.convert("RGB"), dtype=np.float64)
    s0 = float(score_fn(img))
    sign = 1.0 if s0 > 0 else -1.0          # we want to reduce sign * score
    delta = np.zeros_like(base)
    best, used = s0, 1
    h, w, _ = base.shape
    for _ in range(queries):
        trial = delta.copy()
        y, x = rng.integers(0, max(1, h - block + 1)), rng.integers(0, max(1, w - block + 1))
        trial[y:y + block, x:x + block, :] = rng.choice([-eps, eps], size=(1, 1, 3))
        trial = np.clip(trial, -eps, eps)
        s = float(score_fn(_apply(base, trial)))
        used += 1
        if sign * s < sign * best:
            delta, best = trial, s
        if sign * best < 0:
            break
    return {"initial": s0, "final": best, "flipped": bool(sign * best < 0),
            "queries": used, "eps": eps, "drop": float(sign * (s0 - best))}


def robustness_curve(score_fn, images: Sequence[Image.Image],
                     eps_grid: Sequence[float] = (1, 2, 4, 8), **kw) -> Dict:
    rows = []
    for e in eps_grid:
        res = [evasion_attack(score_fn, im, eps=float(e), **kw) for im in images]
        rows.append({"eps": float(e),
                     "flip_rate": float(np.mean([r["flipped"] for r in res])),
                     "mean_drop": float(np.mean([r["drop"] for r in res])),
                     "mean_queries": float(np.mean([r["queries"] for r in res]))})
    return {"rows": rows, "n": len(images),
            "note": "Black-box random search: a lower bound on fragility."}
