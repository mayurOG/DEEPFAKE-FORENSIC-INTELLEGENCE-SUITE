"""Verdict-stability stress test.

Replays the input through a "social-media laundering" battery (JPEG
re-encoding, downscaling, blur, sensor-like noise) and reports how many
degraded copies keep the same call. A fake that only flips under compression
is exactly the case a user should be warned about.
"""
from __future__ import annotations

import io
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter


def _jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _downscale(img: Image.Image, f: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def _noise(img: Image.Image, sigma: float, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    a = a + rng.normal(0, sigma, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


PERTURBATIONS: Dict[str, Callable[[Image.Image], Image.Image]] = {
    "jpeg_q75": lambda im: _jpeg(im, 75),
    "jpeg_q50": lambda im: _jpeg(im, 50),
    "downscale_0.5": lambda im: _downscale(im, 0.5),
    "blur_r1": lambda im: im.convert("RGB").filter(ImageFilter.GaussianBlur(1.0)),
    "noise_s4": lambda im: _noise(im, 4.0),
}


def stability_report(
    images: Sequence[Image.Image],
    score_fn: Callable[[List[Image.Image]], float],
    base_p: Optional[float] = None,
    decision: float = 0.5,
    names: Optional[Sequence[str]] = None,
) -> Dict:
    """``score_fn(list_of_images) -> p_fake`` (one scalar for the whole
    input: all face crops of an image, or the sampled frames of a video).

    Returns the stability score = share of perturbed copies whose call
    (p_fake >= decision) matches the base call, plus per-perturbation p.
    """
    if base_p is None:
        base_p = float(score_fn(list(images)))
    base_call = base_p >= decision
    per: Dict[str, float] = {}
    same = 0
    chosen = list(names) if names else list(PERTURBATIONS)
    for name in chosen:
        fn = PERTURBATIONS[name]
        p = float(score_fn([fn(im) for im in images]))
        per[name] = p
        same += int((p >= decision) == base_call)
    ps = list(per.values())
    return {
        "stability_score": same / len(chosen) if chosen else None,
        "base_p_fake": float(base_p),
        "perturbed_p_fake": per,
        "p_min": float(min(ps)) if ps else None,
        "p_max": float(max(ps)) if ps else None,
        "flips": [n for n, p in per.items() if (p >= decision) != base_call],
    }
