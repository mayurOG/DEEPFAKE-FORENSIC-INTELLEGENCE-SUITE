"""Local-edit / noise-inconsistency expert.

Foundation-model detectors are weak on localized edits. A real camera leaves
a fairly uniform sensor-noise residual across the frame; pasted or
regenerated regions often have a different noise level. This expert maps the
residual's block-wise level and flags outlier blocks and a face-vs-background
mismatch.

It is a *heuristic* (weak weight in fusion) and is automatically distrusted
on heavily compressed images, where sensor noise is destroyed.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

from detector.fusion import Evidence, skipped

_SRM = np.array([
    [0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0],
    [0, -1, 2, -1, 0], [0, 0, 0, 0, 0],
], dtype=np.float32) / 4.0


def noise_residual(gray: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.filter2D(gray.astype(np.float32), -1, _SRM)


def tamper_map(
    img: Image.Image, face_box: Optional[Tuple[int, int, int, int]] = None,
    grid: int = 8, max_side: int = 768,
) -> dict:
    import cv2

    im = img.convert("RGB")
    scale = min(1.0, max_side / max(im.size))
    if scale < 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)),
                       Image.BILINEAR)
        if face_box is not None:
            face_box = tuple(int(v * scale) for v in face_box)
    gray = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
    res = noise_residual(gray)
    H, W = res.shape
    levels = np.zeros((grid, grid), dtype=np.float64)
    for i in range(grid):
        for j in range(grid):
            blk = res[i * H // grid:(i + 1) * H // grid,
                      j * W // grid:(j + 1) * W // grid]
            levels[i, j] = float(blk.std()) if blk.size else 0.0
    med = float(np.median(levels))
    mad = float(np.median(np.abs(levels - med))) * 1.4826 + 1e-6
    z = (levels - med) / mad
    out = {
        "block_levels": levels,
        "robust_z": z,
        "outlier_fraction": float((np.abs(z) > 3.0).mean()),
        "max_abs_z": float(np.abs(z).max()),
        "face_bg_log_ratio": None,
    }
    if face_box is not None:
        x1, y1, x2, y2 = [int(v) for v in face_box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 > 16 and y2 - y1 > 16:
            mask = np.zeros_like(res, dtype=bool)
            mask[y1:y2, x1:x2] = True
            inside, outside = res[mask].std(), res[~mask].std()
            if outside > 1e-6 and inside > 1e-6:
                out["face_bg_log_ratio"] = float(np.log(inside / outside))
    return out


def tamper_heatmap_image(tm: dict, size: Tuple[int, int]) -> Image.Image:
    """Render the robust-z map as a red-heat overlay image."""
    z = np.clip(np.abs(tm["robust_z"]) / 6.0, 0, 1)
    rgb = np.zeros(z.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = (255 * z).astype(np.uint8)
    rgb[..., 1] = (80 * (1 - z)).astype(np.uint8)
    rgb[..., 2] = (160 * (1 - z)).astype(np.uint8)
    return Image.fromarray(rgb).resize(size, Image.NEAREST)


def tamper_evidence(
    img: Image.Image, face_box=None, compression: str = "low",
) -> Evidence:
    if min(img.size) < 96:
        return skipped("tamper", "image too small for noise analysis", True)
    tm = tamper_map(img, face_box)
    ratio = tm["face_bg_log_ratio"]
    raw = 4.0 * tm["outlier_fraction"]
    if ratio is not None:
        raw += 1.5 * max(abs(ratio) - 0.4, 0.0)
    logit = float(np.clip(raw - 0.5, -1.5, 3.0))
    rel = {"low": 0.6, "medium": 0.3, "high": 0.1}.get(compression, 0.3)
    return Evidence(
        name="tamper", logit=logit, reliability=rel, heuristic=True,
        detail={
            "outlier_fraction": round(tm["outlier_fraction"], 4),
            "max_abs_z": round(tm["max_abs_z"], 2),
            "face_bg_log_ratio": None if ratio is None else round(ratio, 3),
        },
    )
