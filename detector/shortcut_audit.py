"""Causal shortcut and evidence-reliability audit (CALM-FORGE Contribution B).

Applies controlled *benign* transformations (resolution, JPEG, blur, codec-like
noise) and measures whether the detector verdict/score changes. The goal is to
separate:

* instability under nuisance factors (possible shortcut reliance), from
* stability that still reacts to manipulation-relevant cues (when known).

This reuses transformation_lab operators where possible and records a structured
audit suitable for reports and research tables.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

log = logging.getLogger("deepfake.shortcut_audit")


@dataclass
class TransformResult:
    name: str
    severity: float  # 0 mild → 1 severe
    p_fake_before: float
    p_fake_after: float
    verdict_before: str
    verdict_after: str
    flipped: bool
    score_delta: float
    category: str  # nuisance | content_preserving | aggressive

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShortcutAuditReport:
    asset_id: Optional[str]
    baseline_verdict: str
    baseline_p_fake: float
    results: List[TransformResult] = field(default_factory=list)
    fragility: float = 0.0
    nuisance_flip_rate: float = 0.0
    likely_shortcut_reliance: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "baseline_verdict": self.baseline_verdict,
            "baseline_p_fake": self.baseline_p_fake,
            "results": [r.to_dict() for r in self.results],
            "fragility": self.fragility,
            "nuisance_flip_rate": self.nuisance_flip_rate,
            "likely_shortcut_reliance": self.likely_shortcut_reliance,
            "notes": self.notes,
            "interpretation": (
                "High nuisance flip rate suggests the decision may rely on "
                "compression/resolution shortcuts rather than stable manipulation "
                "evidence. This is a detector-behaviour audit, not proof about the media."
            ),
        }


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    import io
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _resize(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    nw, nh = max(8, int(w * scale)), max(8, int(h * scale))
    out = img.resize((nw, nh), Image.Resampling.BILINEAR)
    return out.resize((w, h), Image.Resampling.BILINEAR)


def _blur(img: Image.Image, radius: float) -> Image.Image:
    from PIL import ImageFilter
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr = np.clip(arr + np.random.default_rng(0).normal(0, sigma, arr.shape), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


# name, severity, category, factory(img) -> img
NUISANCE_TRANSFORMS: List[Tuple[str, float, str, Callable[[Image.Image], Image.Image]]] = [
    ("jpeg_q75", 0.3, "nuisance", lambda im: _jpeg(im, 75)),
    ("jpeg_q40", 0.55, "nuisance", lambda im: _jpeg(im, 40)),
    ("resize_0.5", 0.4, "nuisance", lambda im: _resize(im, 0.5)),
    ("blur_1.5", 0.45, "nuisance", lambda im: _blur(im, 1.5)),
    ("noise_8", 0.35, "content_preserving", lambda im: _noise(im, 8)),
]


def audit_image(
    img: Image.Image,
    score_fn: Callable[[Image.Image], Tuple[float, str]],
    asset_id: Optional[str] = None,
    transforms: Optional[List] = None,
) -> ShortcutAuditReport:
    """score_fn(image) -> (p_fake, verdict)."""
    base_p, base_v = score_fn(img)
    report = ShortcutAuditReport(
        asset_id=asset_id,
        baseline_verdict=base_v,
        baseline_p_fake=float(base_p),
    )
    ops = transforms or NUISANCE_TRANSFORMS
    weighted_flip = 0.0
    weight_sum = 0.0
    nuisance_flips = 0
    nuisance_n = 0

    for name, severity, category, fn in ops:
        try:
            mutated = fn(img.copy())
            p, v = score_fn(mutated)
        except Exception as e:
            log.warning("transform %s failed: %s", name, e)
            continue
        delta = float(p - base_p)
        # Any verdict change (including to/from Inconclusive) counts as a flip
        # for fragility. Large score moves without a verdict change are tracked
        # via score_delta but do not alone mark flipped.
        verdict_flipped = (v != base_v)
        tr = TransformResult(
            name=name, severity=severity,
            p_fake_before=float(base_p), p_fake_after=float(p),
            verdict_before=base_v, verdict_after=v,
            flipped=verdict_flipped,
            score_delta=delta,
            category=category,
        )
        report.results.append(tr)
        # Milder flips weigh more toward fragility
        w = max(0.05, 1.0 - severity)
        weight_sum += w
        if tr.flipped:
            weighted_flip += w
        if category == "nuisance":
            nuisance_n += 1
            if tr.flipped:
                nuisance_flips += 1

    report.fragility = float(weighted_flip / weight_sum) if weight_sum else 0.0
    report.nuisance_flip_rate = (
        float(nuisance_flips / nuisance_n) if nuisance_n else 0.0
    )
    report.likely_shortcut_reliance = (
        report.nuisance_flip_rate >= 0.5 or report.fragility >= 0.4
    )
    if report.likely_shortcut_reliance:
        report.notes.append(
            "Decision changes under mild JPEG/resize/blur — treat evidence as less reliable."
        )
    else:
        report.notes.append(
            "Verdict relatively stable under tested nuisance transforms."
        )
    return report


def audit_with_pipeline(pipe, path: str, asset_id: Optional[str] = None) -> ShortcutAuditReport:
    """Convenience: score via Pipeline.analyze on temp-saved transforms."""
    import os
    import tempfile

    img = Image.open(path).convert("RGB")

    def score_fn(im: Image.Image):
        fd, tmp = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            im.save(tmp, quality=95)
            res = pipe.analyze(tmp, media_type="image", explain=False, stability=False)
            return float(res.p_fake), str(res.verdict)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    return audit_image(img, score_fn, asset_id=asset_id)
