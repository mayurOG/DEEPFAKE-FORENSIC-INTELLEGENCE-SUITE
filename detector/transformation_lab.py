"""Counterfactual media transformation laboratory.

Re-encoding a file and re-scoring it is easy; the contribution here is what
is *done* with the results:

1. A **robustness profile** per input: for every controlled transformation
   (JPEG at two strengths, downscale, blur, centre crop, noise, and a
   combined social-media pipeline; resample / noise / low-pass / 8-bit
   quantisation for audio) the score under the transformation, the share of
   the original *evidence retained* (same-direction logit magnitude), and
   whether the call flips.
2. A **fragility score** in [0, 1] that weights flips by how *mild* the
   transformation is (a call that flips under a light JPEG is more fragile
   than one that flips only under heavy blur).
3. A **robustness gate** for selective decisions: abstain when the call is
   not stable under transformations that the input could plausibly already
   have undergone. ``plausible_given`` ties this to the media-quality
   profile, so a clean original is not penalised for flipping under a crop
   it never suffered, and an already-compressed file is not penalised for
   JPEG it already has.
4. The evaluation protocol lives in ``train.evaluate_robustness``: does the
   fragility score predict the system's *errors* (AUROC), and does gating on
   it improve selective risk?

All of this is evidence about the detector's behaviour on the input, not
proof about the media.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter

# ---------------------------------------------------------------- image ----


def _jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _resize(img: Image.Image, f: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(16, int(w * f)), max(16, int(h * f))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def _crop(img: Image.Image, keep: float) -> Image.Image:
    w, h = img.size
    dx, dy = int(w * (1 - keep) / 2), int(h * (1 - keep) / 2)
    return img.crop((dx, dy, w - dx, h - dy)).resize((w, h), Image.BICUBIC)


def _noise(img: Image.Image, sigma: float, seed: int = 0) -> Image.Image:
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    n = np.random.default_rng(seed).normal(0, sigma, a.shape)
    return Image.fromarray(np.clip(a + n, 0, 255).astype(np.uint8))


@dataclass(frozen=True)
class Transform:
    name: str
    severity: float          # 0 (barely) .. 1 (destructive)
    fn: Callable[[Any], Any]
    # which compression levels of the *input* already include this effect
    already_in: Sequence[str] = ()

    def plausible_given(self, quality: Optional[Dict[str, Any]]) -> bool:
        """Could the input plausibly have been through this already? If so a
        flip is informative about real-world fragility; if the input is a
        clean original, severe transformations are hypothetical."""
        comp = (quality or {}).get("compression", "unknown")
        return self.severity <= 0.5 or comp in self.already_in or comp == "unknown"


IMAGE_TRANSFORMS: List[Transform] = [
    Transform("jpeg_q85", 0.15, lambda im: _jpeg(im, 85), ("medium", "high")),
    Transform("jpeg_q60", 0.40, lambda im: _jpeg(im, 60), ("high",)),
    Transform("jpeg_q35", 0.70, lambda im: _jpeg(im, 35), ("high",)),
    Transform("downscale_0.75", 0.25, lambda im: _resize(im, 0.75)),
    Transform("downscale_0.5", 0.50, lambda im: _resize(im, 0.5), ("high",)),
    Transform("blur_1.0", 0.30, lambda im: im.convert("RGB").filter(ImageFilter.GaussianBlur(1.0))),
    Transform("blur_2.0", 0.60, lambda im: im.convert("RGB").filter(ImageFilter.GaussianBlur(2.0)), ("high",)),
    Transform("crop_0.9", 0.20, lambda im: _crop(im, 0.9)),
    Transform("noise_4", 0.30, lambda im: _noise(im, 4.0)),
    Transform("social_pipeline", 0.55, lambda im: _jpeg(_resize(im, 0.6), 65), ("high",)),
]


def _as_logits(score_fn, items) -> np.ndarray:
    return np.asarray(score_fn(list(items)), dtype=np.float64).reshape(-1)


def robustness_profile(
    score_fn: Callable[[List[Any]], Sequence[float]], media: Any,
    transforms: Sequence[Transform] = IMAGE_TRANSFORMS,
    quality: Optional[Dict[str, Any]] = None, decision: float = 0.0,
    verdict_fn: Optional[Callable[[float], str]] = None,
) -> Dict[str, Any]:
    """``score_fn(list_of_media) -> logits`` (one scalar per item)."""
    variants = [t.fn(media) for t in transforms]
    z = _as_logits(score_fn, [media] + variants)
    base, zt = float(z[0]), z[1:]
    call = lambda v: (verdict_fn(v) if verdict_fn else ("Fake" if v > decision else "Real"))
    base_call = call(base)
    rows, weight_total, weight_flip, drop = [], 0.0, 0.0, []
    sgn = 1.0 if base > decision else -1.0
    for t, v in zip(transforms, zt):
        v = float(v)
        keep = max(0.0, (v - decision) * sgn) / max(abs(base - decision), 1e-6)
        keep = float(min(keep, 1.0))
        flips = call(v) != base_call
        plaus = t.plausible_given(quality)
        w = (1.0 - t.severity + 0.1) * (1.0 if plaus else 0.25)
        weight_total += w
        weight_flip += w * flips
        drop.append(1.0 - keep)
        rows.append({"transform": t.name, "severity": t.severity, "logit": round(v, 3),
                     "evidence_retained": round(keep, 3), "flips": bool(flips),
                     "plausible": bool(plaus)})
    flip_part = weight_flip / max(weight_total, 1e-9)
    fragility = float(np.clip(0.7 * flip_part + 0.3 * float(np.mean(drop)), 0.0, 1.0))
    return {"base_logit": round(base, 3), "base_call": base_call, "rows": rows,
            "fragile_to": [r["transform"] for r in rows if r["flips"] and r["plausible"]],
            "mean_retained": round(float(1 - np.mean(drop)), 3),
            "fragility": round(fragility, 3)}


def robustness_gate(profile: Dict[str, Any], max_fragility: float = 0.35,
                    max_flip_severity: float = 0.5) -> Dict[str, Any]:
    """Abstain when the call is not stable under plausible, mild transformations."""
    mild_flips = [r["transform"] for r in profile["rows"]
                  if r["flips"] and r["plausible"] and r["severity"] <= max_flip_severity]
    if mild_flips:
        return {"ok": False, "reason": f"verdict flips under mild transformation(s): {', '.join(mild_flips)}"}
    if profile["fragility"] > max_fragility:
        return {"ok": False, "reason": f"fragility {profile['fragility']:.2f} above {max_fragility:.2f}"}
    return {"ok": True, "reason": None}


# ---------------------------------------------------------------- audio ----


def _a_resample(w: np.ndarray, sr: int, low: int) -> np.ndarray:
    n = len(w)
    down = np.interp(np.linspace(0, n - 1, int(n * low / sr)), np.arange(n), w)
    return np.interp(np.linspace(0, len(down) - 1, n), np.arange(len(down)), down)


def _a_noise(w: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    p = float(np.mean(w ** 2)) + 1e-12
    n = np.random.default_rng(seed).normal(0, np.sqrt(p / 10 ** (snr_db / 10)), len(w))
    return (w + n).astype(w.dtype)


def _a_lowpass(w: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    f = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(len(w), 1.0 / sr)
    f[freqs > cutoff] = 0
    return np.fft.irfft(f, n=len(w)).astype(w.dtype)


def _a_quant(w: np.ndarray, bits: int) -> np.ndarray:
    lv = 2 ** (bits - 1)
    return (np.round(np.clip(w, -1, 1) * lv) / lv).astype(w.dtype)


def _a_mulaw(w: np.ndarray, mu: int = 255) -> np.ndarray:
    """Telephony companding (mu-law): the codec distortion of voice calls."""
    x = np.clip(w, -1, 1)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1) * 127.5) / 127.5 - 1
    return (np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu).astype(w.dtype)


def _a_reverb(w: np.ndarray, sr: int, rt60: float = 0.4, seed: int = 0) -> np.ndarray:
    """Synthetic room: exponentially decaying noise impulse response."""
    n = int(sr * rt60)
    ir = np.random.default_rng(seed).normal(0, 1, n) * np.exp(-6.9 * np.arange(n) / n)
    ir[0] = 1.0
    ir /= np.abs(ir).sum() + 1e-9
    out = np.fft.irfft(np.fft.rfft(w, len(w) + n) * np.fft.rfft(ir, len(w) + n))[:len(w)]
    scale = (np.abs(w).max() + 1e-9) / (np.abs(out).max() + 1e-9)
    return (out * scale).astype(w.dtype)


def audio_transforms(sr: int = 16000) -> List[Transform]:
    return [
        Transform("resample_8k", 0.45, lambda w: _a_resample(w, sr, 8000), ("high",)),
        Transform("noise_snr20", 0.35, lambda w: _a_noise(w, 20.0)),
        Transform("noise_snr10", 0.65, lambda w: _a_noise(w, 10.0), ("high",)),
        Transform("lowpass_4k", 0.50, lambda w: _a_lowpass(w, sr, 4000), ("high",)),
        Transform("quant_8bit", 0.40, lambda w: _a_quant(w, 8)),
        Transform("codec_mulaw", 0.45, lambda w: _a_mulaw(w)),
        Transform("reverb_rt60_04", 0.55, lambda w: _a_reverb(w, sr, 0.4), ("high",)),
    ]
