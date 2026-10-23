"""Backward-compatible image API.

The detection logic now lives in :mod:`detector.pipeline`. This wrapper keeps
the original 8-tuple signature working for existing scripts. New code should
call ``get_pipeline().analyze(...)`` and use the full ``AnalysisResult``.

Note: the returned verdict may be ``"Inconclusive"`` (the original code could
only answer Real or Fake).
"""
from PIL import Image

from detector.facial_forensics import analyze_image_quality_and_noise
from detector.pipeline import get_pipeline


def detect_image(input_img, fake_threshold=0.5):
    """Returns (verdict, confidence_str, face_count, annotated_img,
    face_thumbs, ela_img, fft_img, raw_details).

    ``fake_threshold`` is kept for compatibility: it maps to the operating
    mode (<=0.35 sensitive, >=0.65 strict, otherwise balanced)."""
    mode = ("sensitive" if fake_threshold <= 0.35
            else "strict" if fake_threshold >= 0.65 else "balanced")
    res = get_pipeline().analyze(
        input_img, media_type="image", mode=mode, explain=True, stability=False
    )
    img = res.artifacts.get("annotated")
    conf = res.confidence if res.confidence is not None else max(res.p_fake, 1 - res.p_fake)
    q = analyze_image_quality_and_noise(img) if img is not None else {}
    n_faces = next((e["detail"].get("faces", 0) for e in res.evidence
                    if e["name"] == "neural"), 0)
    raw = {
        "fake_probability": res.p_fake,
        "real_probability": 1.0 - res.p_fake,
        "primary_confidence": conf,
        "face_count": n_faces,
        "threshold_used": fake_threshold,
        "quality_metrics": q,
        "calibrated": res.calibrated,
        "label_set": res.label_set,
    }
    return (
        res.verdict, f"{conf:.2f}", n_faces, img,
        res.artifacts.get("face_crops", []), res.artifacts.get("ela"),
        res.artifacts.get("fft"), raw,
    )
