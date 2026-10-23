"""Backward-compatible audio API (see :mod:`detector.pipeline`)."""
import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from detector.audio_io import load_audio
from detector.pipeline import get_pipeline

matplotlib.use("Agg")


def detect_audio(audio_file, chunk_seconds=3):
    """Returns (verdict, confidence_str, segment_preds, mel_img,
    time_series_data). The verdict may be ``"Inconclusive"``."""
    res = get_pipeline().analyze(
        audio_file, media_type="audio", audio_chunk_s=chunk_seconds
    )
    conf = res.confidence if res.confidence is not None else max(res.p_fake, 1 - res.p_fake)
    segs = [
        {"start": s["start"], "end": s["end"],
         "label": "Fake" if s["fake_probability"] >= 0.5 else "Real",
         "confidence": f"{max(s['fake_probability'], 1 - s['fake_probability']):.2f}",
         "fake_prob": s["fake_probability"]}
        for s in res.segments if "fake_probability" in s
    ]
    return res.verdict, f"{conf:.2f}", segs, res.artifacts.get("mel"), res.timeline


def get_waveform_image(audio_path):
    wave = load_audio(audio_path, 16000)
    fig, ax = plt.subplots(figsize=(10, 2))
    if wave is not None:
        ax.plot(wave, color="#38bdf8", alpha=0.85, linewidth=0.8)
    ax.set_title("Audio Amplitude Waveform", fontsize=10, color="white")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(),
                bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)
