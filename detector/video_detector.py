"""Backward-compatible video API (see :mod:`detector.pipeline`)."""
from detector.pipeline import get_pipeline


def detect_video(video_file, frame_skip=15):
    """Returns (verdict, total_analyzed, fake_count, summary, fake_frames,
    annotated_frames, timeline_data, audio_analysis).

    ``frame_skip`` is ignored: frame sampling is now adaptive (coarse Tier-1
    screen, densified around suspicious frames). The verdict is the
    pipeline's calibrated verdict and may be ``"Inconclusive"``."""
    res = get_pipeline().analyze(
        video_file, media_type="video", explain=False, stability=False
    )
    timeline = [
        {"frame": i, "timestamp": t["timestamp"],
         "verdict": "Fake" if t["fake_probability"] >= 0.5 else "Real",
         "confidence": max(t["fake_probability"], 1 - t["fake_probability"]),
         "fake_probability": t["fake_probability"]}
        for i, t in enumerate(res.timeline)
    ]
    fake_count = sum(1 for t in timeline if t["verdict"] == "Fake")
    total = len(timeline)
    summary = {
        "total_analyzed_frames": total,
        "real_frames": total - fake_count,
        "fake_frames": fake_count,
        "fake_percentage": f"{(fake_count / total * 100) if total else 0.0:.1f}%",
        "video_fps": f"{res.quality.get('fps') or 0:.1f}",
    }
    fake_frames = [im for _, im in res.artifacts.get("suspicious_frames", [])]
    return (res.verdict, total, fake_count, summary, fake_frames, [],
            timeline, None)
