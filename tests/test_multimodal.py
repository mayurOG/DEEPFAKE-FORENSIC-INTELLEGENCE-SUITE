import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from detector.audio_detector import detect_audio
from detector.audio_io import compute_vad_mask, is_silent_or_ambient, load_audio
from detector.experts.temporal import temporally_aware_aggregate
from detector.pipeline import detect_scene_boundaries, sample_frames
from detector.video_detector import detect_video
from tests.test_pipeline import _write_video


def test_vad_silence_vs_speech():
    sr = 16000
    # 1. Pure silence
    silence = np.zeros(sr * 3, dtype=np.float32)
    vad_silence = compute_vad_mask(silence, sr)
    assert vad_silence["speech_ratio"] == 0.0
    is_sil, ratio = is_silent_or_ambient(silence, sr)
    assert is_sil is True
    assert ratio == 0.0

    # 2. Voiced harmonic speech-like signal
    t = np.arange(sr * 3) / sr
    speech = (
        0.35 * np.sin(2 * np.pi * 200 * t)
        + 0.25 * np.sin(2 * np.pi * 400 * t)
        + 0.15 * np.sin(2 * np.pi * 800 * t)
    ).astype(np.float32)
    vad_speech = compute_vad_mask(speech, sr)
    assert vad_speech["speech_ratio"] > 0.5
    is_sil, ratio = is_silent_or_ambient(speech, sr)
    assert is_sil is False
    assert ratio > 0.5
    assert len(vad_speech["speech_spans"]) >= 1


def test_audio_pipeline_silence_gating(pipe, tmp_path):
    sr = 16000
    # Create 3 seconds of silence / low ambient noise
    ambient = np.random.normal(0, 1e-4, sr * 3).astype(np.float32)
    audio_path = str(tmp_path / "ambient.wav")
    sf.write(audio_path, ambient, sr)

    res = pipe.analyze(audio_path, media_type="audio", filename="ambient.wav")
    # Silence should trigger a warning
    assert any("speech" in w.lower() for w in res.warnings)
    # Audio evidence detail should indicate no speech detected
    ev = next(e for e in res.evidence if e["name"] == "audio")
    assert ev["detail"].get("speech_detected") is False or ev["detail"].get("speech_ratio", 1.0) < 0.05
    # Should not produce a false-positive Fake verdict
    assert res.verdict in ("Real", "Inconclusive")


def test_temporally_aware_aggregate():
    # 1. Highly correlated frames (repeated values)
    z_corr = [2.0, 2.05, 2.02, 2.08, 2.01, 2.06]
    agg_corr, meta_corr = temporally_aware_aggregate(z_corr)
    # Autocorrelation should be high or variance low
    assert meta_corr["n_eff"] <= len(z_corr)

    # 2. Localized anomaly burst (face swap in 2 frames among 8 real frames)
    z_burst = [-2.0, -2.1, -1.9, 3.5, 3.2, -2.0, -1.8, -2.2]
    agg_burst, meta_burst = temporally_aware_aggregate(z_burst)
    assert meta_burst["anomaly_detected"] is True
    # The anomaly burst should pull the score significantly above plain mean
    assert agg_burst > np.mean(z_burst)

    # 3. Empty input
    agg_empty, meta_empty = temporally_aware_aggregate([])
    assert agg_empty == 0.0
    assert meta_empty["n_frames"] == 0


def test_video_sampling_and_scene_detection(tmp_path):
    import cv2

    video_path = str(tmp_path / "scene_test.mp4")
    vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 64))

    # Scene 1: 30 black frames
    for _ in range(30):
        vw.write(np.zeros((64, 64, 3), dtype=np.uint8))
    # Scene 2: 30 bright white frames (abrupt cut)
    for _ in range(30):
        vw.write(np.ones((64, 64, 3), dtype=np.uint8) * 255)
    vw.release()

    cap = cv2.VideoCapture(video_path)
    boundaries = detect_scene_boundaries(cap, 60, 30.0)
    cap.release()
    # Should detect boundary around frame 30
    assert len(boundaries) >= 3

    # Sample frames with duplicate skipping and scene awareness
    sampled = sample_frames(video_path, n=6, detect_scenes=True, skip_duplicates=True)
    assert len(sampled) >= 2


def test_video_detector_wrapper(tmp_path):
    p = _write_video(tmp_path / "v_wrap.mp4", frames=45, with_audio=False)
    verdict, total, fake_count, summary, fake_frames, annotated, timeline, audio_analysis = detect_video(p)
    assert verdict in ("Real", "Fake", "Inconclusive")
    assert total > 0
    assert "effective_sample_size" in summary or "total_analyzed_frames" in summary
