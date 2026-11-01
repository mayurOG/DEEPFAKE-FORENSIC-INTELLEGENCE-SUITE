"""Robust audio loading (wav / mp3 / flac / audio track of a video).

Uses ffmpeg when available (works on any container, no torchaudio backend
quirks), then falls back to soundfile / torchaudio.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Optional, Tuple

import numpy as np


def _load_ffmpeg(path: str, sr: int) -> Optional[np.ndarray]:
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr),
             "-f", "f32le", "-"],
            capture_output=True, timeout=300,
        )
    except Exception:
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _load_soundfile(path: str, sr: int) -> Optional[np.ndarray]:
    try:
        import soundfile as sf

        x, file_sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if file_sr != sr:
            x = _resample(x, file_sr, sr)
        return x
    except Exception:
        return None


def _load_torchaudio(path: str, sr: int) -> Optional[np.ndarray]:
    try:
        import torch
        import torchaudio

        wav, file_sr = torchaudio.load(path)
        wav = wav.mean(dim=0)
        if file_sr != sr:
            wav = torchaudio.functional.resample(wav, file_sr, sr)
        return wav.numpy().astype(np.float32)
    except Exception:
        return None


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(src, dst)
    return resample_poly(x, dst // g, src // g).astype(np.float32)


def load_audio(path: str, sr: int = 16000) -> Optional[np.ndarray]:
    """Mono float32 waveform standardized at ``sr`` Hz, or None if there is no audio."""
    for loader in (_load_ffmpeg, _load_soundfile, _load_torchaudio):
        x = loader(path, sr)
        if x is not None and len(x) > 0:
            x = np.asarray(x, dtype=np.float32)
            if x.ndim > 1:
                x = x.mean(axis=-1)
            # Normalize digital headroom if clipped
            peak = float(np.abs(x).max()) if len(x) else 0.0
            if peak > 1.0:
                x = x / peak
            return x
    return None


def compute_vad_mask(
    wave: np.ndarray,
    sr: int = 16000,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    energy_threshold_db: float = -38.0,
    entropy_threshold: float = 0.88,
    min_speech_ms: float = 120.0,
    min_silence_ms: float = 200.0,
) -> dict:
    """Voice Activity Detection (VAD) combining short-time energy and spectral entropy.

    Separates active speech from silence and ambient background noise to prevent
    unvoiced room tone or silent audio chunks from producing false-positive
    deepfake voice predictions.

    Returns a dict containing:
      - ``vad_mask``: boolean array per audio sample of speech activity.
      - ``speech_ratio``: fraction of audio containing voiced speech (0.0 to 1.0).
      - ``speech_spans``: list of ``(start_s, end_s)`` speech segment boundaries.
      - ``frame_is_speech``: boolean array per STFT frame.
      - ``frame_times``: timestamp (seconds) for each analysis frame.
      - ``mean_snr_db``: estimated signal-to-noise ratio in decibels.
    """
    wave = np.asarray(wave, dtype=np.float32)
    if len(wave) == 0:
        return {
            "vad_mask": np.zeros(0, dtype=bool),
            "speech_ratio": 0.0,
            "speech_spans": [],
            "frame_is_speech": np.zeros(0, dtype=bool),
            "frame_times": np.zeros(0, dtype=np.float32),
            "mean_snr_db": 0.0,
        }

    frame_len = max(16, int(frame_ms * sr / 1000.0))
    hop_len = max(8, int(hop_ms * sr / 1000.0))
    n_frames = max(1, (len(wave) - frame_len) // hop_len + 1)

    window = np.hanning(frame_len).astype(np.float32)
    energies = np.zeros(n_frames, dtype=np.float32)
    entropies = np.zeros(n_frames, dtype=np.float32)
    frame_times = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_len
        frame = wave[start:start + frame_len]
        frame_times[i] = (start + frame_len / 2.0) / sr

        # Short-time frame energy
        rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
        energies[i] = rms

        # Windowed spectral entropy
        win_frame = frame * window
        spec = np.abs(np.fft.rfft(win_frame))
        power = spec ** 2
        p_norm = power / (power.sum() + 1e-12)
        # Shannon entropy normalized by log(N)
        ent = -np.sum(p_norm * np.log2(p_norm + 1e-12)) / np.log2(len(p_norm) + 1e-12)
        entropies[i] = float(ent)

    # Convert energy to relative dB with absolute silence floor
    max_energy = float(np.percentile(energies, 95)) if len(energies) else 0.0
    abs_silence_floor = 0.005  # -46 dBFS minimum audible threshold for speech candidates

    if max_energy < abs_silence_floor:
        # Audio track is pure silence or low-level ambient hiss
        is_speech = np.zeros(n_frames, dtype=bool)
    else:
        db = 20.0 * np.log10(np.maximum(energies, 1e-6) / max_energy)
        # Initial frame-level speech detection:
        # Must exceed absolute floor + relative threshold, with harmonic low-entropy structure
        is_speech = (energies >= abs_silence_floor) & (db >= energy_threshold_db) & ((entropies <= entropy_threshold) | (db >= -15.0))


    # Temporal hangover and smoothing (morphological closing / minimum duration)
    min_speech_frames = max(1, int(min_speech_ms / hop_ms))
    min_silence_frames = max(1, int(min_silence_ms / hop_ms))

    # 1. Fill short silence gaps within speech (hangover)
    smoothed = is_speech.copy()
    silence_count = 0
    in_speech = False
    for i in range(len(smoothed)):
        if smoothed[i]:
            if in_speech and 0 < silence_count <= min_silence_frames:
                smoothed[i - silence_count:i] = True
            in_speech = True
            silence_count = 0
        else:
            silence_count += 1
            if silence_count > min_silence_frames:
                in_speech = False

    # 2. Drop transient noise spikes shorter than min_speech_frames
    speech_count = 0
    for i in range(len(smoothed)):
        if smoothed[i]:
            speech_count += 1
        else:
            if 0 < speech_count < min_speech_frames:
                smoothed[i - speech_count:i] = False
            speech_count = 0
    if 0 < speech_count < min_speech_frames:
        smoothed[len(smoothed) - speech_count:] = False

    # Build sample-level mask
    vad_mask = np.zeros(len(wave), dtype=bool)
    for i, flag in enumerate(smoothed):
        if flag:
            s = i * hop_len
            e = min(len(wave), s + frame_len)
            vad_mask[s:e] = True

    speech_ratio = float(np.mean(vad_mask)) if len(vad_mask) else 0.0

    # Extract speech spans (start_s, end_s)
    speech_spans: List[Tuple[float, float]] = []
    in_span = False
    span_start = 0.0
    for i, flag in enumerate(smoothed):
        t = float(frame_times[i])
        if flag and not in_span:
            in_span = True
            span_start = max(0.0, t - (hop_ms / 2000.0))
        elif not flag and in_span:
            in_span = False
            speech_spans.append((span_start, t + (hop_ms / 2000.0)))
    if in_span:
        speech_spans.append((span_start, float(len(wave) / sr)))

    # Estimate SNR
    noise_energies = energies[~smoothed] if np.any(~smoothed) else energies
    speech_energies = energies[smoothed] if np.any(smoothed) else energies
    snr_db = float(
        20.0 * np.log10(
            (np.mean(speech_energies) + 1e-6) / (np.mean(noise_energies) + 1e-6)
        )
    )

    return {
        "vad_mask": vad_mask,
        "speech_ratio": speech_ratio,
        "speech_spans": speech_spans,
        "frame_is_speech": smoothed,
        "frame_times": frame_times,
        "mean_snr_db": snr_db,
    }


def is_silent_or_ambient(wave: np.ndarray, sr: int = 16000, min_speech_ratio: float = 0.05) -> Tuple[bool, float]:
    """Returns (True, speech_ratio) if the audio waveform lacks active speech."""
    vad = compute_vad_mask(wave, sr)
    return (vad["speech_ratio"] < min_speech_ratio, vad["speech_ratio"])


def has_audio_track(path: str) -> bool:
    exe = shutil.which("ffprobe")
    if not exe:
        return load_audio(path) is not None
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index", "-of", "csv=p=0", path],
            capture_output=True, timeout=30, text=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False

