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
    """Mono float32 waveform at ``sr`` Hz, or None if there is no audio."""
    for loader in (_load_ffmpeg, _load_soundfile, _load_torchaudio):
        x = loader(path, sr)
        if x is not None and len(x) > 0:
            return x
    return None


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
