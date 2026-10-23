"""Face-signal experts: physiological pulse (rPPG), audio-visual sync and
facial-geometry stability.

All three need *consecutive* frames, so they share one pass that tracks the
face over a short window and records forehead/cheek colour, mouth motion and
landmarks. They are gated by quality checks and report a reliability:
heavily compressed, short, low-fps or small-face clips are *skipped* rather
than allowed to add noise.

IMPORTANT: these are signal-processing heuristics (CHROM rPPG, a mouth-motion
vs audio-envelope correlation proxy for lip-sync, landmark-ratio stability).
They are uncalibrated and carry a small weight in fusion until the weights
are learned with ``train/fit_fusion.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from detector.fusion import Evidence, skipped


@dataclass
class FaceSignals:
    fps: float
    n_frames: int
    face_ratio: float
    face_width_px: float
    forehead_rgb: np.ndarray  # (N, 3), NaN where the face was lost
    cheek_rgb: np.ndarray  # (N, 3)
    mouth_energy: np.ndarray  # (N,) inter-frame mouth-region motion, NaN lost
    landmarks: List[Tuple[int, np.ndarray]] = field(default_factory=list)


def extract_face_signals(
    video_path: str, detector, max_seconds: float = 8.0, detect_every: int = 6,
    max_frames: int = 300, max_width: int = 640, start_frame: int = 0,
) -> Optional[FaceSignals]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    n_target = int(min(max_frames, max(1, max_seconds * fps)))
    fore, cheek, mouth_e, lms = [], [], [], []
    box, misses, prev_mouth, widths = None, 0, None, []
    for idx in range(n_target):
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > max_width:
            frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
            h, w = frame.shape[:2]
        if idx % detect_every == 0 or box is None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect(Image.fromarray(rgb))
            if faces:
                f = max(faces, key=lambda ff: ff.area)
                box, misses = f.box, 0
                if f.landmarks is not None:
                    lms.append((idx, np.asarray(f.landmarks, dtype=np.float64)))
            else:
                misses += 1
                if misses > 2:
                    box = None
        if box is None:
            fore.append([np.nan] * 3)
            cheek.append([np.nan] * 3)
            mouth_e.append(np.nan)
            prev_mouth = None
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 24 or bh < 24:
            fore.append([np.nan] * 3)
            cheek.append([np.nan] * 3)
            mouth_e.append(np.nan)
            prev_mouth = None
            continue
        widths.append(bw)

        def region(fx0, fx1, fy0, fy1):
            r = frame[y1 + int(fy0 * bh):y1 + int(fy1 * bh),
                      x1 + int(fx0 * bw):x1 + int(fx1 * bw)]
            return r.reshape(-1, 3).mean(axis=0)[::-1] if r.size else np.full(3, np.nan)

        fore.append(region(0.30, 0.70, 0.05, 0.22))
        lc, rc = region(0.12, 0.36, 0.50, 0.72), region(0.64, 0.88, 0.50, 0.72)
        cheek.append((lc + rc) / 2.0)
        m = frame[y1 + int(0.65 * bh):y1 + int(0.92 * bh),
                  x1 + int(0.25 * bw):x1 + int(0.75 * bw)]
        if m.size:
            g = cv2.resize(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY), (32, 16)).astype(np.float32)
            mouth_e.append(float(np.abs(g - prev_mouth).mean()) if prev_mouth is not None else np.nan)
            prev_mouth = g
        else:
            mouth_e.append(np.nan)
            prev_mouth = None
    cap.release()
    n = len(fore)
    if n == 0:
        return None
    fore_a = np.asarray(fore, dtype=np.float64)
    ratio = float(np.mean(~np.isnan(fore_a[:, 0])))
    if ratio == 0.0:
        return None
    return FaceSignals(
        fps=float(fps), n_frames=n, face_ratio=ratio,
        face_width_px=float(np.mean(widths)) if widths else 0.0,
        forehead_rgb=fore_a, cheek_rgb=np.asarray(cheek, dtype=np.float64),
        mouth_energy=np.asarray(mouth_e, dtype=np.float64), landmarks=lms,
    )


def _interp_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    idx = np.arange(len(x))
    ok = ~np.isnan(x)
    if ok.sum() < 2:
        return np.zeros_like(x)
    x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return x


# --------------------------------------------------------------------------
# rPPG (CHROM, de Haan & Jeanne 2013)
# --------------------------------------------------------------------------
def chrom_pulse(rgb: np.ndarray, fps: float) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    rgb = np.stack([_interp_nan(rgb[:, c]) for c in range(3)], axis=1)
    rgb = rgb / (rgb.mean(axis=0, keepdims=True) + 1e-9)
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    xs, ys = 3 * R - 2 * G, 1.5 * R + G - 1.5 * B
    nyq = fps / 2.0
    hi = min(3.0, nyq * 0.95)
    b, a = butter(3, [0.7 / nyq, hi / nyq], btype="band")
    xf, yf = filtfilt(b, a, xs), filtfilt(b, a, ys)
    alpha = np.std(xf) / (np.std(yf) + 1e-9)
    return xf - alpha * yf


def pulse_snr(signal: np.ndarray, fps: float) -> Tuple[float, float]:
    """(heart rate in bpm, SNR in dB) of the dominant in-band frequency."""
    n = len(signal)
    nfft = int(2 ** np.ceil(np.log2(max(n * 4, 1024))))
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(signal * win, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fps)
    band = (freqs >= 0.7) & (freqs <= 3.0)
    if not band.any():
        return 0.0, -99.0
    f0 = freqs[band][np.argmax(spec[band])]
    sig = ((np.abs(freqs - f0) < 0.12) | (np.abs(freqs - 2 * f0) < 0.12)) & (freqs >= 0.7) & (freqs <= 4.0)
    p_sig = spec[sig].sum()
    p_noise = spec[band & ~sig].sum() + 1e-12
    return float(60.0 * f0), float(10 * np.log10(p_sig / p_noise + 1e-12))


def rppg_evidence(sig: Optional[FaceSignals], compression: str = "low") -> Evidence:
    if sig is None:
        return skipped("rppg", "no usable face track", True)
    if sig.fps < 15:
        return skipped("rppg", f"frame rate too low ({sig.fps:.0f} fps)", True)
    if sig.n_frames < 4 * sig.fps:
        return skipped("rppg", "less than 4 s of face video", True)
    if sig.face_ratio < 0.8:
        return skipped("rppg", "face lost in more than 20% of frames", True)
    if sig.face_width_px < 80:
        return skipped("rppg", "face too small for colour analysis", True)
    if compression == "high":
        return skipped("rppg", "video too compressed to carry a pulse", True)
    rgb = (sig.forehead_rgb + sig.cheek_rgb) / 2.0
    pulse = chrom_pulse(rgb, sig.fps)
    hr, snr = pulse_snr(pulse, sig.fps)
    # A clean periodic pulse (high SNR) is a sign of a live face.
    logit = float(np.clip((0.0 - snr) * 0.35, -3.0, 3.0))
    rel = 0.5 if compression == "low" else 0.3
    return Evidence(
        name="rppg", logit=logit, reliability=rel, heuristic=True,
        detail={"hr_bpm": round(hr, 1), "snr_db": round(snr, 2),
                "seconds": round(sig.n_frames / sig.fps, 1)},
    )


# --------------------------------------------------------------------------
# Audio-visual sync proxy
# --------------------------------------------------------------------------
def audio_envelope(audio: np.ndarray, sr: int, fps: float, n: int) -> np.ndarray:
    hop = sr / fps
    env = np.zeros(n)
    for i in range(n):
        seg = audio[int(i * hop):int((i + 1) * hop)]
        env[i] = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
    return env


def _smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    return np.convolve(x, np.ones(k) / k, mode="same")


def sync_zscore(
    mouth: np.ndarray, env: np.ndarray, fps: float, max_lag_s: float = 0.3,
    n_null: int = 200, seed: int = 0,
) -> Tuple[float, float, float]:
    """(z, best_corr, best_lag_seconds). z compares the best lagged
    correlation with a null distribution from circularly shifted audio."""
    m = _smooth(_interp_nan(mouth))
    a = _smooth(env)
    m = (m - m.mean()) / (m.std() + 1e-9)
    a = (a - a.mean()) / (a.std() + 1e-9)
    n = len(m)
    L = int(round(max_lag_s * fps))

    def best_corr(av):
        best, best_l = -1.0, 0
        for l in range(-L, L + 1):
            if l >= 0:
                x, y = m[: n - l], av[l:]
            else:
                x, y = m[-l:], av[: n + l]
            if len(x) < 8:
                continue
            r = float(np.mean(x * y))
            if r > best:
                best, best_l = r, l
        return best, best_l

    b, bl = best_corr(a)
    rng = np.random.default_rng(seed)
    nulls = []
    lo, hi = int(fps), max(int(fps) + 1, n - int(fps))
    for _ in range(n_null):
        s = int(rng.integers(lo, hi)) if hi > lo else 0
        nulls.append(best_corr(np.roll(a, s))[0])
    nulls = np.asarray(nulls)
    z = (b - nulls.mean()) / (nulls.std() + 1e-6)
    return float(z), float(b), float(bl / fps)


def sync_evidence(
    sig: Optional[FaceSignals], audio: Optional[np.ndarray], sr: int = 16000,
    compression: str = "low",
) -> Evidence:
    if sig is None:
        return skipped("sync", "no usable face track", True)
    if audio is None or len(audio) == 0:
        return skipped("sync", "no audio track", True)
    if sig.n_frames < 3 * sig.fps:
        return skipped("sync", "less than 3 s of face video", True)
    if sig.face_ratio < 0.7:
        return skipped("sync", "face lost in too many frames", True)
    mouth = sig.mouth_energy
    if np.nanstd(mouth) < 1e-3:
        return skipped("sync", "no visible mouth motion", True)
    env = audio_envelope(audio, sr, sig.fps, sig.n_frames)
    if env.std() < 1e-5 or float(np.percentile(env, 90) / (np.percentile(env, 10) + 1e-9)) < 3.0:
        return skipped("sync", "no clear speech in the audio", True)
    z, r, lag = sync_zscore(mouth, env, sig.fps)
    # Strong coupling (large z) is what a genuine speaker produces.
    logit = float(np.clip((1.5 - z) * 0.8, -3.0, 3.0))
    rel = 0.5 if compression != "high" else 0.3
    return Evidence(
        name="sync", logit=logit, reliability=rel, heuristic=True,
        detail={"sync_z": round(z, 2), "best_corr": round(r, 3),
                "best_lag_s": round(lag, 3)},
    )


# --------------------------------------------------------------------------
# Landmark geometry stability
# --------------------------------------------------------------------------
def _nose_ratio(lm: np.ndarray) -> Optional[float]:
    le, re, nose = lm[0], lm[1], lm[2]
    iod = np.linalg.norm(re - le)
    if iod < 1e-6:
        return None
    eye_c = (le + re) / 2.0
    ang = np.arctan2(re[1] - le[1], re[0] - le[0])
    c, s = np.cos(-ang), np.sin(-ang)
    v = nose - eye_c
    vy = s * v[0] + c * v[1]
    return float(vy / iod)


def geometry_evidence(sig: Optional[FaceSignals]) -> Evidence:
    if sig is None or len(sig.landmarks) < 6:
        return skipped("geometry", "fewer than 6 tracked landmark sets", True)
    ratios = [r for _, lm in sig.landmarks if (r := _nose_ratio(lm)) is not None]
    if len(ratios) < 6:
        return skipped("geometry", "unstable landmarks", True)
    ratios = np.asarray(ratios)
    cv = float(np.std(ratios) / (abs(np.mean(ratios)) + 1e-9))
    logit = float(np.clip((cv - 0.06) * 25.0, -2.0, 2.0))
    return Evidence(
        name="geometry", logit=logit, reliability=0.4, heuristic=True,
        detail={"nose_eye_ratio_cv": round(cv, 4), "n_landmark_sets": len(ratios)},
    )
