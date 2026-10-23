"""Reusable media fingerprints for cross-asset matching (CALM-FORGE).

Produces compact descriptors that support near-duplicate and derivative
search without storing full media:

* Image: aHash / dHash / pHash-style perceptual hashes + colour histogram
* Video: temporal sample of frame perceptual hashes + duration
* Audio: chromaprint-like spectral peaks (lightweight, no external db)

Fingerprints are evidence for *relationship hypotheses*, not deepfake labels.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

log = logging.getLogger("deepfake.fingerprints")


@dataclass
class MediaFingerprint:
    fingerprint_id: str
    media_type: str
    # Perceptual
    ahash: Optional[str] = None
    dhash: Optional[str] = None
    phash: Optional[str] = None
    # Colour / spectral summaries
    hist_rgb: Optional[List[float]] = None
    # Video / audio structure
    frame_hashes: Optional[List[str]] = None
    duration_s: Optional[float] = None
    sample_rate: Optional[int] = None
    audio_peaks: Optional[List[float]] = None
    # Embedding placeholder (filled when a vision/audio encoder is available)
    embedding: Optional[List[float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.embedding is not None and len(self.embedding) > 64:
            d["embedding"] = self.embedding[:64] + ["..."]  # truncate for JSON
            d["embedding_dim"] = len(self.embedding)
        return d


def _to_gray(img: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    g = img.convert("L").resize(size, Image.Resampling.LANCZOS)
    return np.asarray(g, dtype=np.float64)


def ahash64(img: Image.Image) -> str:
    arr = _to_gray(img, (8, 8))
    bits = (arr > arr.mean()).astype(np.uint8).flatten()
    return "".join(str(int(b)) for b in bits)


def dhash64(img: Image.Image) -> str:
    arr = _to_gray(img, (9, 8))
    diff = arr[:, 1:] > arr[:, :-1]
    return "".join(str(int(b)) for b in diff.flatten().astype(np.uint8))


def phash64(img: Image.Image) -> str:
    """Block-mean perceptual hash (8x8).

    Named ``phash`` for API compatibility, but this is intentionally a
    dependency-light block-mean stand-in rather than a true DCT pHash.
    It is correlated with aHash; downstream fusion should not treat them as
    independent strong evidence of the same claim.
    """
    arr = _to_gray(img, (32, 32))
    blocks = arr.reshape(8, 4, 8, 4).mean(axis=(1, 3))
    bits = (blocks > blocks.mean()).astype(np.uint8).flatten()
    return "".join(str(int(b)) for b in bits)


def hamming(a: str, b: str) -> int:
    if not a or not b or len(a) != len(b):
        return max(len(a or ""), len(b or ""))
    return sum(x != y for x, y in zip(a, b))


def hist_rgb8(img: Image.Image) -> List[float]:
    im = img.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    arr = np.asarray(im, dtype=np.float64)
    feats = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=8, range=(0, 256), density=True)
        feats.extend(h.tolist())
    return feats


def fingerprint_image(img: Image.Image) -> MediaFingerprint:
    ah, dh, ph = ahash64(img), dhash64(img), phash64(img)
    fid = hashlib.sha256(f"{ah}:{dh}:{ph}".encode()).hexdigest()[:16]
    return MediaFingerprint(
        fingerprint_id=f"fp_{fid}",
        media_type="image",
        ahash=ah,
        dhash=dh,
        phash=ph,
        hist_rgb=hist_rgb8(img),
        meta={"size": list(img.size)},
    )


def fingerprint_image_path(path: str) -> MediaFingerprint:
    from detector.triage import enforce_path_size_limit, enforce_image_limits
    enforce_path_size_limit(path)
    img = Image.open(path).convert("RGB")
    img.load()
    enforce_image_limits(img)
    return fingerprint_image(img)


def fingerprint_video(path: str, n_frames: int = 8) -> MediaFingerprint:
    """Sample evenly spaced frames and hash each."""
    from detector.triage import enforce_path_size_limit, probe_media_duration, enforce_duration_limit
    from detector.config import get_settings
    enforce_path_size_limit(path)
    real_duration = probe_media_duration(path, "video")
    enforce_duration_limit(real_duration, "video")
    max_f = get_settings().max_video_frames
    if max_f > 0:
        n_frames = min(int(n_frames), int(max_f))
    frame_hashes: List[str] = []
    duration = real_duration
    try:
        from detector.pipeline import sample_frames
        samples = list(sample_frames(path, n_frames))
        for _, t, im in samples:
            if im is not None:
                frame_hashes.append(ahash64(im.convert("RGB")))
                if duration is None:
                    duration = t
    except Exception as e:
        log.warning("video fingerprint fallback: %s", e)
        # Try OpenCV
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
            duration = (n / fps) if n else None
            idxs = np.linspace(0, max(n - 1, 0), n_frames).astype(int) if n else []
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, frame = cap.read()
                if not ok:
                    continue
                rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_hashes.append(ahash64(rgb))
            cap.release()
        except Exception as e2:
            log.warning("cv2 video fingerprint failed: %s", e2)

    head = frame_hashes[0] if frame_hashes else "0" * 64
    fid = hashlib.sha256("|".join(frame_hashes).encode()).hexdigest()[:16]
    return MediaFingerprint(
        fingerprint_id=f"fp_{fid}",
        media_type="video",
        ahash=head,
        frame_hashes=frame_hashes,
        duration_s=float(duration) if duration is not None else None,
        meta={"n_frame_hashes": len(frame_hashes)},
    )


def fingerprint_audio(path: str, sr: int = 16000, max_seconds: float = 30.0) -> MediaFingerprint:
    """Lightweight spectral-peak fingerprint (not a full Chromaprint)."""
    from detector.audio_io import load_audio
    wave = load_audio(path, sr=sr)
    if wave is None or len(wave) < sr // 4:
        fid = hashlib.sha256(path.encode()).hexdigest()[:16]
        return MediaFingerprint(
            fingerprint_id=f"fp_{fid}", media_type="audio",
            meta={"empty": True},
        )
    max_n = int(max_seconds * sr)
    if len(wave) > max_n:
        start = (len(wave) - max_n) // 2
        wave = wave[start:start + max_n]
    # Frame energies + coarse spectrum peaks
    frame = max(int(0.05 * sr), 1)
    n = len(wave) // frame
    if n < 2:
        peaks = [float(np.mean(wave ** 2))]
    else:
        frames = wave[: n * frame].reshape(n, frame)
        energy = (frames ** 2).mean(axis=1)
        # FFT of energy envelope
        spec = np.abs(np.fft.rfft(energy - energy.mean()))
        k = min(16, len(spec))
        peaks = spec[:k].astype(np.float64).tolist()
        s = sum(peaks) or 1.0
        peaks = [p / s for p in peaks]
    fid = hashlib.sha256(np.asarray(peaks).tobytes()).hexdigest()[:16]
    return MediaFingerprint(
        fingerprint_id=f"fp_{fid}",
        media_type="audio",
        audio_peaks=peaks,
        duration_s=len(wave) / float(sr),
        sample_rate=sr,
    )


def fingerprint_asset(path: str, media_type: Optional[str] = None) -> MediaFingerprint:
    from detector.pipeline import infer_media_type
    mt = media_type or infer_media_type(path) or "image"
    if mt == "image":
        return fingerprint_image_path(path)
    if mt == "video":
        return fingerprint_video(path)
    if mt == "audio":
        return fingerprint_audio(path)
    return fingerprint_image_path(path)
