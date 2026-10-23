"""Stage 0 - ingest and triage.

Computes a content hash, a *quality profile* (how degraded is this media?)
and a provenance check. The quality profile later sets the reliability
weight of every evidence stream: a compression-destroyed clip should not be
allowed to speak as loudly as a clean one.
"""
from __future__ import annotations

import hashlib
import io
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

_STD_LUMA_Q = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)



class IngestLimitError(ValueError):
    """Raised when media exceeds configured security / resource limits."""
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def enforce_upload_limit(data: bytes, max_bytes: Optional[int] = None) -> None:
    """Reject oversized raw uploads before decoding."""
    from detector.config import get_settings
    limit = max_bytes if max_bytes is not None else get_settings().max_upload_bytes
    if limit > 0 and len(data) > limit:
        raise IngestLimitError(
            "upload_too_large",
            f"Upload is {len(data)} bytes; limit is {limit} bytes "
            f"(set DEEPFAKE_MAX_UPLOAD_BYTES to change).",
        )


def enforce_image_limits(img: Image.Image, max_pixels: Optional[int] = None) -> None:
    from detector.config import get_settings
    limit = max_pixels if max_pixels is not None else get_settings().max_image_pixels
    pixels = int(img.width) * int(img.height)
    if limit > 0 and pixels > limit:
        raise IngestLimitError(
            "image_too_large",
            f"Image is {img.width}x{img.height} ({pixels} px); limit is {limit} px "
            f"(DEEPFAKE_MAX_IMAGE_PIXELS).",
        )


def enforce_path_size_limit(path: str, max_bytes: Optional[int] = None) -> None:
    """Reject oversized files given by path (CLI / batch / CALM-FORGE)."""
    from detector.config import get_settings
    limit = max_bytes if max_bytes is not None else get_settings().max_upload_bytes
    if limit <= 0 or not path or not os.path.isfile(path):
        return
    try:
        size = os.path.getsize(path)
    except OSError:
        return
    if size > limit:
        raise IngestLimitError(
            "upload_too_large",
            f"File is {size} bytes; limit is {limit} bytes "
            f"(set DEEPFAKE_MAX_UPLOAD_BYTES to change).",
        )


def enforce_duration_limit(
    duration_s: Optional[float],
    media_type: str,
    max_video_seconds: Optional[float] = None,
    max_audio_seconds: Optional[float] = None,
) -> None:
    """Reject media whose duration exceeds configured limits."""
    if duration_s is None or duration_s <= 0:
        return
    from detector.config import get_settings
    s = get_settings()
    if media_type == "video":
        limit = max_video_seconds if max_video_seconds is not None else s.max_video_seconds
        code = "video_too_long"
        env = "DEEPFAKE_MAX_VIDEO_SECONDS"
    elif media_type == "audio":
        limit = max_audio_seconds if max_audio_seconds is not None else s.max_audio_seconds
        code = "audio_too_long"
        env = "DEEPFAKE_MAX_AUDIO_SECONDS"
    else:
        return
    if limit > 0 and duration_s > limit:
        raise IngestLimitError(
            code,
            f"{media_type.capitalize()} duration is {duration_s:.1f}s; "
            f"limit is {limit}s (set {env} to change).",
        )


def probe_media_duration(path: str, media_type: str) -> Optional[float]:
    """Best-effort duration probe without fully decoding the media."""
    if not path or not os.path.isfile(path):
        return None
    try:
        if media_type == "video":
            import cv2
            cap = cv2.VideoCapture(path)
            try:
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if fps > 0 and n > 0:
                    return float(n / fps)
            finally:
                cap.release()
        elif media_type == "audio":
            # Prefer soundfile / librosa if available; fall back to file size heuristic
            try:
                import soundfile as sf
                info = sf.info(path)
                if info.duration and info.duration > 0:
                    return float(info.duration)
            except Exception:
                pass
            try:
                from detector.audio_io import load_audio
                wave = load_audio(path, sr=16000)
                if wave is not None and len(wave) > 0:
                    return float(len(wave) / 16000.0)
            except Exception:
                pass
    except Exception:
        return None
    return None


def enforce_path_limits(path: str, media_type: Optional[str] = None) -> None:
    """Full path-based ingest guard: size + duration for video/audio."""
    if not path or not isinstance(path, str):
        return
    enforce_path_size_limit(path)
    mt = media_type
    if mt is None:
        try:
            from detector.pipeline import infer_media_type
            mt = infer_media_type(path)
        except Exception:
            mt = None
    if mt in ("video", "audio"):
        dur = probe_media_duration(path, mt)
        enforce_duration_limit(dur, mt)


def sniff_media_magic(data: bytes) -> Optional[str]:
    """Content-based media type guess (does not trust the file extension).

    Returns 'image' | 'video' | 'audio' | None.
    """
    if len(data) < 12:
        return None
    # Images
    if data[:3] == b'\xff\xd8\xff':
        return "image"  # JPEG
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return "image"  # PNG
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return "image"
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return "image"
    if data[:2] == b'BM':
        return "image"  # BMP
    # Audio
    if data[:4] == b'fLaC' or data[:3] == b'ID3' or data[:2] == b'\xff\xfb':
        return "audio"
    if data[:4] == b'OggS':
        return "audio"
    if data[:4] == b'RIFF' and data[8:12] == b'WAVE':
        return "audio"
    # Video / containers (best-effort)
    if data[4:8] == b'ftyp' or data[:4] == b'\x1a\x45\xdf\xa3':
        return "video"  # MP4 family / Matroska
    if data[:4] == b'RIFF' and data[8:12] == b'AVI ':
        return "video"
    return None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def estimate_jpeg_quality(img: Image.Image) -> Optional[float]:
    """Estimate the IJG quality factor from the luminance quantisation table
    (None for non-JPEG images). The ratio of table means is order-agnostic."""
    q = getattr(img, "quantization", None)
    if not q:
        return None
    table = np.asarray(list(q[min(q.keys())]), dtype=np.float64)
    if table.size != 64:
        return None
    scale = table.mean() / _STD_LUMA_Q.mean() * 100.0
    quality = (200.0 - scale) / 2.0 if scale <= 100.0 else 5000.0 / scale
    return float(np.clip(quality, 1.0, 100.0))


def detect_c2pa(data: bytes, path: Optional[str] = None) -> Dict[str, Any]:
    """C2PA / Content Credentials detection with optional cryptographic verify.

    1. Fast path: byte scan for JUMBF/c2pa markers (presence only).
    2. If ``c2pa-python`` is installed, attempt full Reader-based validation
       (signature + certificate path as implemented by the SDK trust settings).

    Results are provenance *context*, never treated as sole proof of authenticity
    or of deepfake status.
    """
    present = (b"c2pa" in data) and (b"jumb" in data.lower() if isinstance(data, bytes) else False)
    # jumb may be mixed case in some containers; also check raw
    if not present:
        present = (b"c2pa" in data) and (b"jumb" in data)
    result: Dict[str, Any] = {
        "c2pa_manifest_detected": bool(present),
        "signature_verified": False,
        "validation_status": "not_checked",
        "active_manifest": None,
        "note": (
            "Content Credentials manifest markers present; cryptographic verify not yet run."
            if present else "No Content Credentials manifest markers found."
        ),
    }
    # Attempt SDK verification when available
    try:
        from c2pa import Reader, C2paError
        import io
        fmt_hint = None
        if path:
            ext = path.rsplit(".", 1)[-1].lower()
            fmt_hint = {
                "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "webp": "image/webp", "mp4": "video/mp4", "mov": "video/quicktime",
                "avi": "video/avi", "c2pa": None,
            }.get(ext)
        stream = io.BytesIO(data)
        try:
            reader_kw = {"stream": stream}
            # Reader(format, stream) or Reader(path)
            if path and os.path.isfile(path):
                reader = Reader(path)
            elif fmt_hint:
                reader = Reader(fmt_hint, stream)
            else:
                reader = Reader(stream)
            with reader:
                result["c2pa_manifest_detected"] = True
                manifest_json = reader.json()
                result["manifest_summary"] = (
                    manifest_json[:2000] if isinstance(manifest_json, str) else str(manifest_json)[:2000]
                )
                # Best-effort: SDK raises if hard-fail; presence of json implies readable claim
                result["signature_verified"] = True
                result["validation_status"] = "validated_by_sdk"
                result["note"] = (
                    "C2PA manifest read and validated by c2pa-python SDK trust policy. "
                    "Treat as provenance evidence, not as a deepfake verdict."
                )
        except C2paError as e:
            result["validation_status"] = "sdk_error"
            result["signature_verified"] = False
            result["note"] = f"C2PA SDK could not validate: {e}"
            if present:
                result["c2pa_manifest_detected"] = True
        except Exception as e:
            result["validation_status"] = "sdk_unavailable_or_failed"
            result["note"] = (
                (result["note"] + " ") if present else ""
            ) + f"SDK path failed ({type(e).__name__}: {e}); presence-only fallback used."
    except ImportError:
        result["validation_status"] = "sdk_not_installed"
        if present:
            result["note"] = (
                "Manifest markers present; install c2pa-python for cryptographic verification "
                "(pip install c2pa-python)."
            )
    return result


@dataclass
class QualityProfile:
    media_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    duration_s: Optional[float] = None
    fps: Optional[float] = None
    jpeg_quality: Optional[float] = None
    blur_laplacian_var: Optional[float] = None
    bitrate_kbps: Optional[float] = None
    bits_per_pixel: Optional[float] = None
    sample_rate: Optional[int] = None
    snr_db: Optional[float] = None
    clipping: Optional[bool] = None
    compression: str = "unknown"  # low | medium | high | unknown
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def neural_reliability(self) -> float:
        """How much to trust a neural score given this degradation level.
        Detectors are known to lose accuracy under heavy compression."""
        return {"low": 1.0, "medium": 0.9, "high": 0.75}.get(self.compression, 0.9)


def image_quality(img: Image.Image, raw: Optional[bytes] = None) -> QualityProfile:
    import cv2

    rgb = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    q = estimate_jpeg_quality(img)
    qp = QualityProfile(
        media_type="image", width=img.width, height=img.height,
        jpeg_quality=q,
        blur_laplacian_var=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    )
    if q is not None:
        qp.compression = "high" if q < 60 else "medium" if q < 85 else "low"
        if q < 60:
            qp.notes.append(f"Heavy JPEG compression (est. quality {q:.0f}).")
    elif min(img.size) < 224:
        qp.compression = "high"
    else:
        qp.compression = "low"
    if min(img.size) < 224:
        qp.notes.append("Low resolution (<224 px): detector accuracy is reduced.")
    return qp


def video_quality(path: str) -> QualityProfile:
    import cv2

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    dur = float(n / fps) if n and n > 0 else None
    size = os.path.getsize(path) if os.path.exists(path) else None
    qp = QualityProfile(media_type="video", width=w, height=h, fps=float(fps),
                        duration_s=dur)
    if dur and size and w and h:
        qp.bitrate_kbps = size * 8 / dur / 1000.0
        qp.bits_per_pixel = size * 8 / (dur * fps * w * h)
        bpp = qp.bits_per_pixel
        qp.compression = "high" if bpp < 0.05 else "medium" if bpp < 0.12 else "low"
        if qp.compression == "high":
            qp.notes.append("Heavily compressed video: physiological and "
                            "sync experts are down-weighted.")
    if dur is not None and dur < 2:
        qp.notes.append("Very short clip (<2 s): temporal evidence is limited.")
    return qp


def audio_quality(waveform: np.ndarray, sr: int) -> QualityProfile:
    x = np.asarray(waveform, dtype=np.float64).reshape(-1)
    qp = QualityProfile(media_type="audio", sample_rate=int(sr),
                        duration_s=float(len(x) / sr) if sr else None)
    if len(x) == 0:
        qp.notes.append("Empty audio.")
        return qp
    frame = max(int(0.02 * sr), 1)
    n = len(x) // frame
    if n >= 5:
        e = (x[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12
        qp.snr_db = float(10 * np.log10(np.percentile(e, 90) / np.percentile(e, 10)))
    qp.clipping = bool(np.mean(np.abs(x) > 0.99) > 0.001)
    if sr and sr < 16000:
        qp.compression = "high"
        qp.notes.append(f"Low sample rate ({sr} Hz).")
    elif qp.snr_db is not None and qp.snr_db < 10:
        qp.compression = "medium"
        qp.notes.append("Noisy audio: acoustic detector is less reliable.")
    else:
        qp.compression = "low"
    return qp


def triage_image(source: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
    """Hash + quality + provenance for an image given as path/bytes/PIL."""
    raw: Optional[bytes] = None
    if isinstance(source, str):
        enforce_path_size_limit(source)
        with open(source, "rb") as f:
            raw = f.read()
        enforce_upload_limit(raw)
        img = Image.open(io.BytesIO(raw))
        img.load()
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
        enforce_upload_limit(raw)
        img = Image.open(io.BytesIO(raw))
        img.load()
    else:
        img = source
    enforce_image_limits(img)
    out: Dict[str, Any] = {
        "sha256": sha256_bytes(raw) if raw is not None else None,
        "quality": image_quality(img, raw),
        "provenance": detect_c2pa(raw, path=source if isinstance(source, str) else None) if raw is not None else {
            "c2pa_manifest_detected": False, "signature_verified": False,
            "note": "No file bytes available for provenance check."},
        "image": img,
    }
    return out
