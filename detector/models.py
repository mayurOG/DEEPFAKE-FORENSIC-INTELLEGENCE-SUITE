"""Model backends and registry.

Fixes over the original code:
* every model is loaded **once, lazily** (not at import time, not twice),
* the device is chosen automatically (CUDA / MPS / CPU),
* the "which output is Fake" index is read from ``model.config.id2label``
  instead of being hard-coded,
* inference is **batched**, and every backend also returns an embedding that
  is used for out-of-domain detection.

Backends share one tiny interface so the pipeline can ensemble them and the
tests can inject stubs without downloading anything.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from detector.config import get_settings

log = logging.getLogger("deepfake.models")

_FAKE_WORDS = {
    "fake", "deepfake", "spoof", "spoofed", "synthetic", "generated", "ai",
    "artificial", "manipulated", "forged", "tts", "cloned",
}
_REAL_WORDS = {
    "real", "bonafide", "genuine", "authentic", "human", "natural", "pristine",
}


def find_fake_index(id2label: dict) -> Tuple[int, str]:
    """Return (index of the 'fake' class, how it was found).

    Reads the model config; only falls back to index 0 (the assumption the
    original code made) when the labels are uninformative.
    """
    labels = {int(k): str(v) for k, v in (id2label or {}).items()}
    fake_hits, real_hits = [], []
    for idx, name in labels.items():
        tokens = set(re.split(r"[^a-z0-9]+", name.lower())) - {""}
        if tokens & _FAKE_WORDS and not tokens & _REAL_WORDS:
            fake_hits.append(idx)
        if tokens & _REAL_WORDS and not tokens & _FAKE_WORDS:
            real_hits.append(idx)
    if fake_hits:
        return fake_hits[0], "config"
    if real_hits and len(labels) == 2:
        other = [i for i in labels if i != real_hits[0]]
        return other[0], "config"
    override = os.getenv("FAKE_LABEL_INDEX")
    if override is not None and override.strip().lstrip("-").isdigit():
        return int(override), "env"
    log.warning(
        "Could not infer the Fake class from labels %s; assuming index 0. "
        "Set FAKE_LABEL_INDEX to override.", labels,
    )
    return 0, "fallback"


def fake_logit_from_logits(logits: np.ndarray, fake_idx: int) -> np.ndarray:
    """Binary fake log-odds from class logits (any number of classes)."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        return logits
    if logits.shape[1] == 1:
        return logits[:, 0]
    fake = logits[:, fake_idx]
    rest = np.delete(logits, fake_idx, axis=1)
    m = rest.max(axis=1)
    lse_rest = m + np.log(np.exp(rest - m[:, None]).sum(axis=1))
    return fake - lse_rest


def get_device() -> str:
    pref = get_settings().device.lower()
    if pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - torch missing
        pass
    return "cpu"


def _token() -> Optional[str]:
    return get_settings().hf_token or None


def _pool_hidden(hidden) -> np.ndarray:
    """Mean-pool a hidden state tensor (n, tokens, d) -> (n, d)."""
    h = hidden.detach().float().cpu().numpy()
    if h.ndim == 3:
        return h.mean(axis=1)
    return h.reshape(h.shape[0], -1)


# --------------------------------------------------------------------------
# Image backends
# --------------------------------------------------------------------------
class ImageBackend:
    """Interface: ``predict(images) -> (fake_logits (n,), embeddings (n,d)|None)``."""

    name = "backend"
    kind = "classifier"

    def predict(self, images: Sequence[Image.Image]):
        raise NotImplementedError


class HFClassifierBackend(ImageBackend):
    """A pretrained Hugging Face image classifier (the original default)."""

    kind = "classifier"

    def __init__(self, model_id: str, batch_size: int = 16):
        self.model_id = model_id
        self.name = f"clf:{model_id}"
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._ready = False
        self.fake_index = 0
        self.label_source = "unset"

    def _load(self):
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            import torch
            from transformers import (
                AutoImageProcessor, AutoModelForImageClassification,
            )

            tok = _token()
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_id, token=tok
            )
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_id, token=tok
            )
            self.model.eval()
            self.device = get_device()
            self.model.to(self.device)
            self.fake_index, self.label_source = find_fake_index(
                self.model.config.id2label
            )
            self._torch = torch
            self._ready = True

    def predict(self, images):
        self._load()
        torch = self._torch
        logits_all, emb_all = [], []
        imgs = [im.convert("RGB") for im in images]
        for i in range(0, len(imgs), self.batch_size):
            batch = imgs[i:i + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                out = self.model(**inputs, output_hidden_states=True)
            logits_all.append(out.logits.detach().float().cpu().numpy())
            hs = getattr(out, "hidden_states", None)
            if hs:
                emb_all.append(_pool_hidden(hs[-1]))
        logits = np.concatenate(logits_all, axis=0)
        emb = np.concatenate(emb_all, axis=0) if emb_all else None
        return fake_logit_from_logits(logits, self.fake_index), emb


class ProbeBackend(ImageBackend):
    """Frozen foundation encoder + trained linear probe.

    The recent literature finds frozen features with a linear head
    generalise better than fully fine-tuned or LoRA-tuned detectors, so this
    is the recommended backend once you have trained a probe
    (``python -m train.train_probe``).
    """

    kind = "probe"

    def __init__(self, embed_model_id: str, probe_path: str, batch_size: int = 16):
        self.embed_model_id = embed_model_id
        self.probe_path = probe_path
        self.name = f"probe:{embed_model_id}"
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._ready = False

    def _load(self):
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            import torch
            from transformers import AutoImageProcessor, AutoModel

            tok = _token()
            self.processor = AutoImageProcessor.from_pretrained(
                self.embed_model_id, token=tok
            )
            self.model = AutoModel.from_pretrained(self.embed_model_id, token=tok)
            self.model.eval()
            self.device = get_device()
            self.model.to(self.device)
            z = np.load(self.probe_path)
            self.w, self.b = z["w"].astype(np.float64), float(z["b"])
            self.mean, self.scale = z["mean"], z["scale"]
            self._torch = torch
            self._ready = True

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        self._load()
        return embed_with_encoder(
            self.model, self.processor, images, self.device, self._torch,
            self.batch_size,
        )

    def predict(self, images):
        emb = self.embed(images)
        return self.score_embeddings(emb), emb

    def score_embeddings(self, emb: np.ndarray) -> np.ndarray:
        self._load()
        return probe_logits(emb, self.w, self.b, self.mean, self.scale)


def embed_with_encoder(model, processor, images, device, torch, batch_size=16):
    """Image embeddings from any HF vision encoder (CLIP/SigLIP/DINO/ViT)."""
    out_all = []
    imgs = [im.convert("RGB") for im in images]
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            if hasattr(model, "get_image_features"):
                feats = model.get_image_features(pixel_values=inputs["pixel_values"])
                if not hasattr(feats, "detach"):  # newer versions return objects
                    feats = getattr(feats, "pooler_output", feats[0])
            else:
                o = model(**inputs)
                feats = getattr(o, "pooler_output", None)
                if feats is None:
                    feats = o.last_hidden_state.mean(dim=1)
        out_all.append(feats.detach().float().cpu().numpy())
    return np.concatenate(out_all, axis=0)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def probe_logits(emb, w, b, mean, scale) -> np.ndarray:
    e = (l2_normalize(np.asarray(emb, dtype=np.float64)) - mean) / np.maximum(
        scale, 1e-6
    )
    return e @ w + b


class StubBackend(ImageBackend):
    """Deterministic backend for tests and offline demos.

    ``fn(image) -> (fake_logit, embedding_vector)``. Never used in production
    unless explicitly registered."""

    kind = "stub"

    def __init__(self, fn: Callable, name: str = "stub"):
        self.fn = fn
        self.name = name

    def predict(self, images):
        logits, embs = [], []
        for im in images:
            z, e = self.fn(im)
            logits.append(z)
            embs.append(e)
        emb = None if embs[0] is None else np.stack(embs)
        return np.asarray(logits, dtype=np.float64), emb



class SigLIPBackend(ImageBackend):
    """Complementary semantic branch using a SigLIP / CLIP-style encoder.

    Produces embeddings that can be paired with a linear probe or used for
    semantic consistency checks. Not a pretrained deepfake classifier by
    itself; the probe (or a downstream head) supplies the fake logit.
    """

    kind = "siglip"

    def __init__(self, model_id: str, batch_size: int = 16):
        self.model_id = model_id
        self.name = f"siglip:{model_id}"
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._ready = False

    def _load(self):
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            import torch
            from transformers import AutoProcessor, AutoModel

            tok = _token()
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, token=tok
            )
            self.model = AutoModel.from_pretrained(self.model_id, token=tok)
            self.model.eval()
            self.device = get_device()
            self.model.to(self.device)
            self._torch = torch
            self._ready = True

    def predict(self, images):
        self._load()
        torch = self._torch
        emb_all = []
        imgs = [im.convert("RGB") for im in images]
        for i in range(0, len(imgs), self.batch_size):
            batch = imgs[i:i + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()
                      if k in ("pixel_values", "input_ids", "attention_mask")}
            with torch.inference_mode():
                if hasattr(self.model, "get_image_features"):
                    out = self.model.get_image_features(**inputs)
                else:
                    out = self.model(**inputs).pooler_output
            emb = out.detach().float().cpu().numpy()
            if emb.ndim == 3:
                emb = emb.mean(axis=1)
            emb_all.append(emb)
        emb = np.concatenate(emb_all, axis=0)
        # No native fake logit; return zeros so fusion treats it as embedding-only.
        return np.zeros(len(images), dtype=np.float64), emb


class FrequencyStubBackend(ImageBackend):
    """Placeholder for F3Net / SPSL-style frequency detectors.

    The production frequency path is the trained spectral head in
    ``detector.experts.spectral``. This backend exists so the registry can
    expose a frequency branch and the adaptive controller can budget it.
    When a real F3Net/SPSL checkpoint is available, replace this with a
    proper loader.
    """

    kind = "frequency"
    name = "frequency:stub"

    def predict(self, images):
        n = len(images)
        # Neutral logit; spectral expert supplies the real signal when trained.
        return np.zeros(n, dtype=np.float64), None


# --------------------------------------------------------------------------
# Face detection
# --------------------------------------------------------------------------
@dataclass
class Face:
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2 (original pixels)
    prob: float = 1.0
    landmarks: Optional[np.ndarray] = None  # (5, 2): eyes, nose, mouth corners

    @property
    def area(self) -> float:
        return max(0, self.box[2] - self.box[0]) * max(0, self.box[3] - self.box[1])


class FaceDetector:
    def detect(self, img: Image.Image) -> List[Face]:
        raise NotImplementedError


class MTCNNDetector(FaceDetector):
    def __init__(self, min_prob: float = 0.9):
        self.min_prob = min_prob
        self._m = None
        self._lock = threading.Lock()

    def _load(self):
        if self._m is None:
            with self._lock:
                if self._m is None:
                    from facenet_pytorch import MTCNN

                    dev = get_device()
                    # MTCNN is small and its CPU path is the most portable.
                    self._m = MTCNN(
                        keep_all=True, device="cuda" if dev == "cuda" else "cpu"
                    )

    def detect(self, img):
        self._load()
        try:
            boxes, probs, lms = self._m.detect(img.convert("RGB"), landmarks=True)
        except Exception as e:  # pragma: no cover - defensive
            log.warning("Face detection failed: %s", e)
            return []
        faces: List[Face] = []
        if boxes is None:
            return faces
        for b, p, lm in zip(boxes, probs, lms):
            if p is None or p < self.min_prob:
                continue
            faces.append(Face(tuple(int(round(v)) for v in b), float(p), lm))
        return faces


class HaarFaceDetector(FaceDetector):
    """Fallback detector (OpenCV Haar cascade, no extra dependency). Weaker
    than MTCNN and gives no landmarks, so the geometry expert is skipped."""

    def __init__(self):
        import cv2

        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self._c = cv2.CascadeClassifier(path)
        if self._c.empty():
            raise RuntimeError("OpenCV Haar cascade not found")

    def detect(self, img):
        import cv2

        gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        found = self._c.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
        return [Face((int(x), int(y), int(x + w), int(y + h)), 0.9, None)
                for (x, y, w, h) in found]


class NullFaceDetector(FaceDetector):
    """Last-resort detector: finds no faces, so whole images/frames are
    analysed (the pipeline flags this to the user)."""

    def detect(self, img):
        return []


class StubFaceDetector(FaceDetector):
    """Returns a fixed centred box (tests / offline demos)."""

    def __init__(self, frac: float = 0.6):
        self.frac = frac

    def detect(self, img):
        w, h = img.size
        bw, bh = int(w * self.frac), int(h * self.frac)
        x1, y1 = (w - bw) // 2, (h - bh) // 2
        lm = np.array([
            [x1 + 0.3 * bw, y1 + 0.35 * bh], [x1 + 0.7 * bw, y1 + 0.35 * bh],
            [x1 + 0.5 * bw, y1 + 0.55 * bh],
            [x1 + 0.35 * bw, y1 + 0.78 * bh], [x1 + 0.65 * bw, y1 + 0.78 * bh],
        ])
        return [Face((x1, y1, x1 + bw, y1 + bh), 0.99, lm)]


def crop_with_margin(
    img: Image.Image, box, margin: float = 0.3
) -> Image.Image:
    """Crop a face with extra context. Face-swap blending artifacts live on
    the face boundary, which tight boxes cut off (a flaw of the original)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    mx, my = int(w * margin), int(h * margin)
    W, H = img.size
    return img.crop((max(0, x1 - mx), max(0, y1 - my),
                     min(W, x2 + mx), min(H, y2 + my)))


# --------------------------------------------------------------------------
# Audio backend
# --------------------------------------------------------------------------
class AudioBackend:
    name = "audio"

    def predict(self, chunks: Sequence[np.ndarray], sr: int = 16000):
        raise NotImplementedError


class HFAudioBackend(AudioBackend):
    def __init__(self, model_id: str, batch_size: int = 8):
        self.model_id = model_id
        self.name = f"audio:{model_id}"
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._ready = False

    def _load(self):
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            import torch
            from transformers import (
                AutoFeatureExtractor, AutoModelForAudioClassification,
            )

            tok = _token()
            self.extractor = AutoFeatureExtractor.from_pretrained(
                self.model_id, token=tok
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_id, token=tok
            )
            self.model.eval()
            self.device = get_device()
            self.model.to(self.device)
            self.fake_index, self.label_source = find_fake_index(
                self.model.config.id2label
            )
            self._torch = torch
            self._ready = True

    def predict(self, chunks, sr=16000):
        self._load()
        torch = self._torch
        logits_all, emb_all = [], []
        for i in range(0, len(chunks), self.batch_size):
            batch = [np.asarray(c, dtype=np.float32) for c in chunks[i:i + self.batch_size]]
            inputs = self.extractor(
                batch, sampling_rate=sr, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                out = self.model(**inputs, output_hidden_states=True)
            logits_all.append(out.logits.detach().float().cpu().numpy())
            hs = getattr(out, "hidden_states", None)
            if hs:
                emb_all.append(_pool_hidden(hs[-1]))
        logits = np.concatenate(logits_all, axis=0)
        emb = np.concatenate(emb_all, axis=0) if emb_all else None
        return fake_logit_from_logits(logits, self.fake_index), emb


class StubAudioBackend(AudioBackend):
    kind = "stub"

    def __init__(self, fn: Callable, name: str = "stub-audio"):
        self.fn = fn
        self.name = name

    def predict(self, chunks, sr=16000):
        out = [self.fn(np.asarray(c)) for c in chunks]
        z = np.asarray([o[0] for o in out], dtype=np.float64)
        e = None if out[0][1] is None else np.stack([o[1] for o in out])
        return z, e


# --------------------------------------------------------------------------
# Registry (overridable for tests)
# --------------------------------------------------------------------------
_REG = {"image": None, "audio": None, "face": None}
_REG_LOCK = threading.Lock()


def _probe_encoder_model_id(probe_path: str, fallback: str) -> str:
    with np.load(probe_path, allow_pickle=False) as artifact:
        if "encoder" not in artifact.files:
            return fallback
        value = artifact["encoder"]
        if value.ndim != 0:
            raise ValueError(f"Probe encoder metadata must be scalar: {probe_path}")
        encoder_id = value.item()
    if isinstance(encoder_id, bytes):
        encoder_id = encoder_id.decode("utf-8")
    if not isinstance(encoder_id, str):
        raise ValueError(f"Probe encoder metadata must be a string: {probe_path}")
    encoder_id = encoder_id.strip()
    if not encoder_id:
        return fallback
    if fallback and encoder_id != fallback:
        log.info("Using probe-matched encoder %s instead of configured %s",
                 encoder_id, fallback)
    return encoder_id


def get_image_backends() -> List[ImageBackend]:
    with _REG_LOCK:
        if _REG["image"] is None:
            s = get_settings()
            backends: List[ImageBackend] = [HFClassifierBackend(s.image_model_id)]
            probe = os.path.join(s.artifact_dir, "probe.npz")
            if os.path.exists(probe):
                embed_model_id = _probe_encoder_model_id(probe, s.embed_model_id)
            else:
                embed_model_id = s.embed_model_id
            if embed_model_id and os.path.exists(probe):
                # A probe trained on your data is the primary Tier-1 screen
                # (frozen foundation features generalise best); the
                # pretrained classifier stays as a second, independent vote.
                backends.insert(0, ProbeBackend(embed_model_id, probe))
            # Optional complementary SigLIP semantic branch (recommended stack).
            if getattr(s, "siglip_model_id", "") and s.siglip_model_id:
                try:
                    backends.append(SigLIPBackend(s.siglip_model_id))
                except Exception as e:  # pragma: no cover
                    log.warning("SigLIP backend unavailable: %s", e)
            # Frequency branch placeholder (spectral expert is the real path).
            backends.append(FrequencyStubBackend())
            _REG["image"] = backends
        return _REG["image"]


def get_audio_backend() -> AudioBackend:
    with _REG_LOCK:
        if _REG["audio"] is None:
            _REG["audio"] = HFAudioBackend(get_settings().audio_model_id)
        return _REG["audio"]


def get_face_detector() -> FaceDetector:
    with _REG_LOCK:
        if _REG["face"] is None:
            try:
                import facenet_pytorch  # noqa: F401

                _REG["face"] = MTCNNDetector()
            except ImportError:
                log.warning(
                    "facenet-pytorch is not installed; falling back to the "
                    "OpenCV Haar face detector (less accurate, no landmarks). "
                    "Install it with: pip install --no-deps facenet-pytorch")
                try:
                    _REG["face"] = HaarFaceDetector()
                except Exception:
                    log.warning("No face detector available; analysing whole images.")
                    _REG["face"] = NullFaceDetector()
        return _REG["face"]


def set_backends(image=None, audio=None, face=None) -> None:
    """Inject backends (tests, notebooks, custom models)."""
    with _REG_LOCK:
        if image is not None:
            _REG["image"] = list(image) if isinstance(image, (list, tuple)) else [image]
        if audio is not None:
            _REG["audio"] = audio
        if face is not None:
            _REG["face"] = face


def reset_backends() -> None:
    with _REG_LOCK:
        for k in _REG:
            _REG[k] = None
