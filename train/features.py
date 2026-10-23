"""Embedding extraction with a frozen (or fine-tuned) vision encoder."""
from __future__ import annotations

import argparse
import logging
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from detector import models as M
from detector.config import get_settings
from train.data import Item, load_images, scan_folder

log = logging.getLogger("deepfake.train")


class Encoder:
    """Any Hugging Face vision encoder (CLIP / SigLIP / DINO / ViT), local
    directory or hub id. Lazily loaded."""

    def __init__(self, model_id: str, batch_size: int = 16):
        self.model_id = model_id
        self.batch_size = batch_size
        self._ready = False

    def _load(self):
        if self._ready:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModel

        tok = get_settings().hf_token or None
        self.processor = AutoImageProcessor.from_pretrained(self.model_id, token=tok)
        self.model = AutoModel.from_pretrained(self.model_id, token=tok).eval()
        self.device = M.get_device()
        self.model.to(self.device)
        self.torch = torch
        self._ready = True

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        self._load()
        return M.embed_with_encoder(self.model, self.processor, images,
                                    self.device, self.torch, self.batch_size)


def face_crop(img: Image.Image, detector, use_faces: bool = True,
              margin: float = 0.3) -> Image.Image:
    if not use_faces:
        return img
    faces = detector.detect(img)
    if not faces:
        return img
    f = max(faces, key=lambda ff: ff.area)
    return M.crop_with_margin(img, f.box, margin)


def extract(items: Sequence[Item], encoder: Encoder, detector=None,
            n_frames: int = 8, use_faces: bool = True):
    """Returns dict(emb (N,d), y (N,), groups (N,), paths (N,)). One row per
    image / per sampled video frame."""
    detector = detector or M.get_face_detector()
    embs, ys, gs, ps = [], [], [], []
    batch_size = int(getattr(encoder, "batch_size", 1))
    if batch_size < 1:
        raise ValueError("encoder batch_size must be positive")
    pending_images, pending_rows = [], []

    def flush():
        if not pending_images:
            return
        try:
            encoded = encoder.embed(pending_images)
            if len(encoded) != len(pending_rows):
                raise ValueError("encoder returned a different number of embeddings than images")
            embs.append(encoded)
            ys.extend(row[0] for row in pending_rows)
            gs.extend(row[1] for row in pending_rows)
            ps.extend(row[2] for row in pending_rows)
        except Exception as ex:
            log.warning("batched feature extraction failed for %d images: %s; retrying individually",
                        len(pending_images), ex)
            for im, row in zip(pending_images, pending_rows):
                try:
                    encoded = encoder.embed([im])
                    if len(encoded) != 1:
                        raise ValueError("encoder returned a different number of embeddings than images")
                except Exception as item_ex:
                    log.warning("skip %s: %s", row[2], item_ex)
                    continue
                embs.append(encoded)
                ys.append(row[0])
                gs.append(row[1])
                ps.append(row[2])
        pending_images.clear()
        pending_rows.clear()

    for k, it in enumerate(items):
        try:
            crops = [face_crop(im, detector, use_faces) for im in load_images(it, n_frames)]
            if not crops:
                continue
        except Exception as ex:
            log.warning("skip %s: %s", it.path, ex)
            continue
        for im in crops:
            pending_images.append(im)
            pending_rows.append((it.label, it.group, it.path))
            if len(pending_images) >= batch_size:
                flush()
        if (k + 1) % 50 == 0:
            log.info("extracted %d/%d", k + 1, len(items))
    flush()
    if not embs:
        raise RuntimeError("No features extracted; check the dataset layout.")
    return {"emb": np.concatenate(embs).astype(np.float32),
            "y": np.asarray(ys, dtype=np.int64),
            "groups": np.asarray(gs), "paths": np.asarray(ps)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cache encoder embeddings for a dataset folder.")
    ap.add_argument("--data", required=True, help="folder with real/ and fake/")
    ap.add_argument("--encoder", default=get_settings().embed_model_id,
                    help="HF model id or local dir of the vision encoder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=8, help="frames per video")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="images per encoder batch (reduce for low VRAM)")
    ap.add_argument("--no-faces", action="store_true", help="skip face cropping")
    ap.add_argument("--max-items", type=int, default=None)
    a = ap.parse_args(argv)
    if not a.encoder:
        ap.error("--encoder (or EMBED_MODEL_ID) is required")
    logging.basicConfig(level=logging.INFO)
    items = [i for i in scan_folder(a.data, max_items=a.max_items) if i.media_type in ("image", "video")]
    out = extract(items, Encoder(a.encoder, batch_size=a.batch_size),
                   n_frames=a.frames, use_faces=not a.no_faces)
    np.savez_compressed(a.out, encoder=a.encoder, **out)
    print(f"saved {len(out['y'])} rows to {a.out} (real={int((out['y']==0).sum())}, fake={int((out['y']==1).sum())})")


if __name__ == "__main__":
    main()
