"""Dataset scanning and leakage-safe splitting.

Expected layout (any depth below real/ and fake/):

    data/<dataset_name>/real/**.{jpg,png,mp4,wav,...}
    data/<dataset_name>/fake/**.{jpg,png,mp4,wav,...}

Frames extracted from the same video MUST NOT land on both sides of a
split (that inflates scores massively). ``group_id`` collapses
``vid001_frame0012.jpg`` and ``vid001_frame0099.jpg`` to the group
``vid001`` and splits are done per group.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from detector.pipeline import AUDIO_EXT, IMAGE_EXT, VIDEO_EXT, infer_media_type

_FRAME_SUFFIX = re.compile(r"[_\-\.]?(frame|f|img|image)?[_\-\.]?\d+$", re.I)


@dataclass
class Item:
    path: str
    label: int  # 0 = real, 1 = fake
    group: str
    media_type: str
    dataset: str


def group_id(path: str, root: str = "") -> str:
    rel = os.path.relpath(path, root) if root else path
    d = os.path.dirname(rel)
    stem = os.path.splitext(os.path.basename(rel))[0]
    celeb_identity = re.match(r"^(id\d+)(?:_id\d+)?(?:[_-]\d+)?$", stem, re.I)
    if celeb_identity:
        return celeb_identity.group(1).lower()
    if re.match(r"^(?:real|fake)[_-]?\d+$", stem, re.I):
        return f"{d}/{stem}"
    stem = _FRAME_SUFFIX.sub("", stem) or stem
    return f"{d}/{stem}"


def scan_folder(root: str, media: Optional[str] = None,
                max_items: Optional[int] = None, seed: int = 0) -> List[Item]:
    root = os.path.abspath(root)
    dataset = os.path.basename(root.rstrip("/"))
    items: List[Item] = []
    for label_name, label in (("real", 0), ("fake", 1)):
        base = os.path.join(root, label_name)
        if not os.path.isdir(base):
            continue
        for dp, _, files in os.walk(base):
            for fn in sorted(files):
                mt = infer_media_type(fn)
                if mt is None or (media and mt != media):
                    continue
                p = os.path.join(dp, fn)
                items.append(Item(p, label, group_id(p, root), mt, dataset))
    if max_items and len(items) > max_items:
        rng = np.random.default_rng(seed)
        items = [items[i] for i in rng.permutation(len(items))[:max_items]]
    return items


def group_split(items: Sequence[Item], val_fraction: float = 0.2,
                seed: int = 0) -> Tuple[List[Item], List[Item]]:
    """Split by group so no video/identity appears on both sides."""
    groups = sorted({it.group for it in items})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * val_fraction)))
    val_groups = set(groups[:n_val])
    train = [it for it in items if it.group not in val_groups]
    val = [it for it in items if it.group in val_groups]
    return train, val


def load_images(item: Item, n_frames: int = 8) -> Iterator[Image.Image]:
    """Yield PIL images for an item (one for an image, n frames for a video)."""
    if item.media_type == "image":
        yield Image.open(item.path).convert("RGB")
    elif item.media_type == "video":
        from detector.pipeline import sample_frames

        for _, _, im in sample_frames(item.path, n_frames):
            yield im
    else:
        return


def load_waveform(item: Item, sr: int = 16000, max_seconds: float = 30.0):
    """Load mono float32 waveform for an audio item or video with audio track.

    Returns None if no usable audio is present.
    """
    from detector.audio_io import load_audio
    wave = load_audio(item.path, sr=sr)
    if wave is None or len(wave) < sr // 4:  # < 0.25 s
        return None
    max_samples = int(max_seconds * sr)
    if len(wave) > max_samples:
        # centre crop for long files
        start = (len(wave) - max_samples) // 2
        wave = wave[start:start + max_samples]
    return wave.astype(np.float32)
