import dataclasses
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import models as M  # noqa: E402
from detector.config import get_settings  # noqa: E402
from detector.pipeline import Pipeline  # noqa: E402


def stub_image_fn(im):
    a = np.asarray(im.convert("RGB"), dtype=np.float64)
    z = (a[..., 0].mean() - a[..., 2].mean()) / 8.0
    emb = np.array([a[..., 0].mean(), a[..., 1].mean(), a[..., 2].mean(), a.std()])
    return z, emb


def stub_audio_fn(chunk):
    return float(np.std(chunk) * 20 - 2.0), np.zeros(4)


def make_img(r, g, b, seed=0, size=256):
    rng = np.random.default_rng(seed)
    a = rng.normal(128, 20, (size, size, 3))
    a[..., 0] += r
    a[..., 1] += g
    a[..., 2] += b
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


@pytest.fixture
def art_dir(tmp_path):
    return str(tmp_path / "artifacts")


@pytest.fixture
def settings(art_dir, monkeypatch):
    monkeypatch.setenv("DEEPFAKE_ARTIFACT_DIR", art_dir)
    return dataclasses.replace(get_settings(), artifact_dir=art_dir)


@pytest.fixture
def pipe(settings):
    M.set_backends(image=M.StubBackend(stub_image_fn), face=M.StubFaceDetector(),
                   audio=M.StubAudioBackend(stub_audio_fn))
    yield Pipeline(settings=settings)
    M.reset_backends()


@pytest.fixture
def dataset(tmp_path):
    """Synthetic real/ + fake/ folders (reddish = fake, bluish = real)."""
    root = tmp_path / "data" / "toy"
    for label, (r, b) in {"real": (-50, 50), "fake": (50, -50)}.items():
        (root / label).mkdir(parents=True)
        for i in range(24):
            make_img(r, 0, b, seed=i + (100 if label == "fake" else 0), size=96
                     ).save(root / label / f"{label}_{i // 2}_{i}.png")
    return str(root)


@pytest.fixture(scope="session")
def tiny_encoder(tmp_path_factory):
    """Tiny random ViT saved locally so encoder code runs fully offline."""
    from transformers import ViTConfig, ViTImageProcessor, ViTModel
    d = str(tmp_path_factory.mktemp("tinyvit"))
    cfg = ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                    intermediate_size=64, image_size=32, patch_size=16)
    ViTModel(cfg).save_pretrained(d)
    ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(d)
    return d
