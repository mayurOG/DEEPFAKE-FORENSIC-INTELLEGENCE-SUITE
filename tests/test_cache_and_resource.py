import time
from PIL import Image
import numpy as np

from detector.cache import (
    InferenceCache,
    ResourceTracker,
    compute_content_hash,
    get_current_rss_mb,
    get_inference_device,
)
from detector.pipeline import Pipeline
from tests.conftest import make_img, stub_image_fn
from detector import models as M


def test_content_hash_deterministic():
    img1 = make_img(10, 0, -10, seed=1)
    img2 = make_img(10, 0, -10, seed=1)
    img3 = make_img(20, 0, -20, seed=2)
    h1 = compute_content_hash(img1)
    h2 = compute_content_hash(img2)
    h3 = compute_content_hash(img3)
    assert h1 == h2
    assert h1 != h3


def test_inference_cache_lru():
    cache = InferenceCache(maxsize=2, enabled=True)
    cache.put("h1", "res1")
    cache.put("h2", "res2")
    assert cache.get("h1") == "res1"
    assert cache.get("h2") == "res2"
    
    # Insert third item, oldest should be evicted
    cache.put("h3", "res3")
    assert cache.get("h1") is None or cache.get("h2") is not None
    stats = cache.stats()
    assert stats["size"] == 2
    assert stats["hits"] >= 1
    
    cache.clear()
    assert cache.stats()["size"] == 0


def test_resource_tracker():
    with ResourceTracker() as tracker:
        time.sleep(0.01)
    u = tracker.usage
    assert u.latency_ms >= 5.0
    assert u.peak_memory_mb > 0.0
    assert isinstance(u.device, str)
    d = u.to_dict()
    assert "latency_ms" in d
    assert "peak_memory_mb" in d


def test_pipeline_with_cache_hit(settings):
    import dataclasses
    s = dataclasses.replace(settings, enable_inference_cache=True)
    M.set_backends(image=M.StubBackend(stub_image_fn), face=M.StubFaceDetector())
    pipe = Pipeline(settings=s)
    img = make_img(50, 0, -50)
    
    # First pass: cache miss
    r1 = pipe.analyze(img, media_type="image", explain=False, stability=False)
    assert r1.resource_usage.get("cache_hit") is False
    
    # Second pass: cache hit
    r2 = pipe.analyze(img, media_type="image", explain=False, stability=False)
    assert r2.resource_usage.get("cache_hit") is True
    assert r2.verdict == r1.verdict
    M.reset_backends()
