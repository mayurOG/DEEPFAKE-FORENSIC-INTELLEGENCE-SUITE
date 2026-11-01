"""Inference caching and system resource monitoring (Priority 2, Item 17).

Provides:
1. InferenceCache: Thread-safe in-memory LRU cache with SHA-256 content keying.
   Prevents redundant neural inference passes when the same media is re-analyzed
   or evaluated across multiple dashboards.
2. ResourceMonitor: Measures RSS memory, device status (CUDA, MPS, CPU),
   and microsecond-precision inference latency per pipeline run.
"""
from __future__ import annotations

import collections
import hashlib
import io
import os
import resource
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


def compute_content_hash(source: Any) -> str:
    """Compute SHA-256 for a file path, bytes, or PIL Image."""
    h = hashlib.sha256()
    if isinstance(source, (str, os.PathLike)) and os.path.isfile(source):
        with open(source, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    elif isinstance(source, bytes):
        h.update(source)
        return h.hexdigest()
    elif hasattr(source, "tobytes"):  # PIL Image or numpy array
        try:
            h.update(source.tobytes())
            return h.hexdigest()
        except Exception:
            pass
    return hashlib.sha256(str(source).encode("utf-8")).hexdigest()


@dataclass
class ResourceUsage:
    latency_ms: float
    peak_memory_mb: float
    current_memory_mb: float
    device: str
    cache_hit: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latency_ms": round(self.latency_ms, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "current_memory_mb": round(self.current_memory_mb, 2),
            "device": self.device,
            "cache_hit": self.cache_hit,
            "timestamp": self.timestamp,
        }


def get_current_rss_mb() -> float:
    """Get current process resident set size in megabytes."""
    try:
        # On macOS ru_maxrss is in bytes; on Linux it is in kilobytes.
        import platform
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return float(raw / (1024.0 * 1024.0))
        return float(raw / 1024.0)
    except Exception:
        return 0.0


def get_inference_device() -> str:
    """Detect current torch acceleration device."""
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name()})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps (Apple Silicon Metal)"
        return "cpu"
    except Exception:
        return "cpu"


class ResourceTracker:
    """Context manager for tracking inference elapsed time and memory."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or get_inference_device()
        self.t0 = 0.0
        self.t1 = 0.0
        self.mem_before = 0.0
        self.mem_after = 0.0

    def __enter__(self):
        self.mem_before = get_current_rss_mb()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t1 = time.perf_counter()
        self.mem_after = get_current_rss_mb()

    @property
    def usage(self) -> ResourceUsage:
        elapsed_ms = (self.t1 - self.t0) * 1000.0
        return ResourceUsage(
            latency_ms=elapsed_ms,
            peak_memory_mb=max(self.mem_before, self.mem_after),
            current_memory_mb=self.mem_after,
            device=self.device,
            cache_hit=False,
        )


class InferenceCache:
    """Thread-safe LRU cache for pipeline analysis results."""

    def __init__(self, maxsize: int = 128, enabled: bool = True):
        self.maxsize = maxsize
        self.enabled = enabled
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, content_hash: str, mode: str, explain: bool, stability: bool) -> str:
        return f"{content_hash}_{mode}_{int(explain)}_{int(stability)}"

    def get(self, content_hash: str, mode: str = "balanced", explain: bool = True, stability: bool = False) -> Optional[Any]:
        if not self.enabled or not content_hash:
            return None
        key = self._make_key(content_hash, mode, explain, stability)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                self._cache.move_to_end(key)
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, content_hash: str, result: Any, mode: str = "balanced", explain: bool = True, stability: bool = False) -> None:
        if not self.enabled or not content_hash:
            return
        key = self._make_key(content_hash, mode, explain, stability)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = result
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_ratio = round(self._hits / total, 3) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": hit_ratio,
                "enabled": self.enabled,
            }


_GLOBAL_CACHE = InferenceCache()


def get_inference_cache() -> InferenceCache:
    return _GLOBAL_CACHE
