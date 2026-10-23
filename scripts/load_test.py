"""Concurrent load test for the analysis pipeline / Streamlit backend (P1).

Uses a thread pool against ``Pipeline.analyze`` (same path the UI uses).
Reports throughput, latency distribution, error rate and peak RSS.

    python -m scripts.load_test --image sample.jpg --workers 4 --requests 20
    python -m scripts.load_test --image a.jpg --video b.mp4 --workers 8 --requests 40 \\
        --out artifacts/load_test.json

This is an in-process stress test of the detector, not a full HTTP soak of
Streamlit's websocket layer. For HTTP-level testing, point locust/k6 at a
running ``streamlit run app.py`` instance separately.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("deepfake.loadtest")


def _rss_mb() -> Optional[float]:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return None


def _one(pipe, path: str, media: Optional[str]) -> Tuple[float, Optional[str], Optional[str]]:
    t0 = time.perf_counter()
    err = None
    verdict = None
    try:
        res = pipe.analyze(path, media_type=media, explain=False, stability=False)
        verdict = getattr(res, "verdict", None)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    return time.perf_counter() - t0, verdict, err


def run_load_test(
    jobs: List[Tuple[str, Optional[str]]],
    workers: int,
    requests: int,
) -> Dict[str, Any]:
    from detector.pipeline import get_pipeline, reset_pipeline
    reset_pipeline()
    # Initialise once in main thread
    pipe = get_pipeline()
    lock = threading.Lock()
    # Shared pipeline — backends are locked internally; measure contention.
    latencies: List[float] = []
    errors: List[str] = []
    verdicts: List[str] = []

    # Round-robin paths
    expanded = [jobs[i % len(jobs)] for i in range(requests)]

    t_wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, pipe, path, mt) for path, mt in expanded]
        for fut in as_completed(futs):
            dt, verdict, err = fut.result()
            with lock:
                latencies.append(dt)
                if err:
                    errors.append(err)
                if verdict:
                    verdicts.append(verdict)
    wall = time.perf_counter() - t_wall0

    lat_ms = sorted(x * 1000 for x in latencies)
    def pct(p):
        if not lat_ms:
            return None
        return lat_ms[min(len(lat_ms) - 1, max(0, int(p * len(lat_ms)) - 1))]

    return {
        "workers": workers,
        "requests": requests,
        "wall_seconds": round(wall, 3),
        "throughput_rps": round(requests / wall, 3) if wall > 0 else None,
        "latency_ms": {
            "min": round(lat_ms[0], 1) if lat_ms else None,
            "median": round(statistics.median(lat_ms), 1) if lat_ms else None,
            "p95": round(pct(0.95), 1) if lat_ms else None,
            "p99": round(pct(0.99), 1) if lat_ms else None,
            "max": round(lat_ms[-1], 1) if lat_ms else None,
        },
        "error_rate": round(len(errors) / requests, 4) if requests else None,
        "n_errors": len(errors),
        "sample_errors": errors[:5],
        "verdict_counts": {v: verdicts.count(v) for v in set(verdicts)},
        "rss_mb": _rss_mb(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--video", action="append", default=[])
    ap.add_argument("--audio", action="append", default=[])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--out", default="artifacts/load_test.json")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    jobs: List[Tuple[str, Optional[str]]] = (
        [(p, "image") for p in a.image]
        + [(p, "video") for p in a.video]
        + [(p, "audio") for p in a.audio]
    )
    if not jobs:
        # synthetic fallback
        from PIL import Image
        import tempfile
        img = Image.new("RGB", (224, 224), (90, 120, 150))
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(tmp)
        jobs = [(tmp, "image")]
        log.info("No media given; using synthetic PNG %s", tmp)

    for path, _ in jobs:
        if not os.path.isfile(path):
            raise SystemExit(f"File not found: {path}")

    report = run_load_test(jobs, a.workers, a.requests)
    report["jobs"] = [{"path": p, "media": m} for p, m in jobs]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({
        "throughput_rps": report["throughput_rps"],
        "latency_ms": report["latency_ms"],
        "error_rate": report["error_rate"],
        "rss_mb": report["rss_mb"],
        "out": a.out,
    }, indent=2))
    return report


if __name__ == "__main__":
    main()
