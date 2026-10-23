"""Deployment / inference performance benchmark (P1).

Measures cold-start, warm latency, peak RSS and simple throughput for the
pipeline. Run before promising a public demo.

    python -m scripts.benchmark_inference --image path/to.jpg --out artifacts/bench.json
    python -m scripts.benchmark_inference --image a.jpg --video b.mp4 --runs 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import tracemalloc
from typing import Any, Dict, List, Optional

log = logging.getLogger("deepfake.bench")


def _rss_mb() -> Optional[float]:
    try:
        import resource
        # Linux: ru_maxrss is KB
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return None


def bench_one(pipe, path: str, media: Optional[str], runs: int) -> Dict[str, Any]:
    from detector.pipeline import infer_media_type
    mt = media or infer_media_type(path)
    times = []
    last = None
    for i in range(runs):
        t0 = time.perf_counter()
        last = pipe.analyze(path, media_type=mt, explain=False, stability=False)
        times.append(time.perf_counter() - t0)
    times_ms = [round(t * 1000, 1) for t in times]
    return {
        "path": path,
        "media_type": mt,
        "runs": runs,
        "latency_ms": times_ms,
        "median_ms": sorted(times_ms)[len(times_ms) // 2],
        "p95_ms": sorted(times_ms)[max(0, int(0.95 * len(times_ms)) - 1)],
        "verdict": getattr(last, "verdict", None),
        "tier": getattr(last, "tier", None),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--video", action="append", default=[])
    ap.add_argument("--audio", action="append", default=[])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="artifacts/bench.json")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    report: Dict[str, Any] = {"runs_per_file": a.runs, "items": []}

    # Cold start: import + first pipeline construction
    tracemalloc.start()
    t0 = time.perf_counter()
    from detector.pipeline import get_pipeline, reset_pipeline
    reset_pipeline()
    pipe = get_pipeline()
    cold_s = time.perf_counter() - t0
    report["cold_start_seconds"] = round(cold_s, 3)
    report["rss_mb_after_init"] = _rss_mb()
    current, peak = tracemalloc.get_traced_memory()
    report["tracemalloc_peak_mb_init"] = round(peak / (1024 * 1024), 2)

    jobs = (
        [(p, "image") for p in a.image]
        + [(p, "video") for p in a.video]
        + [(p, "audio") for p in a.audio]
    )
    if not jobs:
        # Synthetic tiny image so the script still produces a number offline
        from PIL import Image
        import tempfile
        img = Image.new("RGB", (256, 256), (128, 100, 80))
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(tmp)
        jobs = [(tmp, "image")]
        report["note"] = "No media provided; used a synthetic 256x256 PNG."

    for path, mt in jobs:
        if not os.path.isfile(path):
            report["items"].append({"path": path, "error": "file not found"})
            continue
        try:
            report["items"].append(bench_one(pipe, path, mt, a.runs))
        except Exception as e:
            report["items"].append({"path": path, "error": f"{type(e).__name__}: {e}"})

    report["rss_mb_after_runs"] = _rss_mb()
    _, peak = tracemalloc.get_traced_memory()
    report["tracemalloc_peak_mb_total"] = round(peak / (1024 * 1024), 2)
    tracemalloc.stop()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({
        "cold_start_s": report["cold_start_seconds"],
        "rss_mb": report.get("rss_mb_after_runs"),
        "n_items": len(report["items"]),
        "out": a.out,
    }, indent=2))
    return report


if __name__ == "__main__":
    main()
