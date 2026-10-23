"""Drift canary: is the detector still fresh against *recent* generators?

Detectors decay as generators improve: a model that scored well last year can
miss most fakes from this year's tools. Keep a small set of labelled samples
from the newest generators you can obtain (``canary/manifest.json``), re-run
it regularly, and compare with the baseline recorded when you last
calibrated. The result is shown as a freshness badge in the app.

    python -m scripts.run_canary --set-baseline     # after (re)calibration
    python -m scripts.run_canary                    # regular check

Manifest format::

    {"items": [{"path": "new_gen_01.jpg", "label": 1,
                "generator": "SomeNewTool", "added": "2026-10"}, ...]}
Paths are relative to the manifest file; label 1 = fake, 0 = real.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

from detector import metrics as MT
from detector.config import get_settings
from detector.pipeline import Pipeline, infer_media_type

DEFAULT_MANIFEST = os.path.join("canary", "manifest.json")


def _paths():
    d = get_settings().artifact_dir
    return (os.path.join(d, "canary_baseline.json"), os.path.join(d, "canary_last.json"))


def run_canary(pipe: Pipeline, manifest_path: str = DEFAULT_MANIFEST,
               mode: str = "balanced") -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    ys, ps, verdicts, by_gen = [], [], [], {}
    for it in manifest.get("items", []):
        path = os.path.join(base_dir, it["path"])
        mt = it.get("media_type") or infer_media_type(path)
        if mt is None or not os.path.exists(path):
            continue
        r = pipe.analyze(path, media_type=mt, mode=mode, explain=False, stability=False)
        ys.append(int(it["label"]))
        ps.append(r.p_fake)
        verdicts.append(r.verdict)
        g = it.get("generator", "unknown")
        d = by_gen.setdefault(g, {"n": 0, "missed": 0})
        d["n"] += 1
        d["missed"] += int(int(it["label"]) == 1 and r.verdict == "Real")
    out: Dict[str, Any] = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "n": len(ys),
        "auc": MT.auc(ys, ps) if ys else None,
        "decisions": MT.selective_metrics(ys, verdicts) if ys else {},
        "by_generator": by_gen,
    }
    out["miss_rate"] = out["decisions"].get("miss_rate") if ys else None
    return out


def freshness(current: Dict[str, Any], baseline: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare a canary run with the baseline. Status: fresh | watch | stale |
    no_baseline | no_data."""
    if not current or not current.get("n"):
        return {"status": "no_data", "reason": "Canary set is empty."}
    if not baseline:
        return {"status": "no_baseline",
                "reason": "No baseline yet; run with --set-baseline after calibrating."}
    reasons = []
    status = "fresh"

    def bump(level):
        nonlocal status
        order = ["fresh", "watch", "stale"]
        if order.index(level) > order.index(status):
            status = level

    a0, a1 = baseline.get("auc"), current.get("auc")
    if a0 is not None and a1 is not None and a0 == a0 and a1 == a1:
        drop = a0 - a1
        if drop > 0.10:
            bump("stale"); reasons.append(f"AUC fell by {drop:.2f}")
        elif drop > 0.05:
            bump("watch"); reasons.append(f"AUC fell by {drop:.2f}")
    m0, m1 = baseline.get("miss_rate"), current.get("miss_rate")
    if m0 is not None and m1 is not None:
        rise = m1 - m0
        if rise > 0.15:
            bump("stale"); reasons.append(f"miss rate rose by {rise:.2f}")
        elif rise > 0.07:
            bump("watch"); reasons.append(f"miss rate rose by {rise:.2f}")
    return {"status": status, "reason": "; ".join(reasons) or "Within baseline."}


def save_run(result: Dict[str, Any], set_baseline: bool = False) -> None:
    base_p, last_p = _paths()
    os.makedirs(os.path.dirname(os.path.abspath(last_p)), exist_ok=True)
    baseline = None
    if os.path.exists(base_p):
        with open(base_p, "r", encoding="utf-8") as f:
            baseline = json.load(f)
    if set_baseline:
        with open(base_p, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        baseline = result
    result = dict(result)
    result["freshness"] = freshness(result, baseline)
    with open(last_p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def load_last() -> Optional[Dict[str, Any]]:
    _, last_p = _paths()
    if not os.path.exists(last_p):
        return None
    try:
        with open(last_p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
