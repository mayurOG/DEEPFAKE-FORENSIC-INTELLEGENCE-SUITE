"""Shared helpers for the EGAFR evaluation scripts.

Protocol points that matter for a fair comparison:

* every configuration is *calibrated as it will be deployed*: its own natural
  path (early exit, controller, ...) is run on the calibration split and the
  calibrators are fitted on those outputs, so a configuration is never judged
  with another configuration's calibration;
* learned reliabilities are fitted first and applied before calibration
  (two passes), so the calibrator sees the logits it will be applied to;
* calibration and test items are disjoint, and nothing from the test split
  is used to choose a threshold.
"""
from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from detector import metrics as MT
from detector.calibration import CalibrationBundle, OODModel, fit_calibrator
from detector.config import get_settings
from detector.pipeline import Pipeline

log = logging.getLogger("deepfake.eval")


def make_pipe(media: str, calibration=None, ood=None, **flags) -> Pipeline:
    s = dataclasses.replace(get_settings(), **flags)
    return Pipeline(settings=s, calibration=calibration, ood=ood)


def score_items(pipe: Pipeline, items, media: str, mode: str = "balanced",
                stability: bool = False, explain: bool = False,
                force_full: bool = False) -> List[Dict[str, Any]]:
    rows = []
    for it in items:
        t = time.time()
        try:
            r = pipe.analyze(it.path, media_type=media, mode=mode, explain=explain,
                             stability=stability, force_full=force_full)
        except Exception as e:  # keep going on a corrupt file
            log.warning("skip %s: %s", it.path, e)
            continue
        used = [e for e in r.evidence if e["status"] == "used"]
        rows.append({
            "path": it.path, "y": it.label, "p": r.p_fake, "verdict": r.verdict,
            "tier": r.tier, "calibrated": r.calibrated,
            "tier1_logit": r.raw.get("tier1_logit"), "final_logit": r.raw.get("final_logit"),
            "emb": r.raw.get("embedding"), "ctx": r.raw.get("ctx"),
            "expert": r.raw.get("expert_logits"),
            "calls": len(used), "latency": time.time() - t,
            "sufficiency": (r.graph.get("fusion") or {}).get("sufficiency"),
            "fragility": (r.robustness or {}).get("fragility"),
            "abstention": r.abstention_reason})
    return rows


def fit_for_config(calib_rows: Sequence[Dict[str, Any]], media: str, alpha: float,
                   min_final: int = 60):
    """(CalibrationBundle, ood dict) fitted on a configuration's own outputs."""
    y = np.array([r["y"] for r in calib_rows])
    bundle = CalibrationBundle()
    if len(np.unique(y)) < 2:
        return bundle, {}
    z1 = np.array([r["tier1_logit"] for r in calib_rows], dtype=np.float64)
    bundle.set(f"{media}.tier1", fit_calibrator(z1, y, alpha=alpha, note="ablation"))
    t2 = [r for r in calib_rows if r["tier"] == "tier2"]
    y2 = np.array([r["y"] for r in t2])
    use = t2 if (len(t2) >= min_final and len(np.unique(y2)) == 2) else list(calib_rows)
    bundle.set(f"{media}.final", fit_calibrator(
        np.array([r["final_logit"] for r in use], dtype=np.float64),
        np.array([r["y"] for r in use]), alpha=alpha, note="ablation"))
    embs = [r["emb"] for r in calib_rows if r["emb"] is not None]
    ood = {media: OODModel().fit(np.stack(embs))} if embs else {}
    return bundle, ood


def summarize(rows: Sequence[Dict[str, Any]], n_boot: int = 500, seed: int = 0) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    y = np.array([r["y"] for r in rows])
    p = np.array([r["p"] for r in rows])
    v = [r["verdict"] for r in rows]
    lat = np.array([r["latency"] for r in rows])
    calls = np.array([r["calls"] for r in rows])
    out = {
        "n": int(len(y)),
        "auroc": MT.bootstrap_ci(MT.auc, y, p, n_boot=n_boot, seed=seed),
        "brier": MT.brier(y, p), "nll": MT.nll(y, p),
        "ece": MT.expected_calibration_error(y, p),
        "forced_accuracy": float(((p >= 0.5).astype(int) == y).mean()),
        "decisions": MT.selective_metrics(y, v),
        "aurc": MT.aurc(y, p), "coverage_at_5pct_risk": MT.coverage_at_risk(y, p, 0.05),
        "early_exit_share": float(np.mean([r["tier"] == "tier1_early_exit" for r in rows])),
        "mean_expert_calls": float(calls.mean()),
        "latency_median_s": float(np.median(lat)), "latency_p95_s": float(np.percentile(lat, 95)),
    }
    return out
