"""Fit calibration, conformal prediction and the out-of-domain model.

Run this on a **held-out labelled set that was not used for training** (and
that resembles your deployment data). It runs the full cascade with early
exit disabled to collect raw scores, then writes:

    artifacts/calibration.json   Platt scaling + conformal quantiles +
                                 false-alarm operating points per stage
    artifacts/ood_<media>.npz    Mahalanobis out-of-domain model

    python -m train.fit_calibration --data data/holdout --media image --alpha 0.1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List, Optional

import numpy as np

from detector import metrics as MT
from detector.calibration import (
    CalibrationBundle, OODModel, fit_calibrator, sigmoid,
)
from detector.config import get_settings
from detector.pipeline import Pipeline
from train.data import Item, scan_folder

log = logging.getLogger("deepfake.train")


def collect_scores(pipe: Pipeline, items: List[Item], media: str):
    rows = []
    for k, it in enumerate(items):
        try:
            r = pipe.analyze(it.path, media_type=media, explain=False,
                             stability=False, force_full=True)
        except Exception as e:
            log.warning("skip %s: %s", it.path, e)
            continue
        rows.append({"y": it.label, "tier1": r.raw.get("tier1_logit"),
                     "final": r.raw.get("final_logit"),
                     "emb": r.raw.get("embedding"), "path": it.path,
                     "ctx": r.raw.get("ctx"), "expert": r.raw.get("expert_logits")})
        if (k + 1) % 25 == 0:
            log.info("scored %d/%d", k + 1, len(items))
    return rows


def fit_from_rows(rows, media: str, alpha: float, bundle: Optional[CalibrationBundle] = None,
                  out_dir: Optional[str] = None):
    """Returns (report dict, CalibrationBundle)."""
    y = np.array([r["y"] for r in rows])
    if len(np.unique(y)) < 2:
        raise ValueError("Need both real and fake examples to calibrate.")
    bundle = bundle or CalibrationBundle()
    report = {"media": media, "n": int(len(y)), "alpha": alpha}
    stages = ["final"] if media == "audio" else ["tier1", "final"]
    for stage in stages:
        z = np.array([r[stage] for r in rows], dtype=np.float64)
        cal = fit_calibrator(z, y, alpha=alpha, note=f"{media}.{stage}")
        bundle.set(f"{media}.{stage}", cal)
        p_raw, p_cal = sigmoid(z), cal.prob_fake(z)
        report[stage] = {
            "auc": MT.auc(y, z), "eer": MT.eer(y, z),
            "tpr_at_1pct_fpr": MT.tpr_at_fpr(y, z, 0.01),
            "ece_before": MT.expected_calibration_error(y, p_raw),
            "ece_after": MT.expected_calibration_error(y, p_cal),
            "a": cal.a, "b": cal.b,
            "q_real": None if cal.q_real is None else str(cal.q_real),
            "q_fake": None if cal.q_fake is None else str(cal.q_fake),
            "op_thresholds": cal.op_thresholds,
        }
    embs = [r["emb"] for r in rows if r["emb"] is not None]
    ood_saved = False
    if embs:
        ood = OODModel().fit(np.stack(embs))
        if ood.fitted and out_dir:
            os.makedirs(out_dir, exist_ok=True)
            ood.save(os.path.join(out_dir, f"ood_{media}.npz"))
            ood_saved = True
    report["ood_fitted"] = ood_saved
    return report, bundle


def fit_conditional_from_rows(rows, media: str, alpha: float, out_dir: str,
                              min_group_n: int = 60) -> dict:
    """EGAFR: context-conditional conformal sets, per-expert reliability
    models and the empirical expert-error correlation, from the same
    calibration rows. In-sample group coverage is reported only as a
    sanity check; evaluate on held-out data with train.evaluate_ablation."""
    from detector.conditional_reliability import (ReliabilityModel, fit_conditional,
                                                  group_coverage, save_conditional)
    from detector.evidence_graph import estimate_correlation
    rows = [r for r in rows if r.get("final") is not None]
    y = np.array([r["y"] for r in rows])
    z = np.array([r["final"] for r in rows], dtype=np.float64)
    ctxs = [r.get("ctx") or {} for r in rows]
    cc = fit_conditional(z, y, ctxs, alpha=alpha, min_group_n=min_group_n)
    names = sorted({k for r in rows for k in (r.get("expert") or {})})
    rel, Z = {}, np.full((len(rows), len(names)), np.nan)
    for j, n in enumerate(names):
        idx = [i for i, r in enumerate(rows) if n in (r.get("expert") or {})]
        for i in idx:
            Z[i, j] = rows[i]["expert"][n]
        rel[n] = ReliabilityModel().fit([rows[i]["expert"][n] for i in idx], y[idx],
                                        [ctxs[i] for i in idx])
    os.makedirs(out_dir, exist_ok=True)
    save_conditional(os.path.join(out_dir, f"conditional_{media}.json"), cc, rel)
    if names:
        corr = estimate_correlation(Z, names, y)
        with open(os.path.join(out_dir, "expert_correlation.json"), "w") as f:
            json.dump(corr, f, indent=2)
    return {"groups": {k: v["n"] for k, v in cc.groups.items()},
            "reliability_fitted": [n for n, m in rel.items() if m.fitted],
            "in_sample_group_coverage": group_coverage(cc, z, y, ctxs),
            "experts": names}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, action="append",
                    help="folder with real/ and fake/ (repeatable)")
    ap.add_argument("--media", required=True, choices=["image", "video", "audio"])
    ap.add_argument("--alpha", type=float, default=get_settings().alpha)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--conditional", action="store_true",
                    help="also fit EGAFR conditional calibration + expert reliabilities")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    s = get_settings()
    items: List[Item] = []
    for d in a.data:
        items += scan_folder(d, media=a.media, max_items=a.max_items)
    pipe = Pipeline()
    rows = collect_scores(pipe, items, a.media)
    path = os.path.join(s.artifact_dir, "calibration.json")
    bundle = CalibrationBundle.load(path)
    report, bundle = fit_from_rows(rows, a.media, a.alpha, bundle, s.artifact_dir)
    bundle.save(path)
    if a.conditional and a.media in ("image", "video"):
        report["conditional"] = fit_conditional_from_rows(rows, a.media, a.alpha, s.artifact_dir)
    with open(os.path.join(s.artifact_dir, f"calibration_report_{a.media}.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
