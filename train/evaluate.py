"""Cross-dataset evaluation of the deployed cascade.

    python -m train.evaluate --data data/ffpp --data data/celebdf --media image \
        --out reports/eval.json

Reports, per dataset (so cross-dataset drops are visible):
  * raw backbone: AUC, EER, TPR@1%FPR (threshold-free)
  * full cascade: AUC of the fused score, ECE, abstain rate, accuracy on the
    answered cases, false-alarm / miss rates, early-exit share, mean latency

Always evaluate on data the detector was NOT trained or calibrated on.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

from detector import metrics as MT
from detector.calibration import sigmoid
from detector.pipeline import Pipeline
from train.data import Item, scan_folder

log = logging.getLogger("deepfake.train")


def evaluate_items(pipe: Pipeline, items: List[Item], media: str, mode: str) -> Dict:
    ys, raw, p_final, verdicts, early, lat = [], [], [], [], [], []
    for it in items:
        t = time.time()
        try:
            r = pipe.analyze(it.path, media_type=media, mode=mode, explain=False, stability=False)
        except Exception as e:
            log.warning("skip %s: %s", it.path, e)
            continue
        lat.append(time.time() - t)
        ys.append(it.label)
        raw.append(r.raw.get("tier1_logit", 0.0))
        p_final.append(r.p_fake)
        verdicts.append(r.verdict)
        early.append(r.tier == "tier1_early_exit")
    if not ys:
        return {"n": 0}
    out = {
        "n": len(ys),
        "backbone": {"auc": MT.auc(ys, raw), "eer": MT.eer(ys, raw),
                     "tpr_at_1pct_fpr": MT.tpr_at_fpr(ys, raw, 0.01)},
        "cascade": {"auc": MT.auc(ys, p_final), "eer": MT.eer(ys, p_final),
                    "tpr_at_1pct_fpr": MT.tpr_at_fpr(ys, p_final, 0.01),
                    "ece": MT.expected_calibration_error(ys, p_final),
                    "brier": MT.brier(ys, p_final)},
        "decisions": MT.selective_metrics(ys, verdicts),
        "early_exit_share": float(np.mean(early)),
        "mean_latency_s": float(np.mean(lat)),
    }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, action="append")
    ap.add_argument("--media", required=True, choices=["image", "video", "audio"])
    ap.add_argument("--mode", default="balanced", choices=["strict", "balanced", "sensitive"])
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    pipe = Pipeline()
    report = {"media": a.media, "mode": a.mode, "datasets": {}}
    for d in a.data:
        items = scan_folder(d, media=a.media, max_items=a.max_items)
        report["datasets"][items[0].dataset if items else d] = evaluate_items(pipe, items, a.media, a.mode)
    text = json.dumps(report, indent=2)
    print(text)
    if a.out:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
