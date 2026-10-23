"""Transformation-robustness evaluation (the protocol for Novelty 4).

    python -m train.evaluate_robustness --calib data/calib --test data/celebdf \
        --media image --out robustness.json

Part 1 - degradation matrix: the same test images pushed through each
controlled transformation (JPEG, downscale, blur, crop, noise, social
pipeline); AUROC, calibration error, abstention, and the share of verdicts
that change relative to the clean image.

Part 2 - failure prediction: does the per-image fragility score from the
robustness profile predict the system's *errors*? Reported as the AUROC of
fragility for "is this call wrong", next to the classifier's own uncertainty
(1 - |2p - 1|) and the evidence-sufficiency statistic, with paired bootstrap
differences. Then the risk-coverage effect of using (1 - fragility) as the
selector, and of the hard robustness gate.

Whether fragility adds anything beyond the classifier's own uncertainty is
the question; the script reports the answer either way.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from detector import metrics as MT
from detector.transformation_lab import IMAGE_TRANSFORMS, robustness_gate
from train._egafr import fit_for_config, make_pipe, score_items
from train.data import scan_folder


def score_images(pipe, pairs, mode, stability=False) -> List[Dict[str, Any]]:
    rows = []
    for name, img, y in pairs:
        try:
            r = pipe.analyze(img, media_type="image", filename=name, mode=mode,
                             explain=False, stability=stability)
        except Exception as e:
            logging.warning("skip %s: %s", name, e)
            continue
        rows.append({"name": name, "y": y, "p": r.p_fake, "verdict": r.verdict,
                     "fragility": (r.robustness or {}).get("fragility"),
                     "gate_ok": ((r.robustness or {}).get("gate") or {}).get("ok", True),
                     "sufficiency": (r.graph.get("fusion") or {}).get("sufficiency")})
    return rows


def _auc_err(y, p, signal):
    err = ((np.asarray(p) >= 0.5).astype(int) != np.asarray(y)).astype(int)
    return err, MT.auc(err, signal) if len(np.unique(err)) > 1 else float("nan")


def run(calib_items, test_items, media, mode, alpha, n_boot=500):
    pipe0 = make_pipe(media)
    bundle, ood = fit_for_config(score_items(pipe0, calib_items, media, mode), media, alpha)
    pipe = make_pipe(media, calibration=bundle, ood=ood)
    imgs = [(it.path, Image.open(it.path).convert("RGB"), it.label) for it in test_items]
    clean = score_images(pipe, imgs, mode, stability=True)
    y = np.array([r["y"] for r in clean])
    p = np.array([r["p"] for r in clean])
    out: Dict[str, Any] = {"n": len(clean), "matrix": {}}
    clean_v = [r["verdict"] for r in clean]
    out["matrix"]["clean"] = {"auroc": MT.auc(y, p), "ece": MT.expected_calibration_error(y, p),
                              **MT.selective_metrics(y, clean_v)}
    for t in IMAGE_TRANSFORMS:
        rows = score_images(pipe, [(n, t.fn(im), yy) for n, im, yy in imgs], mode)
        if len(rows) != len(clean):
            continue
        yt, pt = np.array([r["y"] for r in rows]), np.array([r["p"] for r in rows])
        vt = [r["verdict"] for r in rows]
        out["matrix"][t.name] = {
            "severity": t.severity, "auroc": MT.auc(yt, pt),
            "ece": MT.expected_calibration_error(yt, pt),
            "verdict_change_rate": float(np.mean([a != b for a, b in zip(clean_v, vt)])),
            **MT.selective_metrics(yt, vt)}
    # failure prediction
    frag = np.array([1.0 if r["fragility"] is None else r["fragility"] for r in clean])
    unc = 1.0 - np.abs(2 * p - 1)
    suff = np.array([0.0 if r["sufficiency"] is None else r["sufficiency"] for r in clean])
    err, a_frag = _auc_err(y, p, frag)
    _, a_unc = _auc_err(y, p, unc)
    _, a_suff = _auc_err(y, p, 1 - suff)
    _, a_comb = _auc_err(y, p, unc + frag)
    out["failure_prediction_auroc"] = {
        "n_errors": int(err.sum()), "fragility": a_frag, "classifier_uncertainty": a_unc,
        "insufficiency": a_suff, "uncertainty_plus_fragility": a_comb}
    if 0 < err.sum() < len(err):
        out["failure_prediction_auroc"]["fragility_ci"] = MT.bootstrap_ci(
            lambda e, s: MT.auc(e, s), err, frag, n_boot=n_boot)
        out["failure_prediction_auroc"]["combined_minus_uncertainty"] = MT.paired_bootstrap_diff(
            MT.auc, err, unc + frag, unc, n_boot=n_boot)
    conf_own = np.maximum(p, 1 - p)
    out["selective_risk"] = {
        "aurc_own_confidence": MT.aurc(y, p, conf_own),
        "aurc_robustness_selector": MT.aurc(y, p, 1 - frag),
        "aurc_combined": MT.aurc(y, p, conf_own * (1 - frag)),
        "coverage_at_5pct_risk_own": MT.coverage_at_risk(y, p, 0.05, conf_own),
        "coverage_at_5pct_risk_combined": MT.coverage_at_risk(y, p, 0.05, conf_own * (1 - frag))}
    gated = [("Inconclusive" if not r["gate_ok"] else r["verdict"]) for r in clean]
    out["robustness_gate"] = {"before": MT.selective_metrics(y, clean_v),
                              "after": MT.selective_metrics(y, gated)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", required=True, action="append")
    ap.add_argument("--test", required=True, action="append")
    ap.add_argument("--media", default="image", choices=["image"])
    ap.add_argument("--mode", default="balanced", choices=["strict", "balanced", "sensitive"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    cal = sum([scan_folder(d, media=a.media, max_items=a.max_items) for d in a.calib], [])
    tst = sum([scan_folder(d, media=a.media, max_items=a.max_items) for d in a.test], [])
    if {i.path for i in cal} & {i.path for i in tst}:
        raise SystemExit("calibration and test items overlap")
    res = run(cal, tst, a.media, a.mode, a.alpha, a.boot)
    text = json.dumps(res, indent=2, default=str)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w").write(text)
    return res


if __name__ == "__main__":
    main()
