"""Adaptive controller vs fixed cascade vs baseline policies.

    python -m train.evaluate_adaptive --calib data/calib --test data/celebdf \
        --media image --budgets 0.2 0.5 1.0 --risk-target 0.05 --out adaptive.json

Compares, on the SAME test items and each calibrated as deployed:
  fixed_cascade   the existing pipeline (early exit, then every Tier-2 test)
  voi             value-of-information controller (the proposal)
  fixed|random|quality   baseline selection policies, same budget

For each budget it reports risk/AUROC with bootstrap CIs, abstention, expert
calls and latency, so you can read "lower cost at comparable risk" or "lower
risk at comparable cost" straight off the table - or find that the
controller does NOT help, which is a valid result. No improvement is
assumed anywhere in this script.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

from detector import metrics as MT
from train._egafr import fit_for_config, make_pipe, score_items, summarize
from train.data import scan_folder

import numpy as np

POLICIES = ("voi", "fixed", "random", "quality")


def run(calib_items, test_items, media, mode, alpha, risk_target, budgets, n_boot=500):
    results = {}

    def evaluate(name, **flags):
        pipe = make_pipe(media, **flags)
        cal_rows = score_items(pipe, calib_items, media, mode)
        bundle, ood = fit_for_config(cal_rows, media, alpha)
        pipe = make_pipe(media, calibration=bundle, ood=ood, **flags)
        rows = score_items(pipe, test_items, media, mode)
        results[name] = {"flags": flags, **summarize(rows, n_boot)}
        results[name]["_rows"] = [{k: r[k] for k in ("y", "p", "verdict", "calls", "latency")}
                                  for r in rows]
        return rows

    evaluate("fixed_cascade", controller=False)
    for b in budgets:
        for pol in POLICIES:
            evaluate(f"{pol}@{b}", controller=True, controller_policy=pol,
                     risk_target=risk_target, budget=b)
    base = results["fixed_cascade"]["_rows"]
    yb = np.array([r["y"] for r in base])
    for name, r in results.items():
        if name == "fixed_cascade" or not r.get("n") or len(r["_rows"]) != len(base):
            continue
        pa = np.array([x["p"] for x in r["_rows"]])
        pb = np.array([x["p"] for x in base])
        r["vs_fixed_cascade"] = {
            "auroc_diff": MT.paired_bootstrap_diff(MT.auc, yb, pa, pb, n_boot=n_boot),
            "expert_calls_diff": r["mean_expert_calls"] - results["fixed_cascade"]["mean_expert_calls"],
            "latency_median_diff_s": r["latency_median_s"] - results["fixed_cascade"]["latency_median_s"]}
    for r in results.values():
        r.pop("_rows", None)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", required=True, action="append")
    ap.add_argument("--test", required=True, action="append")
    ap.add_argument("--media", default="image", choices=["image", "video"])
    ap.add_argument("--mode", default="balanced", choices=["strict", "balanced", "sensitive"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--risk-target", type=float, default=0.05)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.2, 0.5, 1.0])
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    cal = sum([scan_folder(d, media=a.media, max_items=a.max_items) for d in a.calib], [])
    tst = sum([scan_folder(d, media=a.media, max_items=a.max_items) for d in a.test], [])
    if {i.path for i in cal} & {i.path for i in tst}:
        raise SystemExit("calibration and test items overlap")
    res = run(cal, tst, a.media, a.mode, a.alpha, a.risk_target, a.budgets, a.boot)
    text = json.dumps(res, indent=2, default=str)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w").write(text)
    return res


if __name__ == "__main__":
    main()
