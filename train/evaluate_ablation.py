"""Component-contribution study (ablation) for the EGAFR modules.

    python -m train.evaluate_ablation --calib data/calib --test data/celebdf \
        --media image --out ablation.json

  A  fixed-weight weighted mean without EGAFR features
  B  fixed weights plus prior-correlation GLS
  C  A + adaptive test selection (controller)
  D  A + conditional calibration + learned reliabilities
  E  full system (controller + dependent fusion + conditional)
  F  full system with the controller disabled at inference
  G  full system with expert-dependency adjustment disabled (rho = 0)
  H  unit-weight averaging
  I  learned precision without dependency adjustment
  J  learned precision with learned-correlation GLS

Every configuration is calibrated as deployed (see train/_egafr.py), on the
calibration split only. The fusion controls compare fixed-weight averaging,
unit-weight averaging, fixed-prior GLS, learned precision without dependency
adjustment, and learned precision with learned-correlation GLS. Outputs
include per-configuration metrics, paired bootstrap differences against the
fixed-weight baseline and averaging / learned-precision controls, plus
per-context abstention and error.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List

import numpy as np

from detector import metrics as MT
from detector.conditional_reliability import ReliabilityModel, fit_conditional, group_key
from train._egafr import fit_for_config, make_pipe, score_items, summarize
from train.data import scan_folder

FULL = dict(controller=True, dependent_fusion=True, conditional=True)
CONFIGS: Dict[str, Dict[str, Any]] = {
    "A_existing": dict(controller=False, dependent_fusion=False, conditional=False,
                        fusion_weights_mode="fixed"),
    "B_dependent_fusion": dict(controller=False, dependent_fusion=True,
                               conditional=False, fusion_weights_mode="fixed",
                               correlation_mode="prior"),
    "C_controller": dict(controller=True, dependent_fusion=False, conditional=False,
                         fusion_weights_mode="fixed"),
    "D_conditional": dict(controller=False, dependent_fusion=False, conditional=True,
                          fusion_weights_mode="fixed"),
    "E_full": dict(FULL),
    "F_full_no_controller": dict(FULL, controller=False),
    "G_full_independent": dict(FULL, independent_experts=True),
    "H_plain_average": dict(controller=False, dependent_fusion=False, conditional=False,
                             fusion_weights_mode="unit"),
    "I_learned_precision_independent": dict(
        controller=False, dependent_fusion=False, conditional=False,
        fusion_weights_mode="artifact"),
    "J_learned_precision_correlation_gls": dict(
        controller=False, dependent_fusion=True, conditional=False,
        fusion_weights_mode="artifact", correlation_mode="artifact"),
}


def group_table(rows) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        if r["tier"] != "tier2" or not r.get("ctx"):
            continue
        g = out.setdefault(group_key(r["ctx"]), {"n": 0, "abstain": 0, "wrong": 0, "answered": 0})
        g["n"] += 1
        if r["verdict"] == "Inconclusive":
            g["abstain"] += 1
        else:
            g["answered"] += 1
            g["wrong"] += int((r["verdict"] == "Fake") != bool(r["y"]))
    return {k: {"n": v["n"], "abstain_rate": v["abstain"] / v["n"],
                "error_among_answered": (v["wrong"] / v["answered"]) if v["answered"] else None}
            for k, v in out.items()}


def run_config(name, flags, calib_items, test_items, media, mode, alpha):
    pipe = _configured_pipe(media, flags)
    models: Dict[str, Any] = {}
    if flags.get("conditional"):
        # pass 1: gather every expert's logit to learn per-expert reliability
        rows1 = score_items(pipe, calib_items, media, mode, force_full=True)
        y = np.array([r["y"] for r in rows1])
        for n in sorted({k for r in rows1 for k in (r.get("expert") or {})}):
            idx = [i for i, r in enumerate(rows1) if n in (r.get("expert") or {})]
            models[f"{media}:{n}"] = ReliabilityModel().fit(
                [rows1[i]["expert"][n] for i in idx], y[idx],
                [rows1[i].get("ctx") or {} for i in idx])
        pipe.reliability_models = models
    # pass 2 (or the only pass): the configuration's natural deployed path
    cal_rows = score_items(pipe, calib_items, media, mode)
    bundle, ood = fit_for_config(cal_rows, media, alpha)
    pipe2 = _configured_pipe(media, flags, calibration=bundle, ood=ood)
    pipe2.reliability_models = models
    if flags.get("conditional"):
        t2 = [r for r in cal_rows if r["tier"] == "tier2" and r.get("ctx")]
        yy = np.array([r["y"] for r in t2])
        if len(t2) >= 40 and len(np.unique(yy)) == 2:
            pipe2.conditional = {media: fit_conditional(
                [r["final_logit"] for r in t2], yy, [r["ctx"] for r in t2], alpha=alpha)}
    rows = score_items(pipe2, test_items, media, mode)
    return rows


def _configured_pipe(media, flags, calibration=None, ood=None):
    pipe_flags = dict(flags)
    weights_mode = pipe_flags.pop("fusion_weights_mode", "artifact")
    correlation_mode = pipe_flags.pop("correlation_mode", "artifact")
    pipe = make_pipe(media, calibration=calibration, ood=ood, **pipe_flags)
    if weights_mode == "fixed":
        pipe.fusion_weights = None
    elif weights_mode == "unit":
        from detector.fusion import DEFAULT_WEIGHTS
        pipe.fusion_weights = {name: 1.0 for name in DEFAULT_WEIGHTS}
    elif weights_mode != "artifact":
        raise ValueError(f"unknown fusion_weights_mode: {weights_mode}")
    if correlation_mode == "prior":
        pipe.expert_corr = None
    elif correlation_mode != "artifact":
        raise ValueError(f"unknown correlation_mode: {correlation_mode}")
    return pipe


def run(calib_items, test_items, media, mode, alpha, n_boot=500, configs=None):
    out: Dict[str, Any] = {}
    keep: Dict[str, List[Dict[str, Any]]] = {}
    for name, flags in (configs or CONFIGS).items():
        rows = run_config(name, flags, calib_items, test_items, media, mode, alpha)
        keep[name] = rows
        out[name] = {"flags": flags, **summarize(rows, n_boot),
                     "by_context_group": group_table(rows)}
    base_name = next(iter(keep))
    baseline_names = list(dict.fromkeys(
        [base_name, "H_plain_average", "I_learned_precision_independent"]))
    for name, rows in keep.items():
        for baseline_name in baseline_names:
            baseline = keep.get(baseline_name)
            if name == baseline_name or baseline is None:
                continue
            baseline_by_path = {r["path"]: r for r in baseline}
            paired = [(row, baseline_by_path[row["path"]]) for row in rows
                      if row["path"] in baseline_by_path]
            if not paired:
                continue
            compared, reference = zip(*paired)
            y = np.array([r["y"] for r in compared])
            pa = np.array([r["p"] for r in compared])
            pb = np.array([r["p"] for r in reference])
            out[name]["vs_" + baseline_name] = {
                "auroc": MT.paired_bootstrap_diff(MT.auc, y, pa, pb, n_boot=n_boot),
                "aurc_lower_is_better": MT.paired_bootstrap_diff(
                    MT.aurc, y, pa, pb, n_boot=n_boot),
                "brier_lower_is_better": MT.paired_bootstrap_diff(
                    lambda yy, pp: MT.brier(yy, pp), y, pa, pb, n_boot=n_boot),
                "mean_expert_calls_diff": float(np.mean([r["calls"] for r in compared])
                                                - np.mean([r["calls"] for r in reference])),
                "n_paired": int(len(paired))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", required=True, action="append")
    ap.add_argument("--test", required=True, action="append")
    ap.add_argument("--media", default="image", choices=["image", "video"])
    ap.add_argument("--mode", default="balanced", choices=["strict", "balanced", "sensitive"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--only", nargs="*", default=None, help="subset of config names")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    cal = sum([scan_folder(d, media=a.media, max_items=a.max_items) for d in a.calib], [])
    tst = sum([scan_folder(d, media=a.media, max_items=a.max_items) for d in a.test], [])
    if {i.path for i in cal} & {i.path for i in tst}:
        raise SystemExit("calibration and test items overlap")
    cfgs = {k: v for k, v in CONFIGS.items() if not a.only or k in a.only or k.split("_")[0] in a.only}
    res = run(cal, tst, a.media, a.mode, a.alpha, a.boot, cfgs)
    text = json.dumps(res, indent=2, default=str)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w").write(text)
    return res


if __name__ == "__main__":
    main()
