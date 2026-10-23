"""End-to-end held-out evaluation protocol (P0).

Enforces a leakage-safe workflow and writes a single report you can publish:

  1. Validate train / calib / test trees (or manifests from prepare_datasets)
  2. Refuse any group overlap between splits
  3. Train heads on **train only**
  4. Fit calibration + expert correlations on **calib only**
  5. Evaluate metrics on **test only** (ideally a different generator)
  6. Optional ablation + robustness on test

    python -m train.run_real_evaluation \\
        --train data/ffpp --calib data/calib --test data/celebdf \\
        --encoder facebook/dinov2-base --media image --out artifacts/real_eval

This still requires *you* to supply real media. The script makes the protocol
correct and the report complete — it cannot invent deepfake datasets.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Set

from train.data import scan_folder

log = logging.getLogger("deepfake.real_eval")


def _groups(root: str, media: str) -> Set[str]:
    return {it.group for it in scan_folder(root, media=media)}


def assert_disjoint(train: List[str], calib: List[str], test: List[str], media: str):
    gt = set()
    for roots, name in ((train, "train"), (calib, "calib"), (test, "test")):
        g = set()
        for r in roots:
            g |= _groups(r, media)
        overlap = gt & g
        if overlap:
            raise SystemExit(
                f"Leakage: {name} shares {len(overlap)} groups with earlier splits "
                f"(e.g. {sorted(list(overlap))[:3]}). "
                "Use scripts.prepare_datasets --split to build clean splits."
            )
        gt |= g
    # pairwise calib/test already covered by cumulative check order train→calib→test
    log.info("Split group-disjointness OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, nargs="+")
    ap.add_argument("--calib", required=True, nargs="+")
    ap.add_argument("--test", required=True, nargs="+")
    ap.add_argument("--encoder", default="facebook/dinov2-base")
    ap.add_argument("--media", default="image", choices=["image", "video", "audio"])
    ap.add_argument("--out", default="artifacts/real_eval")
    ap.add_argument("--skip-train", action="store_true",
                    help="use existing artifacts; only calib+eval")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--finetune", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    os.makedirs(a.out, exist_ok=True)
    os.environ["DEEPFAKE_ARTIFACT_DIR"] = a.out
    report: Dict[str, Any] = {
        "train": a.train, "calib": a.calib, "test": a.test,
        "encoder": a.encoder, "media": a.media, "steps": {},
        "disclaimer": (
            "Metrics below are only meaningful if --test is real media from "
            "generators not used in --train/--calib. Synthetic unit tests are "
            "not a substitute."
        ),
    }

    # 1. Disjointness
    t0 = time.time()
    try:
        assert_disjoint(a.train, a.calib, a.test, a.media)
        report["steps"]["disjointness"] = {"ok": True, "seconds": round(time.time() - t0, 2)}
    except SystemExit as e:
        report["steps"]["disjointness"] = {"ok": False, "error": str(e)}
        with open(os.path.join(a.out, "real_eval_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        raise

    mx = ["--max-items", str(a.max_items)] if a.max_items else []

    # 2. Train on train only
    if not a.skip_train:
        t0 = time.time()
        try:
            if a.media in ("image", "video"):
                from train import finetune_all
                mods = ["image"] if a.media == "image" else ["image", "video"]
                args = [
                    "--train", *a.train, "--encoder", a.encoder,
                    "--out", a.out, "--modalities", *mods, *mx,
                ]
                if a.finetune:
                    args.append("--finetune-image")
                finetune_all.main(args)
            if a.media == "audio":
                from train import train_audio
                train_audio.main(["--train", *a.train, "--out", a.out, *mx])
            report["steps"]["train"] = {"ok": True, "seconds": round(time.time() - t0, 2)}
        except Exception as e:
            log.exception("train failed")
            report["steps"]["train"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    else:
        report["steps"]["train"] = {"ok": True, "skipped": True}

    # 3. Calibration on calib only
    t0 = time.time()
    try:
        from train import fit_calibration
        fit_calibration.main(
            sum([["--data", d] for d in a.calib], [])
            + ["--media", a.media, *mx]
        )
        report["steps"]["calibration"] = {"ok": True, "seconds": round(time.time() - t0, 2)}
    except Exception as e:
        log.exception("calibration failed")
        report["steps"]["calibration"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # 4. Fusion correlations on calib only
    t0 = time.time()
    try:
        from train import fit_fusion_correlations
        fit_fusion_correlations.main(
            ["--data", *a.calib, "--media", a.media,
             "--out", os.path.join(a.out, "expert_corr.json"), *mx]
        )
        report["steps"]["fusion_correlations"] = {
            "ok": True, "seconds": round(time.time() - t0, 2),
            "leakage_safe": True,
        }
    except Exception as e:
        log.exception("correlation fit failed")
        report["steps"]["fusion_correlations"] = {
            "ok": False, "error": f"{type(e).__name__}: {e}",
        }

    # 5. Evaluate on test only
    t0 = time.time()
    try:
        from train import evaluate
        eval_out = os.path.join(a.out, "test_metrics.json")
        evaluate.main(
            sum([["--data", d] for d in a.test], [])
            + ["--media", a.media, "--out", eval_out, *mx]
        )
        metrics = {}
        if os.path.isfile(eval_out):
            with open(eval_out) as f:
                metrics = json.load(f)
        report["steps"]["evaluate_test"] = {
            "ok": True, "seconds": round(time.time() - t0, 2),
            "metrics": metrics,
        }
    except Exception as e:
        log.exception("evaluate failed")
        report["steps"]["evaluate_test"] = {
            "ok": False, "error": f"{type(e).__name__}: {e}",
        }

    # 6. Optional ablation on test
    if not a.skip_ablation:
        t0 = time.time()
        try:
            from train import evaluate_ablation
            ab_out = os.path.join(a.out, "ablation.json")
            evaluate_ablation.main(
                sum([["--calib", d] for d in a.calib], [])
                + sum([["--test", d] for d in a.test], [])
                + ["--media", a.media, "--out", ab_out, *mx]
            )
            report["steps"]["ablation"] = {
                "ok": True, "seconds": round(time.time() - t0, 2), "out": ab_out,
            }
        except Exception as e:
            log.exception("ablation failed")
            report["steps"]["ablation"] = {
                "ok": False, "error": f"{type(e).__name__}: {e}",
            }

    # Expert audit snapshot
    try:
        from scripts.expert_audit import run_audit, to_markdown
        audit = run_audit(artifact_dir=a.out, try_load=False)
        with open(os.path.join(a.out, "expert_audit.json"), "w") as f:
            json.dump(audit, f, indent=2, default=str)
        with open(os.path.join(a.out, "expert_audit.md"), "w") as f:
            f.write(to_markdown(audit))
        report["steps"]["expert_audit"] = {"ok": True, "summary": audit.get("summary")}
    except Exception as e:
        report["steps"]["expert_audit"] = {"ok": False, "error": str(e)}

    path = os.path.join(a.out, "real_eval_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    failed = [k for k, v in report["steps"].items() if not v.get("ok")]
    print(json.dumps({
        "out": path,
        "failed": failed,
        "steps_ok": {k: v.get("ok") for k, v in report["steps"].items()},
    }, indent=2))
    return report


if __name__ == "__main__":
    main()
