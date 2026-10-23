"""Leave-one-domain-out calibration and selective-risk benchmark.

Each ``--domain NAME=ROOT`` must contain ``real/`` and ``fake/`` folders.
For each fold, the named domain is evaluated while calibration is fitted on
all other domains. The currently configured detector checkpoint is held fixed;
this script does not retrain its neural weights.

    python -m train.evaluate_generator_shift \\
        --domain ffpp=data/ffpp --domain flux=data/flux \\
        --domain sd3=data/sd3 --domain social_reupload=data/social_reupload \\
        --media image --out artifacts/generator_shift.json

The report includes selective risk-versus-coverage curves and empirical
conformal coverage per held-out domain. These are benchmark results, not a
claim of formal coverage under arbitrary distribution shift.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from detector import metrics as MT
from detector.calibration import fit_calibrator
from detector.config import get_settings
from detector.pipeline import Pipeline
from train.data import scan_folder

log = logging.getLogger("deepfake.generator_shift")


def parse_domain(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("domains must use NAME=ROOT")
    name, root = value.split("=", 1)
    if not name.strip() or not root.strip():
        raise argparse.ArgumentTypeError("both domain name and root are required")
    return name.strip(), root.strip()


def leave_one_domain_out(
    rows: Sequence[Dict[str, Any]], domain_names: Sequence[str],
    alpha: float = 0.1, mode: str = "balanced",
) -> Dict[str, Any]:
    """Fit a score calibrator on other domains and evaluate each held-out one."""
    if len(set(domain_names)) < 2:
        raise ValueError("at least two domains are required for leave-one-domain-out evaluation")

    folds: Dict[str, Any] = {}
    for held_out in domain_names:
        calibration = [r for r in rows if r["domain"] != held_out]
        test = [r for r in rows if r["domain"] == held_out]
        if not test:
            raise ValueError(f"domain {held_out!r} has no successfully scored items")
        y_cal = np.asarray([r["y"] for r in calibration], dtype=np.int64)
        if len(np.unique(y_cal)) != 2:
            raise ValueError(f"calibration fold excluding {held_out!r} needs both labels")
        z_cal = np.asarray([r["logit"] for r in calibration], dtype=np.float64)
        calibrator = fit_calibrator(z_cal, y_cal, alpha=alpha,
                                    note=f"leave_one_domain_out:{held_out}")

        y = np.asarray([r["y"] for r in test], dtype=np.int64)
        z = np.asarray([r["logit"] for r in test], dtype=np.float64)
        p = calibrator.prob_fake(z)
        sets = [calibrator.label_set(float(pi), mode) for pi in p]
        verdicts = [calibrator.verdict(float(pi), mode) for pi in p]
        conf = np.maximum(p, 1.0 - p)
        curve = MT.risk_coverage_curve(y, p, confidence=conf)
        true_labels = np.where(y == 1, "Fake", "Real")
        set_covered = np.asarray([label in pred_set
                                  for label, pred_set in zip(true_labels, sets)])
        folds[held_out] = {
            "n": int(len(y)),
            "calibration_domains": sorted({r["domain"] for r in calibration}),
            "calibration_n": int(len(calibration)),
            "calibrated_auc": MT.auc(y, p),
            "brier": MT.brier(y, p),
            "ece": MT.expected_calibration_error(y, p),
            "aurc": MT.aurc(y, p, confidence=conf),
            "coverage_at_5pct_risk": MT.coverage_at_risk(
                y, p, 0.05, confidence=conf),
            "risk_coverage": {
                "coverage": curve["coverage"].tolist(),
                "risk": curve["risk"].tolist(),
            },
            "conformal": {
                "target_coverage": 1.0 - alpha,
                "empirical_coverage": float(set_covered.mean()),
                "mean_set_size": float(np.mean([len(s) for s in sets])),
                "abstention_rate": float(np.mean([len(s) != 1 for s in sets])),
                "calibrated": bool(calibrator.fitted),
            },
            "selective_decisions": MT.selective_metrics(y, verdicts),
        }
    return {
        "protocol": "leave_one_domain_out_calibration",
        "alpha": float(alpha),
        "mode": mode,
        "model_weights_retrained_per_fold": False,
        "folds": folds,
        "warning": (
            "The detector weights are held fixed across folds. To claim a "
            "leave-one-generator-out training result, ensure those weights "
            "were trained without every evaluated held-out generator."
        ),
    }


def collect_predictions(domains, media: str, max_items=None):
    pipe = Pipeline()
    rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for name, root in domains:
        items = scan_folder(root, media=media, max_items=max_items)
        if not items:
            raise ValueError(f"no {media} items found in domain {name!r}: {root}")
        counts[name] = len(items)
        for index, item in enumerate(items, 1):
            try:
                result = pipe.analyze(item.path, media_type=media, explain=False,
                                      stability=False, force_full=True)
            except Exception as exc:
                log.warning("skip %s: %s", item.path, exc)
                continue
            logit = result.raw.get("final_logit")
            if logit is None or not np.isfinite(float(logit)):
                log.warning("skip %s: missing/non-finite final logit", item.path)
                continue
            rows.append({"domain": name, "y": item.label, "logit": float(logit)})
            if index % 25 == 0:
                log.info("scored %s: %d/%d", name, index, len(items))
    if not rows:
        raise ValueError("no domain items produced usable final logits")
    return rows, counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True, action="append", type=parse_domain,
                    help="repeat as NAME=ROOT; root contains real/ and fake/")
    ap.add_argument("--media", default="image", choices=["image", "video", "audio"])
    ap.add_argument("--mode", default="balanced",
                    choices=["strict", "balanced", "sensitive"])
    ap.add_argument("--alpha", type=float, default=get_settings().alpha)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    names = [name for name, _ in a.domain]
    if len(set(names)) != len(names):
        raise SystemExit("domain names must be unique")
    if not 0.0 < a.alpha < 1.0:
        raise SystemExit("--alpha must be between 0 and 1")

    started = time.time()
    rows, counts = collect_predictions(a.domain, a.media, a.max_items)
    report = leave_one_domain_out(rows, names, a.alpha, a.mode)
    report.update({
        "media": a.media,
        "requested_domain_counts": counts,
        "scored_items": len(rows),
        "skipped_items": int(sum(counts.values()) - len(rows)),
        "seconds": round(time.time() - started, 3),
        "artifact_dir": get_settings().artifact_dir,
    })
    text = json.dumps(report, indent=2, allow_nan=True)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    return report


if __name__ == "__main__":
    main()
