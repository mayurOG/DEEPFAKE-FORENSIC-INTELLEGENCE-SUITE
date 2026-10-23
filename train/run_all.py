"""Train, calibrate and evaluate everything in one command.

    python -m train.run_all \
        --train data/ffpp data/dfdc_sample \
        --calib data/calib \
        --test  data/celebdf data/wilddeepfake \
        --encoder facebook/dinov2-base \
        --out artifacts

Folder layout for every dataset: ``<dir>/real/...`` and ``<dir>/fake/...``.
Use THREE disjoint sets: --train (fits probe + spectral head), --calib
(fits calibration and the out-of-domain model; never seen by training),
--test (reported numbers; ideally a different dataset/generator, because
cross-dataset results are the honest ones).

Steps (each skippable): features -> probe [-> LayerNorm tuning] -> spectral
head -> calibration -> evaluation. Writes ``<out>/run_summary.json``.
Needs a GPU for a large encoder; the spectral head and probe are cheap.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List

log = logging.getLogger("deepfake.train")


def _step(name: str, summary: Dict, fn):
    t = time.time()
    log.info("== %s ==", name)
    try:
        out = fn()
        summary["steps"][name] = {"ok": True, "seconds": round(time.time() - t, 1),
                                  **({"result": out} if isinstance(out, dict) else {})}
    except SystemExit as e:  # argparse / "need both classes" exits
        summary["steps"][name] = {"ok": False, "error": f"exit: {e}"}
    except Exception as e:  # keep going: later steps may still be useful
        log.exception("%s failed", name)
        summary["steps"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, nargs="+")
    ap.add_argument("--calib", required=True, nargs="+")
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--encoder", required=True, help="hub id or local dir")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--media", default="image", choices=["image", "video", "audio"])
    ap.add_argument("--conditional", action="store_true",
                    help="also fit EGAFR conditional calibration + expert reliabilities")
    ap.add_argument("--finetune", action="store_true",
                    help="also run LayerNorm-only tuning (GPU recommended)")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="encoder images per batch (use 1-2 on GPUs with <=4 GB VRAM)")
    ap.add_argument("--no-faces", action="store_true")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--mode", default="balanced", choices=["strict", "balanced", "sensitive"])
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    os.makedirs(a.out, exist_ok=True)
    # Everything downstream (Pipeline, backends) reads these.
    os.environ["DEEPFAKE_ARTIFACT_DIR"] = a.out
    summary: Dict = {"encoder": a.encoder, "train": a.train, "calib": a.calib,
                     "test": a.test, "steps": {}}
    overlap = set(map(os.path.abspath, a.train)) & set(map(os.path.abspath, a.calib + a.test))
    if overlap:
        raise SystemExit(f"train and calib/test sets must be disjoint, got overlap: {sorted(overlap)}")

    from train import features, fit_calibration, evaluate, train_probe, train_spectral
    mx = ["--max-items", str(a.max_items)] if a.max_items else []
    nf = ["--no-faces"] if a.no_faces else []

    feats = os.path.join(a.out, "train_features.npz")

    def do_features():
        # one file per --train dir would lose provenance; merge them
        import numpy as np
        parts = []
        for k, d in enumerate(a.train):
            p = os.path.join(a.out, f"_feat_{k}.npz")
            features.main(["--data", d, "--encoder", a.encoder, "--out", p,
                           "--batch-size", str(a.batch_size)] + nf + mx)
            with np.load(p, allow_pickle=True) as part:
                parts.append({key: part[key] for key in part.files})
        merged = {}
        for k in parts[0]:
            if parts[0][k].ndim >= 1:
                merged[k] = np.concatenate([p[k] for p in parts])
            else:  # scalar metadata (e.g. encoder id): keep the first
                merged[k] = parts[0][k]
        np.savez(feats, **merged)
        for k in range(len(a.train)):
            os.remove(os.path.join(a.out, f"_feat_{k}.npz"))
        return {"rows": int(len(merged["y"]))}

    _step("features", summary, do_features)
    _step("probe", summary, lambda: train_probe.main(
        ["--train", feats, "--out", os.path.join(a.out, "probe.npz")]) or {})
    if a.finetune:
        from train import finetune_ln
        _step("finetune_ln", summary, lambda: finetune_ln.main(
            ["--data", a.train[0], "--encoder", a.encoder, "--out", a.out,
             "--batch-size", str(a.batch_size)] + nf + mx) or {})
    spec_args = sum([["--data", d] for d in a.train], [])
    _step("spectral_head", summary, lambda: {"val_auc": train_spectral.main(
        spec_args + ["--out", a.out] + mx)["val_auc"]})

    if a.media == "audio":
        from train import train_audio
        def do_audio():
            args = ["--train", *a.train, "--out", a.out]
            if a.max_items:
                args += ["--max-items", str(a.max_items)]
            return train_audio.main(args) or {}
        _step("audio_probe", summary, do_audio)

    os.environ["EMBED_MODEL_ID"] = a.encoder  # probe becomes the Tier-1 backbone
    from detector import models as M
    from detector.pipeline import reset_pipeline
    reset_pipeline()
    if os.environ.get("DEEPFAKE_RUN_ALL_KEEP_BACKENDS") != "1":
        M.reset_backends()

    cal_args = (sum([["--data", d] for d in a.calib], []) + ["--media", a.media] + mx
                + (["--conditional"] if a.conditional else []))
    _step("calibration", summary, lambda: fit_calibration.main(cal_args) or {})
    reset_pipeline()

    if a.test:
        ev_args = sum([["--data", d] for d in a.test], []) + [
            "--media", a.media, "--mode", a.mode,
            "--out", os.path.join(a.out, "evaluation.json")] + mx
        _step("evaluation", summary, lambda: evaluate.main(ev_args) or {})
    with open(os.path.join(a.out, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    failed = [k for k, v in summary["steps"].items() if not v.get("ok")]
    print(json.dumps({"steps": {k: v.get("ok") for k, v in summary["steps"].items()},
                      "failed": failed, "summary": os.path.join(a.out, "run_summary.json")}))
    return summary


if __name__ == "__main__":
    main()
