"""Fit expert-error correlations on a CALIBRATION split only (leakage-safe).

Never pass the final test set here. Correlations feed GLS fusion
(``DEEPFAKE_DEP_FUSION=1``) and the VOI controller.

    python -m train.fit_fusion_correlations --data data/calib --media image \\
        --out artifacts/expert_corr.json

Method:
  1. Run each available expert on the calibration set.
  2. Convert probabilities to residual errors against binary labels.
  3. Estimate pairwise residual correlations with Ledoit-Wolf shrinkage.
  4. Write named correlations and learned relative precisions to the artifacts
     consumed by the inference pipeline.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np

from train.data import scan_folder

log = logging.getLogger("deepfake.train.corr")


def _nearest_correlation(matrix: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix to a positive-definite correlation matrix."""
    matrix = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    matrix = (vectors * np.maximum(values, 1e-6)) @ vectors.T
    scale = np.sqrt(np.maximum(np.diag(matrix), 1e-12))
    matrix = matrix / np.outer(scale, scale)
    np.fill_diagonal(matrix, 1.0)
    matrix = np.clip(matrix, -0.99, 0.99)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def fit_correlations(
    score_matrix: Dict[str, np.ndarray],
    y: np.ndarray,
    shrinkage: float = 0.0,
) -> Dict:
    """Fit residual covariance/correlations and inverse-error-variance weights.

    ``score_matrix`` maps expert names to aligned p_fake arrays (NaN means the
    expert did not run). Ledoit-Wolf is fit pairwise on jointly observed
    residuals so missing experts do not shift rows or get imputed as correct.
    ``shrinkage`` is an optional additional shrinkage toward independence.
    """
    from sklearn.covariance import LedoitWolf

    labels = np.asarray(y, dtype=np.float64).reshape(-1)
    if not np.isfinite(labels).all() or not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("y must contain only finite binary labels")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be between 0 and 1")
    names = sorted(score_matrix.keys())
    residuals: Dict[str, np.ndarray] = {}
    for n in names:
        scores = np.asarray(score_matrix[n], dtype=np.float64).reshape(-1)
        if len(scores) != len(labels):
            raise ValueError(f"{n} has {len(scores)} scores for {len(labels)} labels")
        residuals[n] = np.clip(scores, 0.0, 1.0) - labels

    n = len(names)
    corr = np.eye(n)
    pair_counts = np.zeros((n, n), dtype=np.int64)
    variances: Dict[str, float] = {}
    shrunk_variance_samples: Dict[str, List[float]] = {name: [] for name in names}
    shrinkages = []
    for i, name in enumerate(names):
        values = residuals[name]
        valid = np.isfinite(values)
        pair_counts[i, i] = int(valid.sum())
        variances[name] = (float(np.var(values[valid], ddof=1))
                           if valid.sum() > 1 else float("nan"))
    for i in range(n):
        for j in range(i + 1, n):
            valid = np.isfinite(residuals[names[i]]) & np.isfinite(residuals[names[j]])
            count = int(valid.sum())
            pair_counts[i, j] = pair_counts[j, i] = count
            if count < 5:
                r = 0.0
            else:
                pair = np.column_stack((residuals[names[i]][valid],
                                        residuals[names[j]][valid]))
                estimator = LedoitWolf().fit(pair)
                covariance = estimator.covariance_
                shrunk_variance_samples[names[i]].append(float(covariance[0, 0]))
                shrunk_variance_samples[names[j]].append(float(covariance[1, 1]))
                denom = np.sqrt(max(covariance[0, 0], 0.0)
                                * max(covariance[1, 1], 0.0))
                r = float(covariance[0, 1] / denom) if denom > 1e-12 else 0.0
                shrinkages.append(float(estimator.shrinkage_))
            r = (1.0 - shrinkage) * r
            corr[i, j] = corr[j, i] = r

    corr = _nearest_correlation(corr)
    shrunk_variances = {
        name: (float(np.mean(shrunk_variance_samples[name]))
               if shrunk_variance_samples[name] else variances[name])
        for name in names
    }
    valid_variances = [v for v in shrunk_variances.values()
                       if np.isfinite(v) and v > 1e-12]
    reference_variance = min(valid_variances) if valid_variances else 1.0
    precisions = {
        name: float(np.clip(reference_variance / variance, 0.02, 1.0))
        if np.isfinite(variance) and variance > 1e-12 else 0.02
        for name, variance in shrunk_variances.items()
    }
    return {
        "experts": names,
        "correlation": corr.tolist(),
        "correlation_by_expert": {
            name: {other: float(corr[i, j]) for j, other in enumerate(names)}
            for i, name in enumerate(names)
        },
        "precision_by_expert": precisions,
        "residual_variance_by_expert": variances,
        "shrunk_residual_variance_by_expert": shrunk_variances,
        "pair_sample_counts": pair_counts.tolist(),
        "shrinkage": shrinkage,
        "ledoit_wolf_shrinkage_mean": (float(np.mean(shrinkages))
                                      if shrinkages else None),
        "n_samples": int(len(labels)),
        "n_positive": int(labels.sum()),
        "method": "pairwise_ledoit_wolf_residual_covariance",
        "leakage_note": "Fitted on the provided split only. Do not use test labels.",
    }


def collect_scores_from_pipeline(items, media: str, max_items: Optional[int] = None):
    """Run the live pipeline and collect per-expert p_fake where available."""
    from detector.pipeline import get_pipeline, reset_pipeline
    reset_pipeline()
    pipe = get_pipeline()
    if max_items:
        items = items[:max_items]
    rows: List[Dict[str, float]] = []
    ys: List[int] = []
    for k, it in enumerate(items):
        try:
            res = pipe.analyze(
                it.path, media_type=it.media_type or media,
                explain=False, stability=False, force_full=True,
            )
        except Exception as e:
            log.warning("skip %s: %s", it.path, e)
            continue
        ys.append(it.label)
        # map evidence list
        row: Dict[str, float] = {}
        for ev in res.evidence or []:
            name = ev.get("name") or "unknown"
            pf = ev.get("p_fake_raw")
            if pf is None and ev.get("logit") is not None:
                from detector.calibration import sigmoid
                pf = float(sigmoid(ev["logit"]))
            if pf is None:
                continue
            row[name] = float(pf)
        rows.append(row)
        if (k + 1) % 20 == 0:
            log.info("scored %d/%d", k + 1, len(items))

    # align lengths
    n = len(ys)
    matrix = {}
    names = sorted({name for row in rows for name in row})
    for name in names:
        arr = np.asarray([row.get(name, float("nan")) for row in rows], dtype=np.float64)
        # keep expert only if enough finite scores
        if np.isfinite(arr).sum() >= max(5, n // 10):
            matrix[name] = arr
    return matrix, np.asarray(ys, dtype=np.int64)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, nargs="+",
                    help="CALIBRATION folders only (real/ and fake/)")
    ap.add_argument("--media", default="image", choices=["image", "video", "audio"])
    ap.add_argument("--out", default="artifacts/expert_corr.json")
    ap.add_argument("--shrinkage", type=float, default=0.0,
                    help="additional correlation shrinkage toward zero (0-1); "
                         "Ledoit-Wolf shrinkage is always applied")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    items = []
    for d in a.data:
        items.extend(scan_folder(d, media=a.media, max_items=a.max_items, seed=a.seed))
    if not items:
        raise SystemExit("No calib items found")
    n_real = sum(1 for it in items if it.label == 0)
    n_fake = sum(1 for it in items if it.label == 1)
    if n_real == 0 or n_fake == 0:
        raise SystemExit("Calib split must contain both real and fake")

    log.info("Fitting correlations on %d items (%d real, %d fake) — NOT test",
             len(items), n_real, n_fake)
    matrix, y = collect_scores_from_pipeline(items, a.media, a.max_items)
    if not matrix:
        raise SystemExit("No expert scores were produced; refusing to write fusion artifacts")
    report = fit_correlations(matrix, y, shrinkage=a.shrinkage)
    if len(matrix) < 2:
        log.warning("Only one expert produced scores; learned cross-expert "
                    "correlations are unavailable")
        report["warning"] = "only one expert produced scores"

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    # Write the exact artifact names and schema loaded by Pipeline.reload().
    art = os.environ.get("DEEPFAKE_ARTIFACT_DIR", "artifacts")
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, "expert_correlation.json"), "w", encoding="utf-8") as f:
        json.dump(report.get("correlation_by_expert", {}), f, indent=2)
    with open(os.path.join(art, "fusion_weights.json"), "w", encoding="utf-8") as f:
        json.dump(report.get("precision_by_expert", {}), f, indent=2)
    print(json.dumps({
        "experts": report["experts"],
        "n_samples": report["n_samples"],
        "out": a.out,
        "artifacts": art,
    }, indent=2))
    return report


if __name__ == "__main__":
    main()
