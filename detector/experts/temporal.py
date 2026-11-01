"""Temporal expert: video-level evidence from per-frame scores/embeddings.

Replaces the original "two fake frames = fake video" rule, which made long
real videos fail on a couple of false-positive frames and ignored *when*
the suspicious frames occur.

Two modes:
* statistical (default, no training): trimmed aggregation of per-frame
  log-odds + run-length / flicker features;
* learned (optional): a small 1-D conv head trained on cached frame
  embeddings (``python -m train.train_temporal``).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from detector.fusion import Evidence, skipped


def aggregate_frame_logits(logits: Sequence[float], top_fraction: float = 0.5) -> float:
    """Mean of the most suspicious ``top_fraction`` of frames. A deepfake is
    often only partly manipulated, so a plain mean under-reports it, while a
    single max is too fragile to one false-positive frame."""
    z = np.sort(np.asarray(logits, dtype=np.float64))[::-1]
    if len(z) == 0:
        return 0.0
    k = max(1, int(np.ceil(len(z) * top_fraction)))
    return float(z[:k].mean())


def temporally_aware_aggregate(
    logits: Sequence[float],
    times: Optional[Sequence[float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Temporally-aware frame aggregation with autocorrelation-adjusted effective sample size.

    Accounting for autocorrelation rho between consecutive frames prevents overestimating
    evidence weight and statistical significance when consecutive frames are near-identical.
    Also detects localized anomalies (e.g. face swaps confined to brief intervals).
    """
    z = np.asarray(logits, dtype=np.float64)
    n = len(z)
    if n == 0:
        return 0.0, {"n_frames": 0, "n_eff": 0.0, "autocorr": 0.0, "anomaly_detected": False}
    if n == 1:
        return float(z[0]), {"n_frames": 1, "n_eff": 1.0, "autocorr": 0.0, "anomaly_detected": False}

    mean_z = float(np.mean(z))
    var_z = float(np.var(z))

    # Autocorrelation lag 1
    if var_z > 1e-6 and n > 2:
        z_cent = z - mean_z
        rho1 = float(np.sum(z_cent[:-1] * z_cent[1:]) / ((n - 1) * var_z))
        rho1 = float(np.clip(rho1, -0.99, 0.99))
    else:
        rho1 = 0.0

    # Effective sample size: N_eff = N / (1 + 2 * max(0, rho1))
    n_eff = float(np.clip(n / (1.0 + 2.0 * max(0.0, rho1)), 1.0, float(n)))

    # Localized anomaly detection:
    # If a short sequence of frames has significantly higher fake logits than background
    top_k = max(1, int(np.ceil(n * 0.35)))
    sorted_z = np.sort(z)[::-1]
    top_mean = float(np.mean(sorted_z[:top_k]))
    z_max = float(sorted_z[0])

    is_anomaly = bool(z_max > 1.0 and (z_max - mean_z) > 1.2)
    alpha = 0.65 if is_anomaly else 0.45
    agg_logit = float(alpha * top_mean + (1.0 - alpha) * mean_z)

    metrics = {
        "n_frames": n,
        "n_eff": round(n_eff, 2),
        "autocorr_lag1": round(rho1, 3),
        "mean_logit": round(mean_z, 3),
        "max_logit": round(z_max, 3),
        "anomaly_detected": is_anomaly,
    }
    return agg_logit, metrics


def suspicious_segments(
    logits: Sequence[float], times: Sequence[float], thr: float = 0.0,
    min_len: int = 2,
) -> List[Tuple[float, float]]:
    """Time ranges where consecutive sampled frames score above ``thr``."""
    segs: List[Tuple[float, float]] = []
    start = None
    for i, (z, t) in enumerate(zip(logits, times)):
        if z > thr and start is None:
            start = i
        if (z <= thr or i == len(logits) - 1) and start is not None:
            end = i if z <= thr else i + 1
            if end - start >= min_len:
                segs.append((float(times[start]), float(times[end - 1])))
            start = None
    return segs


def temporal_features(logits, times) -> Dict[str, float]:
    z = np.asarray(logits, dtype=np.float64)
    if len(z) == 0:
        return {}
    run, best = 0, 0
    for v in z:
        run = run + 1 if v > 0 else 0
        best = max(best, run)
    return {
        "n_frames": int(len(z)),
        "mean_logit": float(z.mean()),
        "p90_logit": float(np.percentile(z, 90)),
        "max_logit": float(z.max()),
        "share_above_0": float((z > 0).mean()),
        "longest_run": int(best),
        "flicker": float(np.abs(np.diff(z)).mean()) if len(z) > 1 else 0.0,
    }


# ---- optional learned head ------------------------------------------------
def build_temporal_head(dim: int, hidden: int = 64):
    import torch
    from torch import nn

    class TemporalHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.conv = nn.Sequential(
                nn.Conv1d(dim, hidden, kernel_size=3, padding=1), nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.GELU(),
            )
            self.out = nn.Linear(2 * hidden, 1)

        def forward(self, x):  # x: (B, T, D)
            h = self.conv(self.norm(x).transpose(1, 2))  # (B, H, T)
            pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
            return self.out(pooled).squeeze(-1)

    return TemporalHead()


def load_temporal_head(path: str):
    if not os.path.exists(path):
        return None
    import torch

    blob = torch.load(path, map_location="cpu")
    head = build_temporal_head(blob["dim"], blob.get("hidden", 64))
    head.load_state_dict(blob["state"])
    head.eval()
    return head


def temporal_evidence(
    logits: Sequence[float],
    times: Sequence[float],
    embeddings: Optional[np.ndarray] = None,
    head=None,
    quality_factor: float = 1.0,
) -> Evidence:
    n = len(logits)
    if n < 3:
        return skipped("temporal", "fewer than 3 sampled frames")
    feats = temporal_features(logits, times)
    segs = suspicious_segments(logits, times)
    mode = "statistical"
    z, temp_meta = temporally_aware_aggregate(logits, times)
    if head is not None and embeddings is not None and len(embeddings) == n:
        import torch

        with torch.inference_mode():
            x = torch.tensor(np.asarray(embeddings)[None], dtype=torch.float32)
            z = float(head(x).item())
        mode = "learned_head"
    # Scale reliability by effective sample size rather than raw correlated frame count
    rel = min(1.0, float(temp_meta.get("n_eff", n)) / 8.0) * quality_factor
    return Evidence(
        name="temporal", logit=float(z), reliability=float(rel),
        reason="", heuristic=(mode == "statistical"),
        detail={"mode": mode, **feats, **temp_meta,
                "suspicious_segments": [(round(a, 2), round(b, 2)) for a, b in segs]},
    )

