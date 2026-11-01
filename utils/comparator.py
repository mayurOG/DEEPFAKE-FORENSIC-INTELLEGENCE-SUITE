"""Forensic comparison suite between Reference (A) and Suspect (B) media — v4.1.0.

Provides multi-dimensional similarity analysis and evidence visualizations:
  - Quantitative similarity: SSIM, PSNR, Histogram correlation, Embedding cosine similarity
  - 2D Difference Heatmap (|A - B| spatial modification overlay)
  - 2D Side-by-side ELA & Tamper Noise residuals
  - 3D Spatial Difference Elevation surface (Plotly 3D Surface)
  - 3D Differential FFT Power Spectrum (Plotly 3D Surface)
  - Forensic Profile Radar Comparison (Multi-axis polar chart)
  - Side-by-Side Evidence Logit Comparison (Grouped Bar chart)
  - Forensic Similarity Gauge
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageChops, ImageEnhance


def compute_ssim(im1: np.ndarray, im2: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) between two images."""
    if im1.ndim == 3:
        g1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        g1 = im1.astype(np.float32)
    if im2.ndim == 3:
        g2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        g2 = im2.astype(np.float32)

    if g1.shape != g2.shape:
        g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))

    C1 = (0.01 * 255.0) ** 2
    C2 = (0.03 * 255.0) ** 2
    k = (11, 11)
    sigma = 1.5

    mu1 = cv2.GaussianBlur(g1, k, sigma)
    mu2 = cv2.GaussianBlur(g2, k, sigma)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(g1 ** 2, k, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 ** 2, k, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, k, sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12
    )
    return float(np.clip(np.mean(ssim_map), 0.0, 1.0))


def compute_psnr(im1: np.ndarray, im2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels."""
    if im1.shape != im2.shape:
        im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))
    mse = np.mean((im1.astype(np.float32) - im2.astype(np.float32)) ** 2)
    if mse <= 1e-10:
        return 100.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def compute_hist_correlation(im1: np.ndarray, im2: np.ndarray) -> float:
    """Compute color histogram correlation across RGB channels."""
    if im1.shape != im2.shape:
        im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))
    corrs = []
    for c in range(min(3, im1.shape[2])):
        h1 = cv2.calcHist([im1], [c], None, [64], [0, 256])
        h2 = cv2.calcHist([im2], [c], None, [64], [0, 256])
        cv2.normalize(h1, h1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(h2, h2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        r = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        corrs.append(max(0.0, float(r)))
    return float(np.mean(corrs)) if corrs else 0.0


def compute_image_similarity(
    im_a: Image.Image,
    im_b: Image.Image,
    emb_a: Optional[np.ndarray] = None,
    emb_b: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculate multi-modal similarity scores between Reference A and Suspect B."""
    arr_a = np.asarray(im_a.convert("RGB"), dtype=np.uint8)
    arr_b = np.asarray(im_b.convert("RGB"), dtype=np.uint8)

    ssim_val = compute_ssim(arr_a, arr_b)
    psnr_val = compute_psnr(arr_a, arr_b)
    hist_val = compute_hist_correlation(arr_a, arr_b)

    # Embedding cosine similarity
    cos_sim: Optional[float] = None
    if emb_a is not None and emb_b is not None:
        ea = np.asarray(emb_a, dtype=np.float32).reshape(-1)
        eb = np.asarray(emb_b, dtype=np.float32).reshape(-1)
        if len(ea) == len(eb) and np.linalg.norm(ea) > 0 and np.linalg.norm(eb) > 0:
            cos = float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))
            cos_sim = float(np.clip(cos, -1.0, 1.0))

    # Overall similarity weighting
    if cos_sim is not None:
        overall = (
            0.45 * max(0.0, (cos_sim + 1.0) / 2.0)
            + 0.35 * ssim_val
            + 0.20 * hist_val
        ) * 100.0
    else:
        overall = (0.60 * ssim_val + 0.40 * hist_val) * 100.0

    overall = float(np.clip(overall, 0.0, 100.0))

    if overall >= 88.0:
        verdict = "Identical Subject / Near Duplicate"
        color = "#22c55e"
        badge = "MATCH"
    elif overall >= 65.0:
        verdict = "Same Subject with Alterations / Face Swap"
        color = "#eab308"
        badge = "MODIFIED"
    elif overall >= 40.0:
        verdict = "Different Presentation / Strong Discrepancy"
        color = "#f97316"
        badge = "DISCREPANT"
    else:
        verdict = "Completely Divergent / Different Identity"
        color = "#ef4444"
        badge = "DIVERGENT"

    return {
        "overall_similarity": round(overall, 1),
        "ssim": round(ssim_val * 100.0, 1),
        "psnr_db": round(psnr_val, 1),
        "hist_corr": round(hist_val * 100.0, 1),
        "embed_cosine": round(cos_sim * 100.0, 1) if cos_sim is not None else None,
        "verdict": verdict,
        "badge": badge,
        "color": color,
    }


def generate_difference_heatmap(im_a: Image.Image, im_b: Image.Image) -> Image.Image:
    """Generate 2D spatial difference map (|A - B|) overlay highlighting alterations."""
    a = np.asarray(im_a.convert("RGB"), dtype=np.float32)
    b = np.asarray(im_b.convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))

    diff = np.abs(a - b).mean(axis=-1)  # (H, W) in [0, 255]
    # Normalize to [0, 255]
    diff_norm = np.clip(diff * 2.5, 0, 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(diff_norm, cv2.COLORMAP_TURBO)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Alpha blend: 60% heatmap, 40% original suspect image
    blended = (0.65 * heatmap_colored + 0.35 * b).astype(np.uint8)
    return Image.fromarray(blended)


def plot_3d_difference_surface(im_a: Image.Image, im_b: Image.Image) -> go.Figure:
    """Generate Plotly 3D Elevation Surface showing spatial difference amplitude."""
    a = cv2.resize(np.asarray(im_a.convert("L"), dtype=np.float32), (64, 64))
    b = cv2.resize(np.asarray(im_b.convert("L"), dtype=np.float32), (64, 64))
    diff = np.abs(a - b)

    fig = go.Figure(
        data=[
            go.Surface(
                z=diff,
                colorscale="Viridis",
                colorbar=dict(title="Delta Intensity", len=0.7),
            )
        ]
    )
    fig.update_layout(
        title="3D Spatial Modification Relief (|A - B|)",
        autosize=True,
        height=380,
        margin=dict(l=10, r=10, b=10, t=40),
        scene=dict(
            xaxis=dict(title="X", gridcolor="rgba(51,65,85,0.4)", showbackground=False),
            yaxis=dict(title="Y", gridcolor="rgba(51,65,85,0.4)", showbackground=False),
            zaxis=dict(title="Delta", gridcolor="rgba(51,65,85,0.4)", showbackground=False),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
    )
    return fig


def plot_3d_fft_difference(im_a: Image.Image, im_b: Image.Image) -> go.Figure:
    """Generate Plotly 3D Differential FFT Power Spectrum Surface."""
    a = cv2.resize(np.asarray(im_a.convert("L"), dtype=np.float32), (64, 64))
    b = cv2.resize(np.asarray(im_b.convert("L"), dtype=np.float32), (64, 64))

    fft_a = np.abs(np.fft.fftshift(np.fft.fft2(a)))
    fft_b = np.abs(np.fft.fftshift(np.fft.fft2(b)))

    log_a = np.log1p(fft_a)
    log_b = np.log1p(fft_b)
    diff_fft = log_b - log_a

    fig = go.Figure(
        data=[
            go.Surface(
                z=diff_fft,
                colorscale="Spectral",
                colorbar=dict(title="Δ Log-FFT", len=0.7),
            )
        ]
    )
    fig.update_layout(
        title="3D Differential FFT Power Spectrum Surface (Suspect vs Ref)",
        autosize=True,
        height=380,
        margin=dict(l=10, r=10, b=10, t=40),
        scene=dict(
            xaxis=dict(title="Freq X", gridcolor="rgba(51,65,85,0.4)", showbackground=False),
            yaxis=dict(title="Freq Y", gridcolor="rgba(51,65,85,0.4)", showbackground=False),
            zaxis=dict(title="Δ Spectral", gridcolor="rgba(51,65,85,0.4)", showbackground=False),
            camera=dict(eye=dict(x=-1.4, y=-1.4, z=1.3)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
    )
    return fig


def plot_similarity_gauge(similarity_pct: float, verdict_title: str) -> go.Figure:
    """Generate Plotly indicator gauge showing overall forensic similarity."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=similarity_pct,
            title={"text": f"<b>Forensic Similarity</b><br><span style='font-size:0.8em;color:#94a3b8'>{verdict_title}</span>"},
            number={"suffix": "%", "font": {"size": 38, "color": "#e2e8f0"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                "bar": {"color": "#38bdf8", "thickness": 0.25},
                "bgcolor": "rgba(15,23,42,0.6)",
                "borderwidth": 1,
                "bordercolor": "rgba(56,189,248,0.2)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(239,68,68,0.2)"},
                    {"range": [40, 70], "color": "rgba(245,158,11,0.2)"},
                    {"range": [70, 90], "color": "rgba(56,189,248,0.2)"},
                    {"range": [90, 100], "color": "rgba(34,197,94,0.2)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        height=220,
        margin=dict(l=30, r=30, t=50, b=20),
    )
    return fig


def plot_forensic_radar_comparison(res_a: Dict[str, Any], res_b: Dict[str, Any]) -> go.Figure:
    """Plot polar radar chart comparing forensic dimensions of File A vs File B."""
    qa = res_a.get("quality", {})
    qb = res_b.get("quality", {})

    axes = [
        "P(Manipulated)",
        "Noise Energy",
        "Chroma Variance",
        "Laplacian Sharpness",
        "Re-encoding Stability",
    ]

    p_a = res_a.get("p_fake", 0.0) * 100.0
    p_b = res_b.get("p_fake", 0.0) * 100.0

    noise_a = min(100.0, float(qa.get("srm_noise_energy", 10.0)) * 5.0)
    noise_b = min(100.0, float(qb.get("srm_noise_energy", 10.0)) * 5.0)

    chroma_a = min(100.0, float(qa.get("chroma_variance", 20.0)) * 2.0)
    chroma_b = min(100.0, float(qb.get("chroma_variance", 20.0)) * 2.0)

    sharp_a = min(100.0, math.sqrt(max(0.0, float(qa.get("laplacian_variance", 100.0)))) * 4.0)
    sharp_b = min(100.0, math.sqrt(max(0.0, float(qb.get("laplacian_variance", 100.0)))) * 4.0)

    stab_a = (res_a.get("stability") or {}).get("stability_score", 0.8) * 100.0
    stab_b = (res_b.get("stability") or {}).get("stability_score", 0.8) * 100.0

    vals_a = [p_a, noise_a, chroma_a, sharp_a, stab_a]
    vals_b = [p_b, noise_b, chroma_b, sharp_b, stab_b]

    # Close polar loop
    vals_a.append(vals_a[0])
    vals_b.append(vals_b[0])
    axes_loop = axes + [axes[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=vals_a,
            theta=axes_loop,
            fill="toself",
            name="File A (Reference)",
            line=dict(color="#38bdf8", width=2),
            fillcolor="rgba(56,189,248,0.2)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=vals_b,
            theta=axes_loop,
            fill="toself",
            name="File B (Suspect)",
            line=dict(color="#ef4444", width=2),
            fillcolor="rgba(239,68,68,0.2)",
        )
    )
    fig.update_layout(
        title="Forensic Footprint Comparison (A vs B)",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(51,65,85,0.4)"),
            angularaxis=dict(gridcolor="rgba(51,65,85,0.4)"),
            bgcolor="rgba(8,14,32,0.6)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        height=320,
        margin=dict(l=40, r=40, t=50, b=30),
        legend=dict(orientation="h", y=-0.15, x=0.2),
    )
    return fig


def plot_expert_logits_comparison(res_a: Dict[str, Any], res_b: Dict[str, Any]) -> go.Figure:
    """Grouped bar chart comparing individual expert logits for A and B."""
    ev_a = {e["name"]: float(e.get("logit") or 0.0) for e in res_a.get("evidence", []) if e.get("status") == "used"}
    ev_b = {e["name"]: float(e.get("logit") or 0.0) for e in res_b.get("evidence", []) if e.get("status") == "used"}

    all_names = sorted(list(set(ev_a.keys()) | set(ev_b.keys())))
    if not all_names:
        return go.Figure()

    vals_a = [ev_a.get(n, 0.0) for n in all_names]
    vals_b = [ev_b.get(n, 0.0) for n in all_names]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="File A (Ref)",
            x=all_names,
            y=vals_a,
            marker_color="#38bdf8",
        )
    )
    fig.add_trace(
        go.Bar(
            name="File B (Suspect)",
            x=all_names,
            y=vals_b,
            marker_color="#ef4444",
        )
    )
    fig.add_hline(y=0.0, line_dash="solid", line_color="#64748b")
    fig.update_layout(
        title="Detector Expert Logits Comparison (Log-Odds)",
        barmode="group",
        xaxis_title="Expert Detector",
        yaxis_title="Fake Logit (Positive = Fake)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,14,32,0.6)",
        font=dict(color="#94a3b8"),
        height=320,
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(orientation="h", y=1.12, x=0.25),
        yaxis=dict(gridcolor="rgba(51,65,85,0.4)"),
        xaxis=dict(gridcolor="rgba(51,65,85,0.4)"),
    )
    return fig
