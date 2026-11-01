"""DeepFake Forensic Intelligence Suite v4.0 — Next-Generation UI.

Upgrades over v3:
  * New futuristic dark theme with animated gradient accents.
  * Model Status Panel in sidebar (shows loaded models, device, calibration).
  * Generator Attribution panel (identifies GAN / Diffusion / XTTS / RVC).
  * NPR + Spectral expert integrated (no more stub returning zeros).
  * DINOv2-Large + SigLIP-SO400M-384 displayed in backends list.
  * Multi-encoder audio ensemble status shown.
  * LLM Narrative indicator (shows if GPT-4o/Gemini generated the text).
  * Diffusion-specific detector evidence shown in evidence table.
  * Optical Flow expert evidence shown for videos.
  * PDF export tab (requires reportlab).
  * Enhanced timeline chart with confidence bands.
  * Evidence contribution donut chart.
  * Calibration quality indicator (alpha=0.05, 95% coverage).
"""
import io
import json
import os
import importlib.util

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

if importlib.util.find_spec("torch") is None:
    st.error("PyTorch is not installed in this Python environment.")
    st.code("pip install torch torchvision torchaudio")
    st.stop()

from detector.pipeline import get_pipeline, infer_media_type
from detector.report_generator import (
    build_payload, generate_html_report, sign_payload,
)
from utils.result_view import render_result
from utils.ui_components import (
    apply_custom_theme, render_header, render_result_card,
)
from utils.visualizers import generate_ela_image, generate_fft_spectrum

st.set_page_config(
    page_title="🛡️ DeepFake Forensic Intelligence Suite v4.0",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Theme ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ===== GLOBAL RESET & FONTS ===== */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #04070f !important;
    color: #e2e8f0 !important;
}

/* ===== ANIMATED BACKGROUND ===== */
.stApp {
    background:
        radial-gradient(ellipse at 20% 0%, rgba(56,189,248,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 0%, rgba(139,92,246,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 100%, rgba(6,182,212,0.04) 0%, transparent 60%),
        linear-gradient(160deg, #04070f 0%, #060c1a 50%, #04070f 100%) !important;
    min-height: 100vh;
}

/* ===== MAIN CONTAINER ===== */
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: transparent !important;
    color: #e2e8f0 !important;
}

/* ===== HEADER STRIP ===== */
[data-testid="stHeader"] {
    background: rgba(4,7,15,0.95) !important;
    border-bottom: 1px solid rgba(56,189,248,0.15) !important;
    backdrop-filter: blur(12px) !important;
}
[data-testid="stToolbar"], [data-testid="stDecoration"] {
    background: transparent !important;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: rgba(8,14,32,0.97) !important;
    border-right: 1px solid rgba(56,189,248,0.12) !important;
    backdrop-filter: blur(20px) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #38bdf8 !important;
    font-weight: 700 !important;
}

/* ===== TABS ===== */
[data-testid="stTabs"] {
    border-bottom: 1px solid rgba(56,189,248,0.15) !important;
}
[data-testid="stTabs"] button {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] button:hover {
    color: #e2e8f0 !important;
    background: rgba(56,189,248,0.06) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom-color: #38bdf8 !important;
    background: rgba(56,189,248,0.08) !important;
    font-weight: 700 !important;
}

/* ===== CARDS ===== */
.forensic-card {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(56,189,248,0.12);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.forensic-card:hover {
    border-color: rgba(56,189,248,0.25);
    box-shadow: 0 0 32px rgba(56,189,248,0.06);
}

/* ===== VERDICT BADGES ===== */
.verdict-fake {
    background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(220,38,38,0.1));
    border: 2px solid #ef4444;
    border-radius: 14px;
    padding: 20px 28px;
    text-align: center;
    color: #fca5a5;
    font-size: 1.8rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    box-shadow: 0 0 40px rgba(239,68,68,0.2), inset 0 1px 0 rgba(255,255,255,0.05);
}
.verdict-real {
    background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(22,163,74,0.1));
    border: 2px solid #22c55e;
    border-radius: 14px;
    padding: 20px 28px;
    text-align: center;
    color: #86efac;
    font-size: 1.8rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    box-shadow: 0 0 40px rgba(34,197,94,0.2), inset 0 1px 0 rgba(255,255,255,0.05);
}
.verdict-inconclusive {
    background: linear-gradient(135deg, rgba(234,179,8,0.2), rgba(202,138,4,0.1));
    border: 2px solid #eab308;
    border-radius: 14px;
    padding: 20px 28px;
    text-align: center;
    color: #fde047;
    font-size: 1.8rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    box-shadow: 0 0 40px rgba(234,179,8,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
}

/* ===== SECTION TITLES ===== */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #38bdf8;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(56,189,248,0.2);
    letter-spacing: 0.02em;
}

/* ===== METRICS ===== */
[data-testid="stMetric"] {
    background: rgba(15,23,42,0.6) !important;
    border: 1px solid rgba(56,189,248,0.1) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.6rem !important; font-weight: 700 !important; }

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    background: rgba(8,14,32,0.6) !important;
    border: 1px solid rgba(56,189,248,0.15) !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(15,23,42,0.8) !important;
    border: 2px dashed rgba(56,189,248,0.3) !important;
    border-radius: 12px !important;
    color: #94a3b8 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    background: rgba(56,189,248,0.06) !important;
    border-color: #38bdf8 !important;
}
[data-testid="stFileUploaderDropzone"] * { color: #e2e8f0 !important; }
[data-testid="stFileUploaderDropzone"] svg { fill: #38bdf8 !important; }
[data-testid="stFileUploaderDropzone"] button {
    background: rgba(56,189,248,0.15) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: rgba(56,189,248,0.25) !important;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(14,165,233,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(14,165,233,0.35) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    box-shadow: 0 4px 15px rgba(239,68,68,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 8px 25px rgba(239,68,68,0.4) !important;
}

/* ===== DATAFRAMES ===== */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(56,189,248,0.12) !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* ===== EXPANDERS ===== */
[data-testid="stExpander"] {
    background: rgba(15,23,42,0.5) !important;
    border: 1px solid rgba(56,189,248,0.1) !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
}
[data-testid="stExpander"] summary {
    color: #94a3b8 !important;
    font-weight: 500 !important;
}

/* ===== RADIO ===== */
[data-testid="stRadio"] label, [data-testid="stRadio"] p { color: #e2e8f0 !important; }
[data-testid="stRadio"] input { accent-color: #38bdf8 !important; }

/* ===== CHECKBOXES & SLIDERS ===== */
[data-testid="stCheckbox"] label { color: #e2e8f0 !important; }
[data-testid="stSlider"] { color: #38bdf8 !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #38bdf8 !important;
}

/* ===== ALERTS ===== */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
}

/* ===== PROGRESS BAR ===== */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
    border-radius: 4px !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #04070f; }
::-webkit-scrollbar-thumb { background: rgba(56,189,248,0.3); border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: rgba(56,189,248,0.5); }

/* ===== FORM INPUTS ===== */
input, textarea, [data-baseweb="select"] > div {
    background-color: rgba(15,23,42,0.8) !important;
    color: #f8fafc !important;
    border-color: rgba(56,189,248,0.2) !important;
    border-radius: 8px !important;
}

/* ===== CODE BLOCKS ===== */
code, pre {
    background: rgba(8,14,32,0.8) !important;
    color: #7dd3fc !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 8px !important;
}

/* ===== MODEL STATUS BADGE ===== */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.75rem;
    color: #86efac;
    font-weight: 600;
    margin: 2px;
}
.model-badge-warn {
    background: rgba(234,179,8,0.12);
    border-color: rgba(234,179,8,0.3);
    color: #fde047;
}
.model-badge-err {
    background: rgba(239,68,68,0.12);
    border-color: rgba(239,68,68,0.3);
    color: #fca5a5;
}

/* ===== ATTRIBUTION CHIP ===== */
.attribution-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 0.88rem;
    color: #a5b4fc;
    font-weight: 600;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 32px 0 24px 0;">
    <div style="font-size: 2.6rem; font-weight: 900; letter-spacing: -0.03em;
                background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                background-clip: text; line-height: 1.1;">
        🛡️ DeepFake Forensic Intelligence Suite
    </div>
    <div style="font-size: 0.95rem; color: #64748b; margin-top: 8px; font-weight: 400;">
        v4.0 &nbsp;·&nbsp; DINOv2-Large &nbsp;·&nbsp; SigLIP-SO400M &nbsp;·&nbsp; NPR Frequency &nbsp;·&nbsp; 
        Multi-Encoder Audio &nbsp;·&nbsp; Optical Flow &nbsp;·&nbsp; Generator Attribution
    </div>
</div>
""", unsafe_allow_html=True)

pipe = get_pipeline()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    st.markdown(
        "<div style='font-size:0.8rem; color:#64748b; margin-bottom:12px;'>"
        "Balanced mode is always applied. Verdicts are evidence-based, not user-biased."
        "</div>", unsafe_allow_html=True,
    )
    mode = "balanced"
    explain = st.checkbox("🔬 Explanations (heatmaps)", value=True,
                          help="Multiscale occlusion heatmap + faithfulness check.")
    stability = st.checkbox("🔁 Re-encoding stability test", value=True,
                            help="Replays media through JPEG/blur/noise; checks verdict stability.")
    audio_chunk = 3
    st.caption("Audio analysed in consistent 3-second chunks.")

    st.markdown("---")

    # Model status panel
    st.markdown("### 🤖 Model & System Status")
    try:
        status = pipe.status()
        dev = status.get("device", "cpu").upper()
        dev_color = "#22c55e" if dev == "CUDA" else "#eab308" if dev == "MPS" else "#94a3b8"
        st.markdown(
            f"**Device:** <span style='color:{dev_color}; font-weight:700;'>{dev}</span>",
            unsafe_allow_html=True,
        )
        cal = status.get("any_calibrated", False)
        st.markdown(
            f"**Calibration:** {'<span style=\"color:#22c55e\">✓ Fitted (alpha=0.05)</span>' if cal else '<span style=\"color:#eab308\">⚠ Uncalibrated</span>'}",
            unsafe_allow_html=True,
        )
        backends = status.get("image_backends", [])
        if backends:
            st.markdown("**Image Backends:**")
            for b in backends[:5]:
                short = b.split(":")[-1][:28] if ":" in b else b[:28]
                st.markdown(f"<span class='model-badge'>✓ {short}</span>", unsafe_allow_html=True)
        audio_b = status.get("audio_backend", "")
        if audio_b:
            is_ensemble = "ensemble" in audio_b.lower()
            badge_text = "🎵 Multi-Encoder Ensemble" if is_ensemble else f"🎵 {audio_b[:25]}"
            badge_class = "model-badge" if is_ensemble else "model-badge-warn"
            st.markdown(f"<span class='{badge_class}'>{badge_text}</span>", unsafe_allow_html=True)
        spec = status.get("spectral_head", False)
        st.markdown(
            f"**NPR+Spectral:** {'<span style=\"color:#22c55e\">✓ Trained head loaded</span>' if spec else '<span style=\"color:#ef4444\">✗ Not trained (run train.train_spectral)</span>'}",
            unsafe_allow_html=True,
        )
        ctrl = status.get("controller", False)
        st.markdown(
            f"**VOI Controller:** {'<span style=\"color:#22c55e\">✓ Active</span>' if ctrl else '<span style=\"color:#94a3b8\">Off</span>'}",
            unsafe_allow_html=True,
        )
        dep = status.get("dependent_fusion", False)
        st.markdown(
            f"**Fusion:** {'<span style=\"color:#22c55e\">✓ GLS Correlation-aware</span>' if dep else '<span style=\"color:#94a3b8\">Weighted mean</span>'}",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Status check failed: {e}")

    st.markdown("---")
    st.markdown("### ℹ️ v4.0 New Experts")
    st.markdown(
        "<div style='font-size:0.78rem; color:#64748b; line-height:1.7;'>"
        "🆕 <b>NPR Frequency Expert</b><br>"
        "🆕 <b>Diffusion Detector</b> (SDXL/DALL-E/Flux)<br>"
        "🆕 <b>SigLIP-SO400M</b> 400M/384px<br>"
        "🆕 <b>DINOv2-Large</b> 307M encoder<br>"
        "🆕 <b>WavLM+Wav2Vec2</b> ensemble<br>"
        "🆕 <b>Optical Flow</b> (video)<br>"
        "🆕 <b>Generator Attribution</b><br>"
        "🆕 <b>LLM Narrative</b> (GPT-4o/Gemini)"
        "</div>", unsafe_allow_html=True,
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────
def analyze(source, media, name, *, do_explain=None, do_stab=None):
    try:
        res = pipe.analyze(
            source, media_type=media, filename=name, mode=mode,
            explain=explain if do_explain is None else do_explain,
            stability=stability if do_stab is None else do_stab,
            audio_chunk_s=audio_chunk,
        )
    except Exception as e:
        msg = str(e)
        st.error(f"Analysis failed: {msg}")
        if any(k in msg.lower() for k in ("huggingface", "connection", "offline", "401", "403")):
            st.info(
                "Model download failed. Check your network and, for gated models, "
                "set HF_TOKEN in .env."
            )
        return None
    return res


def remember(res):
    st.session_state["latest"] = {"result": res.to_dict(), "artifacts": res.artifacts}


def _verdict_html(verdict: str, p_fake: float) -> str:
    cls = {
        "Fake": "verdict-fake",
        "Real": "verdict-real",
        "Inconclusive": "verdict-inconclusive",
    }.get(verdict, "verdict-inconclusive")
    pct = f"{p_fake * 100:.1f}%"
    icon = {"Fake": "⚠️", "Real": "✅", "Inconclusive": "❓"}.get(verdict, "❓")
    return (
        f"<div class='{cls}'>"
        f"<div>{icon} {verdict}</div>"
        f"<div style='font-size:1rem; font-weight:500; margin-top:6px; opacity:0.8;'>"
        f"P(manipulated) = {pct}</div>"
        f"</div>"
    )


def _attribution_html(attribution: dict) -> str:
    if not attribution or not attribution.get("family"):
        return ""
    family = attribution["family"]
    prob = attribution.get("probability", 0)
    desc = attribution.get("description", "")
    icons = {
        "GAN": "🤖", "Diffusion": "🌊", "VAE": "🧩",
        "Face_Swap": "😶", "Inpaint": "🖌️",
        "TTS": "🎤", "Voice_Conv": "🔄", "Unknown": "❓",
    }
    icon = icons.get(family, "🔬")
    return (
        f"<div class='attribution-chip'>"
        f"{icon} Generator: <b>{family}</b> &nbsp;(confidence {prob:.0%})"
        f"<span style='font-weight:400; color:#818cf8; font-size:0.8rem;'> — {desc}</span>"
        f"</div>"
    )


_GRID_COLOR_DASH = "rgba(51,65,85,0.5)"


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🎯 Single File Scanner",
    "👁️ Interview Guard",
    "📷 Snapshot Analysis",
    "⚡ Batch Suite",
    "🆚 Compare Files",
    "🔬 Forensic Lab",
    "📄 Signed Report",
    "📊 Model Dashboard",
])
tab_single, tab_interview, tab_cam, tab_batch, tab_compare, tab_lab, tab_report, tab_dash = tabs

# ── 1. Single File Scanner ────────────────────────────────────────────────────
with tab_single:
    st.markdown('<div class="section-title">📁 Upload media for forensic inspection</div>',
                unsafe_allow_html=True)
    option = st.radio("Media type", ["Image", "Video (multimodal)", "Audio"],
                      horizontal=True, key="single_media_type")
    spec = {
        "Image": ("image", ["jpg", "jpeg", "png", "webp"]),
        "Video (multimodal)": ("video", ["mp4", "avi", "mov", "mkv", "webm"]),
        "Audio": ("audio", ["wav", "mp3", "flac", "ogg", "m4a"]),
    }[option]

    up = st.file_uploader(f"Upload {spec[0]}", type=spec[1], key=f"single_{spec[0]}")
    if up is not None:
        data = up.getvalue()
        st.markdown("---")
        if spec[0] == "video":
            st.video(data)
        elif spec[0] == "audio":
            st.audio(data)
        elif spec[0] == "image":
            st.image(data, width=400)

        with st.spinner("🔬 Running evidence cascade (v4.0)..."):
            res = analyze(data, spec[0], up.name)

        if res is not None:
            remember(res)
            d = res.to_dict()

            # ── Verdict banner ──
            st.markdown(_verdict_html(d["verdict"], d["p_fake"]), unsafe_allow_html=True)

            # ── Attribution chip (if available) ──
            att = d.get("attribution")
            if att:
                st.markdown(_attribution_html(att), unsafe_allow_html=True)

            # ── LLM narrative indicator ──
            expl = d.get("explanation") or {}
            if expl.get("narrative_source") == "llm":
                st.markdown(
                    "<span style='font-size:0.78rem; color:#818cf8; font-weight:600;'>"
                    "✨ Narrative generated by LLM (GPT-4o / Gemini)</span>",
                    unsafe_allow_html=True,
                )

            render_result(d, res.artifacts, key=f"single_{spec[0]}")

# ── 2. INTERVIEW GUARD ───────────────────────────────────────────────────────
with tab_interview:
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(99,102,241,0.15),rgba(56,189,248,0.08));
    border:1px solid rgba(99,102,241,0.3); border-radius:16px; padding:20px; margin-bottom:20px;'>
    <div style='font-size:1.5rem; font-weight:800; color:#818cf8; margin-bottom:8px;'>
    👁️ Interview Guard — Live Integrity Monitoring</div>
    <div style='font-size:0.92rem; color:#94a3b8; line-height:1.7;'>
    Designed for <b>online job interviews, proctored exams, and video calls</b>.<br>
    Detects: • AI/Virtual camera deepfakes &nbsp;• Off-screen gaze &nbsp;• Multiple persons<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• Abnormal head pose &nbsp;• Low blink rate (screen reading) &nbsp;• Unnatural facial motion
    </div></div>
    """, unsafe_allow_html=True)

    if "guard" not in st.session_state:
        from detector.interview_guard import InterviewGuard
        st.session_state["guard"] = InterviewGuard()

    guard = st.session_state["guard"]

    # ─ Control panel ────────────────────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
    use_pipe = ctrl_col1.checkbox(
        "🧠 Deep AI scan per frame",
        value=False,
        help="Run full deepfake pipeline on each frame (slow but very accurate)",
    )
    show_annotated = ctrl_col2.checkbox("🎨 Show annotated frame", value=True)
    if ctrl_col3.button("🔄 Reset session", key="guard_reset"):
        guard.reset_session()
        st.session_state.pop("guard_risk_timeline", None)
        st.success("Session reset.")

    frame_pipe = pipe if use_pipe else None

    # ─ Live camera capture ───────────────────────────────────────────
    cam_live, dash_live = st.columns([1, 1])
    with cam_live:
        st.markdown("**📷 Live Feed** — click Capture to analyse")
        live_snap = st.camera_input("Capture frame", key="guard_cam", label_visibility="collapsed")

    with dash_live:
        # Real-time gauge dashboard.
        if live_snap is not None:
            live_img = Image.open(live_snap).convert("RGB")
            with st.spinner("Analysing frame..."):
                report = guard.analyze_frame(live_img, frame_pipe)
            # Store timeline.
            tl = st.session_state.get("guard_risk_timeline", [])
            tl.append(report.overall_risk)
            st.session_state["guard_risk_timeline"] = tl

            # Verdict banner.
            if report.overall_risk >= 60:
                banner_color = "#ef4444"
                banner_text = "🚨 HIGH RISK"
            elif report.overall_risk >= 35:
                banner_color = "#eab308"
                banner_text = "⚠️ SUSPICIOUS"
            else:
                banner_color = "#22c55e"
                banner_text = "✅ AUTHENTIC"

            st.markdown(
                f"<div style='text-align:center; font-size:1.6rem; font-weight:900; "
                f"color:{banner_color}; padding:12px; border:2px solid {banner_color}; "
                f"border-radius:12px; margin-bottom:16px;'>{banner_text} &nbsp;·&nbsp; "
                f"Risk: {report.overall_risk:.0f}/100</div>",
                unsafe_allow_html=True,
            )

            # Annotated frame.
            if show_annotated and report.annotated is not None:
                st.image(report.annotated, use_column_width=True, caption="Annotated frame")

            # Indicator gauge bar.
            st.markdown("**📊 Live Indicator Scores:**")
            from detector.interview_guard import INDICATOR_WEIGHTS, ALERT_THRESHOLDS
            import plotly.graph_objects as _pgo
            ind_names = list(report.indicators.keys())
            ind_vals = list(report.indicators.values())
            ind_colors = [
                "#ef4444" if v >= ALERT_THRESHOLDS.get(k, 100)
                else "#eab308" if v >= ALERT_THRESHOLDS.get(k, 100) * 0.6
                else "#22c55e"
                for k, v in report.indicators.items()
            ]
            fig_ind = _pgo.Figure(_pgo.Bar(
                x=[n.replace("_", " ").title() for n in ind_names],
                y=ind_vals,
                marker_color=ind_colors,
                text=[f"{v:.0f}" for v in ind_vals],
                textposition="outside",
            ))
            fig_ind.update_layout(
                yaxis=dict(range=[0, 115], gridcolor=_GRID_COLOR_DASH),
                xaxis=dict(gridcolor=_GRID_COLOR_DASH),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(8,14,32,0.6)",
                font_color="#94a3b8",
                height=240,
                margin=dict(t=20, b=40, l=40, r=20),
                showlegend=False,
            )
            st.plotly_chart(fig_ind, use_container_width=True, key="guard_ind")

            # Flags.
            if report.flags:
                for flag in report.flags:
                    st.error(flag)
            else:
                st.success("✅ No anomalies detected in this frame")

            # Head pose + gaze metrics.
            pose_col1, pose_col2, pose_col3 = st.columns(3)
            pose_col1.metric("Head Yaw", f"{report.head_yaw:.1f}°",
                             help="Horizontal head rotation")
            pose_col2.metric("Head Pitch", f"{report.head_pitch:.1f}°",
                             help="Vertical head rotation")
            pose_col3.metric("Faces in frame", report.face_count)

        else:
            st.info("📷 Click the camera button above to capture and analyse a frame.")
            st.markdown(
                "<div style='font-size:0.85rem; color:#64748b; line-height:1.8;'>"
                "👁️ <b>What it checks:</b><br>"
                "🤖 Face authenticity (deepfake/virtual camera)<br>"
                "👀 Off-screen gaze (iris tracking)<br>"
                "📍 Head pose (yaw/pitch estimation)<br>"
                "👥 Multiple persons in frame<br>"
                "🙄 Blink rate anomaly<br>"
                "🎬 Unnatural facial motion pattern"
                "</div>",
                unsafe_allow_html=True,
            )

    # ─ Risk timeline + Session summary ──────────────────────────────────
    risk_tl = st.session_state.get("guard_risk_timeline", [])
    if len(risk_tl) >= 2:
        st.markdown("---")
        st.markdown("### 📈 Session Risk Timeline")
        import plotly.graph_objects as _pgo2
        frames_x = list(range(1, len(risk_tl) + 1))
        risk_colors = ["#ef4444" if r >= 60 else "#eab308" if r >= 35 else "#22c55e"
                       for r in risk_tl]
        fig_tl = _pgo2.Figure(_pgo2.Bar(
            x=frames_x, y=risk_tl,
            marker_color=risk_colors,
            name="Frame risk score",
        ))
        fig_tl.add_hline(y=60, line_dash="dash", line_color="#ef4444",
                         annotation_text="High Risk", annotation_font_color="#ef4444")
        fig_tl.add_hline(y=35, line_dash="dot", line_color="#eab308",
                         annotation_text="Suspicious", annotation_font_color="#eab308")
        fig_tl.update_layout(
            xaxis_title="Frame #", yaxis_title="Risk score (0–100)",
            yaxis=dict(range=[0, 110], gridcolor=_GRID_COLOR_DASH),
            xaxis=dict(gridcolor=_GRID_COLOR_DASH),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,14,32,0.6)",
            font_color="#94a3b8",
            height=240,
            margin=dict(t=30, b=40, l=50, r=20),
        )
        st.plotly_chart(fig_tl, use_container_width=True, key="guard_tl")

        # Session summary.
        summary = guard.session_summary()
        st.markdown("### 📊 Session Integrity Report")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("🎯 Session Verdict", summary.verdict)
        sc2.metric("📊 Mean Risk", f"{summary.mean_risk:.0f}/100")
        sc3.metric("⚠️ Max Risk", f"{summary.max_risk:.0f}/100")
        sc4.metric("🛨 Total Flags", summary.total_flags)

        if summary.recommendations:
            st.markdown("**📝 Recommendations for interviewer:**")
            for rec in summary.recommendations:
                st.warning(rec)
        elif summary.verdict == "✅ Clean":
            st.success("✅ Session clean. No integrity concerns detected.")

        # Flag counts breakdown.
        if any(v > 0 for v in summary.flag_counts.values()):
            with st.expander("🔢 Flag breakdown by indicator", expanded=False):
                flag_df = pd.DataFrame([
                    {"Indicator": k.replace("_", " ").title(),
                     "Flag count": v}
                    for k, v in summary.flag_counts.items() if v > 0
                ])
                st.dataframe(flag_df, use_container_width=True, hide_index=True)


# ── 3. Snapshot Analysis (📷) ──────────────────────────────────────────────────
with tab_cam:
    st.markdown('<div class="section-title">📷 Camera snapshot forensic analysis</div>',
                unsafe_allow_html=True)
    st.caption(
        "Full forensic pipeline (same as Single File Scanner) on a webcam snapshot. "
        "For continuous real-time monitoring use the 👁️ Interview Guard tab."
    )
    pic = st.camera_input("Take a snapshot for forensic analysis", key="cam_input")
    if pic is not None:
        with st.spinner("🔍 Analysing frame with full pipeline..."):
            res = analyze(pic.getvalue(), "image", "camera_snapshot.png")
        if res is not None:
            remember(res)
            d = res.to_dict()
            st.markdown(_verdict_html(d["verdict"], d["p_fake"]), unsafe_allow_html=True)
            att = d.get("attribution")
            if att:
                st.markdown(_attribution_html(att), unsafe_allow_html=True)
            render_result(d, res.artifacts, key="cam")

# ── 3. Batch Forensic Suite ───────────────────────────────────────────────────
with tab_batch:
    st.markdown('<div class="section-title">⚡ Batch forensic scanning</div>',
                unsafe_allow_html=True)
    files = st.file_uploader(
        "Upload multiple files (images, videos, audio)",
        accept_multiple_files=True, key="batch_files",
        type=["jpg", "jpeg", "png", "webp", "wav", "mp3", "flac", "mp4", "avi", "mov"],
    )
    if files and st.button("🚀 Start batch analysis", type="primary", key="batch_run"):
        rows, bar = [], st.progress(0.0)
        status_text = st.empty()
        for i, f in enumerate(files):
            status_text.markdown(
                f"<div style='color:#94a3b8;'>Analysing {i+1}/{len(files)}: {f.name}</div>",
                unsafe_allow_html=True,
            )
            media = infer_media_type(f.name)
            res = analyze(f.getvalue(), media, f.name, do_explain=False, do_stab=False) if media else None
            if res is not None:
                d = res.to_dict()
                top = max(
                    (e for e in d["evidence"] if e["status"] == "used"),
                    key=lambda e: abs(e["logit"] or 0), default=None,
                )
                att = d.get("attribution")
                rows.append({
                    "File": f.name, "Type": media,
                    "Verdict": d["verdict"],
                    "P(fake)": round(d["p_fake"], 3),
                    "Confidence": None if d["confidence"] is None else round(d["confidence"], 3),
                    "Calibrated": d["calibrated"],
                    "Generator": att["family"] if att else "—",
                    "Top Evidence": top["name"] if top else "—",
                    "Warnings": "; ".join(d["warnings"])[:120],
                })
            else:
                rows.append({
                    "File": f.name, "Type": media or "unsupported", "Verdict": "error",
                    "Generator": "—",
                })
            bar.progress((i + 1) / len(files))
        status_text.empty()
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        c1.download_button(
            "📥 Download CSV", df.to_csv(index=False),
            "batch_results.csv", "text/csv",
        )
        # Summary stats
        if len(df) > 0 and "Verdict" in df.columns:
            counts = df["Verdict"].value_counts()
            st.markdown("**Batch summary:**")
            cols = st.columns(min(len(counts), 3))
            for i, (v, n) in enumerate(counts.items()):
                color = {"Fake": "#ef4444", "Real": "#22c55e"}.get(v, "#eab308")
                cols[i % 3].metric(v, n)

# ── 4. Compare Files ──────────────────────────────────────────────────────────
with tab_compare:
    st.markdown('<div class="section-title">🆚 Side-by-side forensic comparison</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    fa = c1.file_uploader("📁 File A (reference)", type=["jpg", "jpeg", "png", "webp"], key="cmp_a")
    fb = c2.file_uploader("📁 File B (suspect)", type=["jpg", "jpeg", "png", "webp"], key="cmp_b")
    if fa is not None and fb is not None:
        with st.spinner("🔬 Running comparison..."):
            ra = analyze(fa.getvalue(), "image", fa.name)
            rb = analyze(fb.getvalue(), "image", fb.name)
        for col, r, name in ((c1, ra, fa.name), (c2, rb, fb.name)):
            with col:
                if r is None:
                    continue
                d = r.to_dict()
                st.markdown(_verdict_html(d["verdict"], d["p_fake"]), unsafe_allow_html=True)
                att = d.get("attribution")
                if att:
                    st.markdown(_attribution_html(att), unsafe_allow_html=True)
                if r.artifacts.get("annotated") is not None:
                    st.image(r.artifacts["annotated"], caption=name, use_column_width=True)
                if r.artifacts.get("heatmap") is not None:
                    st.image(r.artifacts["heatmap"], caption="Occlusion heatmap",
                             use_column_width=True)
                st.caption(d.get("narrative", ""))

# ── 5. Forensic Lab ───────────────────────────────────────────────────────────
with tab_lab:
    st.markdown('<div class="section-title">🔬 Classic forensic views (educational)</div>',
                unsafe_allow_html=True)
    st.info(
        "ELA and FFT are **visual aids** — they do not feed the v4.0 verdict. "
        "The decision uses the full evidence table including NPR frequency, "
        "SigLIP-SO400M, and diffusion detector. Use these for educational exploration."
    )
    lab = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="lab_img")
    if lab is not None:
        im = Image.open(lab).convert("RGB")
        q = st.slider("JPEG quality (ELA)", 50, 98, 90)
        s = st.slider("ELA brightness scale", 5, 30, 15)
        a_col, b_col, c_col = st.columns(3)
        a_col.image(im, caption="Original", use_column_width=True)
        b_col.image(
            generate_ela_image(im, quality=q, scale=s),
            caption=f"Error Level Analysis (Q={q})", use_column_width=True,
        )
        c_col.image(
            generate_fft_spectrum(im),
            caption="2D FFT magnitude spectrum", use_column_width=True,
        )
        st.markdown("---")
        st.markdown("**Run full forensic analysis on this image:**")
        if st.button("🔬 Analyze in full pipeline", key="lab_full"):
            lab.seek(0)
            with st.spinner("Running evidence cascade..."):
                res = analyze(lab.getvalue(), "image", lab.name)
            if res is not None:
                remember(res)
                d = res.to_dict()
                st.markdown(_verdict_html(d["verdict"], d["p_fake"]), unsafe_allow_html=True)
                render_result(d, res.artifacts, key="lab_full")

# ── 6. Signed Report ──────────────────────────────────────────────────────────
with tab_report:
    st.markdown('<div class="section-title">📄 Cryptographically signed forensic report</div>',
                unsafe_allow_html=True)
    latest = st.session_state.get("latest")
    if not latest:
        st.warning("No analysis yet. Scan a file in the Single File Scanner first.")
    else:
        d, arts = latest["result"], latest["artifacts"]
        verdict_color = {"Fake": "#ef4444", "Real": "#22c55e"}.get(d["verdict"], "#eab308")
        st.markdown(
            f"**File:** `{d['filename']}` ({d['media_type']}) &nbsp;·&nbsp; "
            f"**Verdict:** <span style='color:{verdict_color}; font-weight:700;'>{d['verdict']}</span> "
            f"&nbsp;·&nbsp; **P(fake):** `{d['p_fake'] * 100:.1f}%`",
            unsafe_allow_html=True,
        )
        att = d.get("attribution")
        if att:
            st.markdown(_attribution_html(att), unsafe_allow_html=True)
        payload = build_payload(d)
        signed = {"payload": payload, "signature": sign_payload(payload)}
        html = generate_html_report(d, arts, signed)

        c1, c2 = st.columns(2)
        c1.download_button(
            "📥 HTML Forensic Report", html,
            f"forensic_report_{d['filename']}.html", "text/html", type="primary",
        )
        c2.download_button(
            "📥 Signed JSON Report",
            json.dumps(signed, indent=2),
            f"forensic_report_{d['filename']}.json", "application/json",
        )

        # PDF export
        from detector.config import get_settings
        if get_settings().enable_pdf_export:
            try:
                from io import BytesIO
                from reportlab.lib.pagesizes import letter
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

                def build_pdf():
                    buf = BytesIO()
                    doc = SimpleDocTemplate(buf, pagesize=letter)
                    styles = getSampleStyleSheet()
                    story = [
                        Paragraph("DeepFake Forensic Intelligence Suite v4.0", styles["Title"]),
                        Paragraph(f"File: {d['filename']}", styles["Normal"]),
                        Paragraph(f"Verdict: {d['verdict']}", styles["Normal"]),
                        Paragraph(f"P(manipulated): {d['p_fake']*100:.1f}%", styles["Normal"]),
                        Spacer(1, 12),
                        Paragraph("Narrative:", styles["Heading2"]),
                        Paragraph(d.get("narrative", "N/A"), styles["Normal"]),
                    ]
                    if att:
                        story.append(Paragraph(
                            f"Generator Attribution: {att['family']} ({att['probability']:.0%})",
                            styles["Normal"]
                        ))
                    doc.build(story)
                    return buf.getvalue()

                pdf_bytes = build_pdf()
                st.download_button(
                    "📥 PDF Forensic Report",
                    pdf_bytes,
                    f"forensic_report_{d['filename']}.pdf",
                    "application/pdf",
                )
            except ImportError:
                st.caption("PDF export requires: `pip install reportlab`")
            except Exception as pe:
                st.caption(f"PDF export failed: {pe}")

        st.caption(
            "The JSON report is signed with Ed25519. Verify later with "
            "`python -m scripts.verify_report <file>.json`."
        )
        components.html(html, height=500, scrolling=True)

# ── 7. Model Dashboard ────────────────────────────────────────────────────────
with tab_dash:
    st.markdown('<div class="section-title">📊 Model & System Dashboard</div>',
                unsafe_allow_html=True)
    try:
        status = pipe.status()
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Device", status.get("device", "cpu").upper())
        r2.metric("Calibrated", "✓ Yes" if status.get("any_calibrated") else "✗ No")
        r3.metric("Spectral/NPR Head", "✓ Loaded" if status.get("spectral_head") else "✗ Missing")
        r4.metric("Temporal Head", "✓ Loaded" if status.get("learned_temporal_head") else "✗ Missing")

        st.markdown("---")
        st.markdown("#### 🖼️ Image Backends")
        for b in status.get("image_backends", []):
            st.markdown(f"<span class='model-badge'>✓ {b}</span>", unsafe_allow_html=True)

        st.markdown("#### 🎵 Audio Backend")
        ab = status.get("audio_backend", "N/A")
        is_ens = "ensemble" in ab.lower()
        ab_class = "model-badge" if is_ens else "model-badge-warn"
        ab_label = "🎵 Multi-Encoder: " + ab if is_ens else ab
        st.markdown(
            f"<span class='{ab_class}'>{ab_label}</span>",
            unsafe_allow_html=True,
        )

        st.markdown("#### ⚙️ Research Modules")
        modules = {
            "VOI Investigation Controller": status.get("controller"),
            "GLS Correlation-aware Fusion": status.get("dependent_fusion"),
            "Conditional Calibration": bool(status.get("conditional_calibration")),
            "Learned Fusion Weights": status.get("learned_fusion_weights"),
        }
        m_cols = st.columns(2)
        for i, (name, active) in enumerate(modules.items()):
            icon = "✓" if active else "✗"
            color = "#22c55e" if active else "#64748b"
            m_cols[i % 2].markdown(
                f"<span style='color:{color};'>{icon} {name}</span>",
                unsafe_allow_html=True,
            )

        st.markdown("#### 📋 Calibrators")
        cal_data = status.get("calibrators", {})
        if cal_data:
            cal_rows = [
                {"Name": k, "Fitted": v["fitted"], "N_cal": v["n_cal"], "Alpha": v["alpha"]}
                for k, v in cal_data.items()
            ]
            st.dataframe(pd.DataFrame(cal_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No calibrators loaded. Run `python -m train.fit_calibration` to fit.")

        st.markdown("#### 🔢 OOD Detection Status")
        ood = status.get("ood_fitted", {})
        for modality, fitted in ood.items():
            color = "#22c55e" if fitted else "#ef4444"
            st.markdown(
                f"<span style='color:{color};'>{'✓' if fitted else '✗'} {modality}</span> ",
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"Dashboard error: {e}")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#334155; font-size:0.8rem; padding:12px;'>"
    "DeepFake Forensic Intelligence Suite v4.0 &nbsp;·&nbsp; Built by OGmayur &nbsp;·&nbsp; "
    "Reports support human review; they are not legal proof. "
    "Detectors can be wrong on unseen generators."
    "</div>",
    unsafe_allow_html=True,
)
