import json
import os
import importlib.util

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

if importlib.util.find_spec("torch") is None:
    st.error("PyTorch is not installed in this Python environment.")
    st.code("run_app.bat  # Windows\npy -3.9 -m streamlit run app.py")
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
    page_title="🛡️ Deepfake Forensic Intelligence Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_custom_theme()
render_header()

pipe = get_pipeline()

# === SIDEBAR ===============================================================
st.sidebar.markdown("### ⚖️ Universal scan mode")
st.sidebar.info(
    "Balanced mode is applied consistently to every input. "
    "Verdicts are based on the available evidence, not a user-selected bias."
)
mode = "balanced"
explain = st.sidebar.checkbox("Explanations (heatmaps)", value=True,
                              help="Occlusion heatmap + faithfulness check. Slower.")
stability = st.sidebar.checkbox("Re-encoding stability test", value=True,
                                help="Replays the media through JPEG/blur/noise and "
                                     "checks the verdict does not flip.")
audio_chunk = 3
st.sidebar.caption("Audio is analyzed in consistent 3-second chunks.")


# === helpers ================================================================
def analyze(source, media, name, *, do_explain=None, do_stab=None):
    """Run the pipeline and store the result for the report tab."""
    try:
        res = pipe.analyze(
            source, media_type=media, filename=name, mode=mode,
            explain=explain if do_explain is None else do_explain,
            stability=stability if do_stab is None else do_stab,
            audio_chunk_s=audio_chunk,
        )
    except Exception as e:  # surface model-download / decode problems clearly
        msg = str(e)
        st.error(f"Analysis failed: {msg}")
        if any(k in msg.lower() for k in ("huggingface", "connection", "offline", "401", "403")):
            st.info("Model download failed. Check your network and, for gated "
                    "models, set HF_TOKEN in a .env file.")
        return None
    return res


def remember(res):
    st.session_state["latest"] = {"result": res.to_dict(), "artifacts": res.artifacts}

def apply_custom_theme():
    st.markdown("""
    <style>
    /* ===== GLOBAL DARK THEME ===== */
    .stApp {
        background: linear-gradient(135deg, #111827 0%, #080d1b 100%);
        color: #f8fafc;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: transparent !important;
        color: #f8fafc !important;
    }

    /* ===== TOP HEADER STRIP ===== */
    [data-testid="stHeader"] {
        background: #6b7280 !important;
        color: #f8fafc !important;
    }

    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        background: #6b7280 !important;
    }

    /* Header buttons and icons */
    [data-testid="stHeader"] button,
    [data-testid="stHeader"] svg {
        color: #e2e8f0 !important;
        fill: currentColor;
    }

    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] {
        background: #111b30 !important;
        border: 1px solid #334766 !important;
        border-radius: 14px !important;
        padding: 10px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #16243b !important;
        border: 2px dashed #526b91 !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        background: #1c304c !important;
        border-color: #38bdf8 !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #e2e8f0 !important;
    }

    [data-testid="stFileUploaderDropzone"] svg {
        fill: #60a5fa !important;
        color: #60a5fa !important;
    }

    /* Browse files button */
    [data-testid="stFileUploaderDropzone"] button {
        background: #243a59 !important;
        color: #ffffff !important;
        border: 1px solid #526b91 !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background: #2563eb !important;
        border-color: #60a5fa !important;
    }

    /* ===== RADIO BUTTONS ===== */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] p {
        color: #e2e8f0 !important;
    }

    [data-testid="stRadio"] input {
        accent-color: #ff4b5c !important;
    }

    /* ===== TEXT AND FORM LABELS ===== */
    label,
    .stMarkdown,
    .stCaption,
    [data-testid="stWidgetLabel"] p {
        color: #e2e8f0;
    }

    /* ===== INPUTS AND SELECT BOXES ===== */
    input,
    textarea,
    [data-baseweb="select"] > div {
        background-color: #16243b !important;
        color: #f8fafc !important;
        border-color: #334766 !important;
    }

    /* ===== TABS ===== */
    [data-testid="stTabs"] button {
        color: #e2e8f0 !important;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #60a5fa !important;
        border-bottom-color: #38bdf8 !important;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: #111a2d !important;
        border-right: 1px solid #26344d;
    }

    [data-testid="stSidebar"] * {
        color: #f1f5f9;
    }

    /* ===== CHECKBOXES AND SLIDERS ===== */
    [data-testid="stCheckbox"] label {
        color: #e2e8f0 !important;
    }

    [data-testid="stSlider"] {
        color: #ff4b5c !important;
    }

    /* ===== UPLOADED FILE DETAILS ===== */
    [data-testid="stFileUploaderFile"] {
        background: #16243b !important;
        border: 1px solid #334766 !important;
        border-radius: 8px !important;
    }

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0b1020;
    }

    ::-webkit-scrollbar-thumb {
        background: #334766;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# === TABS ===================================================================
tab_single, tab_cam, tab_batch, tab_compare, tab_lab, tab_report = st.tabs([
    "🎯 Single File Scanner", "📷 Live Webcam Scanner", "⚡ Batch Forensic Suite",
    "🆚 Dual File Comparison", "🔬 Deep Forensic Lab", "📄 Signed Report Export",
])

with tab_single:
    st.markdown('<div class="section-title">📁 Select media for forensic inspection</div>',
                unsafe_allow_html=True)
    option = st.radio("Media category", ["Image", "Video (multimodal)", "Audio"], horizontal=True)
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
            st.image(data, width=420)
        with st.spinner("🔬 Running the evidence cascade..."):
            res = analyze(data, spec[0], up.name)
        if res is not None:
            remember(res)
            render_result(res.to_dict(), res.artifacts, key=f"single_{spec[0]}")

with tab_cam:
    st.markdown('<div class="section-title">📷 Camera snapshot inspector</div>',
                unsafe_allow_html=True)
    st.caption("A single snapshot cannot show temporal evidence (pulse, lip-sync); "
               "use the video scanner for those.")
    pic = st.camera_input("Take a snapshot for a forensic scan")
    if pic is not None:
        with st.spinner("🔍 Analyzing camera frame..."):
            res = analyze(pic.getvalue(), "image", "camera_snapshot.png")
        if res is not None:
            remember(res)
            render_result(res.to_dict(), res.artifacts, key="cam")

with tab_batch:
    st.markdown('<div class="section-title">⚡ Batch scanning</div>', unsafe_allow_html=True)
    files = st.file_uploader(
        "Upload multiple files", accept_multiple_files=True, key="batch",
        type=["jpg", "jpeg", "png", "webp", "wav", "mp3", "flac", "mp4", "avi", "mov"])
    if files and st.button("🚀 Start batch analysis", type="primary"):
        rows, bar = [], st.progress(0.0)
        for i, f in enumerate(files):
            media = infer_media_type(f.name)
            res = analyze(f.getvalue(), media, f.name, do_explain=False, do_stab=False) if media else None
            if res is not None:
                top = max((e for e in res.evidence if e["status"] == "used"),
                          key=lambda e: abs(e["logit"] or 0), default=None)
                rows.append({
                    "File": f.name, "Type": media, "Verdict": res.verdict,
                    "P(fake)": round(res.p_fake, 3),
                    "Confidence": None if res.confidence is None else round(res.confidence, 3),
                    "Path": res.tier, "Calibrated": res.calibrated,
                    "Strongest evidence": top["name"] if top else "",
                    "Warnings": "; ".join(res.warnings)[:200],
                })
            else:
                rows.append({"File": f.name, "Type": media or "unsupported", "Verdict": "error"})
            bar.progress((i + 1) / len(files))
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("📥 Download CSV", df.to_csv(index=False), "batch_results.csv", "text/csv")

with tab_compare:
    st.markdown('<div class="section-title">🆚 Side-by-side inspector</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    fa = c1.file_uploader("Image A (reference)", type=["jpg", "jpeg", "png", "webp"], key="file_a")
    fb = c2.file_uploader("Image B (suspect)", type=["jpg", "jpeg", "png", "webp"], key="file_b")
    if fa is not None and fb is not None:
        with st.spinner("🔬 Comparing..."):
            ra = analyze(fa.getvalue(), "image", fa.name)
            rb = analyze(fb.getvalue(), "image", fb.name)
        for col, r, name in ((c1, ra, fa.name), (c2, rb, fb.name)):
            with col:
                if r is None:
                    continue
                d = r.to_dict()
                render_result_card(d)
                st.image(r.artifacts["annotated"], caption=name, use_column_width=True)
                if r.artifacts.get("heatmap") is not None:
                    st.image(r.artifacts["heatmap"], caption="Occlusion heatmap",
                             use_column_width=True)
                st.caption(d["narrative"])

with tab_lab:
    st.markdown('<div class="section-title">🔬 Classic forensic views (educational)</div>',
                unsafe_allow_html=True)
    st.info("ELA and FFT are *visual aids*: they do not feed the verdict by themselves and "
            "can mislead on re-compressed images. The decision uses the evidence table.")
    lab = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="lab_img")
    if lab is not None:
        im = Image.open(lab).convert("RGB")
        q = st.slider("JPEG quality (ELA)", 50, 98, 90)
        s = st.slider("ELA brightness scale", 5, 30, 15)
        a, b, c = st.columns(3)
        a.image(im, caption="Original", use_column_width=True)
        b.image(generate_ela_image(im, quality=q, scale=s), caption=f"ELA (Q={q})",
                use_column_width=True)
        c.image(generate_fft_spectrum(im), caption="2D FFT magnitude",
                use_column_width=True)

with tab_report:
    st.markdown('<div class="section-title">📄 Signed forensic report</div>', unsafe_allow_html=True)
    latest = st.session_state.get("latest")
    if not latest:
        st.warning("No analysis yet. Scan a file in the Single File Scanner first.")
    else:
        d, arts = latest["result"], latest["artifacts"]
        st.markdown(f"**File:** `{d['filename']}` ({d['media_type']}) · **Verdict:** `{d['verdict']}` "
                    f"· **P(fake):** `{d['p_fake'] * 100:.1f}%`")
        payload = build_payload(d)
        signed = {"payload": payload, "signature": sign_payload(payload)}
        html = generate_html_report(d, arts, signed)
        c1, c2 = st.columns(2)
        c1.download_button("📥 HTML report", html, f"forensic_report_{d['filename']}.html",
                           "text/html", type="primary")
        c2.download_button("📥 Signed JSON report", json.dumps(signed, indent=2),
                           f"forensic_report_{d['filename']}.json", "application/json")
        st.caption("The JSON is signed (Ed25519). Check it later with "
                   "`python -m scripts.verify_report <file>.json`.")
        components.html(html, height=520, scrolling=True)

st.markdown("---")
st.caption("Built by OGmayur. Reports support human review; they are not proof. "
           "Detectors can be wrong, especially on generators they were not calibrated on.")
