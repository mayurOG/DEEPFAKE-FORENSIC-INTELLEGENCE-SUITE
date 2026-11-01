"""Streamlit rendering of an AnalysisResult — v4.0 Next-Generation.

Upgrades:
  * Generator attribution display (GAN / Diffusion / XTTS / RVC chip).
  * Diffusion-detect and optical-flow evidence shown explicitly.
  * NPR+spectral expert row annotated with version tag.
  * Evidence contribution Plotly donut chart.
  * Confidence band on timeline chart.
  * LLM narrative source indicator.
  * Stability + counterfactual metrics upgraded.
  * Better color coding for verdict-aligned rows.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.ui_components import render_result_card


def evidence_frame(res: dict) -> pd.DataFrame:
    rows = []
    for e in res.get("evidence", []):
        det = e.get("detail") or {}
        expert_ver = det.get("expert_version", "")
        name_str = e["name"]
        if expert_ver:
            name_str += f" [{expert_ver}]"
        if e.get("heuristic"):
            name_str += " (heuristic)"
        rows.append({
            "Evidence": name_str,
            "Status": e["status"],
            "Fake log-odds": e["logit"],
            "P(fake) alone": e.get("p_fake_raw"),
            "Reliability": e["reliability"],
            "Note": e.get("reason") or "",
        })
    return pd.DataFrame(rows)


def _evidence_donut(res: dict):
    """Plotly donut showing % contribution of each used expert."""
    contrib = (res.get("fusion") or {}).get("contributions") or {}
    if not contrib:
        return None
    labels = list(contrib.keys())
    vals = [abs(v) for v in contrib.values()]
    total = sum(vals) + 1e-9
    pcts = [v / total * 100 for v in vals]
    colors = ["#38bdf8", "#818cf8", "#c084fc", "#22c55e", "#f59e0b",
              "#ef4444", "#06b6d4", "#a78bfa", "#fbbf24", "#fb7185"]
    fig = go.Figure(go.Pie(
        labels=labels, values=pcts,
        hole=0.55,
        textinfo="label+percent",
        textfont_size=11,
        marker=dict(colors=colors[:len(labels)], line=dict(color="#04070f", width=2)),
    ))
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8",
        margin=dict(t=30, b=20, l=20, r=20),
        height=240,
        annotations=[dict(
            text="Evidence<br>share", x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color="#64748b"),
        )],
    )
    return fig


def _timeline_chart(res: dict):
    """Plotly line chart of per-frame P(fake) with confidence band."""
    tl = res.get("timeline") or []
    if len(tl) < 2:
        return None
    ts = [t["timestamp"] for t in tl]
    ps = [t["fake_probability"] for t in tl]
    segs = res.get("segments") or []

    fig = go.Figure()
    # Confidence band (±10% heuristic for uncalibrated, ±5% for calibrated).
    band = 0.05 if res.get("calibrated") else 0.10
    fig.add_trace(go.Scatter(
        x=ts + ts[::-1],
        y=[min(1.0, p + band) for p in ps] + [max(0.0, p - band) for p in ps[::-1]],
        fill="toself", fillcolor="rgba(56,189,248,0.08)",
        line=dict(color="rgba(56,189,248,0)"),
        name="Uncertainty band", hoverinfo="skip",
    ))
    # P(fake) line.
    fig.add_trace(go.Scatter(
        x=ts, y=ps,
        mode="lines+markers",
        line=dict(color="#38bdf8", width=2),
        marker=dict(size=5, color=[
            "#ef4444" if p > 0.5 else "#22c55e" for p in ps
        ]),
        name="P(fake)",
    ))
    # Suspicious segments.
    for seg in segs:
        fig.add_vrect(
            x0=seg["start"], x1=seg["end"],
            fillcolor="rgba(239,68,68,0.12)",
            line_width=0,
            annotation_text="⚠", annotation_position="top left",
        )
    # Threshold line.
    fig.add_hline(y=0.5, line_dash="dash", line_color="#64748b",
                  annotation_text="Decision boundary", annotation_font_size=10)
    fig.update_layout(
        xaxis_title="Time (s)", yaxis_title="P(manipulated)",
        yaxis=dict(range=[0, 1]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,14,32,0.6)",
        font_color="#94a3b8",
        xaxis=dict(gridcolor="rgba(51,65,85,0.5)"),
        yaxis_gridcolor="rgba(51,65,85,0.5)",
        margin=dict(t=20, b=40, l=50, r=20),
        height=280,
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
    )
    return fig


def _attribution_section(att: dict):
    """Show generator attribution if available."""
    if not att or not att.get("family"):
        return
    family = att["family"]
    prob = att.get("probability", 0)
    desc = att.get("description", "")
    icons = {
        "GAN": "🤖", "Diffusion": "🌊", "VAE": "🧩",
        "Face_Swap": "😶", "Inpaint": "🖌️",
        "TTS": "🎤", "Voice_Conv": "🔄", "Unknown": "❓",
    }
    icon = icons.get(family, "🔬")
    all_probs = att.get("all_probs") or {}

    with st.expander(f"{icon} Generator Attribution: **{family}** ({prob:.0%} confidence)", expanded=True):
        st.markdown(
            f"<div style='color:#a5b4fc; font-size:0.9rem; margin-bottom:8px;'>{desc}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "_This is supporting context only — it does NOT change the verdict._",
        )
        if all_probs:
            fig = px.bar(
                x=list(all_probs.keys()), y=[v * 100 for v in all_probs.values()],
                labels={"x": "Generator", "y": "Probability (%)"},
                color_discrete_sequence=["#818cf8"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(8,14,32,0.6)",
                font_color="#94a3b8",
                height=200,
                margin=dict(t=10, b=40, l=50, r=20),
                xaxis_gridcolor="rgba(51,65,85,0.5)",
                yaxis_gridcolor="rgba(51,65,85,0.5)",
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_provenance_badge(prov: dict):
    if not prov:
        return
    state = prov.get("validation_state") or ("valid" if prov.get("c2pa_valid") else "absent")
    claimed = prov.get("declared_ai_source")
    signer = (prov.get("certificate_info") or {}).get("issuer") or prov.get("note", "")

    if state == "valid":
        color = "#22c55e"
        bg = "rgba(34,197,94,0.12)"
        border = "rgba(34,197,94,0.35)"
        icon = "🛡️"
        title = "C2PA Cryptographically Verified Manifest"
        status_text = "Content credentials verified with trusted digital certificate."
    elif state == "invalid":
        color = "#ef4444"
        bg = "rgba(239,68,68,0.12)"
        border = "rgba(239,68,68,0.35)"
        icon = "🚨"
        title = "C2PA Manifest Corrupted or Tampered"
        status_text = "Cryptographic signature check failed. Possible asset modification or stripping."
    elif state == "unverifiable":
        color = "#f59e0b"
        bg = "rgba(245,158,11,0.12)"
        border = "rgba(245,158,11,0.35)"
        icon = "⚠️"
        title = "C2PA Manifest Present (Unverified Root)"
        status_text = "Manifest present but certificate chain unverified or self-signed."
    else:
        return

    signer_html = f"&nbsp;·&nbsp; <b>Signer/Issuer:</b> {signer}" if signer else ""
    claimed_html = f"<div style='color:#fca5a5; font-size:0.82rem; margin-top:3px;'><b>Declared AI Generator:</b> {claimed}</div>" if claimed else ""
    st.markdown(f"""
    <div style='background:{bg}; border:1px solid {border}; border-radius:12px; padding:12px 18px; margin-bottom:16px; display:flex; align-items:center; gap:14px;'>
        <div style='font-size:1.8rem;'>{icon}</div>
        <div>
            <div style='color:{color}; font-weight:700; font-size:1.0rem;'>{title}</div>
            <div style='color:#cbd5e1; font-size:0.85rem;'>{status_text}{signer_html}</div>
            {claimed_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_fusion_baselines(res_dict: dict, key: str):
    with st.expander("⚖️ Multi-Expert Fusion Analysis & Baseline Benchmarks", expanded=False):
        fusion = res_dict.get("fusion") or {}
        ev_list = res_dict.get("evidence", [])
        used_ev = [e for e in ev_list if e.get("status") == "used" and e.get("logit") is not None]
        if not used_ev:
            st.caption("No used evidence available for fusion benchmarking.")
            return

        z_vals = [e["logit"] for e in used_ev]
        p_vals = [1.0 / (1.0 + np.exp(-z)) for z in z_vals]
        best_ev = max(used_ev, key=lambda e: e.get("reliability", 0.5))
        best_p = 1.0 / (1.0 + np.exp(-best_ev["logit"]))
        simple_p = float(np.mean(p_vals))
        fixed_p = float(res_dict.get("p_fake", simple_p))
        gls_p = float(1.0 / (1.0 + np.exp(-fusion.get("logit", res_dict.get("tier1_logit", 0.0)))))

        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        f_col1.metric("1. Best Single Expert", f"{best_p*100:.1f}%", help=f"Expert: {best_ev['name']}")
        f_col2.metric("2. Simple Average", f"{simple_p*100:.1f}%", help="Unweighted mean of probabilities")
        f_col3.metric("3. Fixed Weighted", f"{fixed_p*100:.1f}%", help="Standard heuristic weighting")
        f_col4.metric("4. Regularized GLS (Ours)", f"{gls_p*100:.1f}%", help="Covariance-shrinkage generalized least squares")

        df_baselines = pd.DataFrame({
            "Fusion Method": ["1. Best Expert", "2. Simple Average", "3. Fixed Weighted", "4. Learned GLS (Ours)"],
            "P(fake) %": [best_p * 100, simple_p * 100, fixed_p * 100, gls_p * 100],
        })
        fig = px.bar(
            df_baselines, x="P(fake) %", y="Fusion Method", orientation="h",
            color="Fusion Method",
            color_discrete_map={
                "1. Best Expert": "#64748b",
                "2. Simple Average": "#94a3b8",
                "3. Fixed Weighted": "#818cf8",
                "4. Learned GLS (Ours)": "#38bdf8",
            }
        )
        fig.add_vline(x=50.0, line_dash="dash", line_color="#ef4444", annotation_text="Threshold")
        fig.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,14,32,0.6)",
            font_color="#94a3b8",
            height=200,
            margin=dict(t=10, b=25, l=120, r=20),
            xaxis=dict(range=[0, 100], gridcolor="rgba(51,65,85,0.4)"),
            yaxis=dict(gridcolor="rgba(51,65,85,0.4)"),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"fusion_base_{key}")


def _render_audio_forensics(res_dict: dict, artifacts: dict, key: str):
    st.markdown("### 🎙️ Audio Forensics & Voice Activity Analysis")
    audio_ev = next((e for e in res_dict.get("evidence", []) if e.get("name") == "audio"), None)
    det = (audio_ev.get("detail") or {}) if audio_ev else {}
    speech_ratio = det.get("speech_ratio")
    snr_db = det.get("mean_snr_db")
    speech_detected = det.get("speech_detected", True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Voiced Speech Ratio", f"{speech_ratio * 100:.1f}%" if speech_ratio is not None else "100.0%",
              help="Fraction of audio containing active voiced speech detected via VAD")
    m2.metric("Estimated SNR", f"{snr_db:+.1f} dB" if snr_db is not None else "n/a",
              help="Signal-to-noise ratio in decibels")
    m3.metric("Speech Segments", det.get("active_speech_chunks", det.get("chunks", "—")),
              help="Active speech chunk count")
    m4.metric("Silence Gating", "✓ Active" if speech_detected else "⚠️ Gated (Silence)",
              help="Silence / ambient noise gating to prevent false positives")

    if not speech_detected or (speech_ratio is not None and speech_ratio < 0.05):
        st.info("ℹ️ **Voice Activity Gating Active:** The audio track appears to be silence or ambient room tone. Deepfake voice scoring was gated to avoid unvoiced false-positive predictions.")

    tl = res_dict.get("timeline") or []
    if tl:
        fig = go.Figure()
        ts = [t["timestamp"] for t in tl]
        ps = [t["fake_probability"] for t in tl]
        fig.add_trace(go.Bar(
            x=ts, y=ps,
            marker_color=["#ef4444" if p >= 0.5 else "#22c55e" for p in ps],
            name="Chunk P(fake)",
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#64748b", annotation_text="Decision Boundary")
        fig.update_layout(
            title="Per-Chunk Manipulation Probability",
            xaxis_title="Time (seconds)", yaxis_title="P(manipulated)",
            yaxis=dict(range=[0, 1]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,14,32,0.6)",
            font_color="#94a3b8",
            height=240,
            margin=dict(t=35, b=30, l=40, r=20),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"audio_tl_{key}")

    if artifacts.get("mel") is not None:
        st.image(artifacts["mel"], caption="Log-Mel Frequency Spectrogram (16 kHz)", use_column_width=True)


def _render_video_temporal_details(res_dict: dict, key: str):
    temp_ev = next((e for e in res_dict.get("evidence", []) if e.get("name") == "temporal"), None)
    det = (temp_ev.get("detail") or {}) if temp_ev else {}
    n_eff = det.get("n_eff")
    n_frames = det.get("n_frames")
    rho = det.get("autocorr_lag1")
    anomaly = det.get("anomaly_detected", False)

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Sampled Frames", n_frames if n_frames is not None else "—")
    v2.metric("Effective Sample Size (N_eff)", f"{n_eff:.1f}" if n_eff is not None else "—",
              help="Autocorrelation-adjusted independent sample information")
    v3.metric("Temporal Autocorrelation (ρ₁)", f"{rho:+.2f}" if rho is not None else "—",
              help="Frame-to-frame correlation. High = redundant consecutive frames")
    v4.metric("Temporal Anomaly Burst", "🚨 Detected" if anomaly else "✓ Clean",
              help="Targeted manipulation confined to specific frame interval")


def render_result(res_dict: dict, artifacts: dict, key: str = "r"):
    """Full result view used by the single-file scanner — v4.0."""
    render_result_card(res_dict)
    media = res_dict.get("media_type")
    q = res_dict.get("quality", {})
    expl = res_dict.get("explanation") or {}
    st_ = res_dict.get("stability") or {}
    un = res_dict.get("uncertainty") or {}
    cf = expl.get("counterfactual") or {}

    # ── Cryptographic Provenance Badge ─────────────────────────────────────
    _render_provenance_badge(res_dict.get("provenance"))

    # ── Inference Resource & Cache Status ──────────────────────────────────
    res_usage = res_dict.get("resource_usage") or {}
    if res_usage:
        lat = res_usage.get("latency_ms", 0.0)
        peak_mb = res_usage.get("peak_memory_mb", 0.0)
        dev = res_usage.get("device", "cpu")
        hit = res_usage.get("cache_hit", False)
        badge_color = "#10b981" if hit else "#38bdf8"
        hit_label = "⚡ Cache Hit" if hit else "⏱️ Fresh Inference"
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(56, 189, 248, 0.2);
                        border-radius: 8px; padding: 6px 14px; margin-bottom: 12px; display: flex;
                        justify-content: space-between; align-items: center; font-size: 0.8rem; color: #94a3b8;">
                <div><span style="color: {badge_color}; font-weight: 700;">{hit_label}</span> · Latency: <b style="color: #f1f5f9;">{lat:.1f} ms</b></div>
                <div>Peak RAM: <b style="color: #f1f5f9;">{peak_mb:.1f} MB</b> · Device: <span style="color: #38bdf8;">{dev}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Top metrics row ───────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verdict", res_dict["verdict"])
    c2.metric("P(manipulated)", f"{res_dict['p_fake'] * 100:.1f}%")
    c3.metric(
        "Re-encoding stability",
        f"{st_['stability_score'] * 100:.0f}%" if st_.get("stability_score") is not None else "n/a",
    )
    c4.metric(
        "Heatmap faithfulness",
        f"{expl['faithfulness']:+.2f}" if expl.get("faithfulness") is not None else "n/a",
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric(
        "View consistency",
        f"{un['view_spread']:.2f}" if un.get("view_spread") is not None else "n/a",
        help="Spread across mirror/crop TTA views. High = fragile evidence.",
    )
    d2.metric(
        "Flip region",
        f"{cf['flip_fraction'] * 100:.0f}%" if cf.get("flip_fraction") is not None else "n/a",
        help="Smallest image fraction whose removal drops the fake score below threshold.",
    )
    d3.metric(
        "Sufficient region",
        f"{cf['sufficient_fraction'] * 100:.0f}%" if cf.get("sufficient_fraction") is not None else "n/a",
        help="Smallest image fraction that alone keeps the fake score above threshold.",
    )
    cal_text = "✓ Yes (α=0.05)" if res_dict.get("calibrated") else "✗ No"
    d4.metric("Calibrated", cal_text, help="Conformal prediction calibrated at 95% coverage.")

    # ── Narrative ─────────────────────────────────────────────────────────
    with st.expander("🧾 Forensic narrative", expanded=True):
        narrative = res_dict.get("narrative", "")
        if expl.get("narrative_source") == "llm":
            st.markdown(
                "<span style='font-size:0.78rem; color:#818cf8; font-weight:600;'>"
                "✨ Generated by LLM (GPT-4o / Gemini)</span>",
                unsafe_allow_html=True,
            )
        st.markdown(narrative)

    # ── Generator Attribution ─────────────────────────────────────────────
    att = res_dict.get("attribution")
    if att:
        _attribution_section(att)

    # ── Evidence table + donut ────────────────────────────────────────────
    st.markdown("### 🧩 Evidence behind the verdict")
    ev_col, donut_col = st.columns([3, 1])
    with ev_col:
        df_ev = evidence_frame(res_dict)
        if not df_ev.empty:
            st.dataframe(df_ev, use_container_width=True, hide_index=True)
    with donut_col:
        donut_fig = _evidence_donut(res_dict)
        if donut_fig:
            st.plotly_chart(donut_fig, use_container_width=True, key=f"donut_{key}")
        else:
            st.caption("Contribution chart\nunavailable.")

    # ── Multi-Expert Fusion Baselines ─────────────────────────────────────
    _render_fusion_baselines(res_dict, key)

    # ── AUDIO: Forensics & VAD ────────────────────────────────────────────
    if media == "audio":
        _render_audio_forensics(res_dict, artifacts, key)

    # ── IMAGE: Advanced Forensic Visual Analysis ──────────────────────────
    if media == "image":
        st.markdown("### 🖼️ Advanced Forensic Visual Analysis")

        # Top row: Annotated + Occlusion Heatmap + Tamper map.
        vis_cols = st.columns(3)
        if artifacts.get("annotated") is not None:

            vis_cols[0].image(artifacts["annotated"], caption="👤 Face detection overlay",
                              use_column_width=True)
        if artifacts.get("heatmap") is not None:
            vis_cols[1].image(artifacts["heatmap"], use_column_width=True,
                              caption="🔥 Occlusion heatmap: red → fake evidence")
        if artifacts.get("tamper_map") is not None:
            vis_cols[2].image(artifacts["tamper_map"], use_column_width=True,
                              caption="🪬 SRM noise-inconsistency localization")

        st.markdown(
            "<div style='font-size:0.8rem; color:#94a3b8; background:rgba(30,41,59,0.3); "
            "border-left:3px solid #38bdf8; padding:8px 12px; margin:8px 0 16px 0; border-radius:4px;'>"
            "<b>Forensic Notice:</b> Occlusion heatmaps identify spatial regions influencing the detector's "
            "prediction; they do not necessarily indicate the exact pixels altered by an attacker."
            "</div>",
            unsafe_allow_html=True,
        )

        src_img = artifacts.get("source_image")


        # 6-Panel forensic grid.
        st.markdown("#### 🔬 6-Panel Forensic Grid")
        st.caption(
            "**1** Original · **2** Face Mesh (468 landmarks) · **3** ELA residual · "
            "**4** Blending Boundary (swap edges) · **5** SRM Noise · **6** L/R Asymmetry diff"
        )
        if src_img is not None:
            try:
                from utils.advanced_visualizers import forensic_grid
                grid_img = forensic_grid(src_img)
                st.image(grid_img, use_column_width=True)
            except Exception as _ge:
                st.caption(f"Forensic grid: {_ge}")

        # Face region heatmap + 3D FFT surface.
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            if src_img is not None:
                try:
                    from utils.advanced_visualizers import face_region_heatmap
                    st.plotly_chart(face_region_heatmap(src_img),
                                   use_container_width=True, key=f"rhm_{key}")
                except Exception as _e:
                    st.caption(f"Region heatmap: {_e}")
        with r1c2:
            if src_img is not None:
                try:
                    from utils.advanced_visualizers import fft_3d_surface
                    st.plotly_chart(fft_3d_surface(src_img),
                                   use_container_width=True, key=f"fft3_{key}")
                except Exception as _e:
                    st.caption(f"3D FFT: {_e}")

        # NPR direction radar + Frequency radar.
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            if src_img is not None:
                try:
                    from utils.advanced_visualizers import npr_direction_chart
                    st.plotly_chart(npr_direction_chart(src_img),
                                   use_container_width=True, key=f"npr_{key}")
                except Exception as _e:
                    st.caption(f"NPR radar: {_e}")
        with r2c2:
            if src_img is not None:
                try:
                    from utils.advanced_visualizers import frequency_radar
                    st.plotly_chart(frequency_radar(src_img),
                                   use_container_width=True, key=f"frad_{key}")
                except Exception as _e:
                    st.caption(f"Frequency radar: {_e}")

        # Half-face asymmetry + compression map (in expander).
        with st.expander("🧬 Half-Face Asymmetry + Compression Artifact Map", expanded=False):
            cx1, cx2 = st.columns(2)
            with cx1:
                if src_img is not None:
                    try:
                        from utils.advanced_visualizers import half_face_asymmetry
                        st.image(half_face_asymmetry(src_img), use_column_width=True,
                                 caption="Left | Asymmetry Diff | Right(mirrored) — bright = swap boundary")
                    except Exception as _e:
                        st.caption(f"Asymmetry: {_e}")
            with cx2:
                if src_img is not None:
                    try:
                        from utils.advanced_visualizers import compression_artifact_map
                        st.image(compression_artifact_map(src_img), use_column_width=True,
                                 caption="Block-DCT variance — AI images = uniform blocks")
                    except Exception as _e:
                        st.caption(f"Compression map: {_e}")

        # Classic ELA + FFT.
        with st.expander("📐 Classic forensic views (ELA, 2D FFT)", expanded=False):
            c = st.columns(2)
            if artifacts.get("ela") is not None:
                c[0].image(artifacts["ela"], caption="Error Level Analysis", use_column_width=True)
            if artifacts.get("fft") is not None:
                c[1].image(artifacts["fft"], caption="2D FFT magnitude spectrum", use_column_width=True)

    # ── VIDEO: Timeline + Suspicious Frame Contact Sheet ──────────────────
    if media == "video":
        _render_video_temporal_details(res_dict, key)
        tl_fig = _timeline_chart(res_dict)
        if tl_fig:
            st.markdown("### 📈 Manipulation Probability Timeline")
            st.plotly_chart(tl_fig, use_container_width=True, key=f"tl_{key}")

            segs = res_dict.get("segments") or []
            if segs:
                st.markdown(
                    "**⚠ Suspicious segments:** " +
                    ", ".join(f"{s['start']:.1f}s–{s['end']:.1f}s" for s in segs)
                )
        # Suspicious frame contact sheet.
        suspicious = artifacts.get("suspicious_frames") or []
        if suspicious:
            st.markdown("### 🎬 Suspicious Frame Contact Sheet")
            try:
                from utils.advanced_visualizers import suspicious_frame_contact_sheet
                tl_lookup = {round(t["timestamp"], 1): t["fake_probability"]
                             for t in (res_dict.get("timeline") or [])}
                sheet_data = [(ts, frm, tl_lookup.get(round(ts, 1), 0.5))
                              for ts, frm in suspicious[:8]]
                sheet = suspicious_frame_contact_sheet(sheet_data)
                st.image(sheet, use_column_width=True,
                         caption="Ranked by P(fake) — border: 🔴 high risk → 🟢 low risk")
            except Exception as _se:
                frame_cols = st.columns(min(len(suspicious), 4))
                for i, (t, im) in enumerate(suspicious[:4]):
                    frame_cols[i % 4].image(im, caption=f"t={t:.2f}s", use_column_width=True)

    # ── Quality + Provenance ──────────────────────────────────────────────
    with st.expander("📊 Media quality & provenance", expanded=False):
        if q:
            q_rows = [{k: str(v) for k, v in q.items() if v is not None and k != "notes"}]
            st.dataframe(pd.DataFrame(q_rows), use_container_width=True, hide_index=True)
        prov = res_dict.get("provenance") or {}
        if prov:
            st.markdown(f"**Provenance:** {prov.get('note', 'n/a')}")
            if prov.get("c2pa_valid"):
                st.success("✓ C2PA manifest verified — content origin authenticated.")
            elif prov.get("c2pa_valid") is False:
                st.error("⚠ C2PA manifest present but INVALID — possible tampering.")

    # ── Grounded claims ───────────────────────────────────────────────────
    grounded = res_dict.get("grounded") or {}
    claims = grounded.get("claims") or []
    if claims:
        with st.expander("🔍 Grounded forensic claims", expanded=False):
            for c in claims:
                icon = "🟠" if c.get("level") == "strong" else "🔵"
                st.markdown(f"{icon} {c.get('text', '')}")

    # ── Stability ─────────────────────────────────────────────────────────
    if st_ and st_.get("stability_score") is not None:
        with st.expander("🔁 Re-encoding stability report", expanded=False):
            sc = st_.get("stability_score", 0.0)
            color = "#22c55e" if sc >= 0.8 else "#eab308" if sc >= 0.5 else "#ef4444"
            st.markdown(
                f"<span style='color:{color}; font-size:1.1rem; font-weight:700;'>"
                f"{sc * 100:.0f}% of degraded copies agreed with the verdict.</span>",
                unsafe_allow_html=True,
            )
            flips = st_.get("flips") or []
            if flips:
                st.warning(f"Verdict flipped under: {', '.join(flips)}")
            pmin, pmax = st_.get("p_min"), st_.get("p_max")
            if pmin is not None and pmax is not None:
                st.caption(f"P(fake) range across perturbations: {pmin:.2f} – {pmax:.2f}")

    # ── OOD flag ─────────────────────────────────────────────────────────
    if res_dict.get("ood"):
        st.warning(
            "⚠ **Out-of-distribution detected**: this input differs from the calibration "
            "data. Probability estimates may be less reliable than usual.",
        )

    # ── Warnings ─────────────────────────────────────────────────────────
    warns = res_dict.get("warnings") or []
    if warns:
        with st.expander(f"⚠ {len(warns)} forensic warning(s)", expanded=False):
            for w in warns:
                st.warning(w)

    # ── Raw evidence JSON ─────────────────────────────────────────────────
    with st.expander("🔧 Raw evidence JSON (debug)", expanded=False):
        import json
        st.code(json.dumps(res_dict.get("evidence", []), indent=2), language="json")
