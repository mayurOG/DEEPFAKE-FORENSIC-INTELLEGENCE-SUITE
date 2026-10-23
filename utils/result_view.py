"""Streamlit rendering of an AnalysisResult (evidence, stability, heatmaps)."""
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.ui_components import render_result_card


def evidence_frame(res: dict) -> pd.DataFrame:
    rows = []
    for e in res.get("evidence", []):
        rows.append({
            "Evidence": e["name"] + (" (heuristic)" if e.get("heuristic") else ""),
            "Status": e["status"],
            "Fake log-odds": e["logit"],
            "P(fake) alone": e["p_fake_raw"],
            "Reliability": e["reliability"],
            "Note": e.get("reason") or "",
        })
    return pd.DataFrame(rows)


def render_result(res_dict: dict, artifacts: dict, key: str = "r"):
    """Full result view used by the single-file scanner."""
    render_result_card(res_dict)
    media = res_dict.get("media_type")
    q = res_dict.get("quality", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verdict", res_dict["verdict"])
    c2.metric("P(manipulated)", f"{res_dict['p_fake'] * 100:.1f}%")
    st_ = res_dict.get("stability") or {}
    c3.metric("Re-encoding stability",
              f"{st_['stability_score'] * 100:.0f}%" if st_.get("stability_score") is not None else "n/a")
    ex = res_dict.get("explanation") or {}
    c4.metric("Heatmap faithfulness",
              f"{ex['faithfulness']:+.2f}" if ex.get("faithfulness") is not None else "n/a")

    un = res_dict.get("uncertainty") or {}
    cf = ex.get("counterfactual") or {}
    d1, d2, d3 = st.columns(3)
    d1.metric("View consistency (spread)",
              f"{un['view_spread']:.2f}" if un.get("view_spread") is not None else "n/a",
              help="Spread of the score across mirror/crop views. High = fragile evidence.")
    d2.metric("Region that flips the call",
              f"{cf['flip_fraction'] * 100:.0f}%" if cf.get("flip_fraction") is not None else "n/a",
              help="Smallest share of the image whose removal drops the 'fake' score.")
    d3.metric("Region that is enough alone",
              f"{cf['sufficient_fraction'] * 100:.0f}%" if cf.get("sufficient_fraction") is not None else "n/a",
              help="Smallest share of the image that alone keeps the 'fake' score.")

    with st.expander("🧾 Plain-language summary", expanded=True):
        st.text(res_dict.get("narrative", ""))

    st.markdown("### 🧩 Evidence behind the verdict")
    st.dataframe(evidence_frame(res_dict), use_container_width=True, hide_index=True)
    contrib = (res_dict.get("fusion") or {}).get("contributions")
    if contrib:
        st.caption("Share of the fused decision per source: " +
                   ", ".join(f"{k} {v * 100:.0f}%" for k, v in contrib.items()))

    if media == "image":
        st.markdown("### 🖼️ Visual evidence")
        cols = st.columns(3)
        if artifacts.get("annotated") is not None:
            cols[0].image(artifacts["annotated"], caption="Detected faces",
                          use_column_width=True)
        if artifacts.get("heatmap") is not None:
            cols[1].image(artifacts["heatmap"], use_column_width=True,
                          caption="Occlusion heatmap: red regions push the score towards 'fake'")
        if artifacts.get("tamper_map") is not None:
            cols[2].image(artifacts["tamper_map"], use_column_width=True,
                          caption="Noise-inconsistency map (heuristic)")
        with st.expander("Classic forensic views (ELA, FFT)"):
            c = st.columns(2)
            if artifacts.get("ela") is not None:
                c[0].image(artifacts["ela"], caption="Error Level Analysis",
                           use_column_width=True)
            if artifacts.get("fft") is not None:
                c[1].image(artifacts["fft"], caption="2D FFT spectrum",
                           use_column_width=True)
        if artifacts.get("face_crops"):
            cols = st.columns(min(len(artifacts["face_crops"]), 4))
            for i, im in enumerate(artifacts["face_crops"]):
                cols[i % 4].image(im, caption=f"Face region #{i + 1}",
                                  use_column_width=True)

    if media == "video":
        tl = res_dict.get("timeline") or []
        if tl:
            st.markdown("### 📈 Per-frame manipulation probability")
            df = pd.DataFrame(tl)
            fig = px.line(df, x="timestamp", y="fake_probability", markers=True,
                          labels={"timestamp": "Time (s)", "fake_probability": "P(fake)"})
            fig.add_hline(y=0.5, line_dash="dash", line_color="#94a3b8")
            fig.update_yaxes(range=[0, 1])
            fig.update_layout(paper_bgcolor="rgba(15,23,42,0.8)",
                              plot_bgcolor="rgba(15,23,42,0.8)", font_color="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True, key=f"{key}_tl")
        segs = [s for s in res_dict.get("segments", []) if "end" in s and "logit" not in s and "fake_probability" not in s]
        if segs:
            st.info("Suspicious time ranges: " +
                    ", ".join(f"{s['start']:.1f}s-{s['end']:.1f}s" for s in segs))
        sus = artifacts.get("suspicious_frames") or []
        if sus:
            st.markdown("### 🖼️ Most suspicious sampled frames")
            cols = st.columns(min(len(sus), 4))
            for i, (t, im) in enumerate(sus):
                cols[i % 4].image(im, caption=f"t = {t:.1f}s", use_column_width=True)
        if artifacts.get("heatmap") is not None:
            st.image(artifacts["heatmap"], width=360,
                     caption="Occlusion heatmap of the most suspicious face")

    if media == "audio":
        tl = res_dict.get("timeline") or []
        if artifacts.get("mel") is not None:
            st.image(artifacts["mel"], use_column_width=True, caption="Mel spectrogram")
        if tl:
            df = pd.DataFrame(tl)
            fig = px.line(df, x="timestamp", y="fake_probability", markers=True,
                          labels={"timestamp": "Time (s)", "fake_probability": "P(synthetic)"})
            fig.add_hline(y=0.5, line_dash="dash", line_color="#94a3b8")
            fig.update_yaxes(range=[0, 1])
            fig.update_layout(paper_bgcolor="rgba(15,23,42,0.8)",
                              plot_bgcolor="rgba(15,23,42,0.8)", font_color="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True, key=f"{key}_tl")

    _render_egafr(res_dict, key)

    with st.expander("📏 Media quality and provenance"):
        st.json({"quality": q, "provenance": res_dict.get("provenance"),
                 "sha256": res_dict.get("sha256")})
    meta = res_dict.get("metadata")
    if meta:
        with st.expander("📄 File metadata (EXIF / headers)"):
            for w in meta.get("warnings", []):
                st.error(w)
            for n in meta.get("notes", []):
                st.caption(n)
            st.dataframe(pd.DataFrame([{"Property": k, "Value": str(v)} for k, v in meta.items()
                                       if k not in ("exif_data", "warnings", "notes")]),
                         use_container_width=True, hide_index=True)
    with st.expander("⏱️ Timings"):
        st.json(res_dict.get("timings", {}))


def _render_egafr(res: dict, key: str):
    """Evidence graph, investigation log, robustness, contradictions, audit."""
    g = res.get("graph") or {}
    gf = g.get("fusion") or {}
    if res.get("abstention_reason"):
        st.warning(f"Abstention reason: {res['abstention_reason']}")
    att = res.get("attribution")
    if att:
        st.info(f"Likely generator family: **{att['family']}** "
                f"(p={att['probability']:.2f}; supporting context only, never changes the verdict).")
    prov = ((res.get("metadata") or {}).get("provenance")) or {}
    if prov.get("c2pa_present") or prov.get("declared_ai_source"):
        st.caption("Provenance: " + prov.get("note", "") +
                   (f" Declared source: {prov['declared_ai_source']}." if prov.get("declared_ai_source") else ""))
    claims = (res.get("grounded") or {}).get("claims") or []
    if claims:
        with st.expander("Evidence-grounded claims (every statement is tied to recorded evidence)"):
            for c in claims:
                st.write(f"- `{c['id']}`: {c['text']}")
    if gf.get("n_used"):
        with st.expander("🕸️ Evidence graph: what actually drives the decision", expanded=False):
            c = st.columns(3)
            c[0].metric("Evidence sufficiency", f"{gf.get('sufficiency', 0):.2f}",
                        help="z-test that the fused log-odds differs from zero, inflated when "
                             "sources disagree. Separate from classifier confidence.")
            c[1].metric("Independent sources", f"{gf.get('n_eff', 0):.1f} of {gf['n_used']}",
                        help="Correlated experts count for less than one each.")
            c[2].metric("Fused log-odds", f"{gf.get('logit', 0):+.2f}")
            cfs = g.get("counterfactuals") or []
            if cfs:
                st.markdown("**If this source were removed** (leave-one-out):")
                st.dataframe(pd.DataFrame([{
                    "Removed": c_["removed"], "Logit without": c_.get("logit_without"),
                    "Change": c_.get("delta"), "Flips the call": c_.get("flips_decision"),
                } for c_ in cfs]), use_container_width=True, hide_index=True)
            for c_ in g.get("contradictions") or []:
                st.error(f"Contradiction: {c_['experts'][0]} vs {c_['experts'][1]} "
                         f"(logits {c_['logits'][0]:+.1f} / {c_['logits'][1]:+.1f})")
            for s_ in g.get("shared_failure_modes") or []:
                st.caption(f"{s_['experts'][0]} and {s_['experts'][1]} may fail together "
                           f"(rho {s_['rho']:.2f}); agreement counts for less.")
            nodes, edges = g.get("nodes", []), g.get("edges", [])
            dot = ["digraph evidence {", "rankdir=LR;"]
            dot += [f'"{n["id"]}" [label="{n["label"]}"];' for n in nodes]
            dot += [f'"{e["src"]}" -> "{e["dst"]}" [label="{e["type"]}"];' for e in edges]
            dot.append("}")
            st.markdown("**Evidence graph visualization**")
            try:
                st.graphviz_chart("\n".join(dot))
                st.code("\n".join(dot), language="dot")
            except Exception:
                st.code("\n".join(dot), language="dot")
    inv = res.get("investigation") or {}
    if inv.get("policy"):
        with st.expander("🧭 Investigation: why each test ran (or did not)"):
            st.caption(f"Policy {inv['policy']}, budget {inv['budget']}, spent {inv['spent']:.2f}; "
                       f"stopped: {inv['stop_reason'].replace('_', ' ')}; "
                       f"final risk {inv['final_risk']:.3f} (target {inv['risk_target']}).")
            if inv.get("steps"):
                st.dataframe(pd.DataFrame(inv["steps"]), use_container_width=True,
                             hide_index=True)
            if inv.get("not_run"):
                st.dataframe(pd.DataFrame(inv["not_run"]), use_container_width=True,
                             hide_index=True)
    rb = res.get("robustness")
    if rb:
        with st.expander(f"🧪 Transformation robustness (fragility {rb['fragility']:.2f})"):
            st.dataframe(pd.DataFrame(rb["rows"]), use_container_width=True,
                         hide_index=True)
            gate = rb.get("gate") or {}
            if gate.get("ok") is False:
                st.warning(gate.get("reason"))
    cm = res.get("contradiction_map")
    if cm and cm.get("status") in ("ok", "no_reference"):
        with st.expander("🎙️ Audio-visual contradiction map"):
            if cm.get("intervals"):
                for i in cm["intervals"]:
                    st.error(f"Audio and lip motion disagree {i['t0']:.1f}s - {i['t1']:.1f}s "
                             f"(mean correlation {i['mean_corr']:.2f}, min p {i['min_p']:.2f})")
            else:
                st.success("No localized disagreement."
                           + (" A constant offset was treated as benign dubbing/latency."
                              if cm.get("benign_latency") else ""))
            wins = [w for w in cm.get("windows", [])]
            if wins:
                df = pd.DataFrame(wins)
                st.dataframe(df, use_container_width=True, hide_index=True)
    au = res.get("audit") or {}
    if au.get("entries"):
        with st.expander(f"🔗 Chain of custody ({len(au['entries'])} entries, head {au['head'][:12]}…)"):
            sm = au.get("summary", {})
            st.caption("verified = computed here; declared = asserted by the file, not validated; "
                       "inferred = deduced from signal evidence; unknown = nothing known. "
                       f"Counts: {sm}")
            st.dataframe(pd.DataFrame([{"#": e["index"], "kind": e["kind"], "status": e["status"],
                                        "hash": e["hash"][:12]} for e in au["entries"]]),
                         use_container_width=True, hide_index=True)
