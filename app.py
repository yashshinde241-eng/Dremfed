from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import torch

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title     = "DermFed · Federated Skin Cancer Detection",
    page_icon      = "🔬",
    layout         = "wide",
    initial_sidebar_state = "expanded",
)

from utils import (
    CLASS_COLORS,
    CLASS_NAMES,
    DEVICE,
    EVAL_TRANSFORM,
    build_model,
)
from explainability import explain_prediction
from tee_engine import get_engine

# ── Paths ─────────────────────────────────────────────────────────────────────
# Anchored to this file so paths work regardless of working directory
_HERE             = Path(__file__).parent.resolve()
RESULTS_DIR       = _HERE / "results"
METRICS_CSV       = RESULTS_DIR / "fl_metrics.csv"
GLOBAL_MODEL_PATH = _HERE / "models" / "global_model.pt"

# ── Styling ───────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        /* ── Root ─────────────────────────────────────────────── */
        :root {
            --bg-dark:      #0a0e1a;
            --bg-card:      #111827;
            --bg-card2:     #1a2235;
            --accent-cyan:  #00d4ff;
            --accent-green: #00ff88;
            --accent-pink:  #ff4d8f;
            --accent-amber: #ffb347;
            --text-primary: #e8edf5;
            --text-muted:   #7a8599;
            --border:       rgba(0, 212, 255, 0.15);
            --radius:       12px;
        }

        /* ── Global ───────────────────────────────────────────── */
        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif !important;
            background-color: var(--bg-dark) !important;
            color: var(--text-primary) !important;
        }
        .stApp { background-color: var(--bg-dark) !important; }

        /* ── Sidebar ──────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: var(--bg-card) !important;
            border-right: 1px solid var(--border) !important;
        }
        [data-testid="stSidebar"] .stMarkdown p { color: var(--text-muted) !important; }

        /* ── Cards ────────────────────────────────────────────── */
        .derm-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        .derm-card-accent {
            background: linear-gradient(135deg, #0d1b2e 0%, #0a1628 100%);
            border: 1px solid var(--accent-cyan);
            border-radius: var(--radius);
            padding: 1.5rem;
            box-shadow: 0 0 24px rgba(0, 212, 255, 0.08);
        }

        /* ── Hero ─────────────────────────────────────────────── */
        .hero-banner {
            background: linear-gradient(135deg, #0d1b2e 0%, #111827 50%, #0a1628 100%);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2.5rem 3rem;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }
        .hero-banner::before {
            content: '';
            position: absolute; top: -50%; right: -10%;
            width: 400px; height: 400px;
            background: radial-gradient(circle, rgba(0,212,255,0.06) 0%, transparent 70%);
            border-radius: 50%;
        }
        .hero-title {
            font-size: 2.4rem; font-weight: 700;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-green));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 0.4rem 0;
        }
        .hero-sub {
            color: var(--text-muted); font-size: 1rem; font-weight: 400;
            margin: 0;
        }

        /* ── Metric cards ─────────────────────────────────────── */
        .metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
        .metric-box {
            flex: 1; min-width: 140px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.2rem 1.4rem;
            text-align: center;
        }
        .metric-box .val {
            font-size: 2rem; font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        .metric-box .lbl { color: var(--text-muted); font-size: 0.75rem; margin-top: 0.2rem; }
        .cyan  { color: var(--accent-cyan)  !important; }
        .green { color: var(--accent-green) !important; }
        .pink  { color: var(--accent-pink)  !important; }
        .amber { color: var(--accent-amber) !important; }

        /* ── Status badge ─────────────────────────────────────── */
        .badge {
            display: inline-block; padding: 0.2rem 0.75rem;
            border-radius: 99px; font-size: 0.75rem; font-weight: 600;
        }
        .badge-running { background: rgba(0,212,255,0.15); color: var(--accent-cyan); }
        .badge-done    { background: rgba(0,255,136,0.12); color: var(--accent-green); }
        .badge-idle    { background: rgba(122,133,153,0.2); color: var(--text-muted); }

        /* ── Confidence bars ──────────────────────────────────── */
        .conf-row { margin: 0.45rem 0; }
        .conf-label {
            display: flex; justify-content: space-between;
            font-size: 0.78rem; color: var(--text-muted);
            margin-bottom: 3px;
        }
        .conf-track {
            background: rgba(255,255,255,0.05);
            border-radius: 99px; height: 8px;
        }
        .conf-fill {
            height: 8px; border-radius: 99px;
            transition: width 0.4s ease;
        }

        /* ── Prediction result box ────────────────────────────── */
        .pred-result {
            background: linear-gradient(135deg, #0d1b2e, #0a1628);
            border: 2px solid var(--accent-cyan);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 0 32px rgba(0, 212, 255, 0.1);
        }
        .pred-class { font-size: 1.6rem; font-weight: 700; color: var(--accent-cyan); }
        .pred-conf  { font-size: 1rem;   color: var(--text-muted); margin-top: 0.3rem; }

        /* ── Hospital node animation ──────────────────────────── */
        .hospital-grid { display: flex; gap: 1rem; flex-wrap: wrap; }
        .hospital-node {
            flex: 1; min-width: 120px;
            background: var(--bg-card2);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1rem;
            text-align: center;
            transition: border-color 0.3s;
        }
        .hospital-node.active { border-color: var(--accent-cyan); }
        .hospital-node .icon { font-size: 1.8rem; }
        .hospital-node .name { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.3rem; }

        /* ── Tabs ──────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            background: var(--bg-card) !important;
            border-bottom: 1px solid var(--border) !important;
            border-radius: var(--radius) var(--radius) 0 0 !important;
            gap: 0 !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 500 !important;
            color: var(--text-muted) !important;
            border-radius: 0 !important;
            padding: 0.75rem 1.5rem !important;
        }
        .stTabs [aria-selected="true"] {
            color: var(--accent-cyan) !important;
            border-bottom: 2px solid var(--accent-cyan) !important;
            background: transparent !important;
        }

        /* ── Streamlit overrides ──────────────────────────────── */
        .stButton > button {
            background: linear-gradient(135deg, #00aacc, #0099aa) !important;
            color: white !important; font-weight: 600 !important;
            border: none !important; border-radius: 8px !important;
            padding: 0.6rem 1.6rem !important;
            font-family: 'Space Grotesk', sans-serif !important;
            transition: opacity 0.2s !important;
        }
        .stButton > button:hover { opacity: 0.88 !important; }

        .stFileUploader {
            background: var(--bg-card) !important;
            border: 1px dashed var(--border) !important;
            border-radius: var(--radius) !important;
        }

        div[data-testid="stMetric"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            padding: 0.9rem 1rem !important;
        }
        div[data-testid="stMetricValue"] { color: var(--accent-cyan) !important; }

        h1, h2, h3, h4 { color: var(--text-primary) !important; }
        hr { border-color: var(--border) !important; }

        /* ── Plotly theming ───────────────────────────────────── */
        .js-plotly-plot .plotly .main-svg { background: transparent !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Helper components ─────────────────────────────────────────────────────────
def hero() -> None:
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🔬 DermFed</div>
            <p class="hero-sub">
                Privacy-Preserving Federated AI &nbsp;·&nbsp; Skin Lesion Analysis &nbsp;·&nbsp; MobileNetV2
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_bars(probs: list[float]) -> None:
    bars_html = ""
    for i, (cls, prob, color) in enumerate(zip(CLASS_NAMES, probs, CLASS_COLORS)):
        pct = prob * 100
        bars_html += f"""
        <div class="conf-row">
            <div class="conf-label">
                <span>{cls}</span>
                <span>{pct:.1f}%</span>
            </div>
            <div class="conf-track">
                <div class="conf-fill"
                     style="width:{pct:.1f}%; background:{color}; opacity:0.85;">
                </div>
            </div>
        </div>
        """
    st.markdown(bars_html, unsafe_allow_html=True)


# ── Tab A: Simulation ─────────────────────────────────────────────────────────
def tab_simulation() -> None:
    import plotly.graph_objects as go

    st.markdown("### 🏥 Federated Learning Simulation")

    col_info, col_ctrl = st.columns([3, 1])
    with col_info:
        st.markdown(
            """
            <div class="derm-card">
            Multiple hospitals collaboratively train a shared skin-lesion model
            — <em>without sharing raw patient images</em>.
            Only model weight updates are exchanged via FedAvg aggregation.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Hospital visualisation ────────────────────────────────────────────────
    n_clients = st.sidebar.slider("Number of Hospitals", 2, 5, 3)
    hospitals_html = '<div class="hospital-grid">'
    hospital_icons = ["🏥", "🏨", "🏦", "🏛️", "🏗️"]
    for i in range(n_clients):
        hospitals_html += f"""
        <div class="hospital-node active">
            <div class="icon">{hospital_icons[i]}</div>
            <div class="name">Hospital {i}</div>
        </div>"""
    hospitals_html += """
        <div class="hospital-node" style="border-color:rgba(0,212,255,0.4);
             background:linear-gradient(135deg,#0d1b2e,#0a1628);">
            <div class="icon">☁️</div>
            <div class="name">FL Server</div>
        </div>
    </div>"""
    st.markdown(hospitals_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Load / display metrics ────────────────────────────────────────────────
    metrics_placeholder = st.empty()
    chart_placeholder   = st.empty()
    log_placeholder     = st.empty()

    def load_metrics() -> pd.DataFrame | None:
        if METRICS_CSV.exists():
            try:
                df = pd.read_csv(METRICS_CSV)
                df = df.dropna(subset=["val_acc"])
                return df if len(df) > 0 else None
            except Exception:
                return None
        return None

    def render_metrics(df: pd.DataFrame) -> None:
        latest   = df.iloc[-1]
        n_rounds = len(df)

        metrics_placeholder.markdown(
            f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="val cyan">{n_rounds}</div>
                    <div class="lbl">Rounds Complete</div>
                </div>
                <div class="metric-box">
                    <div class="val green">{float(latest.get('val_acc', 0)) * 100:.1f}%</div>
                    <div class="lbl">Global Accuracy</div>
                </div>
                <div class="metric-box">
                    <div class="val amber">{float(latest.get('val_loss', 0)):.4f}</div>
                    <div class="lbl">Val Loss</div>
                </div>
                <div class="metric-box">
                    <div class="val pink">{n_clients}</div>
                    <div class="lbl">Active Clients</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Accuracy / Loss line charts ───────────────────────────────────────
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["round"], y=(df["val_acc"] * 100).round(2),
            mode="lines+markers",
            name="Val Accuracy (%)",
            line=dict(color="#00d4ff", width=2.5),
            marker=dict(size=7, color="#00d4ff",
                        line=dict(width=1, color="#0a0e1a")),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.07)",
        ))

        fig.add_trace(go.Scatter(
            x=df["round"], y=df["val_loss"].round(4),
            mode="lines+markers",
            name="Val Loss",
            yaxis="y2",
            line=dict(color="#ff4d8f", width=2.5, dash="dot"),
            marker=dict(size=7, color="#ff4d8f",
                        line=dict(width=1, color="#0a0e1a")),
        ))

        fig.update_layout(
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(17,24,39,0.6)",
            font          = dict(family="Space Grotesk", color="#7a8599", size=12),
            legend        = dict(orientation="h", yanchor="bottom", y=1.02,
                                 bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
            xaxis = dict(
                title="Federated Round",
                gridcolor="rgba(255,255,255,0.05)",
                linecolor="rgba(255,255,255,0.1)",
                tickfont=dict(color="#7a8599"),
            ),
            yaxis = dict(
                title="Accuracy (%)",
                gridcolor="rgba(255,255,255,0.05)",
                tickfont=dict(color="#00d4ff"),
                titlefont=dict(color="#00d4ff"),
            ),
            yaxis2 = dict(
                title="Loss",
                overlaying="y", side="right",
                tickfont=dict(color="#ff4d8f"),
                titlefont=dict(color="#ff4d8f"),
                gridcolor="rgba(0,0,0,0)",
            ),
            hovermode="x unified",
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    # ── Load & display existing results ──────────────────────────────────────
    df = load_metrics()
    if df is not None:
        render_metrics(df)
    else:
        metrics_placeholder.markdown(
            """
            <div class="derm-card" style="text-align:center; padding:3rem;">
                <div style="font-size:2rem; margin-bottom:1rem;">📊</div>
                <div style="color:var(--text-muted);">
                    No training data yet.<br>
                    Start the FL server and clients to begin.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Auto-refresh toggle ───────────────────────────────────────────────────
    st.markdown("---")
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        auto_refresh = st.toggle("Auto-Refresh", value=False)
    with col_b:
        refresh_rate = st.selectbox("Interval", [5, 10, 30, 60], index=1,
                                    label_visibility="collapsed")
    with col_c:
        if st.button("🔄 Refresh Now"):
            st.rerun()

    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

    # ── Quick-start instructions ──────────────────────────────────────────────
    with st.expander("📋 Quick Start"):
        st.code(
            f"""# Terminal 1 — Start the FL Server
python server.py --rounds 10 --n_clients {n_clients}

# Terminal 2..{n_clients + 1} — Start each Hospital Client
python client.py --hospital_id 0
python client.py --hospital_id 1
{"python client.py --hospital_id 2" if n_clients >= 3 else ""}
{"python client.py --hospital_id 3" if n_clients >= 4 else ""}
{"python client.py --hospital_id 4" if n_clients >= 5 else ""}""",
            language="bash",
        )


# -- Tab B: XAI Inference + TEE -------------------------------------------
def tab_inference() -> None:
    import torch

    st.markdown("### Explainable AI Inference", unsafe_allow_html=True)
    st.caption("Grad-CAM visual explanation  ·  Groq LLaMA 4 Scout clinical narrative  ·  TEE privacy layer")

    @st.cache_resource(show_spinner="Loading global model ...", ttl=60)
    def load_global_model():
        m = build_model()
        if GLOBAL_MODEL_PATH.exists():
            state = torch.load(GLOBAL_MODEL_PATH, map_location=DEVICE)
            m.load_state_dict(state)
            return m, True
        return m, False

    model, model_loaded = load_global_model()
    if not model_loaded:
        st.warning("No trained model found. Complete at least one FL round to enable inference.", icon="⚠️")

    engine = get_engine()
    tee_ok, tee_msg = engine.refresh_status()

    if tee_ok:
        st.markdown(
            f"""<div style="background:rgba(0,255,136,0.07);border:1px solid
            rgba(0,255,136,0.3);border-radius:10px;padding:0.75rem 1.2rem;
            margin-bottom:1rem;font-size:0.82rem;">
            🔒 <b style="color:var(--accent-green);">TEE Active</b> &nbsp;·&nbsp;
            {tee_msg} &nbsp;·&nbsp;
            <span style="color:var(--text-muted);">EXIF stripped · GradCAM overlay
            only sent to VLM · data stays on-device</span></div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div style="background:rgba(255,180,0,0.07);border:1px solid
            rgba(255,180,0,0.3);border-radius:10px;padding:0.75rem 1.2rem;
            margin-bottom:1rem;font-size:0.82rem;">
            ⚠️ <b style="color:var(--accent-amber);">Groq Offline</b> &nbsp;·&nbsp;
            <span style="color:var(--text-muted);">Grad-CAM analysis is available.
            Set <code>GROQ_API_KEY</code> to enable AI narrative.
            Get a free key at <a href='https://console.groq.com' style='color:var(--accent-amber)'>console.groq.com</a>.</span></div>""",
            unsafe_allow_html=True,
        )

    col_up, col_res = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown(
            """<div class="derm-card"><h4 style="margin:0 0 0.5rem 0;">
            Upload Dermatoscopic Image</h4>
            <p style="color:var(--text-muted);font-size:0.82rem;">
            JPG / PNG &nbsp;·&nbsp; EXIF auto-stripped</p></div>""",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader("img", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image", use_column_width=True)
            alpha = st.slider("GradCAM blend intensity", 0.2, 0.8, 0.45, 0.05)

    with col_res:
        if not uploaded:
            st.markdown(
                """<div class="derm-card" style="text-align:center;padding:4rem 2rem;">
                <div style="font-size:3rem;margin-bottom:1rem;">🩺</div>
                <div style="color:var(--text-muted);">Upload a lesion image to get
                Grad-CAM + AI explanation.</div></div>""",
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Running Grad-CAM ..."):
                xai = explain_prediction(model, image, alpha=alpha)

            pred_class = CLASS_NAMES[xai["pred_class"]]
            pred_color = CLASS_COLORS[xai["pred_class"]]
            conf_pct   = xai["confidence"] * 100

            st.markdown(
                f"""<div class="pred-result">
                <div style="font-size:0.75rem;color:var(--text-muted);
                     text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;">
                CNN Prediction</div>
                <div class="pred-class" style="color:{pred_color};">{pred_class}</div>
                <div class="pred-conf">Confidence:
                <b style="color:var(--accent-green);">{conf_pct:.1f}%</b></div>
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<p style='color:var(--text-muted);font-size:0.82rem;'>Class probabilities</p>", unsafe_allow_html=True)
            confidence_bars(xai["all_probs"])

    if uploaded:
        st.markdown("---")
        st.markdown("#### 🔍 Grad-CAM Attention Maps")
        st.caption("Highlighted regions indicate areas most influential to the prediction.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<p style='text-align:center;color:var(--text-muted);font-size:0.78rem;'>Original</p>", unsafe_allow_html=True)
            st.image(image, use_column_width=True)
        with c2:
            st.markdown("<p style='text-align:center;color:var(--text-muted);font-size:0.78rem;'>Grad-CAM Heatmap</p>", unsafe_allow_html=True)
            st.image(xai["heatmap_pil"], use_column_width=True)
        with c3:
            st.markdown("<p style='text-align:center;color:var(--text-muted);font-size:0.78rem;'>Overlay (sent to VLM)</p>", unsafe_allow_html=True)
            st.image(xai["overlay_pil"], use_column_width=True)

        st.markdown("---")
        st.markdown("#### ✨ Clinical Explanation")
        st.caption("Powered by Groq LLaMA 4 Scout — GradCAM overlay only · EXIF-stripped · HTTPS encrypted")

        col_btn, col_info = st.columns([2, 3])
        with col_btn:
            run_vlm = st.button("✨ Generate Groq Explanation", use_container_width=True)
        with col_info:
            st.markdown(
                """<div style="font-size:0.75rem;color:var(--text-muted);
                padding:0.5rem 0.8rem;background:rgba(0,0,0,0.2);border-radius:8px;line-height:1.7;">
                TEE guarantees: EXIF stripped · raw image never sent · only GradCAM overlay
                transmitted · HTTPS encrypted
                </div>""",
                unsafe_allow_html=True,
            )

        if run_vlm:
            with st.spinner("Groq LLaMA analysing GradCAM overlay inside TEE ..."):
                result = engine.generate_explanation(
                    original_image  = image,
                    overlay_image   = xai["overlay_pil"],
                    pred_class_idx  = xai["pred_class"],
                    pred_class_name = CLASS_NAMES[xai["pred_class"]],
                    confidence      = xai["confidence"],
                    all_probs       = xai["all_probs"],
                    class_names     = CLASS_NAMES,
                    region_desc     = xai["region_desc"],
                )

            if result["success"]:
                st.markdown(
                    f"""<div class="derm-card-accent" style="margin-top:1rem;">
                    <div style="font-size:0.72rem;color:var(--text-muted);margin-bottom:0.75rem;display:flex;gap:1.5rem;flex-wrap:wrap;">
                    <span>🔒 Audit ID: <code>{result["audit_id"]}</code></span>
                    <span>⚡ {result["latency_ms"]}ms</span>
                    <span>✨ {result["tee_status"]["vlm_model"]}</span>
                    </div></div>""",
                    unsafe_allow_html=True,
                )
                st.markdown(result["explanation"])
                with st.expander("🔒 Privacy Audit Record"):
                    tee = result["tee_status"]
                    for k, v in tee.items():
                        label = k.replace("_", " ").title()
                        if k == "data_leaves_device":
                            display = "No — data stays local" if not v else "YES — WARNING"
                            icon = "✅" if not v else "❌"
                        elif k in ("pii_scrubbed", "exif_stripped"):
                            display, icon = str(v), "✅"
                        else:
                            display, icon = str(v), "ℹ️"
                        st.markdown(
                            f"<div style='display:flex;gap:1rem;font-size:0.82rem;margin-bottom:0.25rem;'>"
                            f"<span style='color:var(--text-muted);min-width:200px;'>{icon} {label}</span>"
                            f"<code>{display}</code></div>",
                            unsafe_allow_html=True,
                        )
                    st.caption(f"Audit log: {result['tee_status']['audit_log']}")
            else:
                st.error(result['explanation'])

        st.markdown(
            """<div style="margin-top:1.5rem;padding:1rem;background:rgba(255,77,143,0.07);
            border:1px solid rgba(255,77,143,0.25);border-radius:8px;
            font-size:0.78rem;color:var(--text-muted);">
            ⚕️ <b>Medical Disclaimer</b> — For research use only.
            Always consult a qualified dermatologist for clinical decisions.</div>""",
            unsafe_allow_html=True,
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 1.5rem 0 1rem 0;">
                <div style="font-size:2.5rem;">🔬</div>
                <div style="font-size:1.2rem; font-weight:700;
                     background:linear-gradient(135deg,#00d4ff,#00ff88);
                     -webkit-background-clip:text;
                     -webkit-text-fill-color:transparent;">
                    DermFed
                </div>
                <div style="font-size:0.75rem; color:var(--text-muted); margin-top:0.2rem;">
                    XAI · TEE · Federated Learning
                </div>
            </div>
            <hr>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**📦 Stack**")
        stack_items = [
            ("🌸", "Flower (flwr)",   "FL Framework"),
            ("🔥", "PyTorch",          "ML Backend"),
            ("📱", "MobileNetV2",      "Backbone"),
            ("🔍", "Grad-CAM",         "Explainability"),
            ("🔒", "TEE Engine",       "Privacy Layer"),
            ("✨", "Groq LLaMA 4",    "Free cloud VLM"),
            ("🗃️", "HAM10000",          "Dataset"),
        ]
        for icon, name, desc in stack_items:
            st.markdown(
                f"<div style='font-size:0.82rem; margin-bottom:0.4rem; "
                f"color:var(--text-muted);'>{icon} <b style='color:var(--text-primary)'>"
                f"{name}</b> — {desc}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**⚡ Quick Status**")

        model_exists = GLOBAL_MODEL_PATH.exists()
        data_exists  = Path("data/partitions").exists()
        metrics_ok   = METRICS_CSV.exists()

        ollama_ok, _ = get_engine().refresh_status()
        for label, flag in [
            ("Data partitioned", data_exists),
            ("Model available",  model_exists),
            ("Training data",    metrics_ok),
            ("Groq API",         ollama_ok),
        ]:
            badge_cls  = "badge-done" if flag else "badge-idle"
            badge_text = "✓ Ready" if flag else "○ Pending"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"margin-bottom:0.4rem;font-size:0.82rem;'>"
                f"<span style='color:var(--text-muted)'>{label}</span>"
                f"<span class='badge {badge_cls}'>{badge_text}</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.72rem; color:var(--text-muted); text-align:center;'>"
            "Privacy-First Medical AI</div>",
            unsafe_allow_html=True,
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    inject_css()
    sidebar()
    hero()

    tab_a, tab_b = st.tabs(["📊  Simulation Monitor", "🩺  AI Inference"])
    with tab_a:
        tab_simulation()
    with tab_b:
        tab_inference()


if __name__ == "__main__":
    main()
