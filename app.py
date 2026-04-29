"""
DermFed – app.py
Streamlit dashboard: Simulation monitor + Inference engine.

Run:
    streamlit run app.py
"""

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
    predict,
)

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
                Federated Learning · Skin Cancer Detection · HAM10000 Dataset · MobileNetV2
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
            <b>How it works:</b> Multiple hospitals train a shared skin-cancer model
            collaboratively — <em>without ever sharing raw patient images</em>.
            Only encrypted model weight updates travel to the central aggregator
            (FedAvg). This simulates a real-world privacy-preserving clinical AI pipeline.
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
                    <b>Run the FL server and clients</b> to start training,
                    then refresh this page.
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
    with st.expander("📋 How to start the FL training (Windows)"):
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


# ── Tab B: Inference ──────────────────────────────────────────────────────────
def tab_inference() -> None:
    st.markdown("### 🔬 Skin Lesion Classification")

    # ── Load model ────────────────────────────────────────────────────────────
    @st.cache_resource(show_spinner="Loading global model …", ttl=60)
    def load_global_model():
        model = build_model()
        if GLOBAL_MODEL_PATH.exists():
            state = torch.load(GLOBAL_MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state)
            return model, True
        return model, False

    model, model_loaded = load_global_model()

    if not model_loaded:
        st.warning(
            "⚠️ No trained global model found at `models/global_model.pt`.\n\n"
            "The app will use a **randomly-initialised** MobileNetV2 until you "
            "complete at least one federated round. Predictions will not be meaningful.",
            icon="⚠️",
        )

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown(
            """
            <div class="derm-card">
                <h4 style="margin:0 0 0.75rem 0; color:var(--text-primary);">
                    Upload Dermatoscopic Image
                </h4>
                <p style="color:var(--text-muted); font-size:0.85rem;">
                    Accepts JPG, PNG, JPEG · Recommended: ≥ 450 × 450 px
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Choose a dermatoscopic image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # ── About the classes ─────────────────────────────────────────────
            with st.expander("ℹ️ About the 7 diagnostic classes"):
                class_info = {
                    "Actinic Keratoses (akiec)": "Pre-cancerous skin lesion caused by sun exposure.",
                    "Basal Cell Carcinoma (bcc)": "Most common skin cancer; rarely metastasises.",
                    "Benign Keratosis (bkl)":    "Non-cancerous growth including solar lentigines & seborrheic keratosis.",
                    "Dermatofibroma (df)":        "Common benign fibrous nodule usually on legs.",
                    "Melanoma (mel)":             "Most dangerous skin cancer arising from melanocytes.",
                    "Melanocytic Nevi (nv)":      "Common benign moles (>67% of dataset).",
                    "Vascular Lesions (vasc)":    "Includes angiomas, angiokeratomas, pyogenic granulomas.",
                }
                for cls, desc in class_info.items():
                    st.markdown(
                        f"**{cls}** — <span style='color:var(--text-muted);font-size:0.85rem'>{desc}</span>",
                        unsafe_allow_html=True,
                    )

    with col_result:
        if uploaded_file:
            with st.spinner("Analysing …"):
                pred_idx, confidence, probs = predict(model, image)

            pred_class = CLASS_NAMES[pred_idx]
            pred_color = CLASS_COLORS[pred_idx]
            conf_pct   = confidence * 100

            # ── Result box ────────────────────────────────────────────────────
            st.markdown(
                f"""
                <div class="pred-result">
                    <div style="font-size:0.8rem; color:var(--text-muted); 
                         text-transform:uppercase; letter-spacing:0.1em;
                         margin-bottom:0.5rem;">
                        Predicted Diagnosis
                    </div>
                    <div class="pred-class" style="color:{pred_color};">
                        {pred_class}
                    </div>
                    <div class="pred-conf">
                        Confidence: <b style="color:var(--accent-green);">{conf_pct:.1f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<p style='color:var(--text-muted); font-size:0.85rem;'>"
                "Class probability distribution</p>",
                unsafe_allow_html=True,
            )
            confidence_bars(probs)

            # ── Disclaimer ────────────────────────────────────────────────────
            st.markdown(
                """
                <div style="margin-top:1.5rem; padding:1rem;
                     background:rgba(255,77,143,0.07);
                     border:1px solid rgba(255,77,143,0.25);
                     border-radius:8px; font-size:0.78rem;
                     color:var(--text-muted);">
                    ⚕️ <b>Medical Disclaimer</b> — This tool is for educational/research
                    purposes only and does not constitute medical advice. Always consult
                    a qualified dermatologist for clinical diagnosis.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="derm-card" style="text-align:center; padding:4rem 2rem;">
                    <div style="font-size:3rem; margin-bottom:1rem;">🩺</div>
                    <div style="color:var(--text-muted);">
                        Upload a dermatoscopic image on the left to see
                        the AI-powered classification result here.
                    </div>
                </div>
                """,
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
                    v1.0 · Federated Medical AI
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
            ("📱", "MobileNetV2",      "Model Backbone"),
            ("🎈", "Streamlit",        "Dashboard"),
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

        for label, flag in [
            ("Data partitioned", data_exists),
            ("Model available",  model_exists),
            ("Training data",    metrics_ok),
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
            "Built with ❤️ · Privacy-First Medical AI</div>",
            unsafe_allow_html=True,
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    inject_css()
    sidebar()
    hero()

    tab_a, tab_b = st.tabs(["📊  Simulation Monitor", "🩺  Image Inference"])
    with tab_a:
        tab_simulation()
    with tab_b:
        tab_inference()


if __name__ == "__main__":
    main()
