"""
app.py — MHT-CET Admission Prediction System (v3)
Changes vs v2:
  - Synced percentile number-input + slider
  - CAP-style Dream/Target/Safe recommendation logic (30 results default)
  - Probability shown but NOT primary sort key
Run: streamlit run app.py
"""

import os
import sys
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.preprocess import load_and_preprocess, get_filter_options
from utils.model import get_model
from utils.predictor import predict_colleges

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MHT-CET Admission Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (identical theme, small additions for number-input styling) ────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    font-family: 'Inter', sans-serif !important;
    background: #0a0e1a !important;
    color: #e2e8f0 !important;
}
.block-container { padding: 0 1.5rem 2rem 1.5rem !important; max-width: 100% !important; }
[data-testid="stMain"] > div { padding-top: 0 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1224 !important;
    border-right: 1px solid rgba(139,92,246,0.25) !important;
}
[data-testid="stSidebarContent"] { padding: 16px !important; }

.sb-brand {
    text-align: center; padding: 16px 10px 20px;
    border-bottom: 1px solid rgba(139,92,246,0.2); margin-bottom: 18px;
}
.sb-brand .logo {
    width: 60px; height: 60px;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    border-radius: 50%; display: flex; align-items: center;
    justify-content: center; margin: 0 auto 10px; font-size: 26px;
}
.sb-brand h2 { color:#fff; font-size:1.2rem; font-weight:800; letter-spacing:1px; margin:0; }
.sb-brand span { color:#8b5cf6; font-size:0.72rem; font-weight:700; letter-spacing:3px; }

.sb-section-title {
    color:#94a3b8; font-size:0.78rem; font-weight:600;
    text-transform:uppercase; letter-spacing:1.5px;
    margin:16px 0 10px; display:flex; align-items:center; gap:6px;
}
.sb-input-wrap {
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(139,92,246,0.2);
    border-radius:10px; padding:12px 14px; margin-bottom:10px;
}
.sb-input-label { font-size:0.78rem; color:#94a3b8; font-weight:500; margin-bottom:6px; }
.sb-pct-val {
    text-align:center; font-size:1.3rem; font-weight:800;
    color:#8b5cf6; margin-bottom:4px;
}

/* Number input inside sb-input-wrap */
.sb-input-wrap input[type=number] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(139,92,246,0.35) !important;
    border-radius: 8px !important; color: #fff !important;
    font-size: 1.1rem !important; font-weight: 700 !important;
    text-align: center !important; padding: 6px !important;
}

.sb-tip {
    background:rgba(245,158,11,0.08);
    border:1px solid rgba(245,158,11,0.25);
    border-radius:10px; padding:14px; margin-top:14px;
}
.sb-tip-title { color:#f59e0b; font-size:0.82rem; font-weight:600; margin-bottom:6px; }
.sb-tip-body  { color:#94a3b8; font-size:0.76rem; line-height:1.55; }

/* ── Hero ── */
.hero-banner {
    background:linear-gradient(135deg,#1a1040 0%,#2d1b69 40%,#1e1b4b 70%,#0f172a 100%);
    border:1px solid rgba(139,92,246,0.3); border-radius:16px;
    padding:32px 44px; margin:18px 0; position:relative; overflow:hidden;
    display:flex; align-items:center; gap:22px;
}
.hero-banner::before {
    content:''; position:absolute; top:-50%; left:-20%;
    width:60%; height:200%;
    background:radial-gradient(ellipse,rgba(139,92,246,0.15) 0%,transparent 70%);
}
.hero-icon { font-size:52px; flex-shrink:0; }
.hero-text h1 {
    font-size:2.3rem; font-weight:800;
    background:linear-gradient(90deg,#60a5fa,#a78bfa,#f0abfc);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    line-height:1.1; margin:0;
}
.hero-text h2 { color:#e2e8f0; font-size:1.05rem; font-weight:500; margin:4px 0 0; }
.hero-text p  { color:#94a3b8; font-size:0.86rem; margin:5px 0 0; }
.hero-badge {
    margin-left:auto; flex-shrink:0;
    background:linear-gradient(135deg,#7c3aed,#4f46e5);
    border:2px solid rgba(167,139,250,0.5); border-radius:50%;
    width:86px; height:86px; display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    text-align:center; font-size:0.58rem; font-weight:800;
    color:#fff; letter-spacing:1px; line-height:1.4;
}

/* ── Stat cards ── */
.stat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:18px; }
.stat-card {
    border-radius:14px; padding:18px 20px;
    display:flex; align-items:center; gap:14px;
    border:1px solid rgba(255,255,255,0.06);
}
.stat-card.blue   { background:linear-gradient(135deg,#1e3a5f,#1a2d4a); }
.stat-card.purple { background:linear-gradient(135deg,#2d1b69,#231459); }
.stat-card.green  { background:linear-gradient(135deg,#064e3b,#065f46); }
.stat-card.orange { background:linear-gradient(135deg,#451a03,#78350f); }
.stat-icon { width:50px; height:50px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:22px; flex-shrink:0; }
.stat-icon.blue   { background:rgba(59,130,246,0.25); }
.stat-icon.purple { background:rgba(139,92,246,0.25); }
.stat-icon.green  { background:rgba(16,185,129,0.25); }
.stat-icon.orange { background:rgba(245,158,11,0.25); }
.stat-val { font-size:1.8rem; font-weight:800; color:#fff; line-height:1; }
.stat-lbl { font-size:0.76rem; color:#94a3b8; margin-top:2px; }

/* ── Welcome ── */
.welcome-box {
    background:linear-gradient(135deg,#1e2d4a,#1a2438);
    border:1px solid rgba(59,130,246,0.25); border-radius:14px;
    padding:26px 32px; display:flex; align-items:center; gap:22px; margin-bottom:22px;
}
.welcome-box h3 { color:#fff; font-size:1.1rem; font-weight:700; margin-bottom:6px; }
.welcome-box p  { color:#94a3b8; font-size:0.85rem; line-height:1.6; margin:0; }

/* ── Results table ── */
.results-header { display:flex; align-items:center; gap:12px; margin-bottom:14px; flex-wrap:wrap; }
.results-header h3 { color:#fff; font-size:1.1rem; font-weight:700; margin:0; }
.tag-pill {
    background:linear-gradient(90deg,#10b981,#059669);
    color:#fff; font-size:0.70rem; font-weight:700;
    padding:3px 11px; border-radius:20px;
}

/* Section dividers inside table */
.section-row td {
    padding: 6px 12px !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border-bottom: none !important;
}
.section-dream  { background: rgba(239,68,68,0.12) !important; color:#ef4444 !important; }
.section-target { background: rgba(245,158,11,0.12) !important; color:#f59e0b !important; }
.section-safe   { background: rgba(16,185,129,0.12) !important; color:#10b981 !important; }

.res-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
.res-table thead tr { border-bottom:1px solid rgba(255,255,255,0.07); }
.res-table th { padding:10px 12px; text-align:left; color:#64748b; font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.8px; }
.res-table td { padding:12px 12px; border-bottom:1px solid rgba(255,255,255,0.04); color:#e2e8f0; }
.res-table tbody tr:not(.section-row):hover { background:rgba(139,92,246,0.07); }

.prob-safe   { color:#10b981; font-weight:700; }
.prob-target { color:#f59e0b; font-weight:700; }
.prob-dream  { color:#ef4444; font-weight:700; }

.badge-safe   { background:#10b981; color:#fff; padding:3px 10px; border-radius:6px; font-size:.68rem; font-weight:700; }
.badge-target { background:#f59e0b; color:#fff; padding:3px 10px; border-radius:6px; font-size:.68rem; font-weight:700; }
.badge-dream  { background:#ef4444; color:#fff; padding:3px 10px; border-radius:6px; font-size:.68rem; font-weight:700; }
.res-count { color:#64748b; font-size:0.76rem; margin-top:10px; }

/* ── Insight cards ── */
.insight-card {
    background:#111827; border:1px solid rgba(255,255,255,0.06);
    border-radius:14px; padding:20px;
}
.ic-icon { font-size:22px; margin-bottom:8px; }
.ic-lbl  { color:#64748b; font-size:0.74rem; font-weight:500; margin-bottom:3px; }
.ic-val  { color:#fff; font-size:1.55rem; font-weight:800; line-height:1.1; }
.ic-sub  { color:#94a3b8; font-size:0.74rem; margin-top:3px; }

/* ── How it works ── */
.how-title { color:#fff; font-size:1.08rem; font-weight:700; margin:26px 0 14px; }
.how-grid  { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
.how-card  { background:#111827; border:1px solid rgba(255,255,255,0.06); border-radius:14px; padding:22px; position:relative; }
.hc-num { position:absolute; top:14px; right:14px; width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.65rem; font-weight:800; color:#fff; }
.hc-num.blue   { background:#3b82f6; }
.hc-num.purple { background:#8b5cf6; }
.hc-num.orange { background:#f59e0b; }
.hc-icon { font-size:30px; margin-bottom:12px; }
.how-card h4 { color:#fff; font-size:0.92rem; font-weight:700; margin-bottom:7px; }
.how-card p  { color:#64748b; font-size:0.78rem; line-height:1.55; margin:0; }

/* ── Buttons ── */
div.stButton > button {
    background:linear-gradient(90deg,#7c3aed,#4f46e5) !important;
    color:#fff !important; border:none !important;
    border-radius:10px !important; padding:14px 20px !important;
    font-size:0.92rem !important; font-weight:700 !important;
    width:100% !important;
    box-shadow:0 4px 20px rgba(124,58,237,0.4) !important;
}
div[data-testid="stDownloadButton"] > button {
    background:rgba(255,255,255,0.07) !important; color:#e2e8f0 !important;
    border:1px solid rgba(255,255,255,0.15) !important;
    border-radius:8px !important; font-size:0.80rem !important;
    padding:8px 16px !important; width:auto !important;
}

#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ── Cache ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="📂 Loading dataset…")
def load_data():
    return load_and_preprocess(os.path.join(BASE_DIR, "data", "dataset.csv"))

@st.cache_resource(show_spinner="🤖 Training model…")
def load_trained_model(df):
    return get_model(df)


# ── Helpers ───────────────────────────────────────────────────────────────────
def medal(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")

def prob_cls(p: float) -> str:
    if p >= 75: return "prob-safe"
    if p >= 50: return "prob-target"
    return "prob-dream"

def badge_html(label: str) -> str:
    if "Safe"   in label: return '<span class="badge-safe">SAFE ✅</span>'
    if "Target" in label: return '<span class="badge-target">TARGET 🎯</span>'
    return '<span class="badge-dream">DREAM 🔥</span>'

def section_divider(label: str, css_cls: str) -> str:
    return f'<tr class="section-row"><td colspan="8" class="{css_cls}">{label}</td></tr>'


def render_table(results: pd.DataFrame):
    """Render grouped Dream → Target → Safe table with section headers."""
    rows = ""
    prev_type = None
    display_rank = 0

    for i, row in results.iterrows():
        cur_type = row.get("Type", "")

        # Section divider whenever group changes
        if cur_type != prev_type:
            if "Dream"  in cur_type:
                rows += section_divider("🔥 Dream Colleges — Cutoff above your percentile", "section-dream")
            elif "Target" in cur_type:
                rows += section_divider("🎯 Target Colleges — Within your reach", "section-target")
            else:
                rows += section_divider("✅ Safe Colleges — Highly likely admission", "section-safe")
            prev_type = cur_type
            display_rank = 0

        display_rank += 1
        prob  = row.get("Admit_Prob_%", 0)
        med   = medal(display_rank) if display_rank <= 3 and "Safe" in cur_type else ""
        name  = row["Institute Name"].title()
        br    = row["Branch"].title()
        dist  = row["DISTRICT"].title()
        cat   = row["Category"]
        cut   = f"{row['Cutoff']:.2f}"
        rank_cell = f'{med} {display_rank}' if med else str(display_rank)

        rows += f"""<tr>
            <td>{rank_cell}</td>
            <td><b>{name}</b></td>
            <td>{br}</td>
            <td>{dist}</td>
            <td>{cat}</td>
            <td>{cut}</td>
            <td class="{prob_cls(prob)}"><b>{prob}%</b></td>
            <td>{badge_html(cur_type)}</td>
        </tr>"""

    st.markdown(f"""
    <table class="res-table">
        <thead><tr>
            <th>#</th><th>Institute Name</th><th>Branch</th>
            <th>District</th><th>Category</th><th>Cutoff</th>
            <th>Probability</th><th>Status</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>
    <div class="res-count">Showing {len(results)} results &nbsp;·&nbsp;
        🔥 Dream = cutoff above your percentile &nbsp;·&nbsp;
        🎯 Target = within reach &nbsp;·&nbsp;
        ✅ Safe = below your percentile
    </div>
    """, unsafe_allow_html=True)


def make_scatter(results: pd.DataFrame, percentile: float):
    fig, ax = plt.subplots(figsize=(4.8, 3.0), facecolor="#111827")
    ax.set_facecolor("#111827")
    cmap = {"🔥 Dream": "#ef4444", "🎯 Target": "#f59e0b", "✅ Safe": "#10b981"}
    for _, row in results.iterrows():
        c = cmap.get(row["Type"], "#8b5cf6")
        ax.scatter(row["Cutoff"], row["Admit_Prob_%"],
                   color=c, s=55, edgecolors="none", alpha=0.85, zorder=3)
    ax.axvline(x=percentile, color="#8b5cf6", linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"Your %ile ({percentile})")
    ax.set_xlabel("Cutoff", color="#64748b", fontsize=7.5)
    ax.set_ylabel("Probability (%)", color="#64748b", fontsize=7.5)
    ax.set_title("Cutoff vs Prediction Probability", color="#e2e8f0", fontsize=8.5, pad=8)
    ax.tick_params(colors="#64748b", labelsize=6.5)
    for sp in ax.spines.values(): sp.set_edgecolor((1, 1, 1, 0.06))
    ax.grid(True, linestyle="--", alpha=0.1, color="white")
    plt.tight_layout(pad=0.8)
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  — synced percentile input + slider
# ════════════════════════════════════════════════════════════════════════════
def build_sidebar(options):
    with st.sidebar:
        # Brand
        st.markdown("""
        <div class="sb-brand">
            <div class="logo">🎓</div>
            <h2>MHT-CET</h2>
            <span>PREDICTOR</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-section-title">👤 Your Details</div>', unsafe_allow_html=True)

        # ── Synced percentile: number-input + slider (callback-based) ──────
        if "percentile_val" not in st.session_state:
            st.session_state["percentile_val"] = 85.0

        def _sync_from_number():
            st.session_state["percentile_val"] = round(float(st.session_state["_pct_num"]), 2)

        def _sync_from_slider():
            st.session_state["percentile_val"] = round(float(st.session_state["_pct_sld"]), 2)

        st.markdown(
            '<div class="sb-input-wrap">'
            '<div class="sb-input-label">📊 Percentile</div>',
            unsafe_allow_html=True,
        )

        st.number_input(
            "Type percentile",
            min_value=0.0, max_value=100.0,
            value=float(st.session_state["percentile_val"]),
            step=0.01, format="%.2f",
            key="_pct_num",
            on_change=_sync_from_number,
            label_visibility="collapsed",
        )

        st.slider(
            "Slide percentile",
            min_value=0.0, max_value=100.0,
            value=float(st.session_state["percentile_val"]),
            step=0.01,
            key="_pct_sld",
            on_change=_sync_from_slider,
            label_visibility="collapsed",
        )

        percentile = float(st.session_state["percentile_val"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Category
        st.markdown('<div class="sb-input-wrap"><div class="sb-input-label">🏷️ Category</div>', unsafe_allow_html=True)
        cat_idx = options["categories"].index("GOPEN") if "GOPEN" in options["categories"] else 0
        category = st.selectbox("", options["categories"], index=cat_idx, key="cat", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        # Branch
        st.markdown('<div class="sb-input-wrap"><div class="sb-input-label">🎓 Preferred Branch</div>', unsafe_allow_html=True)
        branch = st.selectbox("", options["branches"], key="br", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        # District
        st.markdown('<div class="sb-input-wrap"><div class="sb-input-label">📍 Preferred District</div>', unsafe_allow_html=True)
        dist_list = ["Any District"] + options["districts"]
        dist_sel = st.selectbox("", dist_list, key="dist", label_visibility="collapsed")
        district = None if dist_sel == "Any District" else dist_sel
        st.markdown('</div>', unsafe_allow_html=True)

        # Top N (default 30)
        st.markdown('<div class="sb-input-wrap"><div class="sb-input-label">🔢 Number of Results</div>', unsafe_allow_html=True)
        top_n = st.slider("", 10, 50, 30, 5, key="topn", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        predict_btn = st.button("🚀 Predict Admissions")

        st.markdown("""
        <div class="sb-tip">
            <div class="sb-tip-title">💡 Quick Tip</div>
            <div class="sb-tip-body">
                Results are split into Dream 🔥, Target 🎯, and Safe ✅ colleges —
                just like real CAP counseling. Higher percentile = more Safe colleges!
            </div>
        </div>
        """, unsafe_allow_html=True)

    return percentile, category, branch, district, top_n, predict_btn


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    df      = load_data()
    options = get_filter_options(df)
    clf, enc_branch, enc_district, enc_category = load_trained_model(df)

    percentile, category, branch, district, top_n, predict_btn = build_sidebar(options)

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-icon">🎓</div>
        <div class="hero-text">
            <h1>MHT-CET</h1>
            <h2>Admission Prediction System</h2>
            <p>AI-Powered College Recommendation Engine</p>
        </div>
        <div class="hero-badge">SMART<br>PREDICTOR</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stat cards ────────────────────────────────────────────────────────────
    nc   = df["Institute Name"].nunique()
    nb   = df["Branch"].nunique()
    nd   = df["DISTRICT"].nunique()
    ncat = df["Category"].nunique()
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card blue">
            <div class="stat-icon blue">🏛️</div>
            <div><div class="stat-val">{nc}</div><div class="stat-lbl">Total Colleges</div></div>
        </div>
        <div class="stat-card purple">
            <div class="stat-icon purple">📚</div>
            <div><div class="stat-val">{nb}</div><div class="stat-lbl">Total Branches</div></div>
        </div>
        <div class="stat-card green">
            <div class="stat-icon green">📍</div>
            <div><div class="stat-val">{nd}</div><div class="stat-lbl">Total Districts</div></div>
        </div>
        <div class="stat-card orange">
            <div class="stat-icon orange">🏷️</div>
            <div><div class="stat-val">{ncat}</div><div class="stat-lbl">Total Categories</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Predict ───────────────────────────────────────────────────────────────
    if predict_btn:
        with st.spinner("🔍 Running CAP-style prediction…"):
            results = predict_colleges(
                df, clf, enc_branch, enc_district, enc_category,
                percentile=percentile, category=category,
                branch=branch, district=district, top_n=top_n,
            )

        if results.empty:
            st.warning("⚠️ No colleges found. Try 'Any District' or a different branch.")
            return

        # Counts per type
        n_dream  = (results["Type"].str.contains("Dream")).sum()
        n_target = (results["Type"].str.contains("Target")).sum()
        n_safe   = (results["Type"].str.contains("Safe")).sum()

        # Header row
        col_t, col_dl = st.columns([4, 1])
        with col_t:
            st.markdown(f"""
            <div class="results-header">
                <span style="font-size:1.25rem">🏆</span>
                <h3>Top College Recommendations</h3>
                <span class="tag-pill">{len(results)} Results</span>
                <span style="color:#ef4444;font-size:0.78rem;font-weight:600">🔥 {n_dream} Dream</span>
                <span style="color:#f59e0b;font-size:0.78rem;font-weight:600">🎯 {n_target} Target</span>
                <span style="color:#10b981;font-size:0.78rem;font-weight:600">✅ {n_safe} Safe</span>
            </div>""", unsafe_allow_html=True)
        with col_dl:
            buf = io.StringIO()
            results.to_csv(buf, index=False)
            st.download_button("⬇️ Download CSV", buf.getvalue(),
                               "mhtcet_predictions.csv", "text/csv")

        render_table(results)

        # ── Insight row ───────────────────────────────────────────────────────
        safe_rows   = results[results["Type"].str.contains("Safe")]
        best_safe_p = safe_rows["Admit_Prob_%"].max() if not safe_rows.empty else 0
        avg_p       = results["Admit_Prob_%"].mean()
        top_name    = results.iloc[0]["Institute Name"].title()
        top_br      = results.iloc[0]["Branch"].title()
        short       = (top_name[:18] + "…") if len(top_name) > 18 else top_name

        c1, c2, c3, c4 = st.columns([1, 1, 1, 1.5])
        with c1:
            st.markdown(f"""<div class="insight-card">
                <div class="ic-icon">📈</div>
                <div class="ic-lbl">Best Safe Prob</div>
                <div class="ic-val" style="color:#10b981">{best_safe_p}%</div>
                <div class="ic-sub">Top safe college</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="insight-card">
                <div class="ic-icon">🧮</div>
                <div class="ic-lbl">Average Probability</div>
                <div class="ic-val" style="color:#8b5cf6">{avg_p:.2f}%</div>
                <div class="ic-sub">Across {len(results)} colleges</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="insight-card">
                <div class="ic-icon">👑</div>
                <div class="ic-lbl">Top Dream College</div>
                <div class="ic-val" style="font-size:1rem;color:#ef4444">{short}</div>
                <div class="ic-sub">{top_br[:30]}</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="insight-card" style="padding:14px">', unsafe_allow_html=True)
            fig = make_scatter(results, percentile)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="welcome-box">
            <div style="font-size:46px;flex-shrink:0">🚀</div>
            <div>
                <h3>Set your details in the sidebar</h3>
                <p>Enter your percentile (type or slide), choose your category, preferred branch and district,<br>
                then click <b>Predict Admissions</b> to see your personalised CAP-style shortlist.</p>
            </div>
            <div style="margin-left:auto;font-size:50px;flex-shrink:0">📋</div>
        </div>
        """, unsafe_allow_html=True)

    # ── How it works ──────────────────────────────────────────────────────────
    st.markdown('<div class="how-title">✨ How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="how-grid">
        <div class="how-card">
            <div class="hc-num blue">01</div>
            <div class="hc-icon">🗄️</div>
            <h4>Smart Data Processing</h4>
            <p>We clean and transform real MHT-CET cutoff data for accurate analysis.</p>
        </div>
        <div class="how-card">
            <div class="hc-num purple">02</div>
            <div class="hc-icon">🧠</div>
            <h4>AI-Powered Model</h4>
            <p>Random Forest Classifier predicts your admission chances based on patterns.</p>
        </div>
        <div class="how-card">
            <div class="hc-num orange">03</div>
            <div class="hc-icon">🏆</div>
            <h4>CAP-Style Ranking</h4>
            <p>Colleges split into Dream 🔥, Target 🎯, Safe ✅ — sorted by cutoff proximity, not just probability.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
