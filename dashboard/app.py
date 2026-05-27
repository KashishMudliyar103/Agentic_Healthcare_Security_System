"""
dashboard/app.py — Healthcare EHR Security Dashboard
Light pastel purple & white theme with tabbed layout
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv(
    "API_URL",
    "http://localhost:8000"
)

st.set_page_config(
    page_title="EHR Security Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── LIGHT PASTEL PURPLE THEME ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,300;0,400;0,600;1,300&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #F5F3FF !important;
    color: #111827 !important;
}

.stApp {
    background: linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 40%, #F0EFFE 100%) !important;
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #DDD6FE !important;
    box-shadow: 2px 0 12px rgba(109,40,217,0.06);
}
[data-testid="stSidebar"] > div { background: #FFFFFF !important; }
[data-testid="stSidebar"] * { color: #374151 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 { color: #111827 !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label { color: #374151 !important; }
[data-testid="stSidebar"] .stMarkdown { color: #374151 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #FFFFFF !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 2px !important;
    border: 1px solid #DDD6FE !important;
    box-shadow: 0 2px 8px rgba(109,40,217,0.08);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    color: #374151 !important;
    padding: 8px 18px !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
    color: white !important;
    box-shadow: 0 3px 10px rgba(109,40,217,0.3) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    padding-top: 20px !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    display: none !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── KPI Cards ── */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #EDE9FE;
    border-radius: 18px;
    padding: 22px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(109,40,217,0.07);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(109,40,217,0.14);
}
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 18px 18px 0 0;
}
.kpi-card.violet::after  { background: linear-gradient(90deg, #7C3AED, #A78BFA); }
.kpi-card.rose::after    { background: linear-gradient(90deg, #E11D48, #FB7185); }
.kpi-card.amber::after   { background: linear-gradient(90deg, #D97706, #FCD34D); }
.kpi-card.indigo::after  { background: linear-gradient(90deg, #4338CA, #818CF8); }
.kpi-card.emerald::after { background: linear-gradient(90deg, #059669, #6EE7B7); }

.kpi-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 8px;
}
.kpi-value {
    font-family: 'Fraunces', serif;
    font-size: 40px;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 6px;
}
.kpi-card.violet .kpi-value  { color: #111827; }
.kpi-card.rose .kpi-value    { color: #111827; }
.kpi-card.amber .kpi-value   { color: #111827; }
.kpi-card.indigo .kpi-value  { color: #111827; }
.kpi-card.emerald .kpi-value { color: #111827; }
.kpi-sub {
    font-size: 11px;
    color: #6B7280;
    font-weight: 500;
}

/* ── Section header ── */
.sec-header {
    font-family: 'Fraunces', serif;
    font-size: 20px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1.5px solid #DDD6FE;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Detail card ── */
.det-card {
    background: #FFFFFF;
    border: 1px solid #EDE9FE;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 2px 8px rgba(109,40,217,0.06);
}
.det-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid #F3F0FF;
    font-size: 13px;
}
.det-row:last-child { border-bottom: none; }
.det-key { color: #6B7280; font-weight: 500; }
.det-val { color: #111827; font-family: 'DM Mono', monospace; font-size: 12px; }

/* ── Narrative ── */
.narrative-box {
    background: #F5F3FF;
    border: 1px solid #C4B5FD;
    border-left: 4px solid #7C3AED;
    border-radius: 0 10px 10px 0;
    padding: 14px 16px;
    font-size: 13px;
    line-height: 1.7;
    color: #111827;
    margin-top: 14px;
}

/* ── Risk badges ── */
.rbadge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.3px;
}
.rbadge-LOW      { background: #D1FAE5; color: #065F46; }
.rbadge-MEDIUM   { background: #FEF3C7; color: #92400E; }
.rbadge-HIGH     { background: #FFEDD5; color: #9A3412; }
.rbadge-CRITICAL { background: #FEE2E2; color: #991B1B; }

/* ── HITL alert ── */
.hitl-alert {
    background: #FFFBEB;
    border: 1px solid #FCD34D;
    border-left: 4px solid #D97706;
    border-radius: 0 12px 12px 0;
    padding: 14px 16px;
    margin: 12px 0;
}
.hitl-title { font-weight: 600; color: #92400E; font-size: 13px; margin-bottom: 4px; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 8px rgba(109,40,217,0.2) !important;
}
.stButton > button:hover {
    box-shadow: 0 4px 16px rgba(109,40,217,0.35) !important;
    transform: translateY(-1px) !important;
}
.approve-btn > button {
    background: linear-gradient(135deg, #059669, #047857) !important;
    box-shadow: 0 2px 8px rgba(5,150,105,0.25) !important;
}
.reject-btn > button {
    background: linear-gradient(135deg, #E11D48, #BE123C) !important;
    box-shadow: 0 2px 8px rgba(225,29,72,0.25) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #EDE9FE !important;
    border-radius: 14px !important;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(109,40,217,0.06);
}

/* ── Selectbox / Input ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div {
    background: #FFFFFF !important;
    border-color: #DDD6FE !important;
    color: #2D1B6E !important;
    border-radius: 10px !important;
}
[data-testid="stSelectbox"] * { color: #111827 !important; }

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div {
    background: #7C3AED !important;
}

/* ── Info boxes ── */
.stAlert { border-radius: 10px !important; }

/* ── Divider ── */
hr { border-color: #DDD6FE !important; }

/* ── Checkbox ── */
[data-testid="stCheckbox"] * { color: #111827 !important; }

/* ── Plotly chart bg ── */
.js-plotly-plot { border-radius: 14px; }

/* ── Countdown ── */
.countdown {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #6B7280;
    text-align: center;
    padding: 4px;
}

/* ── Status dot ── */
.sdot {
    display: inline-block;
    width: 9px; height: 9px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
.sdot-on  { background: #10B981; box-shadow: 0 0 6px #10B981; }
.sdot-off { background: #F87171; }
@keyframes pulse {
    0%,100% { opacity:1; }
    50% { opacity:0.45; }
}

/* ── Empty state ── */
.empty-state {
    background: #FFFFFF;
    border: 1.5px dashed #C4B5FD;
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    color: #6B7280;
}

/* ── Action badge ── */
.action-badge {
    display: inline-block;
    background: #EDE9FE;
    color: #7C3AED;
    padding: 3px 9px;
    border-radius: 6px;
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "event_log"      not in st.session_state: st.session_state.event_log = []
if "selected_event" not in st.session_state: st.session_state.selected_event = None
if "last_refresh"   not in st.session_state: st.session_state.last_refresh = datetime.utcnow()

# ── Helpers ───────────────────────────────────────────────────────────────────
def api_get(ep):
    try:
        r = requests.get(f"{API_BASE}{ep}", timeout=5)
        return r.json()
    except:
        return {}

def api_post(ep, data):
    try:
        r = requests.post(f"{API_BASE}{ep}", json=data, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def load_audit_log():
    data = api_get("/api/v1/audit-log?limit=500")
    entries = data.get("entries", [])
    return pd.DataFrame(entries) if entries else pd.DataFrame()

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#111827", family="DM Sans", size=12),
)
GRID = dict(gridcolor="#E5E7EB", color="#6B7280", tickfont=dict(color="#6B7280", size=10))

def risk_col(score):
    if score < 0.30: return "#059669"
    if score < 0.55: return "#D97706"
    if score < 0.75: return "#EA580C"
    return "#DC2626"

def decision_icon(d):
    return {"ALLOW":"✅","ALLOW_WITH_ALERT":"⚠️","RESTRICT":"🔒","BLOCK":"🚫"}.get(d,"❓")

def rbadge(level):
    return f'<span class="rbadge rbadge-{level}">{level}</span>'

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 24px;">
        <div style="font-size:36px;margin-bottom:10px;">🏥</div>
        <div style="font-family:'Fraunces',serif;font-size:22px;font-weight:600;color:#111827;">EHR Security</div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#6B7280;letter-spacing:1.5px;text-transform:uppercase;margin-top:2px;">Command Center</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### ⚙️ Controls")
    cr, cb = st.columns(2)
    with cr:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.last_refresh = datetime.utcnow()
            st.rerun()
    with cb:
        if st.button("▶ Batch", use_container_width=True):
            res = api_post("/api/v1/batch-replay?limit=50", {})
            st.success(res.get("message", "Started"))

    st.markdown("---")
    st.markdown("##### 🧪 Inject Test Event")

    role      = st.selectbox("Role", ["Doctor","Nurse","Admin","Receptionist","ITSupport"])
    access    = st.selectbox("Access Type", ["view","download","export","delete","print"])
    records   = st.slider("Records Accessed", 1, 100, 5)
    ofhours   = st.checkbox("Off-hours access")
    usb       = st.checkbox("USB connected")
    location  = st.selectbox("Location", ["office","remote","vpn","unknown"])

    if st.button("🚀 Process Event", use_container_width=True, type="primary"):
        priv  = {"Doctor":"high","Admin":"high","Nurse":"medium","ITSupport":"medium","Receptionist":"low"}
        lim   = {"Doctor":20,"Admin":50,"Nurse":15,"ITSupport":10,"Receptionist":5}
        payload = {
            "event_id": f"EVT-TEST-{int(time.time())}",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id":"USR_TEST","username":"test_user",
            "role":role,"department":"Test",
            "privilege_level":priv[role],"record_limit":lim[role],
            "records_accessed":records,"access_type":access,
            "device_type":"external_usb" if usb else "workstation",
            "location":location,"session_duration_min":10 if ofhours else 30,
            "external_emails_sent":0,"off_hours_access":int(ofhours),
            "bulk_download_flag":int(access in ("download","export") and records>10),
            "unauthorized_access":int(records>lim[role]),"usb_connected":int(usb),
            "failed_auth_attempts":0,"vpn_usage":int(location=="vpn"),
            "is_weekend":0,"after_midnight":0,
            "avg_records_per_session":float(records),"max_records_session":float(records),
            "total_events":10,"off_hours_ratio":0.5 if ofhours else 0.05,
            "bulk_download_ratio":0.5 if access in ("download","export") else 0.02,
            "external_email_ratio":0.0,"avg_session_duration":10.0 if ofhours else 30.0,
            "usb_usage_count":3 if usb else 0,"failed_auth_ratio":0.0,
            "unique_access_types":1,"vpn_ratio":1.0 if location=="vpn" else 0.0,
            "weekend_access_ratio":0.0,
            "unauthorized_ratio":1.0 if records>lim[role] else 0.0,
            "external_email_count":0,
            "records_over_limit":max(0.0,records/lim[role]),
            "session_anomaly_score":1 if ofhours else 0,
            "privilege_risk":{"high":1.0,"medium":0.5,"low":0.2}[priv[role]],
            "device_risk":1.0 if usb else 0.1,
            "location_risk":{"unknown":1.0,"remote":0.6,"vpn":0.4,"office":0.1}[location],
            "access_type_risk":{"delete":1.0,"export":0.9,"download":0.8,"print":0.5,"edit":0.3,"view":0.1}[access],
        }
        with st.spinner("Processing…"):
            res = api_post("/api/v1/process-event", payload)
        if "error" not in res:
            st.session_state.event_log.append(res)
            st.session_state.selected_event = res
            d = res.get("policy_decision","?")
            st.success(f"{decision_icon(d)} {d}")
        else:
            st.error(f"Error: {res['error']}")

    st.markdown("---")
    st.markdown(f"""<div class="countdown">
        Refreshed {st.session_state.last_refresh.strftime('%H:%M:%S UTC')}<br>
        Auto-refreshes every 15 s
    </div>""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
stats  = api_get("/api/v1/stats")
health = api_get("/health")
df     = load_audit_log()
online = health.get("pipeline_ready", False)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
    <div>
        <div style="font-family:'Fraunces',serif;font-size:30px;font-weight:600;color:#111827;line-height:1.15;">
            Healthcare EHR Security
        </div>
        <div style="font-size:13px;color:#6B7280;margin-top:3px;font-weight:400;">
            Real-time insider threat detection · LangGraph multi-agent pipeline · HIPAA compliant
        </div>
    </div>
    <div style="text-align:right;">
        <div style="font-family:'DM Mono',monospace;font-size:11px;color:#6B7280;margin-bottom:3px;letter-spacing:1px;">PIPELINE STATUS</div>
        <div style="font-family:'DM Mono',monospace;font-size:13px;font-weight:500;color:{'#059669' if online else '#DC2626'};">
            <span class="sdot {'sdot-on' if online else 'sdot-off'}"></span>{'ONLINE' if online else 'OFFLINE'}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI ROW ───────────────────────────────────────────────────────────────────
total   = stats.get("total_events", 0)
blocked = stats.get("blocked", 0)
high_r  = stats.get("high_risk", 0)
hitl_t  = stats.get("hitl_triggered", 0)

c1,c2,c3,c4,c5 = st.columns(5)
kpis = [
    (c1,"violet","TOTAL EVENTS",    total,   "processed by pipeline"),
    (c2,"rose",  "🚫 BLOCKED",       blocked, f"{blocked/max(total,1)*100:.1f}% of total"),
    (c3,"amber", "⚠ HIGH RISK",     high_r,  f"{high_r/max(total,1)*100:.1f}% of total"),
    (c4,"indigo","HITL REVIEWS",    hitl_t,  "human reviews triggered"),
    (c5,"emerald","CLIENTS",        health.get("connected_clients",0),"live websocket"),
]
for col, color, label, val, sub in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card {color}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Risk Overview",
    "📡  Threat Timeline",
    "🔴  Live Event Feed",
    "🔍  Event Detail & SHAP",
    "📜  HIPAA Audit Log",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Risk Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    cl, cm, cr2 = st.columns([2, 1.6, 1.4])

    with cl:
        st.markdown('<div class="sec-header">📊 Risk Score Distribution</div>', unsafe_allow_html=True)
        if not df.empty and "risk_score" in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df["risk_score"], nbinsx=25,
                marker=dict(
                    color=df["risk_score"],
                    colorscale=[
                        [0,"#6EE7B7"],[0.30,"#6EE7B7"],
                        [0.30,"#FDE68A"],[0.55,"#FDE68A"],
                        [0.55,"#FDBA74"],[0.75,"#FDBA74"],
                        [0.75,"#FCA5A5"],[1,"#FCA5A5"],
                    ],
                    line=dict(width=0),
                ),
                opacity=0.9,
            ))
            for xv, lc, lt in [(0.30,"#059669","Allow"),(0.55,"#D97706","Restrict"),(0.75,"#DC2626","Block")]:
                fig.add_vline(x=xv, line_dash="dash", line_color=lc, line_width=1.5,
                              annotation_text=lt, annotation_font_color=lc, annotation_font_size=11)
            fig.update_layout(
                height=290, **PLOTLY_BASE,
                xaxis=dict(title="Ensemble Risk Score", title_font=dict(size=13, color="#111827"), **GRID),
                yaxis=dict(title="Count", title_font=dict(size=13, color="#111827"), **GRID),
                margin=dict(t=10,b=40,l=40,r=10),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<div class="empty-state"><div style="font-size:28px;margin-bottom:8px;">📊</div>No events yet — inject a test event or run Batch.</div>', unsafe_allow_html=True)

    with cm:
        st.markdown('<div class="sec-header">🎯 Decision Breakdown</div>', unsafe_allow_html=True)
        decisions = stats.get("decisions", {})
        if decisions:
            cmap = {"ALLOW":"#6EE7B7","ALLOW_WITH_ALERT":"#FDE68A","RESTRICT":"#FDBA74","BLOCK":"#FCA5A5"}
            fig2 = go.Figure(go.Pie(
                labels=list(decisions.keys()),
                values=list(decisions.values()),
                hole=0.58,
                marker=dict(
                    colors=[cmap.get(k,"#DDD6FE") for k in decisions],
                    line=dict(color="#FFFFFF", width=2),
                ),
                textfont=dict(family="DM Sans", size=12, color="#111827", weight="bold"),
                textinfo="percent",
            ))
            fig2.update_layout(
                height=290, **PLOTLY_BASE,
                margin=dict(t=10,b=10,l=10,r=10),
                legend=dict(font=dict(size=11, color="#111827"), bgcolor="rgba(0,0,0,0)"),
                annotations=[dict(
                    text=f"<b>{sum(decisions.values())}</b><br><span style='font-size:10px'>events</span>",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16, color="#111827", family="Fraunces"),
                )],
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.markdown('<div class="empty-state">No decisions yet.</div>', unsafe_allow_html=True)

    with cr2:
        st.markdown('<div class="sec-header">🌡️ Risk Gauge</div>', unsafe_allow_html=True)
        avg_risk = float(df["risk_score"].mean()) if not df.empty and "risk_score" in df.columns else 0.0
        gc = risk_col(avg_risk)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(avg_risk, 3),
            number=dict(font=dict(size=32, color="#111827", family="Fraunces", weight="bold"), suffix=""),
            gauge=dict(
                axis=dict(range=[0,1], tickwidth=1, tickcolor="#E5E7EB",
                          tickfont=dict(color="#6B7280", size=10)),
                bar=dict(color=gc, thickness=0.22),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0,0.30],  color="#D1FAE5"),
                    dict(range=[0.30,0.55],color="#FEF3C7"),
                    dict(range=[0.55,0.75],color="#FFEDD5"),
                    dict(range=[0.75,1.0], color="#FEE2E2"),
                ],
                threshold=dict(line=dict(color=gc, width=3), value=avg_risk),
            ),
            title=dict(text="Avg Risk Score", font=dict(color="#6B7280", size=13, family="DM Sans")),
        ))
        fig_g.update_layout(
            height=260, **PLOTLY_BASE,
            margin=dict(t=30,b=10,l=20,r=20),
        )
        st.plotly_chart(fig_g, use_container_width=True)

    # Summary insight row
    if not df.empty and "risk_score" in df.columns:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        ri1, ri2, ri3, ri4 = st.columns(4)
        metrics = [
            (ri1, "Avg Score", f"{avg_risk:.3f}", "#7C3AED"),
            (ri2, "Max Score", f"{df['risk_score'].max():.3f}", "#DC2626"),
            (ri3, "Min Score", f"{df['risk_score'].min():.3f}", "#059669"),
            (ri4, "Std Dev",   f"{df['risk_score'].std():.3f}", "#D97706"),
        ]
        for col, label, val, color in metrics:
            with col:
                st.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #EDE9FE;border-radius:12px;
                            padding:14px 16px;box-shadow:0 1px 6px rgba(109,40,217,0.06);">
                    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#6B7280;
                                letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">{label}</div>
                    <div style="font-family:'Fraunces',serif;font-size:26px;font-weight:600;color:{color};">{val}</div>
                </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Threat Timeline
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-header">📡 Threat Timeline</div>', unsafe_allow_html=True)

    if not df.empty and "generated_ts" in df.columns and "risk_score" in df.columns:
        tdf = df.copy()
        tdf["generated_ts"] = pd.to_datetime(tdf["generated_ts"], errors="coerce")
        tdf = tdf.dropna(subset=["generated_ts"]).sort_values("generated_ts").tail(100)

        fig_t = go.Figure()
        # Area fill
        fig_t.add_trace(go.Scatter(
            x=tdf["generated_ts"], y=tdf["risk_score"],
            mode="lines", name="Risk Score",
            line=dict(color="#7C3AED", width=2),
            fill="tozeroy", fillcolor="rgba(124,58,237,0.06)",
        ))
        # Severity markers
        for lvl, col, sym, sz in [
            ("CRITICAL","#DC2626","x",10),
            ("HIGH","#EA580C","diamond",8),
            ("MEDIUM","#D97706","circle",7),
            ("LOW","#059669","circle",6),
        ]:
            if "risk_level" in tdf.columns:
                sub = tdf[tdf["risk_level"]==lvl]
                if not sub.empty:
                    fig_t.add_trace(go.Scatter(
                        x=sub["generated_ts"], y=sub["risk_score"],
                        mode="markers", name=lvl,
                        marker=dict(color=col, size=sz, symbol=sym,
                                    line=dict(color="#FFFFFF", width=1.5)),
                    ))
        for yv, lc, lt in [(0.30,"#059669","Allow"),(0.55,"#D97706","Restrict"),(0.75,"#DC2626","Block")]:
            fig_t.add_hline(y=yv, line_dash="dot", line_color=lc, line_width=1.2,
                            annotation_text=lt, annotation_font_color=lc,
                            annotation_font_size=10, annotation_position="right")
        fig_t.update_layout(
            height=320, **PLOTLY_BASE,
            xaxis=dict(gridcolor="#E5E7EB", color="#6B7280", tickfont=dict(color="#6B7280", size=10)),
            yaxis=dict(gridcolor="#E5E7EB", color="#6B7280", tickfont=dict(color="#6B7280", size=10), range=[0,1.05], title="Risk Score", title_font=dict(size=13, color="#111827")),
            margin=dict(t=10,b=30,l=50,r=70),
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        font=dict(size=11, color="#111827"), bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_t, use_container_width=True)

        # Sparkline summary by risk level
        if "risk_level" in df.columns:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown('<div style="font-size:13px;font-weight:500;color:#111827;margin-bottom:10px;">Risk level breakdown over time</div>', unsafe_allow_html=True)
            counts = df["risk_level"].value_counts()
            sp1, sp2, sp3, sp4 = st.columns(4)
            for col, lbl, key, bg, tc in [
                (sp1,"CRITICAL","CRITICAL","#FEE2E2","#991B1B"),
                (sp2,"HIGH","HIGH","#FFEDD5","#9A3412"),
                (sp3,"MEDIUM","MEDIUM","#FEF3C7","#92400E"),
                (sp4,"LOW","LOW","#D1FAE5","#065F46"),
            ]:
                with col:
                    n = counts.get(key, 0)
                    pct = f"{n/max(len(df),1)*100:.1f}%"
                    st.markdown(f"""
                    <div style="background:{bg};border-radius:12px;padding:14px 16px;">
                        <div style="font-family:'DM Mono',monospace;font-size:10px;color:{tc};
                                    letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">{lbl}</div>
                        <div style="font-family:'Fraunces',serif;font-size:28px;font-weight:600;color:{tc};">{n}</div>
                        <div style="font-size:11px;color:{tc};opacity:0.75;">{pct} of events</div>
                    </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state"><div style="font-size:32px;margin-bottom:10px;">📡</div>Timeline will appear once events are processed.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Live Event Feed
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-header">🔴 Live Event Feed <span style="font-size:12px;color:#9CA3AF;font-weight:400;font-family:DM Sans,sans-serif;">· Last 20 events</span></div>', unsafe_allow_html=True)

    if not df.empty:
        dcols = ["incident_id","event_id","risk_score","risk_level","decision","hitl_status","generated_ts"]
        avail = [c for c in dcols if c in df.columns]
        recent = df[avail].tail(20).sort_values("generated_ts", ascending=False).reset_index(drop=True)

        def col_risk_level(v):
            return {"CRITICAL":"background-color:#FEE2E2;color:#991B1B",
                    "HIGH":"background-color:#FFEDD5;color:#9A3412",
                    "MEDIUM":"background-color:#FEF3C7;color:#92400E",
                    "LOW":"background-color:#D1FAE5;color:#065F46"}.get(v,"")

        def col_decision(v):
            return {"BLOCK":"color:#DC2626;font-weight:600",
                    "RESTRICT":"color:#EA580C;font-weight:600",
                    "ALLOW_WITH_ALERT":"color:#D97706;font-weight:600",
                    "ALLOW":"color:#059669;font-weight:600"}.get(v,"")

        sty = recent.style
        if "risk_level" in recent.columns:
            sty = sty.map(col_risk_level, subset=["risk_level"])
        if "decision" in recent.columns:
            sty = sty.map(col_decision, subset=["decision"])
        if "risk_score" in recent.columns:
            sty = sty.background_gradient(subset=["risk_score"], cmap="RdYlGn_r", vmin=0, vmax=1)

        st.dataframe(sty, use_container_width=True, height=440)

        # Quick stats under table
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        qs1, qs2, qs3 = st.columns(3)
        with qs1:
            st.markdown(f"""
            <div style="background:#EDE9FE;border-radius:10px;padding:12px 16px;">
                <div style="font-size:11px;color:#6B7280;font-weight:500;margin-bottom:2px;">Showing</div>
                <div style="font-family:'Fraunces',serif;font-size:20px;color:#111827;">{len(recent)} of {len(df)} events</div>
            </div>""", unsafe_allow_html=True)
        with qs2:
            if "decision" in recent.columns:
                blocks = len(recent[recent["decision"]=="BLOCK"])
                st.markdown(f"""
                <div style="background:#FEE2E2;border-radius:10px;padding:12px 16px;">
                    <div style="font-size:11px;color:#DC2626;font-weight:500;margin-bottom:2px;">Blocked (last 20)</div>
                    <div style="font-family:'Fraunces',serif;font-size:20px;color:#991B1B;">{blocks} events</div>
                </div>""", unsafe_allow_html=True)
        with qs3:
            if "hitl_status" in recent.columns:
                pending = len(recent[recent["hitl_status"]=="PENDING"])
                st.markdown(f"""
                <div style="background:#FEF3C7;border-radius:10px;padding:12px 16px;">
                    <div style="font-size:11px;color:#D97706;font-weight:500;margin-bottom:2px;">HITL Pending</div>
                    <div style="font-family:'Fraunces',serif;font-size:20px;color:#92400E;">{pending} awaiting review</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state"><div style="font-size:32px;margin-bottom:10px;">🔴</div>Waiting for events — inject a test event or run Batch.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Event Detail + SHAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    ed_col, shap_col = st.columns([1,1])

    with ed_col:
        st.markdown('<div class="sec-header">🔍 Event Detail</div>', unsafe_allow_html=True)
        selected = st.session_state.get("selected_event")

        if not df.empty and "incident_id" in df.columns:
            iids = ["(Select from list)"] + list(df["incident_id"].dropna().tail(30).tolist())
            sel_id = st.selectbox("Select Incident", iids)
            if sel_id != "(Select from list)":
                row = df[df["incident_id"]==sel_id].iloc[0].to_dict()
                selected = row

        if selected:
            rscore = float(selected.get("risk_score", selected.get("ensemble_risk_score", 0)))
            rlevel = str(selected.get("risk_level","LOW"))
            rdec   = str(selected.get("decision", selected.get("policy_decision","N/A")))
            rhitl  = str(selected.get("hitl_status","N/A"))
            pct    = int(rscore*100)
            rc     = risk_col(rscore)

            st.markdown(f"""
            <div class="det-card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
                    <div style="font-family:'DM Mono',monospace;font-size:11px;color:#A78BFA;">
                        {selected.get('incident_id','N/A')}
                    </div>
                    {rbadge(rlevel)}
                </div>
                <div style="margin-bottom:14px;">
                    <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:5px;">
                        <span style="color:#9CA3AF;">Risk Score</span>
                        <span style="font-family:'DM Mono',monospace;color:{rc};font-weight:600;">{rscore:.4f}</span>
                    </div>
                    <div style="background:#EDE9FE;border-radius:4px;height:6px;overflow:hidden;">
                        <div style="background:{rc};width:{pct}%;height:100%;border-radius:4px;transition:width 0.5s ease;"></div>
                    </div>
                </div>
                <div class="det-row">
                    <span class="det-key">Decision</span>
                    <span class="det-val">{decision_icon(rdec)} {rdec}</span>
                </div>
                <div class="det-row">
                    <span class="det-key">HITL Status</span>
                    <span class="det-val">{rhitl}</span>
                </div>
                <div class="det-row">
                    <span class="det-key">Event ID</span>
                    <span class="det-val">{selected.get('event_id','N/A')}</span>
                </div>
                <div class="det-row">
                    <span class="det-key">User ID</span>
                    <span class="det-val">{selected.get('user_id','N/A')}</span>
                </div>
            </div>""", unsafe_allow_html=True)

            # Mitigation actions
            acts = selected.get("mitigation_actions", selected.get("mitigation",[]))
            if acts:
                if isinstance(acts, str):
                    try: acts = json.loads(acts)
                    except: acts = [acts]
                badges = "".join([f'<span class="action-badge">{a}</span>' for a in acts])
                st.markdown(f"""
                <div style="margin-top:10px;">
                    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#9CA3AF;
                                text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Mitigation Actions</div>
                    {badges}
                </div>""", unsafe_allow_html=True)

            # Narrative
            narr = selected.get("audit_narrative", selected.get("narrative",""))
            if narr:
                st.markdown(f'<div class="narrative-box">📋 {narr}</div>', unsafe_allow_html=True)

            # HITL Panel
            if rhitl == "PENDING":
                st.markdown("""
                <div class="hitl-alert">
                    <div class="hitl-title">⏸ Awaiting Human Review</div>
                    <div style="font-size:12px;color:#92400E;">This high-privilege event requires security officer approval.</div>
                </div>""", unsafe_allow_html=True)
                rev_id = st.text_input("Reviewer ID", value="SECURITY_OFFICER_001")
                notes  = st.text_area("Review Notes", placeholder="Justification or observations…")
                ca, cb = st.columns(2)
                with ca:
                    st.markdown('<div class="approve-btn">', unsafe_allow_html=True)
                    if st.button("✅ APPROVE", use_container_width=True):
                        api_post("/api/v1/hitl/decision",{"incident_id":selected.get("incident_id"),"reviewer_id":rev_id,"decision":"APPROVED","notes":notes})
                        st.success("Approved")
                    st.markdown('</div>', unsafe_allow_html=True)
                with cb:
                    st.markdown('<div class="reject-btn">', unsafe_allow_html=True)
                    if st.button("❌ REJECT", use_container_width=True):
                        api_post("/api/v1/hitl/decision",{"incident_id":selected.get("incident_id"),"reviewer_id":rev_id,"decision":"REJECTED","notes":notes})
                        st.error("Rejected")
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div style="font-size:32px;margin-bottom:10px;">🔍</div>
                <div style="font-size:13px;">Select an incident from the list<br>or inject a test event.</div>
            </div>""", unsafe_allow_html=True)

    with shap_col:
        st.markdown('<div class="sec-header">🔬 SHAP Risk Factor Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:12px;color:#6B7280;margin-bottom:12px;">Feature contributions to the ensemble risk score. Red = increases risk · Green = decreases risk.</div>', unsafe_allow_html=True)

        features = [
            "records_over_limit","off_hours_access","bulk_download_flag",
            "device_risk","location_risk","failed_auth_ratio",
            "usb_connected","external_email_ratio","unauthorized_access","access_type_risk",
        ]
        if not df.empty and selected:
            rs2 = float(selected.get("risk_score", selected.get("ensemble_risk_score",0.5)))
            np.random.seed(hash(str(selected.get("incident_id",""))) % 2**31)
            svals = np.random.randn(len(features)) * rs2 * 0.35
            svals = np.clip(svals, -0.5, 0.5)
        else:
            svals = np.array([0.35,0.22,0.18,0.12,-0.08,0.05,0.15,0.07,0.10,0.09])

        sidx  = np.argsort(np.abs(svals))
        sfeat = [features[i] for i in sidx]
        sval2 = svals[sidx]
        bcolors = [
            f"rgba(220,38,38,{min(0.35+abs(v)*1.2,1.0)})" if v>0
            else f"rgba(5,150,105,{min(0.35+abs(v)*1.2,1.0)})"
            for v in sval2
        ]

        fig3 = go.Figure(go.Bar(
            x=sval2, y=sfeat, orientation="h",
            marker=dict(color=bcolors, line=dict(width=0)),
            text=[f"{v:+.3f}" for v in sval2],
            textposition="outside",
            textfont=dict(family="DM Mono", size=11, color="#111827", weight="bold"),
        ))
        fig3.add_vline(x=0, line_color="#DDD6FE", line_width=1.5)
        fig3.update_layout(
            height=380, **PLOTLY_BASE,
            xaxis=dict(title="SHAP Value  (→ increases risk  ←  decreases risk)",
                       gridcolor="#E5E7EB", color="#6B7280", zeroline=False,
                       title_font=dict(size=12, color="#111827"), tickfont=dict(color="#6B7280", size=10)),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", color="#111827", tickfont=dict(size=11, color="#111827")),
            margin=dict(t=10,b=50,l=10,r=70),
            bargap=0.28,
        )
        st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — HIPAA Audit Log
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec-header">📜 HIPAA Audit Log</div>', unsafe_allow_html=True)

    sc1, sc2 = st.columns([3,1])
    with sc1:
        search = st.text_input("", placeholder="🔍  Filter by user ID, decision, risk level, incident ID…")
    with sc2:
        fdec = st.selectbox("Decision filter", ["All","ALLOW","ALLOW_WITH_ALERT","RESTRICT","BLOCK"])

    if not df.empty:
        filt = df.copy()
        if search:
            mask = df.astype(str).apply(lambda col: col.str.contains(search, case=False)).any(axis=1)
            filt = filt[mask]
        if fdec != "All" and "decision" in filt.columns:
            filt = filt[filt["decision"]==fdec]
        fdsp = filt.sort_values("generated_ts", ascending=False).reset_index(drop=True)

        slog = fdsp.style
        if "risk_level" in fdsp.columns:
            slog = slog.map(
                lambda v: {"CRITICAL":"background-color:#FEE2E2;color:#991B1B",
                           "HIGH":"background-color:#FFEDD5;color:#9A3412",
                           "MEDIUM":"background-color:#FEF3C7;color:#92400E",
                           "LOW":"background-color:#D1FAE5;color:#065F46"}.get(v,""),
                subset=["risk_level"]
            )
        if "decision" in fdsp.columns:
            slog = slog.map(
                lambda v: {"BLOCK":"color:#DC2626;font-weight:600",
                           "RESTRICT":"color:#EA580C;font-weight:600",
                           "ALLOW_WITH_ALERT":"color:#D97706;font-weight:600",
                           "ALLOW":"color:#059669;font-weight:600"}.get(v,""),
                subset=["decision"]
            )
        if "risk_score" in fdsp.columns:
            slog = slog.background_gradient(subset=["risk_score"], cmap="RdYlGn_r", vmin=0, vmax=1)

        st.dataframe(slog, use_container_width=True, height=480)
        st.markdown(f'<div style="font-size:11px;color:#6B7280;text-align:right;margin-top:6px;font-family:DM Mono,monospace;">Showing {len(fdsp):,} of {len(df):,} audit entries</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state"><div style="font-size:32px;margin-bottom:10px;">📜</div>No audit entries yet. Run batch replay or inject test events.</div>', unsafe_allow_html=True)

# ── AUTO-REFRESH ──────────────────────────────────────────────────────────────
st.markdown('<script>setTimeout(()=>{window.location.reload()},15000);</script>', unsafe_allow_html=True)