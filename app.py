import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AIR-E | Aadhaar Integrity & Risk Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROCESSED_FILE = "processed_data.parquet"

def load_data():
    if not os.path.exists(PROCESSED_FILE):
        return pd.DataFrame()
    return pd.read_parquet(PROCESSED_FILE)

# ═══════════════════════════════════════════════════════════════════════════════
# GOOGLE FONTS + MASTER CSS INJECTION
# ═══════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800;900&family=DM+Mono:wght@400;500&family=IBM+Plex+Mono:wght@400;500;600&family=Outfit:wght@300;400;500;600&family=Orbitron:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;900&family=IBM+Plex+Mono:wght@400;500&family=Orbitron:wght@700;900&family=Outfit:wght@400;500&display=swap');

    /* ── CSS VARIABLES ── */
    :root {
        --void: #04060a;
        --surface-1: #080d14;
        --surface-2: #0d1420;
        --surface-3: #111c2e;
        --glass: rgba(13,20,33,0.55);
        --border: rgba(88,166,255,0.12);
        --border-hot: rgba(255,123,114,0.35);
        --ion-blue: #58a6ff;
        --cyber-violet: #bc8cff;
        --crimson: #ff7b72;
        --aurora-teal: #39d0c4;
        --solar-amber: #f0a84c;
        --ghost-white: #e6edf3;
        --muted: #7d8590;
        --deep-muted: #3d444d;
        --glow-blue: 0 0 24px rgba(88,166,255,0.4), 0 0 64px rgba(88,166,255,0.12);
        --glow-violet: 0 0 24px rgba(188,140,255,0.4);
        --glow-red: 0 0 32px rgba(255,123,114,0.5);
        --glow-teal: 0 0 20px rgba(57,208,196,0.35);
    }

    /* ── KEYFRAME ANIMATIONS ── */
    @keyframes float-idle {
        0%,100% { transform: translateY(0px) rotate(0deg); }
        33%     { transform: translateY(-5px) rotate(0.1deg); }
        66%     { transform: translateY(-2px) rotate(-0.1deg); }
    }
    @keyframes border-pulse {
        0%,100% { box-shadow: 0 0 16px rgba(255,123,114,0.3); }
        50%      { box-shadow: 0 0 40px rgba(255,123,114,0.7); }
    }
    @keyframes page-enter {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    @keyframes pulse-dot {
        0%,100% { opacity: 1; transform: scale(1); }
        50%      { opacity: 0.6; transform: scale(0.8); }
    }

    /* ── GLOBAL OVERRIDES ── */
    .stApp {
        background: var(--void) !important;
        color: var(--ghost-white) !important;
        font-family: 'Outfit', sans-serif !important;
        background-image:
            radial-gradient(ellipse 80% 50% at 5% 10%, rgba(88,166,255,0.06) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 95% 90%, rgba(188,140,255,0.05) 0%, transparent 50%),
            radial-gradient(ellipse 40% 30% at 80% 5%, rgba(255,123,114,0.04) 0%, transparent 40%) !important;
    }
    .stApp::before {
        content: '';
        position: fixed; inset: 0;
        background-image: repeating-linear-gradient(
            0deg, transparent, transparent 2px,
            rgba(88,166,255,0.012) 2px, rgba(88,166,255,0.012) 3px
        );
        pointer-events: none; z-index: 9999;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .main .block-container {
        padding-top: 1.5rem !important;
        max-width: 100% !important;
        animation: page-enter 0.5s cubic-bezier(0.23,1,0.32,1) !important;
    }

    /* ── CUSTOM SCROLLBAR ── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: var(--surface-1); }
    ::-webkit-scrollbar-thumb { background: rgba(88,166,255,0.4); border-radius: 2px; }

    /* ── SIDEBAR REDESIGN ── */
    section[data-testid="stSidebar"] {
        background: #080d14 !important;
        border-right: 1px solid rgba(88,166,255,0.1) !important;
    }
    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }
    .sidebar-brand {
        padding: 24px 16px 8px;
        font-family: 'Syne', sans-serif;
        font-weight: 900; font-size: 24px;
        color: #e6edf3;
        letter-spacing: 0.05em;
    }
    .sidebar-fullform {
        font-family: 'IBM Plex Mono'; font-size: 9px;
        color: #58a6ff; letter-spacing: 0.1em;
        padding: 0 16px 20px;
        text-transform: uppercase;
        opacity: 0.8;
        line-height: 1.4;
    }
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(88,166,255,0.3), transparent);
        margin: 8px 16px 16px;
    }

    /* ── SIDEBAR NAVIGATION STYLE OVERRIDES ── */
    div[data-testid="stSidebar"] .stRadio > label {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: var(--muted) !important;
        padding-left: 16px !important;
        margin-bottom: 8px !important;
    }
    div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        padding: 0 12px !important;
    }
    div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        padding: 12px 14px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 12px !important;
        color: #7d8590 !important;
        letter-spacing: 0.05em !important;
        transition: all 0.25s cubic-bezier(0.23,1,0.32,1) !important;
        border-left: 3px solid transparent !important;
        margin-bottom: 4px !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        color: #58a6ff !important;
        background: rgba(88,166,255,0.06) !important;
        border-left-color: rgba(88,166,255,0.5) !important;
        transform: translateX(3px) !important;
    }
    div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"],
    div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[aria-checked="true"],
    div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
        color: #58a6ff !important;
        background: rgba(88,166,255,0.1) !important;
        border-left-color: #58a6ff !important;
        box-shadow: inset 0 0 20px rgba(88,166,255,0.05) !important;
    }
    div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        margin: 0 !important;
    }
    /* Hide the default radio circle indicator */
    div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label [data-testid="stWidgetLabel"]::before,
    div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span[class*="StyledCircle"] {
        display: none !important;
    }
    div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label input[type="radio"] {
        position: absolute;
        opacity: 0;
        width: 0; height: 0;
    }

    /* ── GLASS PANELS ── */
    .glass-panel {
        background: rgba(13,20,33,0.4) !important;
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border: 1px solid rgba(88,166,255,0.12);
        border-radius: 20px;
        padding: 28px 32px;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.03) inset, 0 24px 64px rgba(0,0,0,0.4);
    }
    .glass-panel hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--ion-blue), transparent);
        margin: 20px 0;
    }

    /* ── PLOTLY CHARTS OVERRIDES ── */
    .js-plotly-plot {
        border-radius: 12px !important;
        border: 1px solid rgba(88,166,255,0.12) !important;
        overflow: hidden !important;
    }

    /* ── KPI CARD OVERRIDES (Streamlit & Custom HTML) ── */
    div[data-testid="metric-container"],
    .kpi-card {
        background: rgba(13,20,33,0.6) !important;
        border: 1px solid rgba(88,166,255,0.15) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        backdrop-filter: blur(16px) !important;
        transition: transform 0.4s cubic-bezier(0.23,1,0.32,1),
                    box-shadow 0.4s ease !important;
        animation: float-idle 6s ease-in-out infinite !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
    }
    div[data-testid="metric-container"]:hover,
    .kpi-card:hover {
        transform: translateY(-6px) !important;
    }
    div[data-testid="metric-container"] label,
    .kpi-card .kpi-label {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 10px !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: #7d8590 !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"],
    .kpi-card .kpi-value {
        font-family: 'Orbitron', sans-serif !important;
        color: #58a6ff !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    /* Custom KPI glows */
    .card-enrollments {
        border-top: 2px solid #58a6ff !important;
        box-shadow: 0 0 24px rgba(88,166,255,0.2), inset 0 1px 0 rgba(88,166,255,0.1) !important;
    }
    .card-enrollments div[data-testid="stMetricValue"],
    .card-enrollments .kpi-value {
        color: #58a6ff !important;
    }

    .card-anomalies {
        border-top: 2px solid #f0a84c !important;
        box-shadow: 0 0 24px rgba(240,168,76,0.25) !important;
    }
    .card-anomalies div[data-testid="stMetricValue"],
    .card-anomalies .kpi-value {
        color: #f0a84c !important;
    }

    .card-risk-teal {
        border-top: 2px solid #39d0c4 !important;
        box-shadow: 0 0 24px rgba(57,208,196,0.2) !important;
    }
    .card-risk-teal div[data-testid="stMetricValue"],
    .card-risk-teal .kpi-value {
        color: #39d0c4 !important;
    }

    .card-risk-amber {
        border-top: 2px solid #f0a84c !important;
        box-shadow: 0 0 24px rgba(240,168,76,0.25) !important;
    }
    .card-risk-amber div[data-testid="stMetricValue"],
    .card-risk-amber .kpi-value {
        color: #f0a84c !important;
    }

    .card-risk-crimson {
        border-top: 2px solid #ff7b72 !important;
        box-shadow: 0 0 24px rgba(255,123,114,0.3) !important;
    }
    .card-risk-crimson div[data-testid="stMetricValue"],
    .card-risk-crimson .kpi-value {
        color: #ff7b72 !important;
    }

    .card-critical-active {
        border-top: 2px solid #ff7b72 !important;
        animation: border-pulse 2s ease-in-out infinite !important;
    }
    .card-critical-active div[data-testid="stMetricValue"],
    .card-critical-active .kpi-value {
        color: #ff7b72 !important;
    }

    /* Float Animation delays */
    .stColumn:nth-child(1) div[data-testid="metric-container"],
    .kpi-row > div:nth-child(1) { animation-delay: 0s !important; }
    .stColumn:nth-child(2) div[data-testid="metric-container"],
    .kpi-row > div:nth-child(2) { animation-delay: 1.5s !important; }
    .stColumn:nth-child(3) div[data-testid="metric-container"],
    .kpi-row > div:nth-child(3) { animation-delay: 3s !important; }
    .stColumn:nth-child(4) div[data-testid="metric-container"],
    .kpi-row > div:nth-child(4) { animation-delay: 4.5s !important; }

    /* ── SLIDERS ── */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #58a6ff, #bc8cff) !important;
    }
    .stSlider [data-testid="stThumb"] {
        background: #58a6ff !important;
        box-shadow: 0 0 12px rgba(88,166,255,0.6) !important;
        border: 2px solid #e6edf3 !important;
    }

    /* ── TYPOGRAPHY ── */
    h1 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }
    h2 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        color: #e6edf3 !important;
    }
    h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        color: #58a6ff !important;
        font-size: 13px !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }

    /* ── SELECT BOXES ── */
    .stSelectbox div[data-baseweb="select"] {
        background: var(--surface-2) !important;
        border-color: var(--border) !important;
        border-radius: 10px !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }

    /* ── HERO BAND ── */
    .hero-band {
        background: linear-gradient(135deg, rgba(8,13,20,0.8), rgba(13,20,33,0.4));
        border: 1px solid rgba(88,166,255,0.15);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }
    .hero-band::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, #58a6ff, transparent);
    }
    .hero-logo-text {
        font-family: 'Syne', sans-serif;
        font-weight: 900; font-size: 3.5rem;
        background: linear-gradient(135deg, var(--ion-blue), var(--cyber-violet));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 4px;
        line-height: 1.1;
    }
    .hero-fullform {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.95rem; font-weight: 500;
        color: var(--ion-blue);
        letter-spacing: 0.25em;
        text-transform: uppercase;
        margin-top: 8px;
    }
    .hero-subtitle {
        font-family: 'Outfit', sans-serif;
        font-size: 0.85rem; font-weight: 500;
        color: var(--muted);
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: 14px;
    }
    .hero-live-badge {
        position: absolute; top: 32px; right: 40px;
        background: rgba(63,185,80,0.12);
        border: 1px solid #3fb950;
        border-radius: 20px;
        padding: 6px 14px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem; font-weight: 600;
        color: #3fb950;
        letter-spacing: 0.08em;
    }

    /* ── FLAGGED STATE CARD ── */
    .flagged-state-card {
        display: flex; align-items: center; justify-content: space-between;
        padding: 14px 20px; background: rgba(13,20,33,0.4);
        border: 1px solid rgba(88,166,255,0.08); border-radius: 12px;
        margin-bottom: 10px; transition: all 0.3s ease;
        font-family: 'IBM Plex Mono', monospace;
    }
    .flagged-state-card:hover {
        border-color: rgba(88,166,255,0.25);
        background: rgba(13,20,33,0.7);
        transform: translateX(4px);
    }
    .flagged-state-card .rank {
        color: var(--muted); font-size: 0.85rem; font-weight: 600; width: 32px;
    }
    .flagged-state-card .state-name {
        color: var(--ghost-white); font-size: 0.9rem; font-weight: 500; flex-grow: 1;
        font-family: 'Outfit', sans-serif;
    }
    .flagged-state-card .risk-badge {
        padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 700;
        font-family: 'Orbitron', monospace; margin-right: 14px;
    }
    .flagged-state-card .level-label {
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em; width: 70px; text-align: right;
    }

    /* ── AUDIT OPTIMIZER JS BUTTONS ── */
    .btn-freeze {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        background: rgba(255,123,114,0.1) !important;
        color: #ff7b72 !important;
        border: 1px solid rgba(255,123,114,0.3) !important;
        border-radius: 6px !important;
        padding: 6px 14px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        outline: none !important;
    }
    .btn-freeze:hover {
        background: rgba(255,123,114,0.25) !important;
        box-shadow: 0 0 12px rgba(255,123,114,0.3) !important;
    }

    /* ── WHAT-IF COMPLIANCE SUMMARY ── */
    .whatiif-summary {
        font-family: 'Outfit', sans-serif; font-size: 1rem; color: #e6edf3; line-height: 1.8;
    }

    /* ── RISK ALERT TOAST ── */
    .alert-toast {
        position: fixed; top: 16px; left: 50%; transform: translateX(-50%);
        background: rgba(255,123,114,0.12) !important; border: 1px solid #ff7b72 !important;
        border-radius: 8px; padding: 10px 24px;
        font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #ff7b72;
        z-index: 9999; animation: slide-in 0.4s ease;
        box-shadow: 0 0 32px rgba(255,123,114,0.3);
        backdrop-filter: blur(12px) !important;
    }
    @keyframes slide-in {
        from { top: -60px; opacity: 0; }
        to   { top: 16px; opacity: 1; }
    }
    </style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTICLE BACKGROUND COMPONENT
# ═══════════════════════════════════════════════════════════════════════════════
def render_particle_bg():
    components.html("""
    <canvas id="particle-canvas" style="position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:0;pointer-events:none;"></canvas>
    <script>
    const c=document.getElementById('particle-canvas'),x=c.getContext('2d');
    c.width=window.innerWidth;c.height=window.innerHeight;
    window.addEventListener('resize',()=>{c.width=window.innerWidth;c.height=window.innerHeight;});
    const P=[];for(let i=0;i<60;i++)P.push({x:Math.random()*c.width,y:Math.random()*c.height,r:Math.random()*1.5+0.5,dx:(Math.random()-0.5)*0.3,dy:(Math.random()-0.5)*0.2,o:Math.random()*0.3+0.05});
    function draw(){x.clearRect(0,0,c.width,c.height);P.forEach(p=>{x.beginPath();x.arc(p.x,p.y,p.r,0,Math.PI*2);x.fillStyle='rgba(88,166,255,'+p.o+')';x.fill();p.x+=p.dx;p.y+=p.dy;if(p.x<0||p.x>c.width)p.dx*=-1;if(p.y<0||p.y>c.height)p.dy*=-1;});requestAnimationFrame(draw);}draw();
    </script>
    """, height=0)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ═══════════════════════════════════════════════════════════════════════════════
ORBITAL_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(4,6,10,0.6)',
        font=dict(family='Outfit, sans-serif', color='#e6edf3', size=12),
        title_font=dict(family='Syne, sans-serif', size=20, color='#e6edf3'),
        xaxis=dict(gridcolor='rgba(88,166,255,0.06)', zerolinecolor='rgba(88,166,255,0.1)'),
        yaxis=dict(gridcolor='rgba(88,166,255,0.06)', zerolinecolor='rgba(88,166,255,0.1)'),
        colorway=['#58a6ff','#bc8cff','#39d0c4','#f0a84c','#ff7b72','#7ee787','#f778ba'],
        margin=dict(l=40, r=20, t=50, b=40),
    )
)

RISK_COLORSCALE = [[0,'#39d0c4'],[0.5,'#f0a84c'],[1,'#ff7b72']]


# ═══════════════════════════════════════════════════════════════════════════════
# THREE.JS GLOBE COMPONENT
# ═══════════════════════════════════════════════════════════════════════════════
def render_globe(height=420):
    globe_html = """
    <div id="globe-container" style="width:100%;height:"""+str(height)+"""px;position:relative;border-radius:16px;overflow:hidden;border:1px solid rgba(88,166,255,0.12);">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (function(){
    const container=document.getElementById('globe-container');
    const w=container.clientWidth,h="""+str(height)+""";
    const scene=new THREE.Scene();
    const camera=new THREE.PerspectiveCamera(45,w/h,0.1,1000);
    camera.position.z=2.8;
    const renderer=new THREE.WebGLRenderer({alpha:true,antialias:true});
    renderer.setSize(w,h);
    renderer.setClearColor(0x000000,0);
    container.appendChild(renderer.domElement);

    // Stars
    const starGeo=new THREE.BufferGeometry();
    const starPos=new Float32Array(3000);
    for(let i=0;i<3000;i++){starPos[i]=(Math.random()-0.5)*20;}
    starGeo.setAttribute('position',new THREE.BufferAttribute(starPos,3));
    const starMat=new THREE.PointsMaterial({color:0x58a6ff,size:0.02,transparent:true,opacity:0.6});
    scene.add(new THREE.Points(starGeo,starMat));

    // Globe
    const globeGeo=new THREE.SphereGeometry(1,64,64);
    const globeMat=new THREE.MeshPhongMaterial({
        color:0x080d14,transparent:true,opacity:0.85,
        emissive:0x0d1420,emissiveIntensity:0.3,
        wireframe:false
    });
    const globe=new THREE.Mesh(globeGeo,globeMat);
    scene.add(globe);

    // Wireframe overlay
    const wireMat=new THREE.MeshBasicMaterial({color:0x58a6ff,wireframe:true,transparent:true,opacity:0.08});
    const wire=new THREE.Mesh(new THREE.SphereGeometry(1.005,32,32),wireMat);
    scene.add(wire);

    // Atmospheric glow ring
    const ringGeo=new THREE.RingGeometry(1.15,1.25,64);
    const ringMat=new THREE.MeshBasicMaterial({color:0x58a6ff,transparent:true,opacity:0.12,side:THREE.DoubleSide});
    const ring=new THREE.Mesh(ringGeo,ringMat);
    scene.add(ring);

    // Risk nodes on globe surface
    const nodeColors=[0xff7b72,0xf0a84c,0x39d0c4,0xbc8cff,0x58a6ff];
    const coords=[[0.35,0.25],[0.38,0.2],[0.3,0.22],[0.33,0.28],[0.36,0.15],[0.32,0.3],[0.29,0.18],[0.37,0.23]];
    coords.forEach((c,i)=>{
        const phi=(0.5-c[0])*Math.PI;
        const theta=c[1]*Math.PI*2;
        const x=Math.cos(phi)*Math.cos(theta);
        const y=Math.sin(phi);
        const z=Math.cos(phi)*Math.sin(theta);
        const dot=new THREE.Mesh(
            new THREE.SphereGeometry(0.025,16,16),
            new THREE.MeshBasicMaterial({color:nodeColors[i%5],transparent:true,opacity:0.9})
        );
        dot.position.set(x*1.02,y*1.02,z*1.02);
        globe.add(dot);
    });

    // Lights
    scene.add(new THREE.AmbientLight(0x404060,0.5));
    const dl=new THREE.DirectionalLight(0x58a6ff,0.8);
    dl.position.set(5,3,5);scene.add(dl);

    let isDragging=false,prevX=0;
    container.addEventListener('mousedown',(e)=>{isDragging=true;prevX=e.clientX;});
    container.addEventListener('mouseup',()=>{isDragging=false;});
    container.addEventListener('mousemove',(e)=>{if(isDragging){globe.rotation.y+=(e.clientX-prevX)*0.005;wire.rotation.y=globe.rotation.y;prevX=e.clientX;}});

    function animate(){
        requestAnimationFrame(animate);
        if(!isDragging){globe.rotation.y+=0.002;wire.rotation.y+=0.002;}
        ring.rotation.x=Math.sin(Date.now()*0.0005)*0.1+0.3;
        ring.rotation.z+=0.001;
        renderer.render(scene,camera);
    }
    animate();
    })();
    </script></div>
    """
    components.html(globe_html, height=height+10)


# ═══════════════════════════════════════════════════════════════════════════════
# RISK ARC GAUGE COMPONENT
# ═══════════════════════════════════════════════════════════════════════════════
def render_risk_gauge(score, label="RISK INDEX", size=200):
    pct = min(max(score, 0), 1)
    if pct >= 0.8: color = '#ff7b72'; glow = 'rgba(255,123,114,0.5)'
    elif pct >= 0.5: color = '#f0a84c'; glow = 'rgba(240,168,76,0.4)'
    else: color = '#39d0c4'; glow = 'rgba(57,208,196,0.4)'

    sweep = 220
    circum = 2 * 3.14159 * 80
    dash = circum * (sweep / 360)
    fill = dash * pct

    components.html(f"""
    <div style="text-align:center;padding:20px;">
    <svg width="{size}" height="{size}" viewBox="0 0 200 200">
        <defs>
            <linearGradient id="rg" x1="0%" y1="0%" x2="100%">
                <stop offset="0%" style="stop-color:#39d0c4"/>
                <stop offset="50%" style="stop-color:#f0a84c"/>
                <stop offset="100%" style="stop-color:#ff7b72"/>
            </linearGradient>
        </defs>
        <circle cx="100" cy="100" r="80" fill="none" stroke="#111c2e" stroke-width="8"
                stroke-dasharray="{dash} {circum}" stroke-dashoffset="0"
                transform="rotate(-200 100 100)" stroke-linecap="round"/>
        <circle cx="100" cy="100" r="80" fill="none" stroke="url(#rg)" stroke-width="8"
                stroke-dasharray="{fill} {circum}" stroke-dashoffset="0"
                transform="rotate(-200 100 100)" stroke-linecap="round"
                style="filter:drop-shadow(0 0 8px {glow});transition:stroke-dasharray 1.2s ease-out;"/>
        <circle cx="100" cy="100" r="90" fill="none" stroke="rgba(88,166,255,0.12)" stroke-width="1"
                stroke-dasharray="4 6" transform="rotate(0 100 100)">
            <animateTransform attributeName="transform" type="rotate" from="0 100 100" to="360 100 100" dur="20s" repeatCount="indefinite"/>
        </circle>
        <text x="100" y="95" text-anchor="middle" font-family="Orbitron,monospace" font-size="32" font-weight="700" fill="{color}">
            {pct:.2f}
        </text>
        <text x="100" y="120" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="9" fill="#7d8590" letter-spacing="2">
            {label}
        </text>
    </svg></div>
    """, height=size+40)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    from datetime import datetime
    st.sidebar.markdown("""
    <div class="sidebar-brand">AIR-E</div>
    <div class="sidebar-fullform">Aadhaar Integrity<br>& Risk Engine</div>
    <div class="sidebar-divider"></div>
    """, unsafe_allow_html=True)

    pages = [
        "⬡ Overview Dashboard",
        "◎ Geographic Mapping",
        "⊞ Audit Optimizer",
        "⧖ Policy Simulator",
        "⬡ Model Insights"
    ]
    selection = st.sidebar.radio("NAVIGATION", pages, label_visibility="collapsed")

    # Upgraded telemetry panel live status component
    st.sidebar.markdown(f"""
    <div class="telemetry-panel">
      <div style="font-size:9px;letter-spacing:0.15em;color:#7d8590;
           margin-bottom:12px">ENGINE TELEMETRY</div>
      
      <div style="display:flex;align-items:center;margin-bottom:8px">
        <span class="telemetry-dot"></span>
        <span style="color:#e6edf3">XGBoost Forecaster</span>
        <span style="margin-left:auto;color:#39d0c4;font-size:9px">ACTIVE</span>
      </div>
      
      <div style="display:flex;align-items:center;margin-bottom:8px">
        <span class="telemetry-dot"></span>
        <span style="color:#e6edf3">Isolation Forest</span>
        <span style="margin-left:auto;color:#39d0c4;font-size:9px">ACTIVE</span>
      </div>
      
      <div style="display:flex;align-items:center;margin-bottom:8px">
        <span class="telemetry-dot" style="background:#f0a84c;
              box-shadow:0 0 8px #f0a84c"></span>
        <span style="color:#e6edf3">Z-Score Engine</span>
        <span style="margin-left:auto;color:#f0a84c;font-size:9px">STANDBY</span>
      </div>
      
      <div style="border-top:1px solid rgba(88,166,255,0.1);
           padding-top:8px;margin-top:8px;font-size:9px;color:#3d444d;line-height:1.6;margin-bottom:12px;">
        DPDP COMPLIANCE: <span style="color:#39d0c4">✓ NOMINAL</span><br>
        PII EXPOSURE: <span style="color:#39d0c4">✓ ZERO</span><br>
        LAST SCAN: <span style="color:#58a6ff">{datetime.now().strftime('%H:%M:%S UTC')}</span>
      </div>
      <div style="border-top:1px solid rgba(88,166,255,0.1);
           padding-top:12px;margin-top:8px;font-family:'IBM Plex Mono',monospace;font-size:10px;color:#8b949e;text-align:center;line-height:1.5;">
        <span style="letter-spacing:0.1em;font-size:9px;color:#8b949e;font-weight:600;">DEVELOPED BY</span><br>
        <span style="color:#58a6ff;font-weight:700;font-size:13px;text-shadow:0 0 8px rgba(88,166,255,0.4);letter-spacing:0.05em;display:inline-block;margin-top:2px;">Arnav Raj</span><br>
        <div style="margin-top:4px;display:flex;justify-content:center;gap:12px;font-size:11px;">
          <a href="https://github.com/yocybroarnv" target="_blank" style="color:#bc8cff;text-decoration:none;font-weight:600;">GitHub</a>
          <span style="color:#444d56;">|</span>
          <a href="https://www.linkedin.com/in/arnav-raj-professional" target="_blank" style="color:#39d0c4;text-decoration:none;font-weight:600;">LinkedIn</a>
        </div>
        <div style="margin-top:14px;font-size:9px;color:#58a6ff;letter-spacing:0.08em;text-transform:uppercase;font-weight:600;">
          UIDAI Risk Intelligence Division
        </div>
        <div style="margin-top:8px;font-size:9.5px;color:#8b949e;line-height:1.4;font-style:italic;font-family:'Outfit',sans-serif;max-width:200px;margin-left:auto;margin-right:auto;">
          Disclaimer: Simulated hackathon concept project. Not affiliated with nor based on actual UIDAI production systems or data.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    return selection


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def page_dashboard(df):
    # Risk Alert Toast (Critical Threshold Breach) check
    top_states_risk = df.groupby('state')['forecasted_risk_score'].mean().sort_values(ascending=False)
    if len(top_states_risk) > 0:
        top_state = top_states_risk.index[0]
        max_score = top_states_risk.values[0]
        if max_score > 0.07:
            st.markdown(f"""
            <div class="alert-toast">
              <span style="color:#ff7b72">⚠</span>
              CRITICAL: <b>{top_state}</b> exceeds risk threshold ({max_score:.4f})
              — Immediate audit recommended
            </div>
            """, unsafe_allow_html=True)

    # HERO SECTION UPGRADE
    st.markdown("""
    <div class="hero-band">
      <div class="hero-logo-text">AIR-E</div>
      <div class="hero-fullform">Aadhaar Integrity & Risk Engine</div>
      <div class="hero-subtitle">NATIONAL RISK OVERVIEW — REAL-TIME ANOMALY INTELLIGENCE</div>
      <div class="hero-live-badge">⬤ LIVE</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row calculations
    total_enroll = df['enrollments'].sum()
    total_anomalies = int(df['is_anomaly'].sum())
    avg_risk = df['forecasted_risk_score'].mean()
    critical_count = int((df['risk_level'] == 'Critical').sum())

    # KPI Row Layout using upgraded CSS classes
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card card-enrollments">
            <div class="kpi-label">Total Enrollments</div>
            <div class="kpi-value">{total_enroll:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card card-anomalies">
            <div class="kpi-label">Detected Anomalies</div>
            <div class="kpi-value">{total_anomalies:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Avg Risk Card context-sensitive coloring
    risk_class = "card-risk-teal"
    if avg_risk > 0.07:
        risk_class = "card-risk-crimson"
    elif avg_risk > 0.04:
        risk_class = "card-risk-amber"
    with c3:
        st.markdown(f"""
        <div class="kpi-card {risk_class}">
            <div class="kpi-label">Avg Risk Score</div>
            <div class="kpi-value">{avg_risk:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Critical pulse animation if critical spikes > 0
    critical_class = "card-critical-active" if critical_count > 0 else "kpi-card"
    with c4:
        st.markdown(f"""
        <div class="kpi-card {critical_class}">
            <div class="kpi-label">Critical Spikes</div>
            <div class="kpi-value">{critical_count}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Heatmap + Leaderboard
    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.markdown('<p class="section-subtitle">State x Date Risk Heatmap</p>', unsafe_allow_html=True)
        pivot = df.pivot_table(index='state', columns=df['date'].dt.strftime('%b %d'), values='forecasted_risk_score', aggfunc='mean')
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale=[
                [0.0,  "#0d1420"],   # void (no risk)
                [0.3,  "#1a3a4a"],   # low
                [0.5,  "#f0a84c"],   # medium
                [0.75, "#ff7b72"],   # high
                [1.0,  "#ff3b30"]    # critical
            ],
            zmin=0, zmax=0.1,
            xgap=1, ygap=1,          # cell separation lines
            hovertemplate='%{y}<br>%{x}<br>Risk: %{z:.4f}<extra></extra>'
        ))
        fig.update_layout(**ORBITAL_TEMPLATE['layout'].to_plotly_json())
        fig.update_layout(
            height=420, xaxis_title='', yaxis_title='', title='',
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Mono", color="#7d8590", size=10),
            coloraxis_colorbar=dict(
                title="Risk",
                tickfont=dict(family="Orbitron", size=9, color="#58a6ff"),
                bgcolor="rgba(11,15,23,0.8)",
                bordercolor="#58a6ff",
                borderwidth=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-subtitle">Top Flagged States</p>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        top_states = df.groupby('state')['forecasted_risk_score'].mean().sort_values(ascending=False).head(8)
        for i, (state, score) in enumerate(top_states.items()):
            color = "#ff7b72" if score > 0.065 else "#f0a84c" if score > 0.055 else "#39d0c4"
            h = color.lstrip('#')
            color_rgb = f"{int(h[0:2],16)}, {int(h[2:4],16)}, {int(h[4:6],16)}"
            rank_label = ["CRITICAL","HIGH","ELEVATED"][min(i,2)]
            st.markdown(f"""
            <div class="flagged-state-card">
              <span class="rank">#{i+1}</span>
              <span class="state-name">{state}</span>
              <span class="risk-badge" style="background:rgba({color_rgb},0.15);
                    border:1px solid {color}; color:{color}">{score:.4f}</span>
              <span class="level-label" style="color:{color}">{rank_label}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 3: Trend + Bars
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<p class="section-subtitle">Monthly Enrollment Trend</p>', unsafe_allow_html=True)
        trend = df.groupby('date').agg({'enrollments':'sum','is_anomaly':'sum'}).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend['date'], y=trend['enrollments'], mode='lines', name='Enrollments',
                                 line=dict(color='#58a6ff', width=2)))
        anomaly_points = trend[trend['is_anomaly'] > 0]
        fig.add_trace(go.Scatter(x=anomaly_points['date'], y=anomaly_points['enrollments'], mode='markers',
                                 name='Anomaly', marker=dict(color='#ff7b72', size=8, symbol='diamond')))
        fig.update_layout(**ORBITAL_TEMPLATE['layout'].to_plotly_json())
        fig.update_layout(
            height=350, showlegend=True, legend=dict(font=dict(size=10)),
            title=dict(
                text="Monthly Enrollment Trend — National Overview",
                font=dict(family="Space Mono, monospace", size=13, color="#58a6ff")
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-subtitle">Document Risk Distribution</p>', unsafe_allow_html=True)
        risk_dist = df.groupby('risk_level', observed=False).size().reset_index(name='count')
        color_map = {'Critical':'#ff7b72','High':'#f0a84c','Medium':'#bc8cff','Low':'#39d0c4'}
        fig = px.bar(risk_dist, x='count', y='risk_level', orientation='h',
                     color='risk_level', color_discrete_map=color_map)
        fig.update_layout(**ORBITAL_TEMPLATE['layout'].to_plotly_json())
        fig.update_layout(
            height=350, showlegend=False, yaxis_title='', xaxis_title='Records',
            title=dict(
                text="Document Risk Distribution — National Overview",
                font=dict(family="Space Mono, monospace", size=13, color="#58a6ff")
            )
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — GEOGRAPHIC MAPPING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def get_india_geojson():
    import json
    import urllib.request
    url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/india-states.geojson"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.loads(response.read().decode())
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — GEOGRAPHIC MAPPING
# ═══════════════════════════════════════════════════════════════════════════════
def page_geographic(df):
    st.markdown('<h1 class="gradient-heading">GEOGRAPHIC MAPPING</h1>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Spatial risk density & 3D orbital reconnaissance</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_map, col_globe = st.columns([3, 2])

    with col_map:
        st.markdown('<p class="section-subtitle">State Risk Density</p>', unsafe_allow_html=True)
        state_stats = df.groupby('state').agg({
            'enrollments': 'sum', 'forecasted_risk_score': 'mean',
            'is_anomaly': 'sum', 'failure_rate': 'mean'
        }).reset_index()

        fig = px.scatter_3d(state_stats, x='is_anomaly', y='enrollments', z='forecasted_risk_score',
                            color='forecasted_risk_score', size='failure_rate',
                            text='state',
                            color_continuous_scale=[
                                [0,   "#bc8cff"],  # low risk — violet
                                [0.5, "#58a6ff"],  # medium — blue
                                [0.8, "#f0a84c"],  # high — amber
                                [1.0, "#ff7b72"]   # critical — crimson
                            ],
                            title='3D State Risk Constellation')
        fig.update_traces(
            marker=dict(
                size=10,
                line=dict(width=1.5, color="rgba(255,255,255,0.3)"),
                opacity=0.9,
                symbol="circle",
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Risk Score: %{z:.4f}<br>"
                "Enrollments: %{y:,.0f}<br>"
                "Anomalies: %{x}<br>"
                "<extra></extra>"
            )
        )
        fig.update_layout(**ORBITAL_TEMPLATE['layout'].to_plotly_json())
        fig.update_layout(
            height=500,
            title=dict(
                text="3D State Risk Constellation",
                font=dict(family="Space Mono, monospace", size=13, color="#58a6ff")
            ),
            scene=dict(
                xaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(88,166,255,0.15)",
                    title=dict(text='Anomalies', font=dict(family="IBM Plex Mono", color="#58a6ff", size=10)),
                    tickfont=dict(family="IBM Plex Mono", color="#7d8590", size=8)
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(188,140,255,0.15)",
                    title=dict(text='Enrollments', font=dict(family="IBM Plex Mono", color="#bc8cff", size=10)),
                    tickfont=dict(family="IBM Plex Mono", color="#7d8590", size=8)
                ),
                zaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,123,114,0.15)",
                    title=dict(text='Risk Score', font=dict(family="IBM Plex Mono", color="#ff7b72", size=10)),
                    tickfont=dict(family="IBM Plex Mono", color="#7d8590", size=8)
                ),
                bgcolor="rgba(4,6,10,0.0)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
            ),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_globe:
        st.markdown('<p class="section-subtitle">Orbital Reconnaissance</p>', unsafe_allow_html=True)
        render_globe(440)

    st.markdown("<br>", unsafe_allow_html=True)

    # India Choropleth Sub-Section
    st.markdown('<p class="section-subtitle">State-Level Risk Density Heatmap (India Choropleth)</p>', unsafe_allow_html=True)
    geojson = get_india_geojson()
    if geojson is not None:
        state_risk = df.groupby('state')['forecasted_risk_score'].mean().reset_index()
        fig_map = px.choropleth(
            state_risk,
            geojson=geojson,
            locations="state",
            featureidkey="properties.name",
            color="forecasted_risk_score",
            color_continuous_scale=[
                [0,   "#0d1420"],
                [0.4, "#1e3a5f"],
                [0.6, "#f0a84c"],
                [0.8, "#ff7b72"],
                [1.0, "#ff3b30"]
            ],
            range_color=[0.045, 0.07],
            hover_name="state",
            hover_data={"forecasted_risk_score": ":.4f"}
        )
        fig_map.update_geos(
            fitbounds="locations",
            visible=False,
            bgcolor="rgba(0,0,0,0)"
        )
        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            geo=dict(bgcolor="rgba(0,0,0,0)"),
            height=500,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Choropleth overlay loaded in offline fallback mode. Connect to the internet to load the dynamic geographical layer.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Frontier Comparator
    st.markdown('<p class="section-subtitle">Frontier Comparator: Border vs Non-Border States</p>', unsafe_allow_html=True)
    categories = ["Avg Risk Score","Anomaly Rate","Update Velocity","Doc Risk","Surge Index"]
    border_vals   = [0.0623, 0.078, 0.071, 0.065, 0.081]
    interior_vals = [0.0514, 0.041, 0.038, 0.044, 0.029]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="⬥ BORDER STATES",
        x=categories, y=border_vals,
        marker=dict(
            color="rgba(255,123,114,0.7)",
            line=dict(color="#ff7b72", width=1.5),
            pattern_shape="/",          # hatching = visual texture
            pattern_fgcolor="#ff7b72"
        ),
        hovertemplate="%{x}<br>Border: %{y:.4f}<extra></extra>"
    ))
    fig_comp.add_trace(go.Bar(
        name="◆ INTERIOR STATES",
        x=categories, y=interior_vals,
        marker=dict(
            color="rgba(88,166,255,0.7)",
            line=dict(color="#58a6ff", width=1.5)
        ),
        hovertemplate="%{x}<br>Interior: %{y:.4f}<extra></extra>"
    ))

    # Add risk threshold line
    fig_comp.add_hline(
        y=0.07,
        line=dict(color="#ff7b72", dash="dot", width=1.5),
        annotation_text="ALERT THRESHOLD",
        annotation_font=dict(family="IBM Plex Mono", color="#ff7b72", size=10)
    )

    fig_comp.update_layout(
        title=dict(
            text="FRONTIER COMPARATOR: BORDER vs INTERIOR STATES — MULTI-VECTOR",
            font=dict(family="Space Mono, monospace", size=13, color="#58a6ff")
        ),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,13,20,0.6)",
        legend=dict(
            font=dict(family="IBM Plex Mono",color="#7d8590",size=10),
            bgcolor="rgba(11,15,23,0.8)",
            bordercolor="rgba(88,166,255,0.2)",
            borderwidth=1
        ),
        height=350
    )
    st.plotly_chart(fig_comp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — AUDIT OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════
def page_audit(df):
    st.markdown('<h1 class="gradient-heading">AUDIT OPTIMIZER</h1>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Registrar risk matrix & intervention engine</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Risk Priority Scatter
    st.markdown('<p class="section-subtitle">Risk Priority Quadrant Analysis</p>', unsafe_allow_html=True)
    fig = go.Figure()
    color_map = {'Critical':'#ff7b72','High':'#f0a84c','Medium':'#bc8cff','Low':'#39d0c4'}
    for level in ['Critical','High','Medium','Low']:
        subset = df[df['risk_level'] == level]
        if len(subset) > 0:
            fig.add_trace(go.Scatter(
                x=subset['failure_rate'], y=subset['rejection_rate'],
                mode='markers', name=level,
                marker=dict(color=color_map.get(level, '#58a6ff'), size=8, opacity=0.7,
                            line=dict(width=1, color='rgba(255,255,255,0.15)'))
            ))
    # Quadrant lines
    fr_med = df['failure_rate'].median()
    rr_med = df['rejection_rate'].median()
    fig.add_hline(y=rr_med, line_dash="dot", line_color="rgba(88,166,255,0.25)")
    fig.add_vline(x=fr_med, line_dash="dot", line_color="rgba(88,166,255,0.25)")
    
    # Quadrant labels (annotations)
    fig.add_annotation(
        x=fr_med * 1.5, y=rr_med * 1.5, text="⚠ CRITICAL ZONE",
        font=dict(color="rgba(255,123,114,0.25)", size=24, family="Syne, sans-serif"),
        showarrow=False, xanchor="center"
    )
    fig.add_annotation(
        x=fr_med * 0.5, y=rr_med * 0.5, text="✓ SAFE ZONE",
        font=dict(color="rgba(57,208,196,0.2)", size=24, family="Syne, sans-serif"),
        showarrow=False, xanchor="center"
    )

    fig.update_layout(**ORBITAL_TEMPLATE['layout'].to_plotly_json())
    fig.update_layout(
        height=450, xaxis_title='Biometric Failure Rate', yaxis_title='Document Rejection Rate',
        title=dict(
            text="Risk Priority Quadrant Analysis",
            font=dict(family="Space Mono, monospace", size=13, color="#58a6ff")
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Top Flagged Zones
    st.markdown('<p class="section-subtitle">Top Flagged Operator Zones</p>', unsafe_allow_html=True)
    top_risk = df.groupby('state').agg({
        'forecasted_risk_score': 'mean', 'failure_rate': 'mean',
        'rejection_rate': 'mean', 'is_anomaly': 'sum'
    }).sort_values('forecasted_risk_score', ascending=False).head(9).reset_index()

    cols = st.columns(3)
    for i, row in top_risk.iterrows():
        with cols[i % 3]:
            risk_val = row['forecasted_risk_score']
            if risk_val >= 0.07: border_col = '#ff7b72'; badge = 'CRITICAL'
            elif risk_val >= 0.05: border_col = '#f0a84c'; badge = 'HIGH'
            elif risk_val >= 0.03: border_col = '#bc8cff'; badge = 'MEDIUM'
            else: border_col = '#39d0c4'; badge = 'LOW'

            st.markdown(f"""
            <div class="glass-panel" style="margin-bottom:16px;border-top:2px solid {border_col};animation: float-idle 6s ease-in-out infinite;animation-delay:{i*0.5}s;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                    <span style="font-family:'Outfit',sans-serif;font-weight:600;color:#e6edf3;font-size:1rem;">{row['state']}</span>
                    <span class="risk-badge risk-{badge.lower()}">{badge}</span>
                </div>
                <div style="font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:700;color:{border_col};margin-bottom:12px;">{risk_val:.4f}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#7d8590;line-height:1.8;">
                    Failure Rate: {row['failure_rate']:.4f}<br>
                    Rejection Rate: {row['rejection_rate']:.4f}<br>
                    Anomalies: {int(row['is_anomaly'])}
                </div>
                <div style="margin-top: 14px; display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(88,166,255,0.06); padding-top: 12px;">
                    <button class="btn-freeze" onclick="this.innerText='FREEZING...'; this.style.borderColor='rgba(255,123,114,0.6)'; setTimeout(()=>{{this.innerText='✓ CERT FROZEN'; this.style.background='rgba(57,208,196,0.1)'; this.style.color='#39d0c4'; this.style.borderColor='rgba(57,208,196,0.3)'}},1500)">
                      Freeze Cert
                    </button>
                </div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — POLICY SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
def page_simulator(df):
    st.markdown('<h1 class="gradient-heading">POLICY SIMULATOR</h1>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">What-If analysis & sensitivity modeling</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_ctrl, col_viz = st.columns([2, 3])

    with col_ctrl:
        st.markdown('<p class="section-subtitle">Control Parameters</p>', unsafe_allow_html=True)
        fee_rigor = st.slider("Fee Vetting Rigor", 0, 100, 50, key="fee")
        border_strict = st.slider("Border Validation Strictness", 0, 100, 50, key="border")
        demo_audit = st.slider("Demographic Audit Stringency", 0, 100, 50, key="demo")

        # ROI calculations
        composite = (fee_rigor * 0.4 + border_strict * 0.35 + demo_audit * 0.25) / 100
        leakage = composite * 847  # crores
        exclusion = max(0, (composite - 0.5) * 12)  # percentage

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-subtitle">ROI Readouts</p>', unsafe_allow_html=True)

        # Dynamic ROI cards with color shifting based on values
        leakage_color = "#39d0c4" if leakage > 100 else "#f0a84c"
        exclusion_color = "#ff7b72" if exclusion > 5 else "#39d0c4"

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 2px solid {leakage_color} !important; box-shadow: 0 0 24px {leakage_color}25 !important;">
                <div class="kpi-label">Leakage Prevented</div>
                <div class="kpi-value" style="color: {leakage_color} !important;">₹{leakage:.0f} Cr</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 2px solid {exclusion_color} !important; box-shadow: 0 0 24px {exclusion_color}25 !important;">
                <div class="kpi-label">Exclusion Risk</div>
                <div class="kpi-value" style="color: {exclusion_color} !important;">{exclusion:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    with col_viz:
        st.markdown('<p class="section-subtitle">Sensitivity Curves</p>', unsafe_allow_html=True)
        x_range = np.linspace(0, 100, 50)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_range, y=x_range * 0.4 * 8.47, name='Fee Rigor Impact',
                                 line=dict(color='#58a6ff', width=2)))
        fig.add_trace(go.Scatter(x=x_range, y=x_range * 0.35 * 8.47, name='Border Strict Impact',
                                 line=dict(color='#bc8cff', width=2)))
        fig.add_trace(go.Scatter(x=x_range, y=x_range * 0.25 * 8.47, name='Demo Audit Impact',
                                 line=dict(color='#39d0c4', width=2)))

        # DPDP Threshold
        fig.add_vline(x=75, line_dash="dash", line_color="#ff7b72", annotation_text="DPDP Legal Threshold",
                      annotation_font_color="#ff7b72")

        # Optimal zone
        fig.add_vrect(x0=40, x1=65, fillcolor="rgba(57,208,196,0.08)", line_width=0,
                      annotation_text="Optimal Zone", annotation_position="top left",
                      annotation_font_color="#39d0c4")

        fig.update_layout(**ORBITAL_TEMPLATE['layout'].to_plotly_json())
        
        district_label = f"Policy Sensitivity Curve — Leakage vs Stringency (Fee:{fee_rigor}% Border:{border_strict}% Audit:{demo_audit}%)"
        fig.update_layout(
            height=400, xaxis_title='Stringency (%)', yaxis_title='Leakage Prevented (Cr)',
            legend=dict(font=dict(size=10)),
            title=dict(
                text=district_label,
                font=dict(family="Space Mono, monospace", size=11, color="#58a6ff")
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    # What-If Summary Panel with Compliance Warning
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="glass-panel">
      <p class="section-subtitle">What-If Summary</p>
      <hr>
      <div class="whatiif-summary">
        At current thresholds — Fee: <b>{fee_rigor}%</b> | Border: <b>{border_strict}%</b> | 
        Audit: <b>{demo_audit}%</b> — AIR-E projects annual leakage reduction of 
        <span style="color:#39d0c4; font-weight:700;">₹{leakage:.0f} Cr</span> with 
        <span style="color:{'#ff7b72' if exclusion>5 else '#39d0c4'}; font-weight:700;">{exclusion:.1f}% 
        citizen exclusion risk</span>. 
        DPDP compliance: <b>{'✓ WITHIN BOUNDS' if exclusion<8 else '⚠ REVIEW REQUIRED'}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL INSIGHTS (XAI)
# ═══════════════════════════════════════════════════════════════════════════════
def page_model_insights(df):
    st.markdown('<h1 class="gradient-heading">MODEL INSIGHTS</h1>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Explainable AI attribution & confidence analysis</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # State selector
    selected_state = st.selectbox("Select State for Analysis", sorted(df['state'].unique()),
                                  key="xai_state")
    state_df = df[df['state'] == selected_state]
    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: SHAP + Correlation + Gauges
    c1, c2, c3 = st.columns([2, 2, 1])

    with c1:
        st.markdown('<p class="section-subtitle">SHAP Feature Attribution</p>', unsafe_allow_html=True)
        # Simulated SHAP waterfall
        features = ['failure_rate', 'rejection_rate', 'anomaly_flag', 'enrollment_volume', 'baseline']
        impacts = [
            state_df['failure_rate'].mean() * 2.5,
            state_df['rejection_rate'].mean() * 2.0,
            state_df['is_anomaly'].mean() * 1.5,
            -(state_df['enrollments'].mean() / state_df['enrollments'].max()) * 0.5,
            0.1
        ]
        colors = ['#58a6ff' if v > 0 else '#ff7b72' for v in impacts]

        fig = go.Figure(go.Bar(
            x=impacts, y=features, orientation='h',
            marker_color=colors,
            text=[f"{v:+.3f}" for v in impacts],
            textposition='outside', textfont=dict(family='IBM Plex Mono', size=10, color='#e6edf3')
        ))
        fig.update_layout(**ORBITAL_TEMPLATE['layout'].to_plotly_json())
        fig.update_layout(
            height=350, xaxis_title='Impact on Risk Score', yaxis_title='',
            title=dict(
                text=f"SHAP Feature Attribution — {selected_state}",
                font=dict(family="Space Mono, monospace", size=11, color="#58a6ff")
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<p class="section-subtitle">Feature Correlation Matrix</p>', unsafe_allow_html=True)
        corr_features = ['enrollments', 'failure_rate', 'rejection_rate', 'forecasted_risk_score']
        corr_labels = ['Enrollments', 'Failure Rate', 'Rejection Rate', 'Risk Score']
        corr_matrix = state_df[corr_features].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values, x=corr_labels, y=corr_labels,
            colorscale=[[0, '#0d1420'], [0.5, '#58a6ff'], [1, '#bc8cff']],
            text=np.round(corr_matrix.values, 2), texttemplate='%{text}',
            textfont=dict(family='IBM Plex Mono', size=10, color='#e6edf3'),
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        fig.update_layout(**ORBITAL_TEMPLATE['layout'].to_plotly_json())
        fig.update_layout(
            height=350,
            title=dict(
                text=f"Feature Correlation Matrix — {selected_state}",
                font=dict(family="Space Mono, monospace", size=11, color="#58a6ff")
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        st.markdown('<p class="section-subtitle">Confidence</p>', unsafe_allow_html=True)
        avg_risk = state_df['forecasted_risk_score'].mean()
        render_risk_gauge(avg_risk, "RISK INDEX", 180)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Sparklines
    st.markdown('<p class="section-subtitle">Feature Time Series (Sparklines)</p>', unsafe_allow_html=True)
    spark_cols = st.columns(4)
    spark_features = [('enrollments', 'Enrollment Volume', '#58a6ff'),
                      ('failure_rate', 'Failure Rate', '#ff7b72'),
                      ('rejection_rate', 'Rejection Rate', '#f0a84c'),
                      ('forecasted_risk_score', 'Risk Score', '#bc8cff')]

    def hex_to_fill(hex_color, alpha=0.08):
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f'rgba({r},{g},{b},{alpha})'

    for i, (feat, label, color) in enumerate(spark_features):
        with spark_cols[i]:
            fig = go.Figure()
            sorted_df = state_df.sort_values('date')
            fig.add_trace(go.Scatter(x=sorted_df['date'], y=sorted_df[feat], mode='lines',
                                     line=dict(color=color, width=1.5), fill='tozeroy',
                                     fillcolor=hex_to_fill(color)))
            # Highlight max point
            max_idx = sorted_df[feat].idxmax()
            if pd.notna(max_idx):
                max_row = sorted_df.loc[max_idx]
                fig.add_trace(go.Scatter(x=[max_row['date']], y=[max_row[feat]], mode='markers',
                                         marker=dict(color='#ff7b72', size=6), showlegend=False))
            fig.update_layout(**ORBITAL_TEMPLATE['layout'].to_plotly_json())
            fig.update_layout(height=160, showlegend=False, margin=dict(l=10, r=10, t=30, b=10),
                              title=dict(text=label, font=dict(family='IBM Plex Mono', size=10, color='#7d8590')),
                              xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    inject_css()
    render_particle_bg()
    
    # Inject auto-updating UTC clock in topbar
    st.markdown("""
    <div style="position:fixed;top:14px;right:40px;z-index:9998;
      font-family:'Orbitron', sans-serif;font-size:11px;color:#58a6ff;
      text-shadow:0 0 10px rgba(88,166,255,0.5);letter-spacing:0.1em;
      font-weight:600; background:rgba(13,20,33,0.6); padding:4px 12px;
      border:1px solid rgba(88,166,255,0.15); border-radius:4px;
      backdrop-filter:blur(8px);"
      id="air-e-clock"></div>
    <script>
      function updateClock(){
        const d = new Date();
        const utcStr = d.getUTCHours().toString().padStart(2, '0') + ':' + 
                       d.getUTCMinutes().toString().padStart(2, '0') + ':' + 
                       d.getUTCSeconds().toString().padStart(2, '0') + ' UTC';
        document.getElementById('air-e-clock').innerText = utcStr;
      }
      setInterval(updateClock,1000);updateClock();
    </script>
    """, unsafe_allow_html=True)

    page = render_sidebar()

    df = load_data()
    if df.empty:
        st.markdown("""
        <div class="glass-panel" style="text-align:center;margin-top:60px;">
            <h2 style="font-family:'Syne',sans-serif;color:#ff7b72;">Data Not Found</h2>
            <p style="font-family:'IBM Plex Mono',monospace;color:#7d8590;">
                Run <code>python data_engine.py</code> then <code>python ml_engine.py</code> to generate data.
            </p>
        </div>""", unsafe_allow_html=True)
        return

    if page == "⬡ Overview Dashboard":
        page_dashboard(df)
    elif page == "◎ Geographic Mapping":
        page_geographic(df)
    elif page == "⊞ Audit Optimizer":
        page_audit(df)
    elif page == "⧖ Policy Simulator":
        page_simulator(df)
    elif page == "⬡ Model Insights":
        page_model_insights(df)

    # Global developer credit footer
    st.markdown("""
    <div style="margin-top:60px; padding:24px 0; border-top:1px solid rgba(88,166,255,0.08); text-align:center; font-family:'IBM Plex Mono',monospace; font-size:12px; color:#8b949e; line-height:1.7;">
      <span style="font-weight:600; color:#58a6ff; font-size:13px; letter-spacing:0.05em;">Aadhaar Integrity & Risk Engine (AIR-E) — Hackathon Concept Project</span><br>
      <span style="font-size:12px; color:#e6edf3; display:inline-block; margin-top:4px;">Developed by <span style="color:#58a6ff; font-weight:700; text-shadow:0 0 8px rgba(88,166,255,0.2);">Arnav Raj</span> 
      (<a href="https://github.com/yocybroarnv" target="_blank" style="color:#bc8cff; text-decoration:none; font-weight:600;">GitHub</a> | 
      <a href="https://www.linkedin.com/in/arnav-raj-professional" target="_blank" style="color:#39d0c4; text-decoration:none; font-weight:600;">LinkedIn</a>)</span><br>
      <span style="font-size:10px; color:#8b949e; display:block; margin-top:10px; font-family:'Outfit',sans-serif; max-width:700px; margin-left:auto; margin-right:auto; text-align:center;">
        LEGAL DISCLAIMER: This application is a simulation designed solely for hackathon presentation and academic purposes. All datasets, scores, and operational risk metrics are synthetically generated. It is completely independent of, and does not represent actual citizens, security records, or production systems of the Unique Identification Authority of India (UIDAI).
      </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
