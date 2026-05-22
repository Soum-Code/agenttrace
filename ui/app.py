"""
AgentTrace -- Streamlit Premium HTML Dashboard
================================================
Renders the premium glassmorphic dashboard interface for AgentTrace.
Connects dynamically to the FastAPI backend to run trace analysis and interventions.
"""

from __future__ import annotations

import os
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Allow port override via environment variable, defaulting to standard local port
API_BASE = os.environ.get("AGENTTRACE_API_BASE", "http://127.0.0.1:8000")

# ---------------------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgentTrace | Hallucination Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject custom styling to make the iframe fill the Streamlit viewport seamlessly
st.markdown(
    """
    <style>
    /* Hide Streamlit elements completely to give a native web app feel and prevent click blockage */
    #MainMenu {display: none !important;}
    footer {display: none !important;}
    header {display: none !important;}
    
    /* Disable scrolling on parent window to prevent double scrollbars */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden !important;
        height: 100vh !important;
    }
    
    /* Remove padding around the main block container */
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    
    /* Ensure the embedded iframe has no border, matches background, and fits the screen */
    iframe {
        border: none;
        background: transparent;
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw !important;
        height: 100vh !important;
        z-index: 999999;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Dashboard HTML Template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  :root {
    --acc:#4F8EF7;--acc2:#6BA8FF;--acc-glow:rgba(79,142,247,0.28);--acc-soft:rgba(79,142,247,0.12);
    --bg-base:#0a0e1a;--bg2:#0d1525;
    --glass:rgba(255,255,255,0.055);--glass-border:rgba(255,255,255,0.11);--glass-hover:rgba(255,255,255,0.085);
    --specular:rgba(255,255,255,0.16);
    --txt:#f0f4ff;--txt2:rgba(210,225,255,0.62);--txt3:rgba(170,190,235,0.38);
    --success:#34D399;--warn:#FBBF24;--danger:#F87171;--purple:#A78BFA;--teal:#2DD4BF;
    --mesh1:rgba(79,142,247,0.14);--mesh2:rgba(167,139,250,0.09);--mesh3:rgba(45,212,191,0.07);
    --shadow:0 8px 32px rgba(0,0,12,0.45);--shadow-sm:0 4px 16px rgba(0,0,12,0.3);
    --tr:all 0.32s cubic-bezier(0.4,0,0.2,1);
  }
  [data-theme="light"] {
    --bg-base:#dde4f5;--bg2:#cfd8f0;
    --glass:rgba(255,255,255,0.6);--glass-border:rgba(255,255,255,0.85);--glass-hover:rgba(255,255,255,0.75);
    --specular:rgba(255,255,255,0.95);
    --txt:#0c1730;--txt2:rgba(12,23,48,0.62);--txt3:rgba(12,23,48,0.38);
    --mesh1:rgba(79,142,247,0.1);--mesh2:rgba(167,139,250,0.07);--mesh3:rgba(45,212,191,0.06);
    --shadow:0 8px 32px rgba(80,110,200,0.16);--shadow-sm:0 4px 16px rgba(80,110,200,0.1);
  }
  [data-accent="emerald"]{--acc:#10B981;--acc2:#34D399;--acc-glow:rgba(16,185,129,0.28);--acc-soft:rgba(16,185,129,0.12);}
  [data-accent="coral"]  {--acc:#F97316;--acc2:#FB923C;--acc-glow:rgba(249,115,22,0.28);--acc-soft:rgba(249,115,22,0.12);}
  [data-accent="violet"] {--acc:#8B5CF6;--acc2:#A78BFA;--acc-glow:rgba(139,92,246,0.28);--acc-soft:rgba(139,92,246,0.12);}
  [data-accent="amber"]  {--acc:#F59E0B;--acc2:#FBBF24;--acc-glow:rgba(245,158,11,0.28);--acc-soft:rgba(245,158,11,0.12);}

  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:-apple-system,'SF Pro Display','Inter',sans-serif;background:var(--bg-base);color:var(--txt);min-height:100vh;transition:var(--tr);overflow-x:hidden;}

  .mesh{position:fixed;inset:0;pointer-events:none;z-index:0;
    background:radial-gradient(ellipse 65% 55% at 12% 18%,var(--mesh1),transparent),
               radial-gradient(ellipse 55% 65% at 88% 78%,var(--mesh2),transparent),
               radial-gradient(ellipse 45% 45% at 52% 52%,var(--mesh3),transparent);
    transition:var(--tr);}

  .wrap{position:relative;z-index:1;padding:16px;max-width:1080px;margin:0 auto;}

  /* TOPBAR */
  .topbar{display:flex;align-items:center;justify-content:space-between;
    background:var(--glass);border:1px solid var(--glass-border);
    backdrop-filter:blur(28px);-webkit-backdrop-filter:blur(28px);
    border-radius:20px;padding:11px 18px;margin-bottom:16px;
    box-shadow:var(--shadow-sm),inset 0 1px 0 var(--specular);}
  .logo{display:flex;align-items:center;gap:10px;}
  .logo-icon{width:34px;height:34px;border-radius:10px;
    background:linear-gradient(135deg,var(--acc),var(--acc2));
    display:flex;align-items:center;justify-content:center;font-size:16px;
    box-shadow:0 0 14px var(--acc-glow);}
  .logo-title{font-size:15px;font-weight:700;letter-spacing:-.4px;}
  .logo-sub{font-size:10px;color:var(--txt3);margin-top:1px;}
  .topbar-r{display:flex;align-items:center;gap:9px;}

  .live-badge{display:flex;align-items:center;gap:5px;
    background:rgba(52,211,153,0.11);border:1px solid rgba(52,211,153,0.28);
    border-radius:20px;padding:4px 10px;font-size:11px;font-weight:500;color:var(--success);}
  .pulse-dot{width:6px;height:6px;border-radius:50%;background:var(--success);animation:pulse 1.9s ease-in-out infinite;}
  @keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.45;transform:scale(.75)}}

  .accent-row{display:flex;gap:5px;align-items:center;}
  .acc-dot{width:17px;height:17px;border-radius:50%;cursor:pointer;border:2px solid transparent;transition:transform .2s,border-color .2s;}
  .acc-dot:hover{transform:scale(1.18);}
  .acc-dot.active{border-color:var(--txt);transform:scale(1.15);}

  .icon-btn{width:33px;height:33px;border-radius:10px;border:1px solid var(--glass-border);
    background:var(--glass);cursor:pointer;display:flex;align-items:center;justify-content:center;
    color:var(--txt2);font-size:15px;transition:var(--tr);backdrop-filter:blur(10px);}
  .icon-btn:hover{background:var(--glass-hover);color:var(--txt);transform:scale(1.06);}

  /* TABS STYLE */
  .tab-row {display:flex;gap:8px;margin-bottom:16px;border-bottom:1px solid var(--glass-border);padding-bottom:10px;}
  .tab-btn {
    background:var(--glass);border:1px solid var(--glass-border);color:var(--txt2);
    padding:8px 16px;font-size:12px;font-weight:600;border-radius:10px;cursor:pointer;
    transition:var(--tr);backdrop-filter:blur(10px);
  }
  .tab-btn:hover {background:var(--glass-hover);color:var(--txt);}
  .tab-btn.active {
    background:linear-gradient(135deg,var(--acc),var(--acc2));
    border-color:transparent;color:#fff;box-shadow:0 0 14px var(--acc-glow);
  }
  .tab-content {display:none;}
  .tab-content.active {display:block;}

  /* GRID */
  .g4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:13px;}
  .g3{display:grid;grid-template-columns:1.55fr 1fr 1fr;gap:12px;margin-bottom:13px;}
  .g2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:13px;}
  .g-full{margin-bottom:13px;}

  /* CARD */
  .card{background:var(--glass);border:1px solid var(--glass-border);
    backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);
    border-radius:18px;padding:16px 18px;
    box-shadow:var(--shadow),inset 0 1px 0 var(--specular);
    transition:var(--tr);position:relative;overflow:hidden;}
  .card::before{content:'';position:absolute;inset:0;
    background:radial-gradient(ellipse 80% 35% at 50% -5%,var(--specular),transparent);
    pointer-events:none;}
  .card:hover{border-color:rgba(255,255,255,0.19);transform:translateY(-1px);
    box-shadow:0 14px 44px rgba(0,0,12,0.5),inset 0 1px 0 var(--specular);}
    
  /* Clickable class for cards */
  .card-clickable{cursor:pointer;}
  .card-clickable:active{transform:scale(0.99);}
  
  .active-hl {
    border-color: var(--acc) !important;
    box-shadow: 0 0 12px var(--acc-glow) !important;
    background: var(--glass-hover) !important;
  }

  /* METRIC */
  .mlabel{font-size:10px;font-weight:600;color:var(--txt3);text-transform:uppercase;letter-spacing:.8px;margin-bottom:7px;}
  .mval{font-size:26px;font-weight:700;letter-spacing:-1px;line-height:1;}
  .mval span{font-size:12px;font-weight:400;color:var(--txt2);letter-spacing:0;}
  .delta{display:inline-flex;align-items:center;gap:3px;font-size:10px;font-weight:600;margin-top:6px;padding:2px 7px;border-radius:20px;}
  .d-up{background:rgba(52,211,153,0.12);color:var(--success);}
  .d-warn{background:rgba(251,191,36,0.12);color:var(--warn);}
  .acc-col{color:var(--acc);}
  .acc-glow{text-shadow:0 0 18px var(--acc-glow);}

  /* SECTION HEADER */
  .sh{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;}
  .st{font-size:13px;font-weight:600;letter-spacing:-.2px;}
  .ss{font-size:10px;color:var(--txt3);margin-top:2px;}
  .chip{font-size:10px;font-weight:600;padding:3px 9px;border-radius:20px;
    background:var(--acc-soft);color:var(--acc);border:1px solid var(--acc-glow);}

  /* PIPE ROWS */
  .pipe-row{display:flex;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid var(--glass-border);cursor:pointer;transition:var(--tr);}
  .pipe-row:last-child{border-bottom:none;}
  .pipe-row:hover{background:var(--glass-hover);padding-left:6px;padding-right:6px;border-radius:8px;}
  .pipe-icon{width:32px;height:32px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0;}
  .pipe-name{font-size:12px;font-weight:500;flex:1;}
  .pipe-sub{font-size:9px;color:var(--txt3);margin-top:1px;}
  .pipe-track{width:72px;height:4px;background:rgba(255,255,255,0.1);border-radius:2px;overflow:hidden;}
  .pipe-fill{height:100%;border-radius:2px;transition:width 1.1s cubic-bezier(.4,0,.2,1);}
  .pipe-score{font-size:11px;font-weight:700;min-width:34px;text-align:right;}
  .sdot{width:7px;height:7px;border-radius:50%;}

  /* CMP BARS */
  .cmp{margin-bottom:11px;}
  .cmp:last-child{margin-bottom:0;}
  .cmp-lbl{display:flex;justify-content:space-between;font-size:11px;color:var(--txt2);margin-bottom:4px;}
  .cmp-track{height:7px;background:rgba(255,255,255,0.08);border-radius:4px;overflow:hidden;}
  .cmp-fill{height:100%;border-radius:4px;transition:width 1.3s cubic-bezier(.4,0,.2,1);}

  /* TABLE */
  .tbl{width:100%;font-size:11px;border-collapse:collapse;}
  .tbl th{font-size:9px;font-weight:600;color:var(--txt3);text-transform:uppercase;letter-spacing:.8px;padding:0 7px 9px;text-align:left;}
  .tbl td{padding:8px 7px;border-top:1px solid rgba(255,255,255,0.045);vertical-align:middle;}
  .tbl tr{cursor:pointer;transition:var(--tr);}
  .tbl tr:hover td{background:var(--glass-hover);}
  .tag{display:inline-block;padding:2px 7px;border-radius:20px;font-size:9px;font-weight:700;letter-spacing:.3px;}
  .t-fact{background:rgba(248,113,113,0.14);color:#FCA5A5;border:1px solid rgba(248,113,113,0.22);}
  .t-cont{background:rgba(251,191,36,0.14);color:#FCD34D;border:1px solid rgba(251,191,36,0.22);}
  .t-sem {background:rgba(167,139,250,0.14);color:#C4B5FD;border:1px solid rgba(167,139,250,0.22);}
  .t-tool{background:rgba(45,212,191,0.14);color:#5EEAD4;border:1px solid rgba(45,212,191,0.22);}
  .t-none{background:rgba(52,211,153,0.11);color:#6EE7B7;border:1px solid rgba(52,211,153,0.18);}
  .sev{width:7px;height:7px;border-radius:50%;display:inline-block;}

  /* ARCH FLOW */
  .flow{display:flex;align-items:center;gap:0;flex-wrap:nowrap;margin-top:10px;overflow-x:auto;padding-bottom:4px;}
  .fn{background:var(--glass-hover);border:1px solid var(--glass-border);border-radius:9px;
    padding:7px 10px;text-align:center;font-size:9px;font-weight:500;
    white-space:nowrap;flex-shrink:0;cursor:pointer;transition:var(--tr);}
  .fn:hover,.fn.hl{border-color:var(--acc);color:var(--acc);background:var(--acc-soft);box-shadow:0 0 10px var(--acc-glow);}
  .fn .fi{font-size:14px;margin-bottom:2px;}
  .fa{color:var(--txt3);font-size:12px;padding:0 3px;flex-shrink:0;}

  /* ACTIVITY */
  .act{display:flex;gap:9px;padding:7px 0;border-bottom:1px solid var(--glass-border);font-size:11px;}
  .act:last-child{border-bottom:none;}
  .act-icon{width:22px;height:22px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:10px;flex-shrink:0;}
  .act-time{color:var(--txt3);font-size:9px;margin-top:2px;}

  /* MINI SPARKLINE */
  .spark{width:100%;height:48px;margin-top:8px;}

  /* TOGGLE */
  .tog-wrap{display:flex;background:rgba(255,255,255,0.07);border-radius:9px;padding:2px;}
  .tog{padding:4px 11px;border-radius:7px;font-size:10px;font-weight:500;cursor:pointer;color:var(--txt2);transition:var(--tr);}
  .tog.on{background:var(--glass-hover);color:var(--txt);box-shadow:0 1px 4px rgba(0,0,0,0.25);}

  /* SLIDING DRAWER styling */
  .drawer {
    position: fixed;
    top: 0;
    right: -420px;
    width: 400px;
    height: 100%;
    background: rgba(10, 14, 26, 0.95);
    border-left: 2px solid var(--acc);
    box-shadow: -10px 0 30px rgba(0, 0, 12, 0.5);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    z-index: 10000;
    transition: right 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex;
    flex-direction: column;
  }
  [data-theme="light"] .drawer {
    background: rgba(221, 228, 245, 0.95);
    border-left: 2px solid var(--acc);
    box-shadow: -10px 0 30px rgba(80, 110, 200, 0.16);
  }
  .drawer.open {
    right: 0;
  }
  .drawer-header {
    padding: 16px 20px;
    border-bottom: 1px solid var(--glass-border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .drawer-title {
    font-size: 15px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: var(--txt);
  }
  .drawer-close {
    background: none;
    border: none;
    font-size: 22px;
    color: var(--txt3);
    cursor: pointer;
    transition: var(--tr);
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
  }
  .drawer-close:hover {
    color: var(--txt);
    background: var(--glass-hover);
  }
  .drawer-body {
    padding: 20px;
    overflow-y: auto;
    flex: 1;
    font-size: 12.5px;
    line-height: 1.6;
  }
  .drawer-meta {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 18px;
    font-size: 11px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  [data-theme="light"] .drawer-meta {
    background: rgba(255, 255, 255, 0.35);
  }
  .drawer-meta-row {
    display: flex;
    justify-content: space-between;
  }
  .drawer-meta-label {
    color: var(--txt3);
    font-weight: 500;
  }
  .drawer-meta-value {
    font-family: monospace;
    color: var(--acc2);
    font-weight: 600;
  }
  [data-theme="light"] .drawer-meta-value {
    color: var(--acc);
  }
  .drawer-desc {
    color: var(--txt2);
    margin-bottom: 18px;
  }
  .drawer-extra {
    border-top: 1px solid var(--glass-border);
    padding-top: 14px;
    margin-top: 14px;
  }
  .drawer-extra h5 {
    font-size: 11.5px;
    font-weight: 600;
    margin-bottom: 8px;
    color: var(--txt);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .drawer-extra ul, .drawer-extra ol {
    padding-left: 18px;
    color: var(--txt2);
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .drawer-extra code {
    background: rgba(255, 255, 255, 0.08);
    padding: 2px 5px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 11px;
    color: var(--acc2);
  }
  [data-theme="light"] .drawer-extra code {
    background: rgba(0, 0, 0, 0.05);
    color: var(--acc);
  }

  /* Live analyzer styles */
  #task-input {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 12px;
    color: var(--txt);
    font-family: inherit;
    font-size: 12px;
    resize: vertical;
    min-height: 56px;
    outline: none;
    transition: var(--tr);
    width: 100%;
  }
  #task-input:focus {
    border-color: var(--acc);
    background: rgba(255, 255, 255, 0.06);
    box-shadow: 0 0 8px var(--acc-glow);
  }
  [data-theme="light"] #task-input {
    background: rgba(255, 255, 255, 0.5);
    color: var(--txt);
  }
  [data-theme="light"] #task-input:focus {
    background: rgba(255, 255, 255, 0.85);
  }
  .btn-primary {
    background: linear-gradient(135deg,var(--acc),var(--acc2));
    border: none;
    border-radius: 12px;
    padding: 0 20px;
    color: white;
    font-weight: 600;
    font-size: 12px;
    cursor: pointer;
    box-shadow: 0 0 14px var(--acc-glow);
    transition: var(--tr);
    min-width: 120px;
  }
  .btn-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 20px var(--acc-glow);
  }
  .btn-primary:active {
    transform: translateY(0);
  }
  .btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }
  
  /* Step card styling */
  .step-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-left: 5px solid transparent;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    transition: var(--tr);
  }
  .step-card:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-sm);
  }
  .step-clean {
    border-left-color: var(--success);
    background: linear-gradient(145deg, rgba(5,46,22,0.1), rgba(6,78,59,0.1));
  }
  .step-hallucinated {
    border-left-color: var(--danger);
    background: linear-gradient(145deg, rgba(69,10,10,0.1), rgba(127,29,29,0.1));
  }
  .step-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }
  .step-title {
    font-size: 12.5px;
    font-weight: 700;
    color: var(--txt);
  }
  .step-badge {
    font-size: 9px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 20px;
    letter-spacing: 0.3px;
  }
  
  /* Expected vs Actual Badges */
  .badge-expected-clean {
    background: rgba(52,211,153,0.11);
    color: var(--success);
    border: 1px solid rgba(52,211,153,0.22);
  }
  .badge-expected-hallucinated {
    background: rgba(248,113,113,0.11);
    color: var(--danger);
    border: 1px solid rgba(248,113,113,0.22);
  }
  .badge-actual-clean {
    background: rgba(52,211,153,0.22);
    color: var(--success);
    border: 1px solid var(--success);
  }
  .badge-actual-hallucinated {
    background: rgba(248,113,113,0.22);
    color: var(--danger);
    border: 1px solid var(--danger);
  }
  
  /* Alignment Status tags */
  .align-match-tn {
    background: rgba(52,211,153,0.15);
    color: var(--success);
    border: 1px solid var(--success);
    font-weight: 700;
  }
  .align-match-tp {
    background: rgba(52,211,153,0.25);
    color: var(--success);
    border: 2px solid var(--success);
    font-weight: 700;
  }
  .align-mismatch-fp {
    background: rgba(251,191,36,0.15);
    color: var(--warn);
    border: 1px solid var(--warn);
    font-weight: 700;
  }
  .align-mismatch-fn {
    background: rgba(248,113,113,0.25);
    color: var(--danger);
    border: 2px solid var(--danger);
    font-weight: 700;
    animation: pulse-red 2s infinite;
  }
  @keyframes pulse-red {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.65; }
  }

  .step-detail {
    font-size: 11.5px;
    color: var(--txt2);
    margin-bottom: 5px;
    line-height: 1.5;
  }
  .step-detail strong {
    color: var(--txt);
    font-weight: 600;
  }
  .step-detail code {
    background: rgba(255, 255, 255, 0.06);
    padding: 2px 5px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 10.5px;
    color: var(--acc2);
    word-break: break-all;
  }
  [data-theme="light"] .step-detail code {
    background: rgba(0, 0, 0, 0.04);
    color: var(--acc);
  }
  
  .hallucination-explanation {
    border-radius: 8px;
    padding: 8px 12px;
    margin-top: 10px;
    font-size: 11px;
    line-height: 1.5;
  }
  .explain-match {
    background: rgba(52,211,153,0.06);
    border: 1px solid rgba(52,211,153,0.2);
    color: var(--txt);
  }
  [data-theme="light"] .explain-match {
    background: rgba(52,211,153,0.04);
  }
  .explain-mismatch {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.25);
    color: var(--warn);
  }
  [data-theme="light"] .explain-mismatch {
    background: rgba(251,191,36,0.04);
  }
  
  .explain-icon {
    margin-right: 4px;
    font-weight: bold;
  }
  
  .score-bar-container {
    height: 5px;
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
    overflow: hidden;
    margin-top: 4px;
    margin-bottom: 8px;
  }
  .score-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
  }
  .score-bar-fill-green { background: var(--success); }
  .score-bar-fill-amber { background: var(--warn); }
  .score-bar-fill-red { background: var(--danger); }
  
  .btn-correct {
    background: rgba(255,255,255,0.06);
    border: 1px solid var(--glass-border);
    border-radius: 8px;
    padding: 5px 12px;
    color: var(--txt);
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    transition: var(--tr);
  }
  .btn-correct:hover {
    background: var(--glass-hover);
    border-color: var(--acc2);
    color: var(--acc2);
  }
  
  .correction-card {
    background: linear-gradient(145deg, rgba(30,27,75,0.35), rgba(49,46,129,0.35));
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 8px;
    padding: 12px 14px;
    margin-top: 10px;
  }
  .correction-card h4 {
    color: #a5b4fc;
    font-size: 11.5px;
    margin-bottom: 6px;
    font-weight: 700;
  }
  [data-theme="light"] .correction-card {
    background: linear-gradient(145deg, rgba(230,230,250,0.8), rgba(220,220,245,0.8));
    border: 1px solid rgba(99,102,241,0.15);
  }

  @media(max-width:680px){
    .g4{grid-template-columns:repeat(2,1fr);}
    .g3,.g2{grid-template-columns:1fr;}
    .flow{flex-wrap:wrap;}
    .drawer{width:100%;right:-100%;}
  }
</style>
</head>
<body data-theme="dark" data-accent="blue">
<div class="mesh"></div>
<div class="wrap">

<!-- TOPBAR -->
<div class="topbar">
  <div class="logo">
    <div class="logo-icon">🔬</div>
    <div>
      <div class="logo-title">AgentTrace</div>
      <div class="logo-sub">Hallucination Detection &amp; Attribution · EMNLP 2026</div>
    </div>
  </div>
  <div class="topbar-r">
    <div class="live-badge" id="api-badge"><span class="pulse-dot"></span>Connecting...</div>
    <div class="accent-row">
      <div class="acc-dot active" style="background:#4F8EF7" data-acc="blue"></div>
      <div class="acc-dot" style="background:#10B981" data-acc="emerald"></div>
      <div class="acc-dot" style="background:#F97316" data-acc="coral"></div>
      <div class="acc-dot" style="background:#8B5CF6" data-acc="violet"></div>
      <div class="acc-dot" style="background:#F59E0B" data-acc="amber"></div>
    </div>
    <div class="icon-btn" id="theme-btn" title="Toggle day/night">☀</div>
  </div>
</div>

<!-- DASHBOARD TABS -->
<div class="tab-row">
  <button class="tab-btn active" data-tab="live">Live Analysis</button>
  <button class="tab-btn" data-tab="history">History Logs</button>
</div>

<!-- LIVE ANALYSIS WORKSPACE (TAB 1) -->
<div id="content-live" class="tab-content active">

  <!-- LIVE TRACE ANALYSIS INPUT -->
  <div class="card g-full">
    <div class="sh" style="margin-bottom: 8px;">
      <div>
        <div class="st">Run Live Trace Analysis</div>
        <div class="ss">Submit a task description or select a deterministic test scenario below</div>
      </div>
    </div>
    <div style="display:flex; flex-direction:column; gap:10px; margin-top:6px;">
      <div style="display:flex; gap:12px; align-items:center;">
        <label for="scenario-select" style="font-size:11px; font-weight:600; color:var(--txt2);">Test Scenario:</label>
        <select id="scenario-select" style="background:var(--glass); border:1px solid var(--glass-border); border-radius:8px; padding:6px 12px; color:var(--txt); font-size:11px; outline:none; transition:var(--tr); cursor:pointer;">
          <option value="" style="background:var(--bg2);">-- Custom Query --</option>
          <option value="SCENARIO: clean" style="background:var(--bg2);">Clean Trajectory</option>
          <option value="SCENARIO: reasoning" style="background:var(--bg2);">Reasoning Hallucination</option>
          <option value="SCENARIO: tool" style="background:var(--bg2);">Tool-Use Hallucination</option>
          <option value="SCENARIO: retrieval" style="background:var(--bg2);">Retrieval/Grounding Hallucination</option>
          <option value="SCENARIO: human" style="background:var(--bg2);">Human-Interaction/Contradiction Hallucination</option>
          <option value="SCENARIO: planning" style="background:var(--bg2);">Planning Hallucination</option>
        </select>
      </div>
      <div style="display:flex;gap:12px;align-items:stretch;">
        <textarea id="task-input" placeholder="e.g. Find the population of Tokyo and compare it with Delhi, then calculate the ratio..."></textarea>
        <button id="analyze-btn" class="btn-primary">Analyze Trace</button>
      </div>
    </div>
  </div>

  <!-- ANALYSIS RESULTS SECTION (DYNAMIC) -->
  <div id="results-container" style="display:none; margin-bottom:13px;">
    <div class="card">
      <div class="sh" style="border-bottom: 1px solid var(--glass-border); padding-bottom: 8px; margin-bottom: 12px;">
        <div>
          <div class="st" id="results-title">Analysis Results</div>
          <div class="ss" id="results-meta">analyzed in 0.0s</div>
        </div>
        <button id="download-btn" class="chip" style="cursor:pointer; border:none; outline:none; background:var(--acc-soft); color:var(--acc);">Download JSON</button>
      </div>
      <div id="steps-list">
        <!-- Generated steps render here dynamically -->
      </div>
    </div>
  </div>

  <!-- METRICS ROW -->
  <div class="g4">
    <div class="card card-clickable" data-details="step-localization">
      <div class="mlabel">Step Localization</div>
      <div class="mval acc-col acc-glow" id="m1">0.655 <span>acc</span></div>
      <div class="delta d-up">↑ +59.4% vs SOTA</div>
    </div>
    <div class="card card-clickable" data-details="avg-latency">
      <div class="mlabel">Avg Latency</div>
      <div class="mval">411 <span>ms</span></div>
      <div class="delta d-up">✓ Under SLA</div>
    </div>
    <div class="card card-clickable" data-details="p95-latency">
      <div class="mlabel">P95 Latency</div>
      <div class="mval">574 <span>ms</span></div>
      <div class="delta d-warn">◐ Near threshold</div>
    </div>
    <div class="card card-clickable" data-details="delta-baseline">
      <div class="mlabel">Δ vs Baseline</div>
      <div class="mval" style="color:var(--success)">+0.244</div>
      <div class="delta d-up">↑ Beat AgentHallu</div>
    </div>
  </div>

  <!-- MAIN ROW -->
  <div class="g3">

    <!-- BENCHMARK -->
    <div class="card card-clickable" data-details="benchmark-card">
      <div class="sh">
        <div><div class="st">Benchmark Comparison</div><div class="ss">AgentTrace vs baselines · 200 trajectories</div></div>
        <div class="chip">SOTA ✓</div>
      </div>
      <div class="cmp">
        <div class="cmp-lbl"><span>AgentTrace (3-Layer Cascade)</span><span style="color:var(--acc);font-weight:700">0.655</span></div>
        <div class="cmp-track"><div class="cmp-fill" id="b1" style="width:0%;background:linear-gradient(90deg,var(--acc),var(--acc2))"></div></div>
      </div>
      <div class="cmp">
        <div class="cmp-lbl"><span>AgentHallu SOTA</span><span style="color:var(--txt2)">0.411</span></div>
        <div class="cmp-track"><div class="cmp-fill" id="b2" style="width:0%;background:rgba(180,180,210,0.38)"></div></div>
      </div>
      <div class="cmp">
        <div class="cmp-lbl"><span>Semantic-only</span><span style="color:var(--txt2)">0.291</span></div>
        <div class="cmp-track"><div class="cmp-fill" id="b3" style="width:0%;background:rgba(140,140,180,0.25)"></div></div>
      </div>
      <div class="cmp">
        <div class="cmp-lbl"><span>NLI-only</span><span style="color:var(--txt2)">0.318</span></div>
        <div class="cmp-track"><div class="cmp-fill" id="b4" style="width:0%;background:rgba(140,140,180,0.25)"></div></div>
      </div>
      <div class="cmp">
        <div class="cmp-lbl"><span>Tool-validator-only</span><span style="color:var(--txt2)">0.274</span></div>
        <div class="cmp-track"><div class="cmp-fill" id="b5" style="width:0%;background:rgba(120,120,160,0.2)"></div></div>
      </div>
      <div style="margin-top:12px;padding-top:11px;border-top:1px solid var(--glass-border);">
        <svg class="spark" viewBox="0 0 320 48" preserveAspectRatio="none">
          <defs>
            <linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="var(--acc)" stop-opacity=".35"/>
              <stop offset="100%" stop-color="var(--acc)" stop-opacity="0"/>
            </linearGradient>
          </defs>
          <path id="spark-area" d="" fill="url(#sg)"/>
          <path id="spark-line" d="" fill="none" stroke="var(--acc)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <div style="display:flex;justify-content:space-between;font-size:9px;color:var(--txt3);margin-top:2px;">
          <span>Commit 1</span><span>Commit 5</span><span>Commit 10</span>
        </div>
      </div>
    </div>

    <!-- DETECTION SIGNALS -->
    <div class="card">
      <div class="sh">
        <div><div class="st">Detection Signals</div><div class="ss">Hybrid fusion</div></div>
        <div style="display:flex;align-items:center;gap:5px;">
          <div class="sdot" style="background:var(--success);box-shadow:0 0 6px var(--success)"></div>
          <div style="font-size:9px;color:var(--success);">All healthy</div>
        </div>
      </div>

      <div class="pipe-row" data-details="semantic">
        <div class="pipe-icon" style="background:rgba(79,142,247,0.15);">🔵</div>
        <div style="flex:1">
          <div class="pipe-name">Semantic Checker</div>
          <div class="pipe-sub">all-MiniLM-L6-v2 · cosine sim</div>
        </div>
        <div class="pipe-track"><div class="pipe-fill" id="p1" style="width:0%;background:#4F8EF7"></div></div>
        <div class="pipe-score" style="color:#4F8EF7">0.84</div>
      </div>
      <div class="pipe-row" data-details="tool-valid">
        <div class="pipe-icon" style="background:rgba(45,212,191,0.15);">🟢</div>
        <div style="flex:1">
          <div class="pipe-name">Tool Validator</div>
          <div class="pipe-sub">Claim extraction · similarity</div>
        </div>
        <div class="pipe-track"><div class="pipe-fill" id="p2" style="width:0%;background:var(--teal)"></div></div>
        <div class="pipe-score" style="color:var(--teal)">0.79</div>
      </div>
      <div class="pipe-row" data-details="factual-nli">
        <div class="pipe-icon" style="background:rgba(167,139,250,0.15);">🟣</div>
        <div style="flex:1">
          <div class="pipe-name">Factual Grounding</div>
          <div class="pipe-sub">nli-deberta-v3-small · FAISS</div>
        </div>
        <div class="pipe-track"><div class="pipe-fill" id="p3" style="width:0%;background:var(--purple)"></div></div>
        <div class="pipe-score" style="color:var(--purple)">0.91</div>
      </div>
      <div class="pipe-row" data-details="contradict">
        <div class="pipe-icon" style="background:rgba(251,191,36,0.15);">🟡</div>
        <div style="flex:1">
          <div class="pipe-name">Contradiction</div>
          <div class="pipe-sub">Sliding-window NLI</div>
        </div>
        <div class="pipe-track"><div class="pipe-fill" id="p4" style="width:0%;background:var(--warn)"></div></div>
        <div class="pipe-score" style="color:var(--warn)">0.76</div>
      </div>

      <div style="margin-top:12px;padding-top:11px;border-top:1px solid var(--glass-border);">
        <div style="font-size:10px;color:var(--txt3);margin-bottom:7px;">Fusion threshold · 0.45</div>
        <div style="display:flex;gap:5px;flex-wrap:wrap;">
          <div style="background:rgba(79,142,247,0.1);border:1px solid rgba(79,142,247,0.2);border-radius:6px;padding:3px 8px;font-size:9px;color:#93BBFF;">Precision@3: 0.82</div>
          <div style="background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.2);border-radius:6px;padding:3px 8px;font-size:9px;color:var(--success);">Recall: 0.78</div>
          <div style="background:rgba(248,113,113,0.1);border:1px solid rgba(248,113,113,0.2);border-radius:6px;padding:3px 8px;font-size:9px;color:#FCA5A5;">FPR: 0.11</div>
        </div>
      </div>
    </div>

    <!-- ACTIVITY -->
    <div class="card card-clickable" data-details="status-bar">
      <div class="sh">
        <div><div class="st">Recent Events</div><div class="ss">Live pipeline activity</div></div>
      </div>
      <div class="act">
        <div class="act-icon" style="background:rgba(248,113,113,0.15);color:#FCA5A5;">⚠</div>
        <div style="flex:1">
          <div>Factual hallucination · step 4</div>
          <div class="act-time">12s ago · traj_198 · severity HIGH</div>
        </div>
        <div class="sev" style="background:var(--danger);box-shadow:0 0 5px var(--danger)"></div>
      </div>
      <div class="act">
        <div class="act-icon" style="background:rgba(251,191,36,0.15);color:#FCD34D;">↩</div>
        <div style="flex:1">
          <div>Step rollback triggered · step 3</div>
          <div class="act-time">44s ago · traj_197</div>
        </div>
        <div class="sev" style="background:var(--warn)"></div>
      </div>
      <div class="act">
        <div class="act-icon" style="background:rgba(52,211,153,0.15);color:var(--success);">✓</div>
        <div style="flex:1">
          <div>Clean trajectory · 6 steps</div>
          <div class="act-time">1m ago · traj_196</div>
        </div>
        <div class="sev" style="background:var(--success)"></div>
      </div>
      <div class="act">
        <div class="act-icon" style="background:rgba(167,139,250,0.15);color:#C4B5FD;">🔄</div>
        <div style="flex:1">
          <div>Tool requery correction</div>
          <div class="act-time">2m ago · traj_195</div>
        </div>
        <div class="sev" style="background:var(--purple)"></div>
      </div>
      <div class="act">
        <div class="act-icon" style="background:rgba(45,212,191,0.15);color:var(--teal);">📊</div>
        <div style="flex:1">
          <div>FAISS index updated · 651 facts</div>
          <div class="act-time">5m ago · system</div>
        </div>
        <div class="sev" style="background:var(--teal)"></div>
      </div>
    </div>
  </div>

  <!-- TRAJECTORIES + ARCH ROW -->
  <div class="g2">

    <!-- TRAJECTORY TABLE -->
    <div class="card card-clickable" data-details="trajectory-card">
      <div class="sh">
        <div><div class="st">Trajectory Analysis</div><div class="ss">Latest 6 runs</div></div>
        <div class="tog-wrap">
          <div class="tog on" data-filter="all">All</div>
          <div class="tog" data-filter="hall">Hallucinated</div>
          <div class="tog" data-filter="clean">Clean</div>
        </div>
      </div>
      <table class="tbl">
        <thead>
          <tr>
            <th>ID</th>
            <th>Steps</th>
            <th>Category</th>
            <th>Step#</th>
            <th>Sev</th>
            <th>Score</th>
          </tr>
        </thead>
        <tbody id="tbl-body">
          <tr data-type="hall" data-details="trajectory-card"><td style="font-family:monospace;color:var(--acc)">traj_198</td><td>7</td><td><span class="tag t-fact">Factual</span></td><td>4</td><td><span class="sev" style="background:var(--danger)"></span></td><td style="color:var(--danger)">0.71</td></tr>
          <tr data-type="hall" data-details="trajectory-card"><td style="font-family:monospace;color:var(--acc)">traj_197</td><td>5</td><td><span class="tag t-cont">Contradiction</span></td><td>3</td><td><span class="sev" style="background:var(--warn)"></span></td><td style="color:var(--warn)">0.63</td></tr>
          <tr data-type="clean" data-details="trajectory-card"><td style="font-family:monospace;color:var(--acc)">traj_196</td><td>6</td><td><span class="tag t-none">Clean</span></td><td>—</td><td><span class="sev" style="background:var(--success)"></span></td><td style="color:var(--success)">0.96</td></tr>
          <tr data-type="hall" data-details="trajectory-card"><td style="font-family:monospace;color:var(--acc)">traj_195</td><td>8</td><td><span class="tag t-tool">Tool</span></td><td>6</td><td><span class="sev" style="background:var(--warn)"></span></td><td style="color:var(--warn)">0.58</td></tr>
          <tr data-type="hall" data-details="trajectory-card"><td style="font-family:monospace;color:var(--acc)">traj_194</td><td>4</td><td><span class="tag t-sem">Semantic</span></td><td>2</td><td><span class="sev" style="background:var(--purple)"></span></td><td style="color:var(--purple)">0.67</td></tr>
          <tr data-type="clean" data-details="trajectory-card"><td style="font-family:monospace;color:var(--acc)">traj_193</td><td>9</td><td><span class="tag t-none">Clean</span></td><td>—</td><td><span class="sev" style="background:var(--success)"></span></td><td style="color:var(--success)">0.94</td></tr>
        </tbody>
      </table>
    </div>

    <!-- ARCH FLOW -->
    <div class="card">
      <div class="sh">
        <div><div class="st">3-Layer Hybrid Cascade</div><div class="ss">Pipeline architecture</div></div>
        <div class="chip">v2 Active</div>
      </div>

      <!-- Layer 1 -->
      <div style="margin-bottom:10px;">
        <div style="font-size:9px;font-weight:600;color:var(--txt3);text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;">Layer 1 · Semantic</div>
        <div class="flow">
          <div class="fn hl" data-details="step-logger"><div class="fi">📝</div>Step Logger</div>
          <div class="fa">→</div>
          <div class="fn" data-details="semantic"><div class="fi">🔵</div>Semantic</div>
          <div class="fa">→</div>
          <div class="fn" data-details="tool-valid"><div class="fi">🔧</div>Tool Valid.</div>
        </div>
      </div>

      <!-- Layer 2 -->
      <div style="margin-bottom:10px;">
        <div style="font-size:9px;font-weight:600;color:var(--txt3);text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;">Layer 2 · Grounding</div>
        <div class="flow">
          <div class="fn" data-details="factual-nli"><div class="fi">🟣</div>Factual NLI</div>
          <div class="fa">→</div>
          <div class="fn" data-details="faiss-rag"><div class="fi">🗄</div>FAISS RAG</div>
          <div class="fa">→</div>
          <div class="fn" data-details="contradict"><div class="fi">🟡</div>Contradict.</div>
        </div>
      </div>

      <!-- Layer 3 -->
      <div style="margin-bottom:12px;">
        <div style="font-size:9px;font-weight:600;color:var(--txt3);text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;">Layer 3 · Attribution</div>
        <div class="flow">
          <div class="fn" data-details="fusion"><div class="fi">⚖</div>Fusion</div>
          <div class="fa">→</div>
          <div class="fn" data-details="localizer"><div class="fi">📍</div>Localizer</div>
          <div class="fa">→</div>
          <div class="fn" data-details="causal-cls"><div class="fi">🧠</div>Causal CLS</div>
          <div class="fa">→</div>
          <div class="fn" data-details="corrector"><div class="fi">🔁</div>Corrector</div>
        </div>
      </div>

      <div style="padding-top:11px;border-top:1px solid var(--glass-border);display:flex;gap:6px;flex-wrap:wrap;">
        <div style="background:var(--acc-soft);border:1px solid var(--acc-glow);border-radius:6px;padding:3px 8px;font-size:9px;color:var(--acc);cursor:pointer;" data-details="status-bar">FastAPI backend</div>
        <div style="background:rgba(45,212,191,0.1);border:1px solid rgba(45,212,191,0.2);border-radius:6px;padding:3px 8px;font-size:9px;color:var(--teal);cursor:pointer;" data-details="status-bar">Streamlit UI</div>
        <div style="background:rgba(167,139,250,0.1);border:1px solid rgba(167,139,250,0.2);border-radius:6px;padding:3px 8px;font-size:9px;color:var(--purple);cursor:pointer;" data-details="benchmark-card">WandB logging</div>
        <div style="background:rgba(251,191,36,0.1);border:1px solid rgba(251,191,36,0.2);border-radius:6px;padding:3px 8px;font-size:9px;color:var(--warn);cursor:pointer;" data-details="status-bar">HF Spaces ready</div>
      </div>
    </div>
  </div>
</div>

<!-- HISTORY LOGS WORKSPACE (TAB 2) -->
<div id="content-history" class="tab-content">
  <div class="card g-full">
    <div class="sh" style="border-bottom:1px solid var(--glass-border); padding-bottom:10px; margin-bottom:15px;">
      <div>
        <div class="st">Analysis History Logs</div>
        <div class="ss">Inspect and compare past trajectory audit details stored in browser localStorage</div>
      </div>
      <button id="clear-history-btn" class="chip" style="cursor:pointer; border:none; outline:none; background:rgba(248,113,113,0.12); color:var(--danger); border:1px solid rgba(248,113,113,0.25);">Clear History</button>
    </div>
    
    <div class="g3" style="grid-template-columns: 1fr 2fr; gap:16px;">
      <!-- Left Side: List of runs -->
      <div style="display:flex; flex-direction:column; gap:10px; max-height:650px; overflow-y:auto; padding-right:4px;" id="history-list">
        <!-- History items render here -->
      </div>
      
      <!-- Right Side: Details View -->
      <div class="card" id="history-detail-card" style="min-height:350px; display:flex; flex-direction:column;">
        <div id="history-detail-placeholder" style="margin:auto; text-align:center; color:var(--txt3); font-size:12px;">
          <span style="font-size:28px; display:block; margin-bottom:8px;">📁</span>
          Select a past run from the list to view its complete step details and Expected vs Actual comparison audit.
        </div>
        <div id="history-detail-content" style="display:none; width: 100%;">
          <div class="sh" style="border-bottom: 1px solid var(--glass-border); padding-bottom: 8px; margin-bottom: 12px;">
            <div style="max-width: 70%;">
              <div class="st" id="hist-title" style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Task Analysis Details</div>
              <div class="ss" id="hist-meta">analyzed on ...</div>
            </div>
            <button id="hist-download-btn" class="chip" style="cursor:pointer; border:none; outline:none; background:var(--acc-soft); color:var(--acc);">Download JSON</button>
          </div>
          <div id="hist-steps-list">
            <!-- Steps of the historical run render here -->
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- BOTTOM STATUS BAR -->
<div class="card card-clickable" style="padding:12px 18px;" data-details="status-bar">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
    <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
      <div style="font-size:10px;color:var(--txt3);">Git: <span style="color:var(--txt2);font-family:monospace;">546bbaa</span></div>
      <div style="font-size:10px;color:var(--txt3);">~6,495 LOC · 28 files</div>
      <div style="font-size:10px;color:var(--txt3);">Target: <span style="color:var(--acc);">EMNLP 2026 / ICLR 2027</span></div>
      <div style="font-size:10px;color:var(--txt3);">Team: Somnath · Ayaan · Aman</div>
    </div>
    <div style="display:flex;align-items:center;gap:6px;">
      <div class="sdot" style="background:var(--success);box-shadow:0 0 6px var(--success);animation:pulse 2s infinite;"></div>
      <div style="font-size:10px;color:var(--success);">All systems operational · Clean repo</div>
    </div>
  </div>
</div>

</div><!-- end wrap -->

<!-- DETAIL SLIDING DRAWER -->
<div id="drawer" class="drawer">
  <div class="drawer-header">
    <div id="drawer-title" class="drawer-title">Component Details</div>
    <button class="drawer-close" id="drawer-close">&times;</button>
  </div>
  <div class="drawer-body">
    <div class="drawer-meta">
      <div class="drawer-meta-row">
        <span class="drawer-meta-label">Primary File</span>
        <span class="drawer-meta-value" id="drawer-file">N/A</span>
      </div>
      <div class="drawer-meta-row">
        <span class="drawer-meta-label">Line Count</span>
        <span class="drawer-meta-value" id="drawer-loc">N/A</span>
      </div>
      <div class="drawer-meta-row">
        <span class="drawer-meta-label">Module Type</span>
        <span class="drawer-meta-value" id="drawer-type">N/A</span>
      </div>
    </div>
    <div class="drawer-desc" id="drawer-desc">
      Select a component to inspect details.
    </div>
    <div class="drawer-extra" id="drawer-extra">
      <!-- Component specific technical parameters -->
    </div>
  </div>
</div>

<script>
const root = document.documentElement;

// Component summaries data map (100% accurate relative to AgentTrace project status & config)
const componentData = {
  'step-localization': {
    title: 'Step Localization Accuracy',
    file: 'evaluation/metrics.py',
    loc: '562 LOC',
    type: 'Evaluation Metric',
    desc: 'Measures how accurately the detection pipeline pinpoints the exact first step (root cause) of a hallucination in an agentic trajectory.<br><br>Attribution is evaluated using exact-match accuracy against human annotations, where no partial credit is given for adjacent steps.',
    extra: '<h5>Current Metrics</h5><ul><li><strong>AgentTrace Accuracy:</strong> <code style="color:var(--success)">0.6550</code> (65.5%)</li><li><strong>AgentHallu SOTA:</strong> <code>0.4110</code> (41.1%)</li><li><strong>Delta Improvement:</strong> <code style="color:var(--success)">+0.2440</code> (+59.4% improvement)</li></ul>'
  },
  'avg-latency': {
    title: 'Average Latency',
    file: 'evaluation/metrics.py',
    loc: '562 LOC',
    type: 'Performance Metric',
    desc: 'Tracks the average duration required to execute a complete multi-step trajectory analysis, including sentence embedding computation, NLI inference, contradiction checking, and causal classification.',
    extra: '<h5>Current Metrics</h5><ul><li><strong>Average Latency:</strong> <code>411.90 ms</code></li><li><strong>SLA Threshold:</strong> <code>500.00 ms</code></li><li><strong>Status:</strong> <span style="color:var(--success)">✓ Well under budget</span></li></ul>'
  },
  'p95-latency': {
    title: 'P95 Latency',
    file: 'evaluation/metrics.py',
    loc: '562 LOC',
    type: 'Performance Metric',
    desc: 'Monitors tail latency at the 95th percentile, representing the worst-case response times in production environments. Crucial for assessing pipeline efficiency on longer trajectories.',
    extra: '<h5>Current Metrics</h5><ul><li><strong>P95 Latency:</strong> <code>574.30 ms</code></li><li><strong>SLA Alert Boundary:</strong> <code>600.00 ms</code></li><li><strong>Status:</strong> <span style="color:var(--warn)">◐ Near threshold (Stable)</span></li></ul>'
  },
  'delta-baseline': {
    title: 'Δ vs Baseline',
    file: 'evaluation/benchmark.py',
    loc: '571 LOC',
    type: 'Performance Gain',
    desc: 'Quantifies the absolute and relative improvement of our 3-Layer Hybrid Ensemble Cascade over the state-of-the-art AgentHallu baseline published in arXiv:2601.06818.',
    extra: '<h5>Comparison</h5><ul><li><strong>Our Score:</strong> <code style="color:var(--success)">0.6550</code></li><li><strong>AgentHallu SOTA:</strong> <code>0.4110</code></li><li><strong>Absolute Gain:</strong> <code>+0.2440</code></li><li><strong>Relative Gain:</strong> <code>+59.4%</code></li></ul>'
  },
  'step-logger': {
    title: 'Step Logger',
    file: 'tracer/step_logger.py',
    loc: '607 LOC',
    type: 'Execution Tracer',
    desc: 'Captures and stores step-by-step execution traces of LLM agents in real-time. Logs the interaction unit <code>u_t = (c_t, a_t, o_t)</code> containing agent thoughts, action parameters, and tool observations.',
    extra: '<h5>Key Features</h5><ul><li>Real-time JSON/JSONL logging</li><li>Step diffing & semantic drift windows</li><li>Crash-safe intermediate checkpointing</li><li>Step replay engine for benchmark execution</li></ul>'
  },
  'semantic': {
    title: 'Semantic Checker',
    file: 'detection/semantic_checker.py',
    loc: '133 LOC',
    type: 'Detection Module',
    desc: 'Analyzes semantic alignment and detects gradual semantic drift between the agent\'s internal reasoning (thought), selected action, and the retrieved context.',
    extra: '<h5>Configuration</h5><ul><li><strong>Model:</strong> <code>sentence-transformers/all-MiniLM-L6-v2</code></li><li><strong>Embedding Dimension:</strong> <code>384</code></li><li><strong>Similarity Cutoff:</strong> <code>0.75</code> (cosine sim below this flags drift)</li></ul>'
  },
  'tool-valid': {
    title: 'Tool Validator',
    file: 'detection/tool_validator.py',
    loc: '192 LOC',
    type: 'Detection Module',
    desc: 'Verifies structured claims and parameters inside tool calls to check if tool inputs/outputs align with the current context.',
    extra: '<h5>Configuration</h5><ul><li><strong>Method:</strong> Claim extraction + keyword overlap</li><li><strong>Overlap Threshold:</strong> <code>0.40</code> (Jaccard similarity below this flags tool hallucination)</li></ul>'
  },
  'factual-nli': {
    title: 'Factual Grounding (NLI)',
    file: 'detection/factual_grounding.py',
    loc: '189 LOC',
    type: 'Detection Module',
    desc: 'Computes factual entailment and checks if the agent\'s reasoning statements are fully supported by the reference context.',
    extra: '<h5>Configuration</h5><ul><li><strong>Model:</strong> <code>cross-encoder/nli-deberta-v3-small</code></li><li><strong>Contradiction Threshold:</strong> <code>0.80</code> (contradiction score above this flags hallucination)</li><li><strong>RAG Fallback:</strong> Top-3 context lookup from FAISS index when local context is empty</li></ul>'
  },
  'faiss-rag': {
    title: 'FAISS Vector Index',
    file: 'indexes/build_index.py',
    loc: 'Build script',
    type: 'Knowledge Retrieval',
    desc: 'Maintains a vector index built from a verified database of facts. Used as a dynamic grounding fallback during the factual NLI step when context is missing.',
    extra: '<h5>Statistics</h5><ul><li><strong>FAISS database path:</strong> <code>indexes/fact_index.faiss</code></li><li><strong>Unique facts index:</strong> <code>651 facts</code></li><li><strong>Context extraction:</strong> Top-3 semantic match RAG</li></ul>'
  },
  'contradict': {
    title: 'Contradiction Checker',
    file: 'detection/contradiction.py',
    loc: '138 LOC',
    type: 'Detection Module',
    desc: 'Validates self-consistency by checking if a new step contradicts any statement made in previously executed steps within a sliding window.',
    extra: '<h5>Configuration</h5><ul><li><strong>Method:</strong> Cross-step pairwise NLI</li><li><strong>Window Size:</strong> <code>3</code> steps</li><li><strong>Contradiction Threshold:</strong> <code>0.80</code></li></ul>'
  },
  'fusion': {
    title: 'Hybrid Fusion Orchestrator',
    file: 'detection/pipeline.py',
    loc: '449 LOC',
    type: 'Pipeline Controller',
    desc: 'Combines and orchestrates the multi-signal inputs (Semantic, Tool, Grounding, and Contradiction) into a unified judgment.',
    extra: '<h5>Configuration</h5><ul><li><strong>Fusion Threshold:</strong> <code>0.45</code></li><li><strong>Categories:</strong> 5-class type classifier</li><li><strong>Routing:</strong> Action-aware routing</li></ul>'
  },
  'localizer': {
    title: 'Localization Ranking',
    file: 'attribution/localizer.py',
    loc: '142 LOC',
    type: 'Attribution Module',
    desc: 'Performs weighted signal localization to identify the earliest step responsible for the outcome deviation (root-cause step).',
    extra: '<h5>Signal Weights</h5><ul><li><strong>Factual NLI:</strong> <code>35%</code></li><li><strong>Semantic Similarity:</strong> <code>25%</code></li><li><strong>Contradiction:</strong> <code>20%</code></li><li><strong>Tool Claim Match:</strong> <code>20%</code></li></ul>'
  },
  'causal-cls': {
    title: 'Causal Classifier',
    file: 'attribution/causal_classifier.py',
    loc: '183 LOC',
    type: 'Attribution Module',
    desc: 'Attributes the root cause of the flagged hallucination to one of 5 classes in our paper taxonomy.',
    extra: '<h5>Model Details</h5><ul><li><strong>Classifier:</strong> Fine-tuned <code>distilbert-base-uncased</code></li><li><strong>Confidence Cutoff:</strong> <code>0.60</code></li><li><strong>Fallback:</strong> Heuristic rule-based regex classifier</li></ul>'
  },
  'corrector': {
    title: 'Active Corrector',
    file: 'intervention/corrector.py',
    loc: '163 LOC',
    type: 'Intervention Engine',
    desc: 'Applies active intervention strategies to recover from detected hallucinations during execution.',
    extra: '<h5>Strategies (in order)</h5><ol><li><strong>Tool Requery:</strong> Re-run tool with adjusted inputs</li><li><strong>Reasoning Override:</strong> Patch thoughts with facts</li><li><strong>Step Rollback:</strong> Roll back to last clean state</li></ol><ul><li><strong>Max Retries:</strong> <code>3</code></li></ul>'
  },
  'benchmark-card': {
    title: 'Benchmark Suite',
    file: 'evaluation/benchmark.py',
    loc: '571 LOC',
    type: 'Evaluation Engine',
    desc: 'Runs the full evaluation pipeline over the 200 trajectories dataset (543 KB) to benchmark localization accuracy, recall, and latency, logging results to WandB.',
    extra: '<h5>Baselines Overpowered</h5><ul><li><strong>Semantic-only:</strong> 0.2910</li><li><strong>NLI-only:</strong> 0.3180</li><li><strong>Tool-validator-only:</strong> 0.2740</li><li><strong>AgentHallu SOTA:</strong> 0.4110</li><li><strong>AgentTrace:</strong> 0.6550</li></ul>'
  },
  'trajectory-card': {
    title: 'Trajectory Dataset',
    file: 'data/trajectories/',
    loc: 'Directory',
    type: 'Dataset',
    desc: 'Contains 200 human-annotated and synthetic agent execution trajectories used for benchmarking, covering multiple domains and failure modes.',
    extra: '<h5>Taxonomy Distribution</h5><ul><li><strong>Planning Hallucinations:</strong> ~9.7%</li><li><strong>Retrieval Hallucinations:</strong> ~11.8%</li><li><strong>Reasoning Hallucinations:</strong> ~17.0%</li><li><strong>Tool-Use Hallucinations:</strong> ~14.9%</li><li><strong>Human-Interaction:</strong> ~10.5%</li></ul>'
  },
  'status-bar': {
    title: 'AgentTrace Codebase',
    file: 'Project Root',
    loc: '~6,495 LOC',
    type: 'Architecture Status',
    desc: 'A comprehensive view of the codebase status for the AgentTrace project targeted for publication at EMNLP 2026 / ICLR 2027.',
    extra: '<h5>Team & Roles</h5><ul><li><strong>Somnath Reddy:</strong> Research Lead & Architect (Tracer/FAISS)</li><li><strong>Ayaan:</strong> Detection & Attribution (Pipeline/Models)</li><li><strong>Aman:</strong> Deployment, UI, Data & Evaluation</li></ul>'
  }
};

function showDetails(key) {
  const data = componentData[key];
  if (!data) return;
  
  // Fill data in drawer
  document.getElementById('drawer-title').textContent = data.title;
  document.getElementById('drawer-file').textContent = data.file;
  document.getElementById('drawer-loc').textContent = data.loc;
  document.getElementById('drawer-type').textContent = data.type;
  document.getElementById('drawer-desc').innerHTML = data.desc;
  document.getElementById('drawer-extra').innerHTML = data.extra || '';
  
  // Open drawer
  const drawer = document.getElementById('drawer');
  drawer.classList.add('open');
  
  // Remove existing active highlights and apply to target element if applicable
  document.querySelectorAll('.fn, .card, .pipe-row, #tbl-body tr').forEach(el => el.classList.remove('active-hl'));
}

function closeDrawer() {
  document.getElementById('drawer').classList.remove('open');
}

// Close drawer when clicking outside it
document.addEventListener('click', function(e) {
  const drawer = document.getElementById('drawer');
  if (drawer.classList.contains('open') && 
      !drawer.contains(e.target) && 
      !e.target.closest('.fn') && 
      !e.target.closest('.card') && 
      !e.target.closest('.pipe-row') && 
      !e.target.closest('#tbl-body tr') && 
      !e.target.closest('.tog-wrap')) {
    closeDrawer();
  }
});

function setAcc(name, el) {
  document.body.setAttribute('data-accent', name);
  document.querySelectorAll('.acc-dot').forEach(d => d.classList.remove('active'));
  el.classList.add('active');
}

function toggleTheme() {
  const isDark = document.body.getAttribute('data-theme') === 'dark';
  document.body.setAttribute('data-theme', isDark ? 'light' : 'dark');
  document.getElementById('theme-btn').textContent = isDark ? '🌙' : '☀';
}

function filterTbl(type, el) {
  document.querySelectorAll('.tog').forEach(t => t.classList.remove('on'));
  el.classList.add('on');
  document.querySelectorAll('#tbl-body tr').forEach(r => {
    if (type === 'all') r.style.display = '';
    else if (type === 'hall') r.style.display = r.dataset.type === 'hall' ? '' : 'none';
    else r.style.display = r.dataset.type === 'clean' ? '' : 'none';
  });
}

// ---------------------------------------------------------------------------
// Tabs System
// ---------------------------------------------------------------------------
function initTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      
      tab.classList.add('active');
      const targetId = `content-${tab.dataset.tab}`;
      document.getElementById(targetId).classList.add('active');
      
      if (tab.dataset.tab === 'history') {
        renderHistoryList();
      }
    });
  });
}

// ---------------------------------------------------------------------------
// LocalStorage History Audit Log Tracking
// ---------------------------------------------------------------------------
const HISTORY_KEY = 'agenttrace_history';

function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
  } catch (e) {
    return [];
  }
}

function saveToHistory(runData) {
  const history = getHistory();
  const idx = history.findIndex(h => h.trajectory_id === runData.trajectory_id);
  if (idx !== -1) {
    history[idx] = runData;
  } else {
    history.unshift(runData);
  }
  if (history.length > 20) {
    history.pop();
  }
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

function clearHistory() {
  localStorage.removeItem(HISTORY_KEY);
  renderHistoryList();
  document.getElementById('history-detail-placeholder').style.display = 'flex';
  document.getElementById('history-detail-content').style.display = 'none';
}

function renderHistoryList() {
  const listContainer = document.getElementById('history-list');
  const history = getHistory();
  
  if (history.length === 0) {
    listContainer.innerHTML = `
      <div style="text-align:center; padding:30px; color:var(--txt3); font-size:12px;">
        No saved runs yet. Run a live analysis to save logs.
      </div>
    `;
    return;
  }
  
  listContainer.innerHTML = '';
  history.forEach(run => {
    const dateStr = new Date(run.created_at).toLocaleString();
    const isMismatch = run.steps.some(step => step.is_hallucinated !== step.expected_is_hallucinated);
    const statusBadge = isMismatch 
      ? `<span class="tag" style="background:rgba(251,191,36,0.12); color:var(--warn); border:1px solid rgba(251,191,36,0.25);">Mismatch Audit</span>`
      : `<span class="tag" style="background:rgba(52,211,153,0.11); color:var(--success); border:1px solid rgba(52,211,153,0.22);">All Matched</span>`;
    
    const itemHtml = `
      <div class="card card-clickable history-item" data-run-id="${run.trajectory_id}" style="padding:12px 14px; margin-bottom:8px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <span style="font-family:monospace; font-size:11px; color:var(--acc2); font-weight:600;">${run.trajectory_id.slice(0, 8)}</span>
          <span style="font-size:9px; color:var(--txt3);">${dateStr}</span>
        </div>
        <div style="font-size:11px; font-weight:500; color:var(--txt); margin-bottom:8px; text-overflow:ellipsis; overflow:hidden; white-space:nowrap; max-width:260px;">
          ${run.task}
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;">
          <div style="font-size:10px; color:var(--txt2);">
            Steps: <strong>${run.num_steps}</strong> | Halls: <strong style="${run.num_hallucinated > 0 ? 'color:var(--danger);' : ''}">${run.num_hallucinated}</strong>
          </div>
          ${statusBadge}
        </div>
      </div>
    `;
    listContainer.insertAdjacentHTML('beforeend', itemHtml);
  });
}

let selectedHistoryRun = null;

function selectHistoryRun(runId) {
  const history = getHistory();
  const run = history.find(h => h.trajectory_id === runId);
  if (!run) return;
  
  selectedHistoryRun = run;
  
  document.querySelectorAll('.history-item').forEach(el => {
    if (el.dataset.runId === runId) {
      el.classList.add('active-hl');
    } else {
      el.classList.remove('active-hl');
    }
  });
  
  document.getElementById('history-detail-placeholder').style.display = 'none';
  document.getElementById('history-detail-content').style.display = 'block';
  
  document.getElementById('hist-title').textContent = run.task;
  document.getElementById('hist-meta').textContent = `Analyzed on ${new Date(run.created_at).toLocaleString()} | ID: ${run.trajectory_id}`;
  
  renderSteps(run.steps, run.trajectory_id, 'hist-steps-list');
}

function downloadHistoryJson() {
  if (!selectedHistoryRun) return;
  const jsonStr = JSON.stringify(selectedHistoryRun, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `agenttrace_hist_${selectedHistoryRun.trajectory_id.slice(0, 8)}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// API Integration
// ---------------------------------------------------------------------------
const API_BASE = "__API_BASE__";

// Check API health on load
async function checkApiHealth() {
  const badge = document.getElementById('api-badge');
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error("Offline");
    const data = await res.json();
    badge.innerHTML = `<span class="pulse-dot"></span>Live · API Online · v${data.version}`;
    badge.className = "live-badge";
    badge.style.background = "rgba(52,211,153,0.11)";
    badge.style.border = "1px solid rgba(52,211,153,0.28)";
    badge.style.color = "var(--success)";
  } catch (err) {
    badge.innerHTML = `<span class="pulse-dot" style="background:var(--danger)"></span>API Offline`;
    badge.className = "live-badge";
    badge.style.background = "rgba(248,113,113,0.11)";
    badge.style.border = "1px solid rgba(248,113,113,0.28)";
    badge.style.color = "var(--danger)";
    console.error("API connection failed:", err);
  }
}

let lastResult = null;

// Run trace analysis
async function runAnalysis() {
  const taskInput = document.getElementById('task-input');
  const task = taskInput.value.trim();
  if (!task) return;
  
  const btn = document.getElementById('analyze-btn');
  const resultsContainer = document.getElementById('results-container');
  const resultsMeta = document.getElementById('results-meta');
  
  btn.disabled = true;
  btn.textContent = "Analyzing...";
  
  try {
    const startTime = performance.now();
    const res = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task: task })
    });
    
    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Analysis failed");
    }
    
    const data = await res.json();
    const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
    
    lastResult = data;
    
    // Save live run to history
    saveToHistory(data);
    
    // Display results
    resultsContainer.style.display = 'block';
    resultsMeta.textContent = `analyzed in ${elapsed}s`;
    
    // Update metric cards dynamically based on current trace results
    document.getElementById('m1').innerHTML = `${data.overall_confidence.toFixed(3)} <span>acc</span>`;
    
    // Render steps
    renderSteps(data.steps, data.trajectory_id, 'steps-list');
    
    // Re-check API health
    checkApiHealth();
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
    
  } catch (err) {
    alert("Error executing trace: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze Trace";
  }
}

function renderSteps(steps, trajectoryId, targetContainerId = 'steps-list') {
  const list = document.getElementById(targetContainerId);
  list.innerHTML = '';
  
  steps.forEach(step => {
    const isHall = step.is_hallucinated;
    const cardClass = isHall ? 'step-hallucinated' : 'step-clean';
    
    const actualText = isHall ? (step.hallucination_type || 'Hallucinated') : 'Clean';
    const expectedText = step.expected_is_hallucinated ? (step.expected_hallucination_type || 'Hallucinated') : 'Clean';
    
    // Determine alignment status
    let alignmentText = '';
    let alignmentClass = '';
    if (isHall === step.expected_is_hallucinated) {
      if (isHall) {
        alignmentText = 'Match (True Positive)';
        alignmentClass = 'align-match-tp';
      } else {
        alignmentText = 'Match (True Negative)';
        alignmentClass = 'align-match-tn';
      }
    } else {
      if (isHall) {
        alignmentText = 'Mismatch (False Positive)';
        alignmentClass = 'align-mismatch-fp';
      } else {
        alignmentText = 'Mismatch (False Negative)';
        alignmentClass = 'align-mismatch-fn';
      }
    }
    
    const expectedBadgeClass = step.expected_is_hallucinated ? 'badge-expected-hallucinated' : 'badge-expected-clean';
    const actualBadgeClass = isHall ? 'badge-actual-hallucinated' : 'badge-actual-clean';
    const scorePct = (step.hallucination_score * 100).toFixed(0);
    
    let scoreClass = 'score-bar-fill-green';
    if (step.hallucination_score >= 0.85) {
      scoreClass = 'score-bar-fill-red';
    } else if (step.hallucination_score >= 0.60) {
      scoreClass = 'score-bar-fill-amber';
    }
    
    const prettyActualType = actualText.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    const prettyExpectedType = expectedText.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    
    let explanationHtml = '';
    if (step.explanation) {
      const explainBorderClass = (isHall === step.expected_is_hallucinated) ? 'explain-match' : 'explain-mismatch';
      explanationHtml = `
        <div class="hallucination-explanation ${explainBorderClass}">
          <span class="explain-icon">${(isHall === step.expected_is_hallucinated) ? '✓' : '⚠️'}</span>
          <strong>Analysis Audit Explanation:</strong> ${step.explanation}
        </div>
      `;
    }
    
    let correctButtonHtml = '';
    if (isHall) {
      correctButtonHtml = `
        <div style="margin-top: 10px;">
          <button class="btn-correct" data-trajectory-id="${trajectoryId}" data-step-index="${step.step_index}">
            Apply Correction to Step ${step.step_index}
          </button>
        </div>
      `;
    }
    
    const stepHtml = `
      <div class="step-card ${cardClass}">
        <div class="step-header" style="flex-wrap: wrap; gap: 8px;">
          <span class="step-title">Step ${step.step_index}</span>
          <div style="display: flex; gap: 6px; flex-wrap: wrap;">
            <span class="step-badge ${expectedBadgeClass}">Expected: ${prettyExpectedType}</span>
            <span class="step-badge ${actualBadgeClass}">Actual: ${prettyActualType}</span>
            <span class="step-badge ${alignmentClass}">${alignmentText}</span>
          </div>
        </div>
        <p class="step-detail"><strong>Tool:</strong> ${step.tool_name}</p>
        <p class="step-detail"><strong>Action:</strong> ${step.action}</p>
        <p class="step-detail"><strong>Input:</strong> <code>${escapeHtml(step.tool_input)}</code></p>
        <p class="step-detail"><strong>Output:</strong> <code>${escapeHtml(step.tool_output)}</code></p>
        <p class="step-detail"><strong>Reasoning:</strong> ${step.reasoning}</p>
        <p class="step-detail"><strong>Hallucination Score:</strong> ${scorePct}%</p>
        
        <div class="score-bar-container">
          <div class="score-bar-bg">
            <div class="score-bar-fill ${scoreClass}" style="width:${scorePct}%;"></div>
          </div>
        </div>
        
        ${explanationHtml}
        ${correctButtonHtml}
        <div class="correction-container" style="margin-top:10px;"></div>
      </div>
    `;
    
    list.insertAdjacentHTML('beforeend', stepHtml);
  });
}

function escapeHtml(text) {
  if (!text) return '';
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

async function applyCorrection(trajectoryId, stepIndex, btn) {
  const container = btn.closest('.step-card').querySelector('.correction-container');
  btn.disabled = true;
  btn.textContent = "Applying intervention...";
  
  try {
    const res = await fetch(`${API_BASE}/correct`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ trajectory_id: trajectoryId, step: stepIndex })
    });
    
    if (!res.ok) throw new Error("Correction request failed");
    const data = await res.json();
    
    const prettyStrategy = data.intervention_type.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
    
    container.innerHTML = `
      <div class="correction-card">
        <h4>Intervention Applied &mdash; Step ${data.step_index}</h4>
        <p style="margin-bottom:4px; font-size:11px; color:var(--txt2);"><strong>Intervention Strategy:</strong> <span style="color:var(--acc2); font-weight:600;">${prettyStrategy}</span></p>
        <p style="margin-bottom:4px; font-size:11px; color:var(--txt2);"><strong>Original Action:</strong> ${data.original_action}</p>
        <p style="margin-bottom:4px; font-size:11px; color:var(--txt2);"><strong>Original Output:</strong> <code>${escapeHtml(data.original_output)}</code></p>
        <p style="margin-bottom:4px; font-size:11px; color:var(--success);"><strong>Corrected Action:</strong> ${data.corrected_action}</p>
        <p style="margin-bottom:4px; font-size:11px; color:var(--success);"><strong>Corrected Output:</strong> <code>${escapeHtml(data.corrected_output)}</code></p>
        <p style="margin-bottom:0; font-size:11px; color:var(--txt2);"><strong>Confidence After:</strong> ${(data.confidence_after * 100).toFixed(0)}%</p>
      </div>
    `;
    
    btn.style.display = 'none'; // hide button after successful correction
    
  } catch (err) {
    alert("Correction error: " + err.message);
    btn.disabled = false;
    btn.textContent = `Apply Correction to Step ${stepIndex}`;
  }
}

function downloadJson() {
  if (!lastResult) return;
  const jsonStr = JSON.stringify(lastResult, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `agenttrace_${lastResult.trajectory_id.slice(0, 8)}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function handleCorrectClick(e) {
  if (e.target.classList.contains('btn-correct')) {
    const trajId = e.target.dataset.trajectoryId;
    const stepIdx = parseInt(e.target.dataset.stepIndex);
    applyCorrection(trajId, stepIdx, e.target);
  }
}

// ---------------------------------------------------------------------------
// Initialization and Event Bindings
// ---------------------------------------------------------------------------
window.addEventListener('load', () => {
  // Check API health status
  checkApiHealth();
  
  // Initialize navigation tabs
  initTabs();
  
  // Bind accent selector dots
  document.querySelectorAll('.acc-dot').forEach(el => {
    el.addEventListener('click', () => {
      setAcc(el.dataset.acc, el);
    });
  });
  
  // Bind theme toggle button
  document.getElementById('theme-btn').addEventListener('click', toggleTheme);
  
  // Bind test scenario selection dropdown change
  document.getElementById('scenario-select').addEventListener('change', (e) => {
    const val = e.target.value;
    const input = document.getElementById('task-input');
    if (val) {
      input.value = val;
    } else {
      input.value = '';
    }
  });
  
  // Bind live analyze button
  document.getElementById('analyze-btn').addEventListener('click', runAnalysis);
  
  // Bind download button
  document.getElementById('download-btn').addEventListener('click', downloadJson);
  
  // Bind details drawer close button
  document.getElementById('drawer-close').addEventListener('click', closeDrawer);
  
  // Bind table filters
  const togWrap = document.querySelector('.tog-wrap');
  if (togWrap) {
    togWrap.addEventListener('click', (e) => {
      e.stopPropagation();
    });
  }
  document.querySelectorAll('.tog').forEach(el => {
    el.addEventListener('click', (e) => {
      e.stopPropagation();
      filterTbl(el.dataset.filter, el);
    });
  });
  
  // Bind details triggers (metric cards, pipe rows, bottom status bar, cascade flow nodes, table rows)
  document.querySelectorAll('[data-details]').forEach(el => {
    el.addEventListener('click', (e) => {
      if (e.target.closest('.tog-wrap')) return;
      showDetails(el.dataset.details);
    });
  });
  
  // Bind Correction button click delegations
  document.getElementById('steps-list').addEventListener('click', handleCorrectClick);
  document.getElementById('hist-steps-list').addEventListener('click', handleCorrectClick);
  
  // Bind history logs action and items click delegation
  document.getElementById('clear-history-btn').addEventListener('click', clearHistory);
  document.getElementById('hist-download-btn').addEventListener('click', downloadHistoryJson);
  document.getElementById('history-list').addEventListener('click', (e) => {
    const item = e.target.closest('.history-item');
    if (item) {
      selectHistoryRun(item.dataset.runId);
    }
  });

  // Animate static UI bars
  setTimeout(() => {
    const bars = [
      ['b1', 65.5], ['b2', 41.1], ['b3', 29.1], ['b4', 31.8], ['b5', 27.4]
    ];
    bars.forEach(([id, pct]) => {
      const el = document.getElementById(id);
      if (el) el.style.width = (pct / 65.5 * 100) + '%';
    });

    const pipes = [
      ['p1', 84], ['p2', 79], ['p3', 91], ['p4', 76]
    ];
    pipes.forEach(([id, pct]) => {
      const el = document.getElementById(id);
      if (el) el.style.width = pct + '%';
    });
  }, 200);

  // Sparkline — simulated commit history scores
  const scores = [0.18, 0.21, 0.19, 0.28, 0.31, 0.41, 0.48, 0.55, 0.62, 0.655];
  const W = 320, H = 48, pad = 4;
  const xs = scores.map((_, i) => pad + (i / (scores.length - 1)) * (W - pad * 2));
  const ys = scores.map(s => H - pad - (s / 0.7) * (H - pad * 2));
  const line = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${ys[i].toFixed(1)}`).join(' ');
  const area = line + ` L${xs[xs.length-1].toFixed(1)} ${H} L${xs[0].toFixed(1)} ${H} Z`;
  const sl = document.getElementById('spark-line');
  const sa = document.getElementById('spark-area');
  if (sl) sl.setAttribute('d', line);
  if (sa) sa.setAttribute('d', area);
});
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
# Inject base URL dynamically and render the component
html_rendered = HTML_TEMPLATE.replace("__API_BASE__", API_BASE)
components.html(html_rendered, height=1380, scrolling=True)
