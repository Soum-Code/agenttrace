"""
AgentTrace -- Streamlit Demo UI
================================
Interactive frontend for the AgentTrace hallucination detection system.

Features
--------
- Text input for user task description
- Step-by-step agent trace visualisation
- Hallucinated steps highlighted in RED, clean steps in GREEN
- Detailed hallucination explanations per step
- Sidebar with model confidence scores
- Download button to export full trace as JSON
- Loading states with spinners
- Mobile-friendly responsive layout

Run
---
    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
def inject_custom_css() -> None:
    """
    Inject custom CSS for premium styling.

    Applies a dark-themed design system with vibrant accent colours,
    smooth animations, glassmorphism panels, and mobile-responsive
    layout overrides.
    """
    st.markdown(
        """
        <style>
        /* ---- Import Google Font ---- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ---- Global ---- */
        .stApp {
            font-family: 'Inter', sans-serif;
        }

        /* ---- Header ---- */
        .main-header {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            padding: 2rem 2rem 1.5rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        .main-header h1 {
            color: #ffffff;
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0 0 0.3rem 0;
            letter-spacing: -0.5px;
        }
        .main-header p {
            color: rgba(255,255,255,0.65);
            font-size: 1rem;
            margin: 0;
            font-weight: 300;
        }
        .main-header .badge {
            display: inline-block;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            margin-top: 0.6rem;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        /* ---- Metric Cards ---- */
        .metric-row {
            display: flex;
            gap: 0.8rem;
            margin-bottom: 1.2rem;
            flex-wrap: wrap;
        }
        .metric-card {
            flex: 1;
            min-width: 130px;
            background: linear-gradient(145deg, #1a1a2e, #16213e);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            text-align: center;
            box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        }
        .metric-card .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
            line-height: 1.2;
        }
        .metric-card .metric-label {
            font-size: 0.75rem;
            color: rgba(255,255,255,0.5);
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-top: 0.3rem;
            font-weight: 500;
        }
        .metric-blue .metric-value  { color: #60a5fa; }
        .metric-red .metric-value   { color: #f87171; }
        .metric-green .metric-value { color: #4ade80; }
        .metric-amber .metric-value { color: #fbbf24; }

        /* ---- Step Cards ---- */
        .step-card {
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 0.9rem;
            border-left: 5px solid;
            box-shadow: 0 2px 12px rgba(0,0,0,0.15);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .step-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(0,0,0,0.25);
        }
        .step-clean {
            background: linear-gradient(145deg, #052e16, #064e3b);
            border-left-color: #22c55e;
        }
        .step-hallucinated {
            background: linear-gradient(145deg, #450a0a, #7f1d1d);
            border-left-color: #ef4444;
        }
        .step-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.6rem;
            flex-wrap: wrap;
            gap: 0.4rem;
        }
        .step-title {
            font-weight: 700;
            font-size: 1rem;
            color: #ffffff;
        }
        .step-badge {
            font-size: 0.7rem;
            font-weight: 600;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .badge-clean {
            background: rgba(34,197,94,0.2);
            color: #4ade80;
            border: 1px solid rgba(34,197,94,0.3);
        }
        .badge-hallucinated {
            background: rgba(239,68,68,0.2);
            color: #f87171;
            border: 1px solid rgba(239,68,68,0.3);
            animation: pulse-red 2s ease-in-out infinite;
        }
        @keyframes pulse-red {
            0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.3); }
            50%      { box-shadow: 0 0 0 6px rgba(239,68,68,0); }
        }
        .step-detail {
            font-size: 0.85rem;
            color: rgba(255,255,255,0.7);
            margin: 0.25rem 0;
            line-height: 1.5;
        }
        .step-detail strong {
            color: rgba(255,255,255,0.9);
            font-weight: 600;
        }

        /* ---- Explanation callout ---- */
        .hallucination-explanation {
            background: rgba(239,68,68,0.1);
            border: 1px solid rgba(239,68,68,0.25);
            border-radius: 8px;
            padding: 0.8rem 1rem;
            margin-top: 0.6rem;
            font-size: 0.82rem;
            color: #fca5a5;
            line-height: 1.5;
        }
        .hallucination-explanation .explain-icon {
            font-weight: 700;
            color: #f87171;
            margin-right: 0.4rem;
        }

        /* ---- Score bar ---- */
        .score-bar-container {
            margin: 0.5rem 0;
        }
        .score-bar-bg {
            background: rgba(255,255,255,0.08);
            border-radius: 6px;
            height: 8px;
            overflow: hidden;
        }
        .score-bar-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 0.6s ease;
        }
        .score-bar-fill-green { background: linear-gradient(90deg, #22c55e, #4ade80); }
        .score-bar-fill-red   { background: linear-gradient(90deg, #ef4444, #f87171); }
        .score-bar-fill-amber { background: linear-gradient(90deg, #f59e0b, #fbbf24); }

        /* ---- Sidebar confidence list ---- */
        .confidence-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .confidence-item:last-child { border-bottom: none; }
        .confidence-step {
            font-weight: 600;
            font-size: 0.85rem;
            color: #e2e8f0;
        }
        .confidence-score {
            font-weight: 700;
            font-size: 0.85rem;
            padding: 0.15rem 0.5rem;
            border-radius: 6px;
        }
        .conf-high   { background: rgba(34,197,94,0.15);  color: #4ade80; }
        .conf-medium { background: rgba(245,158,11,0.15); color: #fbbf24; }
        .conf-low    { background: rgba(239,68,68,0.15);  color: #f87171; }

        /* ---- Correction result ---- */
        .correction-card {
            background: linear-gradient(145deg, #1e1b4b, #312e81);
            border: 1px solid rgba(99,102,241,0.2);
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            margin-top: 0.8rem;
            box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        }
        .correction-card h4 {
            color: #a5b4fc;
            margin: 0 0 0.6rem 0;
            font-size: 0.95rem;
        }

        /* ---- Status indicator ---- */
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
            animation: blink 1.5s ease-in-out infinite;
        }
        .status-dot-green { background: #22c55e; }
        .status-dot-red   { background: #ef4444; }
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50%      { opacity: 0.4; }
        }

        /* ---- Mobile tweaks ---- */
        @media (max-width: 768px) {
            .main-header { padding: 1.2rem; }
            .main-header h1 { font-size: 1.5rem; }
            .metric-card { min-width: 100px; padding: 0.7rem 0.8rem; }
            .metric-card .metric-value { font-size: 1.3rem; }
            .step-card { padding: 0.9rem 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def check_api_health() -> dict[str, Any] | None:
    """
    Hit GET /health and return the JSON payload.

    Returns:
        Parsed JSON dict on success, None on failure.
    """
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def call_analyze(task: str) -> dict[str, Any] | None:
    """
    Call POST /analyze with the given task description.

    Args:
        task: Natural-language task string from the user.

    Returns:
        Parsed JSON response on success, None on failure.
    """
    try:
        resp = httpx.post(
            f"{API_BASE}/analyze",
            json={"task": task},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"API call failed: {exc}")
        return None


def call_correct(trajectory_id: str, step: int) -> dict[str, Any] | None:
    """
    Call POST /correct to apply intervention on a specific step.

    Args:
        trajectory_id: UUID of the previously analyzed trajectory.
        step: 0-indexed step number to correct.

    Returns:
        Parsed JSON response on success, None on failure.
    """
    try:
        resp = httpx.post(
            f"{API_BASE}/correct",
            json={"trajectory_id": trajectory_id, "step": step},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"Correction failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_header() -> None:
    """Render the main page header with title, subtitle, and version badge."""
    st.markdown(
        """
        <div class="main-header">
            <h1>AgentTrace</h1>
            <p>Step-Level Hallucination Detection &amp; Attribution
               in Multi-Step LLM Agent Workflows</p>
            <span class="badge">Research Preview v0.1</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(data: dict[str, Any]) -> None:
    """
    Render the summary metric cards (total steps, hallucinated, confidence).

    Args:
        data: The full /analyze response dictionary.
    """
    confidence_pct = round(data["overall_confidence"] * 100, 1)
    clean_count = data["num_steps"] - data["num_hallucinated"]

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card metric-blue">
                <p class="metric-value">{data["num_steps"]}</p>
                <p class="metric-label">Total Steps</p>
            </div>
            <div class="metric-card metric-green">
                <p class="metric-value">{clean_count}</p>
                <p class="metric-label">Clean Steps</p>
            </div>
            <div class="metric-card metric-red">
                <p class="metric-value">{data["num_hallucinated"]}</p>
                <p class="metric-label">Hallucinated</p>
            </div>
            <div class="metric-card metric-amber">
                <p class="metric-value">{confidence_pct}%</p>
                <p class="metric-label">Confidence</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _score_bar_html(score: float) -> str:
    """
    Build a mini progress bar showing the hallucination score.

    Args:
        score: Float 0-1 hallucination probability.

    Returns:
        HTML string for the bar.
    """
    pct = round(score * 100, 1)
    if score < 0.4:
        fill_class = "score-bar-fill-green"
    elif score < 0.65:
        fill_class = "score-bar-fill-amber"
    else:
        fill_class = "score-bar-fill-red"

    return (
        f'<div class="score-bar-container">'
        f'  <div class="score-bar-bg">'
        f'    <div class="score-bar-fill {fill_class}" '
        f'         style="width:{pct}%"></div>'
        f'  </div>'
        f'</div>'
    )


def render_step_card(step: dict[str, Any]) -> None:
    """
    Render a single step as a styled card (green=clean, red=hallucinated).

    Shows tool name, input/output, reasoning, hallucination score bar,
    and an explanation callout for hallucinated steps.

    Args:
        step: A single step dictionary from the trajectory.
    """
    is_hall = step["is_hallucinated"]
    card_class = "step-hallucinated" if is_hall else "step-clean"
    badge_class = "badge-hallucinated" if is_hall else "badge-clean"
    badge_text = "HALLUCINATED" if is_hall else "CLEAN"
    score = step["hallucination_score"]

    explanation_html = ""
    if is_hall and step.get("explanation"):
        explanation_html = (
            f'<div class="hallucination-explanation">'
            f'  <span class="explain-icon">&#9888;</span>'
            f'  {step["explanation"]}'
            f'</div>'
        )

    hall_type_html = ""
    if is_hall and step.get("hallucination_type"):
        pretty_type = step["hallucination_type"].replace("_", " ").title()
        hall_type_html = (
            f'<p class="step-detail">'
            f'  <strong>Type:</strong> {pretty_type}'
            f'</p>'
        )

    st.markdown(
        f"""
        <div class="step-card {card_class}">
            <div class="step-header">
                <span class="step-title">Step {step["step_index"]}</span>
                <span class="step-badge {badge_class}">{badge_text}</span>
            </div>
            <p class="step-detail"><strong>Tool:</strong> {step["tool_name"]}</p>
            <p class="step-detail"><strong>Action:</strong> {step["action"]}</p>
            <p class="step-detail"><strong>Input:</strong>
               <code>{step["tool_input"]}</code></p>
            <p class="step-detail"><strong>Output:</strong>
               <code>{step["tool_output"]}</code></p>
            <p class="step-detail"><strong>Reasoning:</strong>
               {step["reasoning"]}</p>
            {hall_type_html}
            <p class="step-detail"><strong>Hallucination Score:</strong>
               {score:.1%}</p>
            {_score_bar_html(score)}
            {explanation_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_confidence(steps: list[dict[str, Any]]) -> None:
    """
    Render a per-step confidence list in the sidebar.

    Each step shows its index, tool name, and hallucination score
    colour-coded by severity.

    Args:
        steps: List of step dictionaries from the trajectory.
    """
    st.sidebar.markdown("### Per-Step Confidence")
    st.sidebar.markdown("---")

    for step in steps:
        score = step["hallucination_score"]
        # Confidence = inverse of hallucination score
        confidence = 1.0 - score

        if confidence >= 0.7:
            css_class = "conf-high"
        elif confidence >= 0.4:
            css_class = "conf-medium"
        else:
            css_class = "conf-low"

        st.sidebar.markdown(
            f"""
            <div class="confidence-item">
                <span class="confidence-step">
                    Step {step["step_index"]} &middot; {step["tool_name"]}
                </span>
                <span class="confidence-score {css_class}">
                    {confidence:.0%}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Overall stats in sidebar
    st.sidebar.markdown("---")
    total = len(steps)
    hallucinated = sum(1 for s in steps if s["is_hallucinated"])
    clean = total - hallucinated
    st.sidebar.metric("Total Steps", total)
    st.sidebar.metric("Clean", clean)
    st.sidebar.metric("Hallucinated", hallucinated)


def render_correction_result(result: dict[str, Any]) -> None:
    """
    Render the correction/intervention result card.

    Args:
        result: The /correct response dictionary.
    """
    st.markdown(
        f"""
        <div class="correction-card">
            <h4>Intervention Applied &mdash; Step {result["step_index"]}</h4>
            <p class="step-detail">
                <strong>Intervention Type:</strong>
                {result["intervention_type"].replace("_", " ").title()}
            </p>
            <p class="step-detail">
                <strong>Original Action:</strong> {result["original_action"]}
            </p>
            <p class="step-detail">
                <strong>Original Output:</strong>
                <code>{result["original_output"]}</code>
            </p>
            <p class="step-detail" style="color:#4ade80;">
                <strong>Corrected Action:</strong> {result["corrected_action"]}
            </p>
            <p class="step-detail" style="color:#4ade80;">
                <strong>Corrected Output:</strong>
                <code>{result["corrected_output"]}</code>
            </p>
            <p class="step-detail">
                <strong>Confidence After:</strong>
                {result["confidence_after"]:.1%}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Entry point for the Streamlit AgentTrace demo.

    Orchestrates the page layout:
      1. Page config and custom CSS
      2. Sidebar with API health and confidence scores
      3. Main area with task input, trace visualization, and corrections
    """
    # ---- Page config ----
    st.set_page_config(
        page_title="AgentTrace | Hallucination Detector",
        page_icon="&#128270;",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    # ---- Sidebar: API health ----
    st.sidebar.markdown("## AgentTrace")
    st.sidebar.markdown("---")

    health = check_api_health()
    if health:
        st.sidebar.markdown(
            '<span class="status-dot status-dot-green"></span>'
            f' **API Online** &nbsp;`v{health["version"]}`',
            unsafe_allow_html=True,
        )
        st.sidebar.caption(
            f"Uptime: {health['uptime_seconds']:.0f}s "
            f"&middot; Models loaded: {'Yes' if health['models_loaded'] else 'No'}"
        )
    else:
        st.sidebar.markdown(
            '<span class="status-dot status-dot-red"></span>'
            " **API Offline**",
            unsafe_allow_html=True,
        )
        st.sidebar.caption(
            "Start the backend:\n\n"
            "`uvicorn api.main:app --reload --port 8000`"
        )

    st.sidebar.markdown("---")

    # ---- Main: Header ----
    render_header()

    # ---- Main: Task input ----
    st.markdown("#### Describe an agent task to analyze")
    task_input = st.text_area(
        "Task description",
        placeholder=(
            "e.g. Find the population of Tokyo and compare it with Delhi, "
            "then calculate the ratio..."
        ),
        height=100,
        label_visibility="collapsed",
    )

    col_run, col_spacer = st.columns([1, 4])
    with col_run:
        run_clicked = st.button(
            "Analyze Trace",
            type="primary",
            use_container_width=True,
            disabled=not task_input.strip(),
        )

    # ---- Run analysis ----
    if run_clicked and task_input.strip():
        if not health:
            st.error(
                "API is offline. Start the backend with: "
                "`uvicorn api.main:app --reload --port 8000`"
            )
            return

        with st.spinner("Running agent and detecting hallucinations..."):
            start = time.time()
            result = call_analyze(task_input.strip())
            elapsed = time.time() - start

        if result:
            st.session_state["last_result"] = result
            st.session_state["elapsed"] = elapsed

    # ---- Display results ----
    result = st.session_state.get("last_result")
    if result:
        elapsed = st.session_state.get("elapsed", 0)

        st.markdown("---")
        st.markdown(
            f"#### Trace Results &nbsp;"
            f'<span style="color:rgba(255,255,255,0.4);font-size:0.8rem;">'
            f"analyzed in {elapsed:.1f}s</span>",
            unsafe_allow_html=True,
        )

        # Metrics
        render_metrics(result)

        # Sidebar confidence
        render_sidebar_confidence(result["steps"])

        # Step cards
        st.markdown("#### Step-by-Step Trace")
        for step in result["steps"]:
            render_step_card(step)

            # Correct button for hallucinated steps
            if step["is_hallucinated"]:
                btn_key = f"correct_{result['trajectory_id']}_{step['step_index']}"
                if st.button(
                    f"Apply Correction to Step {step['step_index']}",
                    key=btn_key,
                    type="secondary",
                ):
                    with st.spinner(
                        f"Applying intervention on step {step['step_index']}..."
                    ):
                        correction = call_correct(
                            result["trajectory_id"],
                            step["step_index"],
                        )
                    if correction:
                        render_correction_result(correction)

        # ---- Download JSON ----
        st.markdown("---")
        json_str = json.dumps(result, indent=2, ensure_ascii=False)
        st.download_button(
            label="Download Full Trace (JSON)",
            data=json_str,
            file_name=f"agenttrace_{result['trajectory_id'][:8]}.json",
            mime="application/json",
            type="primary",
        )

        # Trajectory ID footer
        st.caption(f"Trajectory ID: `{result['trajectory_id']}`")


if __name__ == "__main__":
    main()
