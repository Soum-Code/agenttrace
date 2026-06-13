# Product Requirement Document (PRD) — AgentTrace

## 1. Product Vision & Background
In multi-step Large Language Model (LLM) agent workflows (e.g., ReAct, plan-and-solve), intermediate step failures compound. A single reasoning deviation, factual hallucination, or tool parameter mismatch in step $t$ contaminates the context window, causing absolute task failure by step $t+n$. Coarse evaluation frameworks only check the final output, providing no diagnostic visibility into intermediate trajectories. 

**AgentTrace** is an enterprise-grade step-level hallucination detection, attribution, and real-time intervention system. It sits alongside agent runtimes as an active sentinel to intercept reasoning drifts, attribute errors to a 6-class taxonomy, and perform dynamic code-level rollbacks or tool re-queries before errors cascade.

---

## 2. Target Audience & Personas
*   **AI Systems Engineers:** Need an active, low-latency safeguard to prevent production LLM agents from wasting API credits on corrupted trajectories.
*   **NLP Researchers:** Need a benchmark-aligned framework to inspect intermediate reasoning failures and validate error classification models against State-of-the-Art benchmarks (like AgentHallu).
*   **Enterprise Developers:** Need a visual control-loop interface to inspect agent runs, trace tool calls, and diagnose failure modes on a step-by-step basis.

---

## 3. Core Features & Functional Requirements

### 3.1 Step-Level Hallucination Detection
*   **Requirement:** Intercept and score every step of an active agent trajectory.
*   **Input:** Current step state $\{ \text{action}, \text{tool\_input}, \text{tool\_output}, \text{agent\_reasoning}, t \}$.
*   **Output:** Hallucination score $S(s_t) \in [0, 1]$ and boolean flag `is_hallucinated`.
*   **SLA:** Sub-300ms latency on average to allow real-time interception without stalling agent loops.

### 3.2 6-Class Error Attribution
*   **Requirement:** Classify detected hallucinations into the standardized taxonomy:
    1.  `Planning`: Logical errors in step sequencing or sub-goal decomposition.
    2.  `Retrieval`: Out-of-context or ungrounded knowledge usage.
    3.  `Reasoning`: Contradictions, logical loops, or non-sequitur claims.
    4.  `Tool-Use`: Schema validation failures, invalid parameters, or hallucinations of tool outputs.
    5.  `Human-Interaction`: Misinterpreting user guidelines or constraints.
    6.  `No-Hallucination`: The step is factually grounded and mathematically sound.

### 3.3 3-Layer Hybrid Cascading (Cost & Speed Optimization)
*   **Requirement:** Minimize token consumption and server latency by using an adaptive routing pipeline:
    *   **Layer 1 (Local SLM Ensemble):** Runs lightweight checkers (SentenceTransformers & DeBERTa NLI) for instant validation. Handles $>70\%$ of clear cases.
    *   **Layer 2 (Local Fine-Tuned Llama):** Fine-tuned 8B parameter sequence classifier that handles step classification and attribution.
    *   **Layer 3 (High-Capacity Nemotron Judge):** Triggered only when Layer 2 classification confidence is below $\tau_{\text{conf}} = 0.70$.

### 3.4 Active Intervention Strategies
*   **Requirement:** Allow programmatic recovery from flagged steps:
    *   `tool_requery`: Rerun a tool with corrected arguments.
    *   `reasoning_override`: Inject factual prompts/constraints into the context.
    *   `step_rollback`: Revert agent history to step $t-1$ and force plan regeneration.

### 3.5 Real-Time Interactive Dashboard
*   **Requirement:** Serve a polished, responsive web dashboard with dark-mode glassmorphic styling, enabling:
    *   Dynamic scenario selection (Planning, Retrieval, Tool, etc.).
    *   Live step-by-step annotation (green for clean, red for hallucinated).
    *   Attribution visualization and latency metrics.

---

## 4. Key Performance Indicators (KPIs) & Target Metrics
*   **Step Localization Accuracy:** $\ge 58\%$ (representing a massive improvement over AgentHallu's 41.1% baseline).
*   **Average End-to-End Latency:** $\le 300\text{ ms}$ (P95 latency $\le 450\text{ ms}$).
*   **Tool-Use Corrected Accuracy:** $\ge 95\%$ (measured as successful task completion after intervention).
*   **Cascade Efficiency Rate:** $\ge 75\%$ of steps resolved by Layers 1 & 2 without calling the Layer 3 API.
