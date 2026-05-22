---
title: AgentTrace Demo
emoji: 🚀
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---
# AgentTrace

Step-Level Hallucination Detection and Attribution
in Multi-Step LLM Agent Workflows

Research project targeting EMNLP 2026 / ICLR 2027.
Beating AgentHallu SOTA of 41.1% step localization accuracy.

---

## Team

| Member | GitHub | Role |
|---|---|---|
| P. Somnath Reddy | Soum-Code | Research Lead and Architect |
| Ayaan | AyaanO7 | Detection and Attribution |
| Aman | amano2 | Deployment, UI, Data and Evaluation |

---

## Project Structure

```
AgentTrace/
├── config.py                          # Central parameters, thresholds, and paths
├── requirements.txt                   # Pinned package versions
├── README.md                          # Project documentation
├── PROJECT_STATUS.md                  # Comprehensive module status report
│
├── api/                               # FastAPI Backend Core (Aman)
│   └── main.py                        # Analysis & correction API endpoints
│
├── ui/                                # Streamlit Frontend Dashboard (Aman)
│   └── app.py                         # Premium live visualization interface
│
├── tracer/                            # Execution Tracer (Somnath)
│   └── step_logger.py                 # Real-time logging & step replay engine
│
├── detection/                         # Step Hallucination Detection (Ayaan)
│   ├── semantic_checker.py            # Cosine similarity validation
│   ├── tool_validator.py              # Structured tool claim verification
│   ├── factual_grounding.py           # NLI grounding & dynamic RAG fallback
│   ├── contradiction.py               # Cross-step semantic drift contradiction
│   └── pipeline.py                    # 3-Layer Hybrid Cascade Fusion orchestrator
│
├── attribution/                       # Root Cause Attribution (Ayaan)
│   ├── localizer.py                   # Localization & signal-weighted ranking
│   ├── causal_classifier.py           # Fine-tuned DistilBERT inference fallback
│   └── train_causal_classifier.py     # Causal classifier trainer
│
├── intervention/                      # Active Correction (Ayaan)
│   └── corrector.py                   # Tool requery, reasoning override & rollback
│
├── evaluation/                        # Evaluation, Charts & Paper (Aman & Somnath)
│   ├── metrics.py                     # Metric functions (Accuracy, Recall, Latency)
│   ├── benchmark.py                   # Dataset evaluator & WandB logger
│   ├── ablation.py                    # Component ablation evaluation
│   └── visualizer.py                  # Visualizations and paper tables generator
│
├── data/                              # Datasets & Generators (Aman & Somnath)
│   ├── trajectories/                  # Trajectory dataset storage (gitignored)
│   ├── synthetic_generator.py         # 200 synthetic trajectories generator
│   ├── agenthallu_loader.py           # Benchmark loader
│   └── real_trajectory_generator.py   # Real-world user trajectory capturer
│
├── indexes/                           # Retrieval Indexes (Somnath)
│   ├── build_index.py                 # FAISS vector builder
│   ├── fact_index.faiss               # FAISS vector database
│   └── fact_metadata.json             # Serialized source facts mapping
│
└── paper/                             # Project Publication (Aman)
    ├── main.tex                       # LaTeX source manuscript
    └── figures/                       # Generated charts and diagrams
```

---

## Status

**Phase 2 Development - Complete**
- ✅ **Nemotron Layer 3 judge** status exposed via `/health` API and Streamlit sidebar badge.
- ✅ **3-Member Alignment**: All files & roles redistributed. Dustin Richard removed.
- ✅ **FAISS Index built**: 651 unique facts serialized and stored.
- ✅ **Dynamic RAG Grounding**: Factual grounding pipeline dynamically performs top-3 FAISS context lookup on missing/empty step premises.
- ✅ **Benchmark verified**: Runs locally and inside Docker containers.

---

## SOTA Benchmark Performance

Evaluating our **3-Layer Hybrid Ensemble Cascade** on the benchmark dataset yields state-of-the-art results:

| Metric | Value | AgentHallu SOTA | Delta |
|---|---|---|---|
| **Step Localization Accuracy** | **0.6550** | 0.411 | **+0.2440** (Massive improvement) |
| **Average Latency** | **411.90 ms** | — | High-speed, under-budget |
| **P95 Latency** | **574.30 ms** | — | — |

---

## Quickstart

### 1. Setup Environment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Build FAISS Index
To recreate the vector index from synthetic trajectories:
```bash
python indexes/build_index.py
```

### 3. Run Benchmark Suite
To run the full evaluation over the 200 trajectories:
```bash
python run_benchmark.py
```

### 4. Start Local Dashboard
To start the FastAPI backend and Streamlit frontend:
```bash
# Terminal 1: FastAPI API
uvicorn api.main:app --port 8000

# Terminal 2: Streamlit Dashboard
streamlit run ui/app.py --server.port 7860
```

---

## Tech Stack

Python, FAISS Vector DB, Sentence-Transformers (`all-MiniLM-L6-v2`), DeBERTa (`nli-deberta-v3-small`), DistilBERT, FastAPI, Streamlit, Weights & Biases, Docker, HuggingFace Spaces.
