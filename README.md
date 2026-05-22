---
title: AgentTrace Demo
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# AgentTrace

Step-level hallucination detection and attribution in multi-step LLM agent workflows.

This is a research project designed to identify and trace hallucinations back to their root causes within complex agent pathways, aiming for publication at EMNLP 2026 or ICLR 2027. The framework surpasses the AgentHallu SOTA step localization accuracy baseline of 41.1%.

---

## Team

| Member | GitHub | Role |
|---|---|---|
| P. Somnath Reddy | Soum-Code | Research Lead and Architect |
| Ayaan | AyaanO7 | Detection and Attribution Lead |
| Aman | amano2 | Deployment, UI, Data, and Evaluation Lead |

---

## Project Structure

```
AgentTrace/
├── config.py                          # Central parameters, thresholds, and paths
├── requirements.txt                   # Pinned package versions
├── README.md                          # Project documentation
├── PROJECT_STATUS.md                  # Comprehensive module status report
│
├── api/                               # FastAPI backend core
│   └── main.py                        # Analysis and correction API endpoints
│
├── ui/                                # Streamlit frontend dashboard
│   └── app.py                         # Web interface for live visualization
│
├── tracer/                            # Execution tracer
│   └── step_logger.py                 # Real-time logging and step replay engine
│
├── detection/                         # Step hallucination detection
│   ├── semantic_checker.py            # Cosine similarity validation
│   ├── tool_validator.py              # Structured tool claim verification
│   ├── factual_grounding.py           # NLI grounding and dynamic RAG fallback
│   ├── contradiction.py               # Cross-step semantic drift contradiction
│   └── pipeline.py                    # Three-layer hybrid cascade fusion orchestrator
│
├── attribution/                       # Root cause attribution
│   ├── localizer.py                   # Localization and signal-weighted ranking
│   ├── causal_classifier.py           # Fine-tuned DistilBERT inference fallback
│   └── train_causal_classifier.py     # Causal classifier trainer
│
├── intervention/                      # Active correction
│   └── corrector.py                   # Tool requery, reasoning override, and rollback
│
├── evaluation/                        # Evaluation, charts, and paper assets
│   ├── metrics.py                     # Metric functions (Accuracy, Recall, Latency)
│   ├── benchmark.py                   # Dataset evaluator and WandB logger
│   ├── ablation.py                    # Component ablation evaluation
│   └── visualizer.py                  # Visualizations and paper tables generator
│
├── data/                              # Datasets and generators
│   ├── trajectories/                  # Trajectory dataset storage (gitignored)
│   ├── synthetic_generator.py         # OpenRouter-based trajectory generator
│   ├── agenthallu_loader.py           # Benchmark dataset loader
│   └── real_trajectory_generator.py   # Real-world user trajectory capturer
│
├── indexes/                           # Retrieval indexes
│   ├── build_index.py                 # FAISS vector builder
│   ├── fact_index.faiss               # FAISS vector database
│   └── fact_metadata.json             # Serialized source facts mapping
│
└── paper/                             # Project publication
    ├── main.tex                       # LaTeX source manuscript
    └── figures/                       # Generated charts and diagrams
```

---

## Status

**Phase 2 Development - Complete**
- **Nemotron Layer 3 judge**: Status is exposed via the `/health` API and visualized on the Streamlit sidebar badge.
- **Three-Member Alignment**: Cleaned up the author list and redistributed files and responsibilities across the three members.
- **FAISS Index**: Built and serialized the index from 651 unique facts.
- **Dynamic RAG Grounding**: The factual grounding pipeline automatically performs a top-3 FAISS context lookup when a step's premise is missing or empty.
- **Benchmark Suite**: Verified functionality locally and inside Docker containers.

---

## SOTA Benchmark Performance

Evaluating our three-layer hybrid ensemble cascade on the benchmark dataset yields state-of-the-art results:

| Metric | Value | AgentHallu SOTA | Delta |
|---|---|---|---|
| **Step Localization Accuracy** | **0.6550** | 0.411 | **+0.2440** (Significant SOTA improvement) |
| **Average Latency** | **411.90 ms** | — | Highly responsive and optimized |
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

Python, FAISS Vector Database, Sentence-Transformers (all-MiniLM-L6-v2), DeBERTa (nli-deberta-v3-small), DistilBERT, FastAPI, Streamlit, Weights and Biases, Docker, HuggingFace Spaces.
