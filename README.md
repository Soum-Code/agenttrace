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
| Aman | amano2 | Deployment and UI |
| Dustin | TgoxDustin08 | Data and Evaluation |

---

## Project Structure

agenttrace/ 
├── config.py                          # Member 1: Central parameters, thresholds, and paths
├── requirements.txt                   # Pinned package versions
├── README.md                          # Project documentation
│
├── api/                               # Member 3
│   └── main.py                        # FastAPI backend core
│
├── ui/                                # Member 3
│   └── app.py                         # Streamlit visual demo
│
├── tracer/                            # Member 1
│   └── step_logger.py                 # Real-time step logging logic
│
├── detection/                         # Member 2
│   ├── semantic_checker.py            # Cosine similarity validation
│   ├── tool_validator.py              # Structured tool output validation
│   ├── factual_grounding.py           # NLI grounding score engine
│   └── contradiction.py               # Cross-step semantic drift]
│
├── attribution/                       # Member 2
│   ├── localizer.py                   # Step localization logic
│   └── causal_classifier.py           # Fine-tuned DistilBERT inference
│
├── intervention/                      # Member 2
│   └── corrector.py                   # Targeted fix strategies
│
├── evaluation/                        # Members 1 & 4
│   ├── metrics.py                     # All core metric functions (Member 1)
│   ├── benchmark.py                   # Full baseline evaluation pipeline (Member 1)]
│   ├── ablation.py                    # Component ablation scripts (Member 4)
│   └── visualizer.py                  # Generation of results charts (Member 4)
│
├── data/                              # Members 1 & 4
│   ├── synthetic_generator.py         # Generation of 200 synthetic trajectories (Member 1)
│   ├── agenthallu_loader.py           # Benchmark dataset loader (Member 4)
│   └── real_trajectory_generator.py   # Collection of real-world trajectories (Member 4)
│
├── trajectories/                      # Directory for stored JSON trajectory files (gitignored)
│
├── results/                           # Directory for JSON evaluation results output
│
└── paper/                             # Project Publication
    ├── agenttrace_paper.tex           # Main LaTeX manuscript
    └── figures/  
---

## Status

Month 1 — In Progress

---

## Paper Target

arXiv preprint → EMNLP 2026 Workshop

---

## Tech Stack

Python, LangChain, Sentence-Transformers, FAISS,
DistilBERT, FastAPI, Streamlit, HuggingFace Spaces,
Gemini API, Weights and Biases
