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
| Amano | amano2 | Deployment and UI |
| Dustin | TgoxDustin08 | Data and Evaluation |

---

## Project Structure

agenttrace/
├── config.py
├── tracer/          Member 1
├── detection/       Member 2
├── attribution/     Member 2
├── intervention/    Member 2
├── evaluation/      Members 1 and 4
├── api/             Member 3
├── ui/              Member 3
├── data/            Members 1 and 4
└── paper/           All members

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