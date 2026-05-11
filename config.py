"""
Configuration parameters for AgentTrace Detection and Attribution modules.
All thresholds and model names are centralized here — no hardcoded values in modules.
"""

# ─── Model Names ──────────────────────────────────────────────────────────────
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
CAUSAL_MODEL_NAME = "distilbert-base-uncased"  # Fine-tuned checkpoint path replaces this

# ─── Detection Thresholds ─────────────────────────────────────────────────────
SEMANTIC_SIMILARITY_THRESHOLD = 0.7  # Below this → potential hallucination
NLI_ENTAILMENT_THRESHOLD = 0.5       # Entailment probability must exceed this
NLI_CONTRADICTION_THRESHOLD = 0.7    # Contradiction probability above this → flag

# ─── Tool Validator Settings ──────────────────────────────────────────────────
TOOL_CLAIM_SIMILARITY_THRESHOLD = 0.65  # Claim-to-output match threshold
MAX_CLAIMS_PER_REASONING = 20           # Cap claim extraction to avoid runaway

# ─── Contradiction Detector Settings ─────────────────────────────────────────
CONTRADICTION_WINDOW_SIZE = 3  # Number of previous steps to check for contradictions

# ─── Localizer Settings ──────────────────────────────────────────────────────
LOCALIZATION_SIGNAL_WEIGHTS = {
    "semantic_similarity": 0.25,
    "tool_claim_match": 0.30,
    "nli_score": 0.30,
    "contradiction_with_prev": 0.15,
}

# ─── Causal Classifier Settings ──────────────────────────────────────────────
CAUSAL_LABELS = [
    "Retrieval-Error",
    "Reasoning-Error",
    "Instruction-Drift",
    "Tool-Misuse",
    "Context-Overflow",
]
CAUSAL_CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence for a causal label

# ─── Corrector Settings ──────────────────────────────────────────────────────
CORRECTOR_MAX_RETRIES = 3          # Max correction attempts before escalation
CORRECTOR_STRATEGY_ORDER = [       # Strategies tried in order
    "tool_requery",
    "reasoning_override",
    "step_rollback",
]

# ─── Hallucination Types ─────────────────────────────────────────────────────
TYPE_TOOL_USE = "Tool-Use"
TYPE_FACTUAL = "Factual"
TYPE_CONTRADICTION = "Contradiction"

# ─── Severity Levels ─────────────────────────────────────────────────────────
SEVERITY_LOW = "Low"
SEVERITY_MEDIUM = "Medium"
SEVERITY_HIGH = "High"

# ─── Severity Thresholds (semantic similarity boundaries) ─────────────────────
SEVERITY_HIGH_UPPER = 0.3    # similarity < 0.3 → High
SEVERITY_MEDIUM_UPPER = 0.5  # similarity < 0.5 → Medium, else Low
