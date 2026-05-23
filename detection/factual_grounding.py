"""
Detection Module: Factual Grounding.
Uses Natural Language Inference (NLI) via a cross-encoder to determine
whether the agent's reasoning is entailed by, contradicts, or is neutral
with respect to the tool output / ground truth.
"""

import json
import sys
import os

# Add project root to sys.path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
except ImportError:
    pass

import config


class FactualGrounder:
    """
    Uses NLI to assess whether tool output (premise) entails the agent
    reasoning (hypothesis).  Returns an nli_score representing contradiction
    probability — higher means more likely hallucinated.
    """

    # NLI label indices for cross-encoder/nli-deberta-v3-small
    _LABEL_MAP = {"contradiction": 0, "entailment": 1, "neutral": 2}

    def __init__(self, model=None, tokenizer=None, embedding_model=None):
        """
        Loads the NLI cross-encoder model and tokenizer from config.
        Also loads the FAISS index if it exists.
        """
        try:
            if model is not None:
                self.model = model
                self.tokenizer = tokenizer
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(config.NLI_MODEL_NAME)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    config.NLI_MODEL_NAME
                )
                self.model.eval()
            self.entailment_threshold = config.NLI_ENTAILMENT_THRESHOLD
            self.contradiction_threshold = config.NLI_CONTRADICTION_THRESHOLD
        except Exception as e:
            print(f"Error loading NLI model: {e}")
            self.model = None
            self.tokenizer = None

        # Load FAISS index and metadata for dynamic RAG factual grounding
        self.faiss_index = None
        self.faiss_metadata = None
        self.embedding_model = embedding_model

        index_path = os.path.join(config.CONFIG.paths.index_dir, "fact_index.faiss")
        metadata_path = os.path.join(config.CONFIG.paths.index_dir, "fact_metadata.json")

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                import faiss
                self.faiss_index = faiss.read_index(index_path)
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.faiss_metadata = json.load(f)
                print(f"FactualGrounder: Loaded FAISS index from {index_path} with {self.faiss_index.ntotal} facts.")
            except Exception as e:
                print(f"FactualGrounder: Warning: Failed to load FAISS index: {e}")

    def _lazy_load_embedding_model(self):
        """Loads SentenceTransformer embedding model on-demand to save resources."""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(config.SEMANTIC_MODEL_NAME)
                print(f"FactualGrounder: Loaded SentenceTransformer model '{config.SEMANTIC_MODEL_NAME}' for RAG.")
            except Exception as e:
                print(f"FactualGrounder: Error loading embedding model: {e}")

    # ── public API ──────────────────────────────────────────────────────────

    def ground(self, step_data: dict) -> dict:
        """
        Performs NLI grounding on a single agent step.

        Args:
            step_data (dict): Structured JSON for a single agent step.

        Returns:
            dict: Detection result including nli_score (contradiction probability).
        """
        step_id = step_data.get("step")
        agent_reasoning = step_data.get("agent_reasoning", "")
        # Use ground_truth as primary premise; fall back to tool_output
        premise = step_data.get("ground_truth") or step_data.get("tool_output", "")

        retrieved_facts_list = []
        is_rag_fallback = False

        # RAG fallback: If primary premise is missing, retrieve top-3 facts from FAISS index
        if not premise or not str(premise).strip():
            if self.faiss_index is not None and self.faiss_metadata is not None:
                try:
                    self._lazy_load_embedding_model()
                    if self.embedding_model is not None:
                        query = agent_reasoning
                        query_vector = self.embedding_model.encode([query], convert_to_numpy=True).astype("float32")
                        
                        k = min(3, len(self.faiss_metadata))
                        distances, indices = self.faiss_index.search(query_vector, k)
                        
                        retrieved = []
                        for idx in indices[0]:
                            if 0 <= idx < len(self.faiss_metadata):
                                retrieved.append(self.faiss_metadata[idx])
                        
                        if retrieved:
                            premise = "\n".join(retrieved)
                            retrieved_facts_list = retrieved
                            is_rag_fallback = True
                except Exception as e:
                    print(f"FactualGrounder: Error retrieving facts from FAISS index: {e}")

        if not self.model or not self.tokenizer:
            return self._fallback(step_id, "NLI model not initialized")

        try:
            probs = self._predict_nli(premise, agent_reasoning)

            contradiction_prob = probs[self._LABEL_MAP["contradiction"]]
            entailment_prob = probs[self._LABEL_MAP["entailment"]]
            neutral_prob = probs[self._LABEL_MAP["neutral"]]

            # Use contradiction probability as the main NLI signal
            nli_score = round(float(contradiction_prob), 4)

            hallucination_detected = (
                contradiction_prob >= self.contradiction_threshold
                or entailment_prob < self.entailment_threshold
            )

            # Confidence based on how decisive the NLI verdict is
            confidence = round(float(max(contradiction_prob, 1.0 - entailment_prob)), 2)

            severity = self._compute_severity(contradiction_prob, hallucination_detected)

            return {
                "step": step_id,
                "hallucination_detected": hallucination_detected,
                "confidence": confidence,
                "detection_signals": {
                    "semantic_similarity": None,
                    "tool_claim_match": None,
                    "nli_score": nli_score,
                    "contradiction_with_prev": None,
                },
                "hallucination_type": config.TYPE_FACTUAL if hallucination_detected else None,
                "severity": severity,
                "detail": {
                    "entailment": round(float(entailment_prob), 4),
                    "contradiction": round(float(contradiction_prob), 4),
                    "neutral": round(float(neutral_prob), 4),
                    "is_rag_fallback": is_rag_fallback,
                    "retrieved_facts": retrieved_facts_list
                },
            }

        except Exception as e:
            return self._fallback(step_id, f"NLI inference error: {str(e)}")

    # ── internal helpers ────────────────────────────────────────────────────

    def _predict_nli(self, premise: str, hypothesis: str) -> list:
        """
        Runs NLI inference and returns softmax probabilities.

        Args:
            premise (str): The grounding text (tool output / ground truth).
            hypothesis (str): The agent reasoning to evaluate.

        Returns:
            list: Softmax probabilities [contradiction, entailment, neutral].
        """
        inputs = self.tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        return probs

    def _compute_severity(self, contradiction_prob: float, detected: bool) -> str | None:
        """
        Maps contradiction probability to severity level.

        Args:
            contradiction_prob (float): NLI contradiction probability.
            detected (bool): Whether hallucination was flagged.

        Returns:
            str | None: Severity string or None.
        """
        if not detected:
            return None
        if contradiction_prob > 0.85:
            return config.SEVERITY_HIGH
        if contradiction_prob > 0.6:
            return config.SEVERITY_MEDIUM
        return config.SEVERITY_LOW

    def _fallback(self, step_id: int, error_msg: str) -> dict:
        """
        Generates a safe fallback response on error.
        """
        return {
            "step": step_id,
            "hallucination_detected": False,
            "confidence": 0.0,
            "error": error_msg,
            "detection_signals": {
                "semantic_similarity": None,
                "tool_claim_match": None,
                "nli_score": None,
                "contradiction_with_prev": None,
            },
            "hallucination_type": None,
            "severity": None,
        }


# ── Independent test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    grounder = FactualGrounder()

    # Test 1: Standard NLI with ground_truth
    sample_input = {
        "step": 2,
        "action": "web_search",
        "tool_input": "FIFA 2022 winner",
        "tool_output": "Argentina won FIFA World Cup 2022",
        "agent_reasoning": "France won FIFA 2022, capital is Paris",
        "ground_truth": "Argentina won FIFA 2022",
    }

    print("\n--- Test 1: Standard Factual Grounding ---")
    result = grounder.ground(sample_input)
    print(json.dumps(result, indent=2))

    # Test 2: Dynamic RAG fallback (no ground_truth, no tool_output)
    sample_rag = {
        "step": 3,
        "action": "none",
        "tool_input": None,
        "tool_output": "",
        "agent_reasoning": "Mount Everest is located on the border between Nepal and China.",
    }

    print("\n--- Test 2: Dynamic RAG Grounding Fallback ---")
    result_rag = grounder.ground(sample_rag)
    print(json.dumps(result_rag, indent=2))
