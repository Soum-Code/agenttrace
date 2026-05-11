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

    def __init__(self):
        """
        Loads the NLI cross-encoder model and tokenizer from config.
        """
        try:
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

    sample_input = {
        "step": 2,
        "action": "web_search",
        "tool_input": "FIFA 2022 winner",
        "tool_output": "Argentina won FIFA World Cup 2022",
        "agent_reasoning": "France won FIFA 2022, capital is Paris",
        "ground_truth": "Argentina won FIFA 2022",
    }

    result = grounder.ground(sample_input)
    print(json.dumps(result, indent=2))
