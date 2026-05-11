"""
Detection Module: Contradiction Detector.
Checks whether the agent's current reasoning contradicts its own prior
reasoning steps using NLI in a sliding window.
"""

import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
except ImportError:
    pass

import config


class ContradictionDetector:
    """
    Detects contradictions between the current agent step and previous
    steps using NLI with a configurable sliding window.
    """

    _LABEL_MAP = {"contradiction": 0, "entailment": 1, "neutral": 2}

    def __init__(self):
        """Loads the NLI model for inter-step contradiction detection."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.NLI_MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(config.NLI_MODEL_NAME)
            self.model.eval()
            self.window_size = config.CONTRADICTION_WINDOW_SIZE
            self.contradiction_threshold = config.NLI_CONTRADICTION_THRESHOLD
        except Exception as e:
            print(f"Error loading NLI model for ContradictionDetector: {e}")
            self.model = None
            self.tokenizer = None

    def detect(self, step_data: dict, history: list) -> dict:
        """
        Checks current step reasoning against recent history for contradictions.

        Args:
            step_data: The current agent step JSON.
            history: List of previous step JSONs, ordered chronologically.

        Returns:
            dict: Detection result with contradiction_with_prev signal.
        """
        step_id = step_data.get("step")
        current_reasoning = step_data.get("agent_reasoning", "")

        if not self.model or not self.tokenizer:
            return self._fallback(step_id, "NLI model not initialized")

        if not history:
            return self._no_contradiction(step_id)

        try:
            recent = history[-self.window_size:]
            contradictions = []
            for prev_step in recent:
                prev_reasoning = prev_step.get("agent_reasoning", "")
                if not prev_reasoning:
                    continue
                probs = self._predict_nli(prev_reasoning, current_reasoning)
                contra_prob = probs[self._LABEL_MAP["contradiction"]]
                if contra_prob >= self.contradiction_threshold:
                    contradictions.append({
                        "contradicts_step": prev_step.get("step"),
                        "contradiction_prob": round(float(contra_prob), 4),
                    })

            contradiction_found = len(contradictions) > 0
            max_prob = max((c["contradiction_prob"] for c in contradictions), default=0.0)
            confidence = round(max_prob, 2) if contradiction_found else 0.0

            severity = None
            if contradiction_found:
                if max_prob > 0.9:
                    severity = config.SEVERITY_HIGH
                elif max_prob > 0.75:
                    severity = config.SEVERITY_MEDIUM
                else:
                    severity = config.SEVERITY_LOW

            return {
                "step": step_id,
                "hallucination_detected": contradiction_found,
                "confidence": confidence,
                "detection_signals": {
                    "semantic_similarity": None, "tool_claim_match": None,
                    "nli_score": None, "contradiction_with_prev": contradiction_found,
                },
                "hallucination_type": config.TYPE_CONTRADICTION if contradiction_found else None,
                "severity": severity,
                "detail": {"window_size": self.window_size, "contradictions": contradictions},
            }
        except Exception as e:
            return self._fallback(step_id, f"Contradiction inference error: {str(e)}")

    def _predict_nli(self, premise: str, hypothesis: str) -> list:
        """Runs NLI inference and returns softmax probabilities."""
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.softmax(logits, dim=-1).squeeze().tolist()

    def _no_contradiction(self, step_id: int) -> dict:
        """Returns a clean no-contradiction result for the first step."""
        return {
            "step": step_id, "hallucination_detected": False, "confidence": 0.0,
            "detection_signals": {"semantic_similarity": None, "tool_claim_match": None,
                                  "nli_score": None, "contradiction_with_prev": False},
            "hallucination_type": None, "severity": None,
        }

    def _fallback(self, step_id: int, error_msg: str) -> dict:
        """Generates a safe fallback response on error."""
        return {
            "step": step_id, "hallucination_detected": False, "confidence": 0.0,
            "error": error_msg,
            "detection_signals": {"semantic_similarity": None, "tool_claim_match": None,
                                  "nli_score": None, "contradiction_with_prev": None},
            "hallucination_type": None, "severity": None,
        }


if __name__ == "__main__":
    detector = ContradictionDetector()
    prev = [{"step": 1, "agent_reasoning": "Argentina won the 2022 FIFA World Cup"}]
    curr = {"step": 2, "agent_reasoning": "France won FIFA 2022, capital is Paris"}
    print(json.dumps(detector.detect(curr, prev), indent=2))
