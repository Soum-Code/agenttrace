"""
Detection Module: Tool Validator.
Verifies whether the agent's reasoning accurately reflects the tool output.
Extracts claims from reasoning and checks each against tool output using
sentence-level semantic similarity.
"""

import json
import re
import sys
import os

# Add project root to sys.path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    pass

import config


class ToolValidator:
    """
    Validates that the agent's reasoning faithfully reflects tool output.
    Splits reasoning into sentence-level claims, then checks each claim
    against tool output via cosine similarity.
    """

    def __init__(self):
        """
        Initializes the SentenceTransformer model for claim matching.
        """
        try:
            self.model = SentenceTransformer(config.SEMANTIC_MODEL_NAME)
            self.threshold = config.TOOL_CLAIM_SIMILARITY_THRESHOLD
            self.max_claims = config.MAX_CLAIMS_PER_REASONING
        except Exception as e:
            print(f"Error loading model for ToolValidator: {e}")
            self.model = None

    # ── public API ──────────────────────────────────────────────────────────

    def validate(self, step_data: dict) -> dict:
        """
        Validates a single agent step by comparing reasoning claims to tool output.

        Args:
            step_data (dict): Structured JSON for a single agent step.

        Returns:
            dict: Validation result with tool_claim_match flag, per-claim
                  scores, and hallucination metadata.
        """
        step_id = step_data.get("step")
        agent_reasoning = step_data.get("agent_reasoning", "")
        tool_output = step_data.get("tool_output", "")

        if not self.model:
            return self._fallback(step_id, "Model not initialized")

        try:
            claims = self._extract_claims(agent_reasoning)
            if not claims:
                return self._fallback(step_id, "No claims extracted from reasoning")

            # Encode tool output once
            tool_embedding = self.model.encode(tool_output, convert_to_tensor=True)

            claim_scores = []
            for claim in claims:
                claim_embedding = self.model.encode(claim, convert_to_tensor=True)
                score = util.cos_sim(claim_embedding, tool_embedding).item()
                claim_scores.append({"claim": claim, "similarity": round(score, 4)})

            # A claim matches if its similarity exceeds the threshold
            matched = [c for c in claim_scores if c["similarity"] >= self.threshold]
            mismatched = [c for c in claim_scores if c["similarity"] < self.threshold]

            tool_claim_match = len(mismatched) == 0
            avg_similarity = sum(c["similarity"] for c in claim_scores) / len(claim_scores)

            # Hallucination detected when at least one claim doesn't match
            hallucination_detected = not tool_claim_match

            # Confidence = proportion of mismatched claims weighted by severity
            if hallucination_detected:
                confidence = round(len(mismatched) / len(claim_scores), 2)
            else:
                confidence = round(avg_similarity, 2)

            # Severity by mismatch ratio
            mismatch_ratio = len(mismatched) / len(claim_scores)
            severity = self._compute_severity(mismatch_ratio, hallucination_detected)

            return {
                "step": step_id,
                "hallucination_detected": hallucination_detected,
                "confidence": confidence,
                "detection_signals": {
                    "semantic_similarity": None,
                    "tool_claim_match": tool_claim_match,
                    "nli_score": None,
                    "contradiction_with_prev": None,
                },
                "hallucination_type": config.TYPE_TOOL_USE if hallucination_detected else None,
                "severity": severity,
                "detail": {
                    "total_claims": len(claim_scores),
                    "matched_claims": len(matched),
                    "mismatched_claims": len(mismatched),
                    "per_claim_scores": claim_scores,
                },
            }

        except Exception as e:
            return self._fallback(step_id, f"Inference error: {str(e)}")

    # ── internal helpers ────────────────────────────────────────────────────

    def _extract_claims(self, text: str) -> list:
        """
        Splits reasoning text into individual claims (sentences / clauses).

        Args:
            text (str): The agent reasoning text.

        Returns:
            list: A list of claim strings.
        """
        # Split on sentence-ending punctuation and commas (common clause separator)
        raw = re.split(r'[.;,!?\n]+', text)
        claims = [c.strip() for c in raw if len(c.strip()) > 3]
        return claims[: self.max_claims]

    def _compute_severity(self, mismatch_ratio: float, detected: bool) -> str | None:
        """
        Determines severity based on what fraction of claims are mismatched.

        Args:
            mismatch_ratio (float): Fraction of mismatched claims (0-1).
            detected (bool): Whether a hallucination was detected.

        Returns:
            str | None: Severity level or None if no hallucination.
        """
        if not detected:
            return None
        if mismatch_ratio > 0.6:
            return config.SEVERITY_HIGH
        if mismatch_ratio > 0.3:
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
    validator = ToolValidator()

    sample_input = {
        "step": 2,
        "action": "web_search",
        "tool_input": "FIFA 2022 winner",
        "tool_output": "Argentina won FIFA World Cup 2022",
        "agent_reasoning": "France won FIFA 2022, capital is Paris",
        "ground_truth": "Argentina won FIFA 2022",
    }

    result = validator.validate(sample_input)
    print(json.dumps(result, indent=2))
