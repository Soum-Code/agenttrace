import json
import sys
import os

# Add project root to sys.path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    # This allows the file to be parsed even if dependencies are missing, 
    # though it will fail during initialization.
    pass

import config

class SemanticChecker:
    """
    Detection Module: Semantic Checker.
    Calculates the semantic similarity between agent reasoning and ground truth/tool output
     to identify potential hallucinations.
    """

    def __init__(self):
        """
        Initializes the SentenceTransformer model using the name provided in config.
        """
        try:
            self.model = SentenceTransformer(config.SEMANTIC_MODEL_NAME)
            self.threshold = config.SEMANTIC_SIMILARITY_THRESHOLD
        except Exception as e:
            print(f"Error loading semantic model: {e}")
            self.model = None

    def check(self, step_data: dict) -> dict:
        """
        Performs semantic similarity check on a single agent step.

        Args:
            step_data (dict): The structured JSON for a single agent step.

        Returns:
            dict: The detection result containing hallucination status and confidence.
        """
        step_id = step_data.get("step")
        agent_reasoning = step_data.get("agent_reasoning", "")
        ground_truth = step_data.get("ground_truth", "")
        
        if not self.model:
            return self._generate_fallback(step_id, "Model not initialized")

        try:
            # Encode the reasoning and ground truth
            embeddings = self.model.encode([agent_reasoning, ground_truth], convert_to_tensor=True)
            
            # Compute cosine similarity
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            
            # Hallucination detection logic
            # If similarity is below threshold, we flag it as a hallucination
            hallucination_detected = similarity < self.threshold
            
            # Confidence estimation: 
            # If detected, confidence is higher as similarity decreases.
            # If not detected, confidence is higher as similarity increases.
            if hallucination_detected:
                confidence = 1.0 - similarity
            else:
                confidence = similarity

            # Determine severity based on how far below the threshold the similarity is
            severity = None
            if hallucination_detected:
                if similarity < config.SEVERITY_HIGH_UPPER:
                    severity = config.SEVERITY_HIGH
                elif similarity < config.SEVERITY_MEDIUM_UPPER:
                    severity = config.SEVERITY_MEDIUM
                else:
                    severity = config.SEVERITY_LOW

            return {
                "step": step_id,
                "hallucination_detected": hallucination_detected,
                "confidence": round(confidence, 2),
                "detection_signals": {
                    "semantic_similarity": round(similarity, 2),
                    "tool_claim_match": None,  # To be filled by tool_validator
                    "nli_score": None,         # To be filled by factual_grounding
                    "contradiction_with_prev": None # To be filled by contradiction module
                },
                "hallucination_type": config.TYPE_FACTUAL if hallucination_detected else None,
                "severity": severity
            }

        except Exception as e:
            return self._generate_fallback(step_id, f"Inference error: {str(e)}")

    def _generate_fallback(self, step_id: int, error_msg: str) -> dict:
        """
        Generates a safe fallback response in case of errors.
        """
        return {
            "step": step_id,
            "hallucination_detected": False,
            "confidence": 0.0,
            "error": error_msg,
            "detection_signals": {
                "semantic_similarity": 0.0,
                "tool_claim_match": None,
                "nli_score": None,
                "contradiction_with_prev": None
            },
            "hallucination_type": None,
            "severity": None
        }

if __name__ == "__main__":
    # Independent Test Case
    checker = SemanticChecker()
    
    sample_input = {
      "step": 2,
      "action": "web_search",
      "tool_input": "FIFA 2022 winner",
      "tool_output": "Argentina won FIFA World Cup 2022",
      "agent_reasoning": "France won FIFA 2022, capital is Paris",
      "ground_truth": "Argentina won FIFA 2022"
    }
    
    result = checker.check(sample_input)
    print(json.dumps(result, indent=2))
