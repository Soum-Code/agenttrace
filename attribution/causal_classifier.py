"""
Attribution Module: Causal Classifier.
Classifies the root cause of a detected hallucination into one of the
causal categories defined in config (e.g. Retrieval-Error, Reasoning-Error).
Uses a DistilBERT sequence classifier.
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


class CausalClassifier:
    """
    Given a step flagged as hallucinated, classifies the likely root cause
    using a DistilBERT-based sequence classifier.

    In production this model is fine-tuned on labelled hallucination data.
    For the initial build we use zero-shot heuristics over the logits of
    the base model as a bootstrap, and provide the interface for swapping
    in a fine-tuned checkpoint.
    """

    def __init__(self):
        """Loads the causal classification model from config."""
        self.labels = config.CAUSAL_LABELS
        self.confidence_threshold = config.CAUSAL_CONFIDENCE_THRESHOLD
        
        model_path = config.CONFIG.classifier.finetuned_model_path
        if not os.path.exists(model_path):
            print(f"Local model not found at {model_path}. Falling back to base model.")
            model_path = config.CAUSAL_MODEL_NAME

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=len(self.labels),
                ignore_mismatched_sizes=True,
            )
            self.model.eval()
        except Exception as e:
            print(f"Error loading causal classifier: {e}")
            self.model = None
            self.tokenizer = None

    def classify(self, step_data: dict, detection_result: dict) -> dict:
        """
        Classifies the causal root of a hallucination.

        Args:
            step_data: Original agent step JSON.
            detection_result: Merged detection output for this step.

        Returns:
            dict: Causal classification with label, confidence, and all
                  label probabilities.
        """
        step_id = step_data.get("step")

        if not detection_result.get("hallucination_detected", False):
            return {
                "step": step_id,
                "causal_label": None,
                "causal_confidence": 0.0,
                "message": "No hallucination detected — skipping classification",
            }

        if not self.model or not self.tokenizer:
            return self._fallback_heuristic(step_data, detection_result)

        try:
            # Build a feature string combining all available context
            feature_text = self._build_feature_text(step_data, detection_result)

            inputs = self.tokenizer(
                feature_text, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()

            label_probs = {self.labels[i]: round(probs[i], 4) for i in range(len(self.labels))}
            best_idx = probs.index(max(probs))
            best_label = self.labels[best_idx]
            best_conf = round(probs[best_idx], 4)

            # Fall back to heuristic if model confidence is too low
            if best_conf < self.confidence_threshold:
                return self._fallback_heuristic(step_data, detection_result)

            return {
                "step": step_id,
                "causal_label": best_label,
                "causal_confidence": best_conf,
                "all_label_probs": label_probs,
            }

        except Exception as e:
            return self._fallback_heuristic(step_data, detection_result)

    def _build_feature_text(self, step_data: dict, detection_result: dict) -> str:
        """
        Constructs a single text feature from step data and signals for the classifier.

        Args:
            step_data: Original step JSON.
            detection_result: Merged detection result.

        Returns:
            str: Concatenated feature text.
        """
        parts = [
            f"Action: {step_data.get('action', '')}",
            f"Reasoning: {step_data.get('agent_reasoning', '')}",
            f"Tool Output: {step_data.get('tool_output', '')}",
            f"Hallucination Type: {detection_result.get('hallucination_type', '')}",
            f"Severity: {detection_result.get('severity', '')}",
        ]
        return " | ".join(parts)

    def _fallback_heuristic(self, step_data: dict, detection_result: dict) -> dict:
        """
        Rule-based heuristic when model is unavailable or underconfident.
        Uses detection signals to infer the most likely causal label.

        Args:
            step_data: Original step JSON.
            detection_result: Merged detection result.

        Returns:
            dict: Heuristic causal classification.
        """
        step_id = step_data.get("step")
        signals = detection_result.get("detection_signals", {})
        h_type = detection_result.get("hallucination_type", "")

        # Heuristic rules — uses official taxonomy labels from config
        if signals.get("tool_claim_match") is False:
            label = config.TYPE_TOOL_USE      # "Tool-Use"
            conf = 0.7
        elif h_type == config.TYPE_CONTRADICTION:
            label = config.TYPE_REASONING     # "Reasoning"
            conf = 0.65
        elif signals.get("nli_score") and signals["nli_score"] > 0.8:
            label = config.TYPE_RETRIEVAL     # "Retrieval"
            conf = 0.6
        else:
            label = config.TYPE_REASONING     # "Reasoning"
            conf = 0.5

        return {
            "step": step_id,
            "causal_label": label,
            "causal_confidence": conf,
            "method": "heuristic_fallback",
        }


if __name__ == "__main__":
    classifier = CausalClassifier()

    step = {
        "step": 2, "action": "web_search",
        "tool_output": "Argentina won FIFA World Cup 2022",
        "agent_reasoning": "France won FIFA 2022, capital is Paris",
    }
    detection = {
        "hallucination_detected": True, "hallucination_type": "Tool-Use",
        "severity": "High",
        "detection_signals": {
            "semantic_similarity": 0.21, "tool_claim_match": False,
            "nli_score": 0.89, "contradiction_with_prev": True,
        },
    }

    result = classifier.classify(step, detection)
    print(json.dumps(result, indent=2))
