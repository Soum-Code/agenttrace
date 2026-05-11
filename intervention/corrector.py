"""
Intervention Module: Corrector.
Given a hallucinated step, its causal classification, and the original
trace, proposes and applies a correction strategy.
"""

import json
import copy
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Corrector:
    """
    Proposes a correction for a hallucinated step based on the causal label.
    Strategies are applied in the order defined in config.CORRECTOR_STRATEGY_ORDER.
    """

    def __init__(self):
        """Loads correction parameters from config."""
        self.max_retries = config.CORRECTOR_MAX_RETRIES
        self.strategy_order = config.CORRECTOR_STRATEGY_ORDER

    def correct(self, step_data: dict, causal_result: dict) -> dict:
        """
        Proposes a correction for a single hallucinated step.

        Args:
            step_data: Original agent step JSON.
            causal_result: Output from CausalClassifier.classify().

        Returns:
            dict: Correction proposal with strategy, corrected reasoning,
                  and retry metadata.
        """
        step_id = step_data.get("step")
        causal_label = causal_result.get("causal_label")

        if causal_label is None:
            return {
                "step": step_id,
                "correction_applied": False,
                "message": "No causal label — nothing to correct.",
            }

        try:
            strategy = self._select_strategy(causal_label)
            corrected = self._apply_strategy(strategy, step_data, causal_label)

            return {
                "step": step_id,
                "correction_applied": True,
                "strategy": strategy,
                "causal_label": causal_label,
                "original_reasoning": step_data.get("agent_reasoning", ""),
                "corrected_reasoning": corrected,
                "max_retries": self.max_retries,
            }
        except Exception as e:
            return {
                "step": step_id,
                "correction_applied": False,
                "error": f"Correction error: {str(e)}",
            }

    def _select_strategy(self, causal_label: str) -> str:
        """
        Selects the best correction strategy based on causal label.

        Args:
            causal_label: Root-cause label from CausalClassifier.

        Returns:
            str: Strategy name.
        """
        # Map causal labels to preferred strategies
        label_strategy_map = {
            "Retrieval-Error": "tool_requery",
            "Tool-Misuse": "tool_requery",
            "Reasoning-Error": "reasoning_override",
            "Instruction-Drift": "step_rollback",
            "Context-Overflow": "step_rollback",
        }
        preferred = label_strategy_map.get(causal_label)
        if preferred and preferred in self.strategy_order:
            return preferred
        # Default to first available strategy
        return self.strategy_order[0]

    def _apply_strategy(self, strategy: str, step_data: dict, causal_label: str) -> str:
        """
        Applies the selected correction strategy to produce corrected reasoning.

        Args:
            strategy: The strategy name to apply.
            step_data: Original step data.
            causal_label: Root cause label.

        Returns:
            str: The corrected reasoning text.
        """
        tool_output = step_data.get("tool_output", "")
        ground_truth = step_data.get("ground_truth", "")
        original = step_data.get("agent_reasoning", "")

        if strategy == "tool_requery":
            # Replace reasoning with tool output (the authoritative source)
            return f"[CORRECTED via tool_requery] {tool_output}"

        elif strategy == "reasoning_override":
            # Override with ground truth if available, else tool output
            source = ground_truth if ground_truth else tool_output
            return f"[CORRECTED via reasoning_override] {source}"

        elif strategy == "step_rollback":
            # Flag for rollback — upstream orchestrator should re-execute
            return f"[ROLLBACK REQUESTED] Original: {original} | Cause: {causal_label}"

        return f"[NO STRATEGY MATCHED] {original}"

    def correct_trace(self, trace: list, detection_results: list, causal_results: list) -> list:
        """
        Applies corrections to an entire agent trace.

        Args:
            trace: List of step_data dicts (the original trace).
            detection_results: List of merged detection results per step.
            causal_results: List of CausalClassifier outputs per step.

        Returns:
            list: Corrected trace with annotations.
        """
        corrected_trace = []
        for step_data, det, causal in zip(trace, detection_results, causal_results):
            if det.get("hallucination_detected", False):
                correction = self.correct(step_data, causal)
                patched = copy.deepcopy(step_data)
                if correction.get("correction_applied"):
                    patched["agent_reasoning"] = correction["corrected_reasoning"]
                    patched["_correction_meta"] = correction
                corrected_trace.append(patched)
            else:
                corrected_trace.append(step_data)
        return corrected_trace


if __name__ == "__main__":
    corrector = Corrector()

    step = {
        "step": 2, "action": "web_search",
        "tool_output": "Argentina won FIFA World Cup 2022",
        "agent_reasoning": "France won FIFA 2022, capital is Paris",
        "ground_truth": "Argentina won FIFA 2022",
    }
    causal = {"causal_label": "Tool-Misuse", "causal_confidence": 0.7}

    result = corrector.correct(step, causal)
    print(json.dumps(result, indent=2))
