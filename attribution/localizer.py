"""
Attribution Module: Localizer.
Aggregates detection signals from all four detectors (semantic, tool,
NLI, contradiction) into a single weighted hallucination score per step,
then ranks steps to identify the most likely origin of hallucination.
"""

import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Localizer:
    """
    Fuses multi-signal detection results and produces a per-step
    hallucination risk score.  The step with the highest fused score
    is flagged as the localized origin.
    """

    def __init__(self):
        """Loads signal weights from config."""
        self.weights = config.LOCALIZATION_SIGNAL_WEIGHTS

    def localize(self, detection_results: list) -> dict:
        """
        Aggregates detection signals across steps and localizes the
        hallucination origin.

        Args:
            detection_results: List of merged detection dicts, one per step.
                Each dict must have a 'detection_signals' sub-dict.

        Returns:
            dict: Localization report with per-step scores and the
                  identified origin step.
        """
        try:
            scored_steps = []
            for result in detection_results:
                step_id = result.get("step")
                signals = result.get("detection_signals", {})
                fused = self._fuse_signals(signals)
                scored_steps.append({
                    "step": step_id,
                    "fused_score": round(fused, 4),
                    "signals_used": {k: v for k, v in signals.items() if v is not None},
                })

            # Sort descending by fused score
            scored_steps.sort(key=lambda s: s["fused_score"], reverse=True)

            origin_step = scored_steps[0] if scored_steps else None

            return {
                "localized_step": origin_step["step"] if origin_step else None,
                "origin_fused_score": origin_step["fused_score"] if origin_step else 0.0,
                "per_step_scores": scored_steps,
            }
        except Exception as e:
            return {"error": f"Localization error: {str(e)}",
                    "localized_step": None, "origin_fused_score": 0.0, "per_step_scores": []}

    def _fuse_signals(self, signals: dict) -> float:
        """
        Computes a weighted sum of detection signals.

        Signals are normalized so that higher = more hallucinated:
          - semantic_similarity → inverted (1 - sim)
          - tool_claim_match   → inverted (1 if False, 0 if True)
          - nli_score           → used directly (contradiction prob)
          - contradiction_with_prev → 1 if True, 0 if False

        Args:
            signals: Detection signals dict from a merged step result.

        Returns:
            float: Fused hallucination score in [0, 1].
        """
        values = {}
        total_weight = 0.0

        # Semantic similarity (invert: low sim → high risk)
        sem = signals.get("semantic_similarity")
        if sem is not None:
            values["semantic_similarity"] = 1.0 - sem
            total_weight += self.weights["semantic_similarity"]

        # Tool claim match (invert: False → 1.0 risk)
        tcm = signals.get("tool_claim_match")
        if tcm is not None:
            values["tool_claim_match"] = 0.0 if tcm else 1.0
            total_weight += self.weights["tool_claim_match"]

        # NLI score (contradiction prob — already higher = worse)
        nli = signals.get("nli_score")
        if nli is not None:
            values["nli_score"] = nli
            total_weight += self.weights["nli_score"]

        # Contradiction with previous (boolean → 0/1)
        contra = signals.get("contradiction_with_prev")
        if contra is not None:
            values["contradiction_with_prev"] = 1.0 if contra else 0.0
            total_weight += self.weights["contradiction_with_prev"]

        if total_weight == 0:
            return 0.0

        fused = sum(values[k] * self.weights[k] for k in values) / total_weight
        return fused


if __name__ == "__main__":
    localizer = Localizer()

    merged_results = [
        {
            "step": 1,
            "detection_signals": {
                "semantic_similarity": 0.92,
                "tool_claim_match": True,
                "nli_score": 0.05,
                "contradiction_with_prev": False,
            },
        },
        {
            "step": 2,
            "detection_signals": {
                "semantic_similarity": 0.21,
                "tool_claim_match": False,
                "nli_score": 0.89,
                "contradiction_with_prev": True,
            },
        },
    ]

    report = localizer.localize(merged_results)
    print(json.dumps(report, indent=2))
