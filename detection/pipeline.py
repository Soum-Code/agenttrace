"""
AgentTrace — Real Detection Pipeline
=====================================
Chains all 4 detection modules (semantic, tool, NLI, contradiction)
into a single detector function that can replace mock_detector
in the benchmark runner.

This is the multi-signal fusion approach designed to beat
AgentHallu's 41.1% step localization accuracy.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG, LOCALIZATION_SIGNAL_WEIGHTS
from detection.semantic_checker import SemanticChecker
from detection.tool_validator import ToolValidator


class DetectionPipeline:
    """
    Orchestrates all detection modules and produces a unified
    detection result per step.

    Loads models once at init, then runs all detectors on each step.
    Fuses signals using weighted scoring from config.
    """

    def __init__(self):
        """Initializes all detection sub-modules."""
        print("Loading detection pipeline models...")

        # Semantic checker (sentence-transformers)
        self.semantic_checker = SemanticChecker()
        print("  ✓ SemanticChecker loaded")

        # Tool validator (sentence-transformers, shared model possible)
        self.tool_validator = ToolValidator()
        print("  ✓ ToolValidator loaded")

        # NLI-based modules — lazy load to save memory if not needed
        self.factual_grounder = None
        self.contradiction_detector = None
        self._load_nli_modules()

        self.weights = LOCALIZATION_SIGNAL_WEIGHTS
        self._history = []  # tracks previous steps for contradiction detection
        print("Detection pipeline ready.\n")

    def _load_nli_modules(self):
        """Lazy-loads NLI-based modules with error handling."""
        try:
            from detection.factual_grounding import FactualGrounder
            self.factual_grounder = FactualGrounder()
            print("  ✓ FactualGrounder loaded")
        except Exception as e:
            print(f"  ✗ FactualGrounder failed: {e}")

        try:
            from detection.contradiction import ContradictionDetector
            self.contradiction_detector = ContradictionDetector()
            print("  ✓ ContradictionDetector loaded")
        except Exception as e:
            print(f"  ✗ ContradictionDetector failed: {e}")

    def detect(self, step: dict) -> dict:
        """
        Runs the full multi-signal detection pipeline on a single step.

        This function matches the signature expected by BenchmarkRunner:
        takes a step dict, returns a detection_result dict.

        Args:
            step: A single step dict from a trajectory. Expected keys:
                  step, action, tool_input, tool_output, agent_reasoning,
                  ground_truth_label, hallucination_type.

        Returns:
            dict: Detection result with hallucination_detected, confidence,
                  hallucination_type_predicted, severity.
        """
        # Build the step_data format our detectors expect
        step_data = self._normalize_step(step)

        # ── Run all 4 detectors ─────────────────────────────────────────
        signals = {}

        # 1. Semantic similarity
        sem_result = self.semantic_checker.check(step_data)
        signals["semantic_similarity"] = sem_result.get("detection_signals", {}).get(
            "semantic_similarity"
        )

        # 2. Tool claim validation
        tool_result = self.tool_validator.validate(step_data)
        signals["tool_claim_match"] = tool_result.get("detection_signals", {}).get(
            "tool_claim_match"
        )

        # 3. NLI factual grounding
        if self.factual_grounder:
            nli_result = self.factual_grounder.ground(step_data)
            signals["nli_score"] = nli_result.get("detection_signals", {}).get(
                "nli_score"
            )
        else:
            nli_result = {}
            signals["nli_score"] = None

        # 4. Contradiction with previous steps
        if self.contradiction_detector and self._history:
            contra_result = self.contradiction_detector.detect(
                step_data, self._history
            )
            signals["contradiction_with_prev"] = contra_result.get(
                "detection_signals", {}
            ).get("contradiction_with_prev")
        else:
            contra_result = {}
            signals["contradiction_with_prev"] = None

        # Track history for contradiction detection
        self._history.append(step_data)

        # ── Fuse signals ────────────────────────────────────────────────
        fused_score = self._fuse_signals(signals)

        # Decision: hallucination if fused score exceeds threshold
        # or if any individual detector strongly flags it
        hallucination_detected = (
            fused_score > 0.5
            or sem_result.get("hallucination_detected", False)
            or tool_result.get("hallucination_detected", False)
            or nli_result.get("hallucination_detected", False)
            or contra_result.get("hallucination_detected", False)
        )

        # Confidence = fused score (bounded)
        confidence = round(min(max(fused_score, 0.0), 1.0), 4)

        # Determine type from the strongest signal
        hallucination_type = self._determine_type(
            sem_result, tool_result, nli_result, contra_result
        )

        # Severity from confidence
        severity = self._determine_severity(confidence, hallucination_detected)

        return {
            "hallucination_detected": hallucination_detected,
            "confidence": confidence,
            "hallucination_type_predicted": hallucination_type if hallucination_detected else None,
            "severity": severity,
            "fused_score": fused_score,
            "signals": signals,
        }

    def reset_history(self):
        """Clears step history. Call between trajectories."""
        self._history = []

    def _normalize_step(self, step: dict) -> dict:
        """
        Converts benchmark trajectory step format to detector input format.

        Args:
            step: Raw trajectory step dict.

        Returns:
            dict: Normalized step_data for detectors.
        """
        return {
            "step": step.get("step"),
            "action": step.get("action", ""),
            "tool_input": step.get("tool_input", ""),
            "tool_output": step.get("tool_output", ""),
            "agent_reasoning": step.get("agent_reasoning", ""),
            # Use tool_output as ground_truth proxy (detector compares
            # reasoning against what the tool actually returned)
            "ground_truth": step.get("tool_output", ""),
        }

    def _fuse_signals(self, signals: dict) -> float:
        """
        Weighted fusion of detection signals, same logic as Localizer.

        Args:
            signals: Dict with semantic_similarity, tool_claim_match,
                     nli_score, contradiction_with_prev.

        Returns:
            float: Fused hallucination score [0, 1].
        """
        values = {}
        total_weight = 0.0

        sem = signals.get("semantic_similarity")
        if sem is not None:
            values["semantic_similarity"] = 1.0 - sem
            total_weight += self.weights["semantic_similarity"]

        tcm = signals.get("tool_claim_match")
        if tcm is not None:
            values["tool_claim_match"] = 0.0 if tcm else 1.0
            total_weight += self.weights["tool_claim_match"]

        nli = signals.get("nli_score")
        if nli is not None:
            values["nli_score"] = nli
            total_weight += self.weights["nli_score"]

        contra = signals.get("contradiction_with_prev")
        if contra is not None:
            values["contradiction_with_prev"] = 1.0 if contra else 0.0
            total_weight += self.weights["contradiction_with_prev"]

        if total_weight == 0:
            return 0.0

        return sum(values[k] * self.weights[k] for k in values) / total_weight

    def _determine_type(self, sem, tool, nli, contra) -> str:
        """
        Picks the hallucination type from the strongest detector signal.

        Args:
            sem: SemanticChecker result.
            tool: ToolValidator result.
            nli: FactualGrounder result.
            contra: ContradictionDetector result.

        Returns:
            str: Hallucination type label.
        """
        # Priority: Tool-Use > Contradiction > Factual > Reasoning
        if tool.get("hallucination_detected"):
            return "Tool-Use"
        if contra.get("hallucination_detected"):
            return "Reasoning"
        if nli.get("hallucination_detected"):
            return "Retrieval"
        if sem.get("hallucination_detected"):
            return "Reasoning"
        return "Reasoning"  # default

    def _determine_severity(self, confidence: float, detected: bool) -> str:
        """
        Maps confidence to severity level using config thresholds.

        Args:
            confidence: Fused confidence score.
            detected: Whether hallucination was flagged.

        Returns:
            str or None: Severity label.
        """
        if not detected:
            return None
        thresholds = CONFIG.thresholds.severity_thresholds
        if confidence >= thresholds["High"]:
            return "High"
        if confidence >= thresholds["Medium"]:
            return "Medium"
        return "Low"


# ── Singleton wrapper for benchmark compatibility ───────────────────────

_pipeline_instance = None


def real_detector(step: dict) -> dict:
    """
    Drop-in replacement for mock_detector in BenchmarkRunner.

    Initializes the pipeline on first call, then reuses it.
    Call real_detector.reset() between trajectories.

    Args:
        step: A single step dict from a trajectory.

    Returns:
        dict: Detection result matching benchmark schema.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = DetectionPipeline()
    return _pipeline_instance.detect(step)


def reset_pipeline():
    """Resets step history between trajectories."""
    global _pipeline_instance
    if _pipeline_instance is not None:
        _pipeline_instance.reset_history()


# ── Independent test ────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    pipeline = DetectionPipeline()

    test_step = {
        "step": 2,
        "action": "web_search",
        "tool_input": "FIFA 2022 winner",
        "tool_output": "Argentina won FIFA World Cup 2022",
        "agent_reasoning": "France won FIFA 2022, capital is Paris",
        "ground_truth_label": True,
        "hallucination_type": "Tool-Use",
    }

    result = pipeline.detect(test_step)
    print(json.dumps(result, indent=2))
