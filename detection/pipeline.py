"""
AgentTrace — Real Detection Pipeline (v2)
==========================================
Chains all 4 detection modules (semantic, tool, NLI, contradiction)
into a single detector function that can replace mock_detector
in the benchmark runner.

v2 Changes:
- Replaced OR-fusion with threshold-only fusion (fixes FPR=20%)
- Multi-signal type classifier covering all 5 categories (fixes 0% Planning/Human-Interaction)
- Action-aware type routing (fixes Tool-Use over-prediction)
- No data leakage: never reads ground_truth_label or hallucination_type

Author: P. Somnath Reddy (Research Lead)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG, LOCALIZATION_SIGNAL_WEIGHTS


class DetectionPipeline:
    """
    Orchestrates all detection modules and produces a unified
    detection result per step.

    Loads models once at init, then runs all detectors on each step.
    Fuses signals using weighted scoring — no OR logic.
    """

    # Action categories for type inference
    TOOL_ACTIONS = {"web_search", "api_call", "database_query", "calculator",
                    "code_exec", "file_read", "search", "lookup"}
    PLANNING_ACTIONS = {"plan", "decompose", "schedule", "prioritize",
                        "strategize", "outline", "task_decomposition"}
    HUMAN_ACTIONS = {"ask_user", "human_feedback", "clarify", "confirm",
                     "user_input", "human_input", "ask_human"}

    # Fusion threshold — only flag if fused score exceeds this
    FUSION_THRESHOLD = 0.60

    # Minimum number of active signals required to make a decision
    MIN_SIGNALS_FOR_DETECTION = 2

    def __init__(self):
        """Initializes all detection sub-modules."""
        print("Loading detection pipeline v2...")

        # Semantic checker (sentence-transformers)
        self.semantic_checker = None
        try:
            from detection.semantic_checker import SemanticChecker
            self.semantic_checker = SemanticChecker()
            print("  ✓ SemanticChecker loaded")
        except Exception as e:
            print(f"  ✗ SemanticChecker failed: {e}")

        # Tool validator (sentence-transformers)
        self.tool_validator = None
        try:
            from detection.tool_validator import ToolValidator
            self.tool_validator = ToolValidator()
            print("  ✓ ToolValidator loaded")
        except Exception as e:
            print(f"  ✗ ToolValidator failed: {e}")

        # NLI-based modules
        self.factual_grounder = None
        self.contradiction_detector = None
        self._load_nli_modules()

        self.weights = LOCALIZATION_SIGNAL_WEIGHTS
        self._history = []  # tracks previous steps for contradiction detection
        print("Detection pipeline v2 ready.\n")

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

        IMPORTANT: This method NEVER reads ground_truth_label or
        hallucination_type from the step. It only uses observable
        fields: action, tool_input, tool_output, agent_reasoning.

        Args:
            step: A single step dict from a trajectory.

        Returns:
            dict: Detection result with hallucination_detected, confidence,
                  hallucination_type_predicted, severity.
        """
        # Build the step_data format our detectors expect
        # ONLY uses observable fields — no ground truth leakage
        step_data = self._normalize_step(step)

        # ── Run all 4 detectors ─────────────────────────────────────────
        signals = {}
        detector_results = {}

        # 1. Semantic similarity (reasoning vs tool_output)
        if self.semantic_checker:
            sem_result = self.semantic_checker.check(step_data)
            signals["semantic_similarity"] = sem_result.get(
                "detection_signals", {}
            ).get("semantic_similarity")
            detector_results["semantic"] = sem_result
        else:
            detector_results["semantic"] = {}

        # 2. Tool claim validation
        if self.tool_validator:
            tool_result = self.tool_validator.validate(step_data)
            signals["tool_claim_match"] = tool_result.get(
                "detection_signals", {}
            ).get("tool_claim_match")
            detector_results["tool"] = tool_result
        else:
            detector_results["tool"] = {}

        # 3. NLI factual grounding
        if self.factual_grounder:
            nli_result = self.factual_grounder.ground(step_data)
            signals["nli_score"] = nli_result.get(
                "detection_signals", {}
            ).get("nli_score")
            detector_results["nli"] = nli_result
        else:
            detector_results["nli"] = {}
            signals["nli_score"] = None

        # 4. Contradiction with previous steps
        if self.contradiction_detector and self._history:
            contra_result = self.contradiction_detector.detect(
                step_data, self._history
            )
            signals["contradiction_with_prev"] = contra_result.get(
                "detection_signals", {}
            ).get("contradiction_with_prev")
            detector_results["contradiction"] = contra_result
        else:
            detector_results["contradiction"] = {}
            signals["contradiction_with_prev"] = None

        # Track history for contradiction detection
        self._history.append(step_data)

        # ── Fuse signals (THRESHOLD ONLY — no OR logic) ─────────────────
        fused_score = self._fuse_signals(signals)
        active_signal_count = sum(
            1 for v in signals.values() if v is not None
        )

        # STRICT threshold decision — no OR fallback
        hallucination_detected = (
            fused_score > self.FUSION_THRESHOLD
            and active_signal_count >= self.MIN_SIGNALS_FOR_DETECTION
        )

        # Confidence = fused score
        confidence = round(min(max(fused_score, 0.0), 1.0), 4)

        # Determine type using multi-signal classification
        hallucination_type = None
        if hallucination_detected:
            hallucination_type = self._classify_type(
                step_data, signals, detector_results
            )

        # Severity
        severity = self._determine_severity(confidence, hallucination_detected)

        return {
            "hallucination_detected": hallucination_detected,
            "confidence": confidence,
            "hallucination_type_predicted": hallucination_type,
            "severity": severity,
        }

    def reset_history(self):
        """Clears step history. Call between trajectories."""
        self._history = []

    def _normalize_step(self, step: dict) -> dict:
        """
        Converts benchmark trajectory step format to detector input format.
        ONLY uses observable fields — never reads ground_truth_label.

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
        Weighted fusion of detection signals.

        Converts each signal to a "hallucination risk" score [0,1] then
        computes a weighted average. Only uses signals that are available.

        Args:
            signals: Dict with semantic_similarity, tool_claim_match,
                     nli_score, contradiction_with_prev.

        Returns:
            float: Fused hallucination risk score [0, 1].
        """
        risk_scores = {}
        total_weight = 0.0

        # Semantic: low similarity = high risk
        sem = signals.get("semantic_similarity")
        if sem is not None:
            risk_scores["semantic_similarity"] = 1.0 - sem
            total_weight += self.weights["semantic_similarity"]

        # Tool: no match = high risk
        tcm = signals.get("tool_claim_match")
        if tcm is not None:
            risk_scores["tool_claim_match"] = 0.0 if tcm else 1.0
            total_weight += self.weights["tool_claim_match"]

        # NLI: high contradiction score = high risk
        nli = signals.get("nli_score")
        if nli is not None:
            risk_scores["nli_score"] = nli
            total_weight += self.weights["nli_score"]

        # Contradiction: contradiction found = high risk
        contra = signals.get("contradiction_with_prev")
        if contra is not None:
            risk_scores["contradiction_with_prev"] = 1.0 if contra else 0.0
            total_weight += self.weights["contradiction_with_prev"]

        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            risk_scores[k] * self.weights[k] for k in risk_scores
        )
        return weighted_sum / total_weight

    def _classify_type(
        self, step_data: dict, signals: dict, results: dict
    ) -> str:
        """
        Multi-signal hallucination type classifier.
        Covers all 5 categories using action context + signal patterns.

        Classification logic (no ground truth leakage):
        1. Tool-Use: tool_claim_match=False AND action is a tool action
        2. Human-Interaction: action involves human/user input
        3. Planning: action involves planning/decomposition OR
                     multiple reasoning steps with contradictions
        4. Retrieval: high NLI contradiction score (factual mismatch)
        5. Reasoning: semantic drift without tool/retrieval issues

        Args:
            step_data: Normalized step data.
            signals: Raw signal values.
            results: Full detector result dicts.

        Returns:
            str: One of the 5 hallucination categories.
        """
        action = step_data.get("action", "").lower().strip()
        reasoning = step_data.get("agent_reasoning", "").lower()

        tcm = signals.get("tool_claim_match")
        nli = signals.get("nli_score")
        contra = signals.get("contradiction_with_prev")
        sem = signals.get("semantic_similarity")

        # ── Rule 1: Human-Interaction ────────────────────────────────────
        # If the action involves human/user interaction
        if action in self.HUMAN_ACTIONS or any(
            kw in action for kw in ["user", "human", "ask", "clarif"]
        ):
            return "Human-Interaction"

        # ── Rule 2: Planning ─────────────────────────────────────────────
        # If the action involves planning, OR the reasoning mentions
        # planning-related keywords with contradictions
        if action in self.PLANNING_ACTIONS or any(
            kw in action for kw in ["plan", "decompos", "schedul", "strateg"]
        ):
            return "Planning"

        # Planning from reasoning context: multi-step reasoning with
        # contradictions suggests a planning failure
        planning_keywords = ["first", "then", "next", "step", "plan",
                             "approach", "strategy", "break down", "subtask"]
        planning_score = sum(1 for kw in planning_keywords if kw in reasoning)
        if planning_score >= 3 and contra:
            return "Planning"

        # ── Rule 3: Tool-Use ─────────────────────────────────────────────
        # Tool claim mismatch AND action is a tool action
        if tcm is False and (
            action in self.TOOL_ACTIONS
            or any(kw in action for kw in ["search", "calc", "api", "query",
                                           "look", "fetch", "read"])
        ):
            return "Tool-Use"

        # ── Rule 4: Retrieval ────────────────────────────────────────────
        # High NLI contradiction = factual mismatch from retrieval
        if nli is not None and nli > 0.6:
            return "Retrieval"

        # Semantic drift without tool issues also suggests retrieval error
        if sem is not None and sem < 0.5 and tcm is not False:
            return "Retrieval"

        # ── Rule 5: Reasoning (default) ──────────────────────────────────
        # Contradictions between steps = reasoning failure
        if contra:
            return "Reasoning"

        # General semantic drift = reasoning error
        return "Reasoning"

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
    GUARANTEED: never reads ground_truth_label or hallucination_type.

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

    # Test 1: Tool-Use hallucination (action=web_search, claim mismatch)
    test_tool = {
        "step": 1,
        "action": "web_search",
        "tool_input": "FIFA 2022 winner",
        "tool_output": "Argentina won FIFA World Cup 2022",
        "agent_reasoning": "France won FIFA 2022, capital is Paris",
    }

    # Test 2: Clean step (no hallucination)
    test_clean = {
        "step": 2,
        "action": "calculator",
        "tool_input": "80 - 65",
        "tool_output": "15",
        "agent_reasoning": "The calculator returns 15, so the difference is 15 years.",
    }

    for name, step in [("Tool-Use halluc", test_tool), ("Clean step", test_clean)]:
        result = pipeline.detect(step)
        print(f"\n{name}:")
        print(json.dumps(result, indent=2))
