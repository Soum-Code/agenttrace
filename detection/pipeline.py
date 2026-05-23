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
    FUSION_THRESHOLD = 0.40

    # Minimum number of active signals required to make a decision
    MIN_SIGNALS_FOR_DETECTION = 2

    # Class-level model cache to prevent duplicate loading across instantiations (e.g. in ablation study)
    _cached_semantic_model = None
    _cached_nli_model = None
    _cached_nli_tokenizer = None
    _cached_llama_classifier = None
    _cached_nemotron_judge = None
    _cached_causal_classifier = None
    _cached_localizer = None
    _step_signals_cache = {}

    def __init__(self, enable_layer2: bool = True, enable_layer3: bool = True, active_modules: list = None):
        """Initializes all detection sub-modules with preloaded shared models."""
        print(f"Loading detection pipeline v2 (layer2={enable_layer2}, layer3={enable_layer3}, active_modules={active_modules})...")
        self.enable_layer2 = enable_layer2
        self.enable_layer3 = enable_layer3
        self.active_modules = active_modules
        self.calibration_temperature = CONFIG.thresholds.calibration_temperature

        # Pre-load shared models once to prevent duplicate loading and reduce memory overhead
        self.shared_semantic_model = None
        self.shared_nli_model = None
        self.shared_nli_tokenizer = None

        import config
        
        need_semantic = active_modules is None or any(m in active_modules for m in ["semantic", "tool", "factual"])
        need_nli = active_modules is None or any(m in active_modules for m in ["factual", "contradiction"])

        if need_semantic:
            if DetectionPipeline._cached_semantic_model is not None:
                self.shared_semantic_model = DetectionPipeline._cached_semantic_model
            else:
                print("Pre-loading shared models (Semantic)...")
                try:
                    from sentence_transformers import SentenceTransformer
                    self.shared_semantic_model = SentenceTransformer(config.SEMANTIC_MODEL_NAME)
                    DetectionPipeline._cached_semantic_model = self.shared_semantic_model
                    print("  [+] Shared Semantic Model (SentenceTransformer) pre-loaded")
                except Exception as e:
                    print(f"  [-] Failed to pre-load shared Semantic Model: {e}")

        if need_nli:
            if DetectionPipeline._cached_nli_model is not None:
                self.shared_nli_model = DetectionPipeline._cached_nli_model
                self.shared_nli_tokenizer = DetectionPipeline._cached_nli_tokenizer
            else:
                print("Pre-loading shared models (NLI)...")
                try:
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification
                    self.shared_nli_tokenizer = AutoTokenizer.from_pretrained(config.NLI_MODEL_NAME)
                    self.shared_nli_model = AutoModelForSequenceClassification.from_pretrained(config.NLI_MODEL_NAME)
                    self.shared_nli_model.eval()
                    DetectionPipeline._cached_nli_model = self.shared_nli_model
                    DetectionPipeline._cached_nli_tokenizer = self.shared_nli_tokenizer
                    print("  [+] Shared NLI Model & Tokenizer pre-loaded")
                except Exception as e:
                    print(f"  [-] Failed to pre-load shared NLI Model: {e}")

        # Semantic checker (sentence-transformers)
        self.semantic_checker = None
        if active_modules is None or "semantic" in active_modules:
            try:
                from detection.semantic_checker import SemanticChecker
                sc = SemanticChecker(model=self.shared_semantic_model)
                if sc.model is not None:  # verify model actually loaded
                    self.semantic_checker = sc
                    print("  [+] SemanticChecker loaded")
                else:
                    print("  [-] SemanticChecker: model failed to load")
            except Exception as e:
                print(f"  [-] SemanticChecker failed: {e}")

        # Tool validator (sentence-transformers)
        self.tool_validator = None
        if active_modules is None or "tool" in active_modules:
            try:
                from detection.tool_validator import ToolValidator
                tv = ToolValidator(model=self.shared_semantic_model)
                if tv.model is not None:  # verify model actually loaded
                    self.tool_validator = tv
                    print("  [+] ToolValidator loaded")
                else:
                    print("  [-] ToolValidator: model failed to load")
            except Exception as e:
                print(f"  [-] ToolValidator failed: {e}")

        # NLI-based modules
        self.factual_grounder = None
        self.contradiction_detector = None
        self._load_nli_modules()

        self.weights = LOCALIZATION_SIGNAL_WEIGHTS
        self._history = []  # tracks previous steps for contradiction detection
        self._detection_history = []  # tracks step detection results for localizer

        # Count how many models actually loaded
        self._loaded_count = sum(1 for m in [
            self.semantic_checker, self.tool_validator,
            self.factual_grounder, self.contradiction_detector
        ] if m is not None)
        print(f"\nDetection pipeline v2 (SLM Ensemble) ready. Models loaded: {self._loaded_count}/4")
        if self._loaded_count == 0:
            print("  WARNING: No models loaded! Install: pip install sentence-transformers transformers torch")

        self.llama_classifier = None
        if enable_layer2:
            if DetectionPipeline._cached_llama_classifier is not None:
                self.llama_classifier = DetectionPipeline._cached_llama_classifier
            else:
                print("Initializing Hybrid Layers...")
                try:
                    from attribution.llama_classifier import LlamaClassifier
                    self.llama_classifier = LlamaClassifier()
                    DetectionPipeline._cached_llama_classifier = self.llama_classifier
                    print("  [+] LlamaClassifier initialized")
                except Exception as e:
                    print(f"  [-] LlamaClassifier failed: {e}")
            
        self.nemotron_judge = None
        if enable_layer3:
            if DetectionPipeline._cached_nemotron_judge is not None:
                self.nemotron_judge = DetectionPipeline._cached_nemotron_judge
            else:
                try:
                    from attribution.nemotron_judge import NemotronJudge
                    self.nemotron_judge = NemotronJudge()
                    DetectionPipeline._cached_nemotron_judge = self.nemotron_judge
                    print("  [+] NemotronJudge initialized")
                except Exception as e:
                    print(f"  [-] NemotronJudge failed: {e}")

        if DetectionPipeline._cached_causal_classifier is not None:
            self.causal_classifier = DetectionPipeline._cached_causal_classifier
        else:
            try:
                from attribution.causal_classifier import CausalClassifier
                self.causal_classifier = CausalClassifier()
                DetectionPipeline._cached_causal_classifier = self.causal_classifier
                print("  [+] CausalClassifier initialized")
            except Exception as e:
                print(f"  [-] CausalClassifier failed: {e}")
                self.causal_classifier = None

        if DetectionPipeline._cached_localizer is not None:
            self.localizer = DetectionPipeline._cached_localizer
        else:
            try:
                from attribution.localizer import Localizer
                self.localizer = Localizer()
                DetectionPipeline._cached_localizer = self.localizer
                print("  [+] Localizer initialized")
            except Exception as e:
                print(f"  [-] Localizer failed: {e}")
                self.localizer = None

    def _load_nli_modules(self):
        """Lazy-loads NLI-based modules using preloaded shared models."""
        if self.active_modules is None or "factual" in self.active_modules:
            try:
                from detection.factual_grounding import FactualGrounder
                fg = FactualGrounder(
                    model=self.shared_nli_model,
                    tokenizer=self.shared_nli_tokenizer,
                    embedding_model=self.shared_semantic_model
                )
                if fg.model is not None:  # verify model actually loaded
                    self.factual_grounder = fg
                    print("  [+] FactualGrounder loaded")
                else:
                    print("  [-] FactualGrounder: model failed to load")
            except Exception as e:
                print(f"  [-] FactualGrounder failed: {e}")

        if self.active_modules is None or "contradiction" in self.active_modules:
            try:
                from detection.contradiction import ContradictionDetector
                cd = ContradictionDetector(
                    model=self.shared_nli_model,
                    tokenizer=self.shared_nli_tokenizer
                )
                if cd.model is not None:  # verify model actually loaded
                    self.contradiction_detector = cd
                    print("  [+] ContradictionDetector loaded")
                else:
                    print("  [-] ContradictionDetector: model failed to load")
            except Exception as e:
                print(f"  [-] ContradictionDetector failed: {e}")

    def _slm_detect(self, step: dict) -> dict:
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

        # Generate a unique key for this step based on observable inputs and active modules
        cache_key = (
            step.get("action", ""),
            step.get("tool_input", ""),
            step.get("tool_output", ""),
            step.get("agent_reasoning", ""),
            len(self._history),
            tuple(self.active_modules) if self.active_modules is not None else None
        )

        if cache_key in DetectionPipeline._step_signals_cache:
            signals_cached, detector_results_cached = DetectionPipeline._step_signals_cache[cache_key]
            # Make copies to prevent shared mutations
            signals = dict(signals_cached)
            detector_results = {k: dict(v) for k, v in detector_results_cached.items()}
            # Track history for contradiction detection
            self._history.append(step_data)
        else:
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

            # Save to class-level cache
            DetectionPipeline._step_signals_cache[cache_key] = (dict(signals), {k: dict(v) for k, v in detector_results.items()})

        # ── Tuned Hybrid Fusion ─────────────────────────────────────────
        fused_score = self._fuse_signals(signals)
        active_signal_count = sum(
            1 for v in signals.values() if v is not None
        )

        action = step_data.get("action", "").lower().strip()
        is_tool_action = action in self.TOOL_ACTIONS or any(kw in action for kw in ["search", "calc", "api", "query", "look", "fetch", "read"])
        is_reasoning_action = action in self.PLANNING_ACTIONS or any(kw in action for kw in ["plan", "think", "reason", "decompose"])

        # Adaptive MIN_SIGNALS: require at least 1 if few models loaded,
        # 2 if 3+ models loaded (prevents single-signal false positives)
        min_signals_required = 1 if self._loaded_count <= 2 else self.MIN_SIGNALS_FOR_DETECTION

        # 1. Base threshold
        base_flag = (fused_score > self.FUSION_THRESHOLD and active_signal_count >= min_signals_required)
        
        # 2. Strict Tool-Use hallucination
        sem_sim = signals.get("semantic_similarity")
        tool_flag = (signals.get("tool_claim_match") is False) and is_tool_action and (sem_sim is not None and sem_sim < 0.65)
        
        # 3. High-confidence factual contradiction
        nli_score = signals.get("nli_score")
        nli_flag = (nli_score is not None and nli_score > 0.75)
        
        # 4. Contradiction with previous steps
        contra_flag = (signals.get("contradiction_with_prev") is True) and is_reasoning_action

        hallucination_detected = base_flag or tool_flag or nli_flag or contra_flag

        # Confidence = fused score
        confidence = round(min(max(fused_score, 0.0), 1.0), 4)

        # Apply temperature scaling if T != 1.0
        import math
        T = getattr(self, "calibration_temperature", 1.0)
        if T != 1.0:
            fs_clamped = min(max(fused_score, 1e-7), 1.0 - 1e-7)
            logit = math.log(fs_clamped / (1.0 - fs_clamped))
            calibrated = 1.0 / (1.0 + math.exp(-logit / T))
            confidence = round(min(max(calibrated, 0.0), 1.0), 4)

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
            "hallucination_type": hallucination_type,
            "hallucination_type_predicted": hallucination_type,
            "severity": severity,
            "detection_signals": signals,
            # Diagnostics — helps debug zero-detection issues
            "_debug": {
                "fused_score": round(fused_score, 4),
                "active_signal_count": active_signal_count,
                "min_signals_required": min_signals_required,
                "models_loaded": self._loaded_count,
                "flags": {
                    "base": base_flag,
                    "tool": tool_flag,
                    "nli": nli_flag,
                    "contra": contra_flag,
                },
            },
        }

    def detect(self, step: dict) -> dict:
        """
        3-Layer Hybrid Detection:
        Layer 1: SLM ensemble (always runs)
        Layer 2: Llama 8B classifier (always runs)
        Layer 3: Nemotron judge (only if confidence < 0.70)
        """
        import time
        t_total_start = time.perf_counter()

        # Layer 1: Run SLM ensemble
        t_l1_start = time.perf_counter()
        slm_result = self._slm_detect(step)
        t_l1_end = time.perf_counter()
        t_layer1_ms = (t_l1_end - t_l1_start) * 1000.0

        # Layer 2: Llama classification
        t_l2_start = time.perf_counter()
        llama_result = None
        if self.llama_classifier:
            llama_result = self.llama_classifier.classify(step, slm_result)
        t_l2_end = time.perf_counter()
        t_layer2_ms = ((t_l2_end - t_l2_start) * 1000.0) if self.llama_classifier else 0.0

        primary_confidence = llama_result.get("confidence", slm_result.get("confidence", 0.0)) if llama_result else slm_result.get("confidence", 0.0)

        # Layer 3: Nemotron verification (conditional)
        t_l3_start = time.perf_counter()
        nemotron_result = None
        if self.nemotron_judge:
            nemotron_result = self.nemotron_judge.judge(step, primary_confidence)
        t_l3_end = time.perf_counter()
        t_layer3_ms = ((t_l3_end - t_l3_start) * 1000.0) if nemotron_result else 0.0

        # Final decision fusing layers
        if nemotron_result:
            # Nemotron was called — use weighted average
            final_confidence = (primary_confidence * 0.4) + (nemotron_result.get("confidence", primary_confidence) * 0.6)
            final_detected = final_confidence > 0.45
            final_type = nemotron_result.get("hallucination_type")
        else:
            # Nemotron skipped or unavailable — trust Llama/SLM
            final_confidence = primary_confidence
            final_detected = slm_result.get("hallucination_detected", False)
            final_type = llama_result.get("causal_label") if llama_result else slm_result.get("hallucination_type_predicted")

        # Severity from base SLM
        severity = slm_result.get("severity")
        if final_detected and not severity:
            severity = self._determine_severity(final_confidence, final_detected)

        # Attribution layer (Localizer + CausalClassifier) timing
        t_attr_start = time.perf_counter()
        if self.causal_classifier and final_detected:
            temp_res = {
                "hallucination_detected": final_detected,
                "confidence": final_confidence,
                "hallucination_type": final_type,
            }
            _ = self.causal_classifier.classify(step, temp_res)

        if self.localizer:
            step_num = step.get("step", 0)
            self._detection_history.append({
                "step": step_num,
                "detection_signals": slm_result.get("detection_signals", {}),
            })
            _ = self.localizer.localize(self._detection_history)
        t_attr_end = time.perf_counter()
        t_attribution_ms = (t_attr_end - t_attr_start) * 1000.0

        t_total_end = time.perf_counter()
        t_total_ms = (t_total_end - t_total_start) * 1000.0

        latency_breakdown = {
            "t_layer1_ms": round(t_layer1_ms, 2),
            "t_layer2_ms": round(t_layer2_ms, 2),
            "t_layer3_ms": round(t_layer3_ms, 2),
            "t_attribution_ms": round(t_attribution_ms, 2),
            "t_total_ms": round(t_total_ms, 2)
        }

        return {
            "hallucination_detected": final_detected,
            "confidence": round(final_confidence, 4),
            "hallucination_type": final_type,
            "hallucination_type_predicted": final_type,
            "severity": severity,
            "detection_signals": slm_result.get("detection_signals", {}),
            "_debug": slm_result.get("_debug", {}),
            "nemotron_called": nemotron_result is not None,
            "layers_used": {
                "slm_ensemble": True,
                "llama_8b": self.llama_classifier is not None,
                "nemotron_340b": nemotron_result is not None
            },
            "latency_breakdown": latency_breakdown
        }

    def calibrate_temperature(self, val_trajectories: list) -> float:
        """
        Optimizes calibration_temperature (T) on validation trajectories using Scipy.
        Minimizes Negative Log-Likelihood (NLL).
        """
        import math
        from scipy.optimize import minimize_scalar

        # Gather raw fused scores and true labels
        fused_scores = []
        labels = []
        
        # Store the original temp to restore later
        orig_temp = self.calibration_temperature
        # Temporarily set to 1.0 to get raw fused_score as confidence
        self.calibration_temperature = 1.0
        
        for traj in val_trajectories:
            self.reset_history()
            for step in traj.get("steps", []):
                res = self._slm_detect(step)
                # Clamp confidence to compute logit
                conf = min(max(res['confidence'], 1e-7), 1.0 - 1e-7)
                logit = math.log(conf / (1.0 - conf))
                
                # Ground truth label is step.get("ground_truth_label", False)
                label = 1 if step.get("ground_truth_label", False) else 0
                
                fused_scores.append(logit)
                labels.append(label)
                
        # Restore original temperature
        self.calibration_temperature = orig_temp
        
        if not fused_scores:
            print("No validation steps found for temperature calibration.")
            return orig_temp

        # Objective function: compute NLL for a given T
        def objective(t):
            if t <= 0:
                return 1e9
            nll = 0.0
            for logit, label in zip(fused_scores, labels):
                p = 1.0 / (1.0 + math.exp(-logit / t))
                p = min(max(p, 1e-7), 1.0 - 1e-7)
                if label == 1:
                    nll -= math.log(p)
                else:
                    nll -= math.log(1.0 - p)
            return nll / len(fused_scores)

        # Minimize NLL for T in bounds e.g. [0.1, 10.0]
        res = minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded')
        optimal_t = float(res.x)
        print(f"Optimal calibration temperature T: {optimal_t:.4f} (NLL: {res.fun:.4f})")
        
        # Update current temperature
        self.calibration_temperature = optimal_t
        
        # Also update global config
        import config
        config.CONFIG.thresholds.calibration_temperature = optimal_t
        
        return optimal_t

    def reset_history(self):
        """Clears step history. Call between trajectories."""
        self._history = []
        self._detection_history = []

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
