"""
AgentTrace — Pipeline Diagnostic Script
========================================
Run this on Kaggle to debug zero-detection issues.
Paste each section into separate Kaggle cells.

IMPORTANT: Run this FIRST in Cell 0:
    !pip install sentence-transformers transformers torch --quiet
"""

# ════════════════════════════════════════════════════════════
# CELL 1: Check if models load
# ════════════════════════════════════════════════════════════

import sys
sys.path.insert(0, '/kaggle/working/agenttrace')

print("=" * 60)
print("CELL 1: Model Loading Check")
print("=" * 60)

# Check imports first
try:
    import torch
    print(f"[OK] torch {torch.__version__}")
    print(f"     CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("[FAIL] torch not installed!")

try:
    import sentence_transformers
    print(f"[OK] sentence-transformers {sentence_transformers.__version__}")
except ImportError:
    print("[FAIL] sentence-transformers not installed!")
    print("     FIX: !pip install sentence-transformers")

try:
    import transformers
    print(f"[OK] transformers {transformers.__version__}")
except ImportError:
    print("[FAIL] transformers not installed!")
    print("     FIX: !pip install transformers")

print()

# Now check actual model loading
from detection.semantic_checker import SemanticChecker
from detection.factual_grounding import FactualGrounder
from detection.tool_validator import ToolValidator
from detection.contradiction import ContradictionDetector

sc = SemanticChecker()
fg = FactualGrounder()
tv = ToolValidator()
cd = ContradictionDetector()

print(f"SemanticChecker model:      {'LOADED' if sc.model is not None else 'FAILED'}")
print(f"FactualGrounder model:      {'LOADED' if fg.model is not None else 'FAILED'}")
print(f"ToolValidator model:        {'LOADED' if tv.model is not None else 'FAILED'}")
print(f"ContradictionDetector model:{'LOADED' if cd.model is not None else 'FAILED'}")

loaded = sum(1 for m in [sc.model, fg.model, tv.model, cd.model] if m is not None)
print(f"\nModels loaded: {loaded}/4")
if loaded == 0:
    print("\n*** ALL MODELS FAILED — this is why you get zero detection! ***")
    print("FIX: Run this in a cell before anything else:")
    print("  !pip install sentence-transformers transformers torch --quiet")


# ════════════════════════════════════════════════════════════
# CELL 2: Test individual detectors on a known hallucination
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("CELL 2: Individual Detector Test")
print("=" * 60)

test_step = {
    "step": 1,
    "action": "web_search",
    "tool_input": "FIFA 2022 winner",
    "tool_output": "Argentina won FIFA World Cup 2022",
    "agent_reasoning": "France won FIFA 2022, capital is Paris",
    "ground_truth": "Argentina won FIFA 2022",
    "ground_truth_label": True,
    "hallucination_type": "Tool-Use"
}

print("\nTest step: reasoning says 'France' but tool says 'Argentina'")
print("-" * 60)

# Semantic
r1 = sc.check(test_step)
sim = r1['detection_signals']['semantic_similarity']
print(f"\n1. Semantic Checker:")
print(f"   similarity:  {sim}")
print(f"   detected:    {r1['hallucination_detected']}")
print(f"   error:       {r1.get('error', 'None')}")

# NLI
r2 = fg.ground(test_step)
nli = r2['detection_signals']['nli_score']
print(f"\n2. Factual Grounder (NLI):")
print(f"   nli_score:   {nli}")
print(f"   detected:    {r2['hallucination_detected']}")
print(f"   error:       {r2.get('error', 'None')}")
if 'detail' in r2:
    print(f"   entailment:  {r2['detail'].get('entailment')}")
    print(f"   contradict:  {r2['detail'].get('contradiction')}")
    print(f"   neutral:     {r2['detail'].get('neutral')}")

# Tool
r3 = tv.validate(test_step)
tcm = r3['detection_signals']['tool_claim_match']
print(f"\n3. Tool Validator:")
print(f"   claim_match: {tcm}")
print(f"   detected:    {r3['hallucination_detected']}")
print(f"   error:       {r3.get('error', 'None')}")
if 'detail' in r3:
    print(f"   claims:      {r3['detail'].get('total_claims')}")
    print(f"   mismatched:  {r3['detail'].get('mismatched_claims')}")

# Summary
print("\n" + "-" * 60)
print("EXPECTED: All 3 should detect=True for this hallucinated step")
all_detected = r1['hallucination_detected'] and r2['hallucination_detected'] and r3['hallucination_detected']
print(f"ACTUAL:   {'ALL CORRECT ✓' if all_detected else 'SOME FAILED ✗'}")


# ════════════════════════════════════════════════════════════
# CELL 3: Test full pipeline
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("CELL 3: Full Pipeline Test")
print("=" * 60)

from detection.pipeline import DetectionPipeline
import json

pipeline = DetectionPipeline()

# Reset history
pipeline.reset_history()

# Test hallucinated step
result = pipeline.detect(test_step)

print(f"\nHallucinated step result:")
print(f"  detected:      {result['hallucination_detected']}")
print(f"  confidence:    {result['confidence']}")
print(f"  type:          {result.get('hallucination_type')}")
print(f"  severity:      {result.get('severity')}")
print(f"\n  Signals:")
for k, v in result.get('detection_signals', {}).items():
    print(f"    {k}: {v}")

# Print diagnostics
debug = result.get('_debug', {})
print(f"\n  Diagnostics:")
print(f"    fused_score:       {debug.get('fused_score')}")
print(f"    active_signals:    {debug.get('active_signal_count')}")
print(f"    min_required:      {debug.get('min_signals_required')}")
print(f"    models_loaded:     {debug.get('models_loaded')}")
print(f"    flags:             {debug.get('flags')}")

# Test clean step
test_clean = {
    "step": 2,
    "action": "calculator",
    "tool_input": "640 * 1e9",
    "tool_output": "640000000000",
    "agent_reasoning": "The calculator confirms the GDP is 640 billion, which equals 640000000000.",
    "ground_truth_label": False,
    "hallucination_type": None,
}

result2 = pipeline.detect(test_clean)
print(f"\nClean step result:")
print(f"  detected:      {result2['hallucination_detected']}")
print(f"  confidence:    {result2['confidence']}")

debug2 = result2.get('_debug', {})
print(f"  fused_score:   {debug2.get('fused_score')}")
print(f"  flags:         {debug2.get('flags')}")

print(f"\n{'=' * 60}")
if result['hallucination_detected'] and not result2['hallucination_detected']:
    print("PIPELINE WORKING CORRECTLY ✓")
    print("Hallucinated step detected, clean step passed")
elif not result['hallucination_detected']:
    print("PIPELINE BUG: Hallucinated step NOT detected ✗")
    print("Check the diagnostics above for root cause")
else:
    print("PIPELINE ISSUE: Check clean step false positive")
print(f"{'=' * 60}")
