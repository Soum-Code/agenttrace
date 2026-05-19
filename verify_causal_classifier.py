"""
Local Verification Script for 6-Class Causal Classifier.
Verifies that the fine-tuned model correctly outputs the new 'No-Hallucination' label
and doesn't collapse clean steps to 'Reasoning' / 'Context Fabrication'.
"""

import sys
import os

# Set up paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from attribution.causal_classifier import CausalClassifier
import config

print("=" * 60)
print("Initializing Causal Classifier...")
print("=" * 60)

classifier = CausalClassifier()

if classifier.model is None:
    print("ERROR: Causal classifier failed to load fine-tuned model!")
    sys.exit(1)

print("\nModel successfully loaded from fine-tuned checkpoint!")
print(f"Taxonomy Labels: {classifier.labels}")

# Test Case 1: Clean step (detector flagged it, but classifier should override it as No-Hallucination)
test_clean_step = {
    "action": "web_search(query='population of Tokyo 2026')",
    "agent_reasoning": "Searching for current population of Tokyo to give an accurate answer.",
    "tool_output": "The population of Tokyo in 2026 is estimated to be approximately 14 million."
}
detection_clean = {
    "hallucination_detected": True,
    "hallucination_score": 0.05,
    "hallucination_type": "None",
    "severity": "None"
}

# Test Case 2: Hallucinated Reasoning step (should be classified as Reasoning)
test_hall_step = {
    "action": "calculator(expression='14000000 * 1.05')",
    "agent_reasoning": "Let me calculate the projected population with a 5% increase: 14000000 * 1.05 = 14700000. Wait, the actual population of Tokyo is 37 million, this math makes no sense.",
    "tool_output": "14700000"
}
detection_hall = {
    "hallucination_detected": True,
    "hallucination_score": 0.85,
    "hallucination_type": "Reasoning",
    "severity": "High"
}

print("\n" + "=" * 60)
print("RUNNING TEST CASE 1 (Clean Step - expecting No-Hallucination override)")
print("=" * 60)
res_clean = classifier.classify(test_clean_step, detection_clean)
print(f"Prediction result: {res_clean}")
print(f"Predicted cause: {res_clean.get('causal_label')}")
print(f"Classifier override: {res_clean.get('classifier_override', False)}")

assert res_clean.get('causal_label') == "No-Hallucination", f"Expected No-Hallucination, got {res_clean.get('causal_label')}"
assert res_clean.get('classifier_override') == True, "Expected classifier_override to be True for clean steps"

print("\n" + "=" * 60)
print("RUNNING TEST CASE 2 (Reasoning Hallucination Step)")
print("=" * 60)
res_hall = classifier.classify(test_hall_step, detection_hall)
print(f"Prediction result: {res_hall}")
print(f"Predicted cause: {res_hall.get('causal_label')}")
print(f"Classifier override: {res_hall.get('classifier_override', False)}")

# Check that they predict different classes and res_hall is Reasoning or one of the hallucination labels
assert res_hall.get('causal_label') in ["Reasoning", "Planning", "Retrieval", "Tool-Use", "Human-Interaction"], f"Expected hallucination type, got {res_hall.get('causal_label')}"
assert res_clean.get('causal_label') != res_hall.get('causal_label'), "Failed: Both clean and hallucinated steps classified to same class!"

print("\n" + "=" * 60)
print("ALL TESTS PASSED SUCCESSFULLY! Causal Classifier taxonomy successfully fixed!")
print("=" * 60)
