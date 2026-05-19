"""
AgentTrace - Quick Inference Verification Script
Verifies that the fine-tuned Llama-3.1-8B QLoRA model actually works.
Runs on Kaggle with GPU T4 x2.
"""
import subprocess, sys, os
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "bitsandbytes>=0.46.1", "peft", "accelerate", "transformers"])

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel

# ============================================
# CONFIG
# ============================================
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Set via Kaggle Secrets
MODEL_NAME = "meta-llama/Llama-3.1-8B"
# The fine-tuned LoRA weights from the training notebook output
PEFT_PATH = "/kaggle/input/notebooks/somnath26/agenttrace-llama-3-1-8b-fine-tuning/llama_causal_lora"

LABEL2ID = {
    "reasoning_error": 0,
    "instruction_deviation": 1,
    "context_fabrication": 2,
    "tool_misuse": 3,
    "knowledge_gap": 4,
    "no_hallucination": 5,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# ============================================
# LOAD MODEL
# ============================================
print("=" * 60)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
)
base_model.config.pad_token_id = tokenizer.pad_token_id

print(f"Loading LoRA adapter from {PEFT_PATH}...")
model = PeftModel.from_pretrained(base_model, PEFT_PATH)
model.eval()
print("Model loaded successfully!")
print("=" * 60)

# ============================================
# INFERENCE HELPER
# ============================================
def classify_step(action, tool_output, agent_reasoning):
    """Run inference on a single step and return label + confidence."""
    text = (
        f"Action: {action}\n"
        f"Reasoning: {agent_reasoning}\n"
        f"Tool Output: {tool_output}\n"
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits.float(), dim=-1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = ID2LABEL[pred_idx]
    pred_conf = float(probs[pred_idx])

    return {
        "predicted_label": pred_label,
        "confidence": round(pred_conf, 4),
        "all_probs": {ID2LABEL[i]: round(float(probs[i]), 4) for i in range(NUM_LABELS)},
    }

# ============================================
# TEST CASES
# ============================================
print("\n" + "=" * 60)
print("TEST 1: Clearly hallucinated step (reasoning contradicts tool)")
print("=" * 60)
r1 = classify_step(
    action="web_search",
    tool_output="Argentina won FIFA World Cup 2022",
    agent_reasoning="France won FIFA 2022, capital is Paris",
)
print(f"  Predicted Label : {r1['predicted_label']}")
print(f"  Confidence      : {r1['confidence']}")
print(f"  All Probs       : {r1['all_probs']}")

print("\n" + "=" * 60)
print("TEST 2: Clean step (reasoning matches tool output)")
print("=" * 60)
r2 = classify_step(
    action="web_search",
    tool_output="Argentina won FIFA World Cup 2022",
    agent_reasoning="Argentina won the 2022 World Cup",
)
print(f"  Predicted Label : {r2['predicted_label']}")
print(f"  Confidence      : {r2['confidence']}")
print(f"  All Probs       : {r2['all_probs']}")

print("\n" + "=" * 60)
print("TEST 3: Tool misuse (wrong tool for the task)")
print("=" * 60)
r3 = classify_step(
    action="calculator",
    tool_output="Error: invalid expression",
    agent_reasoning="I need to search the web, let me use the calculator",
)
print(f"  Predicted Label : {r3['predicted_label']}")
print(f"  Confidence      : {r3['confidence']}")
print(f"  All Probs       : {r3['all_probs']}")

print("\n" + "=" * 60)
print("TEST 4: Context fabrication (inventing facts)")
print("=" * 60)
r4 = classify_step(
    action="web_search",
    tool_output="The population of India is 1.4 billion",
    agent_reasoning="According to the search, India has 500 million people and the GDP is $10 trillion",
)
print(f"  Predicted Label : {r4['predicted_label']}")
print(f"  Confidence      : {r4['confidence']}")
print(f"  All Probs       : {r4['all_probs']}")

print("\n" + "=" * 60)
print("TEST 5: Knowledge gap (model doesn't know the answer)")
print("=" * 60)
r5 = classify_step(
    action="web_search",
    tool_output="No results found for this query",
    agent_reasoning="Based on my knowledge, the answer is definitely XYZ corporation founded in 1985",
)
print(f"  Predicted Label : {r5['predicted_label']}")
print(f"  Confidence      : {r5['confidence']}")
print(f"  All Probs       : {r5['all_probs']}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)
results = [r1, r2, r3, r4, r5]
for i, r in enumerate(results, 1):
    print(f"  Test {i}: {r['predicted_label']:25s} (conf={r['confidence']:.4f})")
print("=" * 60)
print("Expected: Test 2 should predict 'no_hallucination' (6th class).")
print("If model outputs varied labels with >0.20 confidence,")
print("the fine-tuning is working correctly!")
print("=" * 60)
