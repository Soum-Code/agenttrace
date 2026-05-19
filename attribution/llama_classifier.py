"""
Attribution Module: Llama 3.1 8B Classifier (Phase 1)
Classifies the root cause of a detected hallucination using
a QLoRA fine-tuned Llama-3.1-8B model.

Until the fine-tuned weights are present locally, this gracefully
falls back to the existing heuristic logic to avoid breaking the app.
"""

import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
    from peft import PeftModel
    import torch
except ImportError:
    pass

import config


class LlamaClassifier:
    """
    Given a step flagged as hallucinated, classifies the likely root cause
    using Llama-3.1-8B.
    """

    def __init__(self):
        self.labels = config.CAUSAL_LABELS
        self.confidence_threshold = config.CAUSAL_CONFIDENCE_THRESHOLD
        
        # Path where the user will save their Kaggle fine-tuned model
        self.model_name = "meta-llama/Llama-3.1-8B"
        self.peft_path = os.path.join(config.CONFIG.paths.models_dir, "llama_causal_lora")
        
        self.model = None
        self.tokenizer = None
        
        # We only try loading if the PEFT weights exist AND a CUDA GPU is available
        # (4-bit quantized Llama requires GPU — no point downloading 16GB on CPU-only machines)
        has_cuda = torch.cuda.is_available()
        if os.path.exists(self.peft_path) and has_cuda:
            print(f"Loading fine-tuned Llama from {self.peft_path}...")
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=len(self.labels),
                    quantization_config=bnb_config,
                    device_map="auto"
                )
                
                self.model = PeftModel.from_pretrained(base_model, self.peft_path)
                self.model.eval()
                print("Llama classifier successfully loaded.")
            except Exception as e:
                print(f"Error loading Llama classifier: {e}")
                self.model = None
                self.tokenizer = None
        elif os.path.exists(self.peft_path) and not has_cuda:
            print("Llama LoRA weights found but no CUDA GPU available. Using fallback heuristic.")
        else:
            print("Llama-3.1-8B LoRA weights not found. Using fallback heuristic.")

    def classify(self, step_data: dict, detection_result: dict) -> dict:
        """
        Classifies the causal root of a hallucination.
        """
        step_id = step_data.get("step")

        if not detection_result.get("hallucination_detected", False):
            return {
                "step": step_id,
                "causal_label": None,
                "causal_confidence": 0.0,
                "message": "No hallucination detected — skipping classification",
            }

        # Fallback to heuristics if model isn't trained/loaded yet
        if not self.model or not self.tokenizer:
            return self._fallback_heuristic(step_data, detection_result)

        try:
            feature_text = self._build_prompt(step_data, detection_result)

            inputs = self.tokenizer(
                feature_text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.model.device)
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
                
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()

            label_probs = {self.labels[i]: round(probs[i], 4) for i in range(len(self.labels))}
            best_idx = probs.index(max(probs))
            best_label = self.labels[best_idx]
            best_conf = round(probs[best_idx], 4)

            if best_conf < self.confidence_threshold:
                return self._fallback_heuristic(step_data, detection_result)

            return {
                "step": step_id,
                "causal_label": best_label,
                "confidence": best_conf,
                "all_label_probs": label_probs,
            }

        except Exception as e:
            return self._fallback_heuristic(step_data, detection_result)

    def _build_prompt(self, step_data: dict, detection_result: dict) -> str:
        parts = [
            f"Action: {step_data.get('action', '')}",
            f"Reasoning: {step_data.get('agent_reasoning', '')}",
            f"Tool Output: {step_data.get('tool_output', '')}",
        ]
        return "\n".join(parts)

    def _fallback_heuristic(self, step_data: dict, detection_result: dict) -> dict:
        step_id = step_data.get("step")
        signals = detection_result.get("detection_signals", {})
        h_type = detection_result.get("hallucination_type_predicted", "")

        if signals.get("tool_claim_match") is False:
            label = config.TYPE_TOOL_USE
            conf = 0.7
        elif h_type == config.TYPE_CONTRADICTION:
            label = config.TYPE_REASONING
            conf = 0.65
        elif signals.get("nli_score") and signals["nli_score"] > 0.8:
            label = config.TYPE_RETRIEVAL
            conf = 0.6
        else:
            label = config.TYPE_REASONING
            conf = 0.5

        return {
            "step": step_id,
            "causal_label": label,
            "confidence": conf,
            "method": "heuristic_fallback",
        }
