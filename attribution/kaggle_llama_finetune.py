"""
AgentTrace - Phase 1 Kaggle Training Script
Llama-3.1-8B QLoRA Fine-tuning for Hallucination Attribution

Instructions for Kaggle:
1. Create a new notebook, select GPU P100 or 2x T4.
2. Upload your `synthetic_trajectories.json` file to the Kaggle input.
3. Paste the contents of this file into cells as indicated below.
"""

# ==========================================
# CELL 1: Install Dependencies
# ==========================================
import subprocess
import sys

def install_deps():
    print("Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-U", "-q",
        "transformers", "peft", "bitsandbytes>=0.46.1", "accelerate", "datasets", "scikit-learn"
    ])

try:
    install_deps()
except Exception as e:
    print(f"Dependency installation warning: {e}")

# ==========================================
# CELL 2: Imports & Setup
# ==========================================
import json
import os
import torch
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configurations
MODEL_NAME = "meta-llama/Llama-3.1-8B"
DATA_PATH = "/kaggle/input/agenttrace-synthetic-data-v1/synthetic_trajectories.json"
OUTPUT_DIR = "./llama_causal_lora"

# Our 6 taxonomy labels mapped to integers
LABELS = ["Planning", "Retrieval", "Reasoning", "Tool-Use", "Human-Interaction", "No-Hallucination"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# ==========================================
# CELL 3: Data Loading & Formatting
# ==========================================
def load_and_format_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        trajectories = json.load(f)
        
    texts = []
    labels = []
    
    for traj in trajectories:
        for step in traj.get("steps", []):
            is_hallucinated = step.get("ground_truth_label", False)
            if is_hallucinated:
                h_type = step.get("hallucination_type")
                if h_type in LABEL2ID:
                    label = h_type
                else:
                    label = "Reasoning"
            else:
                label = "No-Hallucination"
                
            text = (
                f"Action: {step.get('action', '')}\n"
                f"Reasoning: {step.get('agent_reasoning', '')}\n"
                f"Tool Output: {step.get('tool_output', '')}"
            )
            texts.append(text)
            labels.append(LABEL2ID[label])
                    
    return texts, labels

# Find dataset path dynamically
def find_dataset(filename="synthetic_trajectories.json"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if filename in files:
            return os.path.join(root, filename)
    return None

DATA_PATH = find_dataset()
if not DATA_PATH:
    raise FileNotFoundError(
        "Could not find synthetic_trajectories.json anywhere in /kaggle/input! "
        "Please make sure the dataset is successfully added to the kernel."
    )

print(f"Dataset found at: {DATA_PATH}")
texts, labels = load_and_format_data(DATA_PATH)

# Print distribution of classes to verify they are all there
print("Training Data Class Distribution:")
from collections import Counter
class_counts = Counter(labels)
for label_id, count in sorted(class_counts.items()):
    label_name = ID2LABEL[label_id]
    pct = (count / len(labels)) * 100
    print(f"  {label_name:20s}: {count:5d} ({pct:.1f}%)")

print(f"Found {len(texts)} total steps for training.")

# Split and create HuggingFace Dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

# ==========================================
# CELL 4: Tokenization
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Set via Kaggle Secrets

print("Loading Tokenizer...")
from huggingface_hub import login
try:
    login(HF_TOKEN)
except Exception as e:
    print(f"Failed to login to HF: {e}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
# Llama doesn't have a default pad token, so we use eos_token
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format("torch")
val_dataset.set_format("torch")

# ==========================================
# CELL 5: 4-bit QLoRA Model Setup
# ==========================================
print("Loading Model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN
)
model.config.pad_token_id = tokenizer.pad_token_id

# Prepare model for PEFT
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # targeting attention blocks
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==========================================
# CELL 6: Training
# ==========================================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-4,
    per_device_train_batch_size=2,      # Very small to fit in 16GB VRAM
    gradient_accumulation_steps=8,      # Virtual batch size = 16
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,                          # Faster training
    logging_steps=10,
    optim="paged_adamw_8bit"            # Memory efficient optimizer
)

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Weights for our 6 classes: Planning, Retrieval, Reasoning, Tool-Use, Human-Interaction, No-Hallucination
weights = [3.0, 2.0, 1.5, 3.0, 3.0, 1.0]

trainer = WeightedTrainer(
    class_weights=weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

print("Starting Training...")
trainer.train()

# ==========================================
# CELL 7: Save and Zip LoRA Weights
# ==========================================
print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Zipping for download...")
import shutil
shutil.make_archive("llama_causal_lora", "zip", OUTPUT_DIR)
print("Done! You can now download llama_causal_lora.zip from Kaggle output.")
