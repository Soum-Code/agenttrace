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
# !pip install -q transformers peft bitsandbytes accelerate datasets scikit-learn

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
DATA_PATH = "/kaggle/input/agenttrace-data/synthetic_trajectories.json" # UPDATE THIS PATH
OUTPUT_DIR = "./llama_causal_lora"

# Our 5 taxonomy labels mapped to integers
LABELS = ["Planning", "Retrieval", "Reasoning", "Tool-Use", "Human-Interaction"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# ==========================================
# CELL 3: Data Loading & Formatting
# ==========================================
def load_and_format_data(json_path):
    with open(json_path, 'r') as f:
        trajectories = json.load(f)
        
    texts = []
    labels = []
    
    for traj in trajectories:
        for step in traj.get("steps", []):
            if step.get("hallucination_detected"):
                # Ensure the type is one of our 5 classes
                h_type = step.get("hallucination_type")
                if h_type in LABEL2ID:
                    # Format exactly like our LlamaClassifier expects
                    text = (
                        f"Action: {step.get('action', '')}\n"
                        f"Reasoning: {step.get('agent_reasoning', '')}\n"
                        f"Tool Output: {step.get('tool_output', '')}"
                    )
                    texts.append(text)
                    labels.append(LABEL2ID[h_type])
                    
    return texts, labels

print("Loading dataset...")
try:
    texts, labels = load_and_format_data(DATA_PATH)
    print(f"Found {len(texts)} hallucinated steps for training.")
except Exception as e:
    print(f"Dataset load error (make sure path is correct): {e}")

# Split and create HuggingFace Dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

# ==========================================
# CELL 4: Tokenization
# ==========================================
print("Loading Tokenizer...")
# Llama 3 requires a Hugging Face token usually, make sure to login:
# from huggingface_hub import login
# login("YOUR_HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
    device_map="auto"
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
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,                          # Faster training
    logging_steps=10,
    optim="paged_adamw_8bit"            # Memory efficient optimizer
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

print("Starting Training...")
# trainer.train() # UNCOMMENT TO TRAIN

# ==========================================
# CELL 7: Save and Zip LoRA Weights
# ==========================================
# print("Saving model...")
# trainer.save_model(OUTPUT_DIR)
# tokenizer.save_pretrained(OUTPUT_DIR)

# print("Zipping for download...")
# import shutil
# shutil.make_archive("llama_causal_lora", "zip", OUTPUT_DIR)
# print("Done! You can now download llama_causal_lora.zip from Kaggle output.")
