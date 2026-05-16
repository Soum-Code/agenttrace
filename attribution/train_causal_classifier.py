"""
Script to fine-tune the Causal Classifier (DistilBERT) on synthetic trajectory data.
Maps hallucination types to causal root-cause labels and trains sequence classification.
"""

import os
import sys
import json
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from datasets import Dataset
except ImportError:
    print("ERROR: Missing required libraries for training.")
    print("Run: pip install transformers datasets torch")
    sys.exit(1)

import config

def get_causal_label(hallucination_type: str) -> str:
    """Map Detection hallucination_type to Attribution causal label.
    
    The hallucination types from the detection pipeline match the
    5-category taxonomy in CAUSAL_LABELS directly.
    """
    # The taxonomy categories are the causal labels themselves
    valid_labels = set(config.CAUSAL_LABELS)
    if hallucination_type in valid_labels:
        return hallucination_type
    # Fallback for any non-standard types
    return "Reasoning"

def build_feature_text(step: dict) -> str:
    """Replicates _build_feature_text from CausalClassifier."""
    h_type = step.get("hallucination_type", "")
    parts = [
        f"Action: {step.get('action', '')}",
        f"Reasoning: {step.get('agent_reasoning', '')}",
        f"Tool Output: {step.get('tool_output', '')}",
        f"Hallucination Type: {h_type}",
        f"Severity: High", # Assume high for ground truth positives
    ]
    return " | ".join(parts)

def prepare_dataset(data_path: str, labels_list: list):
    """Load JSON trajectories and create HuggingFace Dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        trajectories = json.load(f)

    texts = []
    labels = []
    
    # Create label-to-id mapping
    label2id = {label: i for i, label in enumerate(labels_list)}

    for traj in trajectories:
        for step in traj.get("steps", []):
            if step.get("ground_truth_label", False):
                h_type = step.get("hallucination_type")
                if not h_type:
                    continue
                
                # Build feature text
                feature_text = build_feature_text(step)
                texts.append(feature_text)
                
                # Get causal label ID
                causal_label = get_causal_label(h_type)
                labels.append(label2id[causal_label])

    return Dataset.from_dict({"text": texts, "label": labels})

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    print("=" * 60)
    print("AgentTrace - Training Causal Classifier")
    print("=" * 60)

    # Setup paths
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = project_root / "data" / "trajectories" / "synthetic_trajectories.json"
    model_output_dir = project_root / "models" / "causal_classifier_finetuned"
    
    base_model_name = "distilbert-base-uncased"
    causal_labels = config.CAUSAL_LABELS

    print(f"Loading data from: {data_path}")
    dataset = prepare_dataset(str(data_path), causal_labels)
    
    if len(dataset) == 0:
        print("ERROR: No hallucination samples found in the dataset.")
        sys.exit(1)
        
    print(f"Loaded {len(dataset)} hallucinated steps for training.")
    
    # Split into train and eval (80/20)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"Loading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    print(f"Loading model: {base_model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=len(causal_labels),
        id2label={i: label for i, label in enumerate(causal_labels)},
        label2id={label: i for i, label in enumerate(causal_labels)}
    )

    training_args = TrainingArguments(
        output_dir=str(project_root / "models" / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=str(project_root / "logs" / "causal_classifier"),
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nTraining complete. Saving best model to {model_output_dir}")
    os.makedirs(model_output_dir, exist_ok=True)
    trainer.save_model(str(model_output_dir))
    print("Done!")

if __name__ == "__main__":
    main()
