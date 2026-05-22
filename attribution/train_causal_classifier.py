"""
Script to fine-tune the Causal Classifier (DistilBERT) on synthetic trajectory data.
Maps hallucination types to causal root-cause labels and trains 6-class sequence
classification (5 hallucination types + No-Hallucination for clean steps).

Changes from v1:
  - Added "No-Hallucination" as the 6th class for clean steps
  - Clean steps (ground_truth_label=False/None) are now included in training
  - Class-weighted loss to handle imbalance (clean steps >> hallucinated steps)
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter

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


# ═══════════════════════════════════════════════════════════════
# Class weights to handle imbalance between clean and hallucinated
# Clean steps are more common; rare hallucination types get boosted.
# ═══════════════════════════════════════════════════════════════

CLASS_WEIGHT_MAP = {
    "No-Hallucination": 1.0,       # most common — no boost
    "Tool-Use":         3.0,       # rare, boost
    "Planning":         3.0,       # rare, boost
    "Reasoning":        1.5,       # moderate
    "Retrieval":        2.0,       # uncommon
    "Human-Interaction": 3.0,      # rare, boost
}


def get_causal_label(hallucination_type: str) -> str:
    """Map Detection hallucination_type to Attribution causal label.

    The hallucination types from the detection pipeline match the
    6-category taxonomy in CAUSAL_LABELS directly.
    """
    valid_labels = set(config.CAUSAL_LABELS)
    if hallucination_type in valid_labels:
        return hallucination_type
    # Fallback for any non-standard types
    return "Reasoning"


def build_feature_text(step: dict, is_clean: bool = False) -> str:
    """Replicates _build_feature_text from CausalClassifier.

    For clean steps, hallucination_type and severity are set to 'None'
    so the model learns to associate those patterns with No-Hallucination.
    """
    if is_clean:
        h_type = "None"
        severity = "None"
    else:
        h_type = step.get("hallucination_type", "")
        severity = "High"  # Assume high for ground truth positives

    parts = [
        f"Action: {step.get('action', '')}",
        f"Reasoning: {step.get('agent_reasoning', '')}",
        f"Tool Output: {step.get('tool_output', '')}",
        f"Hallucination Type: {h_type}",
        f"Severity: {severity}",
    ]
    return " | ".join(parts)


def prepare_dataset(data_path: str, labels_list: list):
    """Load JSON trajectories and create HuggingFace Dataset.

    Now includes BOTH hallucinated AND clean steps:
    - Hallucinated steps → mapped to their hallucination type label
    - Clean steps (ground_truth_label=False or absent) → "No-Hallucination"
    """
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
            is_hallucinated = step.get("ground_truth_label", False)

            if is_hallucinated:
                # Hallucinated step — use its type
                h_type = step.get("hallucination_type")
                if not h_type:
                    continue
                causal_label = get_causal_label(h_type)
                feature_text = build_feature_text(step, is_clean=False)
            else:
                # Clean step — label as No-Hallucination
                causal_label = "No-Hallucination"
                feature_text = build_feature_text(step, is_clean=True)

            texts.append(feature_text)
            labels.append(label2id[causal_label])

    return Dataset.from_dict({"text": texts, "label": labels})


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)

    # Per-class accuracy for monitoring
    per_class = precision_recall_fscore_support(labels, predictions, average=None)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


class WeightedTrainer(Trainer):
    """Custom Trainer that applies per-class weights to the loss function.

    This addresses class imbalance: clean steps are far more common than
    any single hallucination type. Without weighting, the model would
    learn to predict "No-Hallucination" for everything.
    """

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

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    print("=" * 60)
    print("AgentTrace - Training Causal Classifier (6-Class)")
    print("=" * 60)

    # Setup paths
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = project_root / "data" / "trajectories" / "synthetic_trajectories.json"
    model_output_dir = project_root / "models" / "causal_classifier_finetuned"

    base_model_name = "distilbert-base-uncased"
    causal_labels = config.CAUSAL_LABELS

    print(f"\nLabel taxonomy ({len(causal_labels)} classes):")
    for i, label in enumerate(causal_labels):
        print(f"  {i}: {label}")

    print(f"\nLoading data from: {data_path}")
    dataset = prepare_dataset(str(data_path), causal_labels)

    if len(dataset) == 0:
        print("ERROR: No samples found in the dataset.")
        sys.exit(1)

    # Print class distribution
    label_counts = Counter(dataset["label"])
    print(f"\nLoaded {len(dataset)} total steps for training:")
    for label_id, count in sorted(label_counts.items()):
        label_name = causal_labels[label_id]
        pct = 100 * count / len(dataset)
        print(f"  {label_name:25s}: {count:4d} ({pct:.1f}%)")

    # Split into train and eval (80/20)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"\nTrain: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    print(f"\nLoading tokenizer: {base_model_name}")
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

    # Build class weight vector in label order
    class_weights = [CLASS_WEIGHT_MAP.get(label, 1.0) for label in causal_labels]
    print(f"\nClass weights: {dict(zip(causal_labels, class_weights))}")

    training_args = TrainingArguments(
        output_dir=str(project_root / "models" / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(project_root / "logs" / "causal_classifier"),
        logging_steps=10,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    # Print final eval metrics
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    eval_results = trainer.evaluate()
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")

    print(f"\nSaving best model to {model_output_dir}")
    os.makedirs(model_output_dir, exist_ok=True)
    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))
    print("Done!")


if __name__ == "__main__":
    main()
