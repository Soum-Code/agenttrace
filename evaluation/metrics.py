"""
AgentTrace — Evaluation Metrics
=================================
Computes all evaluation metrics for step-level hallucination
detection. Compares predicted detections against ground truth
labels across the 5-category hallucination taxonomy.

All functions return a dict with metric name and value.
All functions have full docstrings with examples.

Author: P. Somnath Reddy (Research Lead)
GitHub: github.com/Soum-Code/agenttrace
"""

import os
import sys
from typing import List, Dict, Optional, Tuple
from collections import Counter


# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG


# ════════════════════════════════════════════════════════════
# 1. STEP LOCALIZATION ACCURACY
# ════════════════════════════════════════════════════════════

def step_localization_accuracy(
    predicted_steps: List[int],
    true_steps: List[int],
) -> Dict[str, float]:
    """Compute step localization accuracy (primary metric).

    This is the metric AgentHallu reports at 41.1% SOTA.
    Measures what fraction of truly hallucinated steps
    were correctly identified by the detector.

    Args:
        predicted_steps: List of step numbers flagged as hallucinated.
        true_steps: List of step numbers that are actually hallucinated.

    Returns:
        Dict with 'step_localization_accuracy' as float 0-1.

    Example:
        >>> step_localization_accuracy([2, 4], [2, 3, 4])
        {'step_localization_accuracy': 0.6667}
        >>> step_localization_accuracy([1, 2], [1, 2])
        {'step_localization_accuracy': 1.0}
        >>> step_localization_accuracy([], [1, 2])
        {'step_localization_accuracy': 0.0}
    """
    if not true_steps:
        # No hallucinations to find — perfect score if nothing predicted
        acc = 1.0 if not predicted_steps else 0.0
        return {"step_localization_accuracy": round(acc, 4)}

    # Count how many true hallucinated steps were correctly predicted
    true_set = set(true_steps)
    pred_set = set(predicted_steps)
    correct = len(true_set & pred_set)  # intersection = true positives

    acc = correct / len(true_set)
    return {"step_localization_accuracy": round(acc, 4)}


# ════════════════════════════════════════════════════════════
# 2. PRECISION AT K
# ════════════════════════════════════════════════════════════

def precision_at_k(
    predicted_steps: List[int],
    true_steps: List[int],
    k: int,
) -> Dict[str, float]:
    """Compute precision@k for the top-k predicted hallucinated steps.

    Measures how many of the top-k predictions are actual hallucinations.
    Useful when the detector ranks steps by confidence.

    Args:
        predicted_steps: Step numbers sorted by confidence (highest first).
        true_steps: Step numbers that are actually hallucinated.
        k: Number of top predictions to consider.

    Returns:
        Dict with 'precision_at_k' and the value of k used.

    Example:
        >>> precision_at_k([3, 1, 2], [1, 3], k=2)
        {'precision_at_2': 1.0}
        >>> precision_at_k([3, 1, 2], [1, 3], k=3)
        {'precision_at_3': 0.6667}
    """
    if k <= 0:
        return {f"precision_at_{k}": 0.0}

    # Take only top-k predictions
    top_k = predicted_steps[:k]
    if not top_k:
        return {f"precision_at_{k}": 0.0}

    true_set = set(true_steps)
    hits = sum(1 for s in top_k if s in true_set)  # true positives in top-k

    prec = hits / len(top_k)
    return {f"precision_at_{k}": round(prec, 4)}


# ════════════════════════════════════════════════════════════
# 3. RECALL
# ════════════════════════════════════════════════════════════

def recall(
    predicted_steps: List[int],
    true_steps: List[int],
) -> Dict[str, float]:
    """Compute recall for hallucination detection.

    What fraction of all true hallucinations did we catch?
    Equivalent to step_localization_accuracy but named for
    standard ML terminology.

    Args:
        predicted_steps: Step numbers flagged as hallucinated.
        true_steps: Step numbers actually hallucinated.

    Returns:
        Dict with 'recall' as float 0-1.

    Example:
        >>> recall([1, 2, 3], [2, 3])
        {'recall': 1.0}
        >>> recall([1], [2, 3])
        {'recall': 0.0}
    """
    if not true_steps:
        rec = 1.0 if not predicted_steps else 0.0
        return {"recall": round(rec, 4)}

    true_set = set(true_steps)
    pred_set = set(predicted_steps)
    tp = len(true_set & pred_set)

    rec = tp / len(true_set)
    return {"recall": round(rec, 4)}


# ════════════════════════════════════════════════════════════
# 4. PRECISION
# ════════════════════════════════════════════════════════════

def precision(
    predicted_steps: List[int],
    true_steps: List[int],
) -> Dict[str, float]:
    """Compute precision for hallucination detection.

    What fraction of our predictions were actually correct?

    Args:
        predicted_steps: Step numbers flagged as hallucinated.
        true_steps: Step numbers actually hallucinated.

    Returns:
        Dict with 'precision' as float 0-1.

    Example:
        >>> precision([1, 2, 3], [2, 3])
        {'precision': 0.6667}
        >>> precision([2, 3], [2, 3])
        {'precision': 1.0}
    """
    if not predicted_steps:
        prec = 1.0 if not true_steps else 0.0
        return {"precision": round(prec, 4)}

    true_set = set(true_steps)
    tp = sum(1 for s in predicted_steps if s in true_set)

    prec = tp / len(predicted_steps)
    return {"precision": round(prec, 4)}


# ════════════════════════════════════════════════════════════
# 5. F1 PER CATEGORY
# ════════════════════════════════════════════════════════════

def f1_per_category(
    predictions: List[Optional[str]],
    ground_truth: List[Optional[str]],
    categories: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-category F1 score for hallucination type classification.

    Evaluates how well we classify the TYPE of hallucination
    (Planning, Retrieval, Reasoning, Tool-Use, Human-Interaction).

    Args:
        predictions: Predicted hallucination types (None for non-halluc).
        ground_truth: True hallucination types (None for non-halluc).
        categories: List of category names. Defaults to CONFIG categories.

    Returns:
        Dict mapping each category to its precision/recall/f1.

    Example:
        >>> preds = ['Tool-Use', 'Reasoning', None]
        >>> truth = ['Tool-Use', 'Tool-Use', None]
        >>> result = f1_per_category(preds, truth)
        >>> 'Tool-Use' in result
        True
    """
    cats = categories or CONFIG.classifier.categories

    results = {}
    for cat in cats:
        # True positives, false positives, false negatives for this category
        tp = sum(1 for p, g in zip(predictions, ground_truth)
                 if p == cat and g == cat)
        fp = sum(1 for p, g in zip(predictions, ground_truth)
                 if p == cat and g != cat)
        fn = sum(1 for p, g in zip(predictions, ground_truth)
                 if p != cat and g == cat)

        # Precision
        cat_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # Recall
        cat_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # F1
        cat_f1 = (2 * cat_prec * cat_rec / (cat_prec + cat_rec)
                  if (cat_prec + cat_rec) > 0 else 0.0)

        results[cat] = {
            "precision": round(cat_prec, 4),
            "recall": round(cat_rec, 4),
            "f1": round(cat_f1, 4),
            "support": tp + fn,  # total true instances of this category
        }

    # Macro average across all categories
    valid_cats = [r for r in results.values() if r["support"] > 0]
    if valid_cats:
        macro_f1 = sum(r["f1"] for r in valid_cats) / len(valid_cats)
    else:
        macro_f1 = 0.0

    results["macro_avg"] = {"f1": round(macro_f1, 4)}

    return results


# ════════════════════════════════════════════════════════════
# 6. TASK COMPLETION RATE
# ════════════════════════════════════════════════════════════

def task_completion_rate(
    trajectories_before: List[Dict],
    trajectories_after: List[Dict],
) -> Dict[str, float]:
    """Compute task completion rate improvement after intervention.

    Measures how intervention (correcting hallucinations) improves
    the fraction of trajectories that produce correct final answers.

    Args:
        trajectories_before: Trajectories before intervention.
        trajectories_after: Trajectories after intervention.

    Returns:
        Dict with before/after rates and improvement.

    Example:
        >>> before = [{'final_answer_correct': False}, {'final_answer_correct': True}]
        >>> after = [{'final_answer_correct': True}, {'final_answer_correct': True}]
        >>> result = task_completion_rate(before, after)
        >>> result['improvement']
        0.5
    """
    def _rate(trajectories: List[Dict]) -> float:
        """Fraction of trajectories with correct final answers."""
        if not trajectories:
            return 0.0
        correct = sum(1 for t in trajectories
                      if t.get("final_answer_correct", False))
        return correct / len(trajectories)

    before_rate = _rate(trajectories_before)
    after_rate = _rate(trajectories_after)

    return {
        "task_completion_before": round(before_rate, 4),
        "task_completion_after": round(after_rate, 4),
        "improvement": round(after_rate - before_rate, 4),
    }


# ════════════════════════════════════════════════════════════
# 7. FALSE POSITIVE RATE
# ════════════════════════════════════════════════════════════

def false_positive_rate(
    predicted_steps: List[int],
    true_steps: List[int],
    total_steps: int,
) -> Dict[str, float]:
    """Compute false positive rate for hallucination detection.

    What fraction of non-hallucinated steps were incorrectly flagged?
    Critical metric: high FPR means the detector is too aggressive.

    Args:
        predicted_steps: Step numbers flagged as hallucinated.
        true_steps: Step numbers actually hallucinated.
        total_steps: Total number of steps in the trajectory.

    Returns:
        Dict with 'false_positive_rate' as float 0-1.

    Example:
        >>> false_positive_rate([1, 2, 3], [2], total_steps=5)
        {'false_positive_rate': 0.5}
        >>> false_positive_rate([2], [2], total_steps=5)
        {'false_positive_rate': 0.0}
    """
    true_set = set(true_steps)
    pred_set = set(predicted_steps)

    # True negatives + false positives = all non-hallucinated steps
    non_halluc_steps = total_steps - len(true_set)

    if non_halluc_steps == 0:
        # All steps are hallucinated, FPR is undefined (set to 0)
        return {"false_positive_rate": 0.0}

    # False positives: predicted as hallucinated but actually clean
    fp = len(pred_set - true_set)
    fpr = fp / non_halluc_steps

    return {"false_positive_rate": round(fpr, 4)}


# ════════════════════════════════════════════════════════════
# 8. AVERAGE LATENCY
# ════════════════════════════════════════════════════════════

def average_latency_ms(
    detection_times: List[float],
) -> Dict[str, float]:
    """Compute average detection latency in milliseconds.

    Measures the computational cost of the detection pipeline.
    Important for real-time agent monitoring applications.

    Args:
        detection_times: List of per-step detection times in milliseconds.

    Returns:
        Dict with avg, min, max, and p95 latency.

    Example:
        >>> average_latency_ms([10.0, 20.0, 30.0])
        {'avg_latency_ms': 20.0, 'min_latency_ms': 10.0, ...}
    """
    if not detection_times:
        return {
            "avg_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }

    sorted_times = sorted(detection_times)
    n = len(sorted_times)

    # P95: value at the 95th percentile index
    p95_idx = min(int(n * 0.95), n - 1)

    return {
        "avg_latency_ms": round(sum(sorted_times) / n, 2),
        "min_latency_ms": round(sorted_times[0], 2),
        "max_latency_ms": round(sorted_times[-1], 2),
        "p95_latency_ms": round(sorted_times[p95_idx], 2),
    }


# ════════════════════════════════════════════════════════════
# 9. AGGREGATE — compute all metrics at once
# ════════════════════════════════════════════════════════════

def compute_all_metrics(
    predicted_steps: List[int],
    true_steps: List[int],
    total_steps: int,
    predicted_types: Optional[List[Optional[str]]] = None,
    true_types: Optional[List[Optional[str]]] = None,
    detection_times: Optional[List[float]] = None,
) -> Dict[str, any]:
    """Compute all metrics in a single call.

    Convenience function that aggregates all individual metrics.

    Args:
        predicted_steps: Step numbers flagged as hallucinated.
        true_steps: Step numbers actually hallucinated.
        total_steps: Total steps in the trajectory.
        predicted_types: Predicted hallucination types (optional).
        true_types: True hallucination types (optional).
        detection_times: Per-step latencies in ms (optional).

    Returns:
        Dict containing all computed metrics.

    Example:
        >>> metrics = compute_all_metrics([2], [2, 3], total_steps=5)
        >>> 'step_localization_accuracy' in metrics
        True
    """
    results = {}

    # Core detection metrics
    results.update(step_localization_accuracy(predicted_steps, true_steps))
    results.update(precision(predicted_steps, true_steps))
    results.update(recall(predicted_steps, true_steps))

    # Precision@k for configured k values
    for k in CONFIG.evaluation.top_k_values:
        results.update(precision_at_k(predicted_steps, true_steps, k))

    # False positive rate
    results.update(false_positive_rate(predicted_steps, true_steps, total_steps))

    # F1 per category (if type predictions provided)
    if predicted_types and true_types:
        f1_results = f1_per_category(predicted_types, true_types)
        results["f1_per_category"] = f1_results
        results["macro_f1"] = f1_results["macro_avg"]["f1"]

    # Latency (if timing data provided)
    if detection_times:
        results.update(average_latency_ms(detection_times))

    return results


# ════════════════════════════════════════════════════════════
# TEST BLOCK
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("AgentTrace - Evaluation Metrics Test")
    print("=" * 60)

    # --- Test 1: Step Localization Accuracy ---
    print("\n--- Test 1: Step Localization Accuracy ---")
    r = step_localization_accuracy([2, 4], [2, 3, 4])
    print(f"  Predicted [2,4] vs True [2,3,4]: {r}")
    assert abs(r["step_localization_accuracy"] - 0.6667) < 0.001

    r = step_localization_accuracy([1, 2], [1, 2])
    print(f"  Predicted [1,2] vs True [1,2]:   {r}")
    assert r["step_localization_accuracy"] == 1.0

    # --- Test 2: Precision@K ---
    print("\n--- Test 2: Precision@K ---")
    r = precision_at_k([3, 1, 2], [1, 3], k=2)
    print(f"  Top-2 of [3,1,2] vs True [1,3]: {r}")
    assert r["precision_at_2"] == 1.0

    r = precision_at_k([3, 1, 2], [1, 3], k=3)
    print(f"  Top-3 of [3,1,2] vs True [1,3]: {r}")
    assert abs(r["precision_at_3"] - 0.6667) < 0.001

    # --- Test 3: Recall ---
    print("\n--- Test 3: Recall ---")
    r = recall([1, 2, 3], [2, 3])
    print(f"  Predicted [1,2,3] vs True [2,3]: {r}")
    assert r["recall"] == 1.0

    # --- Test 4: Precision ---
    print("\n--- Test 4: Precision ---")
    r = precision([1, 2, 3], [2, 3])
    print(f"  Predicted [1,2,3] vs True [2,3]: {r}")
    assert abs(r["precision"] - 0.6667) < 0.001

    # --- Test 5: F1 Per Category ---
    print("\n--- Test 5: F1 Per Category ---")
    preds = ["Tool-Use", "Reasoning", None, "Tool-Use"]
    truth = ["Tool-Use", "Tool-Use", None, "Reasoning"]
    r = f1_per_category(preds, truth)
    print(f"  Tool-Use F1:  {r['Tool-Use']}")
    print(f"  Reasoning F1: {r['Reasoning']}")
    print(f"  Macro avg:    {r['macro_avg']}")

    # --- Test 6: Task Completion Rate ---
    print("\n--- Test 6: Task Completion Rate ---")
    before = [
        {"final_answer_correct": False},
        {"final_answer_correct": True},
        {"final_answer_correct": False},
    ]
    after = [
        {"final_answer_correct": True},
        {"final_answer_correct": True},
        {"final_answer_correct": True},
    ]
    r = task_completion_rate(before, after)
    print(f"  Before: {r['task_completion_before']}")
    print(f"  After:  {r['task_completion_after']}")
    print(f"  Improvement: {r['improvement']}")
    assert abs(r["improvement"] - 0.6667) < 0.001

    # --- Test 7: False Positive Rate ---
    print("\n--- Test 7: False Positive Rate ---")
    r = false_positive_rate([1, 2, 3], [2], total_steps=5)
    print(f"  Predicted [1,2,3] vs True [2], 5 steps: {r}")
    assert r["false_positive_rate"] == 0.5

    r = false_positive_rate([2], [2], total_steps=5)
    print(f"  Predicted [2] vs True [2], 5 steps:     {r}")
    assert r["false_positive_rate"] == 0.0

    # --- Test 8: Average Latency ---
    print("\n--- Test 8: Average Latency ---")
    times = [10.0, 15.0, 20.0, 25.0, 100.0]
    r = average_latency_ms(times)
    print(f"  Times {times}: {r}")
    assert r["avg_latency_ms"] == 34.0

    # --- Test 9: Aggregate ---
    print("\n--- Test 9: Compute All Metrics ---")
    r = compute_all_metrics(
        predicted_steps=[2, 4],
        true_steps=[2, 3, 4],
        total_steps=5,
        predicted_types=["Tool-Use", None, None, "Reasoning", None],
        true_types=["Tool-Use", None, "Retrieval", "Reasoning", None],
        detection_times=[12.5, 13.0, 11.8, 14.2, 12.0],
    )
    print(f"  Step loc acc:  {r['step_localization_accuracy']}")
    print(f"  Precision:     {r['precision']}")
    print(f"  Recall:        {r['recall']}")
    print(f"  FPR:           {r['false_positive_rate']}")
    print(f"  Macro F1:      {r['macro_f1']}")
    print(f"  Avg latency:   {r['avg_latency_ms']}ms")

    # --- Baseline comparison ---
    print(f"\n--- AgentHallu Baseline Comparison ---")
    our_acc = r["step_localization_accuracy"]
    baseline = CONFIG.evaluation.agenthallu_baseline
    delta = our_acc - baseline
    status = "ABOVE" if delta > 0 else "BELOW"
    print(f"  Our accuracy:     {our_acc}")
    print(f"  AgentHallu SOTA:  {baseline}")
    print(f"  Delta:            {delta:+.4f} ({status} baseline)")

    print(f"\n{'=' * 60}")
    print("All metrics.py tests passed!")
    print(f"{'=' * 60}")
