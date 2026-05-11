"""
AgentTrace — Benchmark Runner
===============================
Runs the full evaluation pipeline on all trajectories.
Computes metrics, compares against AgentHallu baseline,
generates results table, saves to JSON, and logs to WandB.

Author: P. Somnath Reddy (Research Lead)
GitHub: github.com/Soum-Code/agenttrace
"""

import os
import sys
import json
import time
import datetime
from typing import List, Dict, Optional, Tuple


# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG
from evaluation.metrics import (
    step_localization_accuracy,
    precision_at_k,
    precision,
    recall,
    f1_per_category,
    false_positive_rate,
    average_latency_ms,
    compute_all_metrics,
)


# ════════════════════════════════════════════════════════════
# MOCK DETECTOR (placeholder until Member 2 finishes)
# ════════════════════════════════════════════════════════════

def mock_detector(step: Dict) -> Dict:
    """Placeholder detection function for testing the benchmark pipeline.

    Uses ground truth labels with added noise to simulate a
    realistic but imperfect detector. Will be replaced by
    Member 2's actual detection pipeline.

    Args:
        step: A single step dict from a trajectory.

    Returns:
        detection_result dict matching the team schema.

    Example:
        >>> step = {'ground_truth_label': True, 'hallucination_type': 'Tool-Use'}
        >>> result = mock_detector(step)
        >>> 'hallucination_detected' in result
        True
    """
    import random
    is_halluc = step.get("ground_truth_label", False)
    true_type = step.get("hallucination_type")

    if is_halluc:
        # 80% chance of correct detection (simulates imperfect detector)
        detected = random.random() < 0.80
        confidence = random.uniform(0.65, 0.95) if detected else random.uniform(0.3, 0.55)
        pred_type = true_type if (detected and random.random() < 0.7) else None
    else:
        # 10% false positive rate
        detected = random.random() < 0.10
        confidence = random.uniform(0.6, 0.75) if detected else random.uniform(0.1, 0.4)
        pred_type = random.choice(CONFIG.classifier.categories) if detected else None

    # Assign severity based on confidence thresholds from config
    severity = None
    if detected:
        if confidence >= CONFIG.thresholds.severity_thresholds["High"]:
            severity = "High"
        elif confidence >= CONFIG.thresholds.severity_thresholds["Medium"]:
            severity = "Medium"
        else:
            severity = "Low"

    return {
        "hallucination_detected": detected,
        "confidence": round(confidence, 4),
        "hallucination_type_predicted": pred_type,
        "severity": severity,
    }


# ════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ════════════════════════════════════════════════════════════

class BenchmarkRunner:
    """Runs the full evaluation pipeline on a trajectory dataset.

    Loads trajectories, runs detection on each step, computes
    all metrics, compares against baselines, and saves results.

    Attributes:
        trajectories: Loaded list of trajectory dicts.
        detector_fn: Detection function to evaluate.
        results: Computed benchmark results after run().

    Example:
        >>> runner = BenchmarkRunner()
        >>> runner.load_trajectories()
        >>> runner.run()
        >>> runner.print_results_table()
    """

    def __init__(self, detector_fn=None) -> None:
        """Initialize the benchmark runner.

        Args:
            detector_fn: Detection function that takes a step dict
                and returns a detection_result dict. Defaults to
                mock_detector for pipeline testing.
        """
        self.trajectories: List[Dict] = []
        self.detector_fn = detector_fn or mock_detector
        self.results: Dict = {}
        self.per_trajectory_results: List[Dict] = []

        # Seed for reproducibility
        import random
        random.seed(CONFIG.training.seed)

    def load_trajectories(
        self,
        filepath: Optional[str] = None,
    ) -> int:
        """Load trajectories from JSON file.

        Args:
            filepath: Path to trajectory JSON. Defaults to
                CONFIG synthetic output path.

        Returns:
            Number of trajectories loaded.

        Example:
            >>> runner = BenchmarkRunner()
            >>> count = runner.load_trajectories()
            >>> count > 0
            True
        """
        if filepath is None:
            filepath = os.path.join(
                CONFIG.paths.trajectory_dir,
                CONFIG.synthetic.output_filename,
            )

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                self.trajectories = json.load(f)
            print(f"Loaded {len(self.trajectories)} trajectories from {filepath}")
            return len(self.trajectories)
        except FileNotFoundError:
            print(f"ERROR: File not found: {filepath}")
            return 0
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON: {e}")
            return 0

    def run(self) -> Dict:
        """Run the full benchmark pipeline.

        For each trajectory:
        1. Run detector on every step.
        2. Collect predictions vs ground truth.
        3. Compute per-trajectory metrics.
        4. Aggregate across all trajectories.

        Returns:
            Dict containing all aggregated benchmark results.

        Example:
            >>> runner = BenchmarkRunner()
            >>> runner.load_trajectories()
            >>> results = runner.run()
            >>> 'step_localization_accuracy' in results
            True
        """
        if not self.trajectories:
            print("ERROR: No trajectories loaded. Call load_trajectories() first.")
            return {}

        print(f"\nRunning benchmark on {len(self.trajectories)} trajectories...")
        print(f"Detector: {self.detector_fn.__name__}\n")

        # Aggregated collections across ALL trajectories
        all_predicted_steps = []    # step numbers predicted as hallucinated
        all_true_steps = []         # step numbers truly hallucinated
        all_total_steps = 0         # total step count
        all_predicted_types = []    # predicted hallucination types per step
        all_true_types = []         # true hallucination types per step
        all_latencies = []          # detection time per step
        self.per_trajectory_results = []

        for i, traj in enumerate(self.trajectories):
            traj_id = traj.get("trajectory_id", f"traj_{i+1:03d}")
            steps = traj.get("steps", [])
            total = len(steps)

            # Per-trajectory tracking
            traj_predicted = []
            traj_true = []

            for step in steps:
                step_num = step.get("step", 0)

                # Time the detection
                t0 = time.perf_counter()
                detection = self.detector_fn(step)
                t1 = time.perf_counter()
                latency_ms = (t1 - t0) * 1000.0
                all_latencies.append(latency_ms)

                # Store detection result in the step
                step["detection_result"] = detection

                # Collect predictions
                if detection.get("hallucination_detected", False):
                    traj_predicted.append(step_num)
                    all_predicted_steps.append(
                        all_total_steps + step_num  # global step index
                    )

                # Collect ground truth
                if step.get("ground_truth_label", False):
                    traj_true.append(step_num)
                    all_true_steps.append(
                        all_total_steps + step_num
                    )

                # Collect types for F1 computation
                all_predicted_types.append(
                    detection.get("hallucination_type_predicted")
                )
                all_true_types.append(
                    step.get("hallucination_type")
                )

            # Per-trajectory metrics
            traj_metrics = compute_all_metrics(
                predicted_steps=traj_predicted,
                true_steps=traj_true,
                total_steps=total,
            )
            traj_metrics["trajectory_id"] = traj_id
            traj_metrics["total_steps"] = total
            self.per_trajectory_results.append(traj_metrics)

            all_total_steps += total

            # Progress indicator
            if (i + 1) % 50 == 0 or i == len(self.trajectories) - 1:
                print(f"  [{i+1}/{len(self.trajectories)}] processed")

        # ── Aggregate metrics across all trajectories ──

        # Average per-trajectory step localization accuracy
        avg_sla = sum(
            r["step_localization_accuracy"]
            for r in self.per_trajectory_results
        ) / len(self.per_trajectory_results)

        avg_prec = sum(
            r["precision"] for r in self.per_trajectory_results
        ) / len(self.per_trajectory_results)

        avg_rec = sum(
            r["recall"] for r in self.per_trajectory_results
        ) / len(self.per_trajectory_results)

        avg_fpr = sum(
            r["false_positive_rate"] for r in self.per_trajectory_results
        ) / len(self.per_trajectory_results)

        # F1 per category (global)
        f1_cats = f1_per_category(all_predicted_types, all_true_types)

        # Latency stats
        latency_stats = average_latency_ms(all_latencies)

        # Precision@k (global)
        pak_results = {}
        for k in CONFIG.evaluation.top_k_values:
            pak = precision_at_k(all_predicted_steps, all_true_steps, k)
            pak_results.update(pak)

        # Compile final results
        self.results = {
            "benchmark_id": f"bench_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.datetime.now().isoformat(),
            "num_trajectories": len(self.trajectories),
            "total_steps": all_total_steps,
            "detector": self.detector_fn.__name__,

            # Core metrics (averages across trajectories)
            "step_localization_accuracy": round(avg_sla, 4),
            "precision": round(avg_prec, 4),
            "recall": round(avg_rec, 4),
            "false_positive_rate": round(avg_fpr, 4),

            # Precision@k
            **pak_results,

            # F1 per category
            "f1_per_category": f1_cats,
            "macro_f1": f1_cats["macro_avg"]["f1"],

            # Latency
            **latency_stats,

            # Baselines
            "agenthallu_baseline": CONFIG.evaluation.agenthallu_baseline,
            "agenthallu_tool_baseline": CONFIG.evaluation.agenthallu_tool_baseline,
            "delta_vs_baseline": round(
                avg_sla - CONFIG.evaluation.agenthallu_baseline, 4
            ),
        }

        print(f"\nBenchmark complete!")
        return self.results

    def print_results_table(self) -> str:
        """Generate and print a markdown-formatted results table.

        Returns:
            Markdown string of the results table.

        Example:
            >>> runner.print_results_table()
            | Metric | Value |
            ...
        """
        if not self.results:
            print("No results. Run benchmark first.")
            return ""

        r = self.results

        # Build markdown table
        lines = [
            "",
            "## AgentTrace Benchmark Results",
            "",
            f"**Benchmark ID:** {r['benchmark_id']}",
            f"**Timestamp:** {r['timestamp']}",
            f"**Trajectories:** {r['num_trajectories']}",
            f"**Total Steps:** {r['total_steps']}",
            f"**Detector:** {r['detector']}",
            "",
            "### Core Metrics",
            "",
            "| Metric | Value | AgentHallu SOTA |",
            "|---|---|---|",
            f"| Step Localization Accuracy | **{r['step_localization_accuracy']:.4f}** | {r['agenthallu_baseline']} |",
            f"| Precision | {r['precision']:.4f} | - |",
            f"| Recall | {r['recall']:.4f} | - |",
            f"| False Positive Rate | {r['false_positive_rate']:.4f} | - |",
            f"| Macro F1 | {r['macro_f1']:.4f} | - |",
            f"| Delta vs Baseline | **{r['delta_vs_baseline']:+.4f}** | - |",
            "",
            "### Precision@K",
            "",
            "| K | Precision |",
            "|---|---|",
        ]

        for k in CONFIG.evaluation.top_k_values:
            key = f"precision_at_{k}"
            val = r.get(key, 0.0)
            lines.append(f"| {k} | {val:.4f} |")

        lines.extend([
            "",
            "### F1 Per Hallucination Category",
            "",
            "| Category | Precision | Recall | F1 | Support |",
            "|---|---|---|---|---|",
        ])

        f1_data = r.get("f1_per_category", {})
        for cat in CONFIG.classifier.categories:
            if cat in f1_data:
                cd = f1_data[cat]
                lines.append(
                    f"| {cat} | {cd['precision']:.4f} | {cd['recall']:.4f} "
                    f"| {cd['f1']:.4f} | {cd['support']} |"
                )

        lines.extend([
            "",
            "### Latency",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Avg Latency | {r['avg_latency_ms']:.2f} ms |",
            f"| P95 Latency | {r['p95_latency_ms']:.2f} ms |",
            f"| Max Latency | {r['max_latency_ms']:.2f} ms |",
            "",
        ])

        table = "\n".join(lines)
        print(table)
        return table

    def save_results(self, filepath: Optional[str] = None) -> str:
        """Save benchmark results to JSON file.

        Args:
            filepath: Custom output path. Defaults to results dir.

        Returns:
            Path to the saved file.

        Example:
            >>> path = runner.save_results()
        """
        if not self.results:
            print("No results to save. Run benchmark first.")
            return ""

        if filepath is None:
            filepath = os.path.join(
                CONFIG.paths.results_dir,
                "benchmark_results.json",
            )

        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save main results
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {filepath}")

            # Also save per-trajectory results
            per_traj_path = filepath.replace(".json", "_per_trajectory.json")
            with open(per_traj_path, "w", encoding="utf-8") as f:
                json.dump(
                    self.per_trajectory_results, f,
                    indent=2, ensure_ascii=False,
                )
            print(f"Per-trajectory results saved to {per_traj_path}")

        except IOError as e:
            print(f"SAVE ERROR: {e}")

        return filepath

    def log_to_wandb(self) -> None:
        """Log benchmark results to Weights & Biases.

        Skips silently if WandB is disabled in config or not installed.

        Example:
            >>> runner.log_to_wandb()
        """
        if not CONFIG.wandb.enabled:
            print("WandB logging disabled in config. Skipping.")
            return

        try:
            import wandb
        except ImportError:
            print("WandB not installed. Skipping. Run: pip install wandb")
            return

        if not self.results:
            print("No results to log. Run benchmark first.")
            return

        try:
            # Initialize WandB run
            run = wandb.init(
                project=CONFIG.wandb.project_name,
                entity=CONFIG.wandb.entity,
                tags=CONFIG.wandb.tags,
                config={
                    "detector": self.results.get("detector", "unknown"),
                    "num_trajectories": self.results.get("num_trajectories", 0),
                    "total_steps": self.results.get("total_steps", 0),
                },
                name=self.results.get("benchmark_id", "benchmark"),
            )

            # Log scalar metrics
            wandb.log({
                "step_localization_accuracy": self.results["step_localization_accuracy"],
                "precision": self.results["precision"],
                "recall": self.results["recall"],
                "false_positive_rate": self.results["false_positive_rate"],
                "macro_f1": self.results["macro_f1"],
                "avg_latency_ms": self.results["avg_latency_ms"],
                "delta_vs_baseline": self.results["delta_vs_baseline"],
            })

            # Log F1 per category
            f1_data = self.results.get("f1_per_category", {})
            for cat in CONFIG.classifier.categories:
                if cat in f1_data:
                    wandb.log({f"f1_{cat}": f1_data[cat]["f1"]})

            wandb.finish()
            print("Results logged to WandB successfully.")

        except Exception as e:
            print(f"WandB logging error: {e}")


# ════════════════════════════════════════════════════════════
# TEST BLOCK
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("AgentTrace - Benchmark Runner Test")
    print("=" * 60)

    CONFIG.setup()

    # Disable WandB for local test
    CONFIG.wandb.enabled = False

    runner = BenchmarkRunner()

    # Load trajectories
    count = runner.load_trajectories()
    if count == 0:
        print("\nNo trajectories found. Generate them first:")
        print("  python data/synthetic_generator.py")
        sys.exit(1)

    # Run benchmark
    results = runner.run()

    # Print results table
    runner.print_results_table()

    # Save results
    runner.save_results()

    # Verdict
    print("\n" + "=" * 60)
    delta = results.get("delta_vs_baseline", 0)
    if delta > 0:
        print(f"RESULT: ABOVE AgentHallu baseline by {delta:+.4f}")
    else:
        print(f"RESULT: BELOW AgentHallu baseline by {delta:+.4f}")
    print("(Using mock detector — real results after Member 2 integration)")
    print("=" * 60)
