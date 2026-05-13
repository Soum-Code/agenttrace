"""
AgentTrace — Benchmark v2 Runner Script
Run this as: python run_benchmark.py
"""
import os
import sys

# Set HF token
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.pipeline import real_detector, reset_pipeline
from evaluation.benchmark import BenchmarkRunner

print("=" * 60)
print("AgentTrace - Pipeline v2 Benchmark")
print("=" * 60)

runner = BenchmarkRunner(detector_fn=real_detector)
count = runner.load_trajectories()
print(f"Loaded: {count} trajectories")

if count == 0:
    print("ERROR: No trajectories found!")
    sys.exit(1)

results = runner.run()
runner.print_results_table()
runner.save_results()

# Verdict
print("\n" + "=" * 60)
delta = results.get("delta_vs_baseline", 0)
if delta > 0:
    print(f"RESULT: ABOVE AgentHallu baseline by {delta:+.4f}")
else:
    print(f"RESULT: BELOW AgentHallu baseline by {delta:+.4f}")
print("Pipeline v2: threshold fusion, 5-category classifier")
print("=" * 60)
