import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import sys
import torch
torch.set_num_threads(1)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from detection.pipeline import DetectionPipeline
from evaluation.benchmark import BenchmarkRunner

def main():
    print("Running Combinatorial Ablation Study on 200 Trajectories...\n")
    
    # 7 Configurations:
    # 1. Semantic-only (active_modules=["semantic"])
    # 2. NLI-only (active_modules=["factual"])
    # 3. Tool-validator-only (active_modules=["tool"])
    # 4. Layer 1 SLM ensemble (all 4 modules, enable_layer2=False, enable_layer3=False)
    # 5. Layer 1 + Layer 2 Llama (enable_layer3=False)
    # 6. Layer 1 + Layer 3 Nemotron (enable_layer2=False)
    # 7. Full 3-layer cascade (all enabled)
    
    configs = [
        {
            "name": "Semantic-only",
            "active_modules": ["semantic"],
            "enable_layer2": False,
            "enable_layer3": False
        },
        {
            "name": "NLI-only",
            "active_modules": ["factual"],
            "enable_layer2": False,
            "enable_layer3": False
        },
        {
            "name": "Tool-validator-only",
            "active_modules": ["tool"],
            "enable_layer2": False,
            "enable_layer3": False
        },
        {
            "name": "Layer 1 SLM Ensemble",
            "active_modules": None,
            "enable_layer2": False,
            "enable_layer3": False
        },
        {
            "name": "Layer 1 + Layer 2 Llama",
            "active_modules": None,
            "enable_layer2": True,
            "enable_layer3": False
        },
        {
            "name": "Layer 1 + Layer 3 Nemotron",
            "active_modules": None,
            "enable_layer2": False,
            "enable_layer3": True
        },
        {
            "name": "Full 3-Layer Cascade",
            "active_modules": None,
            "enable_layer2": True,
            "enable_layer3": True
        }
    ]
    
    results = {}
    
    for config_info in configs:
        name = config_info["name"]
        print(f"Evaluating Config: {name} ...")
        
        # Instantiate pipeline for this configuration
        pipeline = DetectionPipeline(
            enable_layer2=config_info["enable_layer2"],
            enable_layer3=config_info["enable_layer3"],
            active_modules=config_info["active_modules"]
        )
        
        # Initialize benchmark runner with this pipeline's detect function
        runner = BenchmarkRunner(detector_fn=pipeline.detect)
        runner.load_trajectories()
        res = runner.run()
        
        # Save key metrics
        results[name] = {
            "step_localization_accuracy": res.get("step_localization_accuracy", 0.0),
            "precision": res.get("precision", 0.0),
            "recall": res.get("recall", 0.0),
            "macro_f1": res.get("macro_f1", 0.0),
            "avg_latency_ms": res.get("avg_latency_ms", 0.0)
        }
        print(f"Config {name} step_localization_accuracy: {results[name]['step_localization_accuracy']:.4f}\n")
        
    # Print comparison table
    print("\n" + "="*80)
    print("Combinatorial Ablation Study Results")
    print("="*80)
    print("| Configuration | Loc Acc | Precision | Recall | F1 | Avg Latency (ms) |")
    print("|---|---|---|---|---|---|")
    for name, metrics in results.items():
        print(f"| {name:<26} | {metrics['step_localization_accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['macro_f1']:.4f} | {metrics['avg_latency_ms']:.2f} |")
    print("="*80)
    
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ablation_results.json")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved ablation results to: {out_path}")

if __name__ == "__main__":
    main()
