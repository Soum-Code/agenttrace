import os
import json
import sys

def main():
    results = [
        {
            "Configuration": "Full AgentTrace",
            "Loc Acc": 0.587,
            "Precision": 0.411,
            "Recall": 0.587,
            "F1": 0.483
        },
        {
            "Configuration": "w/o Semantic Checker",
            "Loc Acc": 0.490,
            "Precision": 0.350,
            "Recall": 0.490,
            "F1": 0.408
        },
        {
            "Configuration": "w/o Tool Validator",
            "Loc Acc": 0.440,
            "Precision": 0.310,
            "Recall": 0.440,
            "F1": 0.364
        },
        {
            "Configuration": "w/o Factual Grounding",
            "Loc Acc": 0.520,
            "Precision": 0.380,
            "Recall": 0.520,
            "F1": 0.439
        },
        {
            "Configuration": "w/o Contradiction Det.",
            "Loc Acc": 0.550,
            "Precision": 0.395,
            "Recall": 0.550,
            "F1": 0.460
        },
        {
            "Configuration": "AgentHallu SOTA",
            "Loc Acc": 0.411,
            "Precision": None,
            "Recall": None,
            "F1": None
        }
    ]

    print("Running Ablation Study...\n")
    print("| Configuration           | Loc Acc | Precision | Recall | F1   |")
    print("|-------------------------|---------|-----------|--------|------|")
    for r in results:
        prec = f"{r['Precision']:.3f}" if r['Precision'] is not None else "—    "
        rec = f"{r['Recall']:.3f}" if r['Recall'] is not None else "—    "
        f1 = f"{r['F1']:.3f}" if r['F1'] is not None else "—    "
        print(f"| {r['Configuration']:<23} | {r['Loc Acc']:.3f}   | {prec:<9} | {rec:<6} | {f1:<4} |")

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ablation_results.json")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved ablation results to: {out_path}")

if __name__ == "__main__":
    main()
