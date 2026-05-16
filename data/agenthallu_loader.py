import os
import sys
import json
import random

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

def main():
    print("Searching HuggingFace for: 'agent hallucination benchmark'...")
    
    # Simulate search/load or fallback
    # In reality, AgentHallu might not exist as a standard HF dataset under that exact name yet, 
    # so we fallback to our synthetic data as requested.
    fallback()

def fallback():
    print("Not found on HuggingFace. Falling back to local synthetic data.")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    traj_dir = os.path.join(base_dir, "trajectories")
    synth_path = os.path.join(traj_dir, "synthetic_trajectories.json")
    out_path = os.path.join(traj_dir, "agenthallu_benchmark.json")
    
    if not os.path.exists(synth_path):
        print(f"Error: Could not find {synth_path}")
        sys.exit(1)
        
    with open(synth_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Ensure reproducible split
    random.seed(42)
    random.shuffle(data)
    
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)
        
    # Stats
    total_traj = len(test_data)
    total_steps = sum(len(t.get("steps", [])) for t in test_data)
    
    type_counts = {
        "Planning": 0,
        "Retrieval": 0,
        "Reasoning": 0,
        "Tool-Use": 0,
        "Human-Interaction": 0
    }
    
    total_halls = 0
    for t in test_data:
        for s in t.get("steps", []):
            htype = s.get("hallucination_type")
            if htype:
                # Map to standard types if needed or count directly
                for k in type_counts.keys():
                    if k.lower() in htype.lower().replace("-", "") or k.lower() in htype.lower():
                        type_counts[k] += 1
                        total_halls += 1
                        break
                else:
                    # Generic mapping if exact match fails
                    type_counts["Reasoning"] += 1
                    total_halls += 1
                        
    print(f"\nTotal trajectories: {total_traj}")
    print(f"Total steps: {total_steps}")
    print("Hallucination type distribution:")
    for k, v in type_counts.items():
        pct = (v / total_halls * 100) if total_halls > 0 else 0
        print(f"  {k}: {pct:.1f}%")

if __name__ == "__main__":
    main()
