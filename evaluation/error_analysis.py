import os
import json
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from detection.pipeline import DetectionPipeline
from evaluation.benchmark import BenchmarkRunner

def analyze_errors():
    print("Running Qualitative Error Analysis on Synthetic Trajectories...", flush=True)
    
    # Use Layer 1 only — no OpenRouter API key needed for error analysis
    pipeline = DetectionPipeline(enable_layer2=False, enable_layer3=False)
    runner = BenchmarkRunner(detector_fn=pipeline.detect)
    runner.load_trajectories()
    
    if not runner.trajectories:
        print("Error: No trajectories loaded.")
        return
        
    errors = []
    
    # Process steps and find misclassifications
    total_trajs = len(runner.trajectories)
    for i, traj in enumerate(runner.trajectories):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Processing trajectory {i+1}/{total_trajs}...", flush=True)
        pipeline.reset_history()
        traj_id = traj.get("trajectory_id", f"traj_{i+1:03d}")
        
        for step in traj.get("steps", []):
            step_num = step.get("step", 0)
            res = pipeline.detect(step)
            
            pred_detected = res.get("hallucination_detected", False)
            gt_detected = step.get("ground_truth_label", False)
            
            if pred_detected != gt_detected:
                # We have a misclassification!
                error_type = "FP" if pred_detected else "FN"
                conf = res.get("confidence", 0.0)
                signals = res.get("detection_signals", {})
                
                # Rule-based logic to determine dominant signal and why wrong
                dominant_signal = "None"
                why_wrong = "Unknown"
                
                if error_type == "FP":
                    # False Positive: predicted hallucination but actually clean
                    sem_sim = signals.get("semantic_similarity")
                    nli = signals.get("nli_score")
                    tcm = signals.get("tool_claim_match")
                    contra = signals.get("contradiction_with_prev")
                    
                    # Determine dominant signal
                    max_signal_val = -1
                    if sem_sim is not None and (1.0 - sem_sim) > max_signal_val:
                        max_signal_val = 1.0 - sem_sim
                        dominant_signal = f"Low Semantic Similarity ({sem_sim:.2f})"
                    if nli is not None and nli > max_signal_val:
                        max_signal_val = nli
                        dominant_signal = f"High NLI Contradiction ({nli:.2f})"
                    if tcm is False and 1.0 > max_signal_val:
                        max_signal_val = 1.0
                        dominant_signal = "Tool Claim Mismatch"
                    if contra is True and 1.0 > max_signal_val:
                        max_signal_val = 1.0
                        dominant_signal = "Contradiction with Previous Steps"
                        
                    # Explanation
                    if nli is not None and nli > 0.75:
                        why_wrong = "NLI model flagged benign semantic mismatch as factual contradiction."
                    elif sem_sim is not None and sem_sim < 0.65:
                        why_wrong = "Semantic similarity checker flagged synonym-rich correct reasoning as semantic drift."
                    elif tcm is False:
                        why_wrong = "Tool validator incorrectly identified formatting differences as a claim mismatch."
                    else:
                        why_wrong = "Aggressive signal fusion threshold triggered a false alarm."
                        
                else:
                    # False Negative: predicted clean but actually hallucinated
                    # Explain why the detector missed it
                    sem_sim = signals.get("semantic_similarity")
                    nli = signals.get("nli_score")
                    
                    dominant_signal = "None"
                    if sem_sim is not None and sem_sim > 0.8:
                        dominant_signal = f"High Semantic Similarity ({sem_sim:.2f})"
                    elif nli is not None and nli < 0.3:
                        dominant_signal = f"Low NLI Contradiction ({nli:.2f})"
                        
                    why_wrong = "The hallucination was linguistically subtle or used matching terminology, bypassing the SLM ensemble."
                    
                errors.append({
                    "trajectory_id": traj_id,
                    "step": step_num,
                    "error_type": error_type,
                    "action": step.get("action", ""),
                    "agent_reasoning": step.get("agent_reasoning", ""),
                    "tool_output": step.get("tool_output", ""),
                    "ground_truth_label": gt_detected,
                    "confidence": conf,
                    "error_magnitude": abs(conf - (1 if gt_detected else 0)),
                    "dominant_signal": dominant_signal,
                    "why_wrong": why_wrong
                })

    # Sort errors by magnitude (highest confidence wrong predictions first)
    errors.sort(key=lambda x: x["error_magnitude"], reverse=True)
    
    # Print top 10 errors
    print("\n--- Top 10 Highest-Confidence Wrong Predictions ---")
    for idx, err in enumerate(errors[:10]):
        print(f"{idx+1}. Traj: {err['trajectory_id']}, Step: {err['step']}, Type: {err['error_type']}, "
              f"Conf: {err['confidence']:.4f}, Magnitude: {err['error_magnitude']:.4f}")
        print(f"   Reasoning: {err['agent_reasoning'][:120]}...")
        print(f"   Dominant Signal: {err['dominant_signal']}")
        print(f"   Why Wrong: {err['why_wrong']}")
        print()

    # Select 6 representative cases (3 FPs, 3 FNs)
    fps = [e for e in errors if e["error_type"] == "FP"]
    fns = [e for e in errors if e["error_type"] == "FN"]
    
    representative_cases = fps[:3] + fns[:3]
    
    # Generate LaTeX table
    tex = r"""\begin{table*}[t]
\centering
\caption{Qualitative Error Analysis: Representative False Positives and False Negatives}
\begin{tabular}{lp{6cm}lp{3cm}p{5cm}}
\toprule
ID & Step Reasoning & Type & Dominant Signal & Error Cause (Rule-Based) \\
\midrule
"""
    for err in representative_cases:
        clean_reasoning = err["agent_reasoning"].replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")
        if len(clean_reasoning) > 100:
            clean_reasoning = clean_reasoning[:97] + "..."
        row_id = err['trajectory_id'] + r"\_" + str(err['step'])
        row_signal = err['dominant_signal'].replace("_", r"\_")
        row_why = err['why_wrong'].replace("_", r"\_")
        tex += f"{row_id} & {clean_reasoning} & {err['error_type']} & {row_signal} & {row_why} \\\\\n"
        
    tex += r"""\bottomrule
\end{tabular}
\end{table*}"""

    # Ensure output directories exist
    paper_dir = os.path.join(CONFIG.paths.project_root, "paper", "figures")
    os.makedirs(paper_dir, exist_ok=True)
    
    tex_path = os.path.join(paper_dir, "error_analysis_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"LaTeX error table saved to: {tex_path}")
    
    # Save JSON results
    out_dir = os.path.join(CONFIG.paths.project_root, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "error_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)
    print(f"Error analysis JSON saved to: {json_path}")
    
if __name__ == "__main__":
    analyze_errors()
