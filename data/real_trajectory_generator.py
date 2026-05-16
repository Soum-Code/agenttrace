import os
import json
import uuid
import sys
from datetime import datetime, timezone
from typing import Any, List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracer.step_logger import StepLogger
from config import PathConfig

class RealTrajectoryGenerator:
    """
    Captures live execution trajectories from a real LLM agent and formats 
    them to match the AgentTrace schema. This enables evaluating the pipeline 
    on real-world live interaction data rather than purely synthetic sets.
    """
    def __init__(self):
        self.paths = PathConfig()
        self.logger = StepLogger(session_id=str(uuid.uuid4()))
        self.trajectories = []
        
    def add_live_trajectory(self, task: str, steps_data: List[Dict[str, Any]], final_answer: str, final_answer_correct: bool = True):
        """
        Manually add a trajectory collected from a live agent run.
        """
        formatted_steps = []
        for i, step in enumerate(steps_data):
            formatted_step = {
                "step": i,
                "action": step.get("action", "unknown_tool"),
                "tool_input": step.get("tool_input", ""),
                "tool_output": step.get("tool_output", ""),
                "agent_reasoning": step.get("agent_reasoning", ""),
                "ground_truth_label": step.get("ground_truth_label", True),
                "hallucination_type": step.get("hallucination_type", None)
            }
            formatted_steps.append(formatted_step)
            
        trajectory = {
            "trajectory_id": str(uuid.uuid4()),
            "task": task,
            "total_steps": len(formatted_steps),
            "steps": formatted_steps,
            "final_answer": final_answer,
            "final_answer_correct": final_answer_correct
        }
        
        self.trajectories.append(trajectory)
        
    def export_to_json(self, filename="real_trajectories.json"):
        """Export collected live trajectories to the data directory."""
        out_path = os.path.join(self.paths.trajectory_dir, filename)
        
        # Load existing if any
        if os.path.exists(out_path):
            with open(out_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
            
        existing_data.extend(self.trajectories)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2)
            
        print(f"Exported {len(self.trajectories)} real trajectories. Total in file: {len(existing_data)}")
        self.trajectories = [] # Reset after export

if __name__ == "__main__":
    print("Real Trajectory Generator Initialized.")
    # Example usage:
    # generator = RealTrajectoryGenerator()
    # generator.add_live_trajectory(task="What is the weather?", steps_data=[...], final_answer="It's sunny.")
    # generator.export_to_json()
    print("Ready to capture live LLM agent interactions.")
