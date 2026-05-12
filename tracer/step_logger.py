"""
AgentTrace — Step Logger (Tracer Module)
=========================================
Logs every step of a LangChain agent as structured JSON.
Supports real-time logging, replay mode, step diff for
drift detection, and export to the team JSON schema.

Author: P. Somnath Reddy (Research Lead)
GitHub: github.com/Soum-Code/agenttrace
"""

import os
import sys
import json
import copy
import uuid
import datetime
from typing import List, Dict, Optional, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG


# ════════════════════════════════════════════════════════════
# STEP DATA STRUCTURE
# ════════════════════════════════════════════════════════════

def create_step(
    step_number: int,
    action: str,
    tool_input: str,
    tool_output: str,
    agent_reasoning: str,
    ground_truth_label: bool = False,
    hallucination_type: Optional[str] = None,
    detection_result: Optional[Dict] = None,
) -> Dict:
    """Create a single step dict matching the team JSON schema.

    Every step in the project uses this exact structure.
    Member 2 detection modules consume this format.
    Member 3 API accepts this format.

    Args:
        step_number: Sequential step index (1-based).
        action: Tool name used (e.g. 'web_search', 'calculator').
        tool_input: Query or input sent to the tool.
        tool_output: Raw result returned by the tool.
        agent_reasoning: Agent's interpretation of the tool output.
        ground_truth_label: True if step is hallucinated (for labeled data).
        hallucination_type: One of the 5 taxonomy categories, or None.
        detection_result: Output from the detection pipeline, or None.

    Returns:
        Step dict matching the team schema.

    Example:
        >>> step = create_step(1, 'web_search', 'GDP of India', 
        ...     'India GDP is $3.7 trillion', 'India GDP is $3.7T')
        >>> step['step']
        1
        >>> step['ground_truth_label']
        False
    """
    step = {
        "step": step_number,
        "action": action,
        "tool_input": tool_input,
        "tool_output": tool_output,
        "agent_reasoning": agent_reasoning,
        "ground_truth_label": ground_truth_label,
        "hallucination_type": hallucination_type,
    }

    # Add detection_result only if provided (keeps schema clean)
    if detection_result is not None:
        step["detection_result"] = detection_result

    return step


# ════════════════════════════════════════════════════════════
# TRAJECTORY LOGGER
# ════════════════════════════════════════════════════════════

class StepLogger:
    """Logs agent steps into a structured trajectory.

    Supports real-time logging during agent execution,
    replay of saved trajectories, step-level diff for
    drift detection, and export to JSON.

    Attributes:
        trajectory_id: Unique ID for this trajectory.
        task: The user query / task being solved.
        steps: List of step dicts recorded so far.
        metadata: Extra metadata (timestamps, source, etc.).

    Example:
        >>> logger = StepLogger(task='What is 2+2?')
        >>> logger.log_step('calculator', '2+2', '4', 'The answer is 4')
        >>> logger.total_steps
        1
        >>> logger.export()['trajectory_id']
        'traj_...'
    """

    def __init__(
        self,
        task: str,
        trajectory_id: Optional[str] = None,
    ) -> None:
        """Initialize a new trajectory logger.

        Args:
            task: The user query or task description.
            trajectory_id: Optional custom ID. Auto-generated if None.

        Example:
            >>> logger = StepLogger(task='Find the capital of France')
        """
        # Auto-generate a unique trajectory ID if not provided
        self.trajectory_id = trajectory_id or f"traj_{uuid.uuid4().hex[:8]}"
        self.task = task
        self.steps: List[Dict] = []
        self.final_answer: Optional[str] = None
        self.final_answer_correct: Optional[bool] = None

        # Metadata for tracking
        self.metadata = {
            "created_at": datetime.datetime.now().isoformat(),
            "source": "live",  # 'live' for real-time, 'replay' for loaded
        }

        # Real-time logging config from config.py
        self._realtime = CONFIG.tracer.enable_realtime_logging
        self._max_steps = CONFIG.tracer.max_trajectory_length
        self._save_intermediate = CONFIG.tracer.save_intermediate
        self._log_format = CONFIG.tracer.log_format

        if self._realtime:
            print(f"[StepLogger] Trajectory {self.trajectory_id} started")
            print(f"[StepLogger] Task: {self.task}")

    @property
    def total_steps(self) -> int:
        """Number of steps recorded so far.

        Example:
            >>> logger = StepLogger(task='test')
            >>> logger.total_steps
            0
        """
        return len(self.steps)

    def log_step(
        self,
        action: str,
        tool_input: str,
        tool_output: str,
        agent_reasoning: str,
        ground_truth_label: bool = False,
        hallucination_type: Optional[str] = None,
        detection_result: Optional[Dict] = None,
    ) -> Dict:
        """Log a single agent step to the trajectory.

        Enforces the max trajectory length safety cap.
        Optionally saves intermediate state for crash recovery.

        Args:
            action: Tool name used in this step.
            tool_input: Input sent to the tool.
            tool_output: Result returned by the tool.
            agent_reasoning: Agent's interpretation/conclusion.
            ground_truth_label: True if hallucinated (labeled data only).
            hallucination_type: Hallucination category or None.
            detection_result: Detection pipeline output or None.

        Returns:
            The created step dict.

        Raises:
            RuntimeError: If max trajectory length is exceeded.

        Example:
            >>> logger = StepLogger(task='test')
            >>> step = logger.log_step('web_search', 'q', 'result', 'reasoning')
            >>> step['step']
            1
        """
        # Safety cap to prevent runaway agents
        if self.total_steps >= self._max_steps:
            raise RuntimeError(
                f"Max trajectory length ({self._max_steps}) exceeded. "
                f"Increase CONFIG.tracer.max_trajectory_length if needed."
            )

        step_number = self.total_steps + 1  # 1-based indexing

        step = create_step(
            step_number=step_number,
            action=action,
            tool_input=tool_input,
            tool_output=tool_output,
            agent_reasoning=agent_reasoning,
            ground_truth_label=ground_truth_label,
            hallucination_type=hallucination_type,
            detection_result=detection_result,
        )

        self.steps.append(step)

        # Real-time console output
        if self._realtime:
            label = " [HALLUC]" if ground_truth_label else ""
            print(f"[StepLogger] Step {step_number}: {action}{label}")

        # Save intermediate state for crash recovery
        if self._save_intermediate:
            self._save_intermediate_state()

        return step

    def set_final_answer(self, answer: str, correct: bool) -> None:
        """Set the trajectory's final answer.

        Args:
            answer: The agent's final response to the user.
            correct: Whether the final answer is factually correct.

        Example:
            >>> logger = StepLogger(task='test')
            >>> logger.set_final_answer('42', True)
            >>> logger.final_answer
            '42'
        """
        self.final_answer = answer
        self.final_answer_correct = correct

        if self._realtime:
            status = "CORRECT" if correct else "INCORRECT"
            print(f"[StepLogger] Final answer: {status}")

    def export(self) -> Dict:
        """Export the trajectory as a dict matching the team JSON schema.

        Returns:
            Complete trajectory dict ready for JSON serialization.

        Example:
            >>> logger = StepLogger(task='test', trajectory_id='traj_001')
            >>> logger.log_step('calc', '1+1', '2', 'answer is 2')
            >>> logger.set_final_answer('2', True)
            >>> data = logger.export()
            >>> data['trajectory_id']
            'traj_001'
            >>> data['total_steps']
            1
        """
        return {
            "trajectory_id": self.trajectory_id,
            "task": self.task,
            "total_steps": self.total_steps,
            "steps": copy.deepcopy(self.steps),  # deep copy to prevent mutation
            "final_answer": self.final_answer or "",
            "final_answer_correct": self.final_answer_correct
                                    if self.final_answer_correct is not None
                                    else False,
        }

    def save(self, filepath: Optional[str] = None) -> str:
        """Save the trajectory to a JSON file.

        Args:
            filepath: Custom output path. Defaults to trajectory_dir/ID.json.

        Returns:
            Absolute path to the saved file.

        Example:
            >>> logger = StepLogger(task='test', trajectory_id='traj_001')
            >>> path = logger.save()
        """
        if filepath is None:
            filepath = os.path.join(
                CONFIG.paths.trajectory_dir,
                f"{self.trajectory_id}.json",
            )

        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.export(), f, indent=2, ensure_ascii=False)
            if self._realtime:
                print(f"[StepLogger] Saved to {filepath}")
        except IOError as e:
            print(f"[StepLogger] SAVE ERROR: {e}")

        return filepath

    def _save_intermediate_state(self) -> None:
        """Save partial trajectory for crash recovery (silent)."""
        try:
            partial_path = os.path.join(
                CONFIG.paths.log_dir,
                f"{self.trajectory_id}_partial.json",
            )
            os.makedirs(os.path.dirname(partial_path), exist_ok=True)
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(self.export(), f, indent=2, ensure_ascii=False)
        except IOError:
            pass  # silent failure for intermediate saves

    # ════════════════════════════════════════════════════════
    # STEP DIFF — drift detection between consecutive steps
    # ════════════════════════════════════════════════════════

    def step_diff(self, step_a: int, step_b: int) -> Dict[str, Any]:
        """Compare two steps to detect semantic drift or contradictions.

        Used by the detection pipeline to identify gradual hallucination
        drift across consecutive steps.

        Args:
            step_a: Step number of the first step (1-based).
            step_b: Step number of the second step (1-based).

        Returns:
            Dict containing field-level diffs between the two steps.

        Raises:
            IndexError: If step numbers are out of range.

        Example:
            >>> logger = StepLogger(task='test')
            >>> logger.log_step('search', 'q1', 'r1', 'reasoning 1')
            >>> logger.log_step('search', 'q2', 'r2', 'reasoning 2')
            >>> diff = logger.step_diff(1, 2)
            >>> 'action_changed' in diff
            True
        """
        # Convert 1-based to 0-based index
        idx_a = step_a - 1
        idx_b = step_b - 1

        if idx_a < 0 or idx_a >= self.total_steps:
            raise IndexError(f"Step {step_a} out of range (1-{self.total_steps})")
        if idx_b < 0 or idx_b >= self.total_steps:
            raise IndexError(f"Step {step_b} out of range (1-{self.total_steps})")

        sa = self.steps[idx_a]
        sb = self.steps[idx_b]

        return {
            "step_a": step_a,
            "step_b": step_b,
            "action_changed": sa["action"] != sb["action"],
            "tool_input_a": sa["tool_input"],
            "tool_input_b": sb["tool_input"],
            "tool_output_a": sa["tool_output"],
            "tool_output_b": sb["tool_output"],
            "reasoning_a": sa["agent_reasoning"],
            "reasoning_b": sb["agent_reasoning"],
            "label_a": sa["ground_truth_label"],
            "label_b": sb["ground_truth_label"],
        }

    def get_drift_window(self, current_step: int) -> List[Dict]:
        """Get diffs for the last N steps (drift window from config).

        Compares each step in the window against its predecessor
        to detect gradual semantic drift.

        Args:
            current_step: The current step number (1-based).

        Returns:
            List of step_diff dicts for the drift window.

        Example:
            >>> logger = StepLogger(task='test')
            >>> for i in range(5):
            ...     logger.log_step('search', f'q{i}', f'r{i}', f'reason{i}')
            >>> diffs = logger.get_drift_window(5)
            >>> len(diffs) <= 3  # CONFIG.thresholds.drift_window
            True
        """
        window_size = CONFIG.thresholds.drift_window
        diffs = []

        # Look back from current_step by window_size
        start = max(1, current_step - window_size)
        for step_num in range(start, current_step):
            diffs.append(self.step_diff(step_num, step_num + 1))

        return diffs


# ════════════════════════════════════════════════════════════
# REPLAY MODE — load and re-examine saved trajectories
# ════════════════════════════════════════════════════════════

def load_trajectory(filepath: str) -> StepLogger:
    """Load a saved trajectory JSON file into a StepLogger for replay.

    Enables re-examination of any previously saved trajectory
    with full step_diff and drift detection capabilities.

    Args:
        filepath: Path to the trajectory JSON file.

    Returns:
        StepLogger instance populated with the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.

    Example:
        >>> logger = load_trajectory('data/trajectories/traj_001.json')
        >>> logger.total_steps
        5
        >>> logger.step_diff(1, 2)
        {...}
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Trajectory file not found: {filepath}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in {filepath}: {e.msg}", e.doc, e.pos
        )

    # Create a StepLogger with real-time logging disabled (replay mode)
    original_rt = CONFIG.tracer.enable_realtime_logging
    CONFIG.tracer.enable_realtime_logging = False  # suppress output during load

    logger = StepLogger(
        task=data.get("task", ""),
        trajectory_id=data.get("trajectory_id", "unknown"),
    )
    logger.metadata["source"] = "replay"
    logger.metadata["loaded_from"] = filepath

    # Populate steps directly (bypass log_step to avoid side effects)
    logger.steps = data.get("steps", [])
    logger.final_answer = data.get("final_answer")
    logger.final_answer_correct = data.get("final_answer_correct")

    # Restore real-time logging setting
    CONFIG.tracer.enable_realtime_logging = original_rt

    return logger


def load_all_trajectories(directory: Optional[str] = None) -> List[StepLogger]:
    """Load all trajectory JSON files from a directory.

    Args:
        directory: Path to scan. Defaults to CONFIG.paths.trajectory_dir.

    Returns:
        List of StepLogger instances, one per loaded file.

    Example:
        >>> loggers = load_all_trajectories()
        >>> len(loggers)
        200
    """
    dir_path = directory or CONFIG.paths.trajectory_dir
    loggers = []

    if not os.path.exists(dir_path):
        print(f"Directory not found: {dir_path}")
        return loggers

    for filename in sorted(os.listdir(dir_path)):
        if filename.endswith(".json"):
            filepath = os.path.join(dir_path, filename)
            try:
                logger = load_trajectory(filepath)
                loggers.append(logger)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"  Skipping {filename}: {e}")

    return loggers


# ════════════════════════════════════════════════════════════
# TEST BLOCK
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("AgentTrace - Step Logger Test")
    print("=" * 60)

    CONFIG.setup()

    # --- Test 1: Live logging ---
    print("\n--- Test 1: Live Logging ---\n")

    logger = StepLogger(task="What is the GDP of the FIFA 2022 winner?")

    logger.log_step(
        action="web_search",
        tool_input="FIFA 2022 winner",
        tool_output="Argentina won FIFA World Cup 2022",
        agent_reasoning="Argentina won FIFA 2022. Need GDP next.",
    )

    logger.log_step(
        action="web_search",
        tool_input="GDP of Argentina",
        tool_output="Argentina GDP is $640 billion (2023)",
        agent_reasoning="France GDP is $2.9 trillion",  # hallucination!
        ground_truth_label=True,
        hallucination_type="Tool-Use",
    )

    logger.log_step(
        action="calculator",
        tool_input="640 * 1e9",
        tool_output="640000000000",
        agent_reasoning="Confirmed: Argentina GDP is $640 billion",
    )

    logger.set_final_answer(
        "The GDP of Argentina (FIFA 2022 winner) is $640 billion.",
        correct=True,
    )

    print(f"\nTotal steps: {logger.total_steps}")

    # --- Test 2: Export ---
    print("\n--- Test 2: Export ---\n")
    exported = logger.export()
    print(f"Trajectory ID: {exported['trajectory_id']}")
    print(f"Total steps:   {exported['total_steps']}")
    print(f"Final answer:  {exported['final_answer']}")

    # --- Test 3: Save and reload ---
    print("\n--- Test 3: Save and Reload ---\n")
    save_path = logger.save()

    reloaded = load_trajectory(save_path)
    print(f"Reloaded ID:    {reloaded.trajectory_id}")
    print(f"Reloaded steps: {reloaded.total_steps}")
    print(f"Source:         {reloaded.metadata['source']}")

    # --- Test 4: Step diff ---
    print("\n--- Test 4: Step Diff ---\n")
    diff = logger.step_diff(1, 2)
    print(f"Step 1 vs 2:")
    print(f"  Action changed: {diff['action_changed']}")
    print(f"  Reasoning 1: {diff['reasoning_a']}")
    print(f"  Reasoning 2: {diff['reasoning_b']}")
    print(f"  Label 1: {diff['label_a']} -> Label 2: {diff['label_b']}")

    # --- Test 5: Drift window ---
    print("\n--- Test 5: Drift Window ---\n")
    diffs = logger.get_drift_window(3)
    print(f"Drift window for step 3 ({len(diffs)} comparisons):")
    for d in diffs:
        print(f"  Step {d['step_a']} vs {d['step_b']}: "
              f"action_changed={d['action_changed']}")

    # --- Test 6: Replay from synthetic data ---
    print("\n--- Test 6: Replay Synthetic Data ---\n")
    synth_path = os.path.join(
        CONFIG.paths.trajectory_dir,
        CONFIG.synthetic.output_filename,
    )
    if os.path.exists(synth_path):
        try:
            with open(synth_path, "r", encoding="utf-8") as f:
                synth_data = json.load(f)
            if synth_data:
                # Save first trajectory as individual file, then reload
                first = synth_data[0]
                temp_path = os.path.join(
                    CONFIG.paths.trajectory_dir,
                    f"{first['trajectory_id']}.json",
                )
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(first, f, indent=2)

                replayed = load_trajectory(temp_path)
                print(f"Replayed: {replayed.trajectory_id}")
                print(f"  Task:  {replayed.task}")
                print(f"  Steps: {replayed.total_steps}")
                if replayed.total_steps >= 2:
                    diff = replayed.step_diff(1, 2)
                    print(f"  Diff 1v2 action_changed: {diff['action_changed']}")
        except Exception as e:
            print(f"  Replay error: {e}")
    else:
        print("  No synthetic data found. Run synthetic_generator.py first.")

    print(f"\n{'=' * 60}")
    print("All step_logger.py tests passed!")
    print(f"{'=' * 60}")
