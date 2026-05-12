"""
AgentTrace — Synthetic Trajectory Generator
=============================================
Generates labeled agent trajectories using OpenRouter API
(OpenAI-compatible). Falls back to direct Gemini API if
OpenRouter key is not set.

Each trajectory contains 3-7 steps with 1-2 deliberate
hallucinations injected, matching the team JSON schema.

Author: P. Somnath Reddy (Research Lead)
GitHub: github.com/Soum-Code/agenttrace
"""

import os
import sys
import json
import time
import random
import re
from typing import List, Dict, Optional

# Add project root to path so config can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

try:
    from openai import OpenAI  # OpenRouter uses OpenAI-compatible SDK
except ImportError:
    print("ERROR: openai package not installed.")
    print("Run: pip install openai")
    sys.exit(1)


# ════════════════════════════════════════════════════════════
# PROMPT TEMPLATE
# ════════════════════════════════════════════════════════════

TRAJECTORY_PROMPT = """You are a synthetic data generator for AI safety research.

Generate a realistic multi-step LLM agent trajectory that solves a complex task.

REQUIREMENTS:
- The trajectory must have exactly {num_steps} steps.
- The agent has these tools: {tools}
- Inject exactly {num_hallucinations} hallucination(s) at random step(s).
- Each hallucination must be one of these types: {hallucination_types}
- Non-hallucinated steps must be factually correct and consistent.
- Hallucinated steps must have a mismatch between tool_output and agent_reasoning,
  OR the agent must fabricate information, OR misuse a tool.

HALLUCINATION TYPE DEFINITIONS:
- Planning: Agent chooses wrong tool or wrong sequence of actions.
- Retrieval: Agent misquotes or fabricates retrieved information.
- Reasoning: Agent draws incorrect conclusion from correct information.
- Tool-Use: Agent ignores or contradicts the tool output.
- Human-Interaction: Agent misinterprets user intent or adds unsolicited info.

OUTPUT FORMAT (strict JSON, no markdown, no code fences):
{{
  "task": "A complex multi-step question requiring tool use",
  "steps": [
    {{
      "step": 1,
      "action": "tool_name",
      "tool_input": "query sent to the tool",
      "tool_output": "factual result from the tool",
      "agent_reasoning": "what the agent concludes from this step",
      "ground_truth_label": false,
      "hallucination_type": null
    }}
  ],
  "final_answer": "The agent's final response to the user",
  "final_answer_correct": true
}}

RULES:
- ground_truth_label is true if the step contains a hallucination, false otherwise.
- hallucination_type is null for non-hallucinated steps.
- final_answer_correct is false if any hallucination corrupts the final answer.
- Make tasks diverse: geography, science, math, history, current events, finance.
- Make hallucinations subtle and realistic, not obviously wrong.

Generate the trajectory now. Output ONLY valid JSON."""


# ════════════════════════════════════════════════════════════
# CORE GENERATOR CLASS
# ════════════════════════════════════════════════════════════

class SyntheticTrajectoryGenerator:
    """Generates labeled synthetic agent trajectories via OpenRouter API.

    Uses the OpenAI-compatible SDK to call models through OpenRouter.
    Falls back to direct Gemini API if OpenRouter key is not set.

    Attributes:
        client: OpenAI client configured for OpenRouter.
        model_name: Model identifier on OpenRouter.
        config: SyntheticDataConfig from config.py.

    Example:
        >>> gen = SyntheticTrajectoryGenerator()
        >>> trajectory = gen.generate_single_trajectory('traj_001')
        >>> print(trajectory['trajectory_id'])
        traj_001
    """

    def __init__(self) -> None:
        """Initialize the generator with OpenRouter API credentials.

        Raises:
            ValueError: If OPENROUTER_API_KEY env var is not set.
        """
        api_key = CONFIG.openrouter.api_key
        if not api_key:
            raise ValueError(
                f"Set {CONFIG.openrouter.api_key_env_var} environment variable. "
                f"Example: export OPENROUTER_API_KEY='sk-or-v1-...'"
            )

        # Initialize OpenAI client pointing to OpenRouter
        self.client = OpenAI(
            base_url=CONFIG.openrouter.base_url,
            api_key=api_key,
        )

        self.model_name = CONFIG.openrouter.model_name
        self.config = CONFIG.synthetic
        self.rate_limit = CONFIG.openrouter.requests_per_minute

        # Seed for reproducibility
        random.seed(CONFIG.training.seed)

    def _build_prompt(self, num_steps: int, num_hallucinations: int) -> str:
        """Build the generation prompt with specific step/hallucination counts.

        Args:
            num_steps: Number of steps for this trajectory.
            num_hallucinations: Number of hallucinations to inject.

        Returns:
            Formatted prompt string.
        """
        return TRAJECTORY_PROMPT.format(
            num_steps=num_steps,
            tools=", ".join(self.config.available_tools),
            num_hallucinations=num_hallucinations,
            hallucination_types=", ".join(self.config.hallucination_types),
        )

    def _parse_response(self, response_text: str) -> Optional[Dict]:
        """Parse and validate the LLM response as trajectory JSON.

        Handles markdown fences, extra text, incomplete JSON.

        Args:
            response_text: Raw text from API response.

        Returns:
            Parsed trajectory dict if valid, None if parsing fails.
        """
        cleaned = response_text.strip()
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)  # opening fence
        cleaned = re.sub(r'\s*```$', '', cleaned)            # closing fence
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', cleaned)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            else:
                return None

        # Validate required top-level keys
        required_keys = {"task", "steps", "final_answer", "final_answer_correct"}
        if not required_keys.issubset(data.keys()):
            missing = required_keys - data.keys()
            print(f"  WARNING: Missing keys: {missing}")
            return None

        # Validate each step
        step_keys = {
            "step", "action", "tool_input", "tool_output",
            "agent_reasoning", "ground_truth_label", "hallucination_type"
        }
        for i, step in enumerate(data["steps"]):
            if not step_keys.issubset(step.keys()):
                missing = step_keys - step.keys()
                print(f"  WARNING: Step {i+1} missing keys: {missing}")
                return None

        return data

    def generate_single_trajectory(self, trajectory_id: str) -> Optional[Dict]:
        """Generate one labeled trajectory via OpenRouter API.

        Args:
            trajectory_id: Unique ID string (e.g. 'traj_001').

        Returns:
            Complete trajectory dict matching team schema, or None on failure.
        """
        num_steps = random.randint(
            self.config.min_steps,
            self.config.max_steps,
        )
        num_hallucinations = random.randint(
            self.config.min_hallucinations,
            min(self.config.max_hallucinations, num_steps - 1),
        )

        prompt = self._build_prompt(num_steps, num_hallucinations)

        try:
            # Call OpenRouter via OpenAI-compatible SDK
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=CONFIG.openrouter.temperature,
                max_tokens=CONFIG.openrouter.max_tokens,
            )
            response_text = response.choices[0].message.content
        except Exception as e:
            print(f"  API ERROR for {trajectory_id}: {e}")
            return None

        # Parse and validate
        trajectory = self._parse_response(response_text)
        if trajectory is None:
            print(f"  PARSE ERROR for {trajectory_id}: could not parse response")
            return None

        # Add metadata
        trajectory["trajectory_id"] = trajectory_id
        trajectory["total_steps"] = len(trajectory["steps"])

        return trajectory

    def generate_dataset(
        self,
        num_trajectories: Optional[int] = None,
        resume: bool = True,
    ) -> List[Dict]:
        """Generate the full synthetic trajectory dataset.

        Includes rate limiting, progress tracking, and crash recovery.

        Args:
            num_trajectories: Override for CONFIG.synthetic.num_trajectories.
            resume: If True, load existing partial results and continue.

        Returns:
            List of all generated trajectory dicts.
        """
        total = num_trajectories or self.config.num_trajectories
        output_path = os.path.join(
            CONFIG.paths.trajectory_dir,
            self.config.output_filename,
        )

        # Resume from partial results
        trajectories = []
        start_idx = 0
        if resume and os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    trajectories = json.load(f)
                start_idx = len(trajectories)
                print(f"Resuming from trajectory {start_idx + 1} "
                      f"({start_idx} already generated)")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Could not resume: {e}. Starting fresh.")
                trajectories = []
                start_idx = 0

        # Rate limit
        request_interval = 60.0 / self.rate_limit
        failed_count = 0
        max_retries = 3

        print(f"\nGenerating {total - start_idx} trajectories...")
        print(f"Rate limit: {self.rate_limit} req/min "
              f"({request_interval:.1f}s between requests)\n")

        for i in range(start_idx, total):
            trajectory_id = f"traj_{i + 1:03d}"
            print(f"[{i + 1}/{total}] Generating {trajectory_id}...", end=" ")

            trajectory = None
            for attempt in range(max_retries):
                trajectory = self.generate_single_trajectory(trajectory_id)
                if trajectory is not None:
                    break
                print(f"  Retry {attempt + 1}/{max_retries}...")
                time.sleep(request_interval)

            if trajectory is not None:
                trajectories.append(trajectory)
                halluc_count = sum(
                    1 for s in trajectory["steps"]
                    if s.get("ground_truth_label", False)
                )
                print(f"OK ({trajectory['total_steps']} steps, "
                      f"{halluc_count} hallucinations)")
            else:
                failed_count += 1
                print(f"FAILED after {max_retries} retries")

            # Save after every success
            self._save_trajectories(trajectories, output_path)

            # Rate limit wait
            if i < total - 1:
                time.sleep(request_interval)

        print(f"\n{'=' * 50}")
        print(f"Generation complete!")
        print(f"  Successful: {len(trajectories)}")
        print(f"  Failed:     {failed_count}")
        print(f"  Saved to:   {output_path}")
        print(f"{'=' * 50}")

        return trajectories

    def _save_trajectories(self, trajectories: List[Dict], output_path: str) -> None:
        """Save trajectories to JSON file with error handling."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(trajectories, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"  SAVE ERROR: {e}")


def validate_dataset(filepath: str) -> None:
    """Load and validate a generated trajectory dataset.

    Prints statistics and checks schema compliance.

    Args:
        filepath: Path to the synthetic_trajectories.json file.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Validation ERROR: {e}")
        return

    print(f"\nDataset: {filepath}")
    print(f"Total trajectories: {len(data)}")

    total_steps = 0
    total_hallucinations = 0
    type_counts = {}
    tool_counts = {}
    step_lengths = []

    for traj in data:
        steps = traj.get("steps", [])
        step_lengths.append(len(steps))
        total_steps += len(steps)

        for step in steps:
            action = step.get("action", "unknown")
            tool_counts[action] = tool_counts.get(action, 0) + 1

            if step.get("ground_truth_label", False):
                total_hallucinations += 1
                h_type = step.get("hallucination_type", "Unknown")
                type_counts[h_type] = type_counts.get(h_type, 0) + 1

    print(f"Total steps: {total_steps}")
    print(f"Avg steps/trajectory: {total_steps / max(len(data), 1):.1f}")
    print(f"Total hallucinations: {total_hallucinations}")
    print(f"Hallucination rate: "
          f"{total_hallucinations / max(total_steps, 1) * 100:.1f}%")

    print(f"\nHallucination types:")
    for h_type, count in sorted(type_counts.items()):
        print(f"  {h_type}: {count}")

    print(f"\nTool usage:")
    for tool, count in sorted(tool_counts.items()):
        print(f"  {tool}: {count}")

    if step_lengths:
        print(f"\nStep count distribution:")
        print(f"  Min: {min(step_lengths)}, Max: {max(step_lengths)}, "
              f"Avg: {sum(step_lengths) / len(step_lengths):.1f}")


# ════════════════════════════════════════════════════════════
# TEST BLOCK
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("AgentTrace - Synthetic Trajectory Generator")
    print("=" * 60)

    CONFIG.setup()

    # Check API key
    if not CONFIG.openrouter.api_key:
        print(f"\nERROR: Set {CONFIG.openrouter.api_key_env_var} first!")
        print("Example: $env:OPENROUTER_API_KEY='sk-or-v1-...'")
        sys.exit(1)

    print(f"\nProvider:    OpenRouter")
    print(f"Model:       {CONFIG.openrouter.model_name}")
    print(f"Trajectories target: {CONFIG.synthetic.num_trajectories}")
    print(f"Steps/traj:  {CONFIG.synthetic.min_steps}-{CONFIG.synthetic.max_steps}")
    print(f"Halluc/traj: {CONFIG.synthetic.min_hallucinations}-"
          f"{CONFIG.synthetic.max_hallucinations}")

    # Generate a small test batch (3 trajectories)
    print("\n--- TEST MODE: Generating 3 trajectories ---\n")
    generator = SyntheticTrajectoryGenerator()
    test_data = generator.generate_dataset(num_trajectories=3, resume=False)

    # Validate the test output
    output_path = os.path.join(
        CONFIG.paths.trajectory_dir,
        CONFIG.synthetic.output_filename,
    )
    if os.path.exists(output_path):
        validate_dataset(output_path)

        if test_data:
            print(f"\n--- Sample trajectory ---")
            print(json.dumps(test_data[0], indent=2, ensure_ascii=False)[:1500])
            print("...")
    else:
        print("No output file created. Check errors above.")
