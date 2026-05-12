"""
AgentTrace — Central Configuration
===================================
Step-Level Hallucination Detection and Attribution
in Multi-Step LLM Agent Workflows

Every module in the project imports from this file.
No hardcoded values exist outside config.py.

Author: P. Somnath Reddy (Research Lead)
GitHub: github.com/Soum-Code/agenttrace
Target: EMNLP 2026 / ICLR 2027
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


# ════════════════════════════════════════════════════════════
# 1. PROJECT PATHS
# ════════════════════════════════════════════════════════════

@dataclass
class PathConfig:
    """All file and directory paths used across the project.

    Attributes:
        project_root: Absolute path to the agenttrace/ top-level directory.
        data_dir: Directory for raw and processed datasets.
        trajectory_dir: Where synthetic and live trajectories are saved.
        results_dir: Evaluation output files go here.
        index_dir: FAISS vector index storage.
        model_cache_dir: Local cache for downloaded HuggingFace models.
        paper_dir: LaTeX source and figures for the paper.
        log_dir: Runtime logs from the step tracer.

    Example:
        >>> paths = PathConfig()
        >>> print(paths.trajectory_dir)
        /path/to/agenttrace/data/trajectories
    """
    # project_root is auto-detected: parent of this config.py file
    project_root: str = field(
        default_factory=lambda: str(Path(__file__).resolve().parent)
    )

    @property
    def data_dir(self) -> str:
        """Directory containing all data assets."""
        return os.path.join(self.project_root, "data")

    @property
    def trajectory_dir(self) -> str:
        """Synthetic and live agent trajectories (JSON files)."""
        return os.path.join(self.data_dir, "trajectories")

    @property
    def results_dir(self) -> str:
        """Benchmark results and evaluation outputs."""
        return os.path.join(self.project_root, "evaluation", "results")

    @property
    def index_dir(self) -> str:
        """FAISS index files for semantic search."""
        return os.path.join(self.project_root, "indexes")

    @property
    def model_cache_dir(self) -> str:
        """Cached HuggingFace model weights."""
        return os.path.join(self.project_root, ".model_cache")

    @property
    def paper_dir(self) -> str:
        """LaTeX paper source directory."""
        return os.path.join(self.project_root, "paper")

    @property
    def log_dir(self) -> str:
        """Step tracer runtime logs."""
        return os.path.join(self.project_root, "logs")

    def ensure_dirs(self) -> None:
        """Create all directories if they do not already exist.

        Call this once at project startup to guarantee the
        folder structure is in place before any module writes.

        Example:
            >>> PathConfig().ensure_dirs()
        """
        for dir_path in [
            self.data_dir,
            self.trajectory_dir,
            self.results_dir,
            self.index_dir,
            self.model_cache_dir,
            self.paper_dir,
            self.log_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)  # no-op if already exists


# ════════════════════════════════════════════════════════════
# 2. MODEL CONFIGURATIONS
# ════════════════════════════════════════════════════════════

@dataclass
class EmbeddingModelConfig:
    """Sentence-transformer model used for semantic similarity.

    Used by: detection/semantic_checker.py, tracer/step_logger.py

    Attributes:
        model_name: HuggingFace model identifier.
        embedding_dim: Output vector dimensionality.
        max_seq_length: Maximum input token length before truncation.
        device: Compute device ('cuda' for T4, 'cpu' as fallback).

    Example:
        >>> emb = EmbeddingModelConfig()
        >>> print(emb.model_name)
        sentence-transformers/all-MiniLM-L6-v2
    """
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384          # all-MiniLM-L6-v2 outputs 384-d vectors
    max_seq_length: int = 256         # sufficient for single agent steps
    device: str = "cuda"              # Kaggle T4 default; falls back in code


@dataclass
class NLIModelConfig:
    """Natural Language Inference cross-encoder for factual grounding.

    Used by: detection/factual_grounding.py

    Attributes:
        model_name: HuggingFace cross-encoder identifier.
        labels: NLI output classes in model's label order.
        entailment_idx: Index of the 'entailment' class.
        contradiction_idx: Index of the 'contradiction' class.
        device: Compute device.

    Example:
        >>> nli = NLIModelConfig()
        >>> print(nli.labels)
        ['contradiction', 'neutral', 'entailment']
    """
    model_name: str = "cross-encoder/nli-deberta-v3-small"
    labels: List[str] = field(
        default_factory=lambda: ["contradiction", "neutral", "entailment"]
    )
    entailment_idx: int = 2           # index in labels list
    contradiction_idx: int = 0        # index in labels list
    device: str = "cuda"


@dataclass
class ClassifierConfig:
    """DistilBERT causal classifier for hallucination-type prediction.

    Fine-tuned by Member 2.
    Used by: attribution/causal_classifier.py

    Attributes:
        base_model: Pre-trained base model identifier.
        num_labels: Number of hallucination cause categories.
        categories: The 5 hallucination cause types from the taxonomy.
        max_seq_length: Maximum input length for classification.
        device: Compute device.

    Example:
        >>> clf = ClassifierConfig()
        >>> print(clf.categories)
        ['Planning', 'Retrieval', 'Reasoning', 'Tool-Use', 'Human-Interaction']
    """
    base_model: str = "distilbert-base-uncased"
    num_labels: int = 5               # matches len(categories)
    categories: List[str] = field(
        default_factory=lambda: [
            "Planning",
            "Retrieval",
            "Reasoning",
            "Tool-Use",
            "Human-Interaction",
        ]
    )
    max_seq_length: int = 512         # longer context for full-step input
    device: str = "cuda"


# ════════════════════════════════════════════════════════════
# 3. DETECTION THRESHOLDS
# ════════════════════════════════════════════════════════════

@dataclass
class ThresholdConfig:
    """All numeric thresholds for the detection pipeline.

    Tuned to beat AgentHallu SOTA (41.1% step localization accuracy).

    Attributes:
        similarity_cutoff: Cosine similarity below this → semantic drift.
        contradiction_cutoff: NLI contradiction score above this → flagged.
        confidence_cutoff: Minimum detector confidence to report a finding.
        tool_mismatch_cutoff: Jaccard overlap below this → tool-use hallucination.
        severity_thresholds: Confidence ranges mapping to Low/Medium/High severity.
        drift_window: Number of consecutive steps to check for gradual drift.

    Example:
        >>> th = ThresholdConfig()
        >>> print(th.similarity_cutoff)
        0.72
    """
    similarity_cutoff: float = 0.72       # cosine sim below this = drift
    contradiction_cutoff: float = 0.80    # NLI contradiction prob above this = flag
    confidence_cutoff: float = 0.60       # minimum detector confidence to report
    tool_mismatch_cutoff: float = 0.40    # Jaccard overlap below this = tool hallucination
    severity_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "High": 0.85,     # confidence >= 0.85 → High severity
            "Medium": 0.70,   # confidence >= 0.70 → Medium severity
            "Low": 0.60,      # confidence >= 0.60 → Low severity (= confidence_cutoff)
        }
    )
    drift_window: int = 3                # number of past steps to compare for drift


# ════════════════════════════════════════════════════════════
# 4. TRAINING PARAMETERS
# ════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """Hyperparameters for fine-tuning the causal classifier.

    Member 2 uses these when training the DistilBERT classifier.
    Logged to WandB for experiment tracking.

    Attributes:
        batch_size: Training batch size (T4 GPU memory safe).
        eval_batch_size: Evaluation batch size.
        learning_rate: AdamW learning rate.
        num_epochs: Total training epochs.
        warmup_ratio: Fraction of steps used for LR warmup.
        weight_decay: L2 regularization factor.
        max_grad_norm: Gradient clipping threshold.
        train_split: Fraction of data for training.
        val_split: Fraction of data for validation.
        test_split: Fraction of data for testing.
        seed: Random seed for reproducibility.

    Example:
        >>> train = TrainingConfig()
        >>> print(f"LR={train.learning_rate}, Epochs={train.num_epochs}")
        LR=2e-05, Epochs=10
    """
    batch_size: int = 16               # fits comfortably on T4 16GB
    eval_batch_size: int = 32          # larger for eval since no gradients
    learning_rate: float = 2e-5        # standard for DistilBERT fine-tuning
    num_epochs: int = 10               # enough for convergence on 200 trajectories
    warmup_ratio: float = 0.1          # 10% of steps for warmup
    weight_decay: float = 0.01         # light L2 regularization
    max_grad_norm: float = 1.0         # clip gradients for stability
    train_split: float = 0.70          # 70% train
    val_split: float = 0.15            # 15% validation
    test_split: float = 0.15           # 15% test
    seed: int = 42                     # reproducibility across all modules


# ════════════════════════════════════════════════════════════
# 5. EVALUATION PARAMETERS
# ════════════════════════════════════════════════════════════

@dataclass
class EvaluationConfig:
    """Parameters for the evaluation and benchmarking pipeline.

    Used by: evaluation/metrics.py, evaluation/benchmark.py

    Attributes:
        top_k_values: Values of k for precision@k computation.
        agenthallu_baseline: SOTA step localization accuracy to beat (41.1%).
        agenthallu_tool_baseline: SOTA tool-use hallucination accuracy (11.6%).
        bootstrap_samples: Number of bootstrap resamples for CI computation.
        significance_level: Alpha for statistical significance tests.

    Example:
        >>> ev = EvaluationConfig()
        >>> print(ev.agenthallu_baseline)
        0.411
    """
    top_k_values: List[int] = field(
        default_factory=lambda: [1, 3, 5]   # precision@1, @3, @5
    )
    agenthallu_baseline: float = 0.411       # 41.1% from arXiv 2601.06818
    agenthallu_tool_baseline: float = 0.116  # 11.6% tool-use accuracy
    bootstrap_samples: int = 1000            # for confidence intervals
    significance_level: float = 0.05         # p < 0.05 for significance


# ════════════════════════════════════════════════════════════
# 6. SYNTHETIC DATA GENERATION
# ════════════════════════════════════════════════════════════

@dataclass
class SyntheticDataConfig:
    """Parameters for generating synthetic agent trajectories.

    Used by: data/synthetic_generator.py

    Attributes:
        num_trajectories: Total trajectories to generate (200 for v1).
        min_steps: Minimum steps per trajectory.
        max_steps: Maximum steps per trajectory.
        min_hallucinations: Minimum hallucinations injected per trajectory.
        max_hallucinations: Maximum hallucinations injected per trajectory.
        available_tools: Tools the synthetic agent can use.
        hallucination_types: The 5-category taxonomy from the paper.
        output_filename: Name of the output JSON file.

    Example:
        >>> syn = SyntheticDataConfig()
        >>> print(syn.available_tools)
        ['web_search', 'calculator', 'knowledge_lookup']
    """
    num_trajectories: int = 200               # dataset size for v1
    min_steps: int = 3                        # minimum steps per trajectory
    max_steps: int = 7                        # maximum steps per trajectory
    min_hallucinations: int = 1               # at least 1 per trajectory
    max_hallucinations: int = 2               # at most 2 per trajectory
    available_tools: List[str] = field(
        default_factory=lambda: [
            "web_search",
            "calculator",
            "knowledge_lookup",
        ]
    )
    hallucination_types: List[str] = field(
        default_factory=lambda: [
            "Planning",
            "Retrieval",
            "Reasoning",
            "Tool-Use",
            "Human-Interaction",
        ]
    )
    output_filename: str = "synthetic_trajectories.json"


# ════════════════════════════════════════════════════════════
# 7. GEMINI API CONFIGURATION
# ════════════════════════════════════════════════════════════

@dataclass
class GeminiConfig:
    """Google Gemini API settings for synthetic data generation.

    API key is read from the GEMINI_API_KEY environment variable.
    Never hardcode the key in source code.

    Used by: data/synthetic_generator.py

    Attributes:
        model_name: Gemini model identifier.
        temperature: Sampling temperature (higher = more creative).
        top_p: Nucleus sampling cutoff.
        top_k: Top-k sampling cutoff.
        max_output_tokens: Maximum tokens per API response.
        api_key_env_var: Name of the environment variable holding the key.
        requests_per_minute: Rate limit to avoid 429 errors on free tier.

    Example:
        >>> gem = GeminiConfig()
        >>> print(gem.model_name)
        gemini-1.5-flash
    """
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.8              # slightly creative for diverse trajectories
    top_p: float = 0.95                   # standard nucleus sampling
    top_k: int = 40                       # standard top-k
    max_output_tokens: int = 4096         # enough for a full trajectory JSON
    api_key_env_var: str = "GEMINI_API_KEY"  # env var name, NOT the actual key
    requests_per_minute: int = 15         # free tier rate limit safety margin

    @property
    def api_key(self) -> Optional[str]:
        """Read API key from environment. Returns None if not set.

        Example:
            >>> import os
            >>> os.environ['GEMINI_API_KEY'] = 'test-key'
            >>> GeminiConfig().api_key
            'test-key'
        """
        return os.environ.get(self.api_key_env_var)


@dataclass
class OpenRouterConfig:
    """OpenRouter API settings (OpenAI-compatible, no quota issues).

    Primary LLM provider for synthetic data generation.
    Falls back to direct Gemini API if not configured.

    Attributes:
        base_url: OpenRouter API endpoint.
        model_name: Model identifier on OpenRouter.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens per response.
        api_key_env_var: Env var name for the API key.
        requests_per_minute: Rate limit safety margin.

    Example:
        >>> orr = OpenRouterConfig()
        >>> print(orr.model_name)
        google/gemini-2.0-flash-001
    """
    base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = "google/gemini-2.0-flash-001"
    temperature: float = 0.8
    max_tokens: int = 4096
    api_key_env_var: str = "OPENROUTER_API_KEY"
    requests_per_minute: int = 20  # OpenRouter has higher limits

    @property
    def api_key(self) -> Optional[str]:
        """Read API key from environment. Returns None if not set."""
        return os.environ.get(self.api_key_env_var)


# ════════════════════════════════════════════════════════════
# 8. WEIGHTS & BIASES CONFIGURATION
# ════════════════════════════════════════════════════════════

@dataclass
class WandBConfig:
    """Weights & Biases experiment tracking settings.

    Used by: evaluation/benchmark.py, attribution/causal_classifier.py

    Attributes:
        project_name: WandB project name (shared by all team members).
        entity: WandB team or username (set to None for personal account).
        enabled: Toggle WandB logging on/off (off for quick local tests).
        log_model: Whether to upload model checkpoints to WandB.
        tags: Default tags applied to every run.

    Example:
        >>> wb = WandBConfig()
        >>> print(wb.project_name)
        agenttrace
    """
    project_name: str = "agenttrace"
    entity: Optional[str] = None          # set to team name if using WandB Teams
    enabled: bool = True                  # set False to skip logging in dev
    log_model: bool = False               # True only for final training runs
    tags: List[str] = field(
        default_factory=lambda: ["hallucination-detection", "agentic-ai", "emnlp2026"]
    )


# ════════════════════════════════════════════════════════════
# 9. STEP TRACER CONFIGURATION
# ════════════════════════════════════════════════════════════

@dataclass
class TracerConfig:
    """Configuration for the step-level agent tracer.

    Used by: tracer/step_logger.py

    Attributes:
        enable_realtime_logging: If True, logs each step as it happens.
        log_format: Output format for log files.
        max_trajectory_length: Safety cap on steps per trajectory.
        save_intermediate: Whether to save partial trajectories during execution.

    Example:
        >>> tr = TracerConfig()
        >>> print(tr.max_trajectory_length)
        20
    """
    enable_realtime_logging: bool = True
    log_format: str = "json"              # 'json' or 'jsonl'
    max_trajectory_length: int = 20       # safety cap to prevent runaway agents
    save_intermediate: bool = True        # save after each step for crash recovery


# ════════════════════════════════════════════════════════════
# 10. MASTER CONFIG — SINGLE ENTRY POINT
# ════════════════════════════════════════════════════════════

@dataclass
class AgentTraceConfig:
    """Master configuration object aggregating all sub-configs.

    Every module imports this single object:
        from config import CONFIG

    Attributes:
        paths: File and directory paths.
        embedding: Sentence-transformer model settings.
        nli: NLI cross-encoder model settings.
        classifier: DistilBERT causal classifier settings.
        thresholds: Detection pipeline thresholds.
        training: Fine-tuning hyperparameters.
        evaluation: Benchmarking parameters.
        synthetic: Synthetic data generation settings.
        gemini: Gemini API settings.
        wandb: Weights & Biases tracking settings.
        tracer: Step tracer settings.

    Example:
        >>> from config import CONFIG
        >>> CONFIG.thresholds.similarity_cutoff
        0.72
        >>> CONFIG.gemini.model_name
        'gemini-1.5-flash'
    """
    paths: PathConfig = field(default_factory=PathConfig)
    embedding: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)
    nli: NLIModelConfig = field(default_factory=NLIModelConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    synthetic: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    tracer: TracerConfig = field(default_factory=TracerConfig)

    def setup(self) -> "AgentTraceConfig":
        """Initialize the project: create directories, validate config.

        Call once at project startup. Returns self for chaining.

        Example:
            >>> CONFIG.setup()
        """
        self.paths.ensure_dirs()          # create all directories
        self._validate()                  # check for obvious misconfigurations
        return self

    def _validate(self) -> None:
        """Run sanity checks on the configuration values.

        Raises:
            ValueError: If any config value is clearly invalid.

        Example:
            >>> AgentTraceConfig()._validate()  # passes silently
        """
        # Training splits must sum to 1.0
        split_sum = (
            self.training.train_split
            + self.training.val_split
            + self.training.test_split
        )
        if abs(split_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Training splits must sum to 1.0, got {split_sum:.4f}"
            )

        # Classifier categories must match num_labels
        if len(self.classifier.categories) != self.classifier.num_labels:
            raise ValueError(
                f"classifier.num_labels ({self.classifier.num_labels}) != "
                f"len(categories) ({len(self.classifier.categories)})"
            )

        # Thresholds must be in valid ranges
        if not (0.0 < self.thresholds.similarity_cutoff < 1.0):
            raise ValueError("similarity_cutoff must be between 0 and 1")
        if not (0.0 < self.thresholds.confidence_cutoff < 1.0):
            raise ValueError("confidence_cutoff must be between 0 and 1")

        # Synthetic data constraints
        if self.synthetic.min_steps > self.synthetic.max_steps:
            raise ValueError("min_steps cannot exceed max_steps")
        if self.synthetic.min_hallucinations > self.synthetic.max_hallucinations:
            raise ValueError("min_hallucinations cannot exceed max_hallucinations")


# ════════════════════════════════════════════════════════════
# GLOBAL INSTANCE — import this everywhere
# ════════════════════════════════════════════════════════════

CONFIG = AgentTraceConfig()
"""
Usage across all modules:

    from config import CONFIG

    # Access any setting
    model = CONFIG.embedding.model_name
    cutoff = CONFIG.thresholds.similarity_cutoff
    lr = CONFIG.training.learning_rate
    traj_dir = CONFIG.paths.trajectory_dir
"""


# ════════════════════════════════════════════════════════════
# FLAT ALIASES — detection modules import these directly
# ════════════════════════════════════════════════════════════

SEMANTIC_MODEL_NAME = CONFIG.embedding.model_name
SEMANTIC_SIMILARITY_THRESHOLD = CONFIG.thresholds.similarity_cutoff
NLI_MODEL_NAME = CONFIG.nli.model_name
NLI_CONTRADICTION_THRESHOLD = CONFIG.thresholds.contradiction_cutoff
NLI_ENTAILMENT_THRESHOLD = 1.0 - CONFIG.thresholds.contradiction_cutoff
TOOL_CLAIM_SIMILARITY_THRESHOLD = CONFIG.thresholds.tool_mismatch_cutoff
MAX_CLAIMS_PER_REASONING = 5
CONTRADICTION_WINDOW_SIZE = CONFIG.thresholds.drift_window
CAUSAL_MODEL_NAME = CONFIG.classifier.base_model
CAUSAL_LABELS = CONFIG.classifier.categories
CAUSAL_CONFIDENCE_THRESHOLD = CONFIG.thresholds.confidence_cutoff
LOCALIZATION_SIGNAL_WEIGHTS = {
    "semantic_similarity": 0.35,
    "tool_claim_match": 0.30,
    "nli_score": 0.25,
    "contradiction_with_prev": 0.10,
}
TYPE_TOOL_USE = "Tool-Use"
TYPE_FACTUAL = "Retrieval"
TYPE_CONTRADICTION = "Reasoning"
SEVERITY_HIGH = "High"
SEVERITY_MEDIUM = "Medium"
SEVERITY_LOW = "Low"
SEVERITY_HIGH_UPPER = 0.3
SEVERITY_MEDIUM_UPPER = 0.5
CORRECTOR_MAX_RETRIES = 3
CORRECTOR_STRATEGY_ORDER = [
    "tool_requery",
    "reasoning_override",
    "step_rollback"
]


# ════════════════════════════════════════════════════════════
# TEST BLOCK — run with: python config.py
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("AgentTrace — Configuration Verification")
    print("=" * 60)

    # Initialize: create dirs + validate
    CONFIG.setup()
    print("\n✓ All directories created successfully")
    print("✓ Configuration validation passed")

    # Print all settings in a structured format
    print(f"\n{'─' * 60}")
    print("PATHS")
    print(f"{'─' * 60}")
    print(f"  Project root:    {CONFIG.paths.project_root}")
    print(f"  Data dir:        {CONFIG.paths.data_dir}")
    print(f"  Trajectory dir:  {CONFIG.paths.trajectory_dir}")
    print(f"  Results dir:     {CONFIG.paths.results_dir}")
    print(f"  Index dir:       {CONFIG.paths.index_dir}")
    print(f"  Model cache:     {CONFIG.paths.model_cache_dir}")
    print(f"  Log dir:         {CONFIG.paths.log_dir}")

    print(f"\n{'─' * 60}")
    print("MODELS")
    print(f"{'─' * 60}")
    print(f"  Embedding:       {CONFIG.embedding.model_name} ({CONFIG.embedding.embedding_dim}d)")
    print(f"  NLI:             {CONFIG.nli.model_name}")
    print(f"  Classifier:      {CONFIG.classifier.base_model} ({CONFIG.classifier.num_labels} labels)")
    print(f"  Categories:      {CONFIG.classifier.categories}")

    print(f"\n{'─' * 60}")
    print("THRESHOLDS")
    print(f"{'─' * 60}")
    print(f"  Similarity:      {CONFIG.thresholds.similarity_cutoff}")
    print(f"  Contradiction:   {CONFIG.thresholds.contradiction_cutoff}")
    print(f"  Confidence:      {CONFIG.thresholds.confidence_cutoff}")
    print(f"  Tool mismatch:   {CONFIG.thresholds.tool_mismatch_cutoff}")
    print(f"  Severity levels: {CONFIG.thresholds.severity_thresholds}")
    print(f"  Drift window:    {CONFIG.thresholds.drift_window} steps")

    print(f"\n{'─' * 60}")
    print("TRAINING")
    print(f"{'─' * 60}")
    print(f"  Batch size:      {CONFIG.training.batch_size}")
    print(f"  Learning rate:   {CONFIG.training.learning_rate}")
    print(f"  Epochs:          {CONFIG.training.num_epochs}")
    print(f"  Splits:          {CONFIG.training.train_split}/{CONFIG.training.val_split}/{CONFIG.training.test_split}")
    print(f"  Seed:            {CONFIG.training.seed}")

    print(f"\n{'─' * 60}")
    print("EVALUATION")
    print(f"{'─' * 60}")
    print(f"  Top-k values:    {CONFIG.evaluation.top_k_values}")
    print(f"  AgentHallu SOTA: {CONFIG.evaluation.agenthallu_baseline} (step loc)")
    print(f"  AgentHallu Tool: {CONFIG.evaluation.agenthallu_tool_baseline} (tool-use)")
    print(f"  Bootstrap N:     {CONFIG.evaluation.bootstrap_samples}")

    print(f"\n{'─' * 60}")
    print("SYNTHETIC DATA")
    print(f"{'─' * 60}")
    print(f"  Trajectories:    {CONFIG.synthetic.num_trajectories}")
    print(f"  Steps range:     {CONFIG.synthetic.min_steps}-{CONFIG.synthetic.max_steps}")
    print(f"  Halluc. range:   {CONFIG.synthetic.min_hallucinations}-{CONFIG.synthetic.max_hallucinations}")
    print(f"  Tools:           {CONFIG.synthetic.available_tools}")
    print(f"  Types:           {CONFIG.synthetic.hallucination_types}")
    print(f"  Output file:     {CONFIG.synthetic.output_filename}")

    print(f"\n{'─' * 60}")
    print("GEMINI API")
    print(f"{'─' * 60}")
    print(f"  Model:           {CONFIG.gemini.model_name}")
    print(f"  Temperature:     {CONFIG.gemini.temperature}")
    print(f"  Max tokens:      {CONFIG.gemini.max_output_tokens}")
    api_status = "✓ SET" if CONFIG.gemini.api_key else "✗ NOT SET"
    print(f"  API key:         {api_status} (env: {CONFIG.gemini.api_key_env_var})")
    print(f"  Rate limit:      {CONFIG.gemini.requests_per_minute} req/min")

    print(f"\n{'\u2500' * 60}")
    print("OPENROUTER API")
    print(f"{'\u2500' * 60}")
    print(f"  Base URL:        {CONFIG.openrouter.base_url}")
    print(f"  Model:           {CONFIG.openrouter.model_name}")
    print(f"  Temperature:     {CONFIG.openrouter.temperature}")
    print(f"  Max tokens:      {CONFIG.openrouter.max_tokens}")
    or_status = "\u2713 SET" if CONFIG.openrouter.api_key else "\u2717 NOT SET"
    print(f"  API key:         {or_status} (env: {CONFIG.openrouter.api_key_env_var})")
    print(f"  Rate limit:      {CONFIG.openrouter.requests_per_minute} req/min")

    print(f"\n{'─' * 60}")
    print("WEIGHTS & BIASES")
    print(f"{'─' * 60}")
    print(f"  Project:         {CONFIG.wandb.project_name}")
    print(f"  Enabled:         {CONFIG.wandb.enabled}")
    print(f"  Tags:            {CONFIG.wandb.tags}")

    print(f"\n{'─' * 60}")
    print("TRACER")
    print(f"{'─' * 60}")
    print(f"  Realtime log:    {CONFIG.tracer.enable_realtime_logging}")
    print(f"  Log format:      {CONFIG.tracer.log_format}")
    print(f"  Max steps:       {CONFIG.tracer.max_trajectory_length}")
    print(f"  Save partial:    {CONFIG.tracer.save_intermediate}")

    print(f"\n{'=' * 60}")
    print("✓ config.py verified — all systems nominal")
    print(f"{'=' * 60}")
