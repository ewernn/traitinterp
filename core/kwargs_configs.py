"""
Pipeline configuration dataclasses.

Bundles CLI/function parameters into single objects that thread through
function calls without 30-arg signatures.

Usage:
    from core.kwargs_configs import SteeringConfig, ExtractionConfig
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SteeringConfig:
    """Configuration for steering evaluation and coefficient search."""
    # Experiment
    experiment: str = ""
    vector_experiment: Optional[str] = None
    extraction_variant: Optional[str] = None

    # Search
    layers_arg: str = "30%-60%"
    coefficients: Optional[List[float]] = None
    n_steps: int = 5
    up_mult: float = 1.3
    down_mult: float = 0.85
    start_mult: float = 0.7
    momentum: float = 0.1

    # Vector
    method: str = "probe"
    component: str = "residual"
    position: str = "response[:5]"

    # Evaluation
    max_new_tokens: int = 64
    min_coherence: float = 77
    subset: int = 5
    relevance_check: bool = True
    judge_provider: str = "openai"
    prompt_set: str = "steering"

    # Judge
    eval_prompt: Optional[str] = None
    use_default_prompt: bool = False
    trait_judge: Optional[str] = None

    # Output
    save_mode: str = "best"
    direction: str = "positive"
    force: bool = False

    # Model loading
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"

    # Modes
    batched: bool = True
    regenerate_responses: bool = False
    baseline_only: bool = False
    questions_file: Optional[str] = None
    ablation: Optional[int] = None  # layer number, or None


@dataclass
class VettingStats:
    """Result from response vetting — quality gate info."""
    pos_passed: int = 0
    pos_total: int = 0
    neg_passed: int = 0
    neg_total: int = 0

    @property
    def passed(self) -> bool:
        """Check if enough responses passed for both polarities."""
        return self.pos_passed >= 10 and self.neg_passed >= 10

    @property
    def pass_rate(self) -> float:
        total = self.pos_total + self.neg_total
        return (self.pos_passed + self.neg_passed) / total if total > 0 else 0


@dataclass
class ExtractionConfig:
    """Configuration for the extraction pipeline."""
    # Experiment
    experiment: str = ""
    model_variant: Optional[str] = None

    # Pipeline control
    only_stages: Optional[set] = None
    force: bool = False
    save_activations: bool = False

    # Methods
    methods: Optional[List[str]] = None
    component: str = "residual"
    position: str = "response[:5]"
    layers: Optional[List[int]] = None

    # Generation
    rollouts: int = 1
    temperature: float = 0.0
    max_new_tokens: Optional[int] = None

    # Vetting
    vet_responses: bool = True
    pos_threshold: int = 60
    neg_threshold: int = 40
    max_concurrent: int = 100
    paired_filter: bool = False
    adaptive: bool = False
    min_pass_rate: float = 0.0
    min_per_polarity: int = 0

    # Data
    val_split: float = 0.1
    logitlens: bool = False

    # Model loading
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    base_model: Optional[bool] = None


@dataclass
class InferenceConfig:
    """Configuration for the inference pipeline."""
    experiment: str = ""
    prompt_set: str = ""
    model_variant: Optional[str] = None
    extraction_variant: Optional[str] = None

    # Pipeline control
    skip_generate: bool = False
    from_activations: bool = False

    # Projection
    traits: Optional[List[str]] = None
    layers: str = "best,best+5"
    component: str = "residual"
    centered: bool = False
    skip_existing: bool = False

    # Generation
    max_new_tokens: int = 50
    temperature: float = 0.0
    no_server: bool = False

    # Model loading
    load_in_8bit: bool = False
    load_in_4bit: bool = False
