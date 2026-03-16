"""
Shared type definitions for trait vector operations.

Input: None (type definitions only)
Output: VectorSpec, ProjectionConfig dataclasses
Usage:
    from core.types import VectorSpec, ProjectionConfig

    spec = VectorSpec(layer=9, component='residual', position='response[:]', method='probe')
    config = ProjectionConfig.single(9, 'residual', 'response[:]', 'probe', weight=0.9)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import torch


@dataclass
class VectorSpec:
    """
    Identifies a trait vector and its weight for projection/steering.

    Attributes:
        layer: Layer index (0-indexed)
        component: Hook component (residual, attn_contribution, mlp_contribution, etc.)
        position: Extraction position (response[:], response[:5], prompt[-1], etc.)
        method: Extraction method (probe, mean_diff, gradient)
        weight: Coefficient for steering, relative weight for projection (default 1.0)
    """
    layer: int
    component: str
    position: str
    method: str
    weight: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'VectorSpec':
        # Filter to only known fields (allows forward compat)
        fields = {'layer', 'component', 'position', 'method', 'weight'}
        return cls(**{k: v for k, v in d.items() if k in fields})


@dataclass
class ProjectionConfig:
    """
    Configuration for projecting/steering with one or more trait vectors.

    Single vector: ProjectionConfig.single(layer, component, position, method, weight)
    Ensemble: ProjectionConfig(vectors=[VectorSpec(...), VectorSpec(...), ...])
    """
    vectors: List[VectorSpec]

    @property
    def is_ensemble(self) -> bool:
        return len(self.vectors) > 1

    @property
    def normalized_weights(self) -> List[float]:
        """Weights normalized to sum to 1.0 (for projection)."""
        total = sum(v.weight for v in self.vectors)
        if total == 0:
            return [1.0 / len(self.vectors)] * len(self.vectors)
        return [v.weight / total for v in self.vectors]

    @classmethod
    def single(cls, layer: int, component: str, position: str,
               method: str, weight: float = 1.0) -> 'ProjectionConfig':
        """Create a config for a single trait vector."""
        return cls(vectors=[VectorSpec(layer, component, position, method, weight)])

    def to_dict(self) -> dict:
        return {'vectors': [v.to_dict() for v in self.vectors]}

    @classmethod
    def from_dict(cls, d: dict) -> 'ProjectionConfig':
        return cls(vectors=[VectorSpec.from_dict(v) for v in d['vectors']])


@dataclass
class ResponseRecord:
    """Canonical schema for inference response JSON files.

    Written by inference/generate_responses.py, read by capture_activations.py
    and project_activations_onto_traits.py. Rollout converters extend with
    turn_boundaries and source fields.

    File pattern: experiments/{exp}/inference/{variant}/responses/{prompt_set}/{id}.json
    """
    prompt: str
    response: str
    tokens: List[str]          # prompt_tokens + response_tokens
    token_ids: List[int]       # prompt_token_ids + response_token_ids
    prompt_end: int            # len(prompt_tokens) — split point
    inference_model: str
    capture_date: str          # ISO timestamp
    system_prompt: str = None
    prompt_note: str = None
    tags: List[str] = None     # e.g., ["rollout", "env_name"]

    def to_dict(self) -> dict:
        d = asdict(self)
        if d['tags'] is None:
            d['tags'] = []
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'ResponseRecord':
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fields})

    @property
    def prompt_tokens(self) -> List[str]:
        return self.tokens[:self.prompt_end]

    @property
    def response_tokens(self) -> List[str]:
        return self.tokens[self.prompt_end:]

    @property
    def prompt_token_ids(self) -> List[int]:
        return self.token_ids[:self.prompt_end]

    @property
    def response_token_ids(self) -> List[int]:
        return self.token_ids[self.prompt_end:]


@dataclass
class ModelConfig:
    """Schema for config/models/*.yaml files.

    Loaded by utils/model_registry.py. Used for hook placement, layer counts,
    SAE paths, and model identification.

    File pattern: config/models/{model-slug}.yaml
    """
    huggingface_id: str
    model_type: str                    # gemma2, llama, mistral, qwen2, olmo2, deepseek_v3, ...
    variant: str                       # base, it, sft, dpo
    supports_system_prompt: bool
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    max_context_length: int = 4096
    vocab_size: Optional[int] = None
    sae: Optional[Dict] = None         # {available, base_path, layer_template, downloaded_layers}
    moe: Optional[Dict] = None         # MoE-specific: {num_experts, top_k, ...}
    mla: Optional[Dict] = None         # Multi-head Latent Attention: {kv_lora_rank, ...}
    notes: Optional[Dict] = None       # Usage notes, quirks

    @classmethod
    def from_dict(cls, d: dict) -> 'ModelConfig':
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fields})

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def activation_scale(activations: torch.Tensor, vector: torch.Tensor) -> float:
    """
    Scale factor to normalize steering relative to activation magnitude.

    Used to make steering coefficients interpretable:
    - weight=0.9 means "perturb by ~90% of activation magnitude in vector direction"

    Args:
        activations: Activation tensor at the hook point
        vector: Trait vector to steer with

    Returns:
        Scaling factor: ||activations|| / ||vector||
    """
    return activations.norm().item() / (vector.norm().item() + 1e-8)
