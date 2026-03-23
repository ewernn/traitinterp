"""
Shared type definitions for trait vector operations.

Input: None (type definitions only)
Output: VectorSpec, VectorResult, JudgeResult, ProjectionConfig, ProjectionEntry, ProjectionRecord, ResponseRecord, ModelConfig, ModelVariant, SteeringEntry
Usage:
    from core.types import VectorSpec, VectorResult, JudgeResult, ProjectionConfig, ModelVariant

    spec = VectorSpec(layer=9, component='residual', position='response[:]', method='probe')
    config = ProjectionConfig.single(9, 'residual', 'response[:]', 'probe', weight=0.9)
    result = JudgeResult(trait_mean=72.5, coherence_mean=85.0, n=10)
    variant = ModelVariant(name='base', model='google/gemma-2-2b', lora=None)
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, NamedTuple, Optional

class ModelVariant(NamedTuple):
    """Model variant resolved from experiment config."""
    name: str
    model: str
    lora: Optional[str] = None


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
class VectorResult:
    """Result from select_vector / select_vectors. Identifies a scored vector."""
    layer: int
    method: str
    position: str
    component: str
    score: Optional[float]          # steering delta (None if unscored)
    direction: Optional[str]        # "positive" or "negative" (None if unscored)
    source: str                     # "steering" or "unscored"
    coefficient: Optional[float]    # best steering coefficient (None if unscored)
    naturalness: Optional[float] = None

    def to_vector_spec(self, weight: float = 1.0) -> VectorSpec:
        """Convert to VectorSpec for use in hooks."""
        return VectorSpec(layer=self.layer, component=self.component,
                         position=self.position, method=self.method, weight=weight)


@dataclass
class JudgeResult:
    """Result from LLM-as-judge scoring of steered responses."""
    trait_mean: Optional[float]
    coherence_mean: Optional[float]
    n: int
    trait_std: float = 0.0
    success_rate: float = 0.0
    min_score: Optional[float] = None
    max_score: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'JudgeResult':
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fields})

    @classmethod
    def empty(cls) -> 'JudgeResult':
        return cls(trait_mean=None, coherence_mean=None, n=0)


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
class ProjectionEntry:
    """One vector's per-token projection scores at a single layer."""
    method: str
    layer: int
    selection_source: str       # "steering" or "unscored"
    baseline: float             # projection of class centroid (for centering)
    prompt: List[float]         # per-token raw projection scores
    response: List[float]       # per-token raw projection scores
    prompt_token_norms: List[float]   # per-token ||h|| at this layer
    response_token_norms: List[float] # per-token ||h|| at this layer

    def to_dict(self, precision: int = 4) -> dict:
        r = precision
        return {
            'method': self.method,
            'layer': self.layer,
            'selection_source': self.selection_source,
            'baseline': round(self.baseline, r),
            'prompt': [round(v, r) for v in self.prompt],
            'response': [round(v, r) for v in self.response],
            'token_norms': {
                'prompt': [round(v, r) for v in self.prompt_token_norms],
                'response': [round(v, r) for v in self.response_token_norms],
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ProjectionEntry':
        norms = d.get('token_norms', {})
        return cls(
            method=d['method'], layer=d['layer'],
            selection_source=d.get('selection_source', 'unknown'),
            baseline=d.get('baseline', 0),
            prompt=d.get('prompt', []), response=d['response'],
            prompt_token_norms=norms.get('prompt', []),
            response_token_norms=norms.get('response', []),
        )

    @classmethod
    def from_vector_result(cls, vr: 'VectorResult', baseline: float,
                           prompt_proj, response_proj,
                           prompt_token_norms: list, response_token_norms: list) -> 'ProjectionEntry':
        """Construct from a VectorResult + computed projections."""
        to_list = lambda x: x.tolist() if hasattr(x, 'tolist') else list(x)
        return cls(
            method=vr.method, layer=vr.layer,
            selection_source=vr.source, baseline=baseline,
            prompt=to_list(prompt_proj), response=to_list(response_proj),
            prompt_token_norms=to_list(prompt_token_norms),
            response_token_norms=to_list(response_token_norms),
        )


@dataclass
class ProjectionRecord:
    """Projection JSON written per prompt per trait.

    File pattern: experiments/{exp}/inference/{variant}/projections/{trait}/{prompt_set}/{id}.json
    """
    prompt_id: str
    prompt_set: str
    n_prompt_tokens: int
    n_response_tokens: int
    component: str
    position: str
    centered: bool
    projections: List[ProjectionEntry]
    projection_date: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self, precision: int = 4) -> dict:
        return {
            'metadata': {
                'prompt_id': self.prompt_id,
                'prompt_set': self.prompt_set,
                'n_prompt_tokens': self.n_prompt_tokens,
                'n_response_tokens': self.n_response_tokens,
                'multi_vector': True,
                'n_vectors': len(self.projections),
                'component': self.component,
                'position': self.position,
                'centered': self.centered,
                'projection_date': self.projection_date,
            },
            'projections': [p.to_dict(precision) for p in self.projections],
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ProjectionRecord':
        meta = d.get('metadata', {})
        projections = [ProjectionEntry.from_dict(p) for p in d.get('projections', [])]
        return cls(
            prompt_id=meta.get('prompt_id', ''),
            prompt_set=meta.get('prompt_set', ''),
            n_prompt_tokens=meta.get('n_prompt_tokens', 0),
            n_response_tokens=meta.get('n_response_tokens', 0),
            component=meta.get('component', 'residual'),
            position=meta.get('position', ''),
            centered=meta.get('centered', False),
            projections=projections,
            projection_date=meta.get('projection_date', ''),
        )


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
    system_prompt: Optional[str] = None
    prompt_note: Optional[str] = None
    tags: Optional[List[str]] = None     # e.g., ["rollout", "env_name"]
    # Extension fields (rollout converters, sentence alignment)
    turn_boundaries: Optional[List[Dict]] = None
    source: Optional[Dict] = None
    sentence_boundaries: Optional[List[Dict]] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d['tags'] is None:
            d['tags'] = []
        # Omit extension fields when not set (avoid polluting standard response JSONs)
        for key in ('turn_boundaries', 'source', 'sentence_boundaries'):
            if d[key] is None:
                del d[key]
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


@dataclass
class SteeringEntry:
    """A discovered steering result entry from the filesystem."""
    trait: str
    model_variant: str
    position: str
    prompt_set: str
    full_path: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SteeringRunRecord:
    """One steering run from results.jsonl.

    Each run tests a specific vector configuration (layer, method, coefficient)
    and records the judge's scores for trait expression and coherence.
    """
    result: JudgeResult
    config: ProjectionConfig
    eval_judge: Optional[str] = None
    timestamp: str = ''
    input_hashes: Optional[Dict[str, str]] = None

    @property
    def layer(self) -> int:
        """Layer of the first (usually only) vector."""
        return self.config.vectors[0].layer

    @property
    def coefficient(self) -> float:
        """Steering coefficient (weight of first vector)."""
        return self.config.vectors[0].weight

    @property
    def method(self) -> str:
        """Method of the first vector."""
        return self.config.vectors[0].method

    def to_dict(self) -> dict:
        d = {
            'type': 'run',
            'result': self.result.to_dict(),
            'config': self.config.to_dict(),
            'eval': {'trait_judge': self.eval_judge},
            'timestamp': self.timestamp,
        }
        if self.input_hashes:
            d['input_hashes'] = self.input_hashes
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'SteeringRunRecord':
        result = JudgeResult.from_dict(d.get('result', {}))
        config_raw = d.get('config', {'vectors': []})
        config = ProjectionConfig.from_dict(config_raw)
        return cls(
            result=result,
            config=config,
            eval_judge=d.get('eval', {}).get('trait_judge'),
            timestamp=d.get('timestamp', ''),
            input_hashes=d.get('input_hashes'),
        )


@dataclass
class SteeringResults:
    """Full loaded steering results from results.jsonl.

    Returned by load_results(). Contains header metadata, optional baseline,
    and all steering run records.
    """
    trait: str
    direction: str
    steering_model: str
    steering_experiment: str
    vector_source: Dict
    eval: Dict
    prompts_file: str
    prompts_hash: str
    baseline: Optional[JudgeResult]
    runs: List[SteeringRunRecord]

    def to_dict(self) -> dict:
        return {
            'trait': self.trait,
            'direction': self.direction,
            'steering_model': self.steering_model,
            'steering_experiment': self.steering_experiment,
            'vector_source': self.vector_source,
            'eval': self.eval,
            'prompts_file': self.prompts_file,
            'prompts_hash': self.prompts_hash,
            'baseline': self.baseline.to_dict() if self.baseline else None,
            'runs': [r.to_dict() for r in self.runs],
        }
