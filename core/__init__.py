"""
Core primitives for trait vector extraction and analysis.
"""

from .types import (
    VectorSpec,
    ProjectionConfig,
    activation_scale,
)
from .hooks import (
    HookManager,                # base: all hook registration
    get_hook_path,              # layer + component -> string path
    detect_contribution_paths,  # auto-detect post-norm architecture
    LayerHook,                  # single-layer base class
    CaptureHook,                # capture from one layer
    SteeringHook,               # steer one layer
    AblationHook,               # ablate direction from one layer
    MultiLayerCapture,          # capture one component across layers
    MultiLayerSteeringHook,     # steer multiple layers simultaneously
    MultiLayerAblationHook,     # ablate direction across all layers
    BatchedLayerSteeringHook,   # different steering per batch slice
)
from .steering import (
    batched_steering_generate,              # batched generation with per-config steering
    multi_trait_batched_steering_generate,   # per-config prompts (multi-trait batching)
)
from .methods import (
    ExtractionMethod,
    MeanDifferenceMethod,
    ProbeMethod,
    GradientMethod,
    RandomBaselineMethod,
    get_method,
)
from .math import (
    projection,
    project_with_config,
    project_single,
    cosine_similarity,
    batch_cosine_similarity,
    orthogonalize,
    separation,
    accuracy,
    effect_size,
    p_value,
    polarity_correct,
    vector_properties,
    distribution_properties,
    remove_massive_dims,
)
from .logit_lens import (
    vector_to_vocab,
    build_common_token_mask,
    get_interpretation_layers,
)
from .generation import (
    HookedGenerator,
    CaptureConfig,
    SteeringConfig,
    TokenOutput,
    SequenceOutput,
)
from .profiling import (
    gpu_profile,
    gpu_timer,
    memory_stats,
    bandwidth_report,
    tensor_size_gb,
    ProfileResult,
)
from .backends import (
    GenerationBackend,
    LocalBackend,
    ServerBackend,
    get_backend,
    SteeringSpec,
    CaptureSpec,
    GenerationConfig,
    CaptureResult,
    TokenResult,
)
