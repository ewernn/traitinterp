"""
Core primitives for trait vector extraction and analysis.
"""

from .types import (
    VectorSpec,
    VectorResult,
    JudgeResult,
    ProjectionConfig,
    ProjectionEntry,
    ProjectionRecord,
    ResponseRecord,
    ModelConfig,
    ModelVariant,
    SteeringEntry,
    SteeringRunRecord,
    SteeringResults,
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
    ProjectionHook,             # project onto vectors on GPU (single layer)
    MultiLayerProjection,       # project across layers (stream-through)
    MultiLayerSteering,           # steer multiple layers simultaneously
    MultiLayerAblation,           # ablate direction across all layers
    ActivationCappingHook,            # clamp projection above threshold (single layer)
    MultiLayerActivationCapping,  # clamp projection above threshold (multi-layer)
    PerSampleSteering,         # different steering per batch slice
)
from .methods import (
    ExtractionMethod,
    MeanDifferenceMethod,
    ProbeMethod,
    GradientMethod,
    RandomBaselineMethod,
    RFMMethod,
    get_method,
)
from .math import (
    projection,
    cosine_similarity,
    batch_cosine_similarity,
    orthogonalize,
    accuracy,
    effect_size,
    polarity_correct,
    remove_massive_dims,
    normalize_projections,
)
from .generation import (
    HookedGenerator,
    CaptureConfig,
    SteeringConfig,
    TokenOutput,
    SequenceOutput,
)
