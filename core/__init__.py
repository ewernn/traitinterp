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
    ModelVariant,
    SteeringEntry,
    SteeringRunRecord,
    SteeringResults,
)
from .architectures import (
    Architecture,               # per-arch hook paths + module tree
    HybridArchitecture,         # subclass for per-layer dispatch (Qwen3.5, DeepSeek V3)
    get_architecture,           # resolve Architecture for a model
    UnsupportedComponentError,  # component absent for this arch/layer
    ArchitectureMismatchError,  # live module tree diverges from adapter
    COMPONENTS,                 # canonical component names
    ModuleSpec,
    LayerPaths,
    layers as get_layers,       # nn.ModuleList of blocks (LoRA-aware)
    layer_prefix,               # dot-path prefix to layers (LoRA-aware)
    inner_model,                # unwrap PeftModel/multimodal
)
from .hooks import (
    HookManager,                # base: all hook registration
    LayerHook,                  # single-layer base class
    CaptureHook,                # capture from one layer (shape-agnostic)
    SteeringHook,               # steer one layer
    AblationHook,               # ablate direction from one layer
    MultiLayerCapture,          # capture one component across layers
    ProjectionHook,             # project onto vectors on GPU (single layer)
    MultiLayerProjection,       # project across layers (stream-through)
    MultiLayerSteering,           # steer multiple layers simultaneously
    MultiLayerAblation,           # ablate direction across all layers
    ActivationCappingHook,        # clamp projection within bounds (single layer)
    MultiLayerActivationCapping,  # clamp projection within bounds (multi-layer)
    PerSampleSteering,            # different steering per batch slice
    PerPositionSteeringHook,      # steer at specific token positions
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
    unit_normalize,
    projection,
    cosine_similarity,
    batch_cosine_similarity,
    orthogonalize,
    accuracy,
    effect_size,
    pearson_correlation,
    polarity_correct,
    auroc,
    remove_massive_dims,
    normalize_projections,
    pairwise_cosine_matrix,
    pca,
    project_out_subspace,
    trait_clusters,
    representational_similarity,
    vector_set_comparison,
    pca_norm_correlation,
)
from .validation import (
    compute_vector_quality,
)
from .generation import (
    HookedGenerator,
    CaptureConfig,
    SteeringConfig,
    TokenOutput,
    SequenceOutput,
)


__all__ = [
    # architectures
    "Architecture", "HybridArchitecture", "get_architecture", "COMPONENTS",
    "UnsupportedComponentError", "ArchitectureMismatchError",
    "ModuleSpec", "LayerPaths", "get_layers", "layer_prefix", "inner_model",
    # hooks
    "HookManager", "LayerHook", "CaptureHook", "SteeringHook", "AblationHook",
    "MultiLayerCapture", "ProjectionHook", "MultiLayerProjection",
    "MultiLayerSteering", "MultiLayerAblation",
    "ActivationCappingHook", "MultiLayerActivationCapping",
    "PerSampleSteering", "PerPositionSteeringHook",
    # types
    "VectorSpec", "VectorResult", "JudgeResult", "ProjectionConfig",
    "ProjectionEntry", "ProjectionRecord", "ResponseRecord", "ModelVariant",
    "SteeringEntry", "SteeringRunRecord", "SteeringResults",
    # methods
    "ExtractionMethod", "MeanDifferenceMethod", "ProbeMethod", "GradientMethod",
    "RandomBaselineMethod", "RFMMethod", "get_method",
    # math
    "unit_normalize", "projection", "cosine_similarity", "batch_cosine_similarity",
    "orthogonalize", "accuracy", "effect_size", "pearson_correlation",
    "polarity_correct", "auroc", "remove_massive_dims", "normalize_projections",
    "pairwise_cosine_matrix", "pca", "project_out_subspace", "trait_clusters",
    "representational_similarity", "vector_set_comparison", "pca_norm_correlation",
    # validation
    "compute_vector_quality",
    # generation
    "HookedGenerator", "CaptureConfig", "SteeringConfig", "TokenOutput", "SequenceOutput",
]
