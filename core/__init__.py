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
    MultiLayerCapture,          # capture one component across layers
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
