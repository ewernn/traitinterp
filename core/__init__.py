"""
Core primitives for trait vector extraction and analysis.
"""

from .hooks import (
    HookManager,        # base: all hook registration
    get_hook_path,      # layer + component -> string path
    LayerHook,          # single-layer base class
    CaptureHook,        # capture from one layer
    SteeringHook,       # steer one layer
    MultiLayerCapture,  # capture one component across layers
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
    cosine_similarity,
    orthogonalize,
    separation,
    accuracy,
    effect_size,
    p_value,
    polarity_correct,
    evaluate_vector,
)
