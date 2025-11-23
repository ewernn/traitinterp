"""
traitlens - A minimal toolkit for extracting and analyzing trait vectors from transformers.

Like pandas for data analysis, traitlens provides primitives for trait analysis.
You build your extraction strategy from these building blocks.
"""

from .hooks import HookManager
from .activations import ActivationCapture
from .compute import (
    mean_difference,
    compute_derivative,
    compute_second_derivative,
    projection,
    cosine_similarity,
    normalize_vectors
)
from .methods import (
    ExtractionMethod,
    MeanDifferenceMethod,
    ICAMethod,
    ProbeMethod,
    GradientMethod,
    get_method
)
from .metrics import (
    # Separation metrics
    separation,
    accuracy,
    effect_size,
    # Statistical metrics
    p_value,
    polarity_correct,
    # Stability metrics
    bootstrap_stability,
    noise_robustness,
    subsample_stability,
    # Vector properties
    sparsity,
    effective_rank,
    top_k_concentration,
    # Cross-vector metrics
    orthogonality,
    cross_trait_accuracy,
    # Convenience functions
    evaluate_vector,
    evaluate_vector_properties,
)

__version__ = "0.3.0"  # Bumped for new metrics module

__all__ = [
    # Core classes
    "HookManager",
    "ActivationCapture",

    # Compute functions
    "mean_difference",
    "compute_derivative",
    "compute_second_derivative",
    "projection",
    "cosine_similarity",
    "normalize_vectors",

    # Extraction methods
    "ExtractionMethod",
    "MeanDifferenceMethod",
    "ICAMethod",
    "ProbeMethod",
    "GradientMethod",
    "get_method",

    # Evaluation metrics
    "separation",
    "accuracy",
    "effect_size",
    "p_value",
    "polarity_correct",
    "bootstrap_stability",
    "noise_robustness",
    "subsample_stability",
    "sparsity",
    "effective_rank",
    "top_k_concentration",
    "orthogonality",
    "cross_trait_accuracy",
    "evaluate_vector",
    "evaluate_vector_properties",
]