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

__version__ = "0.2.0"  # Bumped for new methods module

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
]