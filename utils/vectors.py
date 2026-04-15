"""
Vector loading and discovery primitives for trait vectors.

Low-level functions for loading vectors, metadata, and activation norms from disk.
Discovery of available vectors on disk (discover_vectors).
For best vector selection (using steering results), see utils/vector_selection.py.
"""

import json
import logging
import re
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch


@dataclass
class LoadedVector:
    """A trait vector in memory, paired with its precomputed steering scale.

    Returned by load_vectors() in steering code and consumed by coefficient-search
    routines. `base_coef` = activation_norm / vector_norm at `layer`; multiplying
    it by a coherence-search multiplier produces a candidate steering coefficient.
    """
    layer: int
    vector: torch.Tensor
    base_coef: float


def validate_vector_norm(vector: torch.Tensor, name: str = "trait vector",
                         tol: float = 1e-2) -> None:
    """Warn if a trait vector is not approximately unit norm.

    Projection scores assume unit-norm vectors. Non-unit vectors will scale
    scores by the vector's norm, making cross-trait comparisons meaningless.
    """
    norm = float(vector.float().norm())
    if norm < 1e-6:
        raise ValueError(f"{name} has near-zero norm ({norm:.2e}). Cannot project.")
    if abs(norm - 1.0) > tol:
        warnings.warn(
            f"{name} has norm {norm:.4f} (expected ~1.0). "
            f"Projection scores will be scaled by {norm:.4f}x. "
            f"Use normalize_vector=True or normalize the vector before loading.",
            stacklevel=2,
        )

from core.types import VectorSpec
from utils.paths import (
    get,
    get as get_path,
    get_vector_path,
    get_vector_metadata_path,
    desanitize_position,
)

logger = logging.getLogger(__name__)

# Single source of truth for steering quality thresholds
MIN_COHERENCE = 77
MIN_DELTA = 20
MIN_NATURALNESS = 50



def discover_vectors(
    experiment: str,
    trait: str,
    model_variant: str,
    component: str = None,
    position: str = None,
    layer: int = None,
    method: str = None,
) -> List[dict]:
    """Scan vector files on disk and return list of candidates.

    Each candidate is a dict with keys: layer, method, position, component, path.
    Filters narrow results when provided.
    """
    vectors_dir = get_path('extraction.vectors', experiment=experiment, trait=trait, model_variant=model_variant)
    if not vectors_dir.exists():
        return []

    candidates = []
    filename_pattern = re.compile(r'^layer(\d+)\.pt$')

    for pt_file in vectors_dir.rglob('layer*.pt'):
        rel_parts = pt_file.relative_to(vectors_dir).parts
        if len(rel_parts) != 4:
            continue

        pos_sanitized, comp, file_method, filename = rel_parts
        match = filename_pattern.match(filename)
        if not match:
            continue

        file_layer = int(match.group(1))
        pos = desanitize_position(pos_sanitized)

        if position and pos != position:
            continue
        if component and comp != component:
            continue
        if layer is not None and file_layer != layer:
            continue
        if method and file_method != method:
            continue

        candidates.append({
            'layer': file_layer,
            'method': file_method,
            'position': pos,
            'component': comp,
            'path': pt_file,
        })

    return candidates


def load_vector(
    experiment: str,
    trait: str,
    layer: int,
    model_variant: str,
    method: str = "probe",
    component: str = "residual",
    position: str = "response[:]",
) -> Optional[torch.Tensor]:
    """Load trait vector from experiment. Returns None if not found."""
    vector_file = get_vector_path(experiment, trait, method, layer, model_variant, component, position)
    if not vector_file.exists():
        return None
    return torch.load(vector_file, weights_only=True)


def load_cached_activation_norms(experiment: str, component: str = "residual") -> Dict[int, float]:
    """Load cached activation norms from extraction_evaluation.json.

    Returns {layer: norm} or empty dict if not available.
    """
    eval_path = get('extraction_eval.evaluation', experiment=experiment)
    if not eval_path.exists():
        return {}

    try:
        with open(eval_path) as f:
            data = json.load(f)
        norms = data.get('activation_norms', {})

        # New nested format: {component: {layer: norm}}
        if component in norms and isinstance(norms[component], dict):
            return {int(k): v for k, v in norms[component].items()}

        # Old flat format: {layer: norm} — only valid for residual
        if norms and not any(isinstance(v, dict) for v in norms.values()):
            if component == "residual":
                return {int(k): v for k, v in norms.items()}
            else:
                return {}

        return {}
    except (json.JSONDecodeError, KeyError):
        return {}


def find_vector_method(
    experiment: str,
    trait: str,
    layer: int,
    model_variant: str,
    component: str = "residual",
    position: str = "response[:]",
) -> Optional[str]:
    """Auto-detect vector method for a specific layer."""
    for method in ["probe", "mean_diff", "gradient"]:
        vector_path = get_vector_path(experiment, trait, method, layer, model_variant, component, position)
        if vector_path.exists():
            return method
    return None


def load_vector_metadata(
    experiment: str,
    trait: str,
    method: str,
    model_variant: str,
    component: str = "residual",
    position: str = "response[:]",
) -> Dict[str, Any]:
    """Load vector metadata for a trait/method."""
    metadata_path = get_vector_metadata_path(experiment, trait, method, model_variant, component, position)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No metadata for {experiment}/{trait}/{model_variant}/{method}. "
            f"Re-run extraction to generate metadata."
        )
    with open(metadata_path) as f:
        return json.load(f)


def load_vector_with_baseline(
    experiment: str,
    trait: str,
    method: str,
    layer: int,
    model_variant: str,
    component: str = "residual",
    position: str = "response[:]",
) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    """Load a vector with its baseline and per-vector metadata.

    Returns (vector tensor, baseline float, layer metadata dict).
    """
    vector_path = get_vector_path(experiment, trait, method, layer, model_variant, component, position)
    if not vector_path.exists():
        raise FileNotFoundError(f"Vector not found: {vector_path}")

    vector = torch.load(vector_path, weights_only=True)
    validate_vector_norm(vector, name=f"{trait}/{method}/layer{layer}")

    baseline = 0.0
    layer_metadata = {}
    try:
        metadata = load_vector_metadata(experiment, trait, method, model_variant, component, position)
        layer_info = metadata.get('layers', {}).get(str(layer), {})
        baseline = layer_info.get('baseline', 0.0)
        layer_metadata = layer_info
    except FileNotFoundError:
        logger.warning(f"No metadata found for {method}, baseline=0")

    return vector, baseline, layer_metadata


