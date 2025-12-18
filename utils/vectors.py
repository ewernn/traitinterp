"""
Utility functions for working with trait vectors.

Single source of truth for best layer selection.

Priority:
1. Steering results (ground truth - actual behavioral validation)
2. Cached result in extraction_evaluation.json
3. Effect size (fallback heuristic when steering not available)
4. Default (layer 16, probe method)

Note: Effect size from extraction is a rough heuristic, not a reliable predictor
of steering success. Always run steering evaluation for ground truth.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch

from utils.paths import get as get_path

logger = logging.getLogger(__name__)

# Single source of truth for minimum coherence threshold in steering evaluation
MIN_COHERENCE = 70


def get_best_layer(experiment: str, trait: str) -> dict:
    """
    Get best layer for a trait.

    Priority:
    1. Steering results (ground truth from steering/{trait}/results.json)
    2. Effect size (from extraction_evaluation.json all_results)
    3. Default (layer 16, probe method)

    Args:
        experiment: Experiment name
        trait: Trait path (e.g., "category/trait_name")

    Returns:
        Dict with 'layer', 'method', 'source', 'score'
        source is one of: 'steering', 'effect_size', 'default'

    Example:
        >>> best = get_best_layer('{experiment}', '{category}/{trait}')
        >>> print(f"L{best['layer']} {best['method']} (from {best['source']}: {best['score']:.1f})")
    """
    # 1. Check steering results first (ground truth)
    steering_result = _try_steering_result(experiment, trait)
    if steering_result:
        return steering_result

    # 2. Compute effect_size from all_results
    effect_size_result = _try_effect_size(experiment, trait)
    if effect_size_result:
        return effect_size_result

    # 3. Default fallback
    return {'layer': 16, 'method': 'probe', 'source': 'default', 'score': 0}


def _try_steering_result(experiment: str, trait: str) -> Optional[dict]:
    """Try to get best layer from steering results (ground truth)."""
    steering_path = get_path('steering.results', experiment=experiment, trait=trait)
    if not steering_path.exists():
        return None

    try:
        with open(steering_path) as f:
            data = json.load(f)
        baseline = data.get('baseline', {}).get('trait_mean', 0)
        best_run, best_delta = None, float('-inf')
        for run in data.get('runs', []):
            # Only consider single-layer runs
            if len(run.get('config', {}).get('layers', [])) == 1:
                trait_mean = run.get('result', {}).get('trait_mean')
                coherence = run.get('result', {}).get('coherence_mean', 0)
                if trait_mean is not None and coherence > MIN_COHERENCE:
                    delta = trait_mean - baseline
                    if delta > best_delta:
                        best_delta, best_run = delta, run
        if best_run:
            return {
                'layer': best_run['config']['layers'][0],
                'method': best_run['config'].get('methods', ['probe'])[0],
                'source': 'steering',
                'score': best_delta
            }
    except (json.JSONDecodeError, KeyError):
        pass

    return None


def _try_effect_size(experiment: str, trait: str) -> Optional[dict]:
    """Fallback: use effect_size as heuristic (not a reliable steering predictor)."""
    eval_path = get_path('extraction_eval.evaluation', experiment=experiment)
    if not eval_path.exists():
        return None

    try:
        with open(eval_path) as f:
            results = json.load(f).get('all_results', [])
        trait_results = [r for r in results if r.get('trait') == trait and r.get('val_effect_size')]
        if trait_results:
            best = max(trait_results, key=lambda r: r['val_effect_size'])
            return {
                'layer': best['layer'],
                'method': best['method'],
                'source': 'effect_size',
                'score': best['val_effect_size']
            }
    except (json.JSONDecodeError, KeyError):
        pass

    return None


def load_vector_with_metadata(
    experiment: str,
    trait: str,
    method: str,
    layer: int,
    component: str = "residual"
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Load a vector and its metadata.

    Args:
        experiment: Experiment name
        trait: Trait path (e.g., "category/trait_name")
        method: Extraction method (e.g., "probe", "mean_diff")
        layer: Layer number
        component: Component type (default: "residual")

    Returns:
        Tuple of (vector tensor, metadata dict)

    Raises:
        FileNotFoundError: If vector file doesn't exist
    """
    vectors_dir = get_path('extraction.vectors', experiment=experiment, trait=trait)

    # Build vector filename
    prefix = "" if component == "residual" else f"{component}_"
    vector_path = vectors_dir / f"{prefix}{method}_layer{layer}.pt"

    if not vector_path.exists():
        raise FileNotFoundError(f"Vector not found: {vector_path}")

    vector = torch.load(vector_path, weights_only=True)

    # Load metadata
    metadata = load_vector_metadata(experiment, trait)

    # Add specific vector info to metadata
    metadata['method'] = method
    metadata['layer'] = layer
    metadata['component'] = component

    return vector, metadata


def load_vector_metadata(experiment: str, trait: str) -> Dict[str, Any]:
    """
    Load vector metadata for a trait.

    Args:
        experiment: Experiment name
        trait: Trait path (e.g., "category/trait_name")

    Returns:
        Dict with vector metadata

    Raises:
        FileNotFoundError: If vectors/metadata.json doesn't exist
    """
    metadata_path = get_path('extraction.vectors_metadata', experiment=experiment, trait=trait)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No vectors/metadata.json for {experiment}/{trait}. "
            f"Re-run extraction to generate metadata, or create {metadata_path} manually."
        )

    with open(metadata_path) as f:
        return json.load(f)


def load_vector_with_baseline(
    experiment: str,
    trait: str,
    method: str,
    layer: int,
    component: str = "residual"
) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    """
    Load a vector with its baseline and per-vector metadata.

    The baseline is the projection of the training centroid onto the vector.
    Subtract it from projections to center around 0.

    Args:
        experiment: Experiment name
        trait: Trait path (e.g., "category/trait_name")
        method: Extraction method (e.g., "probe", "mean_diff")
        layer: Layer number
        component: Component type (default: "residual")

    Returns:
        Tuple of (vector tensor, baseline float, per-vector metadata dict)
        baseline is 0.0 if not available in metadata

    Raises:
        FileNotFoundError: If vector file doesn't exist
    """
    vectors_dir = get_path('extraction.vectors', experiment=experiment, trait=trait)

    # Build filenames
    prefix = "" if component == "residual" else f"{component}_"
    vector_path = vectors_dir / f"{prefix}{method}_layer{layer}.pt"
    metadata_path = vectors_dir / f"{prefix}{method}_layer{layer}_metadata.json"

    if not vector_path.exists():
        raise FileNotFoundError(f"Vector not found: {vector_path}")

    vector = torch.load(vector_path, weights_only=True)

    # Load per-vector metadata (contains baseline)
    metadata = {}
    baseline = 0.0
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        baseline = metadata.get('baseline', 0.0)
    else:
        logger.warning(f"No per-vector metadata found at {metadata_path}, baseline=0")

    return vector, baseline, metadata


def get_vector_source_info(experiment: str, trait: str, method: str, layer: int, component: str = "residual") -> Dict[str, Any]:
    """
    Get vector source info for use in results metadata.

    This is the standard format for recording where a vector came from.

    Args:
        experiment: Experiment name
        trait: Trait path
        method: Extraction method
        layer: Layer number
        component: Component type

    Returns:
        Dict with vector source info for embedding in results
    """
    metadata = load_vector_metadata(experiment, trait)

    return {
        "model": metadata.get("extraction_model", "unknown"),
        "experiment": experiment,
        "trait": trait,
        "method": method,
        "layer": layer,
        "component": component
    }
