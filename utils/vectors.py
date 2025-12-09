"""
Utility functions for working with trait vectors.

Single source of truth for best layer selection.

Priority:
1. Cached result in extraction_evaluation.json (if available)
2. Steering results (ground truth)
3. Effect size (best proxy, r=0.898 correlation with steering)
4. Default (layer 16, probe method)
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch

from utils.paths import get as get_path

logger = logging.getLogger(__name__)


def get_best_layer(experiment: str, trait: str) -> dict:
    """
    Get best layer for a trait.

    Args:
        experiment: Experiment name
        trait: Trait path (e.g., "category/trait_name")

    Returns:
        Dict with 'layer', 'method', 'source', 'score'
        source is one of: 'cached', 'steering', 'effect_size', 'default'

    Example:
        >>> best = get_best_layer('gemma-2-2b-it', 'epistemic/optimism')
        >>> print(f"L{best['layer']} {best['method']} (from {best['source']}: {best['score']:.1f})")
    """
    # 1. Check for cached result in extraction_evaluation.json
    eval_path = get_path('extraction_eval.evaluation', experiment=experiment)
    if eval_path.exists():
        try:
            with open(eval_path) as f:
                data = json.load(f)
            best_vectors = data.get('best_vectors', {})
            if trait in best_vectors:
                result = best_vectors[trait]
                return {
                    'layer': result['layer'],
                    'method': result['method'],
                    'source': result.get('source', 'cached'),
                    'score': result.get('score', 0)
                }
        except (json.JSONDecodeError, KeyError):
            pass

    # 2. Compute on-the-fly (fallback if not cached)
    return _compute_best_layer(experiment, trait)


def _compute_best_layer(experiment: str, trait: str) -> dict:
    """Compute best layer on-the-fly (used when not cached)."""

    # Try steering results (ground truth)
    steering_path = get_path('steering.results', experiment=experiment, trait=trait)
    if steering_path.exists():
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
                    if trait_mean is not None and coherence > 70:
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

    # Fall back to effect_size (best proxy for steering, r=0.898)
    eval_path = get_path('extraction_eval.evaluation', experiment=experiment)
    if eval_path.exists():
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

    # Default fallback
    return {'layer': 16, 'method': 'probe', 'source': 'default', 'score': 0}


def compute_all_best_layers(experiment: str) -> dict:
    """
    Compute best layers for all traits in an experiment.

    Used by extraction_evaluation.py to populate best_vectors.

    Returns:
        Dict mapping trait -> {'layer': int, 'method': str, 'source': str, 'score': float}
    """
    eval_path = get_path('extraction_eval.evaluation', experiment=experiment)
    if not eval_path.exists():
        return {}

    try:
        with open(eval_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

    # Get unique traits
    traits = set(r['trait'] for r in data.get('all_results', []) if 'trait' in r)

    # Compute best for each (using on-the-fly computation, not cache)
    return {trait: _compute_best_layer(experiment, trait) for trait in traits}


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
