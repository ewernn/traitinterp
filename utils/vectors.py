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


def find_vector_method(vectors_dir: Path, layer: int, component: str = "residual") -> Optional[str]:
    """Auto-detect best vector method for a layer.

    Args:
        vectors_dir: Path to vectors directory
        layer: Layer number
        component: 'residual' (default) or 'attn_out'

    Returns:
        Method name if found, None otherwise
    """
    for method in ["probe", "mean_diff", "gradient"]:
        if component == "attn_out":
            # Check for attn_out vectors: attn_out_probe_layer8.pt
            if (vectors_dir / f"attn_out_{method}_layer{layer}.pt").exists():
                return method
        else:
            # Standard residual vectors: probe_layer8.pt
            if (vectors_dir / f"{method}_layer{layer}.pt").exists():
                return method
    return None

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


def get_top_N_vectors(experiment: str, trait: str, N: int = 3, methods: list = None) -> list:
    """
    Get top N vectors for a trait, ranked by quality.

    Useful for multi-vector projections where you want to compare
    different extraction methods on the same trait.

    Args:
        experiment: Experiment name
        trait: Trait path (e.g., "category/trait_name")
        N: Number of vectors to return (default: 3)
        methods: List of methods to include (default: ['probe', 'mean_diff', 'gradient'])

    Returns:
        List of dicts with 'layer', 'method', 'source', 'score'
        Sorted by score descending, deduplicated by (method, layer)

    Example:
        >>> vectors = get_top_N_vectors('gemma-2-2b', 'harm/refusal', N=3)
        >>> for v in vectors:
        ...     print(f"L{v['layer']} {v['method']} ({v['source']}: {v['score']:.2f})")
    """
    if methods is None:
        methods = ['probe', 'mean_diff', 'gradient']

    all_vectors = []
    seen = set()  # (method, layer) pairs for deduplication

    # 1. Collect from steering results (ground truth)
    steering_vectors = _get_all_steering_vectors(experiment, trait, methods)
    for v in steering_vectors:
        key = (v['method'], v['layer'])
        if key not in seen:
            seen.add(key)
            all_vectors.append(v)

    # 2. Collect from effect_size (fallback)
    effect_vectors = _get_all_effect_size_vectors(experiment, trait, methods)
    for v in effect_vectors:
        key = (v['method'], v['layer'])
        if key not in seen:
            seen.add(key)
            all_vectors.append(v)

    # 3. Sort by score descending
    all_vectors.sort(key=lambda x: x['score'], reverse=True)

    # 4. Take top N
    return all_vectors[:N]


def _get_all_steering_vectors(experiment: str, trait: str, methods: list) -> list:
    """Get all steering results for a trait across methods."""
    steering_path = get_path('steering.results', experiment=experiment, trait=trait)
    if not steering_path.exists():
        return []

    results = []
    try:
        with open(steering_path) as f:
            data = json.load(f)
        baseline = data.get('baseline', {}).get('trait_mean', 0)

        for run in data.get('runs', []):
            # Only single-layer runs
            if len(run.get('config', {}).get('layers', [])) != 1:
                continue

            method = run['config'].get('methods', ['probe'])[0]
            if method not in methods:
                continue

            trait_mean = run.get('result', {}).get('trait_mean')
            coherence = run.get('result', {}).get('coherence_mean', 0)

            if trait_mean is not None and coherence > MIN_COHERENCE:
                delta = trait_mean - baseline
                results.append({
                    'layer': run['config']['layers'][0],
                    'method': method,
                    'source': 'steering',
                    'score': delta
                })
    except (json.JSONDecodeError, KeyError):
        pass

    return results


def _get_all_effect_size_vectors(experiment: str, trait: str, methods: list) -> list:
    """Get all effect_size results for a trait across methods."""
    eval_path = get_path('extraction_eval.evaluation', experiment=experiment)
    if not eval_path.exists():
        return []

    results = []
    try:
        with open(eval_path) as f:
            all_results = json.load(f).get('all_results', [])

        for r in all_results:
            if r.get('trait') != trait:
                continue
            if r.get('method') not in methods:
                continue
            if not r.get('val_effect_size'):
                continue

            results.append({
                'layer': r['layer'],
                'method': r['method'],
                'source': 'effect_size',
                'score': r['val_effect_size']
            })
    except (json.JSONDecodeError, KeyError):
        pass

    return results


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
