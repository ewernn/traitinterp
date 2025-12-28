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
from typing import Optional, Tuple, Dict, Any, List

import torch

from utils.paths import (
    get as get_path,
    get_vector_dir,
    get_vector_path,
    get_vector_metadata_path,
    get_steering_results_path,
    list_layers,
)

logger = logging.getLogger(__name__)

# Single source of truth for minimum coherence threshold in steering evaluation
MIN_COHERENCE = 70


def find_vector_method(
    experiment: str,
    trait: str,
    layer: int,
    component: str = "residual",
    position: str = "response[:]",
) -> Optional[str]:
    """
    Auto-detect best vector method for a layer.

    Args:
        experiment: Experiment name
        trait: Trait path
        layer: Layer number
        component: Component type
        position: Position string

    Returns:
        Method name if found, None otherwise
    """
    for method in ["probe", "mean_diff", "gradient"]:
        vector_path = get_vector_path(experiment, trait, method, layer, component, position)
        if vector_path.exists():
            return method
    return None


def get_best_layer(
    experiment: str,
    trait: str,
    component: str = "residual",
    position: str = "response[:]",
) -> dict:
    """
    Get best layer for a trait.

    Priority:
    1. Steering results (ground truth from steering/{trait}/{position}/results.json)
    2. Effect size (from extraction_evaluation.json all_results)
    3. Default (layer 16, probe method)

    Args:
        experiment: Experiment name
        trait: Trait path (e.g., "category/trait_name")
        component: Component type
        position: Position string

    Returns:
        Dict with 'layer', 'method', 'source', 'score'
        source is one of: 'steering', 'effect_size', 'default'
    """
    # 1. Check steering results first (ground truth)
    steering_result = _try_steering_result(experiment, trait, position)
    if steering_result:
        return steering_result

    # 2. Compute effect_size from all_results
    effect_size_result = _try_effect_size(experiment, trait, component, position)
    if effect_size_result:
        return effect_size_result

    # 3. Default fallback
    return {'layer': 16, 'method': 'probe', 'source': 'default', 'score': 0}


def get_top_N_vectors(
    experiment: str,
    trait: str,
    component: str = "residual",
    position: str = "response[:]",
    N: int = 3,
    methods: List[str] = None,
) -> list:
    """
    Get top N vectors for a trait, ranked by quality.

    Args:
        experiment: Experiment name
        trait: Trait path
        component: Component type
        position: Position string
        N: Number of vectors to return (default: 3)
        methods: List of methods to include (default: ['probe', 'mean_diff', 'gradient'])

    Returns:
        List of dicts with 'layer', 'method', 'source', 'score'
    """
    if methods is None:
        methods = ['probe', 'mean_diff', 'gradient']

    all_vectors = []
    seen = set()

    # 1. Collect from steering results (ground truth)
    steering_vectors = _get_all_steering_vectors(experiment, trait, position, methods)
    for v in steering_vectors:
        key = (v['method'], v['layer'])
        if key not in seen:
            seen.add(key)
            all_vectors.append(v)

    # 2. Collect from effect_size (fallback)
    effect_vectors = _get_all_effect_size_vectors(experiment, trait, component, position, methods)
    for v in effect_vectors:
        key = (v['method'], v['layer'])
        if key not in seen:
            seen.add(key)
            all_vectors.append(v)

    # 3. Sort by score descending
    all_vectors.sort(key=lambda x: x['score'], reverse=True)

    return all_vectors[:N]


def _get_all_steering_vectors(
    experiment: str,
    trait: str,
    position: str,
    methods: list,
) -> list:
    """Get all steering results for a trait across methods."""
    steering_path = get_steering_results_path(experiment, trait, position)
    if not steering_path.exists():
        return []

    results = []
    try:
        with open(steering_path) as f:
            data = json.load(f)
        baseline = data.get('baseline', {}).get('trait_mean', 0)

        for run in data.get('runs', []):
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


def _get_all_effect_size_vectors(
    experiment: str,
    trait: str,
    component: str,
    position: str,
    methods: list,
) -> list:
    """Get all effect_size results for a trait across methods."""
    eval_path = get_path('extraction_eval.evaluation', experiment=experiment)
    if not eval_path.exists():
        return []

    results = []
    try:
        with open(eval_path) as f:
            data = json.load(f)

        # Check if position/component match (skip if eval doesn't have these fields - old format)
        eval_position = data.get('position')
        eval_component = data.get('component')
        if eval_position and eval_position != position:
            return []
        if eval_component and eval_component != component:
            return []

        all_results = data.get('all_results', [])
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


def _try_steering_result(experiment: str, trait: str, position: str) -> Optional[dict]:
    """Try to get best layer from steering results (ground truth)."""
    steering_path = get_steering_results_path(experiment, trait, position)
    if not steering_path.exists():
        return None

    try:
        with open(steering_path) as f:
            data = json.load(f)
        baseline = data.get('baseline', {}).get('trait_mean', 0)
        best_run, best_delta = None, float('-inf')
        for run in data.get('runs', []):
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


def _try_effect_size(
    experiment: str,
    trait: str,
    component: str,
    position: str,
) -> Optional[dict]:
    """Fallback: use effect_size as heuristic (not a reliable steering predictor)."""
    eval_path = get_path('extraction_eval.evaluation', experiment=experiment)
    if not eval_path.exists():
        return None

    try:
        with open(eval_path) as f:
            data = json.load(f)

        # Check if position/component match (skip if eval doesn't have these fields - old format)
        eval_position = data.get('position')
        eval_component = data.get('component')
        if eval_position and eval_position != position:
            return None
        if eval_component and eval_component != component:
            return None

        results = data.get('all_results', [])
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


def load_vector_metadata(
    experiment: str,
    trait: str,
    method: str,
    component: str = "residual",
    position: str = "response[:]",
) -> Dict[str, Any]:
    """
    Load vector metadata for a trait/method.

    Args:
        experiment: Experiment name
        trait: Trait path
        method: Extraction method
        component: Component type
        position: Position string

    Returns:
        Dict with vector metadata (includes 'layers' dict with per-layer info)

    Raises:
        FileNotFoundError: If metadata.json doesn't exist
    """
    metadata_path = get_vector_metadata_path(experiment, trait, method, component, position)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No metadata for {experiment}/{trait}/{method}. "
            f"Re-run extraction to generate metadata, or create {metadata_path} manually."
        )

    with open(metadata_path) as f:
        return json.load(f)


def load_vector_with_baseline(
    experiment: str,
    trait: str,
    method: str,
    layer: int,
    component: str = "residual",
    position: str = "response[:]",
) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    """
    Load a vector with its baseline and per-vector metadata.

    Args:
        experiment: Experiment name
        trait: Trait path
        method: Extraction method
        layer: Layer number
        component: Component type
        position: Position string

    Returns:
        Tuple of (vector tensor, baseline float, layer metadata dict)

    Raises:
        FileNotFoundError: If vector file doesn't exist
    """
    vector_path = get_vector_path(experiment, trait, method, layer, component, position)

    if not vector_path.exists():
        raise FileNotFoundError(f"Vector not found: {vector_path}")

    vector = torch.load(vector_path, weights_only=True)

    # Load metadata from consolidated file
    baseline = 0.0
    layer_metadata = {}
    try:
        metadata = load_vector_metadata(experiment, trait, method, component, position)
        layer_info = metadata.get('layers', {}).get(str(layer), {})
        baseline = layer_info.get('baseline', 0.0)
        layer_metadata = layer_info
    except FileNotFoundError:
        logger.warning(f"No metadata found for {method}, baseline=0")

    return vector, baseline, layer_metadata
