"""
Utility functions for working with trait vectors.

Single source of truth for vector selection.

Priority:
1. Steering results (ground truth - actual behavioral validation)
2. Effect size (from extraction_evaluation.json)

Note: Effect size is a rough heuristic, not a reliable predictor
of steering success. Always run steering evaluation for ground truth.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import torch

from utils.paths import (
    get as get_path,
    get_vector_path,
    get_vector_metadata_path,
    get_steering_results_path,
    desanitize_position,
)

logger = logging.getLogger(__name__)

# Single source of truth for minimum coherence threshold in steering evaluation
MIN_COHERENCE = 70


# =============================================================================
# Vector Discovery & Scoring
# =============================================================================

def _discover_vectors(
    experiment: str,
    trait: str,
    component: str = None,
    position: str = None,
) -> List[dict]:
    """
    Scan vector files and return list of candidates.

    Args:
        experiment: Experiment name
        trait: Trait path
        component: Filter to this component (or None for all)
        position: Filter to this position (or None for all)

    Returns:
        List of dicts with 'layer', 'method', 'position', 'component', 'path'
    """
    vectors_dir = get_path('extraction.vectors', experiment=experiment, trait=trait)
    if not vectors_dir.exists():
        return []

    candidates = []
    pattern = re.compile(r'^layer(\d+)\.pt$')

    for pt_file in vectors_dir.rglob('layer*.pt'):
        # Parse path: vectors/{position}/{component}/{method}/layer{N}.pt
        rel_parts = pt_file.relative_to(vectors_dir).parts
        if len(rel_parts) != 4:
            continue

        pos_sanitized, comp, method, filename = rel_parts
        match = pattern.match(filename)
        if not match:
            continue

        layer = int(match.group(1))
        pos = desanitize_position(pos_sanitized)

        # Filter if specified
        if position and pos != position:
            continue
        if component and comp != component:
            continue

        candidates.append({
            'layer': layer,
            'method': method,
            'position': pos,
            'component': comp,
            'path': pt_file,
        })

    return candidates


def _score_vector(
    experiment: str,
    trait: str,
    candidate: dict,
    min_coherence: int = MIN_COHERENCE,
) -> Tuple[float, str]:
    """
    Get score for a vector candidate.

    Checks steering results first (ground truth), falls back to effect_size.

    Returns:
        (score, source) where source is 'steering', 'effect_size', or 'none'
    """
    layer = candidate['layer']
    method = candidate['method']
    position = candidate['position']

    # 1. Try steering results
    steering_path = get_steering_results_path(experiment, trait, position)
    if steering_path.exists():
        try:
            with open(steering_path) as f:
                data = json.load(f)
            baseline = data.get('baseline', {}).get('trait_mean', 0)

            for run in data.get('runs', []):
                cfg = run.get('config', {})
                if (cfg.get('layers') == [layer] and
                    cfg.get('methods', ['probe'])[0] == method):
                    result = run.get('result', {})
                    coherence = result.get('coherence_mean', 0)
                    if coherence >= min_coherence:
                        trait_mean = result.get('trait_mean', 0)
                        return trait_mean - baseline, 'steering'
        except (json.JSONDecodeError, KeyError):
            pass

    # 2. Try effect_size
    eval_path = get_path('extraction_eval.evaluation', experiment=experiment)
    if eval_path.exists():
        try:
            with open(eval_path) as f:
                data = json.load(f)

            for r in data.get('all_results', []):
                if (r.get('trait') == trait and
                    r.get('layer') == layer and
                    r.get('method') == method and
                    r.get('val_effect_size')):
                    return r['val_effect_size'], 'effect_size'
        except (json.JSONDecodeError, KeyError):
            pass

    return 0.0, 'none'


# =============================================================================
# Public API
# =============================================================================

def get_best_vector(
    experiment: str,
    trait: str,
    component: str = None,
    position: str = None,
    min_coherence: int = MIN_COHERENCE,
) -> dict:
    """
    Find best vector, optionally filtering by position/component.

    Args:
        experiment: Experiment name
        trait: Trait path (e.g., "category/trait_name")
        component: Component type, or None to search all
        position: Position string, or None to search all
        min_coherence: Minimum coherence for steering results (default: 70)

    Returns:
        Dict with 'layer', 'method', 'position', 'component', 'source', 'score'

    Raises:
        FileNotFoundError: If no vectors found or no evaluation results
    """
    candidates = _discover_vectors(experiment, trait, component, position)

    if not candidates:
        raise FileNotFoundError(
            f"No vectors found for {experiment}/{trait}. Run extraction first."
        )

    # Score each candidate
    for c in candidates:
        c['score'], c['source'] = _score_vector(experiment, trait, c, min_coherence)

    # Filter to those with scores, then find best
    scored = [c for c in candidates if c['source'] != 'none']

    if not scored:
        raise FileNotFoundError(
            f"No steering results or extraction_evaluation.json found for {experiment}/{trait}. "
            f"Run: python analysis/vectors/extraction_evaluation.py --experiment {experiment}"
        )

    best = max(scored, key=lambda c: c['score'])
    # Remove 'path' from result (internal detail)
    return {k: v for k, v in best.items() if k != 'path'}


def get_top_N_vectors(
    experiment: str,
    trait: str,
    component: str = None,
    position: str = None,
    N: int = 3,
    min_coherence: int = MIN_COHERENCE,
) -> List[dict]:
    """
    Get top N vectors for a trait, ranked by quality.

    Args:
        experiment: Experiment name
        trait: Trait path
        component: Component type, or None to search all
        position: Position string, or None to search all
        N: Number of vectors to return (default: 3)
        min_coherence: Minimum coherence for steering results

    Returns:
        List of dicts with 'layer', 'method', 'position', 'component', 'source', 'score'
    """
    candidates = _discover_vectors(experiment, trait, component, position)

    # Score each candidate
    for c in candidates:
        c['score'], c['source'] = _score_vector(experiment, trait, c, min_coherence)

    # Filter to scored, sort by score descending
    scored = [c for c in candidates if c['source'] != 'none']
    scored.sort(key=lambda c: c['score'], reverse=True)

    # Remove 'path' from results
    return [{k: v for k, v in c.items() if k != 'path'} for c in scored[:N]]


def find_vector_method(
    experiment: str,
    trait: str,
    layer: int,
    component: str = "residual",
    position: str = "response[:]",
) -> Optional[str]:
    """
    Auto-detect vector method for a specific layer.

    Returns:
        Method name if found, None otherwise
    """
    for method in ["probe", "mean_diff", "gradient"]:
        vector_path = get_vector_path(experiment, trait, method, layer, component, position)
        if vector_path.exists():
            return method
    return None


# =============================================================================
# Vector Loading
# =============================================================================

def load_vector_metadata(
    experiment: str,
    trait: str,
    method: str,
    component: str = "residual",
    position: str = "response[:]",
) -> Dict[str, Any]:
    """
    Load vector metadata for a trait/method.

    Returns:
        Dict with vector metadata (includes 'layers' dict with per-layer info)

    Raises:
        FileNotFoundError: If metadata.json doesn't exist
    """
    metadata_path = get_vector_metadata_path(experiment, trait, method, component, position)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No metadata for {experiment}/{trait}/{method}. "
            f"Re-run extraction to generate metadata."
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
