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

from core.types import VectorSpec, ProjectionConfig
from utils.paths import (
    get as get_path,
    get_vector_path,
    get_vector_metadata_path,
    get_steering_results_path,
    get_steering_responses_dir,
    desanitize_position,
    sanitize_position,
)

logger = logging.getLogger(__name__)

# Single source of truth for minimum coherence threshold in steering evaluation
MIN_COHERENCE = 80  # Higher than PV paper (75) for better output quality


# =============================================================================
# Vector Discovery & Scoring
# =============================================================================

def _discover_vectors(
    experiment: str,
    trait: str,
    model_variant: str,
    component: str = None,
    position: str = None,
    layer: int = None,
) -> List[dict]:
    """
    Scan vector files and return list of candidates.

    Args:
        experiment: Experiment name
        trait: Trait path
        model_variant: Model variant name
        component: Filter to this component (or None for all)
        position: Filter to this position (or None for all)
        layer: Filter to this layer (or None for all)

    Returns:
        List of dicts with 'layer', 'method', 'position', 'component', 'path'
    """
    vectors_dir = get_path('extraction.vectors', experiment=experiment, trait=trait, model_variant=model_variant)
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

        file_layer = int(match.group(1))
        pos = desanitize_position(pos_sanitized)

        # Filter if specified
        if position and pos != position:
            continue
        if component and comp != component:
            continue
        if layer is not None and file_layer != layer:
            continue

        candidates.append({
            'layer': file_layer,
            'method': method,
            'position': pos,
            'component': comp,
            'path': pt_file,
        })

    return candidates


def _score_vector(
    experiment: str,
    trait: str,
    model_variant: str,
    candidate: dict,
    min_coherence: int = MIN_COHERENCE,
    prompt_set: str = "steering",
) -> Tuple[float, str, Optional[float]]:
    """
    Get score for a vector candidate.

    Checks steering results first (ground truth), falls back to effect_size.
    Finds the BEST run across all coefficients for the given layer/method.

    Returns:
        (score, source, coefficient) where:
        - source is 'steering', 'effect_size', or 'none'
        - coefficient is the optimal coefficient (None for effect_size/none)
    """
    layer = candidate['layer']
    method = candidate['method']
    position = candidate['position']

    # 1. Try steering results
    steering_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)
    if steering_path.exists():
        # Check for stale results (prompts modified after results)
        prompts_path = get_path('datasets.trait_steering', trait=trait)
        if prompts_path.exists() and prompts_path.stat().st_mtime > steering_path.stat().st_mtime:
            import warnings
            warnings.warn(
                f"Steering prompts modified after results for {trait}. "
                f"Results may be stale. Re-run steering evaluation."
            )
        try:
            with open(steering_path) as f:
                data = json.load(f)
            baseline = data.get('baseline', {}).get('trait_mean', 0)

            # Find BEST run across all coefficients for this layer/method
            best_delta = None
            best_coef = None
            for run in data.get('runs', []):
                cfg = run.get('config', {})
                vectors = cfg.get('vectors', [])
                if not vectors:
                    continue
                v = vectors[0]  # Single-vector config
                if v.get('layer') == layer and v.get('method', 'probe') == method:
                    result = run.get('result', {})
                    coherence = result.get('coherence_mean', 0)
                    if coherence >= min_coherence:
                        trait_mean = result.get('trait_mean', 0)
                        delta = trait_mean - baseline
                        if best_delta is None or delta > best_delta:
                            best_delta = delta
                            best_coef = v.get('weight')

            if best_delta is not None:
                return best_delta, 'steering', best_coef
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
                    return r['val_effect_size'], 'effect_size', None
        except (json.JSONDecodeError, KeyError):
            pass

    return 0.0, 'none', None


# =============================================================================
# Public API
# =============================================================================

def get_best_vector(
    experiment: str,
    trait: str,
    model_variant: str,
    component: str = None,
    position: str = None,
    layer: int = None,
    min_coherence: int = MIN_COHERENCE,
    prompt_set: str = "steering",
) -> dict:
    """
    Find best vector, optionally filtering by position/component/layer.

    Args:
        experiment: Experiment name
        trait: Trait path (e.g., "category/trait_name")
        model_variant: Model variant name
        component: Component type, or None to search all
        position: Position string, or None to search all
        layer: Layer number, or None to search all
        min_coherence: Minimum coherence for steering results (default: 75)
        prompt_set: Prompt set for steering results (default: "steering")

    Returns:
        Dict with 'layer', 'method', 'position', 'component', 'source', 'score', 'coefficient'
        (coefficient is None if source is not 'steering')

    Raises:
        FileNotFoundError: If no vectors found or no evaluation results
    """
    candidates = _discover_vectors(experiment, trait, model_variant, component, position, layer)

    if not candidates:
        raise FileNotFoundError(
            f"No vectors found for {experiment}/{trait}/{model_variant}. Run extraction first."
        )

    # Score each candidate
    for c in candidates:
        c['score'], c['source'], c['coefficient'] = _score_vector(experiment, trait, model_variant, c, min_coherence, prompt_set)

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
    model_variant: str,
    component: str = None,
    position: str = None,
    layer: int = None,
    N: int = 3,
    min_coherence: int = MIN_COHERENCE,
    prompt_set: str = "steering",
) -> List[dict]:
    """
    Get top N vectors for a trait, ranked by quality.

    Args:
        experiment: Experiment name
        trait: Trait path
        model_variant: Model variant name
        component: Component type, or None to search all
        position: Position string, or None to search all
        layer: Layer number, or None to search all
        N: Number of vectors to return (default: 3)
        min_coherence: Minimum coherence for steering results
        prompt_set: Prompt set for steering results (default: "steering")

    Returns:
        List of dicts with 'layer', 'method', 'position', 'component', 'source', 'score', 'coefficient'
    """
    candidates = _discover_vectors(experiment, trait, model_variant, component, position, layer)

    # Score each candidate
    for c in candidates:
        c['score'], c['source'], c['coefficient'] = _score_vector(experiment, trait, model_variant, c, min_coherence, prompt_set)

    # Filter to scored, sort by score descending
    scored = [c for c in candidates if c['source'] != 'none']
    scored.sort(key=lambda c: c['score'], reverse=True)

    # Remove 'path' from results
    return [{k: v for k, v in c.items() if k != 'path'} for c in scored[:N]]


def find_vector_method(
    experiment: str,
    trait: str,
    layer: int,
    model_variant: str,
    component: str = "residual",
    position: str = "response[:]",
) -> Optional[str]:
    """
    Auto-detect vector method for a specific layer.

    Returns:
        Method name if found, None otherwise
    """
    for method in ["probe", "mean_diff", "gradient"]:
        vector_path = get_vector_path(experiment, trait, method, layer, model_variant, component, position)
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
    model_variant: str,
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
    """
    Load a vector with its baseline and per-vector metadata.

    Returns:
        Tuple of (vector tensor, baseline float, layer metadata dict)

    Raises:
        FileNotFoundError: If vector file doesn't exist
    """
    vector_path = get_vector_path(experiment, trait, method, layer, model_variant, component, position)

    if not vector_path.exists():
        raise FileNotFoundError(f"Vector not found: {vector_path}")

    vector = torch.load(vector_path, weights_only=True)

    # Load metadata from consolidated file
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


def get_best_steering_responses_path(
    experiment: str,
    trait: str,
    model_variant: str,
    position: str = None,
    min_coherence: int = MIN_COHERENCE,
    prompt_set: str = "steering",
) -> Optional[Path]:
    """
    Get path to the response file for the best steering configuration.

    Args:
        experiment: Experiment name
        trait: Trait path
        model_variant: Model variant name
        position: Position string, or None to search all
        min_coherence: Minimum coherence for steering results
        prompt_set: Prompt set for steering results (default: "steering")

    Returns:
        Path to response JSON file, or None if not found
    """
    try:
        best = get_best_vector(experiment, trait, model_variant, position=position, min_coherence=min_coherence, prompt_set=prompt_set)
    except FileNotFoundError:
        return None

    if best['source'] != 'steering' or best['coefficient'] is None:
        return None

    layer = best['layer']
    coef = best['coefficient']
    pos = best['position']

    # Use proper path helper
    responses_dir = get_steering_responses_dir(experiment, trait, model_variant, pos, prompt_set)

    if not responses_dir.exists():
        return None

    # Find matching response file: L{layer}_c{coef}*.json
    # Coefficient in filename may be rounded differently, so match prefix
    for f in responses_dir.iterdir():
        if f.name.startswith(f"L{layer}_c") and f.suffix == '.json':
            # Parse coefficient from filename
            try:
                parts = f.stem.split('_')
                file_coef = float(parts[1][1:])  # Remove 'c' prefix
                if abs(file_coef - coef) < 0.5:  # Allow small rounding diff
                    return f
            except (IndexError, ValueError):
                continue

    return None


# =============================================================================
# VectorSpec-based API
# =============================================================================

def load_vector_from_spec(
    experiment: str,
    trait: str,
    spec: VectorSpec,
    model_variant: str,
) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    """
    Load a vector using a VectorSpec.

    Args:
        experiment: Experiment name
        trait: Trait path
        spec: VectorSpec identifying the vector
        model_variant: Model variant name

    Returns:
        Tuple of (vector tensor, baseline float, layer metadata dict)
    """
    return load_vector_with_baseline(
        experiment, trait, spec.method, spec.layer, model_variant, spec.component, spec.position
    )


def get_best_vector_spec(
    experiment: str,
    trait: str,
    model_variant: str,
    component: str = None,
    position: str = None,
    layer: int = None,
    weight: float = 1.0,
    min_coherence: int = MIN_COHERENCE,
    prompt_set: str = "steering",
) -> Tuple[VectorSpec, Dict[str, Any]]:
    """
    Find best vector and return as VectorSpec.

    Args:
        experiment: Experiment name
        trait: Trait path
        model_variant: Model variant name
        component: Filter by component (or None for all)
        position: Filter by position (or None for all)
        layer: Filter by layer (or None for all)
        weight: Weight to assign to the VectorSpec (default 1.0)
        min_coherence: Minimum coherence for steering results
        prompt_set: Prompt set for steering results (default: "steering")

    Returns:
        Tuple of (VectorSpec, metadata dict with 'source', 'score', 'coefficient')
    """
    best = get_best_vector(experiment, trait, model_variant, component, position, layer, min_coherence, prompt_set)

    spec = VectorSpec(
        layer=best['layer'],
        component=best['component'],
        position=best['position'],
        method=best['method'],
        weight=weight,
    )

    metadata = {
        'source': best['source'],
        'score': best['score'],
        'coefficient': best.get('coefficient'),
    }

    return spec, metadata


def get_best_projection_config(
    experiment: str,
    trait: str,
    model_variant: str,
    component: str = None,
    position: str = None,
    layer: int = None,
    weight: float = 1.0,
    min_coherence: int = MIN_COHERENCE,
    prompt_set: str = "steering",
) -> Tuple[ProjectionConfig, Dict[str, Any]]:
    """
    Get ProjectionConfig for the best single vector.

    Convenience function for single-vector projection/steering.

    Returns:
        Tuple of (ProjectionConfig, metadata dict)
    """
    spec, metadata = get_best_vector_spec(
        experiment, trait, model_variant, component, position, layer, weight, min_coherence, prompt_set
    )
    return ProjectionConfig(vectors=[spec]), metadata
