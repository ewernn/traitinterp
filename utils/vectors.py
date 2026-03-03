"""
Utility functions for working with trait vectors.

Single source of truth for vector selection based on steering results.
Steering evaluation provides ground truth for vector quality.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import torch

from core.types import VectorSpec, ProjectionConfig
from utils.paths import (
    get,
    get as get_path,
    get_vector_path,
    get_vector_metadata_path,
    get_steering_results_path,
    get_steering_responses_dir,
    get_model_variant,
    desanitize_position,
    sanitize_position,
)

logger = logging.getLogger(__name__)

# Single source of truth for minimum coherence threshold in steering evaluation
MIN_COHERENCE = 77

# Minimum naturalness score (filters AI-mode, robotic responses)
# Only applied when naturalness.json exists for the trait
MIN_NATURALNESS = 50


# =============================================================================
# Vector Loading
# =============================================================================

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
    """
    Load cached activation norms from extraction_evaluation.json for a specific component.

    Returns:
        {layer: norm} or empty dict if not available
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
                return {}  # Flat norms are residual-only, no data for this component

        return {}
    except (json.JSONDecodeError, KeyError):
        return {}


# =============================================================================
# Vector Discovery & Steering Results
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


def _get_steering_result(
    experiment: str,
    trait: str,
    model_variant: str,
    candidate: dict,
    min_coherence: int = MIN_COHERENCE,
    prompt_set: str = "steering",
) -> Optional[Tuple[float, float, str]]:
    """
    Look up steering result for a vector candidate.

    Finds the BEST run across all coefficients for the given layer/method
    that meets the coherence threshold. Direction-aware: picks largest
    positive delta for positive direction, most negative delta for negative.

    Returns:
        (delta, coefficient, direction) if found, None otherwise
    """
    from analysis.steering.results import load_results

    layer = candidate['layer']
    method = candidate['method']
    position = candidate['position']

    steering_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)
    if not steering_path.exists():
        return None

    # Check for stale results (prompts modified after results)
    prompts_path = get_path('datasets.trait_steering', trait=trait)
    if prompts_path.exists() and prompts_path.stat().st_mtime > steering_path.stat().st_mtime:
        import warnings
        warnings.warn(
            f"Steering prompts modified after results for {trait}. "
            f"Results may be stale. Re-run steering evaluation."
        )

    try:
        results_data = load_results(experiment, trait, model_variant, position, prompt_set)
    except (json.JSONDecodeError, IOError, ValueError):
        return None

    direction = results_data.get("direction", "positive")
    sign = 1 if direction == "positive" else -1
    baseline_result = results_data.get("baseline")
    baseline_trait_mean = baseline_result.get("trait_mean", 0) if baseline_result else 0
    runs = results_data.get("runs", [])

    # Find BEST run across all coefficients for this layer/method
    # Direction-aware: positive maximizes delta, negative minimizes delta
    best_delta = None
    best_coef = None
    for run in runs:
        cfg = run.get('config', {})
        vectors = cfg.get('vectors', [])
        if not vectors:
            continue
        v = vectors[0]  # Single-vector config
        if (v.get('layer') == layer
                    and v.get('method', 'probe') == method
                    and v.get('component', 'residual') == candidate.get('component', 'residual')):
            result = run.get('result', {})
            coherence = result.get('coherence_mean', 0)
            if coherence >= min_coherence:
                trait_mean = result.get('trait_mean', 0)
                delta = trait_mean - baseline_trait_mean
                if best_delta is None or delta * sign > best_delta * sign:
                    best_delta = delta
                    best_coef = v.get('weight')

    if best_delta is not None:
        return best_delta, best_coef, direction
    return None


def _load_naturalness_scores(
    experiment: str,
    trait: str,
    model_variant: str,
    position: str,
    prompt_set: str = "steering",
) -> Dict[Tuple[int, str, str], float]:
    """
    Load naturalness scores if available.

    Returns:
        {(layer, method, component): best_naturalness_score} or empty dict
        For layers with multiple coefficients scored, returns the best score.
    """
    nat_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set).parent / "naturalness.json"
    if not nat_path.exists():
        return {}

    try:
        with open(nat_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

    # Group by (layer, method, component), keep best naturalness per group
    scores = {}
    for config_data in data.get("scores", {}).values():
        key = (config_data["layer"], config_data["method"], config_data["component"])
        mean = config_data.get("mean", 0)
        if key not in scores or mean > scores[key]:
            scores[key] = mean

    return scores


# =============================================================================
# Public API
# =============================================================================

def get_best_vector(
    experiment: str,
    trait: str,
    extraction_variant: str = None,
    steering_variant: str = None,
    component: str = None,
    position: str = None,
    layer: int = None,
    min_coherence: int = MIN_COHERENCE,
    min_naturalness: int = MIN_NATURALNESS,
    prompt_set: str = "steering",
) -> dict:
    """
    Find best vector based on steering results.

    Args:
        experiment: Experiment name
        trait: Trait path (e.g., "category/trait_name")
        extraction_variant: Variant for vectors (default: from config defaults.extraction)
        steering_variant: Variant for steering results (default: from config defaults.application)
        component: Component type, or None to search all
        position: Position string, or None to search all
        layer: Layer number, or None to search all
        min_coherence: Minimum coherence threshold (default: MIN_COHERENCE)
        min_naturalness: Minimum naturalness score (default: MIN_NATURALNESS).
            Only applied when naturalness.json exists. Set to 0 to disable.
        prompt_set: Prompt set for steering results (default: "steering")

    Returns:
        Dict with 'layer', 'method', 'position', 'component', 'source', 'score',
        'coefficient', 'direction', and optionally 'naturalness'

    Raises:
        FileNotFoundError: If no vectors found or no steering results
    """
    # Resolve variants from config if not specified
    if extraction_variant is None:
        extraction_variant = get_model_variant(experiment, None, mode="extraction")['name']
    if steering_variant is None:
        steering_variant = get_model_variant(experiment, None, mode="application")['name']

    candidates = _discover_vectors(experiment, trait, extraction_variant, component, position, layer)

    if not candidates:
        raise FileNotFoundError(
            f"No vectors found for {experiment}/{trait}/{extraction_variant}. Run extraction first."
        )

    # Look up steering results for each candidate
    scored = []
    for c in candidates:
        result = _get_steering_result(experiment, trait, steering_variant, c, min_coherence, prompt_set)
        if result is not None:
            delta, coefficient, direction = result
            c['score'] = delta
            c['direction'] = direction
            c['source'] = 'steering'
            c['coefficient'] = coefficient
            scored.append(c)

    if not scored:
        raise FileNotFoundError(
            f"No steering results found for {experiment}/{trait}. "
            f"Run: python analysis/steering/evaluate.py --experiment {experiment} --trait {trait}"
        )

    # Load naturalness scores if available
    nat_scores = {}
    positions_seen = set(c['position'] for c in scored)
    for pos in positions_seen:
        nat_scores.update(
            _load_naturalness_scores(experiment, trait, steering_variant, pos, prompt_set)
        )

    # Annotate candidates with naturalness
    for c in scored:
        key = (c['layer'], c['method'], c['component'])
        if key in nat_scores:
            c['naturalness'] = nat_scores[key]

    # Filter by naturalness if scores exist and threshold > 0
    if nat_scores and min_naturalness > 0:
        filtered = [c for c in scored if c.get('naturalness', 100) >= min_naturalness]
        if filtered:
            scored = filtered
        else:
            best_nat = max(c.get('naturalness', 0) for c in scored)
            raise FileNotFoundError(
                f"All configs for {trait} below naturalness threshold {min_naturalness} "
                f"(best: {best_nat:.0f}). Fix steering questions or set min_naturalness=0."
            )

    # Direction-aware: pick largest delta for positive, most negative for negative
    best = max(scored, key=lambda c: c['score'] * (1 if c['direction'] == 'positive' else -1))
    # Remove 'path' from result (internal detail)
    return {k: v for k, v in best.items() if k != 'path'}


def get_top_N_vectors(
    experiment: str,
    trait: str,
    extraction_variant: str = None,
    steering_variant: str = None,
    component: str = None,
    position: str = None,
    layer: int = None,
    N: int = 3,
    min_coherence: int = MIN_COHERENCE,
    min_naturalness: int = MIN_NATURALNESS,
    prompt_set: str = "steering",
) -> List[dict]:
    """
    Get top N vectors for a trait, ranked by steering delta.

    Args:
        experiment: Experiment name
        trait: Trait path
        extraction_variant: Variant for vectors (default: from config defaults.extraction)
        steering_variant: Variant for steering results (default: from config defaults.application)
        component: Component type, or None to search all
        position: Position string, or None to search all
        layer: Layer number, or None to search all
        N: Number of vectors to return (default: 3)
        min_coherence: Minimum coherence threshold
        min_naturalness: Minimum naturalness score (0 to disable)
        prompt_set: Prompt set for steering results (default: "steering")

    Returns:
        List of dicts with 'layer', 'method', 'position', 'component', 'source',
        'score', 'coefficient', and optionally 'naturalness'
    """
    # Resolve variants from config if not specified
    if extraction_variant is None:
        extraction_variant = get_model_variant(experiment, None, mode="extraction")['name']
    if steering_variant is None:
        steering_variant = get_model_variant(experiment, None, mode="application")['name']

    candidates = _discover_vectors(experiment, trait, extraction_variant, component, position, layer)

    # Look up steering results for each candidate
    scored = []
    for c in candidates:
        result = _get_steering_result(experiment, trait, steering_variant, c, min_coherence, prompt_set)
        if result is not None:
            delta, coefficient, direction = result
            c['score'] = delta
            c['direction'] = direction
            c['source'] = 'steering'
            c['coefficient'] = coefficient
            scored.append(c)

    # Load and apply naturalness scores
    nat_scores = {}
    for pos in set(c['position'] for c in scored):
        nat_scores.update(
            _load_naturalness_scores(experiment, trait, steering_variant, pos, prompt_set)
        )
    for c in scored:
        key = (c['layer'], c['method'], c['component'])
        if key in nat_scores:
            c['naturalness'] = nat_scores[key]

    if nat_scores and min_naturalness > 0:
        filtered = [c for c in scored if c.get('naturalness', 100) >= min_naturalness]
        if filtered:
            scored = filtered

    # Sort by score: direction-aware (largest effect first)
    scored.sort(key=lambda c: c['score'] * (1 if c['direction'] == 'positive' else -1), reverse=True)

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
    extraction_variant: str = None,
    steering_variant: str = None,
    position: str = None,
    min_coherence: int = MIN_COHERENCE,
    prompt_set: str = "steering",
) -> Optional[Path]:
    """
    Get path to the response file for the best steering configuration.

    Args:
        experiment: Experiment name
        trait: Trait path
        extraction_variant: Variant for vectors (default: from config)
        steering_variant: Variant for steering results (default: from config)
        position: Position string, or None to search all
        min_coherence: Minimum coherence for steering results
        prompt_set: Prompt set for steering results (default: "steering")

    Returns:
        Path to response JSON file, or None if not found
    """
    # Resolve steering_variant for response path
    if steering_variant is None:
        steering_variant = get_model_variant(experiment, None, mode="application")['name']

    try:
        best = get_best_vector(experiment, trait, extraction_variant, steering_variant, position=position, min_coherence=min_coherence, prompt_set=prompt_set)
    except FileNotFoundError:
        return None

    if best['source'] != 'steering' or best['coefficient'] is None:
        return None

    layer = best['layer']
    coef = best['coefficient']
    pos = best['position']

    # Use proper path helper
    responses_dir = get_steering_responses_dir(experiment, trait, steering_variant, pos, prompt_set)

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
    extraction_variant: str = None,
    steering_variant: str = None,
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
        extraction_variant: Variant for vectors (default: from config)
        steering_variant: Variant for steering results (default: from config)
        component: Filter by component (or None for all)
        position: Filter by position (or None for all)
        layer: Filter by layer (or None for all)
        weight: Weight to assign to the VectorSpec (default 1.0)
        min_coherence: Minimum coherence for steering results
        prompt_set: Prompt set for steering results (default: "steering")

    Returns:
        Tuple of (VectorSpec, metadata dict with 'source', 'score', 'coefficient')
    """
    best = get_best_vector(experiment, trait, extraction_variant, steering_variant, component, position, layer, min_coherence, prompt_set=prompt_set)

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
    extraction_variant: str = None,
    steering_variant: str = None,
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
        experiment, trait, extraction_variant, steering_variant, component, position, layer, weight, min_coherence, prompt_set
    )
    return ProjectionConfig(vectors=[spec]), metadata
