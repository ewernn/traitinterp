"""
Best vector selection using steering evaluation results.

Scans available vectors, looks up steering results, and returns the best
vector(s) based on direction-aware delta with coherence filtering.

Usage:
    from utils.vector_selection import get_best_vector, get_best_vector_spec, load_trait_vectors
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
    get_model_variant,
    desanitize_position,
)
from utils.vectors import (
    MIN_COHERENCE, MIN_NATURALNESS,
    load_vector_metadata, load_vector_with_baseline,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Internal helpers
# =============================================================================

def _discover_vectors(
    experiment: str,
    trait: str,
    model_variant: str,
    component: str = None,
    position: str = None,
    layer: int = None,
) -> List[dict]:
    """Scan vector files and return list of candidates."""
    vectors_dir = get_path('extraction.vectors', experiment=experiment, trait=trait, model_variant=model_variant)
    if not vectors_dir.exists():
        return []

    candidates = []
    pattern = re.compile(r'^layer(\d+)\.pt$')

    for pt_file in vectors_dir.rglob('layer*.pt'):
        rel_parts = pt_file.relative_to(vectors_dir).parts
        if len(rel_parts) != 4:
            continue

        pos_sanitized, comp, method, filename = rel_parts
        match = pattern.match(filename)
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
    """Look up steering result for a vector candidate.

    Returns (delta, coefficient, direction) if found, None otherwise.
    """
    from utils.steering_results import load_results

    layer = candidate['layer']
    method = candidate['method']
    position = candidate['position']

    steering_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)
    if not steering_path.exists():
        return None

    # Check for stale results
    prompts_path = get_path('datasets.trait_steering', trait=trait)
    if prompts_path.exists() and prompts_path.stat().st_mtime > steering_path.stat().st_mtime:
        import warnings
        warnings.warn(
            f"Steering prompts modified after results for {trait}. Results may be stale."
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

    best_delta = None
    best_coef = None
    for run in runs:
        cfg = run.get('config', {})
        vectors = cfg.get('vectors', [])
        if not vectors:
            continue
        v = vectors[0]
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
    """Load naturalness scores if available. Returns {(layer, method, component): score}."""
    nat_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set).parent / "naturalness.json"
    if not nat_path.exists():
        return {}

    try:
        with open(nat_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

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
    min_delta: float = 0,
    prompt_set: str = "steering",
) -> dict:
    """Find best vector based on steering results.

    Returns dict with 'layer', 'method', 'position', 'component', 'source',
    'score', 'coefficient', 'direction', and optionally 'naturalness'.
    """
    if extraction_variant is None:
        extraction_variant = get_model_variant(experiment, None, mode="extraction")['name']
    if steering_variant is None:
        steering_variant = get_model_variant(experiment, None, mode="application")['name']

    candidates = _discover_vectors(experiment, trait, extraction_variant, component, position, layer)
    if not candidates:
        raise FileNotFoundError(
            f"No vectors found for {experiment}/{trait}/{extraction_variant}. Run extraction first."
        )

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
            f"Run: python steering/run_steering_eval.py --experiment {experiment} --trait {trait}"
        )

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
        else:
            best_nat = max(c.get('naturalness', 0) for c in scored)
            raise FileNotFoundError(
                f"All configs for {trait} below naturalness threshold {min_naturalness} "
                f"(best: {best_nat:.0f}). Fix steering questions or set min_naturalness=0."
            )

    best = max(scored, key=lambda c: c['score'] * (1 if c['direction'] == 'positive' else -1))

    if min_delta > 0 and abs(best['score']) < min_delta:
        raise FileNotFoundError(
            f"Best delta for {trait} is {best['score']:+.1f} (|{abs(best['score']):.1f}| < {min_delta}). "
            f"Vector too weak for reliable use."
        )

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
    """Get top N vectors for a trait, ranked by steering delta."""
    if extraction_variant is None:
        extraction_variant = get_model_variant(experiment, None, mode="extraction")['name']
    if steering_variant is None:
        steering_variant = get_model_variant(experiment, None, mode="application")['name']

    candidates = _discover_vectors(experiment, trait, extraction_variant, component, position, layer)

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

    scored.sort(key=lambda c: c['score'] * (1 if c['direction'] == 'positive' else -1), reverse=True)
    return [{k: v for k, v in c.items() if k != 'path'} for c in scored[:N]]


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
    min_delta: float = 0,
    prompt_set: str = "steering",
) -> Tuple[VectorSpec, Dict[str, Any]]:
    """Find best vector and return as VectorSpec."""
    best = get_best_vector(experiment, trait, extraction_variant, steering_variant, component, position, layer, min_coherence, min_delta=min_delta, prompt_set=prompt_set)

    spec = VectorSpec(
        layer=best['layer'],
        component=best['component'],
        position=best['position'],
        method=best['method'],
        weight=weight,
    )
    return spec, {'source': best['source'], 'score': best['score'], 'coefficient': best.get('coefficient')}


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
    min_delta: float = 0,
    prompt_set: str = "steering",
) -> Tuple[ProjectionConfig, Dict[str, Any]]:
    """Get ProjectionConfig for the best single vector."""
    spec, metadata = get_best_vector_spec(
        experiment, trait, extraction_variant, steering_variant, component, position, layer, weight, min_coherence, min_delta=min_delta, prompt_set=prompt_set
    )
    return ProjectionConfig(vectors=[spec]), metadata


def get_best_steering_responses_path(
    experiment: str,
    trait: str,
    extraction_variant: str = None,
    steering_variant: str = None,
    position: str = None,
    min_coherence: int = MIN_COHERENCE,
    prompt_set: str = "steering",
) -> Optional[Path]:
    """Get path to the response file for the best steering configuration."""
    if steering_variant is None:
        steering_variant = get_model_variant(experiment, None, mode="application")['name']

    try:
        best = get_best_vector(experiment, trait, extraction_variant, steering_variant, position=position, min_coherence=min_coherence, prompt_set=prompt_set)
    except FileNotFoundError:
        return None

    if best['source'] != 'steering' or best['coefficient'] is None:
        return None

    responses_dir = get_steering_responses_dir(experiment, trait, steering_variant, best['position'], prompt_set)
    if not responses_dir.exists():
        return None

    layer, coef = best['layer'], best['coefficient']
    for f in responses_dir.iterdir():
        if f.name.startswith(f"L{layer}_c") and f.suffix == '.json':
            try:
                parts = f.stem.split('_')
                file_coef = float(parts[1][1:])
                if abs(file_coef - coef) < 0.5:
                    return f
            except (IndexError, ValueError):
                continue

    return None


def load_trait_vectors(experiment, extraction_variant, traits, component, layers_spec,
                       available_layers=None):
    """Load trait vectors and group by layer for batched projection.

    Returns:
        trait_vectors: {(cat, trait): [(vector, method, path, layer, metadata,
            source, baseline, position), ...]}
        vectors_by_layer: {layer: Tensor[n_vectors, hidden_dim]}
        hook_index: {(layer, slot_in_stacked): (category, trait_name, vec_list_idx)}
    """
    from utils.process_activations import resolve_layers

    trait_vectors = {}
    vectors_for_layer = {}

    for trait in traits:
        parts = trait.split('/')
        category, trait_name = parts[0], '/'.join(parts[1:]) if len(parts) > 2 else parts[1]
        key = (category, trait_name)

        try:
            vec_spec, spec_metadata = get_best_vector_spec(experiment, trait, extraction_variant=extraction_variant)
        except FileNotFoundError:
            print(f"  Warning: no vectors/steering results for {trait}, skipping")
            continue

        best_layer = vec_spec.layer
        best_method = vec_spec.method
        position = vec_spec.position
        selection_source = spec_metadata.get('source', 'steering')

        if available_layers is None:
            concrete_layers = [best_layer] if best_layer else []
        else:
            concrete_layers = resolve_layers(
                layers_spec or "best,best+5", best_layer, set(available_layers)
            )

        vector_list = []
        for layer in concrete_layers:
            vector, baseline, vec_metadata = load_vector_with_baseline(
                experiment, trait, best_method, layer, extraction_variant,
                component, position,
            )

            vec_list_idx = len(vector_list)
            vector_list.append((vector, best_method, None, layer, vec_metadata,
                                selection_source, baseline, position))

            vectors_for_layer.setdefault(layer, []).append(
                (vector, category, trait_name, vec_list_idx))

        if vector_list:
            trait_vectors[key] = vector_list

    vectors_by_layer = {}
    hook_index = {}
    for layer, vecs in vectors_for_layer.items():
        stacked = torch.stack([v[0] for v in vecs])
        vectors_by_layer[layer] = stacked
        for idx, (_, cat, trait_name, vec_list_idx) in enumerate(vecs):
            hook_index[(layer, idx)] = (cat, trait_name, vec_list_idx)

    return trait_vectors, vectors_by_layer, hook_index
