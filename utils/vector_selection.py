"""
Vector selection: find best vector(s) for a trait using steering evaluation results.

Input:
    Vector files on disk (from extraction pipeline)
    Steering evaluation results (from steering pipeline)

Output:
    VectorResult dataclasses with layer, method, position, component, score, direction, coefficient

Usage:
    from utils.vector_selection import select_vector, select_vectors, get_best_vector_spec

    # Best single vector (returns VectorResult)
    best = select_vector(experiment, trait)
    print(best.layer, best.method, best.score)

    # Top 3 vectors (one per layer)
    top = select_vectors(experiment, trait, n=3)

    # Manual selection (no steering results needed)
    specific = select_vector(experiment, trait, layer=45, method="probe")
"""

import json
import logging
from typing import Optional, Tuple, Dict, Any, List

import torch

from core.types import VectorSpec, JudgeResult, VectorResult
from utils.paths import (
    get as get_path,
    get_steering_results_path,
    get_model_variant,
)
from utils.vectors import (
    MIN_COHERENCE,
    discover_vectors,
    load_vector_with_baseline,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Internal helpers
# =============================================================================

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
    from utils.paths import content_hash

    layer = candidate['layer']
    method = candidate['method']
    position = candidate['position']

    steering_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)
    if not steering_path.exists():
        return None

    try:
        results_data = load_results(experiment, trait, model_variant, position, prompt_set)
    except (json.JSONDecodeError, IOError, ValueError):
        return None

    # Check for stale results (hash-based if available, mtime fallback)
    prompts_path = get_path('datasets.trait_steering', trait=trait)
    if results_data.prompts_hash:
        current_hash = content_hash(prompts_path)
        if current_hash and current_hash != results_data.prompts_hash:
            import warnings
            warnings.warn(
                f"Steering prompts changed since results were recorded for {trait}. "
                f"Re-run: python steering/run_steering_eval.py --experiment {experiment} --trait {trait}"
            )
    elif prompts_path.exists() and prompts_path.stat().st_mtime > steering_path.stat().st_mtime:
        import warnings
        warnings.warn(
            f"Steering prompts modified after results for {trait}. Results may be stale."
        )

    direction = results_data.direction
    sign = 1 if direction == "positive" else -1
    baseline_result = results_data.baseline or JudgeResult.empty()
    baseline_trait_mean = baseline_result.trait_mean or 0

    best_delta = None
    best_coef = None
    for run in results_data.runs:
        v = run.config.vectors[0]
        if (v.layer == layer
                    and v.method == method
                    and v.component == candidate.get('component', 'residual')):
            if (run.result.coherence_mean or 0) >= min_coherence:
                trait_mean = run.result.trait_mean or 0
                delta = trait_mean - baseline_trait_mean
                if best_delta is None or delta * sign > best_delta * sign:
                    best_delta = delta
                    best_coef = v.weight

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


def _rank_value(candidate: dict, sort_by: str, sign: int) -> float:
    """Compute ranking value for a candidate. Currently only supports 'delta'."""
    score = candidate.get('score')
    if score is None:
        return float('-inf')
    return score * sign


def _select_vectors(
    experiment: str,
    trait: str,
    n: int = 1,
    extraction_variant: str = None,
    steering_variant: str = None,
    component: str = None,
    position: str = None,
    layer: int = None,
    method: str = None,
    min_coherence: int = MIN_COHERENCE,
    min_naturalness: int = 0,
    min_delta: float = 0,
    sort_by: str = "delta",
    prompt_set: str = "steering",
) -> List[VectorResult]:
    """Core selection logic: discover, score, dedupe by layer, rank, return top N."""
    # 1. Resolve variants
    if extraction_variant is None:
        extraction_variant = get_model_variant(experiment, None, mode="extraction").name
    if steering_variant is None:
        steering_variant = get_model_variant(experiment, None, mode="application").name

    # 2. Discover candidates
    candidates = discover_vectors(experiment, trait, extraction_variant, component, position, layer, method)
    if not candidates:
        raise FileNotFoundError(
            f"No vectors found for {experiment}/{trait}/{extraction_variant}. Run extraction first."
        )

    # 3. Score via steering results
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

    if scored:
        # 4. Naturalness filter (only when explicitly requested)
        if min_naturalness > 0:
            nat_scores = {}
            for pos in set(c['position'] for c in scored):
                nat_scores.update(
                    _load_naturalness_scores(experiment, trait, steering_variant, pos, prompt_set)
                )
            if nat_scores:
                for c in scored:
                    key = (c['layer'], c['method'], c['component'])
                    if key in nat_scores:
                        c['naturalness'] = nat_scores[key]
                filtered = [c for c in scored if c.get('naturalness', 100) >= min_naturalness]
                if filtered:
                    scored = filtered

        # 5. Min delta gate
        if min_delta > 0:
            scored = [c for c in scored if abs(c['score']) >= min_delta]

        working = scored
    else:
        # No steering results — only allow unscored if user explicitly specified layer/method
        if layer is not None or method is not None:
            for c in candidates:
                c.update(score=None, direction=None, source='unscored', coefficient=None)
            working = candidates
        else:
            return []

    if not working:
        return []

    # 6. Deduplicate by layer — keep best candidate per layer
    direction = next((c['direction'] for c in working if c.get('direction')), 'positive')
    sign = 1 if direction == 'positive' else -1

    best_per_layer = {}
    for c in working:
        val = _rank_value(c, sort_by, sign)
        prev = best_per_layer.get(c['layer'])
        if prev is None or val > _rank_value(prev, sort_by, sign):
            best_per_layer[c['layer']] = c

    # 7. Sort and return top N
    ranked = sorted(best_per_layer.values(), key=lambda c: _rank_value(c, sort_by, sign), reverse=True)
    return [VectorResult(
        layer=c['layer'], method=c['method'], position=c['position'],
        component=c['component'], score=c.get('score'), direction=c.get('direction'),
        source=c.get('source', 'unscored'), coefficient=c.get('coefficient'),
        naturalness=c.get('naturalness'),
    ) for c in ranked[:n]]


# =============================================================================
# Public API
# =============================================================================

def select_vector(
    experiment: str,
    trait: str,
    extraction_variant: str = None,
    steering_variant: str = None,
    component: str = None,
    position: str = None,
    layer: int = None,
    method: str = None,
    min_coherence: int = MIN_COHERENCE,
    min_naturalness: int = 0,
    min_delta: float = 0,
    sort_by: str = "delta",
    prompt_set: str = "steering",
) -> VectorResult:
    """Find best vector for a trait. Returns VectorResult with layer, method, position,
    component, source, score, coefficient, direction, and optionally naturalness."""
    results = _select_vectors(
        experiment, trait, n=1,
        extraction_variant=extraction_variant, steering_variant=steering_variant,
        component=component, position=position, layer=layer, method=method,
        min_coherence=min_coherence, min_naturalness=min_naturalness,
        min_delta=min_delta, sort_by=sort_by, prompt_set=prompt_set,
    )
    if not results:
        raise FileNotFoundError(
            f"No suitable vectors found for {experiment}/{trait}. "
            f"Run: python steering/run_steering_eval.py --experiment {experiment} --trait {trait}"
        )
    return results[0]


def select_vectors(
    experiment: str,
    trait: str,
    n: int = 3,
    extraction_variant: str = None,
    steering_variant: str = None,
    component: str = None,
    position: str = None,
    layer: int = None,
    method: str = None,
    min_coherence: int = MIN_COHERENCE,
    min_naturalness: int = 0,
    min_delta: float = 0,
    sort_by: str = "delta",
    prompt_set: str = "steering",
) -> List[VectorResult]:
    """Get top N vectors for a trait, one per layer, ranked by sort_by."""
    return _select_vectors(
        experiment, trait, n=n,
        extraction_variant=extraction_variant, steering_variant=steering_variant,
        component=component, position=position, layer=layer, method=method,
        min_coherence=min_coherence, min_naturalness=min_naturalness,
        min_delta=min_delta, sort_by=sort_by, prompt_set=prompt_set,
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
    min_delta: float = 0,
    prompt_set: str = "steering",
) -> Tuple[VectorSpec, Dict[str, Any]]:
    """Find best vector and return as VectorSpec."""
    best = select_vector(
        experiment, trait,
        extraction_variant=extraction_variant, steering_variant=steering_variant,
        component=component, position=position, layer=layer,
        min_coherence=min_coherence, min_delta=min_delta, prompt_set=prompt_set,
    )
    spec = best.to_vector_spec(weight=weight)
    return spec, {'source': best.source, 'score': best.score, 'coefficient': best.coefficient}


def load_trait_vectors(experiment, extraction_variant, traits, component, layers_spec,
                       available_layers=None):
    """Load trait vectors and group by layer for batched projection.

    Returns:
        trait_vectors: {(cat, trait): [(vector, method, path, layer, metadata,
            source, baseline, position), ...]}
        vectors_by_layer: {layer: Tensor[n_vectors, hidden_dim]}
        hook_index: {(layer, slot_in_stacked): (category, trait_name, vec_list_idx)}
    """
    from utils.layers import resolve_layers

    trait_vectors = {}
    vectors_for_layer = {}

    for trait in traits:
        parts = trait.split('/')
        category, trait_name = parts[0], '/'.join(parts[1:]) if len(parts) > 2 else parts[1]
        key = (category, trait_name)

        try:
            best = select_vector(experiment, trait, extraction_variant=extraction_variant)
        except FileNotFoundError:
            print(f"  Warning: no vectors/steering results for {trait}, skipping")
            continue

        best_layer = best.layer
        best_method = best.method
        position = best.position
        selection_source = best.source or 'steering'

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
