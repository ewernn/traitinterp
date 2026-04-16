"""
Vector selection: find the best trait vector via a validation hierarchy.

Input:
    Vector files on disk (from the extraction pipeline)
    Steering evaluation results (from the steering pipeline)
    Extraction evaluation (in-distribution held-out metrics)

Output:
    VectorResult with layer, method, position, component, and any available
    scoring metadata (delta, coefficient, direction, coherence, source).

Usage:
    from utils.vector_selection import select_vector, select_vectors, get_best_vector_spec

    best = select_vector(experiment, trait)                  # best single vector
    top = select_vectors(experiment, trait, n=3)             # top N, one per layer
    spec, meta = get_best_vector_spec(experiment, trait)     # best as VectorSpec
"""

import json
import logging
import warnings
from typing import Optional, Tuple, Dict, Any, List

import torch

from core.types import VectorSpec, JudgeResult, VectorResult
from utils.paths import (
    get as get_path,
    get_steering_results_path,
    get_model_variant,
    content_hash,
)
from utils.vectors import (
    MIN_COHERENCE,
    discover_vectors,
    load_vector_with_baseline,
)
from utils.steering_results import load_results

logger = logging.getLogger(__name__)


# =============================================================================
# Scoring via steering results
# =============================================================================

def _warn_if_stale(trait: str, results_data, steering_path) -> None:
    """Warn when the steering.json prompts have changed since results were recorded."""
    prompts_path = get_path('datasets.trait_steering', trait=trait)
    if results_data.prompts_hash:
        current = content_hash(prompts_path)
        if current and current != results_data.prompts_hash:
            warnings.warn(
                f"Steering prompts changed since results were recorded for {trait}. "
                f"Re-run: python steering/run_steering_eval.py --trait {trait}"
            )
    elif prompts_path.exists() and prompts_path.stat().st_mtime > steering_path.stat().st_mtime:
        warnings.warn(f"Steering prompts modified after results for {trait}. Results may be stale.")


def _get_steering_result(
    experiment: str,
    trait: str,
    model_variant: str,
    candidate: dict,
    min_coherence: int,
    prompt_set: str,
) -> Optional[Tuple[float, float, str, float]]:
    """Return (delta, coefficient, direction, coherence) for the best coherent
    run of this candidate (layer, method, component) — or None if none qualify."""
    position = candidate['position']
    steering_path = get_steering_results_path(experiment, trait, model_variant, position, prompt_set)
    if not steering_path.exists():
        return None

    try:
        results_data = load_results(experiment, trait, model_variant, position, prompt_set)
    except (json.JSONDecodeError, IOError, ValueError):
        return None

    _warn_if_stale(trait, results_data, steering_path)

    direction = results_data.direction
    sign = 1 if direction == "positive" else -1
    baseline = (results_data.baseline or JudgeResult.empty()).trait_mean or 0

    best_delta = best_coef = best_coh = None
    for run in results_data.runs:
        v = run.config.vectors[0]
        if (v.layer, v.method, v.component) != (candidate['layer'], candidate['method'], candidate.get('component', 'residual')):
            continue
        coh = run.result.coherence_mean or 0
        if coh < min_coherence:
            continue
        delta = (run.result.trait_mean or 0) - baseline
        if best_delta is None or delta * sign > best_delta * sign:
            best_delta, best_coef, best_coh = delta, v.weight, coh

    if best_delta is None:
        return None
    return best_delta, best_coef, direction, best_coh


# =============================================================================
# Selection paths
# =============================================================================

def _select_vectors(
    experiment: str,
    trait: str,
    n: int,
    extraction_variant: Optional[str],
    steering_variant: Optional[str],
    component: Optional[str],
    position: Optional[str],
    layer: Optional[int],
    method: Optional[str],
    min_coherence: int,
    min_delta: float,
    prompt_set: str,
) -> List[VectorResult]:
    """Score candidates via steering results, dedupe per layer, return top n."""
    if extraction_variant is None:
        extraction_variant = get_model_variant(experiment, None, mode="extraction").name
    if steering_variant is None:
        steering_variant = get_model_variant(experiment, None, mode="application").name

    candidates = discover_vectors(experiment, trait, extraction_variant, component, position, layer, method)
    if not candidates:
        raise FileNotFoundError(
            f"No vectors found for {experiment}/{trait}/{extraction_variant}. Run extraction first."
        )

    scored = []
    for c in candidates:
        result = _get_steering_result(experiment, trait, steering_variant, c, min_coherence, prompt_set)
        if result is None:
            continue
        c['delta'], c['coefficient'], c['direction'], c['coherence'] = result
        c['source'] = 'steering'
        scored.append(c)

    # Without steering results, only proceed if caller asked for a specific (layer, method).
    if not scored:
        if layer is None and method is None:
            return []
        for c in candidates:
            c.update(delta=None, coefficient=None, direction=None, coherence=None, source='unscored')
        scored = candidates

    if min_delta > 0:
        scored = [c for c in scored if c['delta'] is not None and abs(c['delta']) >= min_delta]
    if not scored:
        return []

    direction = next((c['delta'] for c in scored if c.get('direction') == 'negative'), None)
    sign = -1 if direction is not None else 1

    best_per_layer: Dict[int, dict] = {}
    for c in scored:
        d = c.get('delta')
        rank = d * sign if d is not None else float('-inf')
        prev = best_per_layer.get(c['layer'])
        prev_rank = (prev['delta'] * sign) if prev and prev.get('delta') is not None else float('-inf')
        if prev is None or rank > prev_rank:
            best_per_layer[c['layer']] = c

    ranked = sorted(
        best_per_layer.values(),
        key=lambda c: (c['delta'] * sign) if c.get('delta') is not None else float('-inf'),
        reverse=True,
    )
    return [VectorResult(
        layer=c['layer'], method=c['method'], position=c['position'], component=c['component'],
        delta=c.get('delta'), direction=c.get('direction'), source=c.get('source', 'unscored'),
        coefficient=c.get('coefficient'), coherence=c.get('coherence'),
    ) for c in ranked[:n]]


def _select_from_extraction_eval(
    experiment: str,
    trait: str,
    component: Optional[str] = None,
    position: Optional[str] = None,
    layer: Optional[int] = None,
    method: Optional[str] = None,
) -> Optional[VectorResult]:
    """Fallback to extraction_evaluation.json's highest combined_score with polarity_correct.
    Returns None when the file is missing or no rows survive the caller's filters."""
    eval_path = get_path('extraction_eval.evaluation', experiment=experiment)
    if not eval_path.exists():
        return None
    try:
        data = json.loads(eval_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    filters = {'trait': trait, 'component': component, 'position': position, 'layer': layer, 'method': method}
    rows = [
        r for r in data.get('all_results', [])
        if r.get('polarity_correct')
        and all(r.get(k) == v for k, v in filters.items() if v is not None)
    ]
    if not rows:
        return None
    best = max(rows, key=lambda r: r.get('combined_score') or 0)
    return VectorResult(
        layer=best['layer'], method=best['method'], position=best['position'], component=best['component'],
        delta=None, direction=None, coefficient=None, coherence=None,
        source='extraction_eval',
    )


# =============================================================================
# Public API
# =============================================================================

def select_vector(
    experiment: str,
    trait: str,
    extraction_variant: Optional[str] = None,
    steering_variant: Optional[str] = None,
    component: Optional[str] = None,
    position: Optional[str] = None,
    layer: Optional[int] = None,
    method: Optional[str] = None,
    min_coherence: int = MIN_COHERENCE,
    min_delta: float = 0,
    prompt_set: str = "steering",
) -> VectorResult:
    """Return the best vector for a trait using the validation hierarchy:

        1. Causal steering score (gold standard)
        2. In-distribution validation effect size (extraction_evaluation.json)

    (OOD validation — the middle tier in the designed hierarchy — activates
    automatically once `ood_accuracy` / `ood_effect_size` are persisted in
    extraction_evaluation.json. Not wired in yet.)

    Raises FileNotFoundError only when both sources are missing.
    """
    results = _select_vectors(
        experiment, trait, n=1,
        extraction_variant=extraction_variant, steering_variant=steering_variant,
        component=component, position=position, layer=layer, method=method,
        min_coherence=min_coherence, min_delta=min_delta, prompt_set=prompt_set,
    )
    if results:
        return results[0]

    fallback = _select_from_extraction_eval(experiment, trait, component, position, layer, method)
    if fallback is not None:
        return fallback

    raise FileNotFoundError(
        f"No suitable vectors found for {experiment}/{trait}. Run either:\n"
        f"  python analysis/vectors/extraction_evaluation.py --experiment {experiment}\n"
        f"  python steering/run_steering_eval.py --experiment {experiment} --trait {trait}"
    )


def select_vectors(
    experiment: str,
    trait: str,
    n: int = 3,
    extraction_variant: Optional[str] = None,
    steering_variant: Optional[str] = None,
    component: Optional[str] = None,
    position: Optional[str] = None,
    layer: Optional[int] = None,
    method: Optional[str] = None,
    min_coherence: int = MIN_COHERENCE,
    min_delta: float = 0,
    prompt_set: str = "steering",
) -> List[VectorResult]:
    """Top N vectors (one per layer) via the steering-score path.
    No extraction-eval fallback — use select_vector for single-best when steering is absent."""
    return _select_vectors(
        experiment, trait, n=n,
        extraction_variant=extraction_variant, steering_variant=steering_variant,
        component=component, position=position, layer=layer, method=method,
        min_coherence=min_coherence, min_delta=min_delta, prompt_set=prompt_set,
    )


def get_best_vector_spec(
    experiment: str,
    trait: str,
    weight: float = 1.0,
    **select_kwargs,
) -> Tuple[VectorSpec, Dict[str, Any]]:
    """Return best vector as (VectorSpec, metadata dict). Thin wrapper over select_vector."""
    best = select_vector(experiment, trait, **select_kwargs)
    return best.to_vector_spec(weight=weight), {
        'source': best.source, 'delta': best.delta, 'coefficient': best.coefficient,
        'coherence': best.coherence,
    }


def load_trait_vectors(
    experiment: str,
    extraction_variant: str,
    traits: List[str],
    component: Optional[str],
    layers_spec: Optional[str],
    available_layers: Optional[List[int]] = None,
):
    """Load trait vectors and group by layer for batched projection.

    Returns:
        trait_vectors: {(category, trait_name): [(vector, method, None, layer,
            metadata, source, baseline, position), ...]}
        vectors_by_layer: {layer: Tensor[n_vectors, hidden_dim]}
        hook_index: {(layer, slot_in_stacked): (category, trait_name, vec_list_idx)}
    """
    from utils.layers import resolve_layers, parse_layers
    from utils.vectors import find_vector_method

    trait_vectors: Dict[Tuple[str, str], list] = {}
    vectors_for_layer: Dict[int, list] = {}

    for trait in traits:
        parts = trait.split('/')
        category, trait_name = parts[0], '/'.join(parts[1:]) if len(parts) > 2 else parts[1]
        key = (category, trait_name)

        # Try to score the trait (steering → extraction_eval). If neither exists,
        # fall through to explicit numeric layers; otherwise raise loudly (silent
        # skip was the previous bug — it masqueraded as success).
        best_layer = best_method = position = None
        source = 'unscored'
        try:
            best = select_vector(experiment, trait, extraction_variant=extraction_variant)
            best_layer, best_method, position, source = best.layer, best.method, best.position, best.source or 'steering'
        except FileNotFoundError:
            if not (layers_spec and 'best' not in layers_spec):
                raise FileNotFoundError(
                    f"No scored vectors for {trait} — neither steering results nor "
                    f"extraction_evaluation.json is available. Either pass explicit "
                    f"numeric --layers, or run:\n"
                    f"  python analysis/vectors/extraction_evaluation.py --experiment {experiment}\n"
                    f"  python steering/run_steering_eval.py --experiment {experiment} --trait {trait}"
                )

        # Resolve concrete layers.
        if available_layers is not None:
            concrete_layers = resolve_layers(layers_spec or "best,best+5", best_layer, set(available_layers))
        elif best_layer is not None:
            concrete_layers = [best_layer]
        elif layers_spec and 'best' not in layers_spec:
            concrete_layers = parse_layers(layers_spec, 80)
        else:
            concrete_layers = []

        vector_list = []
        load_position = position or 'response[:5]'
        for layer in concrete_layers:
            method = best_method or find_vector_method(
                experiment, trait, layer, extraction_variant, component or 'residual', load_position
            )
            if method is None:
                continue
            vector, baseline, metadata = load_vector_with_baseline(
                experiment, trait, method, layer, extraction_variant, component, load_position,
            )
            vec_list_idx = len(vector_list)
            vector_list.append((vector, method, None, layer, metadata, source, baseline, load_position))
            vectors_for_layer.setdefault(layer, []).append((vector, category, trait_name, vec_list_idx))

        if vector_list:
            trait_vectors[key] = vector_list

    vectors_by_layer: Dict[int, torch.Tensor] = {}
    hook_index: Dict[Tuple[int, int], Tuple[str, str, int]] = {}
    for layer, vecs in vectors_for_layer.items():
        vectors_by_layer[layer] = torch.stack([v[0] for v in vecs])
        for idx, (_, cat, trait_name, vec_list_idx) in enumerate(vecs):
            hook_index[(layer, idx)] = (cat, trait_name, vec_list_idx)

    return trait_vectors, vectors_by_layer, hook_index
