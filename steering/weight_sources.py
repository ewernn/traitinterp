"""
Weight source strategies for multi-layer steering.

Each strategy selects (layer, component) hooks and assigns weights
for simultaneous multi-layer steering via a single global coefficient.

Input: Strategy name, experiment/trait identifiers, strategy parameters
Output: (vectors_and_layers, per_hook_weights, base_coef) tuple

Usage:
    from steering.weight_sources import build_weighted_configs
    vl, weights, base_coef = build_weighted_configs(
        "probe", experiment, trait, extraction_variant, method, position, ...
    )
"""

import sys
import json
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from typing import Optional

from utils.activations import load_activation_metadata
from utils.paths import get
from utils.vectors import MIN_COHERENCE, load_vector

COMPONENTS = ["attn_contribution", "mlp_contribution"]

STRATEGIES = ["probe", "causal", "intersection", "topk"]


# =============================================================================
# Public dispatcher
# =============================================================================

def build_weighted_configs(
    source: str,
    experiment: str,
    trait: str,
    extraction_variant: str,
    method: str,
    position: str,
    weight_threshold: float = 0.01,
    delta_threshold: float = 5.0,
    min_coherence: float = MIN_COHERENCE,
    top_k: int = 5,
    model_variant: str = None,
    prompt_set: str = "steering",
    steering_experiment: str = None,
    device: torch.device = None,
) -> tuple[list, list, float]:
    """Build multi-layer steering configs from the given weight source.

    Args:
        experiment: Experiment for loading vectors and probe weights.
        steering_experiment: Experiment where per-component steering results live.
            Defaults to experiment. Only used by causal/intersection/topk strategies.

    Returns:
        vectors_and_layers: [(layer, vector, component), ...]
        per_hook_weights: normalized weights (sum |w| = 1)
        base_coef: estimated starting coefficient for search
    """
    if source not in STRATEGIES:
        raise ValueError(f"Unknown weight source '{source}'. Choose from: {STRATEGIES}")

    results_experiment = steering_experiment or experiment

    common = dict(
        experiment=experiment, trait=trait, extraction_variant=extraction_variant,
        method=method, position=position, device=device,
    )

    if source == "probe":
        entries = _get_probe_entries(experiment, trait, weight_threshold)
    elif source == "causal":
        entries = _get_causal_entries(
            results_experiment, trait, model_variant, position, prompt_set,
            min_coherence, delta_threshold,
        )
    elif source == "intersection":
        entries = _get_intersection_entries(
            experiment, results_experiment, trait, model_variant, position, prompt_set,
            min_coherence, weight_threshold, delta_threshold,
        )
    elif source == "topk":
        entries = _get_topk_entries(
            results_experiment, trait, model_variant, position, prompt_set,
            min_coherence, top_k,
        )

    if not entries:
        return [], [], 0.0

    return _load_vectors_and_compute_base_coef(entries, **common)


# =============================================================================
# Strategy: probe
# =============================================================================

def _get_probe_entries(
    experiment: str,
    trait: str,
    weight_threshold: float,
) -> list[tuple[int, str, float]]:
    """L1 probe coefficients from multi_layer_probe results."""
    probe_data = _load_probe_weights(experiment, trait)
    if probe_data is None:
        return []

    weight_map = probe_data["weight_map"]  # [n_layers][n_components]
    n_layers = len(weight_map)

    # Collect nonzero weights
    entries = []
    for layer in range(n_layers):
        for comp_idx, comp in enumerate(COMPONENTS):
            w = weight_map[layer][comp_idx]
            if abs(w) > 1e-6:
                entries.append((layer, comp, w))

    if not entries:
        print(f"  No nonzero probe weights for {trait}")
        return []

    # Filter by relative threshold
    total_abs = sum(abs(w) for _, _, w in entries)
    threshold = weight_threshold * total_abs
    filtered = [(l, c, w) for l, c, w in entries if abs(w) >= threshold]

    print(f"  Probe weights: {len(entries)} nonzero, {len(filtered)} above {weight_threshold:.0%} threshold")
    return filtered


def _load_probe_weights(experiment: str, trait: str) -> Optional[dict]:
    """Load two-stage probe results for a trait."""
    results_path = get("experiments.base", experiment=experiment) / "analysis" / "multi_layer_probe" / "results.json"
    if not results_path.exists():
        print(f"  Probe results not found: {results_path}")
        return None

    with open(results_path) as f:
        all_results = json.load(f)

    if trait not in all_results:
        print(f"  Trait {trait} not in probe results")
        return None

    return all_results[trait].get("two_stage")


# =============================================================================
# Strategy: causal
# =============================================================================

def _get_causal_entries(
    experiment: str,
    trait: str,
    model_variant: str,
    position: str,
    prompt_set: str,
    min_coherence: float,
    delta_threshold: float,
) -> list[tuple[int, str, float]]:
    """Weight by actual steering delta from per-component results."""
    deltas, direction = _load_per_component_deltas(
        experiment, trait, model_variant, position, prompt_set, min_coherence,
    )
    if not deltas:
        return []

    sign = 1 if direction == "positive" else -1

    # Filter by minimum delta (in the steering direction)
    entries = [
        (layer, comp, delta)
        for (layer, comp), delta in deltas.items()
        if delta * sign >= delta_threshold
    ]

    if not entries:
        print(f"  No per-component deltas above {delta_threshold} for {trait}")
        return []

    print(f"  Causal weights: {len(deltas)} total, {len(entries)} above delta threshold {delta_threshold}")
    return entries


# =============================================================================
# Strategy: intersection
# =============================================================================

def _get_intersection_entries(
    probe_experiment: str,
    steering_experiment: str,
    trait: str,
    model_variant: str,
    position: str,
    prompt_set: str,
    min_coherence: float,
    weight_threshold: float,
    delta_threshold: float,
) -> list[tuple[int, str, float]]:
    """Only (layer, component) where both probe AND causal signals are strong."""
    probe_entries = _get_probe_entries(probe_experiment, trait, weight_threshold)
    probe_map = {(l, c): w for l, c, w in probe_entries}

    deltas, direction = _load_per_component_deltas(
        steering_experiment, trait, model_variant, position, prompt_set, min_coherence,
    )
    sign = 1 if direction == "positive" else -1

    # Keep only keys present in both, with causal delta above threshold
    entries = []
    for (layer, comp), delta in deltas.items():
        if delta * sign < delta_threshold:
            continue
        if (layer, comp) not in probe_map:
            continue

        # Geometric mean of probe and causal magnitudes
        probe_abs = abs(probe_map[(layer, comp)])
        causal_abs = abs(delta)
        weight = math.sqrt(probe_abs * causal_abs)

        entries.append((layer, comp, weight))

    if not entries:
        print(f"  No intersection entries for {trait}")
        return []

    print(f"  Intersection: {len(probe_map)} probe, {len(deltas)} causal, {len(entries)} overlap")
    return entries


# =============================================================================
# Strategy: topk
# =============================================================================

def _get_topk_entries(
    experiment: str,
    trait: str,
    model_variant: str,
    position: str,
    prompt_set: str,
    min_coherence: float,
    top_k: int,
) -> list[tuple[int, str, float]]:
    """Top K by causal delta, equal weight."""
    deltas, direction = _load_per_component_deltas(
        experiment, trait, model_variant, position, prompt_set, min_coherence,
    )
    if not deltas:
        return []

    sign = 1 if direction == "positive" else -1

    # Sort by delta in the steering direction (descending)
    sorted_items = sorted(deltas.items(), key=lambda x: x[1] * sign, reverse=True)

    # Take top K with positive delta
    entries = []
    for (layer, comp), delta in sorted_items[:top_k]:
        if delta * sign <= 0:
            break
        entries.append((layer, comp, 1.0))

    if not entries:
        print(f"  No positive-delta entries for {trait}")
        return []

    print(f"  Top-{top_k}: selected {len(entries)} hooks from {len(deltas)} candidates")
    return entries


# =============================================================================
# Shared helpers
# =============================================================================

def _load_per_component_deltas(
    experiment: str,
    trait: str,
    model_variant: str,
    position: str,
    prompt_set: str,
    min_coherence: float,
) -> tuple[dict[tuple[int, str], float], str]:
    """Extract best delta per (layer, component) from results.jsonl.

    Reads single-vector entries only (skips multi-vector and residual).
    Returns ({(layer, component): delta}, direction).
    """
    from steering.steering_results import load_results

    data = load_results(experiment, trait, model_variant, position, prompt_set)
    direction = data.get("direction", "positive")
    sign = 1 if direction == "positive" else -1
    baseline = data.get("baseline", {}).get("trait_mean", 0)

    best_deltas = {}
    for run in data.get("runs", []):
        vectors = run.get("config", {}).get("vectors", [])
        if len(vectors) != 1:
            continue
        v = vectors[0]
        comp = v.get("component", "residual")
        if comp not in COMPONENTS:
            continue
        layer = v.get("layer")
        coherence = run.get("result", {}).get("coherence_mean", 0)
        if coherence < min_coherence:
            continue
        delta = run.get("result", {}).get("trait_mean", 0) - baseline
        key = (layer, comp)
        if key not in best_deltas or delta * sign > best_deltas[key] * sign:
            best_deltas[key] = delta

    if not best_deltas:
        print(f"  No per-component steering results for {trait}. "
              f"Run per-component steering first (evaluate.py --component attn_contribution/mlp_contribution)")

    return best_deltas, direction


def _load_vectors_and_compute_base_coef(
    entries: list[tuple[int, str, float]],
    experiment: str,
    trait: str,
    extraction_variant: str,
    method: str,
    position: str,
    device: torch.device = None,
) -> tuple[list, list, float]:
    """Load vectors, normalize weights, compute base_coef.

    entries: [(layer, component, raw_weight), ...]
    Returns: (vectors_and_layers, per_hook_weights, base_coef)
    """
    # Load per-component activation norms
    comp_norms = {}
    for comp in set(c for _, c, _ in entries):
        try:
            meta = load_activation_metadata(experiment, trait, extraction_variant, comp, position)
            comp_norms[comp] = {int(k): v for k, v in meta.get("activation_norms", {}).items()}
        except FileNotFoundError:
            comp_norms[comp] = {}

    vectors_and_layers = []
    weights = []

    for layer, comp, raw_weight in entries:
        vector = load_vector(experiment, trait, layer, extraction_variant, method, comp, position)
        if vector is None:
            print(f"    L{layer} {comp}: vector not found, skipping")
            continue

        if device is not None:
            vector = vector.to(device)

        vec_norm = vector.norm().item()
        if vec_norm == 0:
            continue

        vectors_and_layers.append((layer, vector, comp))
        weights.append(raw_weight)

        act_norm = comp_norms.get(comp, {}).get(layer, 0)
        comp_short = "attn" if "attn" in comp else "mlp"
        print(f"    L{layer:02d} {comp_short}  w={raw_weight:+.4f}  act_norm={act_norm:.1f}  vec_norm={vec_norm:.3f}")

    if not vectors_and_layers:
        return [], [], 0.0

    # Normalize weights so coef search sweeps one global scalar
    abs_sum = sum(abs(w) for w in weights)
    per_hook_weights = [w / abs_sum for w in weights]

    # Base coefficient: weighted average of act_norm/vec_norm, scaled by max weight.
    # Each hook gets coef * w_i, so the dominant hook only sees coef * max(|w_i|).
    # Dividing by max(|w_i|) ensures the dominant hook reaches single-layer-equivalent perturbation.
    base_coef_num = 0.0
    base_coef_den = 0.0
    for (layer, vector, comp), w in zip(vectors_and_layers, per_hook_weights):
        vec_norm = vector.norm().item()
        act_norm = comp_norms.get(comp, {}).get(layer, 0)
        if act_norm > 0 and vec_norm > 0:
            base_coef_num += abs(w) * (act_norm / vec_norm)
            base_coef_den += abs(w)

    if base_coef_den > 0:
        base_coef = base_coef_num / base_coef_den
        max_weight = max(abs(w) for w in per_hook_weights)
        base_coef = base_coef / max_weight
    else:
        base_coef = 100.0

    return vectors_and_layers, per_hook_weights, base_coef
