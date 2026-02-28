"""Shared projection utilities. File reading + activation-norm normalization.

Input: Projection JSON files from inference/project_raw_activations_onto_traits.py
Output: Normalized projection data dicts

Usage:
    from utils.projections import read_projection, read_response_projections
    proj = read_projection(path)  # {prompt, response, token_norms, layer, method, baseline, selection_source}
    scores = read_response_projections(path)  # Just response projection array

    # Activation-norm normalization (makes scores comparable across traits at different layers)
    from utils.projections import load_activation_norms, normalize_fingerprint, get_trait_layers
    trait_layers = get_trait_layers("path/to/checkpoint_method_b/rank32.json")
    norms = load_activation_norms("experiments/my_experiment/analysis/activation_norms_14b.json")
    normalized = normalize_fingerprint(scores_dict, trait_layers, norms)
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np


def read_projection(path, layer=None) -> dict:
    """Read a projection file and return normalized data.

    Handles both single-vector and multi-vector formats transparently.

    Args:
        path: Path to projection JSON file
        layer: Layer to extract (for multi-vector files). If None, uses first/only entry.

    Returns:
        Dict with keys: prompt, response, token_norms, layer, method, baseline, selection_source
    """
    with open(path) as f:
        data = json.load(f)

    projections = data['projections']

    if isinstance(projections, list):
        # Multi-vector format: projections is a list of dicts
        if layer is not None:
            entry = next((e for e in projections if e['layer'] == layer), None)
            if entry is None:
                available = [e['layer'] for e in projections]
                raise ValueError(f"Layer {layer} not found in {path}. Available: {available}")
        else:
            entry = projections[0]

        # token_norms: per-entry (new format) or top-level (old multi-vector)
        if 'token_norms' in entry:
            token_norms = entry['token_norms']
        else:
            token_norms = data.get('token_norms', {})

        return {
            'prompt': entry.get('prompt', []),
            'response': entry.get('response', []),
            'token_norms': token_norms,
            'layer': entry['layer'],
            'method': entry.get('method'),
            'baseline': entry.get('baseline', 0.0),
            'selection_source': entry.get('selection_source'),
        }
    else:
        # Single-vector format: projections is {prompt: [...], response: [...]}
        vector_source = data.get('metadata', {}).get('vector_source', {})

        return {
            'prompt': projections.get('prompt', []),
            'response': projections.get('response', []),
            'token_norms': data.get('token_norms', {}),
            'layer': vector_source.get('layer'),
            'method': vector_source.get('method'),
            'baseline': vector_source.get('baseline', 0.0),
            'selection_source': vector_source.get('selection_source'),
        }


def read_response_projections(path, layer=None) -> list:
    """Convenience: returns just the response projection array."""
    return read_projection(path, layer=layer)['response']


# --- Activation-norm normalization ---

def load_activation_norms(norms_path: str | Path) -> np.ndarray:
    """Load per-layer activation norms from JSON.

    These are mean(||hidden_state||) across all response tokens, precomputed by
    compute_activation_norms.py. Used to normalize probe scores so traits at
    different layers are comparable.
    """
    with open(norms_path) as f:
        data = json.load(f)
    return np.array(data["norms_per_layer"])


def normalize_fingerprint(
    scores: dict[str, float],
    trait_layers: dict[str, int],
    norms_per_layer: np.ndarray,
) -> dict[str, float]:
    """Normalize probe scores by activation magnitude at each trait's layer.

    Raw probe score = h @ v_hat = ||h|| cos(theta). Dividing by mean(||h||) at
    that layer gives a score proportional to cos(theta), comparable across traits
    regardless of which layer they use.

    Args:
        scores: {trait: raw_score}
        trait_layers: {trait: best_layer}
        norms_per_layer: array of mean activation norms per layer
    """
    normalized = {}
    for trait, score in scores.items():
        layer = trait_layers[trait]
        norm = norms_per_layer[layer]
        normalized[trait] = score / norm if norm > 1e-8 else score
    return normalized


def scores_dict_to_vector(scores: dict[str, float], trait_order: list[str]) -> np.ndarray:
    """Convert {trait: score} dict to numpy array in consistent trait order."""
    return np.array([scores.get(t, 0.0) for t in trait_order])


def get_trait_layers(checkpoint_path: str | Path) -> dict[str, int]:
    """Get best layer per trait from a checkpoint Method B JSON.

    These JSONs have a "probes" section with the steering-validated best layer
    for each trait, e.g. {"alignment/deception": {"layer": 27, ...}, ...}.
    """
    with open(checkpoint_path) as f:
        data = json.load(f)
    if "probes" not in data:
        raise ValueError(f"No 'probes' section in {checkpoint_path}")
    return {trait: info["layer"] for trait, info in data["probes"].items()}


def normalize_scores_vector(
    score_dict: dict[str, float],
    traits: list[str],
    trait_layers: dict[str, int],
    norms_per_layer: np.ndarray,
) -> np.ndarray:
    """Normalize probe scores by activation layer norms, returning a numpy vector.

    Equivalent to normalize_fingerprint() followed by scores_dict_to_vector(),
    but returns the array directly in trait order.
    """
    vec = np.array([score_dict.get(t, 0.0) for t in traits])
    for i, trait in enumerate(traits):
        layer = trait_layers[trait]
        norm = norms_per_layer[layer]
        if norm > 1e-8:
            vec[i] /= norm
    return vec
