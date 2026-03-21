"""Projection utilities, fingerprint comparison, and classification.

Handles projection JSON I/O, activation-norm normalization, cosine similarity,
nearest-centroid classification, and fingerprint vector operations.

Usage:
    from utils.projections import read_projection, read_response_projections
    from utils.projections import cosine_sim, nearest_centroid_classify, spearman_corr
    from utils.projections import normalize_fingerprint, get_trait_layers
"""

import json
from pathlib import Path

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

def load_activation_norms(norms_path: str | Path, expected_model: str = None) -> np.ndarray:
    """Load per-layer activation norms from JSON.

    These are mean(||hidden_state||) across all response tokens, precomputed by
    compute_activation_norms.py. Used to normalize probe scores so traits at
    different layers are comparable.

    Validates model name if expected_model is provided (compares final path
    component, e.g. "Qwen2.5-14B-Instruct").
    """
    with open(norms_path) as f:
        data = json.load(f)
    norms = np.array(data["norms_per_layer"])

    if expected_model and "model" in data:
        # Compare short names (last path component)
        norms_model = data["model"].rstrip("/").split("/")[-1]
        expected_short = expected_model.rstrip("/").split("/")[-1]
        if norms_model != expected_short:
            print(f"  Warning: norms computed on {data['model']}, but scoring with {expected_model}")

    return norms


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


# =============================================================================
# Fingerprint comparison metrics (numpy, analysis-side)
# =============================================================================

from collections import defaultdict
from itertools import combinations
from scipy.stats import spearmanr


def load_scores(path) -> dict:
    """Load probe score JSON ({trait: score})."""
    with open(path) as f:
        return json.load(f)


def load_checkpoint_run(path) -> dict:
    """Load checkpoint Method B JSON into structured dict."""
    with open(path) as f:
        return json.load(f)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two numpy vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def pairwise_cosine(vectors: list) -> float:
    """Mean cosine similarity over all pairs."""
    sims = [cosine_sim(a, b) for a, b in combinations(vectors, 2)]
    return float(np.mean(sims)) if sims else 0.0


def cross_group_cosine(group_a: list, group_b: list) -> float:
    """Mean cosine similarity between all cross-group pairs."""
    sims = [cosine_sim(a, b) for a in group_a for b in group_b]
    return float(np.mean(sims)) if sims else 0.0


def separation_gap(within: float, cross: float) -> float:
    """Gap between within-group and cross-group similarity."""
    return within - cross


def spearman_corr(a: np.ndarray, b: np.ndarray) -> tuple:
    """Spearman rank correlation. Returns (rho, p-value)."""
    rho, p = spearmanr(a, b)
    return float(rho), float(p)


def nearest_centroid_classify(train_vecs: dict, test_vecs: list, test_labels: list) -> tuple:
    """Cosine-similarity nearest-centroid classifier.

    Args:
        train_vecs: {label: [vectors]} — training vectors grouped by class
        test_vecs: test vectors to classify
        test_labels: ground truth labels

    Returns:
        (correct, total, confusion_matrix)
    """
    centroids = {label: np.mean(vecs, axis=0) for label, vecs in train_vecs.items()}
    labels_sorted = sorted(centroids.keys())
    centroid_matrix = np.stack([centroids[l] for l in labels_sorted])

    correct, total = 0, 0
    confusion = defaultdict(lambda: defaultdict(int))

    for vec, true_label in zip(test_vecs, test_labels):
        dots = centroid_matrix @ vec
        norms = np.linalg.norm(centroid_matrix, axis=1) * np.linalg.norm(vec)
        cosine_sims = dots / (norms + 1e-12)
        pred_label = labels_sorted[np.argmax(cosine_sims)]
        confusion[true_label][pred_label] += 1
        if pred_label == true_label:
            correct += 1
        total += 1

    return correct, total, dict(confusion)


def compute_model_delta(reverse_model: dict, baseline: dict) -> dict:
    """Compute model delta = reverse_model - baseline (element-wise)."""
    return {t: reverse_model[t] - baseline[t] for t in reverse_model if t in baseline}


def short_name(trait: str) -> str:
    """Extract short trait name: 'alignment/deception' -> 'deception'."""
    return trait.split("/")[-1]
