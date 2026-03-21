"""Projection utilities, fingerprint comparison, and classification.

Handles projection JSON I/O, activation-norm normalization, cosine similarity,
and nearest-centroid classification.

Usage:
    from utils.projections import read_projection, read_response_projections
    from utils.projections import cosine_sim, nearest_centroid_classify
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


# =============================================================================
# Fingerprint comparison metrics (numpy, analysis-side)
# =============================================================================

from collections import defaultdict


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two numpy vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


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


