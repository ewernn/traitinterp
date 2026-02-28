"""Fingerprint utilities — data loading, comparison metrics, classification.

Centralizes functions duplicated across ~20 analysis scripts. All functions
operate on numpy arrays (analysis-side), not torch tensors (GPU-side).

Input: Probe score JSONs, checkpoint Method B JSONs, activation norms
Output: Cosine similarities, classification results, normalized vectors

Usage:
    from utils.fingerprints import (
        load_scores, load_checkpoint_run, cosine_sim, pairwise_cosine,
        nearest_centroid_classify, short_name,
    )
    scores = load_scores("probe_scores/variant_x_eval_combined.json")
    cos = cosine_sim(vec_a, vec_b)
    acc, total, confusion = nearest_centroid_classify(train, test_vecs, test_labels)
"""

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from utils.projections import (
    get_trait_layers,
    load_activation_norms,
    normalize_fingerprint,
    normalize_scores_vector,
    scores_dict_to_vector,
)


# Re-export from projections for single import point
__all__ = [
    # Re-exports
    "get_trait_layers", "load_activation_norms", "normalize_fingerprint",
    "normalize_scores_vector", "scores_dict_to_vector",
    # Data loading
    "load_scores", "load_checkpoint_run",
    # Comparison metrics
    "cosine_sim", "pairwise_cosine", "cross_group_cosine",
    "separation_gap", "spearman_corr",
    # Classification
    "nearest_centroid_classify",
    # Vector operations
    "to_vector", "compute_model_delta", "short_name",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scores(path) -> dict[str, float]:
    """Load probe score JSON ({trait: score}).

    Works for pxs_grid outputs: combined, text_only, reverse_model, per-layer.
    """
    with open(path) as f:
        return json.load(f)


def load_checkpoint_run(path) -> dict:
    """Load checkpoint Method B JSON into structured dict.

    Returns:
        {
            "metadata": {...},
            "probes": {trait: {layer, method, steering_delta, direction}},
            "baseline": {scores, per_eval},
            "checkpoints": [{name, step, reverse_model_scores, model_delta, ...}],
        }
    """
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def pairwise_cosine(vectors: list[np.ndarray]) -> float:
    """Mean cosine similarity over all pairs."""
    sims = [cosine_sim(a, b) for a, b in combinations(vectors, 2)]
    return float(np.mean(sims)) if sims else 0.0


def cross_group_cosine(group_a: list[np.ndarray], group_b: list[np.ndarray]) -> float:
    """Mean cosine similarity between all cross-group pairs."""
    sims = [cosine_sim(a, b) for a in group_a for b in group_b]
    return float(np.mean(sims)) if sims else 0.0


def separation_gap(within: float, cross: float) -> float:
    """Gap between within-group and cross-group similarity."""
    return within - cross


def spearman_corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation between two vectors. Returns (rho, p-value)."""
    rho, p = spearmanr(a, b)
    return float(rho), float(p)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def nearest_centroid_classify(
    train_vecs: dict[str, list[np.ndarray]],
    test_vecs: list[np.ndarray],
    test_labels: list[str],
) -> tuple[int, int, dict[str, dict[str, int]]]:
    """Cosine-similarity nearest-centroid classifier.

    Args:
        train_vecs: {label: [vectors]} — training vectors grouped by class
        test_vecs: test vectors to classify
        test_labels: ground truth labels for test vectors

    Returns:
        (correct, total, confusion_matrix)
        confusion_matrix: {true_label: {predicted_label: count}}
    """
    centroids = {}
    for label, vecs in train_vecs.items():
        centroids[label] = np.mean(vecs, axis=0)

    labels_sorted = sorted(centroids.keys())
    centroid_matrix = np.stack([centroids[l] for l in labels_sorted])

    correct = 0
    total = 0
    confusion = defaultdict(lambda: defaultdict(int))

    for vec, true_label in zip(test_vecs, test_labels):
        dots = centroid_matrix @ vec
        norms = np.linalg.norm(centroid_matrix, axis=1) * np.linalg.norm(vec)
        cosine_sims = dots / (norms + 1e-12)
        pred_idx = np.argmax(cosine_sims)
        pred_label = labels_sorted[pred_idx]

        confusion[true_label][pred_label] += 1
        if pred_label == true_label:
            correct += 1
        total += 1

    return correct, total, dict(confusion)


# ---------------------------------------------------------------------------
# Vector operations
# ---------------------------------------------------------------------------

def to_vector(
    scores: dict[str, float],
    trait_order: list[str],
    trait_layers: dict[str, int] = None,
    norms_per_layer: np.ndarray = None,
) -> np.ndarray:
    """Convert score dict to numpy vector, optionally normalizing.

    If trait_layers and norms_per_layer are provided, normalizes by activation
    magnitude (divides each score by mean ||h|| at that trait's layer).
    """
    if trait_layers is not None and norms_per_layer is not None:
        return normalize_scores_vector(scores, trait_order, trait_layers, norms_per_layer)
    return np.array([scores.get(t, 0.0) for t in trait_order])


def compute_model_delta(
    reverse_model: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    """Compute model delta = reverse_model - baseline (element-wise)."""
    return {trait: reverse_model[trait] - baseline[trait]
            for trait in reverse_model if trait in baseline}


def short_name(trait: str) -> str:
    """Extract short trait name: 'alignment/deception' -> 'deception'."""
    return trait.split("/")[-1]
