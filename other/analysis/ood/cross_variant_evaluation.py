#!/usr/bin/env python3
"""
Cross-variant classification evaluation for OOD formality experiments.

Input:
    - experiments/{experiment}/extraction/{trait}/{model_variant}/activations/{position}/{component}/*_all_layers.pt

Output:
    - Prints 4x4 accuracy matrices for cross-language and cross-topic variants

Usage:
    python analysis/ood/cross_variant_evaluation.py --experiment gemma-2-2b
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import json
import fire
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Tuple

from utils.paths import (
    get_activation_path,
    get_val_activation_path,
    get_activation_metadata_path,
)


def load_activations(
    experiment: str,
    trait: str,
    layer: int,
    component: str = "residual",
    position: str = "response[:]",
    split: str = "train",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load pos/neg activations for a layer."""
    if split == "train":
        path = get_activation_path(experiment, trait, component, position)
    else:
        path = get_val_activation_path(experiment, trait, component, position)

    if not path.exists():
        raise FileNotFoundError(f"Activation file not found: {path}")

    metadata_path = get_activation_metadata_path(experiment, trait, component, position)
    with open(metadata_path) as f:
        metadata = json.load(f)

    # n_examples_pos = train positive count, n_val_pos = val positive count
    if split == "train":
        n_pos = metadata.get("n_examples_pos", 0)
    else:
        n_pos = metadata.get("n_val_pos", 0)

    acts = torch.load(path, weights_only=True)
    layer_acts = acts[:, layer, :].float().numpy()

    pos_acts = layer_acts[:n_pos]
    neg_acts = layer_acts[n_pos:]

    return pos_acts, neg_acts


def train_probe(pos_acts: np.ndarray, neg_acts: np.ndarray) -> LogisticRegression:
    """Train a logistic regression probe."""
    X = np.vstack([pos_acts, neg_acts])
    y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))

    probe = LogisticRegression(max_iter=1000, solver='lbfgs')
    probe.fit(X, y)
    return probe


def evaluate_probe(
    probe: LogisticRegression,
    pos_acts: np.ndarray,
    neg_acts: np.ndarray
) -> float:
    """Evaluate probe accuracy."""
    X = np.vstack([pos_acts, neg_acts])
    y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))
    return probe.score(X, y)


def cross_validate_variants(
    experiment: str,
    variants: List[str],
    layer: int,
    component: str = "residual",
    position: str = "response[:]",
) -> np.ndarray:
    """
    Train probe on each variant, test on all others.
    Returns n×n accuracy matrix.
    """
    n = len(variants)
    matrix = np.zeros((n, n))

    # Load all activations
    train_data = {}
    test_data = {}
    for v in variants:
        trait = f"formality_variations/{v}"
        try:
            train_data[v] = load_activations(experiment, trait, layer, component, position, "train")
            test_data[v] = load_activations(experiment, trait, layer, component, position, "val")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

    # Cross-validate
    for i, train_v in enumerate(variants):
        if train_v not in train_data:
            continue

        # Train probe
        pos_train, neg_train = train_data[train_v]
        probe = train_probe(pos_train, neg_train)

        for j, test_v in enumerate(variants):
            if test_v not in test_data:
                continue

            pos_test, neg_test = test_data[test_v]
            acc = evaluate_probe(probe, pos_test, neg_test)
            matrix[i, j] = acc

    return matrix


def print_matrix(matrix: np.ndarray, labels: List[str], title: str):
    """Print accuracy matrix as formatted table."""
    print(f"\n{title}")
    print("=" * (15 + 10 * len(labels)))

    # Header
    header = f"{'Train \\ Test':<15}"
    for label in labels:
        header += f"{label[:8]:<10}"
    print(header)
    print("-" * (15 + 10 * len(labels)))

    # Rows
    for i, row_label in enumerate(labels):
        row = f"{row_label[:14]:<15}"
        for j, acc in enumerate(matrix[i]):
            if acc > 0:
                if i == j:
                    row += f"{acc:.1%}*    "
                else:
                    row += f"{acc:.1%}     "
            else:
                row += f"{'N/A':<10}"
        print(row)

    # Summary stats
    print("-" * (15 + 10 * len(labels)))

    # Diagonal (in-domain)
    diagonal = np.diag(matrix)
    valid_diag = diagonal[diagonal > 0]
    if len(valid_diag) > 0:
        print(f"In-domain mean: {valid_diag.mean():.1%}")

    # Off-diagonal (cross-domain)
    off_diag = []
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i != j and matrix[i, j] > 0:
                off_diag.append(matrix[i, j])
    if off_diag:
        print(f"Cross-domain mean: {np.mean(off_diag):.1%}")
        print(f"Cross-domain std: {np.std(off_diag):.1%}")


def main(
    experiment: str = "gemma-2-2b",
    layer: int = 12,
    component: str = "residual",
    position: str = "response[:]",
):
    """
    Run cross-variant classification evaluation.

    Args:
        experiment: Experiment name
        layer: Layer to evaluate (default: 12, middle layer)
        component: Activation component
        position: Token position
    """
    print(f"Cross-Variant Classification Evaluation")
    print(f"Experiment: {experiment}")
    print(f"Layer: {layer}, Component: {component}, Position: {position}")

    # Cross-language variants
    lang_variants = ["english", "spanish", "french", "chinese"]
    lang_matrix = cross_validate_variants(
        experiment, lang_variants, layer, component, position
    )
    print_matrix(lang_matrix, lang_variants, "Cross-Language Classification")

    # Cross-topic variants
    topic_variants = ["business", "academic", "social", "technical"]
    topic_matrix = cross_validate_variants(
        experiment, topic_variants, layer, component, position
    )
    print_matrix(topic_matrix, topic_variants, "Cross-Topic Classification")

    # Return results for programmatic use
    return {
        "cross_language": {
            "variants": lang_variants,
            "matrix": lang_matrix.tolist(),
        },
        "cross_topic": {
            "variants": topic_variants,
            "matrix": topic_matrix.tolist(),
        },
    }


if __name__ == "__main__":
    fire.Fire(main)
