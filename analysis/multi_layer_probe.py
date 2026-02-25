"""
Multi-layer probes combining attn/mlp contributions across all layers.

Input: Per-layer activation files from extraction (attn_contribution, mlp_contribution)
Output: Probe weights, accuracies, and weight maps per trait

Two approaches:
  1. Full concatenation: L1-regularized probe on all layer×component activations (184K-dim)
  2. Two-stage scalar: project onto per-layer mean_diff, then probe on 72-dim scalars

Usage:
    python analysis/multi_layer_probe.py --experiment mats-emergent-misalignment
    python analysis/multi_layer_probe.py --experiment mats-emergent-misalignment --traits rm_hack/eval_awareness
    python analysis/multi_layer_probe.py --experiment mats-emergent-misalignment --approach two-stage
    python analysis/multi_layer_probe.py --experiment mats-emergent-misalignment --approach full
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.activations import load_train_activations, load_val_activations, available_layers, load_activation_metadata
from utils.vectors import load_vector
from utils.paths import get


COMPONENTS = ["attn_contribution", "mlp_contribution"]


def discover_traits(experiment: str) -> list[str]:
    """Find all traits with both attn_contribution and mlp_contribution activations."""
    extraction_dir = get("experiments.base", experiment=experiment) / "extraction"
    traits = []
    for category_dir in sorted(extraction_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        for trait_dir in sorted(category_dir.iterdir()):
            if not trait_dir.is_dir():
                continue
            trait = f"{category_dir.name}/{trait_dir.name}"
            # Check both components exist
            has_both = True
            for comp in COMPONENTS:
                try:
                    available_layers(experiment, trait, "base", comp, "response[:5]")
                except FileNotFoundError:
                    has_both = False
                    break
            if has_both:
                traits.append(trait)
    return traits


def load_multi_layer_activations(
    experiment: str,
    trait: str,
    model_variant: str,
    components: list[str],
    position: str = "response[:5]",
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate activations across all layers and components.

    Returns:
        X: [n_examples, n_layers * n_components * hidden_dim]
        y: [n_examples] with 1=positive, 0=negative
    """
    load_fn = load_train_activations if split == "train" else load_val_activations
    layers = available_layers(experiment, trait, model_variant, components[0], position)

    all_features = []  # Will be [n_examples, total_features]
    n_pos = None

    for layer in layers:
        for comp in components:
            pos_acts, neg_acts = load_fn(experiment, trait, model_variant, layer, comp, position)
            if pos_acts is None:
                return None, None

            combined = torch.cat([pos_acts, neg_acts], dim=0).float().cpu()
            all_features.append(combined)

            if n_pos is None:
                n_pos = len(pos_acts)

    # Concatenate along feature dimension: [n_examples, layers * components * hidden_dim]
    X = torch.cat(all_features, dim=1).numpy()
    y = np.concatenate([np.ones(n_pos), np.zeros(len(X) - n_pos)])
    return X, y


def load_two_stage_features(
    experiment: str,
    trait: str,
    model_variant: str,
    components: list[str],
    position: str = "response[:5]",
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Project activations onto per-layer mean_diff directions.

    Returns:
        X: [n_examples, n_layers * n_components] scalar projections
        y: [n_examples] labels
        feature_names: list of "L{layer}_{component}" strings
    """
    load_fn = load_train_activations if split == "train" else load_val_activations
    layers = available_layers(experiment, trait, model_variant, components[0], position)

    projections = []
    feature_names = []
    n_pos = None

    for layer in layers:
        for comp in components:
            # Load mean_diff vector for this layer×component
            vector = load_vector(
                experiment, trait, layer, model_variant,
                method="mean_diff", component=comp, position=position
            )
            if vector is None:
                continue

            vector = vector.float()
            vector_norm = vector / (vector.norm() + 1e-8)

            # Load activations
            pos_acts, neg_acts = load_fn(experiment, trait, model_variant, layer, comp, position)
            if pos_acts is None:
                continue

            combined = torch.cat([pos_acts, neg_acts], dim=0).float()
            if n_pos is None:
                n_pos = len(pos_acts)

            # Project: [n_examples] scalar
            scores = (combined @ vector_norm).numpy()
            projections.append(scores)
            feature_names.append(f"L{layer}_{comp}")

    if not projections:
        return None, None, []

    X = np.column_stack(projections)
    y = np.concatenate([np.ones(n_pos), np.zeros(len(X) - n_pos)])
    return X, y, feature_names


def train_full_concat_probe(X_train, y_train, X_val=None, y_val=None):
    """Train L1-regularized logistic regression on full concatenated features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # L1 regularization with saga solver for sparse high-dim
    probe = LogisticRegression(
        solver="saga",
        l1_ratio=1.0,  # Pure L1
        C=1.0,
        max_iter=2000,
        random_state=42,
    )
    probe.fit(X_scaled, y_train)

    train_acc = probe.score(X_scaled, y_train)

    result = {
        "train_acc": float(train_acc),
        "n_features": X_train.shape[1],
        "n_nonzero": int(np.count_nonzero(probe.coef_)),
        "sparsity": float(1 - np.count_nonzero(probe.coef_) / probe.coef_.size),
    }

    if X_val is not None and y_val is not None:
        X_val_scaled = scaler.transform(X_val)
        result["val_acc"] = float(probe.score(X_val_scaled, y_val))

    return result, probe.coef_[0], scaler


def train_two_stage_probe(X_train, y_train, X_val=None, y_val=None):
    """Train logistic regression on scalar projections (72-dim)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    probe = LogisticRegression(
        solver="lbfgs",
        l1_ratio=0.0,  # Pure L2
        C=1.0,
        max_iter=1000,
        random_state=42,
    )
    probe.fit(X_scaled, y_train)

    train_acc = probe.score(X_scaled, y_train)
    cv_scores = cross_val_score(probe, X_scaled, y_train, cv=5, scoring="accuracy")

    result = {
        "train_acc": float(train_acc),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "n_features": X_train.shape[1],
    }

    if X_val is not None and y_val is not None:
        X_val_scaled = scaler.transform(X_val)
        result["val_acc"] = float(probe.score(X_val_scaled, y_val))

    return result, probe.coef_[0], scaler


def extract_weight_map(weights: np.ndarray, n_layers: int, components: list[str], hidden_dim: int = None):
    """Reshape probe weights into a layer × component map.

    For full concat: aggregates absolute weights per layer×component block.
    For two-stage: weights are already per layer×component.
    """
    n_components = len(components)

    if hidden_dim is not None:
        # Full concat: reshape [n_layers * n_components * hidden_dim] → [n_layers, n_components]
        expected = n_layers * n_components * hidden_dim
        if len(weights) != expected:
            print(f"  Warning: weight size {len(weights)} != expected {expected}")
            return None

        weights_3d = weights.reshape(n_layers, n_components, hidden_dim)
        # Aggregate: mean absolute weight per layer×component
        weight_map = np.abs(weights_3d).mean(axis=2)
    else:
        # Two-stage: reshape [n_layers * n_components] → [n_layers, n_components]
        weight_map = weights.reshape(n_layers, n_components)

    return weight_map


def print_weight_map(weight_map: np.ndarray, components: list[str], n_layers: int, top_k: int = 10):
    """Print the top contributing layer×component pairs."""
    flat = []
    for l in range(n_layers):
        for c_idx, comp in enumerate(components):
            flat.append((abs(weight_map[l, c_idx]), weight_map[l, c_idx], l, comp))

    flat.sort(reverse=True)
    print(f"\n  Top {top_k} layer×component contributions:")
    for i, (abs_w, w, layer, comp) in enumerate(flat[:top_k]):
        comp_short = "attn" if "attn" in comp else "mlp"
        direction = "+" if w > 0 else "-"
        print(f"    {i+1:2d}. L{layer:02d} {comp_short:4s}  {direction}{abs_w:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Multi-layer probe analysis")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--traits", help="Comma-separated traits (default: all)")
    parser.add_argument("--approach", choices=["full", "two-stage", "both"], default="both")
    parser.add_argument("--position", default="response[:5]")
    parser.add_argument("--model-variant", default="base")
    args = parser.parse_args()

    if args.traits:
        traits = [t.strip() for t in args.traits.split(",")]
    else:
        traits = discover_traits(args.experiment)
        print(f"Found {len(traits)} traits with attn+mlp activations")

    # Output directory
    output_dir = get("experiments.base", experiment=args.experiment) / "analysis" / "multi_layer_probe"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for trait in traits:
        print(f"\n{'='*60}")
        print(f"  {trait}")
        print(f"{'='*60}")

        layers = available_layers(args.experiment, trait, args.model_variant, COMPONENTS[0], args.position)
        n_layers = len(layers)
        metadata = load_activation_metadata(args.experiment, trait, args.model_variant, COMPONENTS[0], args.position)
        hidden_dim = metadata["hidden_dim"]
        trait_results = {"n_layers": n_layers, "hidden_dim": hidden_dim}

        # ── Two-stage scalar probe ──
        if args.approach in ("two-stage", "both"):
            print(f"\n  Two-stage scalar probe ({n_layers} layers × {len(COMPONENTS)} components = {n_layers * len(COMPONENTS)} features)")

            X_train, y_train, feature_names = load_two_stage_features(
                args.experiment, trait, args.model_variant, COMPONENTS, args.position, "train"
            )
            X_val, y_val, _ = load_two_stage_features(
                args.experiment, trait, args.model_variant, COMPONENTS, args.position, "val"
            )

            if X_train is not None:
                result, weights, scaler = train_two_stage_probe(
                    X_train, y_train,
                    X_val if X_val is not None else None,
                    y_val if y_val is not None else None,
                )
                print(f"    Train: {result['train_acc']:.1%}  |  CV: {result['cv_mean']:.1%} ± {result['cv_std']:.1%}", end="")
                if "val_acc" in result:
                    print(f"  |  Val: {result['val_acc']:.1%}")
                else:
                    print()

                weight_map = extract_weight_map(weights, n_layers, COMPONENTS)
                if weight_map is not None:
                    print_weight_map(weight_map, COMPONENTS, n_layers)
                    result["weight_map"] = weight_map.tolist()
                    result["feature_names"] = feature_names

                trait_results["two_stage"] = result
            else:
                print("    No data available")

        # ── Full concatenation probe ──
        if args.approach in ("full", "both"):
            total_features = n_layers * len(COMPONENTS) * hidden_dim
            print(f"\n  Full concatenation probe ({n_layers} layers × {len(COMPONENTS)} components × {hidden_dim} dim = {total_features:,} features)")

            X_train, y_train = load_multi_layer_activations(
                args.experiment, trait, args.model_variant, COMPONENTS, args.position, "train"
            )
            X_val, y_val = load_multi_layer_activations(
                args.experiment, trait, args.model_variant, COMPONENTS, args.position, "val"
            )

            if X_train is not None:
                n_train = len(X_train)
                print(f"    {n_train} training examples, {total_features:,} features (ratio: {total_features/n_train:.0f}:1)")

                result, weights, scaler = train_full_concat_probe(
                    X_train, y_train,
                    X_val if X_val is not None else None,
                    y_val if y_val is not None else None,
                )
                print(f"    Train: {result['train_acc']:.1%}", end="")
                if "val_acc" in result:
                    print(f"  |  Val: {result['val_acc']:.1%}", end="")
                print()
                print(f"    Non-zero weights: {result['n_nonzero']:,} / {total_features:,} ({1-result['sparsity']:.2%})")

                weight_map = extract_weight_map(weights, n_layers, COMPONENTS, hidden_dim)
                if weight_map is not None:
                    print_weight_map(weight_map, COMPONENTS, n_layers)
                    result["weight_map"] = weight_map.tolist()

                trait_results["full_concat"] = result
            else:
                print("    No data available")

        all_results[trait] = trait_results

    # Save results
    output_path = output_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {output_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Trait':<35s} {'Two-stage CV':>14s} {'Full Train':>14s} {'Full Val':>14s} {'Sparsity':>10s}")
    print(f"  {'─'*35} {'─'*14} {'─'*14} {'─'*14} {'─'*10}")
    for trait, res in all_results.items():
        ts = res.get("two_stage", {})
        fc = res.get("full_concat", {})
        ts_cv = f"{ts['cv_mean']:.1%} ± {ts['cv_std']:.1%}" if ts else "—"
        fc_train = f"{fc['train_acc']:.1%}" if fc else "—"
        fc_val = f"{fc['val_acc']:.1%}" if fc.get("val_acc") else "—"
        fc_sp = f"{fc.get('sparsity', 0):.1%}" if fc else "—"
        print(f"  {trait:<35s} {ts_cv:>14s} {fc_train:>14s} {fc_val:>14s} {fc_sp:>10s}")


if __name__ == "__main__":
    main()
