#!/usr/bin/env python3
"""
Evaluate alignment between trait vectors and SAE decoder directions.

Computes cosine similarity between a trait vector and all SAE decoder directions,
then looks up the top-aligned features in Neuronpedia labels.

Usage:
    python sae/evaluate_trait_alignment.py --experiment gemma_2b_cognitive_nov21 --trait behavioral/refusal --layers 10,16,22
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import json
import argparse
from sae_lens import SAE
from utils.paths import get_vector_path


def load_trait_vector(
    experiment: str,
    trait: str,
    method: str = "probe",
    layer: int = 16,
    model_variant: str = "base",
    component: str = "residual",
    position: str = "response[:]",
):
    """Load a trait vector from the experiment."""
    vector_path = get_vector_path(experiment, trait, method, layer, model_variant, component, position)

    if not vector_path.exists():
        raise FileNotFoundError(f"No vector found at {vector_path}")

    vector = torch.load(vector_path, map_location="cpu", weights_only=True)
    if isinstance(vector, dict):
        vector = vector.get('vector', vector.get('weights'))
    return vector / vector.norm()  # Normalize


def load_feature_labels(layer: int):
    """Load feature labels for a layer."""
    labels_path = Path(f"sae/gemma-scope-2b-pt-res-canonical/layer_{layer}_width_16k_canonical/feature_labels.json")
    if not labels_path.exists():
        return None
    with open(labels_path) as f:
        return json.load(f)


def evaluate_alignment(trait_vector: torch.Tensor, sae: SAE, labels: dict, top_k: int = 20):
    """
    Compute alignment between trait vector and SAE decoder directions.

    Returns top-k aligned features with their labels.
    """
    # SAE decoder: [d_sae, d_model] - each row is a feature direction
    W_dec = sae.W_dec.detach().cpu().float()  # [16384, 2304]

    # Normalize decoder directions
    W_dec_norm = W_dec / W_dec.norm(dim=1, keepdim=True)

    # Ensure trait vector is float32 (same as SAE)
    trait_vector = trait_vector.float()

    # Cosine similarity: trait_vector @ W_dec.T
    # trait_vector: [2304], W_dec_norm: [16384, 2304]
    similarities = (W_dec_norm @ trait_vector).squeeze()  # [16384]

    # Get top-k (both positive and negative alignment)
    top_pos_vals, top_pos_idx = similarities.topk(top_k)
    top_neg_vals, top_neg_idx = (-similarities).topk(top_k)

    features_dict = labels.get('features', {}) if labels else {}

    results = {
        'positive_alignment': [],  # Features that point same direction as trait
        'negative_alignment': [],  # Features that point opposite direction
    }

    for val, idx in zip(top_pos_vals.tolist(), top_pos_idx.tolist()):
        feature_info = features_dict.get(str(idx), {})
        results['positive_alignment'].append({
            'feature_idx': idx,
            'similarity': val,
            'description': feature_info.get('description', f'Feature {idx} (no label)')
        })

    for val, idx in zip(top_neg_vals.tolist(), top_neg_idx.tolist()):
        feature_info = features_dict.get(str(idx), {})
        results['negative_alignment'].append({
            'feature_idx': idx,
            'similarity': -val,  # Flip back to show actual negative value
            'description': feature_info.get('description', f'Feature {idx} (no label)')
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trait-SAE alignment")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    parser.add_argument("--trait", type=str, required=True, help="Trait path (e.g., behavioral/refusal)")
    parser.add_argument("--layers", type=str, default="10,16,22", help="Comma-separated layers to evaluate")
    parser.add_argument("--method", type=str, default="probe", help="Extraction method (default: probe)")
    parser.add_argument("--vector-layer", type=int, default=16, help="Layer the trait vector was extracted from")
    parser.add_argument("--model-variant", type=str, default="base", help="Model variant (default: base)")
    parser.add_argument("--component", type=str, default="residual", help="Component (default: residual)")
    parser.add_argument("--position", type=str, default="response[:]", help="Position (default: response[:])")
    parser.add_argument("--top-k", type=int, default=10, help="Top features to show")
    args = parser.parse_args()

    layers = [int(l) for l in args.layers.split(",")]

    print(f"=" * 70)
    print(f"TRAIT-SAE ALIGNMENT EVALUATION")
    print(f"=" * 70)
    print(f"Experiment: {args.experiment}")
    print(f"Trait: {args.trait}")
    print(f"Trait vector: {args.method}_layer{args.vector_layer}")
    print(f"SAE layers: {layers}")
    print()

    # Load trait vector
    print("Loading trait vector...")
    trait_vector = load_trait_vector(
        args.experiment, args.trait, args.method, args.vector_layer,
        args.model_variant, args.component, args.position
    )
    print(f"  Shape: {trait_vector.shape}, Norm: {trait_vector.norm():.4f}")
    print()

    # Evaluate each layer
    for layer in layers:
        print(f"-" * 70)
        print(f"LAYER {layer}")
        print(f"-" * 70)

        # Load SAE
        print(f"Loading SAE for layer {layer}...")
        try:
            sae = SAE.from_pretrained(
                "gemma-scope-2b-pt-res-canonical",
                f"layer_{layer}/width_16k/canonical",
                device="cpu"
            )
        except Exception as e:
            print(f"  ERROR loading SAE: {e}")
            continue

        # Load labels
        labels = load_feature_labels(layer)
        if labels:
            print(f"  Labels: {labels['stats']['downloaded']} features")
        else:
            print(f"  Labels: NOT FOUND - run: python sae/download_subset.py -l {layer} -n 1000")

        # Evaluate alignment
        results = evaluate_alignment(trait_vector, sae, labels, args.top_k)

        print(f"\n  TOP {args.top_k} POSITIVE ALIGNMENT (same direction as trait):")
        for i, feat in enumerate(results['positive_alignment']):
            print(f"    {i+1:2}. [{feat['feature_idx']:5}] sim={feat['similarity']:+.3f}  {feat['description'][:60]}")

        print(f"\n  TOP {args.top_k} NEGATIVE ALIGNMENT (opposite direction):")
        for i, feat in enumerate(results['negative_alignment']):
            print(f"    {i+1:2}. [{feat['feature_idx']:5}] sim={feat['similarity']:+.3f}  {feat['description'][:60]}")

        print()

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
