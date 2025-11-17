#!/usr/bin/env python3
"""
Encode raw activations to SAE features for visualization.

This script takes raw activation files from inference and encodes them
using the GemmaScope SAE, then saves the encoded features in a compact format.

Usage:
    # Encode all prompts for an experiment
    python analysis/encode_sae_features.py --experiment gemma_2b_cognitive_nov20

    # Encode specific trait
    python analysis/encode_sae_features.py --experiment gemma_2b_cognitive_nov20 --trait refusal
"""

import torch
from sae_lens import SAE
from pathlib import Path
import json
import argparse
from tqdm import tqdm


def load_sae(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_16/width_16k/canonical",
    device="cpu"
):
    """Load the SAE model."""
    print(f"Loading SAE: {release} / {sae_id}")
    sae = SAE.from_pretrained(release, sae_id, device=device)
    print(f"✓ Loaded: {sae.cfg.d_in} → {sae.cfg.d_sae} features")
    return sae


def encode_inference_file(
    activation_file: Path,
    sae: SAE,
    output_file: Path,
    top_k: int = 50
):
    """
    Encode a single inference activation file to SAE features.

    Args:
        activation_file: Path to raw activations (.pt file)
        sae: Loaded SAE model
        output_file: Where to save encoded features
        top_k: Number of top features to keep per token (for compact storage)
    """
    # Load raw activations
    # Format depends on what inference script saved - adapt as needed
    data = torch.load(activation_file, map_location="cpu")

    # Try to extract raw activations
    # This depends on your inference script's output format
    if isinstance(data, dict):
        # Option 1: Dict with 'activations' key
        if 'activations' in data:
            raw_acts = data['activations']
        # Option 2: Dict with layer-specific keys
        elif 'layer_16' in data:
            raw_acts = data['layer_16']
        # Option 3: Your format from monitor_dynamics.py
        elif 'attention_weights' in data:
            # This file doesn't have raw activations, skip
            print(f"  ⚠️  Skipping {activation_file.name} - no raw activations found")
            return None
        else:
            print(f"  ⚠️  Unknown format in {activation_file.name}")
            print(f"      Keys: {list(data.keys())}")
            return None
    elif isinstance(data, torch.Tensor):
        # Option 4: Just a tensor
        raw_acts = data
    else:
        print(f"  ⚠️  Unknown format: {type(data)}")
        return None

    # Ensure correct shape [seq_len, hidden_dim]
    if raw_acts.dim() == 3:
        # [batch, seq, hidden] → [seq, hidden] (take first batch item)
        raw_acts = raw_acts[0]

    expected_dim = sae.cfg.d_in
    if raw_acts.shape[-1] != expected_dim:
        print(f"  ⚠️  Dimension mismatch: got {raw_acts.shape[-1]}, expected {expected_dim}")
        return None

    # Encode to SAE features
    with torch.no_grad():
        features = sae.encode(raw_acts)  # [seq_len, d_sae]

    # Get top-k per token for compact storage
    top_k_values, top_k_indices = features.topk(k=top_k, dim=-1)

    # Prepare output
    output_data = {
        # Metadata
        'sae_release': 'gemma-scope-2b-pt-res-canonical',
        'sae_id': 'layer_16/width_16k/canonical',
        'sae_labels_path': 'sae/gemma-scope-2b-pt-res-canonical/layer_16_width_16k_canonical/feature_labels.json',

        # Original prompt/response if available
        'prompt': data.get('prompt', ''),
        'response': data.get('response', ''),
        'tokens': data.get('tokens', []),

        # Encoded features (top-k only for compact storage)
        'top_k_features': {
            'indices': top_k_indices,  # [seq_len, k]
            'values': top_k_values,    # [seq_len, k]
            'k': top_k
        },

        # Stats
        'num_tokens': raw_acts.shape[0],
        'avg_active_features': (features.abs() > 1e-6).sum(dim=-1).float().mean().item(),
    }

    # Save
    torch.save(output_data, output_file)

    return {
        'file': output_file.name,
        'num_tokens': output_data['num_tokens'],
        'avg_active': output_data['avg_active_features']
    }


def encode_experiment(
    experiment: str,
    trait: str = None,
    layer: int = 16,
    device: str = "cpu",
    top_k: int = 50
):
    """
    Encode all inference activations for an experiment to SAE features.

    Args:
        experiment: Experiment name (e.g., 'gemma_2b_cognitive_nov20')
        trait: Optional trait name (if None, process all traits)
        layer: Layer number to encode (default: 16)
        device: Device to run SAE on
        top_k: Number of top features to keep per token
    """
    base_path = Path("experiments") / experiment

    if not base_path.exists():
        print(f"✗ Experiment not found: {base_path}")
        return

    # Find traits to process
    if trait:
        traits = [trait]
    else:
        # Find all trait directories
        traits = [d.name for d in base_path.iterdir() if d.is_dir()]

    print(f"Processing {len(traits)} trait(s) in experiment '{experiment}'")
    print()

    # Load SAE once
    sae = load_sae(device=device)
    print()

    # Process each trait
    total_encoded = 0
    total_failed = 0

    for trait_name in traits:
        trait_path = base_path / trait_name / "inference"

        if not trait_path.exists():
            print(f"⚠️  No inference data for trait '{trait_name}'")
            continue

        # Find activation files
        # Look in various possible locations
        activation_files = []

        # Check layer_internal_states/
        layer_states_dir = trait_path / "layer_internal_states"
        if layer_states_dir.exists():
            activation_files.extend(layer_states_dir.glob(f"*_layer{layer}.pt"))

        # Check residual_stream_activations/
        residual_dir = trait_path / "residual_stream_activations"
        if residual_dir.exists():
            activation_files.extend(residual_dir.glob("prompt_*.pt"))

        if not activation_files:
            print(f"⚠️  No activation files found for '{trait_name}'")
            continue

        print(f"Trait: {trait_name}")
        print(f"  Found {len(activation_files)} activation file(s)")

        # Create output directory
        output_dir = trait_path / "sae_features"
        output_dir.mkdir(exist_ok=True)

        # Encode each file
        for act_file in tqdm(activation_files, desc=f"  Encoding"):
            output_file = output_dir / f"{act_file.stem}_sae.pt"

            result = encode_inference_file(act_file, sae, output_file, top_k=top_k)

            if result:
                total_encoded += 1
            else:
                total_failed += 1

        print(f"  ✓ Encoded {len(activation_files)} files → {output_dir}")
        print()

    print("="*60)
    print("ENCODING COMPLETE")
    print("="*60)
    print(f"Total encoded: {total_encoded}")
    print(f"Failed: {total_failed}")

    if total_encoded > 0:
        print(f"\n✓ SAE features saved to: experiments/{experiment}/*/inference/sae_features/")
        print(f"\nNext steps:")
        print(f"  1. Download feature labels: python sae/download_feature_labels.py")
        print(f"  2. View in visualization: python -m http.server 8000")


def main():
    parser = argparse.ArgumentParser(description="Encode activations to SAE features")
    parser.add_argument("--experiment", type=str, required=True,
                       help="Experiment name (e.g., 'gemma_2b_cognitive_nov20')")
    parser.add_argument("--trait", type=str, default=None,
                       help="Specific trait to process (default: all)")
    parser.add_argument("--layer", type=int, default=16,
                       help="Layer number (default: 16)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run SAE on (default: cpu)")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Number of top features to keep per token (default: 50)")

    args = parser.parse_args()

    encode_experiment(
        experiment=args.experiment,
        trait=args.trait,
        layer=args.layer,
        device=args.device,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()
