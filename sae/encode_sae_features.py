#!/usr/bin/env python3
"""
Encode raw activations to SAE features for visualization.

Input: Raw activation .pt files from inference/raw/residual/{prompt_set}/
Output: Encoded SAE features with top-k per token

Usage:
    # Encode all prompt sets for an experiment
    python sae/encode_sae_features.py --experiment gemma_2b_cognitive_nov21

    # Encode specific prompt set
    python sae/encode_sae_features.py --experiment gemma_2b_cognitive_nov21 --prompt-set single_trait

    # Use MPS on Mac
    python sae/encode_sae_features.py --experiment gemma_2b_cognitive_nov21 --device mps
"""

import torch
from sae_lens import SAE
from pathlib import Path
import json
import argparse
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.paths import get as get_path


def load_sae(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_16/width_16k/canonical",
    device="cpu"
):
    """Load the SAE model from HuggingFace."""
    print(f"Loading SAE: {release} / {sae_id}")
    sae = SAE.from_pretrained(release, sae_id, device=device)
    print(f"  Input dim: {sae.cfg.d_in}")
    print(f"  SAE features: {sae.cfg.d_sae}")
    return sae


def encode_raw_file(
    raw_file: Path,
    sae: SAE,
    output_file: Path,
    layer: int = 16,
    component: str = "residual",
    top_k: int = 50,
    include_prompt: bool = True
):
    """
    Encode a single raw activation file to SAE features.

    Args:
        raw_file: Path to raw .pt file from capture_raw_activations.py
        sae: Loaded SAE model
        output_file: Where to save encoded features
        layer: Which layer to encode (default: 16)
        component: Which component - 'attn_out', 'residual'
        top_k: Number of top features to keep per token
        include_prompt: Whether to also encode prompt tokens (default: response only)

    Returns:
        dict with encoding stats, or None if failed
    """
    # Load raw data
    data = torch.load(raw_file, map_location="cpu", weights_only=False)

    # Expected format from capture_raw_activations.py:
    # {
    #   'prompt': {'text', 'tokens', 'token_ids', 'activations': {layer: {component: tensor}}, 'attention'},
    #   'response': {'text', 'tokens', 'token_ids', 'activations': {layer: {component: tensor}}, 'attention'}
    # }

    if not isinstance(data, dict) or 'response' not in data:
        print(f"  Skipping {raw_file.name} - unexpected format")
        return None

    response = data['response']
    if 'activations' not in response:
        print(f"  Skipping {raw_file.name} - no activations in response")
        return None

    if layer not in response['activations']:
        print(f"  Skipping {raw_file.name} - layer {layer} not found")
        return None

    layer_data = response['activations'][layer]
    if component not in layer_data:
        print(f"  Skipping {raw_file.name} - component '{component}' not found (available: {list(layer_data.keys())})")
        return None

    # Get response activations
    response_acts = layer_data[component]  # [seq_len, hidden_dim]
    response_tokens = response.get('tokens', [])
    prompt_tokens = []
    n_prompt_tokens = 0

    # Include prompt activations and tokens
    if include_prompt and 'prompt' in data:
        prompt = data['prompt']
        prompt_tokens = prompt.get('tokens', [])
        n_prompt_tokens = len(prompt_tokens)
        if 'activations' in prompt and layer in prompt['activations']:
            prompt_acts = prompt['activations'][layer].get(component)
            if prompt_acts is not None:
                response_acts = torch.cat([prompt_acts, response_acts], dim=0)

    # Combined tokens (prompt + response)
    all_tokens = prompt_tokens + response_tokens

    # Verify dimensions
    expected_dim = sae.cfg.d_in
    if response_acts.shape[-1] != expected_dim:
        print(f"  Skipping {raw_file.name} - dim mismatch: {response_acts.shape[-1]} vs {expected_dim}")
        return None

    # Move to same device as SAE
    response_acts = response_acts.to(sae.device)

    # Encode to SAE features
    with torch.no_grad():
        features = sae.encode(response_acts)  # [seq_len, d_sae]

    # Get top-k per token for compact storage
    top_k_values, top_k_indices = features.topk(k=min(top_k, features.shape[-1]), dim=-1)

    # Compute sparsity stats
    active_per_token = (features.abs() > 1e-6).sum(dim=-1).float()

    # Prepare output
    output_data = {
        # Metadata
        'source_file': str(raw_file.name),
        'sae_release': 'gemma-scope-2b-pt-res-canonical',
        'sae_id': 'layer_16/width_16k/canonical',
        'layer': layer,
        'component': component,

        # Original text
        'prompt_text': data['prompt']['text'] if 'prompt' in data else '',
        'response_text': response.get('text', ''),
        'tokens': all_tokens,
        'n_prompt_tokens': n_prompt_tokens,

        # Encoded features (top-k only)
        'top_k_indices': top_k_indices.cpu(),  # [seq_len, k]
        'top_k_values': top_k_values.cpu(),    # [seq_len, k]
        'k': top_k,

        # Stats
        'num_tokens': response_acts.shape[0],
        'avg_active_features': active_per_token.mean().item(),
        'min_active_features': active_per_token.min().item(),
        'max_active_features': active_per_token.max().item(),
    }

    # Save as JSON (for web visualization)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert tensors to lists for JSON serialization
    json_data = {
        'source_file': output_data['source_file'],
        'sae_release': output_data['sae_release'],
        'sae_id': output_data['sae_id'],
        'layer': output_data['layer'],
        'component': output_data['component'],
        'prompt_text': output_data['prompt_text'],
        'response_text': output_data['response_text'],
        'tokens': output_data['tokens'],
        'n_prompt_tokens': output_data['n_prompt_tokens'],
        'top_k_indices': output_data['top_k_indices'].tolist(),
        'top_k_values': output_data['top_k_values'].tolist(),
        'k': output_data['k'],
        'num_tokens': output_data['num_tokens'],
        'avg_active_features': output_data['avg_active_features'],
        'min_active_features': output_data['min_active_features'],
        'max_active_features': output_data['max_active_features'],
    }

    # Save as JSON with .pt.json extension (so visualization can fetch it)
    json_output = output_file.with_suffix('.pt.json')
    with open(json_output, 'w') as f:
        json.dump(json_data, f)

    return {
        'file': raw_file.name,
        'num_tokens': output_data['num_tokens'],
        'avg_active': output_data['avg_active_features']
    }


def encode_experiment(
    experiment: str,
    prompt_set: str = None,
    layer: int = 16,
    component: str = "residual",
    device: str = "cpu",
    top_k: int = 50
):
    """
    Encode all raw activations for an experiment to SAE features.

    Args:
        experiment: Experiment name
        prompt_set: Specific prompt set (if None, process all)
        layer: Layer to encode
        component: Component - 'attn_out', 'residual'
        device: Device for SAE
        top_k: Number of top features per token
    """
    # Raw activations location
    raw_base = get_path('inference.raw_residual', experiment=experiment, prompt_set='').parent

    if not raw_base.exists():
        print(f"No raw activations found at: {raw_base}")
        return

    # Find prompt sets to process
    if prompt_set:
        prompt_sets = [raw_base / prompt_set]
    else:
        prompt_sets = [d for d in raw_base.iterdir() if d.is_dir()]

    if not prompt_sets:
        print(f"No prompt sets found in {raw_base}")
        return

    print(f"Experiment: {experiment}")
    print(f"Raw data: {raw_base}")
    print(f"Prompt sets: {[p.name for p in prompt_sets]}")
    print(f"Layer: {layer}, Component: {component}")
    print()

    # Load SAE once
    sae = load_sae(device=device)
    print()

    # Output base directory
    output_base = get_path('inference.sae', experiment=experiment)

    # Process each prompt set
    total_encoded = 0
    total_failed = 0

    for ps_dir in prompt_sets:
        if not ps_dir.is_dir():
            continue

        raw_files = sorted(ps_dir.glob("*.pt"))
        if not raw_files:
            print(f"  No .pt files in {ps_dir.name}")
            continue

        print(f"Prompt set: {ps_dir.name} ({len(raw_files)} files)")

        output_dir = output_base / ps_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        for raw_file in tqdm(raw_files, desc=f"  Encoding", leave=False):
            output_file = output_dir / f"{raw_file.stem}_sae.pt"

            result = encode_raw_file(
                raw_file, sae, output_file,
                layer=layer, component=component, top_k=top_k
            )

            if result:
                total_encoded += 1
            else:
                total_failed += 1

        print(f"  Saved to: {output_dir}")

    print()
    print("=" * 60)
    print("ENCODING COMPLETE")
    print("=" * 60)
    print(f"Encoded: {total_encoded}")
    print(f"Failed: {total_failed}")
    print(f"\nOutput: {output_base}")


def main():
    parser = argparse.ArgumentParser(description="Encode raw activations to SAE features")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name")
    parser.add_argument("--prompt-set", type=str, default=None,
                        help="Specific prompt set (default: all)")
    parser.add_argument("--layer", type=int, default=16,
                        help="Layer to encode (default: 16)")
    parser.add_argument("--component", type=str, default="residual",
                        choices=["attn_out", "residual"],
                        help="Activation component (default: residual)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for SAE (cpu, cuda, mps)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top features per token (default: 50)")

    args = parser.parse_args()

    encode_experiment(
        experiment=args.experiment,
        prompt_set=args.prompt_set,
        layer=args.layer,
        component=args.component,
        device=args.device,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()
