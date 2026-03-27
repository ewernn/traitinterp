#!/usr/bin/env python3
"""
Project raw captured activations onto trait vectors at all layers.
Produces per-trait, per-layer signal-to-noise metrics.

Input:
    - experiments/starter/inference/qwen3.5-9b/raw/residual/starter_prompts/general/*.pt
    - experiments/starter/extraction/starter_traits/*/qwen3.5-9b/vectors/response_all/residual/probe/

Output:
    - results/inference_layer_profiles.json

Usage:
    python dev/tasks/detection-layer-profiling/results/project_all_layers.py
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/dev/trait-interp")
sys.path.insert(0, str(ROOT))

from utils.vectors import load_vector_with_baseline

RESULTS_DIR = Path(__file__).parent


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-variant", default="qwen3.5-9b")
    args = parser.parse_args()

    experiment = "starter"
    model_variant = args.model_variant
    prompt_set = "starter_prompts/general"
    n_layers = 32

    raw_dir = ROOT / f"experiments/{experiment}/inference/{model_variant}/raw/residual/{prompt_set}"
    response_dir = ROOT / f"experiments/{experiment}/inference/{model_variant}/responses/{prompt_set}"

    raw_files = sorted(raw_dir.glob("*.pt"), key=lambda f: int(f.stem))
    print(f"Found {len(raw_files)} raw activation files")

    traits = [
        "starter_traits/sycophancy",
        "starter_traits/evil",
        "starter_traits/refusal",
        "starter_traits/concealment",
        "starter_traits/hallucination",
        "starter_traits/golden_gate_bridge",
    ]

    # Load vectors for all traits at all layers
    vectors = {}  # (trait, layer) -> vector tensor
    baselines = {}
    for trait in traits:
        for layer in range(n_layers):
            try:
                v, baseline, _ = load_vector_with_baseline(
                    experiment, trait, "probe", layer, model_variant, "residual", "response[:]"
                )
                vectors[(trait, layer)] = v
                baselines[(trait, layer)] = baseline
            except FileNotFoundError:
                pass

    print(f"Loaded {len(vectors)} trait-layer vectors")

    # Process each prompt
    all_projections = {}  # prompt_id -> {trait -> {layer -> [per_token_scores]}}

    for raw_file in raw_files:
        prompt_id = int(raw_file.stem)

        # Load raw activations: dict with 'prompt' and 'response' sections
        # Each section has activations[layer_idx]['residual'] -> tensor [n_tokens, hidden_dim]
        raw = torch.load(raw_file, map_location='cpu', weights_only=False)

        # Get response text from the raw file itself
        prompt_text = raw.get('prompt', {}).get('text', '')[:100] if isinstance(raw.get('prompt'), dict) else ''
        response_section = raw.get('response', {})
        response_text = response_section.get('text', '')[:100] if isinstance(response_section, dict) else ''

        # Extract response activations per layer
        response_acts = response_section.get('activations', {}) if isinstance(response_section, dict) else {}

        prompt_projections = {}

        for trait in traits:
            trait_name = trait.split('/')[-1]
            layer_scores = {}

            for layer in range(n_layers):
                if (trait, layer) not in vectors:
                    continue
                if layer not in response_acts:
                    continue

                v = vectors[(trait, layer)]
                baseline = baselines[(trait, layer)]

                # Get activations for this layer
                layer_data = response_acts[layer]
                if isinstance(layer_data, dict) and 'residual' in layer_data:
                    layer_acts = layer_data['residual'].float()
                else:
                    continue

                if layer_acts.dim() != 2:
                    continue

                # Cosine similarity
                v_float = v.float()
                v_norm = v_float / v_float.norm()
                act_norms = layer_acts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                cos_sim = (layer_acts @ v_norm.unsqueeze(-1)).squeeze(-1) / act_norms.squeeze(-1)

                scores = cos_sim.numpy().tolist()
                layer_scores[layer] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'max': float(np.max(scores)),
                    'min': float(np.min(scores)),
                    'range': float(np.max(scores) - np.min(scores)),
                }

            prompt_projections[trait_name] = layer_scores

        all_projections[prompt_id] = {
            'prompt': prompt_text[:100],
            'response': response_text[:100],
            'projections': prompt_projections,
        }

    # Aggregate: for each trait, compute per-layer signal quality
    print(f"\n{'='*80}")
    print("INFERENCE LAYER PROFILES — Per-Token Signal Quality")
    print(f"{'='*80}")

    trait_layer_stats = {}

    for trait in traits:
        trait_name = trait.split('/')[-1]
        layer_means = defaultdict(list)
        layer_ranges = defaultdict(list)

        for pid, pdata in all_projections.items():
            if trait_name in pdata['projections']:
                for layer_str, scores in pdata['projections'][trait_name].items():
                    layer = int(layer_str)
                    layer_means[layer].append(scores['mean'])
                    layer_ranges[layer].append(scores['range'])

        print(f"\n  {trait_name}:")
        print(f"  {'Layer':<6} {'Mean proj':<12} {'Mean range':<12} {'Std of means':<14}")

        trait_stats = {}
        for layer in sorted(layer_means.keys()):
            means = layer_means[layer]
            ranges = layer_ranges[layer]
            avg_mean = np.mean(means)
            avg_range = np.mean(ranges)
            std_means = np.std(means)
            print(f"  L{layer:<4} {avg_mean:<12.4f} {avg_range:<12.4f} {std_means:<14.4f}")

            trait_stats[layer] = {
                'avg_mean_projection': float(avg_mean),
                'avg_token_range': float(avg_range),
                'std_across_prompts': float(std_means),
                'signal_quality': float(avg_range * std_means) if std_means > 0 else 0,
            }

        # Best layer by signal quality (range × variance across prompts)
        if trait_stats:
            best_layer = max(trait_stats, key=lambda l: trait_stats[l]['signal_quality'])
            print(f"  Best inference layer (by signal quality): L{best_layer}")

        trait_layer_stats[trait_name] = trait_stats

    # Save
    output = RESULTS_DIR / "inference_layer_profiles.json"
    json.dump({
        'trait_layer_stats': trait_layer_stats,
        'per_prompt': all_projections,
    }, open(output, 'w'), indent=2, default=str)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
