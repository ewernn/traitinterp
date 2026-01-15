#!/usr/bin/env python3
"""
Analyze massive activation dimensions in raw activations.

Identifies which dimensions have abnormally large values (the "massive activations"
from Sun et al. 2024) and tracks their behavior across tokens and layers.

Default mode uses calibration dataset (datasets/inference/massive_dims/calibration_50.json)
to compute model-specific massive dims independent of any specific experiment.

Input: Raw activations (auto-captured if missing for calibration)
Output: JSON with massive dim stats for visualization

Usage:
    # Default: calibrate model using Alpaca prompts
    python analysis/massive_activations.py --experiment gemma-2-2b

    # Analyze specific prompt set (research mode)
    python analysis/massive_activations.py --experiment gemma-2-2b --prompt-set jailbreak_subset

    # Include per-token analysis (verbose)
    python analysis/massive_activations.py --experiment gemma-2-2b --prompt-set X --per-token
"""

import argparse
import json
import subprocess
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get as get_path, get_model_variant

# Calibration dataset path
CALIBRATION_DATASET = Path(__file__).parent.parent / 'datasets' / 'inference' / 'massive_dims' / 'calibration_50.json'
CALIBRATION_PROMPT_SET = '_calibration'  # Internal name for calibration runs


def find_massive_dims(
    activations: Dict[int, Dict[str, torch.Tensor]],
    top_k: int = 5,
    threshold_ratio: float = 100.0,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Find massive activation dimensions per layer.

    Args:
        activations: {layer: {'residual': [seq_len, hidden_dim], ...}}
        top_k: Number of top dimensions to return per layer
        threshold_ratio: Ratio vs median to consider "massive"

    Returns:
        {layer: [{'dim': int, 'max_val': float, 'mean_val': float, 'ratio': float}, ...]}
    """
    results = {}

    for layer, layer_data in activations.items():
        if 'residual' not in layer_data:
            continue

        residual = layer_data['residual'].float()  # [seq_len, hidden_dim]

        # Compute max absolute value per dimension across all tokens
        dim_max = residual.abs().max(dim=0).values  # [hidden_dim]
        dim_mean = residual.abs().mean(dim=0)  # [hidden_dim]

        # Find top-k dimensions by max value
        top_vals, top_dims = dim_max.topk(top_k)

        # Compute ratio to median
        median_val = dim_max.median().item()

        layer_results = []
        for i in range(top_k):
            dim_idx = top_dims[i].item()
            max_val = top_vals[i].item()
            mean_val = dim_mean[dim_idx].item()
            ratio = max_val / median_val if median_val > 0 else float('inf')

            layer_results.append({
                'dim': dim_idx,
                'max_val': round(max_val, 2),
                'mean_val': round(mean_val, 4),
                'ratio': round(ratio, 1),
                'is_massive': ratio > threshold_ratio,
            })

        results[layer] = layer_results

    return results


def track_dim_values(
    activations: Dict[int, Dict[str, torch.Tensor]],
    dims: List[int],
) -> Dict[int, Dict[int, List[float]]]:
    """
    Track specific dimensions' values across tokens for each layer.

    Args:
        activations: {layer: {'residual': [seq_len, hidden_dim], ...}}
        dims: List of dimension indices to track

    Returns:
        {layer: {dim: [val_token_0, val_token_1, ...]}}
    """
    results = {}

    for layer, layer_data in activations.items():
        if 'residual' not in layer_data:
            continue

        residual = layer_data['residual'].float()  # [seq_len, hidden_dim]

        layer_results = {}
        for dim in dims:
            if dim < residual.shape[1]:
                values = residual[:, dim].tolist()
                layer_results[dim] = [round(v, 4) for v in values]

        results[layer] = layer_results

    return results


def compute_mean_alignment(
    activations: Dict[int, Dict[str, torch.Tensor]],
) -> Dict[int, Dict[str, float]]:
    """
    Compute how much each token aligns with mean direction.

    Returns:
        {layer: {'mean': float, 'min': float, 'max': float, 'std': float}}
    """
    results = {}

    for layer, layer_data in activations.items():
        if 'residual' not in layer_data:
            continue

        residual = layer_data['residual'].float()  # [seq_len, hidden_dim]

        # Compute mean direction
        mean_dir = residual.mean(dim=0)  # [hidden_dim]
        mean_dir_norm = mean_dir / (mean_dir.norm() + 1e-8)

        # Compute cosine similarity with mean for each token
        token_norms = residual.norm(dim=1, keepdim=True)
        token_normalized = residual / (token_norms + 1e-8)
        cosines = (token_normalized @ mean_dir_norm).tolist()

        results[layer] = {
            'mean': round(sum(cosines) / len(cosines), 4),
            'min': round(min(cosines), 4),
            'max': round(max(cosines), 4),
            'std': round(torch.tensor(cosines).std().item(), 4),
        }

    return results


def analyze_prompt(
    pt_path: Path,
    top_k: int = 5,
    track_dims: List[int] = None,
    per_token: bool = True,
) -> Dict[str, Any]:
    """Analyze a single prompt's activations."""

    data = torch.load(pt_path, weights_only=False, map_location='cpu')

    # Combine prompt and response activations
    prompt_acts = data.get('prompt', {}).get('activations', {})
    response_acts = data.get('response', {}).get('activations', {})

    # Get tokens
    prompt_tokens = data.get('prompt', {}).get('tokens', [])
    response_tokens = data.get('response', {}).get('tokens', [])

    # Find massive dims in prompt (more stable than response)
    massive_dims = find_massive_dims(prompt_acts, top_k=top_k)

    # Determine which dims to track (union of top dims across layers)
    if track_dims is None:
        all_massive = set()
        for layer_dims in massive_dims.values():
            for d in layer_dims:
                if d['is_massive']:
                    all_massive.add(d['dim'])
        track_dims = sorted(all_massive)[:5]  # Limit to top 5

    result = {
        'massive_dims': {int(k): v for k, v in massive_dims.items()},
        'tracked_dims': track_dims,
    }

    # Only include per-token analysis if requested
    if per_token:
        result['prompt_tokens'] = prompt_tokens
        result['response_tokens'] = response_tokens

        # Track dim values across tokens
        prompt_dim_values = track_dim_values(prompt_acts, track_dims)
        response_dim_values = track_dim_values(response_acts, track_dims)

        result['prompt_dim_values'] = {int(k): {int(d): v for d, v in dims.items()}
                                       for k, dims in prompt_dim_values.items()}
        result['response_dim_values'] = {int(k): {int(d): v for d, v in dims.items()}
                                         for k, dims in response_dim_values.items()}

        # Compute mean alignment
        prompt_mean_align = compute_mean_alignment(prompt_acts)
        response_mean_align = compute_mean_alignment(response_acts)

        result['prompt_mean_alignment'] = {int(k): v for k, v in prompt_mean_align.items()}
        result['response_mean_alignment'] = {int(k): v for k, v in response_mean_align.items()}

    return result


def compute_layer_stats(
    pt_files: List[Path],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Compute per-layer mean activations and find massive dims across all prompts.

    Returns:
        {
            'top_dims_by_layer': {layer: [dim1, dim2, ...]},  # top-k dims per layer
            'dim_magnitude_by_layer': {dim: [mag_L0, mag_L1, ...]},  # normalized per layer
            'layer_norms': {layer: mean_norm},  # average ||h|| per layer
        }
    """
    # Accumulate response activations across all prompts
    layer_sums = {}  # {layer: sum tensor}
    layer_norm_sums = {}  # {layer: sum of L2 norms}
    layer_counts = {}  # {layer: token count}

    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=False, map_location='cpu')
        response_acts = data.get('response', {}).get('activations', {})

        for layer, layer_data in response_acts.items():
            if 'residual' not in layer_data:
                continue
            residual = layer_data['residual'].float()  # [n_tokens, hidden_dim]

            if layer not in layer_sums:
                layer_sums[layer] = torch.zeros(residual.shape[1])
                layer_norm_sums[layer] = 0.0
                layer_counts[layer] = 0

            layer_sums[layer] += residual.sum(dim=0)
            layer_norm_sums[layer] += residual.norm(dim=1).sum().item()  # sum of per-token norms
            layer_counts[layer] += residual.shape[0]

    # Compute mean per layer and average norm
    layer_means = {}  # {layer: mean tensor}
    layer_norms = {}  # {layer: average ||h||}
    for layer in sorted(layer_sums.keys()):
        layer_means[layer] = layer_sums[layer] / layer_counts[layer]
        layer_norms[layer] = round(layer_norm_sums[layer] / layer_counts[layer], 1)

    # Find top-k dims per layer and collect all candidate dims
    top_dims_by_layer = {}
    all_candidate_dims = set()

    for layer, mean_vec in layer_means.items():
        top_vals, top_dims = mean_vec.abs().topk(top_k)
        top_dims_list = top_dims.tolist()
        top_dims_by_layer[layer] = top_dims_list
        all_candidate_dims.update(top_dims_list)

    # Compute normalized magnitude for all candidate dims at each layer
    dim_magnitude_by_layer = {}
    for dim in sorted(all_candidate_dims):
        magnitudes = []
        for layer in sorted(layer_means.keys()):
            mean_vec = layer_means[layer]
            layer_avg = mean_vec.abs().mean().item()
            normalized = abs(mean_vec[dim].item()) / layer_avg if layer_avg > 0 else 0
            magnitudes.append(round(normalized, 3))
        dim_magnitude_by_layer[dim] = magnitudes

    return {
        'top_dims_by_layer': {int(k): v for k, v in top_dims_by_layer.items()},
        'dim_magnitude_by_layer': {int(k): v for k, v in dim_magnitude_by_layer.items()},
        'layer_norms': {int(k): v for k, v in layer_norms.items()},
    }


def aggregate_stats(
    all_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate stats across all prompts."""

    if not all_results:
        return {}

    n_prompts = len(all_results)

    # Aggregate mean alignment stats (only if per-token data exists)
    mean_alignment = {}
    if 'prompt_mean_alignment' in all_results[0]:
        for layer in all_results[0]['prompt_mean_alignment'].keys():
            layer_means = [r['prompt_mean_alignment'].get(layer, {}).get('mean', 0)
                           for r in all_results]
            mean_alignment[layer] = round(sum(layer_means) / len(layer_means), 4)

    return {
        'n_prompts': n_prompts,
        'mean_alignment_by_layer': {int(k): v for k, v in mean_alignment.items()},
    }


def ensure_calibration_activations(experiment: str, model_variant: str) -> Path:
    """
    Ensure calibration activations exist, capturing them if necessary.

    Returns the directory containing .pt files.
    """
    # Check if calibration activations exist
    raw_dir = Path(get_path('inference.raw_residual', experiment=experiment, model_variant=model_variant, prompt_set=CALIBRATION_PROMPT_SET))

    if raw_dir.exists() and list(raw_dir.glob('*.pt')):
        print(f"Using existing calibration activations: {raw_dir}")
        return raw_dir

    # Need to capture calibration activations
    print(f"Calibration activations not found. Capturing from {CALIBRATION_DATASET}...")

    if not CALIBRATION_DATASET.exists():
        raise FileNotFoundError(f"Calibration dataset not found: {CALIBRATION_DATASET}")

    # Run capture_raw_activations.py
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / 'inference' / 'capture_raw_activations.py'),
        '--experiment', experiment,
        '--prompts-file', str(CALIBRATION_DATASET),
        '--prompt-set-name', CALIBRATION_PROMPT_SET,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to capture calibration activations")

    return raw_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--model-variant', default=None, help='Model variant (default: from experiment config)')
    parser.add_argument('--prompt-set', default=None,
                        help='Prompt set to analyze (default: calibration dataset)')
    parser.add_argument('--prompts-file', default=None,
                        help='Direct path to prompts JSON (overrides --prompt-set lookup)')
    parser.add_argument('--prompt-ids', type=str, default=None,
                        help='Comma-separated prompt IDs (default: all)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Top K dims to track per layer')
    parser.add_argument('--per-token', action='store_true',
                        help='Include per-token analysis (verbose, for research)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: auto)')
    args = parser.parse_args()

    # Resolve model variant
    variant = get_model_variant(args.experiment, args.model_variant, mode="application")
    model_variant = variant['name']

    # Determine mode: calibration (default) or prompt-set analysis
    is_calibration = args.prompt_set is None
    prompt_set_name = CALIBRATION_PROMPT_SET if is_calibration else args.prompt_set

    if is_calibration:
        print("=== Calibration Mode ===")
        print("Computing model-specific massive dims from neutral prompts")
        raw_dir = ensure_calibration_activations(args.experiment, model_variant)
    else:
        print(f"=== Analysis Mode: {args.prompt_set} ===")
        raw_dir = Path(get_path('inference.raw_residual', experiment=args.experiment, model_variant=model_variant, prompt_set=args.prompt_set))
        if not raw_dir.exists():
            print(f"No raw activations found at {raw_dir}")
            print(f"Run: python inference/capture_raw_activations.py --experiment {args.experiment} --prompt-set {args.prompt_set}")
            return

    pt_files = sorted(raw_dir.glob('*.pt'))
    if not pt_files:
        print(f"No .pt files in {raw_dir}")
        return

    # Filter by prompt IDs if specified
    if args.prompt_ids:
        ids = set(args.prompt_ids.split(','))
        pt_files = [f for f in pt_files if f.stem in ids]

    print(f"Analyzing {len(pt_files)} prompts...")

    all_results = []
    per_prompt = {} if args.per_token else None

    for pt_file in pt_files:
        prompt_id = pt_file.stem
        print(f"  {prompt_id}...", end=' ')

        result = analyze_prompt(pt_file, top_k=args.top_k, per_token=args.per_token)
        all_results.append(result)

        if per_prompt is not None:
            per_prompt[prompt_id] = result

        # Print top massive dim for this prompt
        if result['tracked_dims']:
            print(f"massive dims: {result['tracked_dims']}")
        else:
            print("no massive dims found")

    # Aggregate stats
    aggregate = aggregate_stats(all_results)

    # Compute per-layer stats for visualization
    print("\nComputing layer stats for visualization...")
    layer_stats = compute_layer_stats(pt_files, top_k=args.top_k)
    aggregate['top_dims_by_layer'] = layer_stats['top_dims_by_layer']
    aggregate['dim_magnitude_by_layer'] = layer_stats['dim_magnitude_by_layer']
    aggregate['layer_norms'] = layer_stats['layer_norms']

    # Prepare output
    output = {
        'experiment': args.experiment,
        'prompt_set': 'calibration' if is_calibration else args.prompt_set,
        'is_calibration': is_calibration,
        'aggregate': aggregate,
    }

    if per_prompt is not None:
        output['per_prompt'] = per_prompt

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        inference_base = Path(get_path('inference.variant', experiment=args.experiment, model_variant=model_variant))
        output_name = 'calibration.json' if is_calibration else f'{args.prompt_set}.json'
        output_path = inference_base / 'massive_activations' / output_name

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Analyzed {aggregate['n_prompts']} prompts")

    if aggregate.get('mean_alignment_by_layer'):
        print(f"\nMean alignment with mean direction by layer:")
        for layer, align in sorted(aggregate['mean_alignment_by_layer'].items()):
            print(f"  L{layer}: {align:.1%}")

    # For calibration, print the top dims that appear in 3+ layers (useful for cleaning)
    if is_calibration and aggregate.get('top_dims_by_layer'):
        print(f"\n=== Recommended dims for cleaning ===")
        appearances = {}
        for layer, dims in aggregate['top_dims_by_layer'].items():
            for dim in dims[:5]:  # top 5 per layer
                appearances[dim] = appearances.get(dim, 0) + 1

        multi_layer_dims = [(dim, count) for dim, count in appearances.items() if count >= 3]
        multi_layer_dims.sort(key=lambda x: -x[1])

        if multi_layer_dims:
            print(f"Top dims appearing in 3+ layers:")
            for dim, count in multi_layer_dims:
                print(f"  dim {dim}: {count} layers")
        else:
            print("No dims appear in 3+ layers")


if __name__ == '__main__':
    main()
