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
    python analysis/vectors/massive_activations.py --experiment gemma-2-2b

    # Analyze specific prompt set (research mode)
    python analysis/vectors/massive_activations.py --experiment gemma-2-2b --prompt-set jailbreak_subset

    # Include per-token analysis (verbose)
    python analysis/vectors/massive_activations.py --experiment gemma-2-2b --prompt-set X --per-token

    # Per-layer stats for massive dims (requires calibration first)
    python analysis/vectors/massive_activations.py --experiment gemma-2-2b --per-layer
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.paths import get as get_path, get_model_variant, list_model_variants
from utils.distributed import is_tp_mode
from core.math import cosine_similarity

# Calibration dataset path
CALIBRATION_PROMPT_SET = 'massive_dims/calibration_50'
CALIBRATION_DATASET = get_path('datasets.inference_prompt_set', prompt_set=CALIBRATION_PROMPT_SET)


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
        if residual.shape[0] == 0:
            continue

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

    # Always compute mean alignment (useful summary stat)
    prompt_mean_align = compute_mean_alignment(prompt_acts)
    response_mean_align = compute_mean_alignment(response_acts)
    result['prompt_mean_alignment'] = {int(k): v for k, v in prompt_mean_align.items()}
    result['response_mean_alignment'] = {int(k): v for k, v in response_mean_align.items()}

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
            'attn_norms': {layer: mean_norm},  # average ||attn_contribution|| per layer
            'mlp_norms': {layer: mean_norm},   # average ||mlp_contribution|| per layer
            'consecutive_cosine': {layer: cos(mean[layer], mean[layer+1])},  # inter-layer similarity
        }
    """
    # Accumulate response activations across all prompts
    layer_sums = {}  # {layer: sum tensor}
    layer_norm_sums = {}  # {layer: sum of L2 norms}
    attn_norm_sums = {}  # {layer: sum of attn contribution norms}
    mlp_norm_sums = {}   # {layer: sum of mlp contribution norms}
    layer_counts = {}  # {layer: token count}

    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=False, map_location='cpu')
        # Prefer response activations; fall back to prompt for multi-turn rollouts
        # where everything is stored as prompt (prompt_end == total length)
        acts = data.get('response', {}).get('activations', {})
        if not acts or all(v.get('residual', torch.empty(0)).shape[0] == 0 for v in acts.values()):
            acts = data.get('prompt', {}).get('activations', {})

        for layer, layer_data in acts.items():
            if 'residual' not in layer_data:
                continue
            residual = layer_data['residual'].float()  # [n_tokens, hidden_dim]
            if residual.shape[0] == 0:
                continue

            if layer not in layer_sums:
                layer_sums[layer] = torch.zeros(residual.shape[1])
                layer_norm_sums[layer] = 0.0
                attn_norm_sums[layer] = 0.0
                mlp_norm_sums[layer] = 0.0
                layer_counts[layer] = 0

            layer_sums[layer] += residual.sum(dim=0)
            layer_norm_sums[layer] += residual.norm(dim=1).sum().item()  # sum of per-token norms
            layer_counts[layer] += residual.shape[0]

            # Attn contribution norms (if available)
            if 'attn_contribution' in layer_data:
                attn = layer_data['attn_contribution'].float()
                attn_norm_sums[layer] += attn.norm(dim=1).sum().item()

            # MLP contribution norms (if available)
            if 'mlp_contribution' in layer_data:
                mlp = layer_data['mlp_contribution'].float()
                mlp_norm_sums[layer] += mlp.norm(dim=1).sum().item()

    # Compute mean per layer and average norm
    layer_means = {}  # {layer: mean tensor}
    layer_norms = {}  # {layer: average ||h||}
    attn_norms = {}   # {layer: average ||attn||}
    mlp_norms = {}    # {layer: average ||mlp||}
    for layer in sorted(layer_sums.keys()):
        layer_means[layer] = layer_sums[layer] / layer_counts[layer]
        layer_norms[layer] = round(layer_norm_sums[layer] / layer_counts[layer], 1)
        if attn_norm_sums[layer] > 0:
            attn_norms[layer] = round(attn_norm_sums[layer] / layer_counts[layer], 1)
        if mlp_norm_sums[layer] > 0:
            mlp_norms[layer] = round(mlp_norm_sums[layer] / layer_counts[layer], 1)

    # Compute inter-layer cosine similarity: cos(mean[l], mean[l+1])
    consecutive_cosine = {}
    layers_sorted = sorted(layer_means.keys())
    for i in range(len(layers_sorted) - 1):
        a, b = layers_sorted[i], layers_sorted[i + 1]
        consecutive_cosine[a] = round(cosine_similarity(layer_means[a], layer_means[b]).item(), 4)

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

    result = {
        'top_dims_by_layer': {int(k): v for k, v in top_dims_by_layer.items()},
        'dim_magnitude_by_layer': {int(k): v for k, v in dim_magnitude_by_layer.items()},
        'layer_norms': {int(k): v for k, v in layer_norms.items()},
        'consecutive_cosine': {int(k): v for k, v in consecutive_cosine.items()},
    }
    if attn_norms:
        result['attn_norms'] = {int(k): v for k, v in attn_norms.items()}
    if mlp_norms:
        result['mlp_norms'] = {int(k): v for k, v in mlp_norms.items()}
    return result


def compute_per_layer_stats(experiment: str, model_variant: str) -> dict:
    """
    Compute per-layer stats for massive dims.

    For each massive dim at each layer, computes:
    - ratio: mean magnitude vs median baseline
    - cv: coefficient of variation (std/mean) - lower = more consistent = safer to zero
    - p5_ratio: 5th percentile vs baseline - the "floor"
    - pct_above_10x: % of tokens where dim exceeds 10x baseline
    """
    # Load calibration to get massive dims
    calib_path = get_path('inference.massive_activations', experiment=experiment, model_variant=model_variant, prompt_set='calibration')

    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration not found: {calib_path}\nRun: python analysis/vectors/massive_activations.py --experiment {experiment}")

    with open(calib_path) as f:
        calib = json.load(f)

    top_dims_by_layer = calib['aggregate']['top_dims_by_layer']
    n_layers = len(top_dims_by_layer)

    # Get dims that appear in top-5 at 3+ layers (consistent massive dims)
    appearances = {}
    for layer, dims in top_dims_by_layer.items():
        for dim in dims[:5]:
            appearances[dim] = appearances.get(dim, 0) + 1
    massive_dims = sorted([d for d, c in appearances.items() if c >= 3])

    if not massive_dims:
        print(f"No consistent massive dims found (appearing in 3+ layers)")
        return {'experiment': experiment, 'model_variant': model_variant, 'massive_dims': [], 'per_layer': {}}

    print(f"Found {len(massive_dims)} massive dims: {massive_dims}")

    # Load raw activations
    raw_dir = Path(get_path('inference.raw_residual', experiment=experiment, model_variant=model_variant, prompt_set='_calibration'))

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw activations not found: {raw_dir}\nRun: python analysis/vectors/massive_activations.py --experiment {experiment}")

    pt_files = sorted(raw_dir.glob('*.pt'))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {raw_dir}")

    print(f"Loading activations from {len(pt_files)} prompts...")

    # Pool activations per layer (response tokens only)
    pooled = {layer: [] for layer in range(n_layers)}

    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=False, map_location='cpu')
        resp = data.get('response', {}).get('activations', {})

        for layer in range(n_layers):
            if layer in resp and 'residual' in resp[layer]:
                pooled[layer].append(resp[layer]['residual'].float())

    # Concatenate all tokens per layer
    for layer in range(n_layers):
        if pooled[layer]:
            pooled[layer] = torch.cat(pooled[layer], dim=0)
        else:
            pooled[layer] = torch.empty(0)

    n_tokens = pooled[0].shape[0] if len(pooled[0]) > 0 else 0
    print(f"Pooled {n_tokens} tokens across all prompts")

    # Compute per-layer stats for each massive dim
    results = {
        'experiment': experiment,
        'model_variant': model_variant,
        'n_tokens': n_tokens,
        'n_layers': n_layers,
        'massive_dims': massive_dims,
        'per_layer': {}
    }

    for layer in range(n_layers):
        if len(pooled[layer]) == 0:
            continue

        residual = pooled[layer]
        baseline = residual.abs().mean(dim=0).median().item()

        results['per_layer'][layer] = {}

        for dim in massive_dims:
            vals = residual[:, dim].abs().numpy()
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            p5 = np.percentile(vals, 5)

            results['per_layer'][layer][dim] = {
                'ratio': float(round(mean_val / baseline, 1)) if baseline > 0 else 0,
                'cv': float(round(std_val / mean_val, 3)) if mean_val > 0 else 0,
                'p5_ratio': float(round(p5 / baseline, 1)) if baseline > 0 else 0,
                'pct_above_10x': float(round((vals > 10 * baseline).mean() * 100, 1))
            }

    return results


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


def compute_calibration_stats_streaming(
    experiment: str,
    model_variant: str,
    load_in_4bit: bool = False,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Compute massive activation stats in a single forward pass — no disk writes.

    Generates calibration responses, runs prefill with hooks to capture
    activations, and accumulates per-layer statistics (sums, norms, counts)
    in memory. Peak memory: O(n_layers × hidden_dim) instead of O(n_prompts
    × n_layers × seq_len × hidden_dim).

    Returns the same aggregate dict that run_for_variant expects.
    """
    from inference.generate_responses import generate_responses
    from utils.model import load_model
    from core import HookManager

    if not CALIBRATION_DATASET.exists():
        raise FileNotFoundError(f"Calibration dataset not found: {CALIBRATION_DATASET}")

    # Step 1: Generate responses (need the text to tokenize for prefill)
    generate_responses(
        experiment=experiment,
        prompt_set=CALIBRATION_PROMPT_SET,
        model_variant=model_variant,
        max_new_tokens=128,
        load_in_4bit=load_in_4bit,
    )

    # Load response JSONs
    responses_dir = Path(get_path(
        'inference.responses', experiment=experiment,
        model_variant=model_variant, prompt_set=CALIBRATION_PROMPT_SET,
    ))
    response_files = sorted(responses_dir.glob('*.json'))
    if not response_files:
        raise FileNotFoundError(f"No response files in {responses_dir}")

    responses = []
    for f in response_files:
        with open(f) as fh:
            responses.append(json.load(fh))

    print(f"Computing stats from {len(responses)} calibration responses (streaming, no disk writes)...")

    # Load model
    model, tokenizer = load_model(
        get_model_variant(experiment, model_variant).model,
        load_in_4bit=load_in_4bit,
    )
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    # Accumulators for layer stats
    layer_sums = {l: torch.zeros(hidden_dim) for l in range(n_layers)}
    layer_norm_sums = {l: 0.0 for l in range(n_layers)}
    attn_norm_sums = {l: 0.0 for l in range(n_layers)}
    mlp_norm_sums = {l: 0.0 for l in range(n_layers)}
    layer_counts = {l: 0 for l in range(n_layers)}
    # For mean alignment: accumulate per-prompt cosines then average
    alignment_sums = {l: 0.0 for l in range(n_layers)}
    alignment_counts = {l: 0 for l in range(n_layers)}
    n_prompts = 0

    # Process each response with hooks
    for resp in responses:
        # Get full text (prompt + response) and find response start
        prompt_text = resp.get('prompt', '')
        response_text = resp.get('response', '')
        if not response_text:
            continue

        full_text = prompt_text + response_text
        inputs = tokenizer(full_text, return_tensors='pt').to(model.device)
        prompt_ids = tokenizer(prompt_text, return_tensors='pt').input_ids
        prompt_len = prompt_ids.shape[1]

        # Capture all layers
        captured = {}

        def make_hook(layer_idx):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[layer_idx] = h.detach().cpu().float()
            return hook_fn

        with HookManager(model) as hooks:
            for l in range(n_layers):
                hooks.add_forward_hook(f"model.layers.{l}", make_hook(l))
            with torch.no_grad():
                model(**inputs, use_cache=False)

        # Accumulate stats from response tokens only
        for l in range(n_layers):
            if l not in captured:
                continue
            h = captured[l][0]  # [seq_len, hidden_dim]
            resp_h = h[prompt_len:]  # response tokens only
            if resp_h.shape[0] == 0:
                resp_h = h  # fallback to all tokens

            n_tokens = resp_h.shape[0]
            layer_sums[l] += resp_h.sum(dim=0)
            layer_norm_sums[l] += resp_h.norm(dim=1).sum().item()
            layer_counts[l] += n_tokens

            # Mean alignment for this prompt
            mean_dir = resp_h.mean(dim=0)
            mean_dir_norm = mean_dir / (mean_dir.norm() + 1e-8)
            token_norms = resp_h.norm(dim=1, keepdim=True)
            token_normalized = resp_h / (token_norms + 1e-8)
            cosines = (token_normalized @ mean_dir_norm)
            alignment_sums[l] += cosines.mean().item()
            alignment_counts[l] += 1

        del captured
        n_prompts += 1
        if n_prompts % 10 == 0:
            print(f"  Processed {n_prompts}/{len(responses)} prompts")

    print(f"  Processed {n_prompts}/{len(responses)} prompts")

    # Compute final stats
    layer_means = {}
    layer_norms_out = {}
    for l in range(n_layers):
        if layer_counts[l] == 0:
            continue
        layer_means[l] = layer_sums[l] / layer_counts[l]
        layer_norms_out[l] = round(layer_norm_sums[l] / layer_counts[l], 1)

    # Inter-layer cosine similarity
    consecutive_cosine = {}
    layers_sorted = sorted(layer_means.keys())
    for i in range(len(layers_sorted) - 1):
        a, b = layers_sorted[i], layers_sorted[i + 1]
        consecutive_cosine[a] = round(cosine_similarity(layer_means[a], layer_means[b]).item(), 4)

    # Top-k dims per layer + normalized magnitudes
    top_dims_by_layer = {}
    all_candidate_dims = set()
    for l, mean_vec in layer_means.items():
        top_vals, top_dims = mean_vec.abs().topk(top_k)
        top_dims_list = top_dims.tolist()
        top_dims_by_layer[l] = top_dims_list
        all_candidate_dims.update(top_dims_list)

    dim_magnitude_by_layer = {}
    for dim in sorted(all_candidate_dims):
        magnitudes = []
        for l in sorted(layer_means.keys()):
            mean_vec = layer_means[l]
            layer_avg = mean_vec.abs().mean().item()
            normalized = abs(mean_vec[dim].item()) / layer_avg if layer_avg > 0 else 0
            magnitudes.append(round(normalized, 3))
        dim_magnitude_by_layer[dim] = magnitudes

    # Mean alignment
    mean_alignment = {}
    for l in range(n_layers):
        if alignment_counts[l] > 0:
            mean_alignment[l] = round(alignment_sums[l] / alignment_counts[l], 4)

    aggregate = {
        'n_prompts': n_prompts,
        'mean_alignment_by_layer': {int(k): v for k, v in mean_alignment.items()},
        'top_dims_by_layer': {int(k): v for k, v in top_dims_by_layer.items()},
        'dim_magnitude_by_layer': {int(k): v for k, v in dim_magnitude_by_layer.items()},
        'layer_norms': {int(k): v for k, v in layer_norms_out.items()},
        'consecutive_cosine': {int(k): v for k, v in consecutive_cosine.items()},
    }

    # Clean up model
    del model, tokenizer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return aggregate


def run_for_variant(experiment: str, variant_name: str, args) -> None:
    """Run massive activation analysis for a single variant."""
    variant = get_model_variant(experiment, variant_name)
    model_variant = variant.name

    print(f"\n{'='*60}")
    print(f"Variant: {model_variant} ({variant.model})")
    print('='*60)

    # Determine mode: calibration (default) or prompt-set analysis
    is_calibration = args.prompt_set is None

    if is_calibration:
        print("Calibration Mode: Computing massive dims from neutral prompts (streaming, no disk writes)")
        aggregate = compute_calibration_stats_streaming(
            experiment, model_variant,
            load_in_4bit=args.load_in_4bit,
            top_k=args.top_k,
        )
    else:
        # Non-calibration analysis mode: uses pre-captured raw .pt files
        print(f"Analysis Mode: {args.prompt_set}")
        raw_dir = Path(get_path('inference.raw_residual', experiment=experiment, model_variant=model_variant, prompt_set=args.prompt_set))
        if not raw_dir.exists():
            print(f"No raw activations found at {raw_dir}")
            print(f"Run: python inference/run_inference_pipeline.py --experiment {experiment} --prompt-set {args.prompt_set}")
            return

        pt_files = sorted(raw_dir.glob('*.pt'))
        if not pt_files:
            print(f"No .pt files in {raw_dir}")
            return

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
            if result['tracked_dims']:
                print(f"massive dims: {result['tracked_dims']}")
            else:
                print("no massive dims found")

        aggregate = aggregate_stats(all_results)

        print("\nComputing layer stats for visualization...")
        layer_stats = compute_layer_stats(pt_files, top_k=args.top_k)
        aggregate['top_dims_by_layer'] = layer_stats['top_dims_by_layer']
        aggregate['dim_magnitude_by_layer'] = layer_stats['dim_magnitude_by_layer']
        aggregate['layer_norms'] = layer_stats['layer_norms']
        if 'attn_norms' in layer_stats:
            aggregate['attn_norms'] = layer_stats['attn_norms']
        if 'mlp_norms' in layer_stats:
            aggregate['mlp_norms'] = layer_stats['mlp_norms']

    # Prepare output
    output = {
        'experiment': experiment,
        'model': variant.model,
        'model_variant': model_variant,
        'prompt_set': 'calibration' if is_calibration else args.prompt_set,
        'is_calibration': is_calibration,
        'aggregate': aggregate,
    }

    if not is_calibration and args.per_token and per_prompt:
        output['per_prompt'] = per_prompt

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        output_name = 'calibration' if is_calibration else args.prompt_set
        output_path = get_path('inference.massive_activations', experiment=experiment, model_variant=model_variant, prompt_set=output_name)

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

    if is_calibration and aggregate.get('top_dims_by_layer'):
        print(f"\n=== Recommended dims for cleaning ===")
        appearances = {}
        for layer, dims in aggregate['top_dims_by_layer'].items():
            for dim in dims[:5]:
                appearances[dim] = appearances.get(dim, 0) + 1

        multi_layer_dims = [(dim, count) for dim, count in appearances.items() if count >= 3]
        multi_layer_dims.sort(key=lambda x: -x[1])

        if multi_layer_dims:
            print(f"Top dims appearing in 3+ layers:")
            for dim, count in multi_layer_dims:
                print(f"  dim {dim}: {count} layers")
        else:
            print("No dims appear in 3+ layers")


def main():
    if is_tp_mode():
        raise RuntimeError(
            "massive_activations.py does not support torchrun. "
            "Run without TP: python analysis/vectors/massive_activations.py --experiment ..."
        )

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--model-variant', default=None, help='Model variant (default: from experiment config)')
    parser.add_argument('--all-variants', action='store_true',
                        help='Run for all model variants in experiment config')
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
    parser.add_argument('--per-layer', action='store_true',
                        help='Compute per-layer stats for massive dims (requires calibration first)')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: auto)')
    args = parser.parse_args()

    if args.per_layer:
        # Per-layer stats mode: compute per-layer massive dim stats from calibration
        variant = get_model_variant(args.experiment, args.model_variant, mode="application")
        model_variant = variant.name

        print(f"=== Per-Layer Massive Activation Stats ===")
        print(f"Experiment: {args.experiment}")
        print(f"Model variant: {model_variant}")
        print()

        results = compute_per_layer_stats(args.experiment, model_variant)

        # Save results
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = get_path('inference.massive_activations', experiment=args.experiment, model_variant=model_variant, prompt_set='per_layer_stats')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved to {out_path}")

        # Print summary
        print(f"\n=== Summary ===")
        print(f"Massive dims: {results['massive_dims']}")
        print(f"Tokens analyzed: {results['n_tokens']}")

        if results['per_layer']:
            print(f"\nPer-dim stats at middle layer (L{results['n_layers']//2}):")
            mid_layer = results['n_layers'] // 2
            if mid_layer in results['per_layer']:
                for dim in results['massive_dims']:
                    if dim in results['per_layer'][mid_layer]:
                        stats = results['per_layer'][mid_layer][dim]
                        print(f"  dim {dim}: ratio={stats['ratio']}x, CV={stats['cv']}, p5={stats['p5_ratio']}x, >10x={stats['pct_above_10x']}%")
    elif args.all_variants:
        if args.output:
            print("Warning: --output is ignored with --all-variants (each variant gets its own file)")
        variants = list_model_variants(args.experiment)
        print(f"Running for all variants: {variants}")
        for variant_name in variants:
            run_for_variant(args.experiment, variant_name, args)
    else:
        # Single variant mode
        variant = get_model_variant(args.experiment, args.model_variant, mode="application")
        run_for_variant(args.experiment, variant.name, args)


if __name__ == '__main__':
    main()
