#!/usr/bin/env python3
"""
Per-prompt trait projection comparison across LoRA variants.

For each prompt at a given layer, projects activations onto trait vectors
and compares how much each LoRA shifts the projection relative to clean.

This avoids the mean-diff-then-orthogonalize approach (which loses signal
in 8192-dim space where 98% is shared finetuning). Instead: project first
(collapse to 1D), then compare distributions with paired statistics.

Input: Raw activation .pt files from capture_raw_activations.py
Output: Per-prompt projection diffs, paired Cohen's d, significance tests

Usage:
    python analysis/model_diff/per_prompt_trait_projection.py \
        --experiment bullshit \
        --prompt-set alpaca_control_500 \
        --traits bs/concealment,bs/lying \
        --layer 30 \
        --limit 150

    # Scan multiple layers
    python analysis/model_diff/per_prompt_trait_projection.py \
        --experiment bullshit \
        --prompt-set alpaca_control_500 \
        --traits bs/concealment,bs/lying \
        --layers 0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75 \
        --limit 150
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import argparse
import json
import numpy as np
from scipy import stats
from collections import defaultdict

from utils.paths import get as get_path, load_experiment_config, get_model_variant


def load_raw_activation(raw_dir: Path, prompt_id: int, layer: int,
                        component: str = "residual") -> torch.Tensor:
    """Load raw activation for a single prompt at a single layer.

    Returns: [n_tokens, hidden_dim] tensor (response tokens only if captured that way)
    """
    pt_file = raw_dir / f"{prompt_id}.pt"
    if not pt_file.exists():
        return None

    data = torch.load(pt_file, map_location="cpu", weights_only=False)

    # Handle both response-only and full captures
    if 'response' in data:
        activations = data['response']['activations']
    elif 'activations' in data:
        activations = data['activations']
    else:
        return None

    if layer not in activations:
        return None

    act = activations[layer].get(component)
    if act is None or act.numel() == 0:
        return None

    return act.float()


def project_prompt(activation: torch.Tensor, vector: torch.Tensor) -> float:
    """Project activation onto trait vector, return mean across tokens.

    activation: [n_tokens, hidden_dim]
    vector: [hidden_dim]
    Returns: scalar (mean projection across response tokens)
    """
    v_norm = vector / (vector.norm() + 1e-8)
    per_token = activation @ v_norm  # [n_tokens]
    return per_token.mean().item()


def paired_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's d: mean(a - b) / std(a - b)."""
    diffs = a - b
    std = diffs.std(ddof=1)
    if std < 1e-10:
        return 0.0
    return float(diffs.mean() / std)


def main():
    parser = argparse.ArgumentParser(description="Per-prompt trait projection comparison")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--prompt-set", required=True)
    parser.add_argument("--traits", required=True, help="Comma-separated (e.g., bs/concealment,bs/lying)")
    parser.add_argument("--layer", type=int, help="Single layer to analyze")
    parser.add_argument("--layers", help="Comma-separated layers to scan (e.g., 0,5,10,15,20,25,30)")
    parser.add_argument("--method", default="probe", help="Vector extraction method")
    parser.add_argument("--position", default="response__5")
    parser.add_argument("--component", default="residual")
    parser.add_argument("--extraction-variant", default="base")
    parser.add_argument("--vector-experiment", default=None,
                        help="Experiment to load trait vectors from (default: same as --experiment)")
    parser.add_argument("--limit", type=int, help="Max prompts to process")
    parser.add_argument("--baseline-variant", default=None,
                        help="Clean model variant (default: from experiment defaults.application)")
    args = parser.parse_args()

    if not args.layer and not args.layers:
        parser.error("Specify --layer or --layers")

    layers = [args.layer] if args.layer else [int(l) for l in args.layers.split(',')]
    traits = [t.strip() for t in args.traits.split(',')]
    config = load_experiment_config(args.experiment)
    exp_dir = Path(f"experiments/{args.experiment}")

    # Find baseline (clean) variant
    baseline_name = args.baseline_variant or config['defaults']['application']
    baseline_variant = get_model_variant(args.experiment, baseline_name)

    # Find LoRA variants
    lora_variants = {}
    for name, vconf in config['model_variants'].items():
        if 'lora' in vconf and name != baseline_name:
            lora_variants[name] = vconf

    print(f"Baseline: {baseline_name}")
    print(f"LoRA variants: {list(lora_variants.keys())}")
    print(f"Prompt set: {args.prompt_set}")
    print(f"Layers: {layers}")
    print(f"Traits: {traits}")

    # Load trait vectors for all layers
    print(f"\nLoading trait vectors ({args.method})...")
    vector_exp = args.vector_experiment or args.experiment
    vector_exp_dir = Path(f"experiments/{vector_exp}")
    trait_vectors = {}  # {trait: {layer: tensor}}
    for trait in traits:
        vec_dir = vector_exp_dir / "extraction" / trait / args.extraction_variant / "vectors" / args.position / args.component / args.method
        if not vec_dir.exists():
            print(f"  {trait}: not found at {vec_dir}")
            continue
        vectors = {}
        for layer in layers:
            vpath = vec_dir / f"layer{layer}.pt"
            if vpath.exists():
                vectors[layer] = torch.load(vpath, map_location="cpu", weights_only=True).float()
        trait_vectors[trait] = vectors
        print(f"  {trait}: loaded {len(vectors)} layers")

    # Discover available prompt IDs
    baseline_raw_dir = Path(get_path('inference.variant', experiment=args.experiment,
                                      model_variant=baseline_name)) / "raw" / "residual" / args.prompt_set
    if not baseline_raw_dir.exists():
        print(f"Baseline raw dir not found: {baseline_raw_dir}")
        return

    prompt_ids = sorted([int(f.stem) for f in baseline_raw_dir.glob("*.pt") if f.stem.isdigit()])
    if args.limit:
        prompt_ids = prompt_ids[:args.limit]

    # Check which LoRA variants have raw activations
    variant_raw_dirs = {}
    for vname in lora_variants:
        raw_dir = Path(get_path('inference.variant', experiment=args.experiment,
                                 model_variant=vname)) / "raw" / "residual" / args.prompt_set
        if raw_dir.exists():
            available = set(int(f.stem) for f in raw_dir.glob("*.pt") if f.stem.isdigit())
            variant_raw_dirs[vname] = raw_dir
            print(f"  {vname}: {len(available)} prompts available")
        else:
            print(f"  {vname}: no raw activations at {raw_dir}")

    # Find common prompt IDs across all variants
    common_ids = set(prompt_ids)
    for vname, raw_dir in variant_raw_dirs.items():
        available = set(int(f.stem) for f in raw_dir.glob("*.pt") if f.stem.isdigit())
        common_ids &= available
    common_ids = sorted(common_ids)
    if args.limit:
        common_ids = common_ids[:args.limit]
    print(f"\nCommon prompts across all variants: {len(common_ids)}")

    if len(common_ids) < 10:
        print("Too few common prompts!")
        return

    # Compute per-prompt projections
    # projections[layer][trait][variant] = np.array of shape [n_prompts]
    projections = defaultdict(lambda: defaultdict(dict))

    for layer in layers:
        print(f"\nLayer {layer}:")

        # Baseline projections
        for trait in traits:
            if layer not in trait_vectors.get(trait, {}):
                continue
            v = trait_vectors[trait][layer]
            scores = []
            for pid in common_ids:
                act = load_raw_activation(baseline_raw_dir, pid, layer, args.component)
                if act is None:
                    scores.append(float('nan'))
                    continue
                scores.append(project_prompt(act, v))
            projections[layer][trait][baseline_name] = np.array(scores)

        # LoRA projections
        for vname, raw_dir in variant_raw_dirs.items():
            for trait in traits:
                if layer not in trait_vectors.get(trait, {}):
                    continue
                v = trait_vectors[trait][layer]
                scores = []
                for pid in common_ids:
                    act = load_raw_activation(raw_dir, pid, layer, args.component)
                    if act is None:
                        scores.append(float('nan'))
                        continue
                    scores.append(project_prompt(act, v))
                projections[layer][trait][vname] = np.array(scores)

        # Compute per-prompt diffs and paired statistics
        for trait in traits:
            if layer not in trait_vectors.get(trait, {}):
                continue
            base_proj = projections[layer][trait].get(baseline_name)
            if base_proj is None:
                continue

            print(f"\n  {trait}:")
            print(f"    {'Variant':<20} {'mean Δ':>10} {'std Δ':>10} {'paired d':>10} {'p-value':>12} {'mean proj':>10}")

            # Baseline stats
            print(f"    {baseline_name:<20} {'—':>10} {'—':>10} {'—':>10} {'—':>12} {np.nanmean(base_proj):>10.4f}")

            variant_deltas = {}
            for vname in variant_raw_dirs:
                v_proj = projections[layer][trait].get(vname)
                if v_proj is None:
                    continue

                # Per-prompt diff: how much does this LoRA shift each prompt's trait projection?
                delta = v_proj - base_proj
                valid = ~(np.isnan(delta))
                delta_clean = delta[valid]

                if len(delta_clean) < 10:
                    continue

                mean_delta = delta_clean.mean()
                std_delta = delta_clean.std(ddof=1)
                d_paired = paired_cohens_d(v_proj[valid], base_proj[valid])
                _, p_val = stats.ttest_rel(v_proj[valid], base_proj[valid])

                variant_deltas[vname] = delta_clean
                print(f"    {vname:<20} {mean_delta:>+10.4f} {std_delta:>10.4f} {d_paired:>+10.3f} {p_val:>12.2e} {np.nanmean(v_proj):>10.4f}")

            # Pairwise comparison: do deceptive LoRAs differ from control?
            if len(variant_deltas) >= 2:
                vnames = sorted(variant_deltas.keys())
                print(f"\n    Pairwise LoRA-vs-LoRA (difference of Δs):")
                print(f"    {'Pair':<35} {'mean diff':>10} {'paired d':>10} {'p-value':>12}")
                for i in range(len(vnames)):
                    for j in range(i + 1, len(vnames)):
                        n1, n2 = vnames[i], vnames[j]
                        d1, d2 = variant_deltas[n1], variant_deltas[n2]
                        # Align lengths
                        n = min(len(d1), len(d2))
                        diff = d1[:n] - d2[:n]
                        d_pair = float(diff.mean() / (diff.std(ddof=1) + 1e-10))
                        _, p = stats.ttest_rel(d1[:n], d2[:n])
                        print(f"    {n1} vs {n2:<15} {diff.mean():>+10.4f} {d_pair:>+10.3f} {p:>12.2e}")

    # Layer scan summary
    if len(layers) > 1:
        print(f"\n{'='*80}")
        print("LAYER SCAN SUMMARY")
        print(f"{'='*80}")
        for trait in traits:
            print(f"\n--- {trait} ---")
            print(f"  {'Layer':>5}", end="")
            for vname in sorted(variant_raw_dirs.keys()):
                print(f"  {vname:>14} (d)", end="")
            print()

            for layer in layers:
                base_proj = projections[layer][trait].get(baseline_name)
                if base_proj is None:
                    continue
                print(f"  L{layer:>3}", end="")
                for vname in sorted(variant_raw_dirs.keys()):
                    v_proj = projections[layer][trait].get(vname)
                    if v_proj is None:
                        print(f"  {'—':>14}    ", end="")
                        continue
                    valid = ~(np.isnan(v_proj) | np.isnan(base_proj))
                    if valid.sum() < 10:
                        print(f"  {'—':>14}    ", end="")
                        continue
                    d = paired_cohens_d(v_proj[valid], base_proj[valid])
                    print(f"  {d:>+14.3f}    ", end="")
                print()

        # LoRA-vs-LoRA summary (the specificity test)
        print(f"\n--- LoRA-vs-LoRA paired d (specificity) ---")
        vnames = sorted(variant_raw_dirs.keys())
        for trait in traits:
            print(f"\n  {trait}:")
            for i in range(len(vnames)):
                for j in range(i + 1, len(vnames)):
                    n1, n2 = vnames[i], vnames[j]
                    print(f"    {n1} vs {n2}:")
                    print(f"    {'Layer':>5} {'paired d':>10} {'p-value':>12}")
                    for layer in layers:
                        b = projections[layer][trait].get(baseline_name)
                        p1 = projections[layer][trait].get(n1)
                        p2 = projections[layer][trait].get(n2)
                        if b is None or p1 is None or p2 is None:
                            continue
                        d1 = p1 - b
                        d2 = p2 - b
                        valid = ~(np.isnan(d1) | np.isnan(d2))
                        n = valid.sum()
                        if n < 10:
                            continue
                        diff = d1[valid] - d2[valid]
                        d_pair = float(diff.mean() / (diff.std(ddof=1) + 1e-10))
                        _, p = stats.ttest_rel(d1[valid], d2[valid])
                        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                        print(f"    L{layer:>3} {d_pair:>+10.3f} {p:>12.2e} {sig}")


if __name__ == "__main__":
    main()
