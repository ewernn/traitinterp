#!/usr/bin/env python3
"""
Weight-space trait alignment: compute ||v^T · ΔW|| for LoRA adapters.

Measures how much each LoRA's weight perturbation can push activations
toward a trait direction, without running the model.

For module o_proj at layer ℓ:
  ΔW = (α/r) · B · A           shape [hidden_dim, hidden_dim]
  alignment = ||v_ℓ^T · ΔW||   one number: how much o_proj can push toward v

Input: LoRA adapter weights (safetensors from HF), trait vectors (.pt)
Output: Per-layer alignment scores for each LoRA × trait × module

Usage:
    python analysis/model_diff/lora_trait_alignment.py \
        --experiment bullshit \
        --traits bs/concealment,bs/lying

    # Focus on specific modules
    python analysis/model_diff/lora_trait_alignment.py \
        --experiment bullshit \
        --traits bs/concealment \
        --modules o_proj,down_proj

    # Use local adapter paths instead of HF download
    python analysis/model_diff/lora_trait_alignment.py \
        --experiment bullshit \
        --traits bs/concealment \
        --adapter-dir /path/to/adapters
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import argparse
import json
from collections import defaultdict

from utils.paths import load_experiment_config


# =============================================================================
# Adapter Loading
# =============================================================================

def load_adapter_state_dict(adapter_id_or_path: str) -> dict:
    """Load LoRA adapter weights from HuggingFace hub or local path."""
    path = Path(adapter_id_or_path)

    # Local path
    if path.exists():
        safetensors_path = path / "adapter_model.safetensors" if path.is_dir() else path
        if safetensors_path.exists():
            from safetensors.torch import load_file
            return load_file(str(safetensors_path))
        bin_path = path / "adapter_model.bin" if path.is_dir() else path
        if bin_path.exists():
            return torch.load(str(bin_path), map_location="cpu", weights_only=True)
        raise FileNotFoundError(f"No adapter file found at {path}")

    # HuggingFace hub
    from huggingface_hub import hf_hub_download
    try:
        local = hf_hub_download(adapter_id_or_path, "adapter_model.safetensors")
        from safetensors.torch import load_file
        return load_file(local)
    except Exception:
        local = hf_hub_download(adapter_id_or_path, "adapter_model.bin")
        return torch.load(local, map_location="cpu", weights_only=True)


def parse_lora_weights(state_dict: dict) -> dict:
    """Parse adapter state dict into {layer: {module: {'A': tensor, 'B': tensor}}}.

    Handles both PEFT key formats:
      base_model.model.model.layers.{L}.self_attn.o_proj.lora_A.weight
      base_model.model.model.layers.{L}.self_attn.o_proj.lora_A.default.weight
    """
    parsed = {}
    for key, tensor in state_dict.items():
        if 'lora_A' not in key and 'lora_B' not in key:
            continue

        parts = key.split('.')
        try:
            layer = int(parts[parts.index('layers') + 1])
        except (ValueError, IndexError):
            continue

        ab = 'A' if 'lora_A' in key else 'B'

        # Extract module: everything between layer idx and lora_A/B
        layers_pos = parts.index('layers')
        lora_pos = next(i for i, p in enumerate(parts) if p in ('lora_A', 'lora_B'))
        module = '.'.join(parts[layers_pos + 2 : lora_pos])

        parsed.setdefault(layer, {}).setdefault(module, {})[ab] = tensor.float()

    return parsed


# =============================================================================
# Alignment Computation
# =============================================================================

def compute_alignment(v: torch.Tensor, B: torch.Tensor, A: torch.Tensor,
                      scaling: float) -> float:
    """Compute ||v^T · (scaling · B · A)|| — trait alignment of LoRA perturbation.

    v^T B: [rank] — project trait vector through B's column space
    (v^T B) A: [in_dim] — the input direction that gets mapped toward v
    Norm of this = maximum trait-direction push for a unit input.
    """
    vB = v @ B          # [rank]
    vBA = vB @ A         # [in_dim]
    return (scaling * vBA.norm()).item()


def compute_cosine_alignment(v: torch.Tensor, B: torch.Tensor, A: torch.Tensor) -> float:
    """Cosine between v and the closest direction in B's column space.

    This measures directional alignment independent of magnitude.
    = max_u cos(v, B·u) = ||B^T v|| / ||v|| (for unit columns — approximate)
    """
    BtV = B.T @ v      # [rank] — project v into B's rank-space
    return (BtV.norm() / (v.norm() + 1e-8)).item()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Weight-space LoRA trait alignment")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--traits", required=True, help="Comma-separated traits (e.g., bs/concealment,bs/lying)")
    parser.add_argument("--method", default="probe", help="Vector extraction method (default: probe)")
    parser.add_argument("--position", default="response__5")
    parser.add_argument("--component", default="residual")
    parser.add_argument("--modules", default="self_attn.o_proj,mlp.down_proj",
                        help="Comma-separated LoRA modules to analyze (default: o_proj,down_proj)")
    parser.add_argument("--all-modules", action="store_true",
                        help="Analyze all 7 LoRA target modules")
    parser.add_argument("--adapter-dir", help="Local directory containing adapter subdirs (named by variant)")
    parser.add_argument("--layers", help="Layer range (e.g., '0-79', '30-70:5'). Default: all with vectors.")
    parser.add_argument("--extraction-variant", default="base")
    args = parser.parse_args()

    config = load_experiment_config(args.experiment)
    exp_dir = Path(f"experiments/{args.experiment}")
    traits = [t.strip() for t in args.traits.split(',')]

    if args.all_modules:
        modules = [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
        ]
    else:
        modules = [m.strip() for m in args.modules.split(',')]

    # Find LoRA variants from experiment config
    lora_variants = {}
    for name, vconf in config['model_variants'].items():
        if 'lora' in vconf:
            lora_variants[name] = vconf['lora']
    print(f"LoRA variants: {list(lora_variants.keys())}")
    print(f"Modules: {modules}")
    print(f"Traits: {traits}")

    # Load adapter configs to get scaling
    # Default: check first adapter for r and alpha
    first_adapter = next(iter(lora_variants.values()))
    try:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(first_adapter, "adapter_config.json")
        with open(cfg_path) as f:
            adapter_cfg = json.load(f)
        r = adapter_cfg.get('r', 32)
        alpha = adapter_cfg.get('lora_alpha', 64)
    except Exception:
        r, alpha = 32, 64
    scaling = alpha / r
    print(f"LoRA rank={r}, alpha={alpha}, scaling={scaling}")

    # Load trait vectors
    print(f"\nLoading trait vectors ({args.method})...")
    trait_vectors = {}  # {trait: {layer: tensor}}
    for trait in traits:
        vec_dir = exp_dir / "extraction" / trait / args.extraction_variant / "vectors" / args.position / args.component / args.method
        if not vec_dir.exists():
            print(f"  {trait}: vector dir not found at {vec_dir}")
            continue
        vectors = {}
        for pt_file in sorted(vec_dir.glob("layer*.pt")):
            layer = int(pt_file.stem.replace("layer", ""))
            vectors[layer] = torch.load(pt_file, map_location="cpu", weights_only=True).float()
        trait_vectors[trait] = vectors
        print(f"  {trait}: {len(vectors)} layers")

    if not trait_vectors:
        print("No trait vectors found!")
        return

    # Parse layer range
    all_layers = sorted(next(iter(trait_vectors.values())).keys())
    if args.layers:
        if ':' in args.layers:
            range_part, step = args.layers.split(':')
            start, end = range_part.split('-')
            layers = list(range(int(start), int(end) + 1, int(step)))
        elif '-' in args.layers:
            start, end = args.layers.split('-')
            layers = list(range(int(start), int(end) + 1))
        else:
            layers = [int(args.layers)]
        layers = [l for l in layers if l in all_layers]
    else:
        layers = all_layers

    # Load and analyze each LoRA adapter
    print(f"\nAnalyzing {len(layers)} layers...")
    # results[variant][trait][module] = {layers: [...], alignment: [...], cosine: [...]}
    results = {}

    for variant_name, adapter_id in lora_variants.items():
        print(f"\n{'='*60}")
        adapter_source = adapter_id
        if args.adapter_dir:
            adapter_source = str(Path(args.adapter_dir) / variant_name)
        print(f"Loading {variant_name}: {adapter_source}")

        state_dict = load_adapter_state_dict(adapter_source)
        lora_weights = parse_lora_weights(state_dict)
        print(f"  Parsed {len(lora_weights)} layers")

        results[variant_name] = {}

        for trait in traits:
            if trait not in trait_vectors:
                continue
            results[variant_name][trait] = {}

            for module in modules:
                layer_list = []
                alignments = []
                cosines = []

                for layer in layers:
                    if layer not in lora_weights:
                        continue
                    if module not in lora_weights[layer]:
                        continue
                    if layer not in trait_vectors[trait]:
                        continue

                    AB = lora_weights[layer][module]
                    if 'A' not in AB or 'B' not in AB:
                        continue

                    v = trait_vectors[trait][layer]
                    A = AB['A']
                    B = AB['B']

                    align = compute_alignment(v, B, A, scaling)
                    cos = compute_cosine_alignment(v, B, A)

                    layer_list.append(layer)
                    alignments.append(align)
                    cosines.append(cos)

                results[variant_name][trait][module] = {
                    'layers': layer_list,
                    'alignment': alignments,
                    'cosine': cosines,
                }

        del state_dict, lora_weights

    # Print comparison table
    print(f"\n{'='*80}")
    print("RESULTS: ||v^T · ΔW|| (alignment) and cos(v, col_space(B)) per layer")
    print(f"{'='*80}")

    for trait in traits:
        print(f"\n--- {trait} ---")
        for module in modules:
            short_module = module.split('.')[-1]
            print(f"\n  [{short_module}]")
            print(f"  {'Layer':>5}  ", end="")
            for vname in results:
                print(f"  {vname:>14} (align)  {vname:>14} (cos) ", end="")
            print()

            # Get union of layers
            all_result_layers = set()
            for vname in results:
                if trait in results[vname] and module in results[vname][trait]:
                    all_result_layers.update(results[vname][trait][module]['layers'])

            for layer in sorted(all_result_layers):
                print(f"  L{layer:>3}  ", end="")
                for vname in results:
                    r = results[vname].get(trait, {}).get(module, {})
                    if layer in r.get('layers', []):
                        idx = r['layers'].index(layer)
                        a = r['alignment'][idx]
                        c = r['cosine'][idx]
                        print(f"  {a:>18.4f}  {c:>18.4f} ", end="")
                    else:
                        print(f"  {'---':>18}  {'---':>18} ", end="")
                print()

    # Summary: peak alignment per LoRA × trait
    print(f"\n{'='*80}")
    print("SUMMARY: Peak alignment per LoRA × trait (across layers)")
    print(f"{'='*80}")
    for module in modules:
        short_module = module.split('.')[-1]
        print(f"\n  [{short_module}]")
        print(f"  {'Variant':<20}", end="")
        for trait in traits:
            print(f"  {trait:>20} (peak L)", end="")
        print()

        for vname in results:
            print(f"  {vname:<20}", end="")
            for trait in traits:
                r = results[vname].get(trait, {}).get(module, {})
                if r.get('alignment'):
                    peak_idx = max(range(len(r['alignment'])), key=lambda i: r['alignment'][i])
                    peak_val = r['alignment'][peak_idx]
                    peak_layer = r['layers'][peak_idx]
                    print(f"  {peak_val:>14.4f} (L{peak_layer:<2})", end="")
                else:
                    print(f"  {'---':>20}", end="")
            print()

    # Cross-alignment: does deception LoRA align MORE with deception traits than controls?
    if len(traits) >= 1 and len(results) >= 2:
        print(f"\n{'='*80}")
        print("SPECIFICITY: alignment ratio (each LoRA's alignment with each trait)")
        print("If deception LoRA specifically encodes deception, its ratio should be higher")
        print(f"{'='*80}")
        for module in modules:
            short_module = module.split('.')[-1]
            print(f"\n  [{short_module}] — sum of alignment across layers 30-70")
            print(f"  {'Variant':<20}", end="")
            for trait in traits:
                print(f"  {trait:>20}", end="")
            print()

            for vname in results:
                print(f"  {vname:<20}", end="")
                for trait in traits:
                    r = results[vname].get(trait, {}).get(module, {})
                    # Sum alignment in layers 30-70 (where signal peaks per plan)
                    total = sum(
                        a for l, a in zip(r.get('layers', []), r.get('alignment', []))
                        if 30 <= l <= 70
                    )
                    print(f"  {total:>20.4f}", end="")
                print()


if __name__ == "__main__":
    main()
