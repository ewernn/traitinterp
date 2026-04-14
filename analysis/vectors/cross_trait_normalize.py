"""Cross-trait normalization for Sofroniew et al. 2026 extraction method.

Implements the post-hoc vector transformations from §1.1.4 steps 4-5:
  - Step 4: Grand mean subtraction across the trait group ("+gm" suffix)
  - Step 5: Project out top PCs of neutral corpus activations ("+pc{N}" suffix)

Saves transformed vectors under composable method names:
  - {source}+gm                     (just grand-mean centered)
  - {source}+gm+pc{variance*100}    (centered AND PC-denoised)

See docs/extraction_guide.md "Composable Method Names" for the convention.

Input:
  - Per-trait saved activations: experiments/{exp}/extraction/{trait}/{variant}/activations/
  - Neutral corpus activations: same path structure for a reference trait
    (e.g. 'ant_emotion_concepts/_neutral', leading-underscore = reference)

Output:
  Vectors under the new method names:
    experiments/{exp}/extraction/{trait}/{variant}/vectors/{position}/{component}/{source}+gm/layer{N}.pt
    experiments/{exp}/extraction/{trait}/{variant}/vectors/{position}/{component}/{source}+gm+pc{N}/layer{N}.pt

  Plus a PC basis cache (shared across traits, per (reference_trait, layer)):
    experiments/{exp}/extraction/{reference_trait}/{variant}/reference_pcs/{position}/{component}/layer{L}.pt

Usage:
    python experiments/ant_emotion_concepts/scripts/cross_trait_normalize.py \
        --experiment ant_emotion_concepts \
        --layers 1,7,13,19,25,31,37,43,49,55,61,67,73,79 \
        --neutral-trait ant_emotion_concepts/_neutral

    # Skip neutral PC step (only save +gm vectors):
    python .../cross_trait_normalize.py --experiment ant_emotion_concepts --layers 53 --no-pc

    # Dry run:
    python .../cross_trait_normalize.py --experiment ant_emotion_concepts --layer 53 --dry-run
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.math import (
    grand_mean_center,
    compute_top_pcs_by_variance,
    denoise_with_pcs,
)
from utils.paths import get as get_path, discover_traits, sanitize_position
from utils.load_activations import load_train_activations


# =============================================================================
# Activation loading
# =============================================================================

def load_trait_means(experiment, traits, layer, model_variant, component, position):
    """Load saved activations for each trait and compute per-trait mean."""
    means = {}
    failed = []
    for trait in traits:
        try:
            pos_acts, _ = load_train_activations(
                experiment, trait, model_variant, layer,
                component=component, position=position,
            )
            if pos_acts.numel() == 0:
                failed.append((trait, "empty"))
                continue
            means[trait] = pos_acts.float().mean(dim=0)
        except Exception as e:
            failed.append((trait, str(e)[:80]))
    return means, failed


def load_reference_activations(experiment, reference_trait, layer, model_variant, component, position):
    """Load activations from the reference corpus (e.g., neutral dialogues)."""
    pos_acts, _ = load_train_activations(
        experiment, reference_trait, model_variant, layer,
        component=component, position=position,
    )
    if pos_acts.numel() == 0:
        raise ValueError(f"No activations found for reference '{reference_trait}' at layer {layer}")
    return pos_acts.float()


# =============================================================================
# PC basis caching
# =============================================================================

def get_reference_pcs_path(experiment, reference_trait, layer, model_variant, component, position):
    """Path to cached PC basis for a reference trait at a layer."""
    san_pos = sanitize_position(position)
    return (
        get_path('experiments.base', experiment=experiment)
        / 'extraction' / reference_trait / model_variant
        / 'reference_pcs' / san_pos / component / f'layer{layer}.pt'
    )


def activations_hash(activations: torch.Tensor) -> str:
    """Short deterministic hash of an activation tensor for cache invalidation."""
    flat = activations.cpu().contiguous().view(-1).to(torch.float32)
    h = hashlib.sha256()
    h.update(str(tuple(activations.shape)).encode())
    n = flat.numel()
    sample_ids = torch.linspace(0, n - 1, steps=min(256, n)).long()
    h.update(flat[sample_ids].numpy().tobytes())
    return h.hexdigest()[:16]


def compute_or_load_reference_pcs(
    experiment, reference_trait, layer, model_variant, component, position,
    variance_threshold=0.5, force=False,
):
    """Load cached PC basis for a reference trait, or compute + cache it.

    Returns:
        basis: [K, hidden_dim] tensor
        n_pcs: K
        cumulative_variance: float
        from_cache: bool
    """
    cache_path = get_reference_pcs_path(
        experiment, reference_trait, layer, model_variant, component, position,
    )

    acts = load_reference_activations(
        experiment, reference_trait, layer, model_variant, component, position,
    )
    act_hash = activations_hash(acts)

    if cache_path.exists() and not force:
        cached = torch.load(cache_path, weights_only=False)
        if (
            cached.get('activation_hash') == act_hash
            and cached.get('variance_threshold') == variance_threshold
        ):
            return (
                cached['basis'],
                int(cached['n_pcs']),
                float(cached['cumulative_variance']),
                True,
            )
        print(f"    Cache stale (hash or threshold mismatch), recomputing...")

    basis, variance_ratio, n_pcs = compute_top_pcs_by_variance(
        acts, variance_threshold=variance_threshold,
    )
    cumvar = float(variance_ratio.sum().item())

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'basis': basis.cpu(),
        'variance_ratio': variance_ratio.cpu(),
        'n_pcs': n_pcs,
        'cumulative_variance': cumvar,
        'activation_hash': act_hash,
        'variance_threshold': variance_threshold,
        'reference_trait': reference_trait,
        'layer': layer,
        'model_variant': model_variant,
        'component': component,
        'position': position,
        'n_reference_samples': int(acts.shape[0]),
        'timestamp': datetime.now().isoformat(),
    }, cache_path)

    return basis, n_pcs, cumvar, False


# =============================================================================
# Vector save with composable method names
# =============================================================================

def save_transformed_vectors(
    experiment, vectors_dict, layer, model_variant, component, position,
    method_name, transforms_meta, dry_run=False,
):
    """Save a dict of transformed vectors under a composed method name.

    Unit-normalizes each vector before saving.
    """
    san_pos = sanitize_position(position)
    count = 0
    for trait, vec in vectors_dict.items():
        norm = vec.norm()
        if norm < 1e-8:
            print(f"    WARNING: {trait} has near-zero norm, skipping")
            continue
        unit_vec = (vec / norm).to(torch.float32)

        if dry_run:
            count += 1
            continue

        vector_dir = (
            get_path('experiments.base', experiment=experiment)
            / 'extraction' / trait / model_variant
            / 'vectors' / san_pos / component / method_name
        )
        vector_dir.mkdir(parents=True, exist_ok=True)
        torch.save(unit_vec, vector_dir / f'layer{layer}.pt')

        meta = {
            'model_variant': model_variant,
            'trait': trait,
            'method': method_name,
            'component': component,
            'position': position,
            'layer': layer,
            'transforms': transforms_meta,
            'pre_norm': float(norm),
            'timestamp': datetime.now().isoformat(),
        }
        with open(vector_dir / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

        count += 1
    return count


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--layer', type=int, default=None, help='Single layer')
    parser.add_argument('--layers', type=str, default=None, help='Comma-separated layers')
    parser.add_argument('--model-variant', default=None)
    parser.add_argument('--component', default='residual')
    parser.add_argument('--position', default='response[50:]')
    parser.add_argument('--category', default='ant_emotion_concepts',
                        help='Trait category to normalize across')
    parser.add_argument('--neutral-trait', default='ant_emotion_concepts/_neutral',
                        help='Reference trait path for neutral corpus activations')
    parser.add_argument('--variance-threshold', type=float, default=0.5,
                        help='Cumulative neutral variance fraction to remove (default: 0.5)')
    parser.add_argument('--method', default='mean_diff',
                        help='Base method to transform (default: mean_diff)')
    parser.add_argument('--no-pc', action='store_true',
                        help='Skip neutral-PC step; only save {source}+gm vectors')
    parser.add_argument('--force-pc', action='store_true',
                        help='Recompute PC basis ignoring cache')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print stats without saving vectors')
    args = parser.parse_args()

    if args.layer is not None:
        layers = [args.layer]
    elif args.layers is not None:
        layers = [int(x) for x in args.layers.split(',')]
    else:
        parser.error('Specify --layer or --layers')

    model_variant = args.model_variant
    if model_variant is None:
        from utils.paths import get_default_variant
        model_variant = get_default_variant(args.experiment, mode='extraction')
        print(f"Auto-detected model variant: {model_variant}")

    # Note: discover_traits now excludes leading-underscore (reference) traits by default
    traits = discover_traits(args.category)
    print(f"Found {len(traits)} traits in category '{args.category}'")
    if len(traits) < 2:
        print("ERROR: Need at least 2 traits for cross-trait normalization")
        sys.exit(1)

    gm_method = f"{args.method}+gm"
    pc_pct = int(round(args.variance_threshold * 100))
    pc_method = f"{args.method}+gm+pc{pc_pct}"
    print(f"Output methods: {gm_method}" + (f", {pc_method}" if not args.no_pc else ""))

    summary = {"layers": {}, "gm_method": gm_method, "pc_method": pc_method if not args.no_pc else None}

    for layer in layers:
        print(f"\n{'='*60}\nLayer {layer}\n{'='*60}")

        print(f"\n  Loading activations for {len(traits)} traits...")
        means, failed = load_trait_means(
            args.experiment, traits, layer, model_variant, args.component, args.position,
        )
        print(f"    Loaded {len(means)}/{len(traits)} traits ({len(failed)} failed)")
        if len(means) < 2:
            print("    ERROR: Need at least 2 loaded traits, skipping layer")
            continue
        all_stack = torch.stack(list(means.values()))
        print(f"    Mean activation norm: {all_stack.norm(dim=-1).mean():.2f} ± {all_stack.norm(dim=-1).std():.2f}")

        print(f"\n  Grand-mean centering...")
        centered, grand = grand_mean_center(means)
        centered_stack = torch.stack(list(centered.values()))
        print(f"    ||grand_mean|| = {grand.norm():.2f}")
        print(f"    ||centered|| = {centered_stack.norm(dim=-1).mean():.2f} ± {centered_stack.norm(dim=-1).std():.2f}")

        n_gm = save_transformed_vectors(
            args.experiment, centered, layer, model_variant,
            args.component, args.position,
            method_name=gm_method,
            transforms_meta=[
                {'type': 'grand_mean_center', 'n_traits_in_group': len(means),
                 'group_category': args.category}
            ],
            dry_run=args.dry_run,
        )
        print(f"    Saved {n_gm} vectors as method='{gm_method}'")

        layer_summary = {'n_gm': n_gm}

        if args.no_pc:
            summary["layers"][str(layer)] = layer_summary
            continue

        print(f"\n  Loading/computing PC basis for '{args.neutral_trait}'...")
        try:
            pc_basis, n_pcs, cumvar, from_cache = compute_or_load_reference_pcs(
                args.experiment, args.neutral_trait, layer, model_variant,
                args.component, args.position,
                variance_threshold=args.variance_threshold, force=args.force_pc,
            )
            src = "cache" if from_cache else "fresh"
            print(f"    {n_pcs} PCs explaining {cumvar*100:.1f}% variance (from {src})")
        except (FileNotFoundError, ValueError) as e:
            print(f"    WARNING: {e}")
            print(f"    Skipping neutral-PC denoising for layer {layer}")
            layer_summary['n_pc_skipped_reason'] = str(e)[:200]
            summary["layers"][str(layer)] = layer_summary
            continue

        denoised = denoise_with_pcs(centered, pc_basis)
        denoised_stack = torch.stack(list(denoised.values()))
        print(f"    ||denoised|| = {denoised_stack.norm(dim=-1).mean():.2f} ± {denoised_stack.norm(dim=-1).std():.2f}")

        n_pc = save_transformed_vectors(
            args.experiment, denoised, layer, model_variant,
            args.component, args.position,
            method_name=pc_method,
            transforms_meta=[
                {'type': 'grand_mean_center', 'n_traits_in_group': len(means),
                 'group_category': args.category},
                {'type': 'project_out_subspace',
                 'reference_trait': args.neutral_trait,
                 'variance_threshold': args.variance_threshold,
                 'n_pcs_removed': n_pcs,
                 'actual_cumulative_variance': cumvar},
            ],
            dry_run=args.dry_run,
        )
        print(f"    Saved {n_pc} vectors as method='{pc_method}'")

        layer_summary['n_pc'] = n_pc
        layer_summary['n_pcs_removed'] = n_pcs
        summary["layers"][str(layer)] = layer_summary

    print(f"\n{'='*60}\nNormalization complete\n{'='*60}")
    print(f"  Layers processed: {len(summary['layers'])}")
    print(f"  Methods saved: {gm_method}" + (f", {pc_method}" if not args.no_pc else ""))
    if args.dry_run:
        print(f"  DRY RUN -- no files written")
    else:
        summary_path = (
            get_path('experiments.base', experiment=args.experiment)
            / 'results' / 'cross_trait_normalize_summary.json'
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary: {summary_path}")


if __name__ == '__main__':
    main()
