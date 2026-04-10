"""Cross-trait normalization for Anthropic Emotion Concepts extraction method.

Implements Sofroniew et al. 2026 §1.1.4 steps 4-7:
  1. Load per-emotion mean activations (from --save-activations .pt files)
  2. Subtract grand mean across all emotions
  3. PCA on neutral corpus activations, find top PCs explaining threshold% variance
  4. Project those PCs out of each emotion vector
  5. Normalize to unit length, save as standard vector .pt files

Input:
  - Per-emotion saved activations: experiments/{exp}/extraction/{trait}/{variant}/activations/
  - Neutral corpus activations: same path structure for a "_neutral" pseudo-trait
Output:
  - Denoised vectors: experiments/{exp}/extraction/{trait}/{variant}/vectors/{position}/{component}/denoised/

Usage:
    python experiments/ant_emotion_concepts/scripts/cross_trait_normalize.py \
        --experiment ant_emotion_concepts \
        --layer 53 \
        --neutral-trait ant_emotion_concepts/_neutral

    # Multiple layers:
    python experiments/ant_emotion_concepts/scripts/cross_trait_normalize.py \
        --experiment ant_emotion_concepts \
        --layers 40,45,50,53,55,60 \
        --neutral-trait ant_emotion_concepts/_neutral

    # Dry run (print stats, don't save):
    python experiments/ant_emotion_concepts/scripts/cross_trait_normalize.py \
        --experiment ant_emotion_concepts --layer 53 --dry-run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from core.math import project_out_subspace
from utils.paths import get as get_path, discover_traits

from shared import filter_neutral_traits, grand_mean_subtract, denoise_with_neutral_pcs


def load_emotion_means(experiment, traits, layer, model_variant, component, position):
    """Load saved activations for each emotion and compute mean.

    Returns:
        means: dict {trait_path: [hidden_dim] tensor}
        failed: list of trait paths that couldn't be loaded
    """
    from utils.load_activations import load_train_activations

    means = {}
    failed = []

    for trait in traits:
        try:
            pos_acts, neg_acts = load_train_activations(
                experiment, trait, model_variant, layer,
                component=component, position=position
            )
            if pos_acts.numel() == 0:
                print(f"    SKIP {trait}: empty activations")
                failed.append(trait)
                continue
            means[trait] = pos_acts.float().mean(dim=0)
        except (FileNotFoundError, Exception) as e:
            print(f"    SKIP {trait}: {e}")
            failed.append(trait)

    return means, failed


def load_neutral_activations(experiment, neutral_trait, layer, model_variant, component, position):
    """Load neutral corpus activations for PCA denoising.

    Returns: [n_neutral, hidden_dim] tensor
    """
    from utils.load_activations import load_train_activations

    pos_acts, _ = load_train_activations(
        experiment, neutral_trait, model_variant, layer,
        component=component, position=position
    )
    if pos_acts.numel() == 0:
        raise ValueError(f"No neutral activations found for {neutral_trait} at layer {layer}")

    return pos_acts.float()


def denoise_and_save(experiment, centered_means, neutral_acts, grand_mean,
                     layer, model_variant, component, position,
                     variance_threshold=0.5, output_method='denoised', dry_run=False):
    """Denoise centered vectors via neutral PCs, normalize, and save.

    Returns: (n_saved, n_pcs, denoised_dict or None)
    """
    from utils.paths import sanitize_position

    # Denoise using shared utility
    denoised_dict = denoise_with_neutral_pcs(
        centered_means, neutral_acts, variance_threshold=variance_threshold,
    )

    # Infer n_pcs from the print output of denoise_with_neutral_pcs
    # (it already prints the count). We also need it for metadata, so recompute.
    from core.math import pca
    n_components = min(50, len(neutral_acts))
    components, variance_ratio, _ = pca(neutral_acts.float(), n_components=n_components)
    cumvar = torch.cumsum(variance_ratio, dim=0)
    n_pcs = int((cumvar < variance_threshold).sum().item()) + 1
    n_pcs = min(n_pcs, len(components))
    neutral_basis = components[:n_pcs]

    count = 0
    for trait, denoised_vec in denoised_dict.items():
        # Normalize to unit length
        norm = denoised_vec.norm()
        if norm < 1e-8:
            print(f"    WARNING: {trait} has near-zero norm after denoising, skipping")
            continue
        vector = denoised_vec / norm

        if dry_run:
            count += 1
            continue

        # Save in standard vector format
        san_pos = sanitize_position(position)
        vector_dir = (
            get_path('experiments.base', experiment=experiment)
            / 'extraction' / trait / model_variant
            / 'vectors' / san_pos / component / output_method
        )
        vector_dir.mkdir(parents=True, exist_ok=True)

        vector_path = vector_dir / f'layer{layer}.pt'
        torch.save(vector.to(torch.float32), vector_path)

        # Save metadata
        centered_vec = centered_means[trait]
        meta_path = vector_dir / 'metadata.json'
        meta = {
            'model_variant': model_variant,
            'trait': trait,
            'method': output_method,
            'component': component,
            'position': position,
            'layer': layer,
            'normalization': {
                'type': 'anthropic_emotion_concepts',
                'grand_mean_subtracted': True,
                'neutral_pcs_removed': n_pcs,
                'pre_denoise_norm': float(centered_vec.norm()),
                'post_denoise_norm': float(norm),
            },
            'timestamp': datetime.now().isoformat(),
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        count += 1

    return count, n_pcs, neutral_basis


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--layer', type=int, default=None, help='Single layer')
    parser.add_argument('--layers', type=str, default=None, help='Comma-separated layers')
    parser.add_argument('--model-variant', default=None)
    parser.add_argument('--component', default='residual')
    parser.add_argument('--position', default='response[50:]')
    parser.add_argument('--category', default='ant_emotion_concepts',
                        help='Trait category to normalize across')
    parser.add_argument('--neutral-trait', default='ant_emotion_concepts/_neutral',
                        help='Trait path for neutral corpus activations')
    parser.add_argument('--variance-threshold', type=float, default=0.5,
                        help='Fraction of neutral variance to remove (default: 0.5)')
    parser.add_argument('--output-method', default='denoised',
                        help='Method name for saved vectors (default: denoised)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print stats without saving vectors')
    args = parser.parse_args()

    # Resolve layers
    if args.layer is not None:
        layers = [args.layer]
    elif args.layers is not None:
        layers = [int(x) for x in args.layers.split(',')]
    else:
        parser.error('Specify --layer or --layers')

    # Resolve model variant
    model_variant = args.model_variant
    if model_variant is None:
        from utils.paths import get_default_variant
        model_variant = get_default_variant(args.experiment, mode='extraction')
        print(f"Auto-detected model variant: {model_variant}")

    # Discover emotion traits (excluding _neutral pseudo-trait)
    traits = filter_neutral_traits(discover_traits(args.category))
    print(f"Found {len(traits)} emotion traits in {args.category}")

    if len(traits) < 2:
        print("ERROR: Need at least 2 traits for cross-trait normalization")
        sys.exit(1)

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        # Step 1: Load per-emotion mean activations
        print(f"\n  Loading activations for {len(traits)} emotions...")
        means, failed = load_emotion_means(
            args.experiment, traits, layer, model_variant,
            args.component, args.position
        )
        print(f"    Loaded {len(means)}/{len(traits)} emotions ({len(failed)} failed)")

        if len(means) < 2:
            print("    ERROR: Need at least 2 loaded emotions, skipping layer")
            continue

        all_means = torch.stack(list(means.values()))
        print(f"    Mean activation norm: {all_means.norm(dim=-1).mean():.1f} "
              f"± {all_means.norm(dim=-1).std():.1f}")

        # Step 2: Grand mean subtraction
        print(f"\n  Computing grand mean and centering...")
        centered, grand_mean = grand_mean_subtract(means)
        centered_stack = torch.stack(list(centered.values()))
        print(f"    Grand mean norm: {grand_mean.norm():.1f}")
        print(f"    Centered mean norm: {centered_stack.norm(dim=-1).mean():.1f} "
              f"± {centered_stack.norm(dim=-1).std():.1f}")

        # Step 3: Load neutral activations
        print(f"\n  Loading neutral corpus activations from '{args.neutral_trait}'...")
        try:
            neutral_acts = load_neutral_activations(
                args.experiment, args.neutral_trait, layer,
                model_variant, args.component, args.position
            )
            print(f"    Loaded {neutral_acts.shape[0]} neutral samples")
            print(f"    Neutral activation norm: {neutral_acts.norm(dim=-1).mean():.1f} "
                  f"± {neutral_acts.norm(dim=-1).std():.1f}")
        except (FileNotFoundError, ValueError) as e:
            print(f"    WARNING: {e}")
            print(f"    Skipping neutral-PC denoising (saving centered-only vectors)")
            neutral_acts = None

        # Step 4: Denoise and save
        action = "Would save" if args.dry_run else "Saving"
        print(f"\n  {action} {len(centered)} vectors (method='{args.output_method}')...")

        if neutral_acts is not None:
            print(f"\n  PCA on neutral activations (threshold={args.variance_threshold})...")
            n_saved, k, neutral_basis = denoise_and_save(
                args.experiment, centered, neutral_acts, grand_mean,
                layer, model_variant, args.component, args.position,
                args.variance_threshold, args.output_method, args.dry_run
            )
            actual_var_msg = ""  # denoise_with_neutral_pcs already prints stats
        else:
            k = 0
            n_saved = len(centered)
            neutral_basis = torch.empty(0, centered_stack.shape[-1])

        # Post-denoise stats
        if not args.dry_run and neutral_acts is not None and k > 0:
            sample_trait = list(centered.keys())[0]
            sample_denoised = project_out_subspace(centered[sample_trait], neutral_basis)
            sample_norm = sample_denoised.norm()
            original_norm = centered[sample_trait].norm()
            print(f"    Sample ({sample_trait.split('/')[-1]}): "
                  f"pre={original_norm:.2f} -> post={sample_norm:.2f} "
                  f"({sample_norm/original_norm*100:.1f}% retained)")

        print(f"    {'Would save' if args.dry_run else 'Saved'} {n_saved} vectors")

    # Summary
    print(f"\n{'='*60}")
    print(f"Cross-trait normalization complete")
    print(f"  Emotions: {len(means)}")
    print(f"  Layers: {layers}")
    print(f"  Neutral PCs removed: {k}")
    print(f"  Output method: {args.output_method}")
    if args.dry_run:
        print(f"  DRY RUN -- no files written")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
