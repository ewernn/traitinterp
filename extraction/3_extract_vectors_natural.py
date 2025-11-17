#!/usr/bin/env python3
"""Extract vectors for natural elicitation - simplified version"""

import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from traitlens.methods import MeanDifferenceMethod, ProbeMethod

def extract_vectors_natural(experiment, trait, layer=16):
    """Extract vectors using mean_diff and probe methods"""

    print("="*80)
    print("EXTRACTING VECTORS - NATURAL ELICITATION")
    print(f"Experiment: {experiment}")
    print(f"Trait: {trait}")
    print(f"Layer: {layer}")
    print("="*80)

    # Load activations
    acts_dir = Path('experiments') / experiment / trait / 'extraction' / 'activations'
    vectors_dir = Path('experiments') / experiment / trait / 'extraction' / 'vectors'
    vectors_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading activations...")
    pos_acts = torch.load(acts_dir / f'pos_layer{layer}.pt')
    neg_acts = torch.load(acts_dir / f'neg_layer{layer}.pt')

    print(f"  Positive: {pos_acts.shape}")
    print(f"  Negative: {neg_acts.shape}")

    # Extract vectors
    print(f"\n{'='*80}")
    print("EXTRACTING VECTORS")
    print(f"{'='*80}\n")

    # Mean difference
    print("Method 1: Mean Difference")
    mean_diff = MeanDifferenceMethod()
    result_md = mean_diff.extract(pos_acts, neg_acts)
    vector_md = result_md['vector']

    md_file = vectors_dir / f'mean_diff_layer{layer}.pt'
    torch.save(vector_md, md_file)
    print(f"  ✅ Saved to {md_file}")
    print(f"     Norm: {vector_md.norm():.2f}")

    # Probe
    print("\nMethod 2: Linear Probe")
    probe = ProbeMethod()
    result_probe = probe.extract(pos_acts, neg_acts)
    vector_probe = result_probe['vector']

    probe_file = vectors_dir / f'probe_layer{layer}.pt'
    torch.save(vector_probe, probe_file)
    print(f"  ✅ Saved to {probe_file}")
    print(f"     Norm: {vector_probe.norm():.2f}")
    print(f"     Train accuracy: {result_probe['train_acc']:.3f}")

    # Save metadata (skipped for now - vectors are what matter)
    # metadata = {
    #     'experiment': experiment,
    #     'trait': trait,
    #     'layer': layer,
    #     'mean_diff': {
    #         'norm': vector_md.norm().item(),
    #         'pos_mean': float(result_md['pos_mean']),
    #         'neg_mean': float(result_md['neg_mean']),
    #         'contrast': float(result_md['pos_mean'] - result_md['neg_mean'])
    #     },
    #     'probe': {
    #         'norm': vector_probe.norm().item(),
    #         'train_acc': result_probe['train_acc'],
    #         'pos_mean': result_probe['pos_scores'].mean().item(),
    #         'neg_mean': result_probe['neg_scores'].mean().item(),
    #         'contrast': result_probe['pos_scores'].mean().item() - result_probe['neg_scores'].mean().item()
    #     }
    # }
    #
    # import json
    # with open(vectors_dir / f'metadata_layer{layer}.json', 'w') as f:
    #     json.dump(metadata, f, indent=2)

    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE!")
    print(f"{'='*80}\n")
    print(f"Vectors saved to: {vectors_dir}")
    print(f"\nNext step:")
    print(f"  python extraction/validate_natural_vectors.py --experiment {experiment} --trait {trait} --layer {layer} --method probe")

    return vectors_dir

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--trait', type=str, required=True)
    parser.add_argument('--layer', type=int, default=16)

    args = parser.parse_args()

    vectors_dir = extract_vectors_natural(
        experiment=args.experiment,
        trait=args.trait,
        layer=args.layer
    )

    print(f"\n✅ SUCCESS: Vectors extracted to {vectors_dir}")
