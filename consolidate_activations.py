#!/usr/bin/env python3
"""Consolidate per-layer activation files into all_layers.pt format"""

import torch
from pathlib import Path
import sys

def consolidate_activations(experiment, trait):
    """Convert pos_layer{N}.pt, neg_layer{N}.pt -> all_layers.pt format"""

    acts_dir = Path('experiments') / experiment / trait / 'extraction' / 'activations'

    if not acts_dir.exists():
        print(f"❌ Directory not found: {acts_dir}")
        return False

    # Check if already has consolidated format
    if (acts_dir / 'all_layers.pt').exists():
        print(f"✓ all_layers.pt already exists")
        return True

    print(f"Consolidating activations for {experiment}/{trait}...")

    # Find all layer files
    pos_files = sorted(acts_dir.glob('pos_layer*.pt'))
    neg_files = sorted(acts_dir.glob('neg_layer*.pt'))

    if not pos_files or not neg_files:
        print(f"❌ No per-layer activation files found")
        return False

    print(f"  Found {len(pos_files)} positive layers, {len(neg_files)} negative layers")

    # Load first file to get dimensions
    first_pos = torch.load(pos_files[0])
    first_neg = torch.load(neg_files[0])

    n_pos = first_pos.shape[0]
    n_neg = first_neg.shape[0]
    n_layers = len(pos_files)
    hidden_dim = first_pos.shape[1]

    print(f"  Shape: {n_pos} pos, {n_neg} neg, {n_layers} layers, {hidden_dim} dim")

    # Create consolidated tensors
    # Format: [n_examples, n_layers, hidden_dim]
    pos_all = torch.zeros(n_pos, n_layers, hidden_dim)
    neg_all = torch.zeros(n_neg, n_layers, hidden_dim)

    # Load and stack each layer
    for i, (pos_file, neg_file) in enumerate(zip(pos_files, neg_files)):
        pos_layer = torch.load(pos_file)
        neg_layer = torch.load(neg_file)

        pos_all[:, i, :] = pos_layer
        neg_all[:, i, :] = neg_layer

        if (i + 1) % 5 == 0:
            print(f"  Loaded {i+1}/{n_layers} layers")

    # Concatenate pos and neg: [n_pos + n_neg, n_layers, hidden_dim]
    all_layers = torch.cat([pos_all, neg_all], dim=0)

    print(f"  Final shape: {all_layers.shape}")

    # Save consolidated format
    output_file = acts_dir / 'all_layers.pt'
    torch.save(all_layers, output_file)
    print(f"✅ Saved to {output_file}")

    # Update metadata
    import json
    metadata_file = acts_dir / 'metadata.json'
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)

        metadata['n_examples_pos'] = n_pos
        metadata['n_examples_neg'] = n_neg
        metadata['consolidated'] = True

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Updated metadata")

    return True

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--trait', type=str, required=True)

    args = parser.parse_args()

    success = consolidate_activations(args.experiment, args.trait)
    sys.exit(0 if success else 1)
