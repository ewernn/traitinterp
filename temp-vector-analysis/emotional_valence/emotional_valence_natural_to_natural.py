#!/usr/bin/env python3
"""
Same-distribution test: Natural → Natural
Split natural data 80/20, extract vectors from 80%, test on 20%
"""
import torch
import json
from pathlib import Path
import numpy as np

print("="*80)
print("SAME-DISTRIBUTION TEST: Natural → Natural")
print("emotional_valence")
print("="*80)

# Load natural activations
acts_dir = Path('experiments/gemma_2b_cognitive_nov20/emotional_valence/extraction/activations')

# Get counts from metadata
metadata_file = acts_dir / 'metadata.json'
with open(metadata_file) as f:
    metadata = json.load(f)

# Note: metadata says 201 total, but let me load actual files to verify
print("\nLoading natural activations from per-layer files...")

# Load from first layer to get counts
pos_acts_l0 = torch.load(acts_dir / 'pos_layer0.pt')
neg_acts_l0 = torch.load(acts_dir / 'neg_layer0.pt')

n_pos = pos_acts_l0.shape[0]
n_neg = neg_acts_l0.shape[0]

print(f"Positive examples: {n_pos}")
print(f"Negative examples: {n_neg}")
print(f"Total: {n_pos + n_neg}")

# Split 80/20
n_pos_train = int(n_pos * 0.8)
n_neg_train = int(n_neg * 0.8)

print(f"\nTrain: {n_pos_train} pos + {n_neg_train} neg = {n_pos_train + n_neg_train}")
print(f"Test:  {n_pos - n_pos_train} pos + {n_neg - n_neg_train} neg = {(n_pos - n_pos_train) + (n_neg - n_neg_train)}")

# Test all layers
results = {}

print(f"\n{'='*80}")
print("EXTRACTING VECTORS FROM TRAINING SPLIT & TESTING")
print(f"{'='*80}")

# Import extraction methods
import sys
sys.path.append('.')
from traitlens.methods import MeanDifferenceMethod, ProbeMethod

mean_diff = MeanDifferenceMethod()
probe = ProbeMethod()

for layer in range(26):
    print(f"\nLayer {layer}:")
    results[layer] = {}

    # Load activations for this layer
    pos_acts = torch.load(acts_dir / f'pos_layer{layer}.pt')
    neg_acts = torch.load(acts_dir / f'neg_layer{layer}.pt')

    # Split train/test
    pos_train_acts = pos_acts[:n_pos_train]
    pos_test_acts = pos_acts[n_pos_train:]
    neg_train_acts = neg_acts[:n_neg_train]
    neg_test_acts = neg_acts[n_neg_train:]

    # Extract vectors from training split
    for method_name, method in [('mean_diff', mean_diff), ('probe', probe)]:
        # Extract vector
        result = method.extract(pos_train_acts, neg_train_acts)
        vector = result['vector'].to(torch.float32)

        # Project test data onto vector
        pos_proj = pos_test_acts.to(torch.float32) @ vector
        neg_proj = neg_test_acts.to(torch.float32) @ vector

        # Check if sign needs flipping
        pos_mean = pos_proj.mean().item()
        neg_mean = neg_proj.mean().item()

        if pos_mean < neg_mean:
            # Wrong sign, flip
            vector = -vector
            pos_proj = -pos_proj
            neg_proj = -neg_proj
            sign_flipped = True
        else:
            sign_flipped = False

        # Calculate accuracy (threshold = 0)
        pos_correct = (pos_proj > 0).sum().item()
        neg_correct = (neg_proj < 0).sum().item()
        total_correct = pos_correct + neg_correct
        total = len(pos_proj) + len(neg_proj)
        accuracy = total_correct / total

        results[layer][method_name] = {
            'accuracy': accuracy,
            'correct': total_correct,
            'total': total,
            'sign_flipped': sign_flipped,
            'pos_mean': pos_mean,
            'neg_mean': neg_mean
        }

        print(f"  {method_name:12s}: {accuracy*100:5.1f}% ({total_correct}/{total})" +
              (f" [sign flipped]" if sign_flipped else ""))

# Save results
output_file = Path('/tmp/emotional_valence_natural_to_natural_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

# Find best performers
best_overall = []
for layer in range(26):
    for method in ['mean_diff', 'probe']:
        if method in results[layer]:
            acc = results[layer][method]['accuracy']
            best_overall.append((acc, layer, method))

best_overall.sort(reverse=True)

print("\nTop 10 performers:")
for i, (acc, layer, method) in enumerate(best_overall[:10], 1):
    print(f"{i:2d}. {method:12s} layer {layer:2d}: {acc*100:5.1f}%")

print(f"\nResults saved to: {output_file}")
print("\nNote: Only tested mean_diff and probe (ICA/gradient require more complex setup)")
