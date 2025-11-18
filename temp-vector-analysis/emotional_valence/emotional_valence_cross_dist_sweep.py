#!/usr/bin/env python3
"""
Full cross-distribution sweep for emotional_valence
Tests all 26 layers × 4 methods (instruction → natural)
"""

import torch
import json
from pathlib import Path

# Configuration
TRAIT = "emotional_valence"
EXPERIMENT = "gemma_2b_cognitive_nov20"
N_LAYERS = 26
METHODS = ['mean_diff', 'probe', 'ica', 'gradient']

# Paths
base_dir = Path(f'experiments/{EXPERIMENT}/{TRAIT}/extraction')
vectors_dir = base_dir / 'vectors'
activations_dir = base_dir / 'activations'

# Load natural test data (100 pos, 101 neg)
print("="*80)
print(f"CROSS-DISTRIBUTION TEST: {TRAIT}")
print("="*80)
print(f"\nLoading natural test data...")

n_pos_test = 100
n_neg_test = 101

results = {}

for layer in range(N_LAYERS):
    print(f"\n{'='*80}")
    print(f"LAYER {layer}")
    print(f"{'='*80}")

    # Load test activations (natural)
    pos_test_file = activations_dir / f'pos_layer{layer}.pt'
    neg_test_file = activations_dir / f'neg_layer{layer}.pt'

    if not pos_test_file.exists() or not neg_test_file.exists():
        print(f"⚠️  Skipping layer {layer} - activation files not found")
        continue

    pos_test_acts = torch.load(pos_test_file)
    neg_test_acts = torch.load(neg_test_file)

    print(f"Test data loaded:")
    print(f"  Positive: {pos_test_acts.shape}")
    print(f"  Negative: {neg_test_acts.shape}")

    layer_results = {}

    for method in METHODS:
        # Load vector (from instruction-based training)
        vector_file = vectors_dir / f'{method}_layer{layer}.pt'

        if not vector_file.exists():
            print(f"  ⚠️  {method}: vector not found")
            continue

        vector = torch.load(vector_file).to(torch.float32)

        # Project test data onto vector (convert to float32 for compatibility)
        pos_proj = pos_test_acts.to(torch.float32) @ vector
        neg_proj = neg_test_acts.to(torch.float32) @ vector

        # Check if sign needs flipping
        pos_mean = pos_proj.mean().item()
        neg_mean = neg_proj.mean().item()

        if pos_mean < neg_mean:
            # Wrong polarity - flip
            vector = -vector
            pos_proj = -pos_proj
            neg_proj = -neg_proj
            pos_mean = -pos_mean
            neg_mean = -neg_mean
            sign_flipped = True
        else:
            sign_flipped = False

        # Calculate accuracy (threshold = 0)
        pos_correct = (pos_proj > 0).sum().item()
        neg_correct = (neg_proj < 0).sum().item()
        total_correct = pos_correct + neg_correct
        total_examples = n_pos_test + n_neg_test
        accuracy = total_correct / total_examples

        # Calculate separation
        separation = pos_mean - neg_mean

        layer_results[method] = {
            'accuracy': accuracy,
            'pos_correct': pos_correct,
            'neg_correct': neg_correct,
            'total_correct': total_correct,
            'pos_mean': pos_mean,
            'neg_mean': neg_mean,
            'separation': separation,
            'sign_flipped': sign_flipped,
            'vector_norm': vector.norm().item()
        }

        print(f"  {method:12s}: {accuracy*100:5.1f}% ({total_correct}/{total_examples}) | sep={separation:6.2f} | {'FLIPPED' if sign_flipped else 'OK'}")

    results[f'layer_{layer}'] = layer_results

# Save results
output_file = Path('/tmp/emotional_valence_cross_dist_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print(f"{'='*80}\n")

# Find best performing layer per method
for method in METHODS:
    best_layer = None
    best_acc = 0

    for layer in range(N_LAYERS):
        layer_key = f'layer_{layer}'
        if layer_key in results and method in results[layer_key]:
            acc = results[layer_key][method]['accuracy']
            if acc > best_acc:
                best_acc = acc
                best_layer = layer

    if best_layer is not None:
        best_result = results[f'layer_{best_layer}'][method]
        print(f"{method:12s}: Layer {best_layer:2d} → {best_acc*100:5.1f}% ({best_result['total_correct']}/{n_pos_test + n_neg_test})")
    else:
        print(f"{method:12s}: No results")

print(f"\n✅ Results saved to {output_file}")
print(f"\n{'='*80}")
print("NEXT: Analyze layer profiles and compare to uncertainty_calibration")
print(f"{'='*80}")
