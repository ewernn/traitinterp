#!/usr/bin/env python3
"""
Same-distribution test: Instruction → Instruction
Split instruction-based data 80/20, test vectors on held-out 20%
"""
import torch
import json
from pathlib import Path

print("="*80)
print("SAME-DISTRIBUTION TEST: Instruction → Instruction")
print("emotional_valence")
print("="*80)

# Load instruction-based activations
acts_file = Path('experiments/gemma_2b_cognitive_nov20/emotional_valence/extraction/activations/all_layers.pt')
acts = torch.load(acts_file)

print(f"\nLoaded instruction-based activations: {acts.shape}")
print(f"  (n_examples, n_layers, hidden_dim)")

# Check metadata for split point
import json
metadata_file = Path('experiments/gemma_2b_cognitive_nov20/emotional_valence/extraction/activations/metadata.json')
with open(metadata_file) as f:
    metadata = json.load(f)

n_pos = metadata['n_positive']
n_neg = metadata['n_negative']
print(f"\nPositive examples: {n_pos}")
print(f"Negative examples: {n_neg}")

# Split 80/20 for training/testing
n_pos_train = int(n_pos * 0.8)
n_neg_train = int(n_neg * 0.8)

# Training data (80%)
pos_train = acts[:n_pos_train]
neg_train = acts[n_pos:n_pos+n_neg_train]

# Test data (20%)
pos_test = acts[n_pos_train:n_pos]
neg_test = acts[n_pos+n_neg_train:]

print(f"\nTrain: {n_pos_train} pos + {n_neg_train} neg = {n_pos_train + n_neg_train}")
print(f"Test:  {n_pos - n_pos_train} pos + {n_neg - n_neg_train} neg = {(n_pos - n_pos_train) + (n_neg - n_neg_train)}")

# Test all layers and methods
results = {}
vectors_dir = Path('experiments/gemma_2b_cognitive_nov20/emotional_valence/extraction/vectors')

print(f"\n{'='*80}")
print("TESTING ALL LAYERS × METHODS")
print(f"{'='*80}")

for layer in range(26):
    print(f"\nLayer {layer}:")
    results[layer] = {}

    # Get test activations for this layer
    pos_test_acts = pos_test[:, layer, :]
    neg_test_acts = neg_test[:, layer, :]

    for method in ['mean_diff', 'probe', 'ica', 'gradient']:
        vector_file = vectors_dir / f'{method}_layer{layer}.pt'

        if not vector_file.exists():
            print(f"  ⚠️  {method}: vector not found")
            continue

        vector = torch.load(vector_file).to(torch.float32)

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

        results[layer][method] = {
            'accuracy': accuracy,
            'correct': total_correct,
            'total': total,
            'sign_flipped': sign_flipped,
            'pos_mean': pos_proj.mean().item(),
            'neg_mean': neg_proj.mean().item()
        }

        print(f"  {method:12s}: {accuracy*100:5.1f}% ({total_correct}/{total})" +
              (f" [sign flipped]" if sign_flipped else ""))

# Save results
output_file = Path('/tmp/emotional_valence_instruction_to_instruction_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

# Find best performers
best_overall = []
for layer in range(26):
    for method in ['mean_diff', 'probe', 'ica', 'gradient']:
        if method in results[layer]:
            acc = results[layer][method]['accuracy']
            best_overall.append((acc, layer, method))

best_overall.sort(reverse=True)

print("\nTop 10 performers:")
for i, (acc, layer, method) in enumerate(best_overall[:10], 1):
    print(f"{i:2d}. {method:12s} layer {layer:2d}: {acc*100:5.1f}%")

print(f"\nResults saved to: {output_file}")
