#!/usr/bin/env python3
"""
Cross-distribution test: Natural → Instruction
Use natural vectors (layer 16 only) to test on instruction data
"""
import torch
import json
from pathlib import Path

print("="*80)
print("CROSS-DISTRIBUTION TEST: Natural → Instruction")
print("emotional_valence")
print("="*80)

# Load natural vectors (only have layer 16)
vectors_dir = Path('experiments/gemma_2b_cognitive_nov20/emotional_valence/extraction/vectors')
layer = 16

print(f"\nLoading natural vectors (layer {layer} only)...")
natural_vectors = {}
for method in ['mean_diff', 'probe']:
    vector_file = vectors_dir / f'{method}_layer{layer}.pt'
    if vector_file.exists():
        # Check timestamp - newer files are from natural extraction
        import os
        mtime = os.path.getmtime(vector_file)
        from datetime import datetime
        mod_date = datetime.fromtimestamp(mtime)
        # Natural extraction was today (Nov 17), instruction was Nov 16
        if '2025-11-17' in str(mod_date):
            natural_vectors[method] = torch.load(vector_file)
            print(f"  ✓ {method}: {vector_file.name} (modified {mod_date})")

if not natural_vectors:
    print("⚠️  No natural vectors found! Need to extract first.")
    print("    (Looking for vectors modified on 2025-11-17)")
    exit(1)

# Load instruction test data
print(f"\nLoading instruction-based test activations...")
acts_file = Path('experiments/gemma_2b_cognitive_nov20/emotional_valence/extraction/activations/all_layers.pt')
acts = torch.load(acts_file)

metadata_file = Path('experiments/gemma_2b_cognitive_nov20/emotional_valence/extraction/activations/metadata.json')
with open(metadata_file) as f:
    metadata = json.load(f)

n_pos = metadata['n_positive']
n_neg = metadata['n_negative']

# Use all instruction data as test set
pos_test_acts = acts[:n_pos, layer, :]
neg_test_acts = acts[n_pos:n_pos+n_neg, layer, :]

print(f"Test data: {n_pos} pos + {n_neg} neg = {n_pos + n_neg}")

# Test natural vectors on instruction data
print(f"\n{'='*80}")
print(f"TESTING NATURAL VECTORS ON INSTRUCTION DATA")
print(f"{'='*80}")

results = {}

for method, vector in natural_vectors.items():
    vector = vector.to(torch.float32)

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

    results[method] = {
        'accuracy': accuracy,
        'correct': total_correct,
        'total': total,
        'sign_flipped': sign_flipped,
        'pos_mean': pos_mean,
        'neg_mean': neg_mean
    }

    print(f"\n{method:12s}: {accuracy*100:5.1f}% ({total_correct}/{total})" +
          (f" [sign flipped]" if sign_flipped else ""))

# Save results
output_file = Path('/tmp/emotional_valence_natural_to_instruction_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

print(f"\nResults for layer {layer}:")
for method in sorted(results.keys()):
    acc = results[method]['accuracy']
    print(f"  {method:12s}: {acc*100:5.1f}%")

print(f"\nResults saved to: {output_file}")
print("\nNote: Only tested layer 16 (only layer with natural vectors)")
