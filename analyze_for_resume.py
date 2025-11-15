#!/usr/bin/env python3
"""Quick analysis to get resume-worthy metrics."""
import csv
from pathlib import Path
from collections import defaultdict

exp_dir = Path("experiments/gemma_2b_cognitive_nov20")

print("=" * 70)
print("RESUME-WORTHY METRICS")
print("=" * 70)

# Count traits
traits = [d.name for d in exp_dir.iterdir() if d.is_dir() and (d / "responses").exists()]
print(f"\n✓ Extracted {len(traits)} behavioral traits")

# Count examples and calculate separation
total_examples = 0
trait_separations = {}

for trait in traits:
    pos_file = exp_dir / trait / "responses" / "pos.csv"
    neg_file = exp_dir / trait / "responses" / "neg.csv"

    if not pos_file.exists() or not neg_file.exists():
        continue

    # Read scores
    pos_scores = []
    neg_scores = []

    try:
        with open(pos_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('trait_score'):
                    try:
                        pos_scores.append(float(row['trait_score']))
                    except:
                        pass

        with open(neg_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('trait_score'):
                    try:
                        neg_scores.append(float(row['trait_score']))
                    except:
                        pass

        if pos_scores and neg_scores:
            total_examples += len(pos_scores) + len(neg_scores)
            separation = abs(sum(pos_scores)/len(pos_scores) - sum(neg_scores)/len(neg_scores))
            trait_separations[trait] = {
                'pos_mean': sum(pos_scores)/len(pos_scores),
                'neg_mean': sum(neg_scores)/len(neg_scores),
                'separation': separation,
                'n_pos': len(pos_scores),
                'n_neg': len(neg_scores)
            }
    except Exception as e:
        print(f"  Warning: Could not process {trait}: {e}")

print(f"✓ Processed {total_examples:,} total examples")

# Count vectors
vector_count = 0
methods_found = set()
layers_found = set()

for trait_dir in exp_dir.iterdir():
    if not trait_dir.is_dir():
        continue
    vectors_dir = trait_dir / "vectors"
    if vectors_dir.exists():
        for vec_file in vectors_dir.glob("*.pt"):
            vector_count += 1
            # Parse filename: method_layerN.pt
            parts = vec_file.stem.split('_')
            if len(parts) >= 2:
                method = parts[0]
                methods_found.add(method)
                if 'layer' in parts[1]:
                    layer_num = parts[1].replace('layer', '')
                    try:
                        layers_found.add(int(layer_num))
                    except:
                        pass

print(f"✓ Extracted {vector_count:,} steering vectors")
print(f"  - Methods: {', '.join(sorted(methods_found))}")
print(f"  - Layers: {min(layers_found)}-{max(layers_found)} ({len(layers_found)} layers)")

# Show top performing traits
print(f"\n{'='*70}")
print("TRAIT SEPARATION QUALITY")
print(f"{'='*70}")
print(f"{'Trait':<25} {'POS Mean':<10} {'NEG Mean':<10} {'Separation':<12}")
print("-" * 70)

for trait, stats in sorted(trait_separations.items(), key=lambda x: x[1]['separation'], reverse=True)[:10]:
    print(f"{trait:<25} {stats['pos_mean']:>8.1f}   {stats['neg_mean']:>8.1f}   {stats['separation']:>10.1f}")

# Calculate average separation
avg_sep = sum(s['separation'] for s in trait_separations.values()) / len(trait_separations)
print("-" * 70)
print(f"{'AVERAGE':<25} {'':>8}   {'':>8}   {avg_sep:>10.1f}")

print(f"\n{'='*70}")
print("KEY METRICS FOR RESUME")
print(f"{'='*70}")
print(f"""
1. Scale: {len(traits)} traits × {total_examples//len(traits):,} examples = {total_examples:,} total examples
2. Methods: {len(methods_found)} extraction methods ({', '.join(sorted(methods_found))})
3. Coverage: {len(layers_found)} layers analyzed (L{min(layers_found)}-L{max(layers_found)})
4. Quality: Average trait separation = {avg_sep:.1f} points (0-100 scale)
5. Output: {vector_count:,} steering vectors extracted
""")

# Find best traits for case studies
print("RECOMMENDED TRAITS FOR RESUME (highest separation):")
for trait, stats in sorted(trait_separations.items(), key=lambda x: x[1]['separation'], reverse=True)[:3]:
    print(f"  • {trait.replace('_', ' ').title()}: {stats['separation']:.1f} point separation")

print(f"\n{'='*70}")
print("SUGGESTED RESUME BULLET")
print(f"{'='*70}")
print(f"""
LLM Behavioral Monitoring Framework | Python, PyTorch, Transformers
• Extracted steering vectors for {len(traits)} behavioral traits (refusal, sycophancy,
  uncertainty) from Gemma-2B using contrastive activation analysis
• Processed {total_examples:,} examples across {len(layers_found)} model layers with {len(methods_found)} extraction methods
  (mean difference, linear probes, ICA, gradient optimization)
• Achieved {avg_sep:.0f}-point average separation on 0-100 scale between positive and
  negative trait expressions, enabling reliable behavioral classification
• Built automated pipeline generating {vector_count:,} steering vectors for real-time
  LLM monitoring and interpretability analysis
""")
