#!/usr/bin/env python3
"""
4×4 Distribution Matrix Analysis for Refusal Trait

Tests cross-distribution robustness of trait vectors:
- Inst→Inst: Instruction-trained on instruction-test
- Inst→Nat: Instruction-trained on natural-test
- Nat→Nat: Natural-trained on natural-test
- Nat→Inst: Natural-trained on instruction-test

Sweeps all 4 methods × 26 layers to find optimal configurations.
"""

import sys
import torch
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def load_activations(experiment, trait, distribution):
    """Load activations for a given distribution"""
    acts_dir = Path('experiments') / experiment / trait / 'extraction' / 'activations'

    # Load all_layers format: [n_examples, n_layers, hidden_dim]
    all_layers = torch.load(acts_dir / 'all_layers.pt')

    # Load metadata to get pos/neg split
    metadata_file = acts_dir / 'metadata.json'
    with open(metadata_file) as f:
        metadata = json.load(f)

    n_pos = metadata['n_examples_pos'] if 'n_examples_pos' in metadata else metadata['n_positive']
    n_neg = metadata['n_examples_neg'] if 'n_examples_neg' in metadata else metadata['n_negative']

    # Split into pos/neg
    pos_acts = all_layers[:n_pos]  # [n_pos, n_layers, hidden_dim]
    neg_acts = all_layers[n_pos:n_pos+n_neg]  # [n_neg, n_layers, hidden_dim]

    print(f"  {distribution:10s} - Pos: {n_pos:3d}, Neg: {n_neg:3d}, Shape: {all_layers.shape}")

    return pos_acts, neg_acts

def test_vector_on_activations(vector, pos_acts, neg_acts):
    """
    Test a vector on activations

    Args:
        vector: [hidden_dim]
        pos_acts: [n_pos, hidden_dim]
        neg_acts: [n_neg, hidden_dim]

    Returns:
        accuracy: float
    """
    # Convert to float32 for consistency
    vector = vector.float()
    pos_acts = pos_acts.float()
    neg_acts = neg_acts.float()

    # Normalize vector
    vector_norm = vector / vector.norm()

    # Project activations onto vector
    pos_scores = (pos_acts @ vector_norm).numpy()
    neg_scores = (neg_acts @ vector_norm).numpy()

    # Create labels (1=positive, 0=negative)
    pos_labels = np.ones(len(pos_scores))
    neg_labels = np.zeros(len(neg_scores))

    # Combine
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([pos_labels, neg_labels])

    # Predict based on threshold of 0
    predictions = (all_scores > 0).astype(int)

    # Compute accuracy
    acc = accuracy_score(all_labels, predictions)

    # Compute separation (for analysis)
    separation = pos_scores.mean() - neg_scores.mean()

    return {
        'accuracy': acc,
        'separation': float(separation),
        'pos_mean': float(pos_scores.mean()),
        'neg_mean': float(neg_scores.mean()),
        'pos_std': float(pos_scores.std()),
        'neg_std': float(neg_scores.std())
    }

def run_4x4_matrix(experiment='gemma_2b_cognitive_nov20'):
    """Run complete 4×4 distribution matrix analysis"""

    print("="*80)
    print("4×4 DISTRIBUTION MATRIX ANALYSIS - REFUSAL TRAIT")
    print("="*80)
    print()

    # Load activations for both distributions
    print("Loading activations...")
    inst_pos, inst_neg = load_activations(experiment, 'refusal', 'instruction')
    nat_pos, nat_neg = load_activations(experiment, 'refusal_natural', 'natural')

    n_layers = inst_pos.shape[1]
    hidden_dim = inst_pos.shape[2]

    print(f"\nDataset info:")
    print(f"  Layers: {n_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    print()

    # Methods to test
    methods = ['mean_diff', 'probe', 'ica', 'gradient']

    # Results storage
    results = {
        'experiment': experiment,
        'n_layers': n_layers,
        'methods': methods,
        'quadrants': {}
    }

    # Define test quadrants
    quadrants = [
        ('inst_inst', 'refusal', inst_pos, inst_neg, 'Instruction → Instruction'),
        ('inst_nat', 'refusal', nat_pos, nat_neg, 'Instruction → Natural'),
        ('nat_nat', 'refusal_natural', nat_pos, nat_neg, 'Natural → Natural'),
        ('nat_inst', 'refusal_natural', inst_pos, inst_neg, 'Natural → Instruction'),
    ]

    # Run tests for each quadrant
    for quad_name, vector_trait, test_pos, test_neg, description in quadrants:
        print("="*80)
        print(f"QUADRANT: {description}")
        print("="*80)
        print()

        quadrant_results = {
            'description': description,
            'vector_source': vector_trait,
            'test_source': 'instruction' if 'inst' in quad_name.split('_')[1] else 'natural',
            'methods': {}
        }

        # Test each method
        for method in methods:
            print(f"Testing method: {method}")
            method_results = []

            # Test each layer
            for layer in tqdm(range(n_layers), desc=f"  {method}"):
                # Load vector
                vector_dir = Path('experiments') / experiment / vector_trait / 'extraction' / 'vectors'
                vector_file = vector_dir / f'{method}_layer{layer}.pt'

                if not vector_file.exists():
                    print(f"    ⚠️  Vector not found: {vector_file}")
                    continue

                vector = torch.load(vector_file, map_location='cpu')

                # Get test activations for this layer
                test_pos_layer = test_pos[:, layer, :]  # [n_pos, hidden_dim]
                test_neg_layer = test_neg[:, layer, :]  # [n_neg, hidden_dim]

                # Test vector
                result = test_vector_on_activations(vector, test_pos_layer, test_neg_layer)
                result['layer'] = layer

                method_results.append(result)

            # Find best layer for this method
            if method_results:
                best = max(method_results, key=lambda x: x['accuracy'])
                avg_acc = np.mean([r['accuracy'] for r in method_results])

                print(f"    Best layer: {best['layer']:2d} - Acc: {best['accuracy']:.3f}")
                print(f"    Avg accuracy: {avg_acc:.3f}")

                quadrant_results['methods'][method] = {
                    'best_layer': best['layer'],
                    'best_accuracy': best['accuracy'],
                    'best_separation': best['separation'],
                    'avg_accuracy': float(avg_acc),
                    'all_layers': method_results
                }

            print()

        results['quadrants'][quad_name] = quadrant_results
        print()

    # Summary analysis
    print("="*80)
    print("SUMMARY: BEST CONFIGURATIONS PER QUADRANT")
    print("="*80)
    print()

    summary = {}
    for quad_name, quad_data in results['quadrants'].items():
        print(f"{quad_data['description']:30s}")

        best_overall = None
        best_acc = 0

        for method, method_data in quad_data['methods'].items():
            acc = method_data['best_accuracy']
            layer = method_data['best_layer']

            print(f"  {method:12s}: {acc:.3f} @ L{layer:2d}")

            if acc > best_acc:
                best_acc = acc
                best_overall = {
                    'method': method,
                    'layer': layer,
                    'accuracy': acc,
                    'separation': method_data['best_separation']
                }

        summary[quad_name] = best_overall
        print(f"  → Best: {best_overall['method']:12s} @ L{best_overall['layer']:2d} = {best_overall['accuracy']:.1%}")
        print()

    results['summary'] = summary

    # Key insight: cross-distribution performance
    print("="*80)
    print("KEY INSIGHT: CROSS-DISTRIBUTION ROBUSTNESS")
    print("="*80)
    print()

    inst_inst_best = summary['inst_inst']['accuracy']
    inst_nat_best = summary['inst_nat']['accuracy']
    nat_nat_best = summary['nat_nat']['accuracy']
    nat_inst_best = summary['nat_inst']['accuracy']

    print(f"Instruction-trained vectors:")
    print(f"  In-distribution (Inst→Inst):  {inst_inst_best:.1%}")
    print(f"  Cross-distribution (Inst→Nat): {inst_nat_best:.1%}")
    print(f"  Drop: {(inst_inst_best - inst_nat_best)*100:.1f}pp")
    print()

    print(f"Natural-trained vectors:")
    print(f"  In-distribution (Nat→Nat):     {nat_nat_best:.1%}")
    print(f"  Cross-distribution (Nat→Inst): {nat_inst_best:.1%}")
    print(f"  Drop: {(nat_nat_best - nat_inst_best)*100:.1f}pp")
    print()

    # Winner determination
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()

    # Which method wins for cross-distribution?
    inst_cross_method = summary['inst_nat']['method']
    nat_cross_method = summary['nat_inst']['method']

    if inst_cross_method == nat_cross_method:
        print(f"✅ CONSISTENT: {inst_cross_method.upper()} wins for both cross-distribution tests")
    else:
        print(f"⚠️  MIXED: Inst→Nat prefers {inst_cross_method}, Nat→Inst prefers {nat_cross_method}")

    # Compare to baseline traits
    print()
    print("Comparison to other traits:")
    print("  uncertainty_calibration: Gradient @ L14 = 96.1% (low separability)")
    print("  emotional_valence: Probe @ all layers = 100% (high separability)")
    print(f"  refusal: {summary['inst_nat']['method'].capitalize()} @ L{summary['inst_nat']['layer']} = {summary['inst_nat']['accuracy']:.1%}")
    print()

    # Save results
    output_file = Path('/tmp/refusal_full_4x4_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to {output_file}")
    print()

    return results

if __name__ == '__main__':
    results = run_4x4_matrix()
