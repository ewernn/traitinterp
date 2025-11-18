#!/usr/bin/env python3
"""
Formality Trait Analysis

Tests natural-trained vectors on natural data to find optimal method/layer.
Expected: High separability, probe should excel (similar to emotional_valence).
"""

import sys
import torch
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score

def test_vector_on_activations(vector, pos_acts, neg_acts):
    """Test a vector on activations"""
    # Convert to float32
    vector = vector.float()
    pos_acts = pos_acts.float()
    neg_acts = neg_acts.float()

    # Normalize vector
    vector_norm = vector / vector.norm()

    # Project activations onto vector
    pos_scores = (pos_acts @ vector_norm).numpy()
    neg_scores = (neg_acts @ vector_norm).numpy()

    # Create labels (1=formal, 0=casual)
    pos_labels = np.ones(len(pos_scores))
    neg_labels = np.zeros(len(neg_scores))

    # Combine
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([pos_labels, neg_labels])

    # Predict based on threshold of 0
    predictions = (all_scores > 0).astype(int)

    # Compute accuracy
    acc = accuracy_score(all_labels, predictions)

    # Compute separation
    separation = pos_scores.mean() - neg_scores.mean()

    return {
        'accuracy': acc,
        'separation': float(separation),
        'pos_mean': float(pos_scores.mean()),
        'neg_mean': float(neg_scores.mean())
    }

def run_formality_analysis(experiment='gemma_2b_cognitive_nov20'):
    """Run formality trait analysis"""

    print("="*80)
    print("FORMALITY TRAIT ANALYSIS - NATURAL ELICITATION")
    print("="*80)
    print()

    trait = 'formality_natural'

    # Load activations
    acts_dir = Path('experiments') / experiment / trait / 'extraction' / 'activations'
    all_acts = torch.load(acts_dir / 'all_layers.pt')

    # Load metadata
    metadata_file = acts_dir / 'metadata.json'
    with open(metadata_file) as f:
        metadata = json.load(f)

    n_pos = metadata['n_examples_pos'] if 'n_examples_pos' in metadata else metadata['n_positive']
    n_neg = metadata['n_examples_neg'] if 'n_examples_neg' in metadata else metadata['n_negative']

    pos_acts = all_acts[:n_pos]  # [n_pos, n_layers, hidden_dim]
    neg_acts = all_acts[n_pos:n_pos+n_neg]  # [n_neg, n_layers, hidden_dim]

    n_layers = pos_acts.shape[1]
    hidden_dim = pos_acts.shape[2]

    print(f"Dataset info:")
    print(f"  Formal examples: {n_pos}")
    print(f"  Casual examples: {n_neg}")
    print(f"  Layers: {n_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    print()

    # Methods to test
    methods = ['mean_diff', 'probe', 'ica', 'gradient']

    # Results storage
    results = {
        'experiment': experiment,
        'trait': trait,
        'n_layers': n_layers,
        'methods': {}
    }

    # Test each method
    for method in methods:
        print(f"Testing method: {method}")
        method_results = []

        # Test each layer
        for layer in range(n_layers):
            # Load vector
            vector_dir = Path('experiments') / experiment / trait / 'extraction' / 'vectors'
            vector_file = vector_dir / f'{method}_layer{layer}.pt'

            if not vector_file.exists():
                continue

            vector = torch.load(vector_file, map_location='cpu')

            # Get test activations for this layer
            test_pos_layer = pos_acts[:, layer, :]  # [n_pos, hidden_dim]
            test_neg_layer = neg_acts[:, layer, :]  # [n_neg, hidden_dim]

            # Test vector
            result = test_vector_on_activations(vector, test_pos_layer, test_neg_layer)
            result['layer'] = layer

            method_results.append(result)

        # Find best layer
        if method_results:
            best = max(method_results, key=lambda x: x['accuracy'])
            avg_acc = np.mean([r['accuracy'] for r in method_results])

            print(f"  Best layer: {best['layer']:2d} - Acc: {best['accuracy']:.3f}")
            print(f"  Avg accuracy: {avg_acc:.3f}")

            results['methods'][method] = {
                'best_layer': best['layer'],
                'best_accuracy': best['accuracy'],
                'best_separation': best['separation'],
                'avg_accuracy': float(avg_acc),
                'all_layers': method_results
            }

        print()

    # Summary
    print("="*80)
    print("SUMMARY: FORMALITY TRAIT PERFORMANCE")
    print("="*80)
    print()

    best_overall = None
    best_acc = 0

    for method, method_data in results['methods'].items():
        acc = method_data['best_accuracy']
        layer = method_data['best_layer']

        print(f"{method:12s}: {acc:.1%} @ L{layer:2d}")

        if acc > best_acc:
            best_acc = acc
            best_overall = {
                'method': method,
                'layer': layer,
                'accuracy': acc,
                'separation': method_data['best_separation']
            }

    results['best_overall'] = best_overall

    print()
    print(f"→ Winner: {best_overall['method'].upper()} @ L{best_overall['layer']} = {best_overall['accuracy']:.1%}")
    print()

    # Comparison to other traits
    print("="*80)
    print("COMPARISON TO OTHER TRAITS")
    print("="*80)
    print()

    print("Cross-trait pattern:")
    print("  uncertainty_calibration (low sep):  Gradient @ L14 = 96.1%")
    print("  refusal (moderate sep):             Gradient @ L11 = 91.7%")
    print("  emotional_valence (high sep):       Probe @ all layers = 100%")
    print(f"  formality (high sep):               {best_overall['method'].capitalize()} @ L{best_overall['layer']} = {best_overall['accuracy']:.1%}")
    print()

    # Hypothesis validation
    if best_overall['method'] == 'probe' and best_overall['accuracy'] >= 0.95:
        print("✅ HYPOTHESIS CONFIRMED: High separability → Probe excels")
    elif best_overall['accuracy'] >= 0.95:
        print(f"⚠️  UNEXPECTED: {best_overall['method'].capitalize()} wins (expected Probe for high separability)")
    else:
        print(f"⚠️  LOWER THAN EXPECTED: {best_overall['accuracy']:.1%} (expected >95% for high separability)")

    print()

    # Save results
    output_file = Path('/tmp/formality_natural_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to {output_file}")
    print()

    return results

if __name__ == '__main__':
    results = run_formality_analysis()
