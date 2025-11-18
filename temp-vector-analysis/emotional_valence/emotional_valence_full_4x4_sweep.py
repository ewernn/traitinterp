#!/usr/bin/env python3
"""
Complete 4×4 Sweep for emotional_valence
4 train/test combinations × 4 extraction methods × 26 layers = 416 tests

Quadrants:
1. Instruction → Instruction (same-distribution baseline)
2. Instruction → Natural (cross-distribution)
3. Natural → Natural (same-distribution baseline)
4. Natural → Instruction (reverse cross-distribution)
"""
import sys
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score

sys.path.append('/Users/ewern/Desktop/code/trait-interp')

def load_instruction_activations(trait, layer):
    """Load instruction-based activations"""
    base = Path(f'experiments/gemma_2b_cognitive_nov20/{trait}/extraction/activations')
    all_layers = torch.load(base / 'all_layers.pt')  # [n_examples, 26, 2304]

    # Get metadata for pos/neg split
    import json
    metadata = json.load(open(base / 'metadata.json'))
    n_pos = metadata.get('n_positive', metadata.get('n_examples_pos', 100))
    n_neg = metadata.get('n_negative', metadata.get('n_examples_neg', 101))

    pos_acts = all_layers[:n_pos, layer, :]
    neg_acts = all_layers[n_pos:n_pos+n_neg, layer, :]

    return pos_acts, neg_acts

def load_natural_activations(trait, layer):
    """Load natural activations"""
    base = Path(f'experiments/gemma_2b_cognitive_nov20/{trait}/extraction/activations')
    pos_acts = torch.load(base / f'pos_layer{layer}.pt')
    neg_acts = torch.load(base / f'neg_layer{layer}.pt')
    return pos_acts, neg_acts

def load_vector(trait, method, layer, source='natural'):
    """Load extracted vector"""
    vectors_dir = Path(f'experiments/gemma_2b_cognitive_nov20/{trait}/extraction/vectors')
    vector_file = vectors_dir / f'{method}_layer{layer}.pt'
    if not vector_file.exists():
        return None
    return torch.load(vector_file)

def test_vector(vector, pos_acts, neg_acts):
    """Test vector on activations, return accuracy and correct sign"""
    if vector is None:
        return None, None

    # Convert to float32 if needed
    vector = vector.to(torch.float32)
    pos_acts = pos_acts.to(torch.float32)
    neg_acts = neg_acts.to(torch.float32)

    # Project
    pos_proj = pos_acts @ vector
    neg_proj = neg_acts @ vector

    # Check both polarities
    correct_polarity = (pos_proj.mean() > neg_proj.mean())

    if correct_polarity:
        correct = (pos_proj > 0).sum() + (neg_proj < 0).sum()
    else:
        correct = (pos_proj < 0).sum() + (neg_proj > 0).sum()

    total = len(pos_acts) + len(neg_acts)
    accuracy = correct.item() / total

    return accuracy, correct_polarity

def run_full_sweep(trait='emotional_valence'):
    """Run complete 4×4 sweep"""

    methods = ['mean_diff', 'probe', 'gradient', 'ica']

    # Define quadrants
    quadrants = {
        'inst_to_inst': {
            'train_loader': load_instruction_activations,
            'test_loader': load_instruction_activations,
            'vector_source': 'instruction',
            'split_ratio': 0.8  # 80/20 split
        },
        'inst_to_nat': {
            'train_loader': load_instruction_activations,
            'test_loader': load_natural_activations,
            'vector_source': 'instruction',
            'split_ratio': None  # Full train, full test (different distributions)
        },
        'nat_to_nat': {
            'train_loader': load_natural_activations,
            'test_loader': load_natural_activations,
            'vector_source': 'natural',
            'split_ratio': 0.8  # 80/20 split
        },
        'nat_to_inst': {
            'train_loader': load_natural_activations,
            'test_loader': load_instruction_activations,
            'vector_source': 'natural',
            'split_ratio': None  # Full train, full test (different distributions)
        }
    }

    print("="*80)
    print("COMPLETE 4×4 SWEEP: emotional_valence")
    print("="*80)
    print(f"Quadrants: {len(quadrants)}")
    print(f"Methods: {len(methods)}")
    print(f"Layers: 26")
    print(f"Total tests: {len(quadrants) * len(methods) * 26} = 416")
    print("="*80)

    # Store all results
    results = {q: {m: [] for m in methods} for q in quadrants.keys()}

    # Run all tests
    for quad_name, quad_config in quadrants.items():
        print(f"\n{'='*80}")
        print(f"QUADRANT: {quad_name}")
        print(f"{'='*80}")

        for layer in range(26):
            print(f"\nLayer {layer}:", end=" ")

            # Load test data
            test_pos, test_neg = quad_config['test_loader'](trait, layer)

            # If same-distribution, need to split
            if quad_config['split_ratio'] is not None:
                # Load train data (same as test source)
                train_pos, train_neg = quad_config['train_loader'](trait, layer)

                # Split 80/20
                n_pos_train = int(len(train_pos) * quad_config['split_ratio'])
                n_neg_train = int(len(train_neg) * quad_config['split_ratio'])

                # Use last 20% for testing
                test_pos = train_pos[n_pos_train:]
                test_neg = train_neg[n_neg_train:]

            for method in methods:
                # Load vector
                vector = load_vector(trait, method, layer, source=quad_config['vector_source'])

                # Test
                if vector is not None:
                    acc, correct_sign = test_vector(vector, test_pos, test_neg)
                    if acc is not None:
                        results[quad_name][method].append(acc * 100)
                        print(f"{method}:{acc*100:.1f}%", end=" ")
                    else:
                        results[quad_name][method].append(None)
                        print(f"{method}:ERR", end=" ")
                else:
                    results[quad_name][method].append(None)
                    print(f"{method}:N/A", end=" ")

            print()  # Newline after layer

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY: Best Accuracy Per Quadrant")
    print(f"{'='*80}\n")

    for quad_name in quadrants.keys():
        print(f"\n{quad_name.upper().replace('_', ' → ')}:")
        print("-" * 60)

        for method in methods:
            accs = [a for a in results[quad_name][method] if a is not None]
            if accs:
                best_acc = max(accs)
                best_layer = accs.index(best_acc)
                avg_acc = np.mean(accs)
                print(f"  {method:12s}: Best={best_acc:.1f}% (L{best_layer}), Avg={avg_acc:.1f}%")
            else:
                print(f"  {method:12s}: No data")

    # Save detailed results
    import json
    output_file = '/tmp/emotional_valence_full_4x4_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Complete results saved to {output_file}")
    print(f"{'='*80}\n")

    return results

if __name__ == '__main__':
    run_full_sweep()
