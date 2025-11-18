#!/usr/bin/env python3
"""
Generic Cross-Distribution Analysis Script

Runs 4×4 cross-distribution sweep for any trait with both instruction and natural data.
Tests all 4 extraction methods (mean_diff, probe, ICA, gradient) across all 26 layers.

Usage:
    python scripts/run_cross_distribution.py --trait uncertainty_calibration
    python scripts/run_cross_distribution.py --trait refusal
"""
import sys
import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def load_instruction_activations(trait, layer):
    """Load instruction-based activations for a specific layer"""
    base = Path(f'experiments/gemma_2b_cognitive_nov20/{trait}/extraction/activations')

    # Try old format first (all_layers.pt)
    all_layers_file = base / 'all_layers.pt'
    if all_layers_file.exists():
        all_layers = torch.load(all_layers_file)  # [n_examples, 26, 2304]

        # Get metadata for pos/neg split
        metadata_file = base / 'metadata.json'
        if metadata_file.exists():
            metadata = json.load(open(metadata_file))
            n_pos = metadata.get('n_positive', metadata.get('n_examples_pos', 100))
            n_neg = metadata.get('n_negative', metadata.get('n_examples_neg', 101))
        else:
            # Assume 100/100 split
            n_pos = 100
            n_neg = 100

        pos_acts = all_layers[:n_pos, layer, :]
        neg_acts = all_layers[n_pos:n_pos+n_neg, layer, :]
        return pos_acts, neg_acts

    # Try new format (separate pos/neg files)
    pos_file = base / f'pos_layer{layer}.pt'
    neg_file = base / f'neg_layer{layer}.pt'
    if pos_file.exists() and neg_file.exists():
        pos_acts = torch.load(pos_file)
        neg_acts = torch.load(neg_file)
        return pos_acts, neg_acts

    raise FileNotFoundError(f"No activation files found for {trait} at layer {layer}")


def load_natural_activations(trait_base, layer):
    """Load natural elicitation activations"""
    trait_natural = f"{trait_base}_natural"
    base = Path(f'experiments/gemma_2b_cognitive_nov20/{trait_natural}/extraction/activations')

    pos_file = base / f'pos_layer{layer}.pt'
    neg_file = base / f'neg_layer{layer}.pt'

    if not pos_file.exists() or not neg_file.exists():
        raise FileNotFoundError(f"Natural activations not found for {trait_natural}")

    pos_acts = torch.load(pos_file)
    neg_acts = torch.load(neg_file)
    return pos_acts, neg_acts


def load_vector(trait, method, layer, source='instruction'):
    """Load extracted vector"""
    if source == 'natural':
        trait = f"{trait}_natural"

    vectors_dir = Path(f'experiments/gemma_2b_cognitive_nov20/{trait}/extraction/vectors')
    vector_file = vectors_dir / f'{method}_layer{layer}.pt'

    if not vector_file.exists():
        return None

    return torch.load(vector_file)


def test_vector(vector, pos_acts, neg_acts):
    """
    Test vector on activations.
    Returns: (accuracy, separation, pos_mean, neg_mean, pos_std, neg_std)
    """
    if vector is None:
        return None

    # Convert to float32 if needed
    vector = vector.to(torch.float32)
    pos_acts = pos_acts.to(torch.float32)
    neg_acts = neg_acts.to(torch.float32)

    # Project
    pos_proj = pos_acts @ vector
    neg_proj = neg_acts @ vector

    # Check polarity (positive examples should have higher projection)
    correct_polarity = (pos_proj.mean() > neg_proj.mean())

    if correct_polarity:
        correct = (pos_proj > 0).sum() + (neg_proj < 0).sum()
        pos_mean = pos_proj.mean().item()
        neg_mean = neg_proj.mean().item()
    else:
        # Flip sign if polarity is wrong
        correct = (pos_proj < 0).sum() + (neg_proj > 0).sum()
        pos_mean = -pos_proj.mean().item()
        neg_mean = -neg_proj.mean().item()

    total = len(pos_acts) + len(neg_acts)
    accuracy = correct.item() / total
    separation = abs(pos_mean - neg_mean)

    pos_std = pos_proj.std().item()
    neg_std = neg_proj.std().item()

    return {
        'accuracy': accuracy,
        'separation': separation,
        'pos_mean': pos_mean,
        'neg_mean': neg_mean,
        'pos_std': pos_std,
        'neg_std': neg_std
    }


def run_quadrant(trait, quadrant_name, vector_source, test_source):
    """Run one quadrant (train source → test source)"""
    methods = ['mean_diff', 'probe', 'ica', 'gradient']
    n_layers = 26

    results = {
        'description': f"{vector_source.title()} → {test_source.title()}",
        'vector_source': trait if vector_source == 'instruction' else f"{trait}_natural",
        'test_source': test_source,
        'methods': {}
    }

    for method in methods:
        method_results = {
            'all_layers': []
        }

        for layer in tqdm(range(n_layers), desc=f"{quadrant_name} - {method}", leave=False):
            try:
                # Load vector
                vector = load_vector(trait, method, layer, source=vector_source)
                if vector is None:
                    continue

                # Load test activations
                if test_source == 'instruction':
                    pos_acts, neg_acts = load_instruction_activations(trait, layer)
                else:  # natural
                    pos_acts, neg_acts = load_natural_activations(trait, layer)

                # Test
                metrics = test_vector(vector, pos_acts, neg_acts)
                if metrics:
                    metrics['layer'] = layer
                    method_results['all_layers'].append(metrics)

            except Exception as e:
                print(f"Warning: Failed at {method} layer {layer}: {e}")
                continue

        # Find best layer
        if method_results['all_layers']:
            best = max(method_results['all_layers'], key=lambda x: x['accuracy'])
            method_results['best_layer'] = best['layer']
            method_results['best_accuracy'] = best['accuracy']
            method_results['best_separation'] = best['separation']
            method_results['avg_accuracy'] = sum(x['accuracy'] for x in method_results['all_layers']) / len(method_results['all_layers'])

        results['methods'][method] = method_results

    return results


def run_full_4x4(trait):
    """Run complete 4×4 cross-distribution analysis"""
    print(f"\n{'='*60}")
    print(f"Cross-Distribution Analysis: {trait}")
    print(f"{'='*60}\n")

    # Define quadrants
    quadrants = {
        'inst_inst': {
            'vector_source': 'instruction',
            'test_source': 'instruction'
        },
        'inst_nat': {
            'vector_source': 'instruction',
            'test_source': 'natural'
        },
        'nat_inst': {
            'vector_source': 'natural',
            'test_source': 'instruction'
        },
        'nat_nat': {
            'vector_source': 'natural',
            'test_source': 'natural'
        }
    }

    results = {
        'experiment': 'gemma_2b_cognitive_nov20',
        'trait': trait,
        'n_layers': 26,
        'methods': ['mean_diff', 'probe', 'ica', 'gradient'],
        'quadrants': {}
    }

    for quadrant_name, config in quadrants.items():
        print(f"\n{quadrant_name}: {config['vector_source']} → {config['test_source']}")
        try:
            quadrant_results = run_quadrant(
                trait,
                quadrant_name,
                config['vector_source'],
                config['test_source']
            )
            results['quadrants'][quadrant_name] = quadrant_results
        except Exception as e:
            print(f"ERROR in {quadrant_name}: {e}")
            continue

    # Save results
    output_dir = Path('results/cross_distribution_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{trait}_full_4x4_results.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")

    # Print summary
    print("Summary:")
    for quadrant_name, quadrant_data in results['quadrants'].items():
        print(f"\n{quadrant_name}:")
        for method, method_data in quadrant_data['methods'].items():
            if 'best_accuracy' in method_data:
                print(f"  {method:12s}: {method_data['best_accuracy']*100:5.1f}% @ layer {method_data['best_layer']}")


def main():
    parser = argparse.ArgumentParser(description='Run cross-distribution analysis')
    parser.add_argument('--trait', type=str, required=True, help='Trait name (e.g., uncertainty_calibration)')
    args = parser.parse_args()

    run_full_4x4(args.trait)


if __name__ == '__main__':
    main()
