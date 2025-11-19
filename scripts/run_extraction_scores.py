#!/usr/bin/env python3
"""
Generate extraction scores for all traits by testing on training data.

This runs the inst_inst quadrant only (vectors tested on same data used for extraction).
Gives us extraction performance metrics for all traits.
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


def load_vector(trait, method, layer):
    """Load extracted vector"""
    vectors_dir = Path(f'experiments/gemma_2b_cognitive_nov20/{trait}/extraction/vectors')
    vector_file = vectors_dir / f'{method}_layer{layer}.pt'

    if not vector_file.exists():
        return None

    return torch.load(vector_file)


def test_vector(vector, pos_acts, neg_acts):
    """
    Test vector on activations.
    Returns: dict with accuracy, separation, pos/neg mean/std
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


def run_extraction_scores(trait):
    """Run extraction score analysis for a single trait"""
    print(f"\n{'='*60}")
    print(f"Extraction Scores: {trait}")
    print(f"{'='*60}\n")

    methods = ['mean_diff', 'probe', 'ica', 'gradient']
    n_layers = 26

    results = {
        'experiment': 'gemma_2b_cognitive_nov20',
        'trait': trait,
        'n_layers': 26,
        'methods': ['mean_diff', 'probe', 'ica', 'gradient'],
        'quadrants': {
            'inst_inst': {
                'description': 'Instruction → Instruction',
                'vector_source': trait,
                'test_source': 'instruction',
                'methods': {}
            }
        }
    }

    for method in methods:
        method_results = {
            'all_layers': []
        }

        for layer in tqdm(range(n_layers), desc=f"{method}", leave=False):
            try:
                # Load vector
                vector = load_vector(trait, method, layer)
                if vector is None:
                    continue

                # Load test activations
                pos_acts, neg_acts = load_instruction_activations(trait, layer)

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

        results['quadrants']['inst_inst']['methods'][method] = method_results

    # Save results
    output_dir = Path('results/cross_distribution_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{trait}_full_4x4_results.json'

    # If file exists, merge with existing data (preserve other quadrants)
    if output_file.exists():
        existing = json.load(open(output_file))
        existing['quadrants']['inst_inst'] = results['quadrants']['inst_inst']
        results = existing

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")

    # Print summary
    print("Summary:")
    for method, method_data in results['quadrants']['inst_inst']['methods'].items():
        if 'best_accuracy' in method_data:
            print(f"  {method:12s}: {method_data['best_accuracy']*100:5.1f}% @ layer {method_data['best_layer']}")


def main():
    parser = argparse.ArgumentParser(description='Generate extraction scores for traits')
    parser.add_argument('--trait', type=str, help='Single trait to process')
    parser.add_argument('--all', action='store_true', help='Process all untested traits')
    args = parser.parse_args()

    # Traits that need testing
    untested_traits = [
        'abstract_concrete', 'commitment_strength', 'context_adherence',
        'convergent_divergent', 'instruction_boundary', 'local_global',
        'paranoia_trust', 'power_dynamics', 'retrieval_construction',
        'serial_parallel', 'sycophancy', 'temporal_focus'
    ]

    if args.all:
        print(f"Processing {len(untested_traits)} traits...")
        for trait in untested_traits:
            try:
                run_extraction_scores(trait)
            except Exception as e:
                print(f"ERROR processing {trait}: {e}")
                continue
    elif args.trait:
        run_extraction_scores(args.trait)
    else:
        print("Usage:")
        print("  python scripts/run_extraction_scores.py --trait refusal")
        print("  python scripts/run_extraction_scores.py --all")
        print(f"\nUntested traits ({len(untested_traits)}):")
        for t in untested_traits:
            print(f"  - {t}")


if __name__ == '__main__':
    main()
