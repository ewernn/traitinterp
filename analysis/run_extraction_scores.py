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


def load_instruction_activations(experiment, trait_path, layer):
    """Load instruction-based activations for a specific layer

    Args:
        experiment: Experiment name (e.g., gemma_2b_cognitive_nov20)
        trait_path: Full trait path including category (e.g., behavioral/refusal)
        layer: Layer number (0-25)
    """
    base = Path(f'experiments/{experiment}/extraction/{trait_path}/extraction/activations')

    if not base.exists():
        raise FileNotFoundError(
            f"Activations directory not found: {base}\n"
            f"Expected: experiments/{experiment}/extraction/{{category}}/{{trait}}/extraction/activations/"
        )

    # Use separate pos/neg files
    pos_file = base / f'pos_layer{layer}.pt'
    neg_file = base / f'neg_layer{layer}.pt'

    if not (pos_file.exists() and neg_file.exists()):
        raise FileNotFoundError(f"Activation files not found for {trait_path} at layer {layer}")

    pos_acts = torch.load(pos_file)
    neg_acts = torch.load(neg_file)
    return pos_acts, neg_acts


def load_vector(experiment, trait_path, method, layer):
    """Load extracted vector

    Args:
        experiment: Experiment name (e.g., gemma_2b_cognitive_nov20)
        trait_path: Full trait path including category (e.g., behavioral/refusal)
        method: Extraction method (mean_diff, probe, ica, gradient)
        layer: Layer number (0-25)
    """
    vectors_dir = Path(f'experiments/{experiment}/extraction/{trait_path}/extraction/vectors')
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


def run_extraction_scores(experiment, trait_path):
    """Run extraction score analysis for a single trait

    Args:
        experiment: Experiment name (e.g., gemma_2b_cognitive_nov20)
        trait_path: Full trait path including category (e.g., behavioral/refusal)
    """
    print(f"\n{'='*60}")
    print(f"Extraction Scores: {trait_path}")
    print(f"Experiment: {experiment}")
    print(f"{'='*60}\n")

    # Verify structure
    exp_dir = Path(f'experiments/{experiment}/extraction/{trait_path}')
    if not exp_dir.exists():
        raise FileNotFoundError(
            f"Trait directory not found: {exp_dir}\n"
            f"Expected: experiments/{experiment}/extraction/{{category}}/{{trait}}/"
        )

    methods = ['mean_diff', 'probe', 'ica', 'gradient']
    n_layers = 26

    # Extract trait name without category for output filename
    trait_name = trait_path.split('/')[-1]

    results = {
        'experiment': experiment,
        'trait': trait_path,
        'n_layers': 26,
        'methods': ['mean_diff', 'probe', 'ica', 'gradient'],
        'quadrants': {
            'inst_inst': {
                'description': 'Instruction → Instruction',
                'vector_source': trait_path,
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
                vector = load_vector(experiment, trait_path, method, layer)
                if vector is None:
                    continue

                # Load test activations
                pos_acts, neg_acts = load_instruction_activations(experiment, trait_path, layer)

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

    # Save results to experiment's validation directory
    output_dir = Path(f'experiments/{experiment}/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{trait_name}_full_4x4_results.json'

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
    parser = argparse.ArgumentParser(
        description='Generate extraction scores for traits',
        epilog='Example: python analysis/run_extraction_scores.py --trait behavioral/refusal'
    )
    parser.add_argument('--experiment', type=str, default='gemma_2b_cognitive_nov20',
                       help='Experiment name (default: gemma_2b_cognitive_nov20)')
    parser.add_argument('--trait', type=str,
                       help='Trait path: category/trait_name (e.g., behavioral/refusal)')
    parser.add_argument('--all', action='store_true',
                       help='Process all traits in experiment')
    args = parser.parse_args()

    # Validate trait format if provided
    if args.trait and '/' not in args.trait:
        raise ValueError(
            f"Trait must include category: got '{args.trait}'\n"
            f"Expected format: category/trait_name (e.g., behavioral/refusal)"
        )

    if args.all:
        # Auto-discover all traits in experiment
        from analysis.cross_distribution_scanner import scan_experiment
        exp_path = Path(f'experiments/{args.experiment}')
        exp_data = scan_experiment(exp_path)

        traits = [t['name'] for t in exp_data['traits']]
        print(f"Processing {len(traits)} traits from {args.experiment}...")

        for trait_path in traits:
            try:
                run_extraction_scores(args.experiment, trait_path)
            except Exception as e:
                print(f"ERROR processing {trait_path}: {e}")
                continue
    elif args.trait:
        run_extraction_scores(args.experiment, args.trait)
    else:
        print("Usage:")
        print("  python analysis/run_extraction_scores.py --trait behavioral/refusal")
        print("  python analysis/run_extraction_scores.py --all")
        print("\nNote: Use --trait category/trait_name format")


if __name__ == '__main__':
    main()
