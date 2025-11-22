#!/usr/bin/env python3
"""
Rank all extracted vectors for an experiment based on validation accuracy.

This script loads the results from `evaluate_on_validation.py` and provides a
ranked list of the best-performing vectors (trait, layer, method combinations).

Usage:
    python analysis/rank_vectors.py --experiment gemma_2b_cognitive_nov21 --top-n 20
"""

import json
import argparse
from pathlib import Path
import sys

# Ensure the project root is in the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.paths import get as getpath

def rank_vectors(experiment: str, top_n: int):
    """
    Loads validation results, ranks vectors by accuracy, and prints a summary.

    Args:
        experiment: The name of the experiment directory.
        top_n: The number of top vectors to display.
    """
    print(f"🔬 Analyzing experiment: {experiment}")

    # Construct the path to the validation results file
    try:
        results_path = getpath('validation.evaluation', experiment=experiment)
    except KeyError:
        print(f"❌ Error: Could not find path key for 'validation.evaluation'. Check config/paths.yaml.")
        return

    if not results_path.exists():
        print(f"❌ Error: Validation results file not found at '{results_path}'")
        print(f"Please run the evaluation script first:")
        print(f"  python analysis/evaluate_on_validation.py --experiment {experiment}")
        return

    # Load the JSON data
    print(f"📄 Loading results from {results_path}...")
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        if 'all_results' not in data or not isinstance(data['all_results'], list):
            print("❌ Error: JSON file is missing the 'all_results' list.")
            return

        all_results = data['all_results']
        print(f"  Found {len(all_results)} total vector evaluations.")

    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode JSON from '{results_path}'. The file might be corrupted.")
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred while reading the file: {e}")
        return

    # Filter out results where accuracy is None and sort
    valid_results = [r for r in all_results if r.get('val_accuracy') is not None]
    
    if not valid_results:
        print("🤔 No results with 'val_accuracy' found to rank.")
        return
        
    sorted_results = sorted(valid_results, key=lambda x: x['val_accuracy'], reverse=True)

    # Print the ranked table
    print("\n" + "="*80)
    print(f"🏆 Top {top_n} Vectors Ranked by Validation Accuracy")
    print("="*80)
    print(f"{ 'Rank':<5} | {'Trait':<30} | {'Layer':<5} | {'Method':<10} | {'Accuracy':<10} | {'Polarity OK?':<12}")
    print("-"*80)

    for i, res in enumerate(sorted_results[:top_n]):
        rank = i + 1
        trait = res.get('trait', 'N/A')
        layer = res.get('layer', 'N/A')
        method = res.get('method', 'N/A')
        accuracy = res.get('val_accuracy', 0) * 100
        polarity_ok = '✅' if res.get('polarity_correct') else '❌'

        print(f"{rank:<5} | {trait:<30} | {layer:<5} | {method:<10} | {accuracy:>8.2f}% | {polarity_ok:^12}")

    print("="*80)

    # Also, find the best layer and method for a specific trait, e.g., 'refusal'
    # This is an example of how you can programmatically find the best vector for a deep dive
    target_trait = 'behavioral_tendency/refusal' 
    if any(r['trait'] == target_trait for r in sorted_results):
        best_for_trait = max(
            [r for r in valid_results if r['trait'] == target_trait],
            key=lambda x: x['val_accuracy']
        )
        print("\n💡 Example: Best vector for a deep dive on 'refusal'")
        print(f"   - Trait: {best_for_trait['trait']}")
        print(f"   - Layer: {best_for_trait['layer']}")
        print(f"   - Method: {best_for_trait['method']}")
        print(f"   - Accuracy: {best_for_trait['val_accuracy']*100:.2f}%")
        print("\nYou can use this layer number for a Tier 3 single-layer capture.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rank extracted vectors by validation accuracy."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Name of the experiment to analyze (e.g., 'gemma_2b_cognitive_nov21')."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top-ranked vectors to display."
    )
    args = parser.parse_args()
    rank_vectors(args.experiment, args.top_n)
