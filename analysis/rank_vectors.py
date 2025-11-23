#!/usr/bin/env python3
"""
Rank extracted vectors for an experiment.

Two modes:
- Quick (default): Analyze vector files directly (norm, size, method stats)
- Detailed (--detailed): Read validation results to rank by accuracy

Usage:
    # Quick ranking by vector properties
    python analysis/rank_vectors.py --experiment {experiment_name}

    # Detailed ranking by validation accuracy (requires evaluate_on_validation.py first)
    python analysis/rank_vectors.py --experiment {experiment_name} --detailed
"""

import torch
import json
from pathlib import Path
import sys
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.paths import get as get_path

import pandas as pd
import fire


def load_vector_metadata(vector_path: Path) -> Dict:
    """Load metadata from vector file if it exists."""
    metadata_path = vector_path.parent / f"{vector_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)

    # Extract from filename
    parts = vector_path.stem.split('_')
    method = parts[0] if parts else "unknown"
    layer = int(parts[1].replace('layer', '')) if len(parts) > 1 and 'layer' in parts[1] else -1

    return {"method": method, "layer": layer}


def analyze_trait_vectors(trait_dir: Path) -> List[Dict]:
    """Analyze all vectors for a single trait."""
    vectors_dir = trait_dir / "vectors"
    if not vectors_dir.exists():
        return []

    results = []
    for vector_file in vectors_dir.glob("*.pt"):
        try:
            vector = torch.load(vector_file, weights_only=True)
            metadata = load_vector_metadata(vector_file)

            result = {
                "file": vector_file.name,
                "method": metadata.get("method", "unknown"),
                "layer": metadata.get("layer", -1),
                "norm": float(vector.norm().item()) if hasattr(vector, 'norm') else 0,
                "size": vector.shape[0] if hasattr(vector, 'shape') else 0,
                "nonzero": int((vector != 0).sum().item()) if hasattr(vector, 'shape') else 0,
            }

            if "separation_score" in metadata:
                result["separation"] = metadata["separation_score"]
            elif "separation" in metadata:
                result["separation"] = metadata["separation"]

            results.append(result)
        except Exception as e:
            print(f"Error loading {vector_file}: {e}")
            continue

    return results


def quick_ranking(experiment: str, output: str = None, top_k: int = 3):
    """Quick ranking by vector properties (norm, size)."""
    exp_dir = get_path('extraction.base', experiment=experiment)

    if not exp_dir.exists():
        print(f"Experiment directory not found: {exp_dir}")
        return

    all_results = []

    # Find all traits
    for category_dir in exp_dir.iterdir():
        if not category_dir.is_dir():
            continue

        for trait_dir in category_dir.iterdir():
            if not trait_dir.is_dir():
                continue

            trait_name = f"{category_dir.name}/{trait_dir.name}"
            vectors_dir = trait_dir / "vectors"

            if not vectors_dir.exists():
                continue

            print(f"\nAnalyzing {trait_name}...")

            trait_results = analyze_trait_vectors(trait_dir)

            for result in trait_results:
                result["category"] = category_dir.name
                result["trait"] = trait_dir.name
                result["full_trait"] = trait_name
                all_results.append(result)

            if trait_results:
                sorted_results = sorted(trait_results, key=lambda x: x.get("norm", 0), reverse=True)
                print(f"  Top {min(top_k, len(sorted_results))} vectors by norm:")
                for i, result in enumerate(sorted_results[:top_k]):
                    print(f"    {i+1}. {result['file']}: norm={result['norm']:.2f}")

    if all_results:
        df = pd.DataFrame(all_results)

        print("\n" + "="*80)
        print("SUMMARY - Best Vector Per Trait")
        print("="*80)

        best_per_trait = []
        for trait in df['full_trait'].unique():
            trait_df = df[df['full_trait'] == trait]
            best = trait_df.loc[trait_df['norm'].idxmax()]
            best_per_trait.append({
                'Trait': best['trait'][:20],
                'Category': best['category'][:15],
                'Best Method': best['method'],
                'Layer': int(best['layer']) if best['layer'] != -1 else 'unknown',
                'Norm': f"{best['norm']:.2f}",
                'File': best['file'][:30]
            })

        summary_df = pd.DataFrame(best_per_trait)
        print(summary_df.to_string(index=False))

        print("\n" + "="*80)
        print("METHOD STATISTICS")
        print("="*80)

        method_stats = df.groupby('method').agg({
            'norm': ['mean', 'std', 'max'],
            'file': 'count'
        }).round(2)
        print(method_stats)

        print("\n" + "="*80)
        print("LAYER DISTRIBUTION")
        print("="*80)

        layer_counts = df[df['layer'] != -1]['layer'].value_counts().sort_index()
        print(f"Vectors per layer:")
        for layer, count in layer_counts.items():
            print(f"  Layer {int(layer):2d}: {'█' * (count // 2)} ({count})")

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"\nResults saved to {output_path}")

    return all_results


def detailed_ranking(experiment: str, top_n: int = 20):
    """Detailed ranking by validation accuracy (requires validation results)."""
    print(f"Analyzing experiment: {experiment}")

    results_path = get_path('validation.evaluation', experiment=experiment)

    if not results_path.exists():
        print(f"Validation results file not found at '{results_path}'")
        print(f"Run evaluation first:")
        print(f"  python analysis/evaluate_on_validation.py --experiment {experiment}")
        return

    print(f"Loading results from {results_path}...")
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)

        if 'all_results' not in data or not isinstance(data['all_results'], list):
            print("JSON file is missing the 'all_results' list.")
            return

        all_results = data['all_results']
        print(f"  Found {len(all_results)} total vector evaluations.")

    except json.JSONDecodeError:
        print(f"Could not decode JSON from '{results_path}'.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    valid_results = [r for r in all_results if r.get('val_accuracy') is not None]

    if not valid_results:
        print("No results with 'val_accuracy' found.")
        return

    sorted_results = sorted(valid_results, key=lambda x: x['val_accuracy'], reverse=True)

    print("\n" + "="*80)
    print(f"Top {top_n} Vectors by Validation Accuracy")
    print("="*80)
    print(f"{'Rank':<5} | {'Trait':<30} | {'Layer':<5} | {'Method':<10} | {'Accuracy':<10} | {'Polarity OK?':<12}")
    print("-"*80)

    for i, res in enumerate(sorted_results[:top_n]):
        rank = i + 1
        trait = res.get('trait', 'N/A')
        layer = res.get('layer', 'N/A')
        method = res.get('method', 'N/A')
        accuracy = res.get('val_accuracy', 0) * 100
        polarity_ok = 'yes' if res.get('polarity_correct') else 'no'

        print(f"{rank:<5} | {trait:<30} | {layer:<5} | {method:<10} | {accuracy:>8.2f}% | {polarity_ok:^12}")

    print("="*80)


def main(experiment: str,
         detailed: bool = False,
         output: str = None,
         top_k: int = 3,
         top_n: int = 20):
    """
    Rank vectors for an experiment.

    Args:
        experiment: Experiment name
        detailed: Use detailed mode (validation accuracy) instead of quick mode
        output: Optional output CSV file (quick mode only)
        top_k: Top vectors per trait (quick mode)
        top_n: Top vectors overall (detailed mode)
    """
    if detailed:
        detailed_ranking(experiment, top_n)
    else:
        quick_ranking(experiment, output, top_k)


if __name__ == "__main__":
    fire.Fire(main)
