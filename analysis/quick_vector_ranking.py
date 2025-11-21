#!/usr/bin/env python3
"""
Quick ranking of existing vectors by separation score.
Simpler than full evaluation - just loads vectors and computes basic metrics.

Usage:
    python analysis/quick_vector_ranking.py --experiment gemma_2b_cognitive_nov21
"""

import torch
import json
from pathlib import Path
import fire
from typing import Dict, List
import pandas as pd

def load_vector_metadata(vector_path: Path) -> Dict:
    """Load metadata from vector file if it exists."""
    # Try to load accompanying metadata file
    metadata_path = vector_path.parent / f"{vector_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)

    # Try to extract from filename
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
            # Load vector
            vector = torch.load(vector_file, weights_only=True)

            # Get metadata
            metadata = load_vector_metadata(vector_file)

            # Compute basic stats
            result = {
                "file": vector_file.name,
                "method": metadata.get("method", "unknown"),
                "layer": metadata.get("layer", -1),
                "norm": float(vector.norm().item()) if hasattr(vector, 'norm') else 0,
                "size": vector.shape[0] if hasattr(vector, 'shape') else 0,
                "nonzero": int((vector != 0).sum().item()) if hasattr(vector, 'shape') else 0,
            }

            # Try to load separation score from metadata
            if "separation_score" in metadata:
                result["separation"] = metadata["separation_score"]
            elif "separation" in metadata:
                result["separation"] = metadata["separation"]

            results.append(result)

        except Exception as e:
            print(f"Error loading {vector_file}: {e}")
            continue

    return results

def main(experiment: str = "gemma_2b_cognitive_nov21",
         output: str = None,
         top_k: int = 3):
    """
    Quick ranking of existing vectors by basic metrics.

    Args:
        experiment: Experiment name
        output: Optional output CSV file
        top_k: Number of top vectors to show per trait
    """
    exp_dir = Path(f"experiments/{experiment}/extraction")

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

            # Analyze vectors for this trait
            trait_results = analyze_trait_vectors(trait_dir)

            # Add trait name to each result
            for result in trait_results:
                result["category"] = category_dir.name
                result["trait"] = trait_dir.name
                result["full_trait"] = trait_name
                all_results.append(result)

            # Show top vectors for this trait
            if trait_results:
                sorted_results = sorted(trait_results,
                                      key=lambda x: x.get("norm", 0),
                                      reverse=True)

                print(f"  Top {min(top_k, len(sorted_results))} vectors by norm:")
                for i, result in enumerate(sorted_results[:top_k]):
                    print(f"    {i+1}. {result['file']}: norm={result['norm']:.2f}")

    # Create summary DataFrame
    if all_results:
        df = pd.DataFrame(all_results)

        print("\n" + "="*80)
        print("SUMMARY - All Vectors")
        print("="*80)

        # Group by trait and find best vector
        best_per_trait = []
        for trait in df['full_trait'].unique():
            trait_df = df[df['full_trait'] == trait]
            # Get vector with highest norm (as proxy for strength)
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

        # Method statistics
        print("\n" + "="*80)
        print("METHOD STATISTICS")
        print("="*80)

        method_stats = df.groupby('method').agg({
            'norm': ['mean', 'std', 'max'],
            'file': 'count'
        }).round(2)
        print(method_stats)

        # Layer distribution
        print("\n" + "="*80)
        print("LAYER DISTRIBUTION")
        print("="*80)

        layer_counts = df[df['layer'] != -1]['layer'].value_counts().sort_index()
        print(f"Vectors per layer:")
        for layer, count in layer_counts.items():
            print(f"  Layer {int(layer):2d}: {'█' * (count // 2)} ({count})")

        # Save results if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"\nResults saved to {output_path}")

        # Vector availability matrix
        print("\n" + "="*80)
        print("VECTOR AVAILABILITY MATRIX")
        print("="*80)
        print("(Shows which method×layer combinations exist)")

        # Create pivot table
        for trait in df['full_trait'].unique():
            trait_df = df[df['full_trait'] == trait]
            if len(trait_df) > 3:  # Only show if multiple vectors
                print(f"\n{trait}:")
                pivot = trait_df.pivot_table(
                    index='method',
                    columns='layer',
                    values='norm',
                    aggfunc='first'
                )
                # Show only layers that have at least one vector
                pivot = pivot.dropna(axis=1, how='all')
                if not pivot.empty:
                    # Replace values with checkmarks for readability
                    pivot_display = pivot.applymap(lambda x: '✓' if pd.notna(x) else '')
                    print(pivot_display.to_string())

    return all_results

if __name__ == "__main__":
    fire.Fire(main)