#!/usr/bin/env python3
"""
Analyze high vs low prompts from monitoring results.
Generates text-based comparisons that can be shared without visual access.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

def load_monitoring_data(json_path: str) -> List[Dict]:
    """Load monitoring results from JSON."""
    with open(json_path) as f:
        return json.load(f)

def analyze_single_prompt(result: Dict) -> Dict:
    """Compute statistics for a single prompt's trait projections."""
    stats = {}

    for trait, scores in result.get("trait_scores", {}).items():
        if scores:
            stats[trait] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "max": max(scores),
                "min": min(scores),
                "first_10_avg": np.mean(scores[:10]) if len(scores) >= 10 else np.mean(scores),
                "last_10_avg": np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores),
                "peak_position": scores.index(max(scores, key=abs)),
                "trajectory": "increasing" if scores[-1] > scores[0] else "decreasing"
            }

    return stats

def compare_high_low_groups(results: List[Dict]) -> Dict:
    """Group by trait/polarity and compare."""
    comparisons = {}

    # Group results by trait and polarity
    grouped = {}
    for result in results:
        metadata = result.get("metadata", {})
        trait = metadata.get("trait", "unknown")
        polarity = metadata.get("polarity", metadata.get("type", "unknown"))

        key = f"{trait}_{polarity}"
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)

    # Compare high vs low for each trait
    traits = set(m.get("metadata", {}).get("trait", "unknown") for m in results)

    for trait in traits:
        if trait == "unknown":
            continue

        high_key = f"{trait}_high"
        low_key = f"{trait}_low"

        if high_key in grouped and low_key in grouped:
            high_results = grouped[high_key]
            low_results = grouped[low_key]

            # Compute aggregate statistics
            high_stats = aggregate_stats(high_results, trait)
            low_stats = aggregate_stats(low_results, trait)

            comparisons[trait] = {
                "high": high_stats,
                "low": low_stats,
                "contrast": high_stats["mean"] - low_stats["mean"],
                "separation_quality": abs(high_stats["mean"] - low_stats["mean"]) /
                                     (high_stats["std"] + low_stats["std"] + 0.001)
            }

    return comparisons

def aggregate_stats(results: List[Dict], target_trait: str) -> Dict:
    """Aggregate statistics across multiple prompts."""
    all_scores = []
    all_means = []

    for result in results:
        if target_trait in result.get("trait_scores", {}):
            scores = result["trait_scores"][target_trait]
            all_scores.extend(scores)
            all_means.append(np.mean(scores))

    if not all_scores:
        return {"mean": 0, "std": 0, "n_samples": 0}

    return {
        "mean": np.mean(all_scores),
        "std": np.std(all_scores),
        "mean_of_means": np.mean(all_means),
        "std_of_means": np.std(all_means),
        "n_samples": len(results),
        "n_tokens": len(all_scores)
    }

def identify_decision_points(result: Dict, trait: str, threshold: float = 0.5) -> List[int]:
    """Find token positions where trait crosses threshold."""
    scores = result.get("trait_scores", {}).get(trait, [])
    crossings = []

    for i in range(1, len(scores)):
        if abs(scores[i]) > threshold and abs(scores[i-1]) <= threshold:
            crossings.append(i)

    return crossings

def generate_text_report(comparisons: Dict, results: List[Dict]) -> str:
    """Generate human-readable report."""
    report = []
    report.append("="*70)
    report.append("HIGH VS LOW PROMPT COMPARISON REPORT")
    report.append("="*70)
    report.append("")

    # Sort traits by contrast strength
    sorted_traits = sorted(comparisons.keys(),
                          key=lambda t: abs(comparisons[t]["contrast"]),
                          reverse=True)

    for trait in sorted_traits:
        comp = comparisons[trait]
        report.append(f"\n{'='*70}")
        report.append(f"TRAIT: {trait.upper()}")
        report.append(f"{'='*70}")

        report.append(f"\nContrast Score: {comp['contrast']:.3f}")
        report.append(f"Separation Quality: {comp['separation_quality']:.3f}")

        report.append(f"\n📊 HIGH PROMPTS (n={comp['high']['n_samples']}):")
        report.append(f"   Mean projection: {comp['high']['mean']:.3f}")
        report.append(f"   Std deviation:   {comp['high']['std']:.3f}")
        report.append(f"   Tokens tracked:  {comp['high']['n_tokens']}")

        report.append(f"\n📊 LOW PROMPTS (n={comp['low']['n_samples']}):")
        report.append(f"   Mean projection: {comp['low']['mean']:.3f}")
        report.append(f"   Std deviation:   {comp['low']['std']:.3f}")
        report.append(f"   Tokens tracked:  {comp['low']['n_tokens']}")

        # Quality assessment
        report.append(f"\n✅ QUALITY ASSESSMENT:")
        if abs(comp['contrast']) > 1.0:
            report.append(f"   ★★★ EXCELLENT - Strong separation")
        elif abs(comp['contrast']) > 0.5:
            report.append(f"   ★★☆ GOOD - Clear separation")
        elif abs(comp['contrast']) > 0.2:
            report.append(f"   ★☆☆ MODERATE - Some separation")
        else:
            report.append(f"   ☆☆☆ POOR - Weak separation")

        # Check if polarity is as expected
        if comp['contrast'] > 0:
            report.append(f"   Direction: HIGH > LOW ✓ (as expected)")
        else:
            report.append(f"   Direction: HIGH < LOW ⚠️ (inverted?)")

    # Find problematic examples
    report.append(f"\n\n{'='*70}")
    report.append("POTENTIAL ISSUES")
    report.append("="*70)

    # Find prompts where target trait isn't strongest
    issues = find_contamination(results)
    if issues:
        report.append("\n⚠️ Prompts with trait contamination:")
        for issue in issues[:5]:  # Show first 5
            report.append(f"   - {issue}")
    else:
        report.append("\n✅ No major contamination detected")

    return "\n".join(report)

def find_contamination(results: List[Dict]) -> List[str]:
    """Find prompts where non-target traits dominate."""
    issues = []

    for result in results:
        metadata = result.get("metadata", {})
        target_trait = metadata.get("trait", "unknown")
        polarity = metadata.get("polarity", metadata.get("type", "unknown"))

        if target_trait == "unknown" or polarity not in ["high", "low"]:
            continue

        # Check if target trait has expected polarity
        trait_scores = result.get("trait_scores", {})
        if target_trait in trait_scores:
            target_mean = np.mean(trait_scores[target_trait])

            # Find strongest trait
            strongest_trait = None
            strongest_mean = 0
            for trait, scores in trait_scores.items():
                mean_score = abs(np.mean(scores))
                if mean_score > strongest_mean:
                    strongest_mean = mean_score
                    strongest_trait = trait

            # Check for issues
            if strongest_trait != target_trait:
                prompt_preview = result["prompt"][:50] + "..."
                issues.append(
                    f"'{prompt_preview}' - Target: {target_trait} ({target_mean:.2f}), "
                    f"but {strongest_trait} dominates ({strongest_mean:.2f})"
                )

            if polarity == "high" and target_mean < 0.2:
                prompt_preview = result["prompt"][:50] + "..."
                issues.append(
                    f"'{prompt_preview}' - HIGH {target_trait} but mean only {target_mean:.2f}"
                )

    return issues

def export_for_spreadsheet(comparisons: Dict, output_path: str = "high_low_comparison.csv"):
    """Export comparison data for spreadsheet analysis."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Trait", "High Mean", "High Std", "Low Mean", "Low Std",
            "Contrast", "Separation Quality", "Quality Rating"
        ])

        for trait, comp in comparisons.items():
            quality = "Excellent" if abs(comp['contrast']) > 1.0 else \
                     "Good" if abs(comp['contrast']) > 0.5 else \
                     "Moderate" if abs(comp['contrast']) > 0.2 else "Poor"

            writer.writerow([
                trait,
                f"{comp['high']['mean']:.3f}",
                f"{comp['high']['std']:.3f}",
                f"{comp['low']['mean']:.3f}",
                f"{comp['low']['std']:.3f}",
                f"{comp['contrast']:.3f}",
                f"{comp['separation_quality']:.3f}",
                quality
            ])

    print(f"CSV exported to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze high vs low prompt monitoring results")
    parser.add_argument(
        "--input",
        default="pertoken/results/gemma_2b_single_trait_teaching.json",
        help="Path to monitoring results JSON"
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export comparison to CSV"
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=0,
        help="Number of example prompts to show per trait"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading monitoring data from {args.input}...")
    results = load_monitoring_data(args.input)
    print(f"Loaded {len(results)} results")

    # Compare high vs low
    comparisons = compare_high_low_groups(results)

    # Generate report
    report = generate_text_report(comparisons, results)
    print(report)

    # Export if requested
    if args.export_csv:
        export_for_spreadsheet(comparisons)

    # Show specific examples if requested
    if args.show_examples > 0:
        print(f"\n\n{'='*70}")
        print(f"EXAMPLE PROMPTS")
        print(f"{'='*70}")

        for trait in list(comparisons.keys())[:3]:  # First 3 traits
            print(f"\n{trait.upper()}:")

            # Get high and low examples
            high_examples = [r for r in results
                           if r.get("metadata", {}).get("trait") == trait
                           and r.get("metadata", {}).get("polarity") == "high"]
            low_examples = [r for r in results
                          if r.get("metadata", {}).get("trait") == trait
                          and r.get("metadata", {}).get("polarity") == "low"]

            print(f"\n  HIGH examples:")
            for ex in high_examples[:args.show_examples]:
                mean_score = np.mean(ex["trait_scores"][trait])
                print(f"    [{mean_score:+.2f}] {ex['prompt'][:60]}...")

            print(f"\n  LOW examples:")
            for ex in low_examples[:args.show_examples]:
                mean_score = np.mean(ex["trait_scores"][trait])
                print(f"    [{mean_score:+.2f}] {ex['prompt'][:60]}...")

    return comparisons

if __name__ == "__main__":
    main()