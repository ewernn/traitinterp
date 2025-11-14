#!/usr/bin/env python3
"""
Export per-token trajectories for specific prompts in shareable text format.
"""

import json
import numpy as np
from typing import Dict, List
import argparse

def load_monitoring_data(json_path: str) -> List[Dict]:
    """Load monitoring results from JSON."""
    with open(json_path) as f:
        return json.load(f)

def format_trajectory(result: Dict, trait: str, width: int = 70) -> str:
    """Format a single trajectory as ASCII art."""
    scores = result.get("trait_scores", {}).get(trait, [])
    tokens = result.get("tokens", [])

    if not scores:
        return "No data"

    # Normalize to 0-width range for visualization
    min_score = min(scores)
    max_score = max(scores)
    range_score = max_score - min_score if max_score != min_score else 1

    lines = []
    lines.append(f"Range: [{min_score:.1f}, {max_score:.1f}]")
    lines.append("-" * width)

    for i, (token, score) in enumerate(zip(tokens[:50], scores[:50])):  # Limit to 50 tokens
        # Calculate position on ASCII chart
        normalized = (score - min_score) / range_score
        pos = int(normalized * (width - 1))

        # Create line with marker
        line = [' '] * width
        line[pos] = '█'

        # Add zero line if in range
        if min_score < 0 < max_score:
            zero_pos = int((0 - min_score) / range_score * (width - 1))
            if line[zero_pos] == ' ':
                line[zero_pos] = '|'

        # Format token (truncate if needed)
        token_str = token[:15].ljust(15)
        lines.append(f"{i:3d} {token_str} {''.join(line)} {score:+6.2f}")

    if len(scores) > 50:
        lines.append(f"... ({len(scores) - 50} more tokens)")

    return '\n'.join(lines)

def find_interesting_examples(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Find examples that demonstrate each pattern well."""
    interesting = {
        "strong_refusal": [],
        "weak_refusal": [],
        "high_uncertainty": [],
        "low_uncertainty": [],
        "increasing_trait": [],
        "decreasing_trait": [],
        "spike_pattern": [],
        "stable_pattern": []
    }

    for result in results:
        metadata = result.get("metadata", {})
        trait = metadata.get("trait", "unknown")

        if trait in result.get("trait_scores", {}):
            scores = result["trait_scores"][trait]
            if len(scores) < 10:
                continue

            mean_score = np.mean(scores)
            std_score = np.std(scores)
            first_half = np.mean(scores[:len(scores)//2])
            second_half = np.mean(scores[len(scores)//2:])

            # Categorize
            if trait == "refusal":
                if mean_score > 70:
                    interesting["strong_refusal"].append(result)
                elif mean_score < 50:
                    interesting["weak_refusal"].append(result)

            if trait == "uncertainty":
                if mean_score > 20:
                    interesting["high_uncertainty"].append(result)
                elif mean_score < -10:
                    interesting["low_uncertainty"].append(result)

            # Trajectory patterns
            if second_half - first_half > 20:
                interesting["increasing_trait"].append(result)
            elif first_half - second_half > 20:
                interesting["decreasing_trait"].append(result)

            # Variability patterns
            if std_score > 30:
                interesting["spike_pattern"].append(result)
            elif std_score < 10:
                interesting["stable_pattern"].append(result)

    # Keep only top examples
    for key in interesting:
        interesting[key] = interesting[key][:3]

    return interesting

def export_example_set(results: List[Dict], output_path: str = "trajectory_examples.txt"):
    """Export a curated set of trajectory examples."""
    interesting = find_interesting_examples(results)

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PER-TOKEN TRAJECTORY EXAMPLES\n")
        f.write("="*80 + "\n\n")

        for pattern_name, examples in interesting.items():
            if not examples:
                continue

            f.write(f"\n{'-'*80}\n")
            f.write(f"PATTERN: {pattern_name.replace('_', ' ').upper()}\n")
            f.write(f"{'-'*80}\n\n")

            for i, example in enumerate(examples, 1):
                metadata = example.get("metadata", {})
                trait = metadata.get("trait", "unknown")

                f.write(f"Example {i}: {trait} trait\n")
                f.write(f"Prompt: {example['prompt'][:100]}...\n\n")

                trajectory = format_trajectory(example, trait)
                f.write(trajectory + "\n\n")

    print(f"Exported examples to {output_path}")
    return output_path

def export_specific_prompt(results: List[Dict], prompt_substring: str, output_path: str = None):
    """Export trajectory for a specific prompt containing substring."""
    matches = []
    for result in results:
        if prompt_substring.lower() in result["prompt"].lower():
            matches.append(result)

    if not matches:
        print(f"No prompts found containing: {prompt_substring}")
        return None

    if len(matches) > 1:
        print(f"Found {len(matches)} matches. Using first one.")

    result = matches[0]
    metadata = result.get("metadata", {})

    if output_path is None:
        output_path = f"trajectory_{prompt_substring[:20].replace(' ', '_')}.txt"

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PER-TOKEN TRAJECTORY EXPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Prompt: {result['prompt']}\n")
        f.write(f"Metadata: {metadata}\n\n")

        # Export trajectory for each trait
        for trait in result.get("trait_scores", {}).keys():
            f.write(f"\n{'-'*80}\n")
            f.write(f"TRAIT: {trait.upper()}\n")
            f.write(f"{'-'*80}\n\n")

            trajectory = format_trajectory(result, trait)
            f.write(trajectory + "\n")

        # Add statistics
        f.write(f"\n{'-'*80}\n")
        f.write("STATISTICS\n")
        f.write(f"{'-'*80}\n\n")

        for trait, scores in result.get("trait_scores", {}).items():
            if scores:
                f.write(f"{trait:15s}: mean={np.mean(scores):+6.2f}, "
                       f"std={np.std(scores):6.2f}, "
                       f"min={min(scores):+6.2f}, "
                       f"max={max(scores):+6.2f}\n")

    print(f"Exported trajectory to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Export per-token trajectories")
    parser.add_argument(
        "--input",
        default="pertoken/results/gemma_2b_single_trait_teaching.json",
        help="Path to monitoring results JSON"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Export specific prompt containing this substring"
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Export curated example set"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path"
    )

    args = parser.parse_args()

    # Load data
    results = load_monitoring_data(args.input)
    print(f"Loaded {len(results)} results")

    # Export based on mode
    if args.prompt:
        export_specific_prompt(results, args.prompt, args.output)
    elif args.examples:
        export_example_set(results, args.output or "trajectory_examples.txt")
    else:
        # Default: export interesting examples
        export_example_set(results, args.output or "trajectory_examples.txt")

if __name__ == "__main__":
    main()