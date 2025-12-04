#!/usr/bin/env python3
"""
Rebuild results.json from response files.

This script scans all response JSON files and reconstructs results.json.
Use this to recover from race condition data loss when running parallel sweeps.

Input:
    - experiments/{experiment}/steering/{trait}/responses/*.json

Output:
    - experiments/{experiment}/steering/{trait}/results_rebuilt.json (new file)
    - Comparison with existing results.json (if present)

Usage:
    python analysis/steering/rebuild_results.py --experiment gemma-2-2b-it --trait epistemic/optimism
    python analysis/steering/rebuild_results.py --experiment gemma-2-2b-it --trait epistemic/optimism --apply
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from statistics import mean, stdev

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.paths import get as get_path


def parse_filename(filename: str) -> Optional[Dict]:
    """
    Parse response filename to extract config.

    Examples:
        L0_probe_c13p6_2025-12-04_05-55-16.467598.json
        L10_11_12_probe_probe_probe_c100p0_100p0_100p0_2025-12-03_...json
        L0_1_2_mean_diff_mean_diff_mean_diff_c5p0_5p0_5p0_2025-12-04_...json
    """
    # Remove .json extension
    name = filename.replace('.json', '')

    # Extract timestamp (at the end, format: YYYY-MM-DD_HH-MM-SS.microseconds)
    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d+)$', name)
    if not timestamp_match:
        return None
    timestamp_str = timestamp_match.group(1)
    # Convert to ISO format
    timestamp = timestamp_str.replace('_', 'T').replace('-', ':', 2)  # Fix time separators
    # Actually need more careful conversion
    parts = timestamp_str.split('_')
    if len(parts) == 2:
        date_part = parts[0]
        time_part = parts[1].replace('-', ':')
        timestamp = f"{date_part}T{time_part}"

    # Remove timestamp from name
    name = name[:timestamp_match.start()].rstrip('_')

    # Parse layers (starts with L, numbers separated by _)
    layers_match = re.match(r'L([\d_]+)_', name)
    if not layers_match:
        return None
    layers_str = layers_match.group(1)
    layers = [int(x) for x in layers_str.split('_')]

    # Remove layers prefix
    name = name[layers_match.end():]

    # Find coefficients (starts with c, format like c100p0 or c100p0_100p0_...)
    coef_match = re.search(r'_c([\dp_]+)$', name)
    if not coef_match:
        return None
    coef_str = coef_match.group(1)
    # Split by _ and convert pN to .N
    coef_parts = coef_str.split('_')
    coefficients = []
    for part in coef_parts:
        # Convert 13p6 to 13.6
        coef_val = float(part.replace('p', '.'))
        coefficients.append(coef_val)

    # Extract methods (between layers and coefficients)
    methods_str = name[:coef_match.start()]
    methods = methods_str.split('_')
    methods = [m for m in methods if m]  # Remove empty strings

    # Validate: should have same number of layers, methods, and coefficients
    if len(layers) != len(methods) or len(layers) != len(coefficients):
        # This can happen with different naming conventions
        # Try to handle single-layer case
        if len(layers) == 1 and len(methods) >= 1 and len(coefficients) >= 1:
            methods = [methods[0]]
            coefficients = [coefficients[0]]
        else:
            print(f"  Warning: Could not parse {filename} - mismatched counts")
            return None

    return {
        'layers': layers,
        'methods': methods,
        'coefficients': coefficients,
        'component': 'residual',  # Assume residual (most common)
        'timestamp': timestamp,
    }


def compute_stats(responses: List[Dict]) -> Dict:
    """Compute aggregate statistics from responses."""
    trait_scores = [r['trait_score'] for r in responses if r.get('trait_score') is not None]
    coherence_scores = [r['coherence_score'] for r in responses if r.get('coherence_score') is not None]

    result = {
        'n': len(trait_scores),
    }

    if trait_scores:
        result['trait_mean'] = mean(trait_scores)
        if len(trait_scores) > 1:
            result['trait_std'] = stdev(trait_scores)

    if coherence_scores:
        result['coherence_mean'] = mean(coherence_scores)

    return result


def rebuild_results(experiment: str, trait: str) -> Tuple[Dict, int, int]:
    """
    Rebuild results.json from response files.

    Returns:
        (rebuilt_data, n_files_processed, n_files_skipped)
    """
    responses_dir = get_path('steering.responses', experiment=experiment, trait=trait)

    if not responses_dir.exists():
        raise FileNotFoundError(f"Responses directory not found: {responses_dir}")

    # Find all response files
    response_files = sorted(responses_dir.glob('*.json'))
    print(f"Found {len(response_files)} response files")

    runs = []
    n_processed = 0
    n_skipped = 0

    for filepath in response_files:
        # Parse filename
        config = parse_filename(filepath.name)
        if config is None:
            print(f"  Skipping: {filepath.name} (could not parse)")
            n_skipped += 1
            continue

        # Load responses
        try:
            with open(filepath) as f:
                responses = json.load(f)
        except json.JSONDecodeError:
            print(f"  Skipping: {filepath.name} (invalid JSON)")
            n_skipped += 1
            continue

        # Compute stats
        result = compute_stats(responses)

        if result['n'] == 0:
            print(f"  Skipping: {filepath.name} (no valid responses)")
            n_skipped += 1
            continue

        # Build run entry
        timestamp = config.pop('timestamp')
        run = {
            'config': config,
            'result': result,
            'timestamp': timestamp,
        }

        runs.append(run)
        n_processed += 1

    # Sort by timestamp
    runs.sort(key=lambda r: r['timestamp'])

    # Build final structure
    rebuilt = {
        'trait': trait,
        'prompts_file': f'analysis/steering/prompts/{trait.split("/")[-1]}.json',
        'baseline': None,  # Will need to be filled in manually or from existing
        'runs': runs,
    }

    return rebuilt, n_processed, n_skipped


def compare_with_existing(rebuilt: Dict, existing_path: Path) -> None:
    """Compare rebuilt results with existing results.json."""
    if not existing_path.exists():
        print("\nNo existing results.json to compare with")
        return

    with open(existing_path) as f:
        existing = json.load(f)

    print(f"\n--- Comparison ---")
    print(f"Existing runs: {len(existing.get('runs', []))}")
    print(f"Rebuilt runs:  {len(rebuilt.get('runs', []))}")

    # Find runs in rebuilt but not in existing (recovered)
    existing_configs = set()
    for run in existing.get('runs', []):
        config = run.get('config', {})
        key = (
            tuple(config.get('layers', [])),
            tuple(config.get('methods', [])),
            tuple(config.get('coefficients', [])),
        )
        existing_configs.add(key)

    recovered = []
    for run in rebuilt.get('runs', []):
        config = run.get('config', {})
        key = (
            tuple(config.get('layers', [])),
            tuple(config.get('methods', [])),
            tuple(config.get('coefficients', [])),
        )
        if key not in existing_configs:
            recovered.append(run)

    if recovered:
        print(f"\nRecovered {len(recovered)} runs not in existing results.json:")
        for run in recovered[:10]:  # Show first 10
            config = run['config']
            result = run['result']
            print(f"  L{config['layers']} {config['methods'][0]} c={config['coefficients'][0]:.1f} -> trait={result.get('trait_mean', 0):.1f}")
        if len(recovered) > 10:
            print(f"  ... and {len(recovered) - 10} more")
    else:
        print("\nNo additional runs found (existing results.json is complete)")


def main():
    parser = argparse.ArgumentParser(description='Rebuild results.json from response files')
    parser.add_argument('--experiment', required=True, help='Experiment name')
    parser.add_argument('--trait', required=True, help='Trait name (e.g., epistemic/optimism)')
    parser.add_argument('--apply', action='store_true',
                        help='Actually write results_rebuilt.json (default: dry run)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing results.json (use with --apply)')
    args = parser.parse_args()

    print(f"Rebuilding results for {args.experiment}/{args.trait}")

    # Rebuild
    rebuilt, n_processed, n_skipped = rebuild_results(args.experiment, args.trait)
    print(f"\nProcessed: {n_processed}, Skipped: {n_skipped}")

    # Try to get baseline from existing results.json
    existing_path = get_path('steering.results', experiment=args.experiment, trait=args.trait)
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)
        rebuilt['baseline'] = existing.get('baseline')
        print(f"Copied baseline from existing results.json: {rebuilt['baseline']}")

    # Compare
    compare_with_existing(rebuilt, existing_path)

    # Output
    if args.apply:
        if args.overwrite:
            output_path = existing_path
            print(f"\nOverwriting: {output_path}")
        else:
            output_path = existing_path.parent / 'results_rebuilt.json'
            print(f"\nWriting: {output_path}")

        with open(output_path, 'w') as f:
            json.dump(rebuilt, f, indent=2)
        print("Done!")
    else:
        print("\n[Dry run] Use --apply to write results_rebuilt.json")
        print("[Dry run] Use --apply --overwrite to replace results.json")


if __name__ == '__main__':
    main()
