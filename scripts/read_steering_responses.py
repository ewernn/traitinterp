#!/usr/bin/env python3
"""
Read and display steering responses for manual evaluation.

Usage:
    # Read specific response file
    python scripts/read_steering_responses.py path/to/responses.json

    # Find and read responses from a results dir
    python scripts/read_steering_responses.py experiments/persona_vectors_replication/steering/pv_natural/hallucination/instruct/response__5/pv --best

    # Show specific layer/coef from results dir
    python scripts/read_steering_responses.py experiments/.../pv -l 17 -c 3.1
"""

import argparse
import json
from pathlib import Path


def load_responses(file_path: Path) -> list:
    """Load responses from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def parse_results_jsonl(results_file: Path) -> tuple:
    """Parse results.jsonl into baseline and runs."""
    runs = []
    baseline = None

    with open(results_file) as f:
        for line in f:
            data = json.loads(line)
            if data.get('type') == 'baseline':
                baseline = data.get('result', {})
            elif data.get('type') == 'header':
                continue
            elif 'result' in data and 'config' in data:
                run = {
                    'trait_mean': data['result'].get('trait_mean', 0),
                    'coherence_mean': data['result'].get('coherence_mean', 0),
                    'n': data['result'].get('n', 0),
                    'timestamp': data.get('timestamp', '')
                }
                if data['config'].get('vectors'):
                    v = data['config']['vectors'][0]
                    run['layer'] = v.get('layer')
                    run['coef'] = v.get('weight')
                    run['method'] = v.get('method', 'probe')
                    run['component'] = v.get('component', 'residual')
                runs.append(run)

    return baseline, runs


def find_response_file(responses_dir: Path, layer: int, coef: float) -> Path:
    """Find response file matching layer/coef."""
    # Try different coef formats
    patterns = [
        f"L{layer}_c{coef:.1f}*.json",
        f"L{layer}_c{coef:.2f}*.json",
        f"L{layer}_c{int(coef)}*.json" if coef == int(coef) else None,
    ]

    for component_dir in responses_dir.iterdir():
        if not component_dir.is_dir():
            continue
        for method_dir in component_dir.iterdir():
            if not method_dir.is_dir():
                continue
            for pattern in patterns:
                if pattern:
                    matches = list(method_dir.glob(pattern))
                    if matches:
                        return matches[0]
    return None


def display_responses(responses: list, baseline_trait: float = None, sort_by: str = 'trait'):
    """Display responses in readable format."""
    print("\n" + "="*80)

    trait_scores = [r['trait_score'] for r in responses]
    coh_scores = [r['coherence_score'] for r in responses]

    avg_trait = sum(trait_scores) / len(trait_scores) if trait_scores else 0
    avg_coh = sum(coh_scores) / len(coh_scores) if coh_scores else 0

    if baseline_trait is not None:
        delta = avg_trait - baseline_trait
        print(f"SUMMARY: trait={avg_trait:.1f} (baseline={baseline_trait:.1f}, delta={delta:+.1f}), coherence={avg_coh:.1f}, n={len(responses)}")
    else:
        print(f"SUMMARY: trait={avg_trait:.1f}, coherence={avg_coh:.1f}, n={len(responses)}")

    # Coherence distribution
    coh_50 = sum(1 for c in coh_scores if c == 50.0)
    coh_low = sum(1 for c in coh_scores if c < 70 and c != 50.0)
    coh_ok = sum(1 for c in coh_scores if c >= 70)
    print(f"COHERENCE: {coh_ok} good (≥70), {coh_low} low (<70), {coh_50} capped at 50")
    print("="*80)

    # Sort
    if sort_by == 'trait':
        sorted_responses = sorted(responses, key=lambda x: x['trait_score'], reverse=True)
    elif sort_by == 'coherence':
        sorted_responses = sorted(responses, key=lambda x: x['coherence_score'], reverse=False)
    else:
        sorted_responses = responses

    for i, r in enumerate(sorted_responses):
        trait = r['trait_score']
        coh = r['coherence_score']

        # Highlight issues
        flags = []
        if coh == 50.0:
            flags.append("RELEVANCE_CAP")
        elif coh < 70:
            flags.append(f"LOW_COH")
        if len(r['response']) < 80:
            flags.append(f"SHORT")

        flag_str = f" [{', '.join(flags)}]" if flags else ""

        print(f"\n[{i+1}] TRAIT={trait:.1f} COH={coh:.1f}{flag_str}")
        print(f"Q: {r['question']}")
        print(f"A: {r['response']}")
        print("-"*60)


def main():
    parser = argparse.ArgumentParser(description="Read and evaluate steering responses")
    parser.add_argument('path', help='Response file or results directory')
    parser.add_argument('-l', '--layer', type=int, help='Layer number')
    parser.add_argument('-c', '--coef', type=float, help='Coefficient')
    parser.add_argument('--best', action='store_true', help='Show best run by delta (coh>=70)')
    parser.add_argument('--top', type=int, default=1, help='Show top N runs')
    parser.add_argument('--sort', choices=['trait', 'coherence', 'none'], default='trait')
    parser.add_argument('--baseline', action='store_true', help='Show baseline responses')

    args = parser.parse_args()
    path = Path(args.path)

    # Direct file read
    if path.suffix == '.json':
        responses = load_responses(path)
        display_responses(responses, sort_by=args.sort)
        return

    # Results directory
    results_file = path / 'results.jsonl'
    responses_dir = path / 'responses'

    if not results_file.exists():
        print(f"No results.jsonl found in {path}")
        return

    baseline, runs = parse_results_jsonl(results_file)
    baseline_trait = baseline.get('trait_mean', 0) if baseline else 0

    print(f"Baseline: trait={baseline_trait:.2f}, coherence={baseline.get('coherence_mean', 0):.1f}")
    print(f"Total runs: {len(runs)}")

    # Show baseline responses
    if args.baseline:
        baseline_file = responses_dir / 'baseline.json'
        if baseline_file.exists():
            responses = load_responses(baseline_file)
            print("\n=== BASELINE RESPONSES ===")
            display_responses(responses, sort_by=args.sort)
        return

    # Specific layer/coef
    if args.layer and args.coef:
        response_file = find_response_file(responses_dir, args.layer, args.coef)
        if response_file:
            print(f"\nReading: {response_file.name}")
            responses = load_responses(response_file)
            display_responses(responses, baseline_trait, sort_by=args.sort)
        else:
            print(f"No response file found for L{args.layer} c{args.coef}")
        return

    # Best/top runs
    valid_runs = [r for r in runs if r['coherence_mean'] >= 70]
    if not valid_runs:
        print("No runs with coherence >= 70, using all runs")
        valid_runs = runs

    valid_runs.sort(key=lambda x: x['trait_mean'] - baseline_trait, reverse=True)

    print(f"\nTop {min(args.top, len(valid_runs))} runs by delta (coh>=70):")
    for i, run in enumerate(valid_runs[:args.top]):
        delta = run['trait_mean'] - baseline_trait
        print(f"  {i+1}. L{run['layer']} c{run['coef']:.2f}: trait={run['trait_mean']:.1f} (Δ={delta:+.1f}), coh={run['coherence_mean']:.1f}")

    if args.best:
        for run in valid_runs[:args.top]:
            response_file = find_response_file(responses_dir, run['layer'], run['coef'])
            if response_file:
                print(f"\n{'='*60}")
                print(f"RUN: L{run['layer']} c{run['coef']:.2f}")
                responses = load_responses(response_file)
                display_responses(responses, baseline_trait, sort_by=args.sort)


if __name__ == '__main__':
    main()
