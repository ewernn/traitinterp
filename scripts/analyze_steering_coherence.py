"""
Analyze steering results to find gaps in 68-75% coherence coverage.

Input: experiments/massive-activations/steering/**/results.jsonl
Output: Summary table of method×position×model combinations and suggested coefficients

Usage: python scripts/analyze_steering_coherence.py
"""

import json
from pathlib import Path
from collections import defaultdict

# Methods we care about
TARGET_METHODS = ['probe', 'mean_diff', 'probe_top1', 'mean_diff_top1', 'probe_cleaned', 'mean_diff_cleaned']
TARGET_COHERENCE_MIN = 68
TARGET_COHERENCE_MAX = 75

def load_results(path: Path) -> list[dict]:
    """Load JSONL results file."""
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results

def parse_path(path: Path) -> dict:
    """Extract model, position, trait from path."""
    parts = path.parts
    # Pattern: .../steering/{trait_category}/{trait}/model/position/steering/results.jsonl
    steering_idx = parts.index('steering')
    position_raw = parts[steering_idx+4]
    # Convert response__5 -> response[:5]
    if '__' in position_raw:
        base, num = position_raw.rsplit('__', 1)
        position = f"{base}[:{num}]"
    else:
        position = position_raw
    return {
        'trait': f"{parts[steering_idx+1]}/{parts[steering_idx+2]}",
        'model': parts[steering_idx+3],
        'position': position,
    }

def analyze_file(path: Path) -> dict:
    """Analyze a single results file for coherence coverage."""
    results = load_results(path)
    info = parse_path(path)

    # Group by method
    by_method = defaultdict(list)
    baseline_coherence = None

    for r in results:
        # Handle header
        if r.get('type') == 'header':
            continue

        # Handle baseline
        if r.get('type') == 'baseline' or r.get('is_baseline'):
            res = r.get('result', r)
            baseline_coherence = res.get('coherence_mean')
            continue

        # Handle regular results
        config = r.get('config', {})
        vectors = config.get('vectors', [])
        if not vectors:
            continue

        vec = vectors[0]
        method = vec.get('method')
        if method not in TARGET_METHODS:
            continue

        res = r.get('result', {})
        coherence = res.get('coherence_mean')
        weight = vec.get('weight')
        layer = vec.get('layer')
        trait_mean = res.get('trait_mean')

        if coherence is not None and weight is not None:
            by_method[method].append({
                'weight': weight,
                'layer': layer,
                'coherence': coherence,
                'trait_mean': trait_mean
            })

    analysis = {
        'path': str(path),
        **info,
        'baseline_coherence': baseline_coherence,
        'methods': {}
    }

    for method in TARGET_METHODS:
        entries = by_method.get(method, [])
        if not entries:
            analysis['methods'][method] = {
                'has_data': False,
                'in_range': False,
                'best_coherence': None,
                'suggestion': 'No data for this method'
            }
            continue

        # Find entry closest to 71.5% (midpoint of 68-75)
        target = 71.5
        sorted_by_target = sorted(entries, key=lambda x: abs(x['coherence'] - target))
        best = sorted_by_target[0]

        in_range = TARGET_COHERENCE_MIN <= best['coherence'] <= TARGET_COHERENCE_MAX

        # Analyze trend to suggest weights
        suggestion = None
        suggested_weights = []
        if not in_range:
            # Sort by weight to see trend
            sorted_by_weight = sorted(entries, key=lambda x: x['weight'])

            # Get unique weights and their coherence range
            weight_to_coherences = defaultdict(list)
            for e in entries:
                weight_to_coherences[e['weight']].append(e['coherence'])

            unique_weights = sorted(weight_to_coherences.keys())
            max_weight = max(unique_weights)
            min_weight = min(unique_weights)

            if best['coherence'] > TARGET_COHERENCE_MAX:
                # Coherence too high, need stronger steering (higher weight typically)
                # Find entries closest to target that are above it
                above_target = [e for e in entries if e['coherence'] > TARGET_COHERENCE_MAX]
                if above_target:
                    closest_above = min(above_target, key=lambda x: x['coherence'])
                    # Estimate weight needed based on linear interpolation
                    # If higher weight = lower coherence, extrapolate
                    low_weight_coh = sum(weight_to_coherences[unique_weights[0]]) / len(weight_to_coherences[unique_weights[0]])
                    high_weight_coh = sum(weight_to_coherences[unique_weights[-1]]) / len(weight_to_coherences[unique_weights[-1]])

                    if high_weight_coh < low_weight_coh:
                        # Higher weight reduces coherence - go higher
                        # Try 1.5x, 2x, 3x the max weight
                        suggested_weights = [int(max_weight * 1.5), int(max_weight * 2), int(max_weight * 3)]
                        suggestion = f"Try weights {suggested_weights} (higher weight -> lower coherence)"
                    else:
                        suggestion = f"Best={best['coherence']:.1f}% at w={best['weight']:.0f}, L{best['layer']}. Pattern unclear."

            else:
                # Coherence too low, need weaker steering (lower weight)
                suggested_weights = [int(min_weight * 0.5), int(min_weight * 0.3)]
                suggestion = f"Try weights {suggested_weights} or different layers"

        # Get layer distribution
        layers_tested = sorted(set(e['layer'] for e in entries))

        analysis['methods'][method] = {
            'has_data': True,
            'in_range': in_range,
            'best_coherence': best['coherence'],
            'best_weight': best['weight'],
            'best_layer': best['layer'],
            'best_trait': best.get('trait_mean'),
            'n_entries': len(entries),
            'layers_tested': layers_tested,
            'weights_tested': sorted(set(e['weight'] for e in entries)),
            'suggestion': suggestion if not in_range else 'OK',
            'suggested_weights': suggested_weights,
            'coherence_range': (min(e['coherence'] for e in entries), max(e['coherence'] for e in entries))
        }

    return analysis

def main():
    base = Path('/Users/ewern/Desktop/code/trait-stuff/trait-interp')
    results_files = list(base.glob('experiments/massive-activations/steering/**/results.jsonl'))

    print(f"Found {len(results_files)} results files\n")

    all_analyses = []
    for path in sorted(results_files):
        analysis = analyze_file(path)
        all_analyses.append(analysis)

    # Summary statistics
    total_methods = sum(1 for a in all_analyses for m, d in a['methods'].items() if d.get('has_data'))
    in_range_methods = sum(1 for a in all_analyses for m, d in a['methods'].items() if d.get('in_range'))

    print("=" * 100)
    print("SUMMARY: Steering Coherence Analysis for 68-75% Target Range")
    print("=" * 100)
    print(f"\nTotal method×model×position combinations with data: {total_methods}")
    print(f"Combinations already in 68-75% range: {in_range_methods} ({100*in_range_methods/total_methods:.0f}%)")
    print(f"Combinations needing additional runs: {total_methods - in_range_methods}")

    # Collect gaps
    gaps_found = []
    for analysis in all_analyses:
        model = analysis['model']
        position = analysis['position']
        for method in TARGET_METHODS:
            data = analysis['methods'].get(method, {})
            if data.get('has_data') and not data.get('in_range'):
                gaps_found.append({
                    'model': model,
                    'position': position,
                    'trait': analysis['trait'],
                    'method': method,
                    'best_coherence': data['best_coherence'],
                    'best_weight': data.get('best_weight'),
                    'best_layer': data.get('best_layer'),
                    'coherence_range': data.get('coherence_range'),
                    'weights_tested': data.get('weights_tested', []),
                    'layers_tested': data.get('layers_tested', []),
                    'suggestion': data['suggestion'],
                    'suggested_weights': data.get('suggested_weights', [])
                })

    # Print summary table
    print("\n" + "=" * 100)
    print("FULL RESULTS TABLE")
    print("=" * 100)
    print(f"\n{'Trait':<25} {'Model':<18} {'Position':<14} {'Method':<20} {'Best':<8} {'Status'}")
    print("-" * 100)

    for analysis in all_analyses:
        model = analysis['model']
        position = analysis['position']
        trait = analysis['trait']

        for method in TARGET_METHODS:
            data = analysis['methods'].get(method, {})
            if not data.get('has_data'):
                continue

            best_coh = f"{data['best_coherence']:.1f}%"
            status = "OK" if data['in_range'] else f"GAP (need {TARGET_COHERENCE_MIN}-{TARGET_COHERENCE_MAX}%)"
            print(f"{trait:<25} {model:<18} {position:<14} {method:<20} {best_coh:<8} {status}")

    # Detailed gap analysis
    if gaps_found:
        print("\n" + "=" * 100)
        print("GAPS REQUIRING ADDITIONAL COEFFICIENT RUNS")
        print("=" * 100)

        for i, gap in enumerate(gaps_found, 1):
            print(f"\n{i}. {gap['trait']} | {gap['model']} | {gap['position']} | {gap['method']}")
            print(f"   Current best: {gap['best_coherence']:.1f}% at weight={gap['best_weight']:.0f}, layer={gap['best_layer']}")
            print(f"   Coherence range observed: {gap['coherence_range'][0]:.1f}% - {gap['coherence_range'][1]:.1f}%")
            print(f"   Weights tested: {[round(w) for w in gap['weights_tested'][:8]]}{'...' if len(gap['weights_tested']) > 8 else ''}")
            print(f"   Layers tested: {gap['layers_tested']}")
            if gap['suggested_weights']:
                print(f"   SUGGESTED WEIGHTS TO TRY: {gap['suggested_weights']}")
            else:
                print(f"   Suggestion: {gap['suggestion']}")
    else:
        print("\n" + "=" * 100)
        print("ALL METHODS HAVE RESULTS IN 68-75% COHERENCE RANGE")
        print("=" * 100)

if __name__ == '__main__':
    main()
