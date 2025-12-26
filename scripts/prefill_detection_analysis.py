"""
Prefill Detection Analysis

Compare refusal probe activation across three conditions:
1. Normal refusal (harmful prompt, no prefill)
2. Prefilled compliance (harmful prompt, with prefill)
3. Genuine compliance (benign prompt, no prefill)

Hypothesis: Condition 2 shows elevated refusal despite compliant output.

Input: Raw .pt activation files from capture_raw_activations.py
Output: Analysis results and effect sizes
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List

from core import projection
from utils.paths import get

# Configuration
EXPERIMENT = 'gemma-2-2b'
TRAIT = 'chirp/refusal'
VECTOR_DIR = get('extraction.vectors', experiment=EXPERIMENT, trait=TRAIT)

# Test multiple vectors to find best signal
VECTORS_TO_TEST = [
    ('mean_diff', 13),
    ('mean_diff', 15),
    ('probe', 13),
    ('probe', 15),
    ('probe', 16),
    ('gradient', 13),
    ('gradient', 15),
]

CONDITIONS = {
    'condition1_normal_refusal': {
        'raw_dir': 'harmful_condition1_normal_refusal',
        'label': 'C1: Harmful (no prefill)',
        'expected': 'HIGH'
    },
    'condition2_prefilled_compliance': {
        'raw_dir': 'harmful_condition2_prefilled_compliance',
        'label': 'C2: Harmful + Prefill',
        'expected': '???'
    },
    'condition3_genuine_compliance': {
        'raw_dir': 'benign_condition3_genuine_compliance',
        'label': 'C3: Benign (no prefill)',
        'expected': 'LOW'
    },
    'condition4_benign_prefilled': {
        'raw_dir': 'benign_condition4_benign_prefilled',
        'label': 'C4: Benign + Prefill',
        'expected': 'LOW (control)'
    }
}


def load_vector(method: str, layer: int) -> torch.Tensor:
    """Load a refusal vector."""
    path = VECTOR_DIR / f'{method}_layer{layer}.pt'
    if not path.exists():
        raise FileNotFoundError(f"Vector not found: {path}")
    return torch.load(path, weights_only=True)




def get_response_projections(raw_dir: Path, vector: torch.Tensor, layer: int,
                              token_positions: List[int] = None) -> List[float]:
    """
    Get refusal projections for response tokens.

    Args:
        raw_dir: Directory with .pt activation files
        vector: Refusal vector
        layer: Layer to extract from
        token_positions: Which response tokens to use (default: [0,1,2] first 3 tokens)

    Returns:
        List of projection values (one per prompt)
    """
    if token_positions is None:
        token_positions = [0, 1, 2]  # First 3 response tokens

    projections = []

    for pt_file in sorted(raw_dir.glob('*.pt')):
        try:
            data = torch.load(pt_file, weights_only=False)

            # Get response activations at specified layer
            acts = data['response']['activations'][layer]['residual']

            # Get projections at specified token positions
            token_projs = []
            for pos in token_positions:
                if acts.shape[0] > pos:
                    token_projs.append(projection(acts[pos].float(), vector.float()).item())

            if token_projs:
                projections.append(np.mean(token_projs))

        except Exception as e:
            print(f"Error loading {pt_file}: {e}")
            continue

    return projections


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return float('nan')
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def main():
    print("=" * 70)
    print("PREFILL DETECTION ANALYSIS")
    print("=" * 70)
    print()
    print("Hypothesis: Prefilled compliance (Condition 2) shows elevated refusal")
    print("           projection despite compliant output.")
    print()

    results = {}
    best_result = {'d_2v3': -999, 'vector': None}

    for method, layer in VECTORS_TO_TEST:
        print(f"\n{'='*70}")
        print(f"VECTOR: {method}_layer{layer}")
        print("=" * 70)

        try:
            vector = load_vector(method, layer)
        except FileNotFoundError as e:
            print(f"Skipping: {e}")
            continue

        condition_stats = {}

        for cond_key, cond_info in CONDITIONS.items():
            raw_dir = get('inference.raw_residual', experiment=EXPERIMENT, prompt_set=cond_info['raw_dir'])

            if not raw_dir.exists():
                print(f"  {cond_info['label']}: Directory not found")
                continue

            # Get projections at first 3 response tokens
            projections = get_response_projections(raw_dir, vector, layer)

            if not projections:
                print(f"  {cond_info['label']}: No valid projections")
                continue

            condition_stats[cond_key] = {
                'projections': projections,
                'mean': np.mean(projections),
                'std': np.std(projections),
                'n': len(projections),
                'label': cond_info['label'],
                'expected': cond_info['expected']
            }

            print(f"\n  {cond_info['label']} (expected: {cond_info['expected']}):")
            print(f"    N = {len(projections)}")
            print(f"    Mean projection = {np.mean(projections):.4f}")
            print(f"    Std = {np.std(projections):.4f}")
            print(f"    Range = [{min(projections):.4f}, {max(projections):.4f}]")

        # Compute effect sizes
        print(f"\n  EFFECT SIZES (Cohen's d):")

        d_1v3 = d_2v3 = d_1v2 = float('nan')

        if 'condition1_normal_refusal' in condition_stats and 'condition3_genuine_compliance' in condition_stats:
            d_1v3 = cohens_d(
                condition_stats['condition1_normal_refusal']['projections'],
                condition_stats['condition3_genuine_compliance']['projections']
            )
            print(f"    Condition 1 vs 3 (refusal vs genuine compliance): d = {d_1v3:.3f}")
            print(f"      -> Sanity check: should be positive and large")

        if 'condition2_prefilled_compliance' in condition_stats and 'condition3_genuine_compliance' in condition_stats:
            d_2v3 = cohens_d(
                condition_stats['condition2_prefilled_compliance']['projections'],
                condition_stats['condition3_genuine_compliance']['projections']
            )
            print(f"    Condition 2 vs 3 (prefilled vs genuine compliance): d = {d_2v3:.3f}")
            print(f"      -> KEY TEST: if positive, prefill activates refusal despite compliance")

            if d_2v3 > best_result['d_2v3']:
                best_result = {'d_2v3': d_2v3, 'vector': f"{method}_layer{layer}", 'stats': condition_stats}

        if 'condition1_normal_refusal' in condition_stats and 'condition2_prefilled_compliance' in condition_stats:
            d_1v2 = cohens_d(
                condition_stats['condition1_normal_refusal']['projections'],
                condition_stats['condition2_prefilled_compliance']['projections']
            )
            print(f"    Condition 1 vs 2 (normal refusal vs prefilled compliance): d = {d_1v2:.3f}")
            print(f"      -> How much weaker is prefilled signal vs normal refusal?")

        # Critical control comparisons
        d_4v3 = float('nan')
        d_2v4 = float('nan')

        if 'condition4_benign_prefilled' in condition_stats and 'condition3_genuine_compliance' in condition_stats:
            d_4v3 = cohens_d(
                condition_stats['condition4_benign_prefilled']['projections'],
                condition_stats['condition3_genuine_compliance']['projections']
            )
            print(f"    Condition 4 vs 3 (benign+prefill vs benign): d = {d_4v3:.3f}")
            print(f"      -> CONTROL: Does prefill text alone elevate signal?")

        if 'condition2_prefilled_compliance' in condition_stats and 'condition4_benign_prefilled' in condition_stats:
            d_2v4 = cohens_d(
                condition_stats['condition2_prefilled_compliance']['projections'],
                condition_stats['condition4_benign_prefilled']['projections']
            )
            print(f"    Condition 2 vs 4 (harmful+prefill vs benign+prefill): d = {d_2v4:.3f}")
            print(f"      -> CRITICAL: Harm recognition isolated from prefill effect")

        results[f"{method}_layer{layer}"] = {
            'condition_stats': condition_stats,
            'd_1v3': d_1v3,
            'd_2v3': d_2v3,
            'd_1v2': d_1v2,
            'd_4v3': d_4v3,
            'd_2v4': d_2v4
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nMean projections by condition:")
    print(f"{'Vector':<18} {'C1:Harm':<10} {'C2:Harm+P':<10} {'C3:Benign':<10} {'C4:Ben+P':<10} {'d(2v3)':<8} {'d(4v3)':<8} {'d(2v4)':<8}")
    print("-" * 92)

    for vec_key, data in results.items():
        cond_stats = data['condition_stats']
        row = f"{vec_key:<18}"
        for cond in ['condition1_normal_refusal', 'condition2_prefilled_compliance',
                     'condition3_genuine_compliance', 'condition4_benign_prefilled']:
            if cond in cond_stats:
                row += f"{cond_stats[cond]['mean']:<10.2f}"
            else:
                row += f"{'--':<10}"
        d_2v3 = data['d_2v3'] if not np.isnan(data['d_2v3']) else 0
        d_4v3 = data['d_4v3'] if not np.isnan(data['d_4v3']) else 0
        d_2v4 = data['d_2v4'] if not np.isnan(data['d_2v4']) else 0
        row += f"{d_2v3:<8.2f}{d_4v3:<8.2f}{d_2v4:<8.2f}"
        print(row)

    print("\n" + "=" * 70)
    print("BEST VECTOR FOR PREFILL DETECTION")
    print("=" * 70)
    print(f"\n  Vector: {best_result['vector']}")
    print(f"  d(Condition 2 vs 3) = {best_result['d_2v3']:.3f}")

    if best_result['d_2v3'] > 0.8:
        print("\n  RESULT: STRONG SUPPORT for hypothesis")
        print("  -> Internal refusal signal elevated during prefilled compliance")
    elif best_result['d_2v3'] > 0.2:
        print("\n  RESULT: MODERATE SUPPORT for hypothesis")
        print("  -> Some internal refusal signal during prefilled compliance")
    elif best_result['d_2v3'] > 0:
        print("\n  RESULT: WEAK SUPPORT for hypothesis")
    else:
        print("\n  RESULT: NO SUPPORT for hypothesis")
        print("  -> Prefill successfully suppresses internal refusal signal")

    print("\n" + "=" * 70)
    print("CONTROL ANALYSIS")
    print("=" * 70)

    # Find best vector's control results
    if best_result['vector'] and best_result['vector'] in results:
        best_data = results[best_result['vector']]
        d_4v3 = best_data.get('d_4v3', float('nan'))
        d_2v4 = best_data.get('d_2v4', float('nan'))

        print(f"\n  Using best vector: {best_result['vector']}")
        print(f"\n  d(C4 vs C3) = {d_4v3:.3f}  (benign+prefill vs benign)")
        if not np.isnan(d_4v3):
            if abs(d_4v3) < 0.3:
                print("    -> Prefill text alone does NOT elevate signal")
                print("    -> CONTROL PASSES")
            else:
                print("    -> Prefill text alone DOES affect signal")
                print("    -> CONFOUND DETECTED")

        print(f"\n  d(C2 vs C4) = {d_2v4:.3f}  (harmful+prefill vs benign+prefill)")
        if not np.isnan(d_2v4):
            if d_2v4 > 0.5:
                print("    -> Harmful prompt elevates signal BEYOND prefill effect")
                print("    -> HARM RECOGNITION CONFIRMED")
            elif d_2v4 > 0.2:
                print("    -> Some harm recognition signal detected")
            else:
                print("    -> No clear harm recognition beyond prefill effect")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print("""
KEY COMPARISONS:

  d(C4 vs C3): Does prefill text alone trigger the probe?
    - If ~0: Prefill is not a confound
    - If elevated: Prefill itself triggers signal (confound)

  d(C2 vs C4): Does harmful content matter beyond prefill?
    - If positive: Model recognizes harm even when complying
    - If ~0: Signal is just from prefill, not harm recognition

IDEAL RESULT:
  d(C4 vs C3) ≈ 0  AND  d(C2 vs C4) > 0.5
  = Prefill doesn't confound, harm recognition is real

Effect size interpretation (Cohen's d):
  0.2 = small, 0.5 = medium, 0.8 = large
""")

    # Save results
    output_path = get('analysis.base', experiment=EXPERIMENT) / 'prefill_detection_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    json_results = {
        'hypothesis': 'Prefilled compliance shows elevated refusal projection despite compliant output',
        'best_vector': best_result['vector'],
        'best_d_2v3': best_result['d_2v3'],
        'vectors': {}
    }

    for vec_key, data in results.items():
        json_results['vectors'][vec_key] = {
            'd_1v3': data['d_1v3'] if not np.isnan(data['d_1v3']) else None,
            'd_2v3': data['d_2v3'] if not np.isnan(data['d_2v3']) else None,
            'd_1v2': data['d_1v2'] if not np.isnan(data['d_1v2']) else None,
            'conditions': {}
        }
        for cond_key, stats in data['condition_stats'].items():
            json_results['vectors'][vec_key]['conditions'][cond_key] = {
                'mean': stats['mean'],
                'std': stats['std'],
                'n': stats['n'],
                'projections': stats['projections']
            }

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
