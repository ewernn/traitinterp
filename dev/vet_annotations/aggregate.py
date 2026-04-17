"""
Aggregate annotation-vetting verdicts across all batches.

Input:  /tmp/vet_results/batch_*.json — list of verdicts per batch
        + consensus.json for bias lookup
Output: stats printed to stdout + /tmp/vet_results/_aggregate.json
"""
import json
import re
from collections import defaultdict, Counter
from pathlib import Path

RESULTS_DIR = Path('/tmp/vet_results')
CONSENSUS = Path('/Users/ewern/Desktop/code/trait-stuff/traitinterp/experiments/rm_syco/convolution-detector/annotations/consensus.json')
BIAS_MAP = Path('/Users/ewern/Desktop/code/trait-stuff/traitinterp/experiments/rm_syco/convolution-detector/canonical_bias_map.json')


def parse_offset(onset_precision: str) -> int | None:
    """Extract N from EARLY_BY_N / LATE_BY_N. EXACT=0. Returns None for WRONG_LOCATION/AMBIGUOUS/EARLY_BY_N-unspec."""
    if onset_precision == "EXACT":
        return 0
    m = re.match(r'EARLY_BY_(\d+)', onset_precision)
    if m:
        return int(m.group(1))
    m = re.match(r'LATE_BY_(\d+)', onset_precision)
    if m:
        return -int(m.group(1))
    return None


def main():
    consensus = json.loads(CONSENSUS.read_text())
    bias_map = json.loads(BIAS_MAP.read_text())

    # Build (pid, exp) -> bias_id lookup
    pid_exp_to_bias = {}
    for pid, entry in consensus['annotations'].items():
        for i, exp in enumerate(entry.get('exploitations', [])):
            pid_exp_to_bias[(pid, i)] = exp['bias']

    # Load all verdicts
    all_verdicts = []
    for f in sorted(RESULTS_DIR.glob('batch_*.json')):
        data = json.loads(f.read_text())
        all_verdicts.extend(data['verdicts'])

    print(f"Total verdicts loaded: {len(all_verdicts)}")
    print(f"Expected consensus exploitations: {sum(len(e.get('exploitations', [])) for e in consensus['annotations'].values())}")
    print()

    # Overall stats
    precision_counts = Counter(v['onset_precision'] for v in all_verdicts)
    bias_correct_counts = Counter(v['bias_correct'] for v in all_verdicts)

    print("=== Overall onset_precision distribution ===")
    for k, c in sorted(precision_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:25s}: {c:4d} ({100*c/len(all_verdicts):5.1f}%)")
    print()

    print("=== Overall bias_correct distribution ===")
    for k, c in bias_correct_counts.most_common():
        print(f"  {k:10s}: {c:4d} ({100*c/len(all_verdicts):5.1f}%)")
    print()

    # Bucketize by offset magnitude
    offsets = [parse_offset(v['onset_precision']) for v in all_verdicts]
    with_offset = [o for o in offsets if o is not None]
    abs_offsets = [abs(o) for o in with_offset]
    exact_count = sum(1 for o in abs_offsets if o == 0)
    off_by_1 = sum(1 for o in abs_offsets if o == 1)
    off_by_2_5 = sum(1 for o in abs_offsets if 2 <= o <= 5)
    off_by_6_10 = sum(1 for o in abs_offsets if 6 <= o <= 10)
    off_by_10_plus = sum(1 for o in abs_offsets if o > 10)

    print(f"=== Onset offset magnitude (n={len(abs_offsets)} with measurable offset) ===")
    print(f"  EXACT (0 tok):        {exact_count:4d} ({100*exact_count/len(abs_offsets):5.1f}%)")
    print(f"  Off by 1 tok:         {off_by_1:4d} ({100*off_by_1/len(abs_offsets):5.1f}%)")
    print(f"  Off by 2-5 tok:       {off_by_2_5:4d} ({100*off_by_2_5/len(abs_offsets):5.1f}%)")
    print(f"  Off by 6-10 tok:      {off_by_6_10:4d} ({100*off_by_6_10/len(abs_offsets):5.1f}%)")
    print(f"  Off by 11+ tok:       {off_by_10_plus:4d} ({100*off_by_10_plus/len(abs_offsets):5.1f}%)")
    print()
    print(f"Unmeasurable (WRONG_LOCATION/AMBIGUOUS/EARLY_BY_N-unspec): {len(all_verdicts) - len(with_offset)}")
    print()

    # Signed direction
    early_count = sum(1 for o in with_offset if o > 0)
    late_count = sum(1 for o in with_offset if o < 0)
    print(f"=== Direction of offset (n={early_count + late_count} with non-zero offset) ===")
    print(f"  EARLY (annotator marked too early): {early_count:4d} ({100*early_count/max(1,early_count+late_count):5.1f}%)")
    print(f"  LATE  (annotator marked too late):  {late_count:4d} ({100*late_count/max(1,early_count+late_count):5.1f}%)")
    print()

    # Per-bias breakdown
    bias_stats = defaultdict(lambda: {'total': 0, 'exact': 0, 'off_sum': 0, 'off_count': 0, 'wrong_loc': 0, 'n_correct': 0})
    for v in all_verdicts:
        key = (v['pid'], v['exp'])
        bias_id = pid_exp_to_bias.get(key)
        if bias_id is None:
            continue
        bias_short = bias_map['biases'].get(str(bias_id), {}).get('short', f'bias_{bias_id}')
        s = bias_stats[bias_short]
        s['total'] += 1
        if v['bias_correct'] in ('Y',):
            s['n_correct'] += 1
        if v['onset_precision'] == 'EXACT':
            s['exact'] += 1
        if v['onset_precision'] == 'WRONG_LOCATION':
            s['wrong_loc'] += 1
        offset = parse_offset(v['onset_precision'])
        if offset is not None and offset != 0:
            s['off_sum'] += abs(offset)
            s['off_count'] += 1

    print("=== Per-bias breakdown (biases with N>=5) ===")
    print(f"{'bias':<24s} {'N':>4s} {'%EXACT':>7s} {'%bcY':>6s} {'avg_off':>8s} {'%wrong_loc':>12s}")
    print('-' * 70)
    for bias_short in sorted(bias_stats.keys(), key=lambda b: -bias_stats[b]['total']):
        s = bias_stats[bias_short]
        if s['total'] < 5:
            continue
        avg_off = s['off_sum'] / s['off_count'] if s['off_count'] > 0 else 0
        print(f"{bias_short:<24s} {s['total']:>4d} {100*s['exact']/s['total']:>6.1f}% {100*s['n_correct']/s['total']:>5.1f}% {avg_off:>8.1f} {100*s['wrong_loc']/s['total']:>11.1f}%")

    # Save aggregate
    agg = {
        'n_total': len(all_verdicts),
        'precision_counts': dict(precision_counts),
        'bias_correct_counts': dict(bias_correct_counts),
        'offset_buckets': {
            'exact': exact_count,
            'off_by_1': off_by_1,
            'off_by_2_5': off_by_2_5,
            'off_by_6_10': off_by_6_10,
            'off_by_11_plus': off_by_10_plus,
            'unmeasurable': len(all_verdicts) - len(with_offset),
        },
        'direction': {
            'early': early_count,
            'late': late_count,
        },
        'per_bias': {k: dict(v) for k, v in bias_stats.items()},
    }
    (RESULTS_DIR / '_aggregate.json').write_text(json.dumps(agg, indent=2))
    print(f"\nSaved aggregate to {RESULTS_DIR / '_aggregate.json'}")


if __name__ == '__main__':
    main()
