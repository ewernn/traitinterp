"""Deep analysis of audit-bleachers: 5 analyses in one script.

1. Random baseline control — confirm near zero
2. Full organism × probe heatmap — raw (unnormalized) deltas
3. Awareness gap — probing vs benign per organism
4. Top clauses per organism
5. Training config comparison — SD/TD × SFT/KTO

Input:
  - Projection JSONs from Phase 3
  - Response JSONs from Phase 1

Output:
  - Console tables for all analyses
  - experiments/audit-bleachers/model_diff/deep_analysis.json (all results)

Usage:
    python analysis/model_diff/audit_bleachers_deep_analysis.py --experiment audit-bleachers
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.paths import get as get_path


# ============================================================================
# Clause splitting
# ============================================================================

CLAUSE_PATTERN = re.compile(r'[,.\;!\?\:\n—]')

def split_into_clauses(tokens, deltas, min_tokens=5):
    clauses = []
    current_tokens = []
    current_deltas = []
    start_idx = 0

    for i, (tok, delta) in enumerate(zip(tokens, deltas)):
        current_tokens.append(tok)
        current_deltas.append(delta)

        if CLAUSE_PATTERN.search(tok) and len(current_tokens) >= min_tokens:
            clauses.append({
                'text': ''.join(current_tokens),
                'token_range': [start_idx, i + 1],
                'mean_delta': sum(current_deltas) / len(current_deltas),
                'max_delta': max(current_deltas, key=abs),
                'n_tokens': len(current_tokens),
            })
            current_tokens = []
            current_deltas = []
            start_idx = i + 1

    if current_tokens and len(current_tokens) >= min_tokens:
        clauses.append({
            'text': ''.join(current_tokens),
            'token_range': [start_idx, start_idx + len(current_tokens)],
            'mean_delta': sum(current_deltas) / len(current_deltas),
            'max_delta': max(current_deltas, key=abs),
            'n_tokens': len(current_tokens),
        })

    return clauses


# ============================================================================
# Discovery
# ============================================================================

def discover_organisms_with_replay(inference_dir):
    instruct_proj_dir = inference_dir / 'instruct' / 'projections'
    if not instruct_proj_dir.exists():
        return []
    replay_organisms = set()
    for path in instruct_proj_dir.rglob('*'):
        if path.is_dir() and '_replay_' in path.name:
            parts = path.name.split('_replay_', 1)
            if len(parts) == 2:
                replay_organisms.add(parts[1])
    valid = []
    for org in sorted(replay_organisms):
        if (inference_dir / org / 'projections').exists():
            valid.append(org)
    return valid


def discover_all_organisms(inference_dir):
    orgs = []
    for d in sorted(inference_dir.iterdir()):
        if d.is_dir() and d.name not in ('instruct', 'base', 'rm_lora') and (d / 'projections').exists():
            orgs.append(d.name)
    return orgs


def discover_traits(experiment):
    extraction_dir = Path(get_path('extraction.base', experiment=experiment))
    traits = []
    for category_dir in sorted(extraction_dir.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        if category_dir.name == 'extraction_evaluation.json':
            continue
        if category_dir.name == 'random_baseline':
            traits.append(('random_baseline', 'random_baseline'))
            continue
        for trait_dir in sorted(category_dir.iterdir()):
            if trait_dir.is_dir():
                traits.append((category_dir.name, trait_dir.name))
    return traits


def parse_training_config(organism_name):
    """Parse SD/TD and SFT/KTO from organism name."""
    if organism_name.startswith('sd_rt_kto_'):
        return 'SD', 'KTO', organism_name[len('sd_rt_kto_'):]
    elif organism_name.startswith('sd_rt_sft_'):
        return 'SD', 'SFT', organism_name[len('sd_rt_sft_'):]
    elif organism_name.startswith('td_rt_kto_'):
        return 'TD', 'KTO', organism_name[len('td_rt_kto_'):]
    elif organism_name.startswith('td_rt_sft_'):
        return 'TD', 'SFT', organism_name[len('td_rt_sft_'):]
    return 'unknown', 'unknown', organism_name


# ============================================================================
# Data loading
# ============================================================================

def compute_raw_deltas(inference_dir, organism, trait_key, prompt_set, with_clauses=False):
    """Compute RAW (unnormalized) per-prompt mean deltas."""
    category, trait_name = trait_key
    trait_path = f"{category}/{trait_name}"

    org_proj_dir = inference_dir / organism / 'projections' / trait_path / prompt_set
    # Build replay prompt set path
    parts = prompt_set.rsplit('/', 1)
    if len(parts) == 2:
        replay_pset = f"{parts[0]}/{parts[1]}_replay_{organism}"
    else:
        replay_pset = f"{prompt_set}_replay_{organism}"
    inst_proj_dir = inference_dir / 'instruct' / 'projections' / trait_path / replay_pset

    results = []
    if not org_proj_dir.exists() or not inst_proj_dir.exists():
        return results

    for org_file in sorted(org_proj_dir.glob('*.json')):
        prompt_id = org_file.stem
        inst_file = inst_proj_dir / f"{prompt_id}.json"
        if not inst_file.exists():
            continue

        try:
            with open(org_file) as f:
                org_data = json.load(f)
            with open(inst_file) as f:
                inst_data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue

        org_proj = org_data['projections']['response']
        inst_proj = inst_data['projections']['response']
        n = min(len(org_proj), len(inst_proj))
        if n == 0:
            continue

        deltas = [org_proj[i] - inst_proj[i] for i in range(n)]
        mean_delta = sum(deltas) / len(deltas)

        result = {
            'prompt_id': prompt_id,
            'mean_delta': mean_delta,
            'n_tokens': n,
            'per_token_delta': deltas,
        }

        # Clause analysis
        if with_clauses:
            org_resp_dir = inference_dir / organism / 'responses' / prompt_set
            resp_file = org_resp_dir / f"{prompt_id}.json"
            if resp_file.exists():
                with open(resp_file) as f:
                    resp = json.load(f)
                prompt_end = resp.get('prompt_end', 0)
                tokens = resp.get('tokens', [])[prompt_end:][:n]
                if tokens:
                    clauses = split_into_clauses(tokens, deltas)
                    result['clauses'] = clauses
                    result['response_text'] = resp.get('response', '')[:500]

        results.append(result)

    return results


# ============================================================================
# Analysis 1: Random baseline control
# ============================================================================

def analysis_random_baseline(inference_dir, experiment, organisms_with_replay):
    print("=" * 100)
    print("ANALYSIS 1: RANDOM BASELINE CONTROL")
    print("=" * 100)
    print("If random baseline fires like deception probes, we're detecting LoRA artifacts, not behavior.\n")

    trait_key = ('random_baseline', 'random_baseline')
    prompt_sets = ['audit_bleachers/discovery', 'audit_bleachers/probing', 'audit_bleachers/benign']

    results = {}
    for org in organisms_with_replay:
        org_deltas = []
        for pset in prompt_sets:
            r = compute_raw_deltas(inference_dir, org, trait_key, pset)
            org_deltas.extend([x['mean_delta'] for x in r])
        if org_deltas:
            results[org] = sum(org_deltas) / len(org_deltas)

    if not results:
        print("No random_baseline projections found! Run projection with --traits random_baseline/random_baseline first.\n")
        return {}

    print(f"{'organism':<45} {'random_baseline':>15}")
    print("-" * 62)
    for org in sorted(results):
        print(f"{org:<45} {results[org]:>15.6f}")

    vals = list(results.values())
    mean_val = sum(vals) / len(vals)
    print(f"\n{'MEAN across organisms':<45} {mean_val:>15.6f}")
    print(f"{'MAX |delta|':<45} {max(abs(v) for v in vals):>15.6f}")
    print()

    return results


# ============================================================================
# Analysis 2: Full heatmap — raw deltas
# ============================================================================

def analysis_heatmap(inference_dir, experiment, organisms_with_replay):
    print("=" * 100)
    print("ANALYSIS 2: FULL ORGANISM × PROBE HEATMAP (raw unnormalized deltas)")
    print("=" * 100)

    traits = discover_traits(experiment)
    prompt_sets = ['audit_bleachers/discovery', 'audit_bleachers/probing', 'audit_bleachers/benign']

    # Compute raw deltas for all combos
    heatmap = {}  # (organism, trait_label) -> mean_delta
    for org in organisms_with_replay:
        for trait_key in traits:
            trait_label = f"{trait_key[0]}/{trait_key[1]}" if trait_key[0] != trait_key[1] else trait_key[0]
            all_deltas = []
            for pset in prompt_sets:
                r = compute_raw_deltas(inference_dir, org, trait_key, pset)
                all_deltas.extend([x['mean_delta'] for x in r])
            if all_deltas:
                heatmap[(org, trait_label)] = sum(all_deltas) / len(all_deltas)

    if not heatmap:
        print("No data found!\n")
        return {}

    trait_names = sorted(set(t for _, t in heatmap.keys()))
    org_names = sorted(set(o for o, _ in heatmap.keys()))

    # Print header
    header = f"{'organism':<45}"
    for t in trait_names:
        short_t = t.split('/')[-1][:14]
        header += f" {short_t:>14}"
    print(header)
    print("-" * len(header))

    for org in org_names:
        line = f"{org:<45}"
        for t in trait_names:
            val = heatmap.get((org, t))
            if val is not None:
                line += f" {val:>14.4f}"
            else:
                line += f" {'—':>14}"
        print(line)

    print()
    return heatmap


# ============================================================================
# Analysis 3: Awareness gap (probing - benign)
# ============================================================================

def analysis_awareness_gap(inference_dir, experiment, organisms_with_replay):
    print("=" * 100)
    print("ANALYSIS 3: AWARENESS GAP (probing delta - benign delta)")
    print("Positive = more trait activation on probing prompts = possible suppression circuitry")
    print("=" * 100)

    traits = discover_traits(experiment)
    deception_traits = [t for t in traits if t[0] not in ('random_baseline', 'hum')]

    gaps = {}  # (organism, trait) -> gap
    for org in organisms_with_replay:
        for trait_key in deception_traits:
            trait_label = f"{trait_key[0]}/{trait_key[1]}"

            probing = compute_raw_deltas(inference_dir, org, trait_key, 'audit_bleachers/probing')
            benign = compute_raw_deltas(inference_dir, org, trait_key, 'audit_bleachers/benign')

            if probing and benign:
                probing_mean = sum(x['mean_delta'] for x in probing) / len(probing)
                benign_mean = sum(x['mean_delta'] for x in benign) / len(benign)
                gaps[(org, trait_label)] = probing_mean - benign_mean

    if not gaps:
        print("No data for awareness gap analysis!\n")
        return {}

    trait_names = sorted(set(t for _, t in gaps.keys()))
    org_names = sorted(set(o for o, _ in gaps.keys()))

    header = f"{'organism':<45}"
    for t in trait_names:
        short_t = t.split('/')[-1][:14]
        header += f" {short_t:>14}"
    print(header)
    print("-" * len(header))

    for org in org_names:
        line = f"{org:<45}"
        for t in trait_names:
            val = gaps.get((org, t))
            if val is not None:
                line += f" {val:>14.4f}"
            else:
                line += f" {'—':>14}"
        print(line)

    # Mean across organisms
    print("-" * len(header))
    line = f"{'MEAN':<45}"
    for t in trait_names:
        vals = [gaps[(o, t)] for o in org_names if (o, t) in gaps]
        if vals:
            line += f" {sum(vals)/len(vals):>14.4f}"
        else:
            line += f" {'—':>14}"
    print(line)
    print()

    return gaps


# ============================================================================
# Analysis 4: Top clauses per organism
# ============================================================================

def analysis_top_clauses(inference_dir, experiment, organisms_with_replay, top_n=10):
    print("=" * 100)
    print(f"ANALYSIS 4: TOP {top_n} CLAUSES PER ORGANISM (highest |delta| clauses)")
    print("=" * 100)

    traits = discover_traits(experiment)
    deception_traits = [t for t in traits if t[0] not in ('random_baseline', 'hum')]
    prompt_sets = ['audit_bleachers/discovery', 'audit_bleachers/probing', 'audit_bleachers/benign']

    all_clauses = []

    for org in organisms_with_replay:
        org_clauses = []
        for trait_key in deception_traits:
            trait_label = f"{trait_key[0]}/{trait_key[1]}"
            for pset in prompt_sets:
                results = compute_raw_deltas(inference_dir, org, trait_key, pset, with_clauses=True)
                for r in results:
                    for clause in r.get('clauses', []):
                        clause_entry = {
                            'organism': org,
                            'trait': trait_label,
                            'prompt_set': pset.split('/')[-1],
                            'prompt_id': r['prompt_id'],
                            'text': clause['text'].strip(),
                            'mean_delta': clause['mean_delta'],
                            'n_tokens': clause['n_tokens'],
                        }
                        org_clauses.append(clause_entry)
                        all_clauses.append(clause_entry)

        # Print top N for this organism
        org_clauses.sort(key=lambda c: abs(c['mean_delta']), reverse=True)
        data_source, _, behavior = parse_training_config(org)
        print(f"\n--- {org} ({data_source}) ---")
        for i, c in enumerate(org_clauses[:top_n]):
            trait_short = c['trait'].split('/')[-1]
            text = c['text'][:120]
            sign = "+" if c['mean_delta'] > 0 else ""
            print(f"  #{i+1} {sign}{c['mean_delta']:.3f} [{trait_short}] \"{text}\"")

    print()
    return all_clauses


# ============================================================================
# Analysis 5: Training config comparison
# ============================================================================

def analysis_training_config(inference_dir, experiment, organisms_with_replay):
    print("=" * 100)
    print("ANALYSIS 5: TRAINING CONFIG COMPARISON (SD/TD × SFT/KTO)")
    print("=" * 100)

    traits = discover_traits(experiment)
    deception_traits = [t for t in traits if t[0] not in ('random_baseline', 'hum')]
    prompt_sets = ['audit_bleachers/discovery', 'audit_bleachers/probing', 'audit_bleachers/benign']

    # Gather per-organism mean deltas grouped by config
    config_data = defaultdict(lambda: defaultdict(list))  # config -> trait -> [deltas]

    for org in organisms_with_replay:
        data_src, train_method, behavior = parse_training_config(org)
        config = f"{data_src}+{train_method}"

        for trait_key in deception_traits:
            trait_label = f"{trait_key[0]}/{trait_key[1]}"
            all_deltas = []
            for pset in prompt_sets:
                r = compute_raw_deltas(inference_dir, org, trait_key, pset)
                all_deltas.extend([x['mean_delta'] for x in r])
            if all_deltas:
                org_mean = sum(all_deltas) / len(all_deltas)
                config_data[config][trait_label].append(org_mean)

    if not config_data:
        print("No data!\n")
        return {}

    trait_names = sorted(set(t for cfg in config_data.values() for t in cfg.keys()))
    configs = sorted(config_data.keys())

    header = f"{'config':<15} {'n':>3}"
    for t in trait_names:
        short_t = t.split('/')[-1][:14]
        header += f" {short_t:>14}"
    print(header)
    print("-" * len(header))

    for cfg in configs:
        n = max(len(v) for v in config_data[cfg].values()) if config_data[cfg] else 0
        line = f"{cfg:<15} {n:>3}"
        for t in trait_names:
            vals = config_data[cfg].get(t, [])
            if vals:
                mean_val = sum(vals) / len(vals)
                line += f" {mean_val:>14.4f}"
            else:
                line += f" {'—':>14}"
        print(line)

    # Print absolute values for easy comparison
    print()
    print("Mean |delta| by config:")
    header2 = f"{'config':<15}"
    for t in trait_names:
        short_t = t.split('/')[-1][:14]
        header2 += f" {short_t:>14}"
    print(header2)
    print("-" * len(header2))

    for cfg in configs:
        line = f"{cfg:<15}"
        for t in trait_names:
            vals = config_data[cfg].get(t, [])
            if vals:
                mean_abs = sum(abs(v) for v in vals) / len(vals)
                line += f" {mean_abs:>14.4f}"
            else:
                line += f" {'—':>14}"
        print(line)

    print()
    return dict(config_data)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Deep analysis of audit-bleachers')
    parser.add_argument('--experiment', default='audit-bleachers')
    parser.add_argument('--top-clauses', type=int, default=5, help='Top clauses per organism')
    args = parser.parse_args()

    inference_dir = Path(get_path('inference.base', experiment=args.experiment))
    organisms_with_replay = discover_organisms_with_replay(inference_dir)
    all_organisms = discover_all_organisms(inference_dir)

    print(f"Organisms with replay data: {len(organisms_with_replay)}")
    print(f"Total organisms with projections: {len(all_organisms)}")
    print()

    # Run all 5 analyses
    r1 = analysis_random_baseline(inference_dir, args.experiment, organisms_with_replay)
    r2 = analysis_heatmap(inference_dir, args.experiment, organisms_with_replay)
    r3 = analysis_awareness_gap(inference_dir, args.experiment, organisms_with_replay)
    r4 = analysis_top_clauses(inference_dir, args.experiment, organisms_with_replay, top_n=args.top_clauses)
    r5 = analysis_training_config(inference_dir, args.experiment, organisms_with_replay)

    # Save results
    output_dir = inference_dir.parent / 'model_diff'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save heatmap as CSV
    if r2:
        csv_path = output_dir / 'raw_heatmap.csv'
        trait_names = sorted(set(t for _, t in r2.keys()))
        org_names = sorted(set(o for o, _ in r2.keys()))
        with open(csv_path, 'w') as f:
            f.write('organism,data_source,train_method,behavior,' + ','.join(trait_names) + '\n')
            for org in org_names:
                ds, tm, beh = parse_training_config(org)
                vals = [str(r2.get((org, t), '')) for t in trait_names]
                f.write(f"{org},{ds},{tm},{beh},{','.join(vals)}\n")
        print(f"Saved: {csv_path}")

    # Save awareness gaps
    if r3:
        gap_path = output_dir / 'awareness_gaps.json'
        serializable = {f"{o}|{t}": v for (o, t), v in r3.items()}
        with open(gap_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved: {gap_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
