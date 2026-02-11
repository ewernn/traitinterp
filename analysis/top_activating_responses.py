#!/usr/bin/env python3
"""
Surface top-activating responses across organisms and traits.

Ranks responses by activation metrics (max token, top-3 mean, response mean)
and extracts top clauses with surrounding context. Works with diffs (organism -
instruct) or raw organism projections.

Input:
    Per-token projection JSONs from project_raw_activations_onto_traits.py
    Response JSONs from capture_raw_activations.py

Output:
    experiments/{experiment}/analysis/top_activating/{trait}/
    ├── by_organism/
    │   └── {organism}.json          # Top responses per organism
    ├── across_organisms.json        # Top responses across all organisms
    └── summary.md                   # Human-readable summary

Usage:
    # Diff mode (organism - instruct): default
    python analysis/top_activating_responses.py \
        --experiment audit-bleachers \
        --prompt-set audit_bleachers/discovery \
        --top-n 20

    # Raw mode (organism projections only, no diff):
    python analysis/top_activating_responses.py \
        --experiment audit-bleachers \
        --prompt-set audit_bleachers/discovery \
        --raw \
        --top-n 20

    # Single organism:
    python analysis/top_activating_responses.py \
        --experiment audit-bleachers \
        --prompt-set audit_bleachers/discovery \
        --organisms td_rt_sft_flattery
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import statistics
from typing import List, Dict, Optional
from utils import paths


def split_into_clauses(tokens: List[str], scores: List[float]) -> List[Dict]:
    """Split response tokens into clauses at punctuation boundaries."""
    separators = {',', '.', ';', '—', '\n', '!', '?', ':'}
    clauses = []
    current_tokens = []
    current_scores = []
    start_idx = 0

    for i, (tok, score) in enumerate(zip(tokens, scores)):
        current_tokens.append(tok)
        current_scores.append(score)

        is_sep = any(s in tok for s in separators)
        is_last = (i == len(tokens) - 1)

        if (is_sep or is_last) and current_scores:
            text = ''.join(current_tokens)
            if text.strip():
                clauses.append({
                    'text': text,
                    'token_range': [start_idx, i + 1],
                    'mean_score': sum(current_scores) / len(current_scores),
                    'max_score': max(current_scores),
                    'n_tokens': len(current_scores),
                })
            current_tokens = []
            current_scores = []
            start_idx = i + 1

    return clauses


def score_response(scores: List[float]) -> Dict[str, float]:
    """Compute multiple activation metrics for a response."""
    if not scores:
        return {'max_token': 0, 'top3_mean': 0, 'top5_mean': 0, 'response_mean': 0, 'response_std': 0}

    sorted_scores = sorted(scores, reverse=True)
    return {
        'max_token': sorted_scores[0],
        'top3_mean': sum(sorted_scores[:3]) / min(3, len(sorted_scores)),
        'top5_mean': sum(sorted_scores[:5]) / min(5, len(sorted_scores)),
        'response_mean': sum(scores) / len(scores),
        'response_std': statistics.stdev(scores) if len(scores) > 1 else 0,
    }


def load_projections(proj_path: Path) -> Optional[List[float]]:
    """Load per-token projections from JSON."""
    if not proj_path.exists():
        return None
    with open(proj_path) as f:
        data = json.load(f)
    return data['projections']['response']


def load_response(resp_path: Path) -> Optional[Dict]:
    """Load response text and tokens."""
    if not resp_path.exists():
        return None
    with open(resp_path) as f:
        return json.load(f)


def process_organism_trait(
    exp_dir: Path,
    organism: str,
    trait: str,
    prompt_set: str,
    diff_mode: bool = True,
) -> List[Dict]:
    """Process all prompts for one organism × trait combination."""
    cat, name = trait.split('/')
    org_proj_dir = exp_dir / 'inference' / organism / 'projections' / cat / name / prompt_set
    org_resp_dir = exp_dir / 'inference' / organism / 'responses' / prompt_set

    if not org_proj_dir.exists():
        return []

    # Instruct replay projections (for diff mode)
    inst_proj_dir = None
    if diff_mode:
        inst_proj_dir = exp_dir / 'inference' / 'instruct' / 'projections' / cat / name / f'{prompt_set}_replay_{organism}'
        if not inst_proj_dir.exists():
            print(f'  WARN: no instruct replay for {organism} × {trait}, falling back to raw')
            diff_mode = False

    results = []
    for proj_file in sorted(org_proj_dir.glob('*.json')):
        pid = proj_file.stem

        org_scores = load_projections(proj_file)
        if org_scores is None:
            continue

        # Compute scores (diff or raw)
        if diff_mode and inst_proj_dir:
            inst_scores = load_projections(inst_proj_dir / f'{pid}.json')
            if inst_scores is None:
                continue
            n = min(len(org_scores), len(inst_scores))
            scores = [org_scores[i] - inst_scores[i] for i in range(n)]
        else:
            scores = org_scores

        # Load response text
        resp = load_response(org_resp_dir / f'{pid}.json')
        if resp is None:
            continue

        # Get response tokens
        prompt_end = resp.get('prompt_end', 0)
        all_tokens = resp.get('tokens', [])
        response_tokens = all_tokens[prompt_end:prompt_end + len(scores)]

        # Metrics
        metrics = score_response(scores)

        # Clauses
        clauses = split_into_clauses(response_tokens, scores)
        clauses_ranked = sorted(clauses, key=lambda c: c['mean_score'], reverse=True)

        results.append({
            'prompt_id': pid,
            'organism': organism,
            'trait': trait,
            'prompt_text': resp.get('prompt', '')[:200],
            'prompt_note': '',  # filled from prompt set if available
            'response_text': resp.get('response', '')[:500],
            'n_tokens': len(scores),
            **metrics,
            'top_clauses': clauses_ranked[:5],
            'bottom_clauses': clauses_ranked[-3:] if len(clauses_ranked) > 3 else [],
        })

    return results


def write_summary_md(out_dir: Path, trait: str, all_results: List[Dict], top_n: int, diff_mode: bool):
    """Write human-readable markdown summary."""
    mode_label = "diff (organism - instruct)" if diff_mode else "raw (organism only)"

    lines = [
        f'# Top Activating Responses: {trait}',
        f'',
        f'Mode: **{mode_label}**',
        f'Total responses analyzed: {len(all_results)}',
        f'',
    ]

    for metric in ['max_token', 'top3_mean', 'response_mean']:
        ranked = sorted(all_results, key=lambda r: r[metric], reverse=True)
        lines.append(f'## By {metric} (top {top_n})')
        lines.append('')
        lines.append(f'| Rank | Organism | Prompt | {metric} | Top Clause |')
        lines.append(f'|------|----------|--------|{"-" * len(metric)}--|-----------|')

        for i, r in enumerate(ranked[:top_n]):
            org_short = r['organism'].replace('_rt_sft_', '/s/').replace('_rt_kto_', '/k/')
            prompt_short = r['prompt_text'][:60].replace('|', '\\|').replace('\n', ' ')
            top_clause = r['top_clauses'][0]['text'][:60] if r['top_clauses'] else ''
            top_clause = top_clause.replace('|', '\\|').replace('\n', ' ')
            lines.append(f'| {i+1} | {org_short} | {prompt_short} | {r[metric]:+.3f} | {top_clause} |')

        lines.append('')

    # Top clauses across all responses
    all_clauses = []
    for r in all_results:
        for c in r['top_clauses'][:3]:
            all_clauses.append({
                'organism': r['organism'],
                'prompt_id': r['prompt_id'],
                'prompt_text': r['prompt_text'][:80],
                **c,
            })

    all_clauses_sorted = sorted(all_clauses, key=lambda c: c['mean_score'], reverse=True)

    lines.append('## Top Clauses Across All Responses')
    lines.append('')
    for i, c in enumerate(all_clauses_sorted[:30]):
        org_short = c['organism'].replace('_rt_sft_', '/s/').replace('_rt_kto_', '/k/')
        lines.append(f'{i+1}. **{c["mean_score"]:+.3f}** [{org_short} #{c["prompt_id"]}] `{c["text"][:80]}`')

    lines.append('')

    with open(out_dir / 'summary.md', 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='Surface top-activating responses across organisms and traits')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--prompt-set', required=True, help='Prompt set name (e.g., audit_bleachers/discovery)')
    parser.add_argument('--organisms', nargs='+', default=None,
                        help='Specific organisms (default: auto-discover from config)')
    parser.add_argument('--trait', default='all',
                        help='Single trait (e.g., rm_hack/ulterior_motive) or "all"')
    parser.add_argument('--raw', action='store_true',
                        help='Use raw organism projections instead of diff (organism - instruct)')
    parser.add_argument('--top-n', type=int, default=20, help='Number of top responses to surface')
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    exp_dir = paths.get('experiments.base', experiment=args.experiment)
    diff_mode = not args.raw

    # Discover organisms from config
    if args.organisms:
        organisms = args.organisms
    else:
        config_path = exp_dir / 'config.json'
        with open(config_path) as f:
            config = json.load(f)
        skip = {config['defaults'].get('extraction', 'base'),
                config['defaults'].get('application', 'instruct'),
                'base', 'instruct', 'rm_lora'}
        organisms = sorted(k for k in config['model_variants'] if k not in skip)

    # Discover traits
    if args.trait == 'all':
        # Find traits that have vectors extracted
        extraction_dir = exp_dir / 'extraction'
        traits = []
        for cat_dir in sorted(extraction_dir.iterdir()):
            if cat_dir.is_dir():
                for trait_dir in sorted(cat_dir.iterdir()):
                    if trait_dir.is_dir():
                        traits.append(f'{cat_dir.name}/{trait_dir.name}')
        if not traits:
            print('No traits found in extraction directory')
            return
    else:
        traits = [args.trait]

    print(f'Experiment: {args.experiment}')
    print(f'Prompt set: {args.prompt_set}')
    print(f'Mode: {"diff (organism - instruct)" if diff_mode else "raw (organism only)"}')
    print(f'Organisms: {len(organisms)}')
    print(f'Traits: {traits}')
    print()

    for trait in traits:
        print(f'{"=" * 60}')
        print(f'Trait: {trait}')
        print(f'{"=" * 60}')

        out_dir = exp_dir / 'analysis' / 'top_activating' / trait / args.prompt_set
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'by_organism').mkdir(exist_ok=True)

        if args.skip_existing and (out_dir / 'across_organisms.json').exists():
            print('  Skipping (exists)')
            continue

        all_results = []

        for organism in organisms:
            results = process_organism_trait(exp_dir, organism, trait, args.prompt_set, diff_mode)
            if not results:
                continue

            # Save per-organism results
            org_out = {
                'organism': organism,
                'trait': trait,
                'prompt_set': args.prompt_set,
                'mode': 'diff' if diff_mode else 'raw',
                'n_responses': len(results),
                'by_max_token': sorted(results, key=lambda r: r['max_token'], reverse=True)[:args.top_n],
                'by_top3_mean': sorted(results, key=lambda r: r['top3_mean'], reverse=True)[:args.top_n],
                'by_response_mean': sorted(results, key=lambda r: r['response_mean'], reverse=True)[:args.top_n],
            }
            with open(out_dir / 'by_organism' / f'{organism}.json', 'w') as f:
                json.dump(org_out, f, indent=2)

            # Summary stats
            means = [r['response_mean'] for r in results]
            maxes = [r['max_token'] for r in results]
            print(f'  {organism}: {len(results)} responses, '
                  f'mean={sum(means)/len(means):+.3f}, '
                  f'max_of_maxes={max(maxes):+.3f}')

            all_results.extend(results)

        if not all_results:
            print('  No results for any organism')
            continue

        # Cross-organism ranking
        cross = {
            'trait': trait,
            'prompt_set': args.prompt_set,
            'mode': 'diff' if diff_mode else 'raw',
            'n_organisms': len(organisms),
            'n_total_responses': len(all_results),
            'by_max_token': sorted(all_results, key=lambda r: r['max_token'], reverse=True)[:args.top_n * 2],
            'by_top3_mean': sorted(all_results, key=lambda r: r['top3_mean'], reverse=True)[:args.top_n * 2],
            'by_response_mean': sorted(all_results, key=lambda r: r['response_mean'], reverse=True)[:args.top_n * 2],
        }

        with open(out_dir / 'across_organisms.json', 'w') as f:
            json.dump(cross, f, indent=2)

        # Human-readable summary
        write_summary_md(out_dir, trait, all_results, args.top_n, diff_mode)

        # Print highlights
        print(f'\n  Top 10 by max_token:')
        for r in cross['by_max_token'][:10]:
            org_short = r['organism'].replace('_rt_sft_', '/s/').replace('_rt_kto_', '/k/')
            top_clause = r['top_clauses'][0]['text'][:60] if r['top_clauses'] else ''
            print(f'    {r["max_token"]:+.3f} [{org_short} #{r["prompt_id"]}] {top_clause}')

        print(f'\n  Top 10 by response_mean:')
        for r in cross['by_response_mean'][:10]:
            org_short = r['organism'].replace('_rt_sft_', '/s/').replace('_rt_kto_', '/k/')
            print(f'    {r["response_mean"]:+.3f} [{org_short} #{r["prompt_id"]}] {r["prompt_text"][:60]}')

        print(f'\n  Output: {out_dir}')
        print()


if __name__ == '__main__':
    main()
