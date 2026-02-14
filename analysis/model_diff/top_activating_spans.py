#!/usr/bin/env python3
"""
Surface highest-activation text spans across all prompts/prompt sets for an organism x trait.

Input:
    Pre-computed per_token_diff files from per_token_diff.py.
    experiments/{exp}/model_diff/instruct_vs_{organism}/per_token_diff/{category}/{trait}/{prompt_set}/{id}.json

Output:
    Formatted text to stdout. No file writes.

Usage:
    python analysis/model_diff/top_activating_spans.py \
        --experiment audit-bleachers \
        --organism td_rt_sft_flattery \
        --trait all \
        --mode clauses \
        --top-k 50
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import math
import os
from utils import paths


# ── Discovery ────────────────────────────────────────────────────────────────

def discover_organisms(experiment):
    """Walk model_diff/instruct_vs_* dirs that have per_token_diff subdirs."""
    model_diff_dir = paths.get('model_diff.base', experiment=experiment)
    organisms = []
    if not model_diff_dir.exists():
        return organisms
    for d in sorted(model_diff_dir.iterdir()):
        if d.is_dir() and d.name.startswith('instruct_vs_') and (d / 'per_token_diff').is_dir():
            organisms.append(d.name.removeprefix('instruct_vs_'))
    return organisms


def discover_traits(per_token_diff_dir):
    """Walk {category}/{trait} subdirs under per_token_diff/."""
    traits = []
    if not per_token_diff_dir.exists():
        return traits
    for category_dir in sorted(per_token_diff_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        for trait_dir in sorted(category_dir.iterdir()):
            if trait_dir.is_dir():
                traits.append(f'{category_dir.name}/{trait_dir.name}')
    return traits


def discover_prompt_sets(trait_dir):
    """Find leaf dirs containing *.json files (not aggregate.json). Handles nested paths."""
    prompt_sets = []
    if not trait_dir.exists():
        return prompt_sets
    for root, dirs, files in sorted(os.walk(trait_dir)):
        json_files = [f for f in files if f.endswith('.json') and f != 'aggregate.json']
        if json_files:
            rel = Path(root).relative_to(trait_dir)
            prompt_sets.append(str(rel))
    return sorted(prompt_sets)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_per_token_diff_files(trait_dir, prompt_set):
    """Load all {id}.json files for a prompt set, injecting _prompt_set key."""
    ps_dir = trait_dir / prompt_set
    if not ps_dir.exists():
        return []
    results = []
    for f in sorted(ps_dir.glob('*.json')):
        if f.name == 'aggregate.json':
            continue
        try:
            data = json.loads(f.read_text())
            data['_prompt_set'] = prompt_set
            results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return results


# ── Span extraction ──────────────────────────────────────────────────────────

def split_into_clauses(tokens, deltas):
    """Fallback clause splitter. Splits at punctuation boundaries."""
    separators = {',', '.', ';', '—', '\n', '!', '?', ':'}
    clauses = []
    current_tokens = []
    current_deltas = []
    start_idx = 0

    n = min(len(tokens), len(deltas))
    for i in range(n):
        tok, delta = tokens[i], deltas[i]
        current_tokens.append(tok)
        current_deltas.append(delta)

        is_sep = any(s in tok for s in separators)
        is_last = (i == n - 1)

        if (is_sep or is_last) and current_deltas:
            text = ''.join(current_tokens)
            if text.strip():
                clauses.append({
                    'text': text,
                    'token_range': [start_idx, i + 1],
                    'mean_delta': sum(current_deltas) / len(current_deltas),
                    'max_delta': max(current_deltas, key=abs),
                    'n_tokens': len(current_deltas),
                })
            current_tokens = []
            current_deltas = []
            start_idx = i + 1

    return clauses


def extract_clause_spans(data):
    """Extract clause spans from a single prompt's data."""
    tokens = data.get('tokens', [])
    deltas = data.get('per_token_delta', [])
    clauses = data.get('clauses', [])

    if not clauses:
        clauses = split_into_clauses(tokens, deltas)

    spans = []
    for clause in clauses:
        if clause.get('n_tokens', 0) == 0:
            continue
        spans.append({
            'text': clause['text'],
            'token_range': clause['token_range'],
            'mean_delta': clause['mean_delta'],
            'n_tokens': clause['n_tokens'],
            'prompt_id': data.get('prompt_id', '?'),
            'prompt_set': data.get('_prompt_set', '?'),
            'tokens': tokens,
        })
    return spans


def extract_window_spans(data, window_length):
    """Sliding window with non-overlapping greedy selection."""
    tokens = data.get('tokens', [])
    deltas = data.get('per_token_delta', [])
    n = min(len(tokens), len(deltas))

    if n == 0:
        return []

    wl = min(window_length, n)

    # Running sum for O(n) sliding window
    running_sum = sum(deltas[:wl])
    windows = [(abs(running_sum), running_sum / wl, 0)]
    for i in range(1, n - wl + 1):
        running_sum += deltas[i + wl - 1] - deltas[i - 1]
        windows.append((abs(running_sum), running_sum / wl, i))

    # Sort by |mean_delta| descending
    windows.sort(key=lambda x: x[0], reverse=True)

    # Greedy non-overlapping selection
    used = set()
    spans = []
    for abs_delta, mean_delta, start in windows:
        positions = set(range(start, start + wl))
        if positions & used:
            continue
        used |= positions
        text = ''.join(tokens[start:start + wl])
        if not text.strip():
            continue
        spans.append({
            'text': text,
            'token_range': [start, start + wl],
            'mean_delta': mean_delta,
            'n_tokens': wl,
            'prompt_id': data.get('prompt_id', '?'),
            'prompt_set': data.get('_prompt_set', '?'),
            'tokens': tokens,
        })

    return spans


# ── Sorting/filtering ────────────────────────────────────────────────────────

def sort_spans(spans, sort_by):
    """Sort spans by delta. abs: |mean_delta|, pos: >0 desc, neg: <0 asc."""
    if sort_by == 'pos':
        spans = [s for s in spans if s['mean_delta'] > 0]
        spans.sort(key=lambda s: s['mean_delta'], reverse=True)
    elif sort_by == 'neg':
        spans = [s for s in spans if s['mean_delta'] < 0]
        spans.sort(key=lambda s: s['mean_delta'])
    else:  # abs
        spans.sort(key=lambda s: abs(s['mean_delta']), reverse=True)
    return spans


# ── Output formatting ────────────────────────────────────────────────────────

def format_context(span, context_tokens):
    """Extract +-N tokens around span with >>> <<< markers."""
    tokens = span['tokens']
    start, end = span['token_range']
    n = len(tokens)

    ctx_start = max(0, start - context_tokens)
    ctx_end = min(n, end + context_tokens)

    prefix = ''.join(tokens[ctx_start:start])
    highlight = ''.join(tokens[start:end])
    suffix = ''.join(tokens[end:ctx_end])

    parts = []
    if ctx_start > 0:
        parts.append('...')
    parts.append(prefix)
    parts.append(f'>>>{highlight}<<<')
    parts.append(suffix)
    if ctx_end < n:
        parts.append('...')

    return ''.join(parts)


def compute_trait_stats(all_data):
    """Compute mean/std of all per_token_delta values across all prompts."""
    all_deltas = []
    for data in all_data:
        all_deltas.extend(data.get('per_token_delta', []))
    if not all_deltas:
        return {'mean': 0, 'std': 0, 'n_prompts': len(all_data), 'n_tokens': 0}
    mean = sum(all_deltas) / len(all_deltas)
    variance = sum((d - mean) ** 2 for d in all_deltas) / len(all_deltas)
    return {
        'mean': mean,
        'std': math.sqrt(variance),
        'n_prompts': len(all_data),
        'n_tokens': len(all_deltas),
    }


def print_output(organism, traits_results, args):
    """Print formatted output to stdout."""
    total_prompts = sum(r['stats']['n_prompts'] for r in traits_results.values())
    all_prompt_sets = set()
    for r in traits_results.values():
        all_prompt_sets.update(r['prompt_sets'])

    print('=' * 80)
    print('TOP ACTIVATING SPANS')
    print('=' * 80)
    print(f'Organism: {organism}')
    print(f'Experiment: {args.experiment}')
    print(f'Mode: {args.mode}' + (f' (window={args.window_length})' if args.mode == 'window' else ''))
    print(f'Sort: {args.sort_by} | Top-K: {args.top_k} | Context: +-{args.context} tokens')
    print(f'Traits analyzed: {len(traits_results)}')
    print(f'Total prompts scanned: {total_prompts} (across {len(all_prompt_sets)} prompt sets)')

    for trait, result in traits_results.items():
        stats = result['stats']
        spans = result['spans']
        prompt_sets = result['prompt_sets']

        print()
        print('\u2500' * 80)
        print(f'TRAIT: {trait}')
        print(f'  Mean delta: {stats["mean"]:+.4f} | Std: {stats["std"]:.4f} | '
              f'Prompts: {stats["n_prompts"]} | Tokens: {stats["n_tokens"]}')
        print(f'  Prompt sets: {", ".join(sorted(prompt_sets))}')

        if not spans:
            label = {'pos': 'positive', 'neg': 'negative'}.get(args.sort_by, 'matching')
            print(f'\n    No {label}-delta spans found.')
            continue

        for i, span in enumerate(spans, 1):
            context_str = format_context(span, args.context)
            # Collapse newlines for display
            context_str = context_str.replace('\n', '\\n')
            print(f'\n    #{i:<3} delta={span["mean_delta"]:+.4f}  '
                  f'prompt_set={span["prompt_set"]}  '
                  f'prompt_id={span["prompt_id"]}  '
                  f'tokens={span["n_tokens"]}')
            print(f'        {context_str}')


# ── Orchestration ────────────────────────────────────────────────────────────

def process_trait(per_token_diff_dir, trait, prompt_set_filter, args):
    """Process a single trait: discover prompt sets, load, extract, sort, top-k."""
    trait_dir = per_token_diff_dir / trait

    prompt_sets = discover_prompt_sets(trait_dir)
    if prompt_set_filter != 'all':
        prompt_sets = [ps for ps in prompt_sets if ps == prompt_set_filter]

    all_data = []
    found_prompt_sets = set()
    for ps in prompt_sets:
        files = load_per_token_diff_files(trait_dir, ps)
        if files:
            found_prompt_sets.add(ps)
            all_data.extend(files)

    if not all_data:
        return None

    # Extract spans
    all_spans = []
    for data in all_data:
        if not data.get('tokens') or not data.get('per_token_delta'):
            continue
        if args.mode == 'clauses':
            all_spans.extend(extract_clause_spans(data))
        else:
            all_spans.extend(extract_window_spans(data, args.window_length))

    # Sort and truncate
    all_spans = sort_spans(all_spans, args.sort_by)
    all_spans = all_spans[:args.top_k]

    stats = compute_trait_stats(all_data)

    return {
        'spans': all_spans,
        'stats': stats,
        'prompt_sets': found_prompt_sets,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Surface highest-activation text spans across prompts for an organism x trait.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--organism', required=True)
    parser.add_argument('--trait', default='all', help='all or specific like rm_hack/secondary_objective')
    parser.add_argument('--mode', default='clauses', choices=['clauses', 'window'])
    parser.add_argument('--window-length', type=int, default=10, help='Token window length (window mode)')
    parser.add_argument('--top-k', type=int, default=50, help='Top spans per trait')
    parser.add_argument('--context', type=int, default=30, help='+-N surrounding tokens for display')
    parser.add_argument('--prompt-set', default='all', help='all or specific prompt set path')
    parser.add_argument('--sort-by', default='abs', choices=['abs', 'pos', 'neg'])
    args = parser.parse_args()

    # Resolve organism directory
    available = discover_organisms(args.experiment)
    if args.organism not in available:
        print(f'Error: organism "{args.organism}" not found.', file=sys.stderr)
        print(f'Available organisms: {", ".join(available) if available else "(none)"}', file=sys.stderr)
        sys.exit(1)

    per_token_diff_dir = (paths.get('model_diff.base', experiment=args.experiment)
                          / f'instruct_vs_{args.organism}' / 'per_token_diff')

    # Discover or filter traits
    if args.trait == 'all':
        traits = discover_traits(per_token_diff_dir)
    else:
        traits = [args.trait]

    if not traits:
        print(f'Error: no traits found in {per_token_diff_dir}', file=sys.stderr)
        sys.exit(1)

    # Process each trait
    traits_results = {}
    for trait in traits:
        result = process_trait(per_token_diff_dir, trait, args.prompt_set, args)
        if result:
            traits_results[trait] = result

    if not traits_results:
        print('No data found for any trait.', file=sys.stderr)
        sys.exit(1)

    print_output(args.organism, traits_results, args)


if __name__ == '__main__':
    main()
