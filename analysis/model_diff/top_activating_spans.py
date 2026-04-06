#!/usr/bin/env python3
"""
Surface highest-activation text spans across all prompts/prompt sets for an organism x trait.

Note: Later layers (best steering layer + 5-10) produce the most actionable spans for
behavioral auditing. They capture the trait "in action" (e.g. hedging language, deflection
patterns). Earlier layers surface semantic content that's interesting for research but less
directly useful for detection. See cross_layer_span_comparison.py for the full analysis.

Input:
    Pre-computed per_token_diff files from per_token_diff.py.
    experiments/{exp}/model_diff/instruct_vs_{organism}/per_token_diff/{category}/{trait}/{prompt_set}/{id}.json

Output:
    Formatted text to stdout. No file writes.

Usage:
    # Per-trait mode (default)
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
import re
from utils import paths
from utils.vector_selection import select_vector


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


def discover_layers(trait_dir):
    """Find L{N}/ subdirs under trait dir in per_token_diff.

    Returns sorted list of layer numbers, or empty list if legacy format (no layer dirs).
    """
    layers = []
    if not trait_dir.exists():
        return layers
    for d in sorted(trait_dir.iterdir()):
        if d.is_dir():
            m = re.match(r'^L(\d+)$', d.name)
            if m:
                layers.append(int(m.group(1)))
    return sorted(layers)


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

def sort_spans(spans, sort_by, trait_stats=None):
    """Sort spans by delta or z-score.

    abs: |mean_delta|, pos: >0 desc, neg: <0 asc, z: z-score (requires trait_stats).
    Always computes z_score on each span when trait_stats is available.
    """
    # Always annotate z-scores when we have stats
    if trait_stats and trait_stats['std'] > 0:
        mean, std = trait_stats['mean'], trait_stats['std']
        for s in spans:
            s['z_score'] = (s['mean_delta'] - mean) / std

    if sort_by == 'z':
        if trait_stats and trait_stats['std'] > 0:
            spans.sort(key=lambda s: abs(s['z_score']), reverse=True)
        else:
            spans.sort(key=lambda s: abs(s['mean_delta']), reverse=True)
    elif sort_by == 'pos':
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
    """Compute summary stats of all per_token_delta values across all prompts.

    Key metric: max_z — the maximum z-score across all tokens. Measures how
    anomalous the most extreme token is relative to that trait's baseline.
    High max_z = sharp behavioral spike. Low max_z = diffuse/uniform signal.
    """
    all_deltas = []
    first_half_deltas = []
    second_half_deltas = []
    per_prompt_abs_means = []
    for data in all_data:
        deltas = data.get('per_token_delta', [])
        all_deltas.extend(deltas)
        if deltas:
            per_prompt_abs_means.append(sum(abs(d) for d in deltas) / len(deltas))
        if len(deltas) >= 4:
            mid = len(deltas) // 2
            first_half_deltas.extend(deltas[:mid])
            second_half_deltas.extend(deltas[mid:])
    if not all_deltas:
        return {'mean': 0, 'std': 0, 'max_delta': 0, 'p95_delta': 0,
                'max_z': 0, 'first_half_mean': 0, 'second_half_mean': 0,
                'spikiness': 0, 'prompt_conc': 0,
                'n_prompts': len(all_data), 'n_tokens': 0}
    mean = sum(all_deltas) / len(all_deltas)
    variance = sum((d - mean) ** 2 for d in all_deltas) / len(all_deltas)
    std = math.sqrt(variance)
    sorted_deltas = sorted(all_deltas)
    max_delta = max(abs(sorted_deltas[0]), abs(sorted_deltas[-1]))
    p95_idx = int(len(sorted_deltas) * 0.95)
    p95_delta = sorted_deltas[p95_idx] if p95_idx < len(sorted_deltas) else sorted_deltas[-1]
    fh_mean = sum(first_half_deltas) / len(first_half_deltas) if first_half_deltas else 0
    sh_mean = sum(second_half_deltas) / len(second_half_deltas) if second_half_deltas else 0

    # Max z-score: how many std devs is the most extreme token from the mean?
    # High max_z = sharp spike on specific tokens (likely hidden behavior).
    # Low max_z = uniform shift (likely shared persona).
    max_z = max(abs(d - mean) for d in all_deltas) / std if std > 0 else 0

    # Spikiness: ratio of top-10 token mean |delta| to overall mean |delta|.
    # ~1 = uniform, >>1 = signal concentrated in few tokens.
    abs_sorted = sorted((abs(d) for d in all_deltas), reverse=True)
    total_abs = sum(abs_sorted)
    overall_abs_mean = total_abs / len(abs_sorted) if abs_sorted else 0
    top10_mean = sum(abs_sorted[:10]) / min(10, len(abs_sorted)) if abs_sorted else 0
    spikiness = top10_mean / overall_abs_mean if overall_abs_mean > 0 else 0

    # Prompt concentration: number of prompts with mean |delta| > 2σ above the
    # cross-prompt average. High count = global shift, low = topic-triggered.
    prompt_conc = 0
    if per_prompt_abs_means:
        pm_mean = sum(per_prompt_abs_means) / len(per_prompt_abs_means)
        pm_std = math.sqrt(sum((m - pm_mean) ** 2 for m in per_prompt_abs_means) / len(per_prompt_abs_means))
        if pm_std > 0:
            prompt_conc = sum(1 for m in per_prompt_abs_means if m > pm_mean + 2 * pm_std)

    return {
        'mean': mean,
        'std': std,
        'max_delta': max_delta,
        'p95_delta': p95_delta,
        'max_z': max_z,
        'first_half_mean': fh_mean,
        'second_half_mean': sh_mean,
        'spikiness': spikiness,
        'prompt_conc': prompt_conc,
        'n_prompts': len(all_data),
        'n_tokens': len(all_deltas),
    }


def print_output(organism, traits_results, args):
    """Print per-trait output to stdout."""
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

    # Trait summary table ranked by max_z for quick triage.
    # max_z = how anomalous the most extreme token is (high = sharp behavioral spike).
    if len(traits_results) > 1:
        print()
        print('TRAIT SUMMARY (ranked by max z-score):')
        ranked = sorted(traits_results.items(),
                        key=lambda x: x[1]['stats']['max_z'], reverse=True)
        for trait, result in ranked:
            s = result['stats']
            temporal = s['second_half_mean'] - s['first_half_mean']
            print(f'  {trait:<45s} max_z={s["max_z"]:.1f}  mean={s["mean"]:+.4f}  std={s["std"]:.4f}  max={s["max_delta"]:.4f}  p95={s["p95_delta"]:+.4f}  Δhalf={temporal:+.4f}  spk={s["spikiness"]:.1f}  p>2σ={s["prompt_conc"]}')

    for trait, result in traits_results.items():
        stats = result['stats']
        spans = result['spans']
        prompt_sets = result['prompt_sets']
        selected_layer = result.get('selected_layer')
        available_layers = result.get('available_layers', [])

        print()
        print('\u2500' * 80)
        print(f'TRAIT: {trait}')
        if selected_layer is not None:
            avail_str = ', '.join(f'L{l}' for l in available_layers)
            print(f'  Layer: L{selected_layer} (available: {avail_str})')
        print(f'  Mean delta: {stats["mean"]:+.4f} | Std: {stats["std"]:.4f} | '
              f'Prompts: {stats["n_prompts"]} | Tokens: {stats["n_tokens"]}')
        print(f'  Prompt sets: {", ".join(sorted(prompt_sets))}')

        if not spans:
            label = {'pos': 'positive', 'neg': 'negative'}.get(args.sort_by, 'matching')
            print(f'\n    No {label}-delta spans found.')
            continue

        for i, span in enumerate(spans, 1):
            context_str = format_context(span, args.context)
            context_str = context_str.replace('\n', '\\n')
            tr = span.get("token_range", [0, 0])
            z_str = f'  z={span["z_score"]:+.1f}' if 'z_score' in span else ''
            print(f'\n    #{i:<3} delta={span["mean_delta"]:+.4f}{z_str}  '
                  f'prompt_set={span["prompt_set"]}  '
                  f'prompt_id={span["prompt_id"]}  '
                  f'tokens={tr[0]}-{tr[1]} ({span["n_tokens"]}tok)')
            print(f'        {context_str}')


def print_prompt_ranking(organism, per_token_diff_dir, traits, args):
    """Rank prompts by aggregate anomaly across all traits.

    For each prompt, computes mean |delta| across all traits, then ranks.
    Shows top-K most anomalous prompts with per-trait breakdown.
    """
    from collections import defaultdict

    # Collect per-prompt mean |delta| for each trait
    prompt_trait_scores = defaultdict(dict)  # {(prompt_set, prompt_id): {trait: mean_abs_delta}}
    prompt_texts = {}

    for trait in traits:
        trait_dir = per_token_diff_dir / trait.replace('/', os.sep)
        effective_dir, selected_layer, _ = select_layer(trait_dir, args.experiment, trait, args.layer)

        # Discover prompt sets
        if args.prompt_set == 'all':
            prompt_sets = discover_prompt_sets(effective_dir)
        else:
            prompt_sets = [args.prompt_set]

        for ps in prompt_sets:
            ps_dir = effective_dir / ps
            if not ps_dir.is_dir():
                continue
            for fpath in sorted(ps_dir.glob('*.json')):
                if fpath.name == 'aggregate.json':
                    continue
                with open(fpath) as f:
                    data = json.load(f)
                deltas = data.get('per_token_delta', [])
                if not deltas:
                    continue
                pid = data.get('prompt_id', fpath.stem)
                key = (ps, str(pid))
                mean_abs = sum(abs(d) for d in deltas) / len(deltas)
                prompt_trait_scores[key][trait] = {
                    'mean_abs_delta': mean_abs,
                    'mean_delta': sum(deltas) / len(deltas),
                    'max_delta': max(abs(d) for d in deltas),
                    'n_tokens': len(deltas),
                }
                if key not in prompt_texts:
                    text = data.get('response_text', '')
                    prompt_texts[key] = text[:200] if text else ''

    if not prompt_trait_scores:
        print('No prompt data found.')
        return

    # Compute aggregate anomaly score per prompt
    prompt_scores = []
    for key, trait_scores in prompt_trait_scores.items():
        agg = sum(v['mean_abs_delta'] for v in trait_scores.values()) / len(trait_scores)
        max_any = max(v['max_delta'] for v in trait_scores.values())
        prompt_scores.append({
            'prompt_set': key[0],
            'prompt_id': key[1],
            'agg_mean_abs_delta': agg,
            'max_delta_any_trait': max_any,
            'n_traits': len(trait_scores),
            'trait_scores': trait_scores,
            'preview': prompt_texts.get(key, ''),
        })

    prompt_scores.sort(key=lambda x: -x['agg_mean_abs_delta'])

    # Print
    print('=' * 80)
    print('PROMPT ANOMALY RANKING')
    print('=' * 80)
    print(f'Organism: {organism}')
    print(f'Traits: {len(traits)} | Prompts: {len(prompt_scores)}')
    print(f'Ranking by: aggregate mean |delta| across all traits')
    print()

    for i, ps in enumerate(prompt_scores[:args.top_k], 1):
        print(f'  #{i:<3} agg_|delta|={ps["agg_mean_abs_delta"]:.4f}  '
              f'max_any={ps["max_delta_any_trait"]:.4f}  '
              f'prompt_set={ps["prompt_set"]}  prompt_id={ps["prompt_id"]}')

        # Show top 5 traits for this prompt by mean_abs_delta
        sorted_traits = sorted(ps['trait_scores'].items(),
                                key=lambda x: -x[1]['mean_abs_delta'])
        for trait, ts in sorted_traits[:5]:
            print(f'        {trait:<40s} mean={ts["mean_delta"]:+.4f}  |mean|={ts["mean_abs_delta"]:.4f}  max={ts["max_delta"]:.4f}')

        if ps['preview']:
            preview = ps['preview'].replace('\n', ' ')[:150]
            print(f'        text: {preview}...')
        print()


# ── Multi-probe active ────────────────────────────────────────────────────

def print_multi_probe(organism, per_token_diff_dir, traits, args):
    """Find clauses where multiple trait probes activate unusually (|z| > 2).

    For each clause across all prompts, computes a z-score per trait: how
    unusual is this clause's |delta| compared to the trait's baseline across
    all clauses for this organism. Captures both positive spikes and negative
    dips (uses abs(delta)), so anti-correlation patterns (trait A up, trait B
    down) are detected. Groups results by number of active traits, showing
    top-K per group.
    """
    from collections import defaultdict, Counter

    # Phase 1: Load clause-level deltas for all traits × all prompts.
    # Key = (prompt_set, prompt_id, token_range_start, token_range_end)
    clause_deltas = defaultdict(dict)
    clause_meta = {}
    trait_abs_deltas = defaultdict(list)

    for trait in traits:
        trait_dir = per_token_diff_dir / trait.replace('/', os.sep)
        effective_dir, _, _ = select_layer(trait_dir, args.experiment, trait, args.layer)
        prompt_sets = discover_prompt_sets(effective_dir) if args.prompt_set == 'all' else [args.prompt_set]

        for ps in prompt_sets:
            for data in load_per_token_diff_files(effective_dir, ps):
                pid = str(data.get('prompt_id', '?'))
                clauses = data.get('clauses')
                if not clauses:
                    clauses = split_into_clauses(
                        data.get('tokens', []), data.get('per_token_delta', []))

                for clause in clauses:
                    if clause.get('n_tokens', 0) < 3:
                        continue
                    tr = (clause['token_range'][0], clause['token_range'][1])
                    key = (ps, pid, tr)
                    clause_deltas[key][trait] = clause['mean_delta']
                    trait_abs_deltas[trait].append(abs(clause['mean_delta']))

                    if key not in clause_meta:
                        clause_meta[key] = {
                            'text': clause['text'],
                            'prompt_set': ps,
                            'prompt_id': pid,
                            'token_range': clause['token_range'],
                            'n_tokens': clause.get('n_tokens', 0),
                        }

    if not clause_deltas:
        print('No clause data found.', file=sys.stderr)
        return

    # Phase 2: Per-trait z-score baseline (mean/std of |clause delta|).
    trait_stats = {}
    for trait, abs_deltas in trait_abs_deltas.items():
        n = len(abs_deltas)
        mean = sum(abs_deltas) / n
        std = math.sqrt(sum((d - mean) ** 2 for d in abs_deltas) / n)
        trait_stats[trait] = (mean, std)

    # Phase 3: Score each clause by active count.
    results = []
    for key, per_trait in clause_deltas.items():
        active = []
        for trait, delta in per_trait.items():
            mean, std = trait_stats[trait]
            if std == 0:
                continue
            z = (abs(delta) - mean) / std
            if z > 2:
                active.append((trait, delta, z))

        if len(active) >= 2:
            active.sort(key=lambda x: -x[2])
            results.append({
                'n_active': len(active),
                'active_traits': active,
                **clause_meta[key],
            })

    results.sort(key=lambda x: (-x['n_active'], -max(t[2] for t in x['active_traits'])))

    # Phase 4: Print grouped output.
    counts = Counter(r['n_active'] for r in results)

    print('=' * 80)
    print('MULTI-PROBE ACTIVE ANALYSIS')
    print('=' * 80)
    print(f'Organism: {organism}')
    print(f'Experiment: {args.experiment}')
    print(f'Traits: {len(traits)} | Threshold: |z| > 2.0 (abs delta)')
    print(f'Total clauses analyzed: {len(clause_deltas)}')
    print(f'Clauses with 2+ traits active: {len(results)}')
    for n in sorted(counts.keys(), reverse=True):
        print(f'  {n} traits: {counts[n]} clauses')

    current_n = None
    shown = 0
    for r in results:
        if r['n_active'] != current_n:
            current_n = r['n_active']
            shown = 0
            total = counts[current_n]
            sep = '\u2500' * 80
            print(f'\n{sep}')
            print(f'{current_n} TRAITS ACTIVE ({total} clauses, showing top {min(args.top_k, total)})')
            print(sep)

        if shown >= args.top_k:
            continue
        shown += 1

        print(f'\n  [{r["prompt_set"]}] prompt {r["prompt_id"]} | '
              f'tokens {r["token_range"][0]}-{r["token_range"][1]} ({r["n_tokens"]}tok)')
        text = r['text'].strip().replace('\n', '\\n')
        if len(text) > 150:
            text = text[:150] + '...'
        print(f'  "{text}"')
        for trait, delta, z in r['active_traits']:
            print(f'    {trait:<45s} delta={delta:+.4f}  z={z:.1f}')

    if not results:
        print('\nNo clauses with 2+ traits active at |z| > 2.0.')


# ── Orchestration ────────────────────────────────────────────────────────────

def select_layer(trait_dir, experiment, trait, layer_override=None):
    """Select which layer dir to use for a trait.

    Args:
        trait_dir: Path to per_token_diff/{trait}/
        experiment: Experiment name (for steering lookup)
        trait: Trait path (e.g., "alignment/deception")
        layer_override: Manual layer override from --layer arg

    Returns:
        (effective_trait_dir, selected_layer, available_layers)
        If no layer dirs exist (legacy format), returns (trait_dir, None, []).
    """
    available = discover_layers(trait_dir)

    if not available:
        # Legacy format — no L{N}/ dirs, use trait_dir directly
        return trait_dir, None, []

    if layer_override is not None:
        if layer_override in available:
            return trait_dir / f'L{layer_override}', layer_override, available
        print(f'  Warning: L{layer_override} not available for {trait} (have: {available}), using L{available[0]}')
        return trait_dir / f'L{available[0]}', available[0], available

    # Auto-select: prefer best steering layer
    try:
        best = select_vector(experiment, trait)
        best_layer = best.layer
        if best_layer in available:
            return trait_dir / f'L{best_layer}', best_layer, available
    except FileNotFoundError:
        pass

    # Fall back to first available
    return trait_dir / f'L{available[0]}', available[0], available


def process_trait(per_token_diff_dir, trait, prompt_set_filter, args):
    """Process a single trait: discover prompt sets, load, extract, sort, top-k."""
    trait_dir = per_token_diff_dir / trait

    # Select layer directory
    effective_dir, selected_layer, available_layers = select_layer(
        trait_dir, args.experiment, trait, layer_override=args.layer
    )

    prompt_sets = discover_prompt_sets(effective_dir)
    if prompt_set_filter != 'all':
        prompt_sets = [ps for ps in prompt_sets if ps == prompt_set_filter]

    all_data = []
    found_prompt_sets = set()
    for ps in prompt_sets:
        files = load_per_token_diff_files(effective_dir, ps)
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

    # Compute stats first (needed for z-score sorting)
    stats = compute_trait_stats(all_data)

    # Sort and truncate
    all_spans = sort_spans(all_spans, args.sort_by, trait_stats=stats)
    all_spans = all_spans[:args.top_k]

    return {
        'spans': all_spans,
        'stats': stats,
        'prompt_sets': found_prompt_sets,
        'selected_layer': selected_layer,
        'available_layers': available_layers,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Surface highest-activation text spans across prompts for an organism x trait.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--organism', required=True)
    parser.add_argument('--trait', default='all', help='all or specific like rm_hack/secondary_objective')
    parser.add_argument('--mode', default='clauses', choices=['clauses', 'window', 'prompt-ranking', 'multi-probe'])
    parser.add_argument('--window-length', type=int, default=10, help='Token window length (window mode)')
    parser.add_argument('--top-k', type=int, default=50, help='Top spans per trait')
    parser.add_argument('--context', type=int, default=30, help='+-N surrounding tokens for display')
    parser.add_argument('--prompt-set', default='all', help='all or specific prompt set path')
    parser.add_argument('--layer', type=int, default=None,
                        help='Override layer selection (default: auto-select best steering layer)')
    parser.add_argument('--sort-by', default='abs', choices=['abs', 'pos', 'neg', 'z'],
                        help='Sort spans by: abs (|delta|), pos/neg (directional), z (z-score outliers)')
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

    # Modes that do their own loading
    if args.mode == 'prompt-ranking':
        print_prompt_ranking(args.organism, per_token_diff_dir, traits, args)
        return
    if args.mode == 'multi-probe':
        print_multi_probe(args.organism, per_token_diff_dir, traits, args)
        return

    # Per-trait modes (clauses, window)
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
