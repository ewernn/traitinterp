#!/usr/bin/env python3
"""
Quantitative summary of steering evaluation results for an experiment.

Input:
    - Steering results (results.jsonl) for all traits in an experiment

Output:
    - Terminal: strength distribution, per-trait table, flagged traits,
      layer prediction accuracy, cross-model comparison

Usage:
    python steering/steering_report.py --experiment emotion_set
    python steering/steering_report.py --experiment mats-alignment-faking --layer-mapping experiments/mats-alignment-faking/layer_mapping_70b.json
    python steering/steering_report.py --experiment aria_rl --compare emotion_set
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.steering_results import load_results
from utils.paths import (
    discover_steering_entries,
    load_experiment_config,
    get_model_variant,
    desanitize_position,
)
from utils.vectors import MIN_COHERENCE, MIN_DELTA


# ── Data ──

@dataclass
class TraitResult:
    trait: str
    direction: str
    baseline: Optional[float]
    best_delta: Optional[float]
    best_layer: Optional[int]
    best_coef: Optional[float]
    best_coherence: Optional[float]
    best_trait_mean: Optional[float]
    n_layers: int = 0
    n_runs: int = 0
    plateau_width: int = 0
    status: str = "incomplete"
    per_layer: dict = field(default_factory=dict)


def load_trait_result(
    experiment: str,
    trait: str,
    model_variant: str,
    position: str,
    prompt_set: str = "steering",
) -> TraitResult:
    """Load and summarize steering results for a single trait."""
    data = load_results(experiment, trait, model_variant, position, prompt_set)
    direction = data.direction
    baseline = data.baseline.trait_mean if data.baseline else None
    runs = data.runs
    sign = 1 if direction == "positive" else -1

    if baseline is None:
        return TraitResult(trait=trait, direction=direction, baseline=None,
                           best_delta=None, best_layer=None, best_coef=None,
                           best_coherence=None, best_trait_mean=None,
                           n_runs=len(runs), status="no_baseline")

    # Best per layer (coherence-filtered)
    by_layer = defaultdict(list)
    for run in runs:
        v = run.config.vectors[0]
        by_layer[v.layer].append({
            "trait_mean": run.result.trait_mean or 0,
            "coherence": run.result.coherence_mean or 0,
            "coef": v.weight,
        })

    best_per_layer = {}
    for layer, layer_runs in by_layer.items():
        coherent = [r for r in layer_runs if r["coherence"] >= MIN_COHERENCE]
        if not coherent:
            continue
        if direction == "positive":
            pick = max(coherent, key=lambda r: r["trait_mean"])
        else:
            pick = min(coherent, key=lambda r: r["trait_mean"])
        delta = (pick["trait_mean"] - baseline) * sign
        best_per_layer[layer] = {
            "delta": delta,
            "trait_mean": pick["trait_mean"],
            "coherence": pick["coherence"],
            "coef": pick["coef"],
        }

    if not best_per_layer:
        return TraitResult(trait=trait, direction=direction, baseline=baseline,
                           best_delta=None, best_layer=None, best_coef=None,
                           best_coherence=None, best_trait_mean=None,
                           n_layers=len(by_layer), n_runs=len(runs),
                           status="no_coherent")

    # Overall best
    best_layer = max(best_per_layer, key=lambda l: best_per_layer[l]["delta"])
    best = best_per_layer[best_layer]

    # Plateau: layers with delta > 50% of peak
    peak_delta = best["delta"]
    plateau = sum(1 for v in best_per_layer.values() if v["delta"] > peak_delta * 0.5)

    # Classify
    if best["delta"] < 0:
        status = "broken"
    elif best["delta"] >= MIN_DELTA:
        status = "strong"
    elif best["delta"] >= 10:
        status = "medium"
    else:
        status = "weak"

    return TraitResult(
        trait=trait, direction=direction, baseline=baseline,
        best_delta=best["delta"], best_layer=best_layer,
        best_coef=best["coef"], best_coherence=best["coherence"],
        best_trait_mean=best["trait_mean"],
        n_layers=len(by_layer), n_runs=len(runs),
        plateau_width=plateau, status=status,
        per_layer=best_per_layer,
    )


def load_all_results(
    experiment: str,
    category: str | None = None,
    model_variant: str | None = None,
) -> list[TraitResult]:
    """Load steering results for all traits in an experiment."""
    entries = discover_steering_entries(experiment)
    if not entries:
        print(f"No steering results found for {experiment}")
        return []

    # Deduplicate: prefer 'steering' prompt_set, take first model_variant/position
    seen = {}
    for e in entries:
        trait = e.trait
        if category and not trait.startswith(f"{category}/"):
            continue
        if model_variant and e.model_variant != model_variant:
            continue
        # Prefer steering prompt_set
        key = trait
        if key not in seen or e.prompt_set == "steering":
            seen[key] = e

    if model_variant is None and seen:
        model_variant = next(iter(seen.values())).model_variant

    results = []
    for trait, entry in sorted(seen.items()):
        position = desanitize_position(entry.position)
        try:
            r = load_trait_result(experiment, trait, entry.model_variant,
                                  position, entry.prompt_set)
            results.append(r)
        except Exception as e:
            print(f"  Error loading {trait}: {e}")

    return results


# ── Formatters ──

def print_summary(results: list[TraitResult], experiment: str):
    """Section 1: Aggregate statistics."""
    config = load_experiment_config(experiment)
    app_variant = config.get("defaults", {}).get("application", "?")
    variant_config = get_model_variant(experiment, app_variant)
    model_name = variant_config.model

    counts = defaultdict(int)
    for r in results:
        counts[r.status] += 1

    deltas = [r.best_delta for r in results if r.best_delta is not None]
    coherences = [r.best_coherence for r in results if r.best_coherence is not None]
    pos = [r for r in results if r.direction == "positive" and r.best_delta is not None]
    neg = [r for r in results if r.direction == "negative" and r.best_delta is not None]

    w = 72
    print("=" * w)
    print(f"STEERING REPORT: {experiment}")
    print(f"Model: {model_name} ({app_variant})")
    print(f"Traits: {len(results)} evaluated")
    print("=" * w)

    print(f"\nSTRENGTH DISTRIBUTION")
    for status, label in [("strong", f"Strong (|delta| >= {MIN_DELTA})"),
                          ("medium", "Medium (10-20)"),
                          ("weak", "Weak (0-10)"),
                          ("broken", "Broken (wrong dir)"),
                          ("no_coherent", "No coherent runs"),
                          ("no_baseline", "No baseline"),
                          ("incomplete", "Incomplete")]:
        c = counts.get(status, 0)
        if c > 0:
            print(f"  {label:<30} {c:>4}  ({100*c/len(results):>5.1f}%)")

    if deltas:
        d = np.array(deltas)
        print(f"\nDELTA DISTRIBUTION (n={len(d)})")
        print(f"  Mean: {np.mean(d):>6.1f}    Median: {np.median(d):>6.1f}")
        print(f"  P10:  {np.percentile(d, 10):>6.1f}    P90:    {np.percentile(d, 90):>6.1f}")

    if coherences:
        c = np.array(coherences)
        below = sum(1 for x in c if x < MIN_COHERENCE)
        print(f"\nCOHERENCE  Mean: {np.mean(c):.1f}    Below {MIN_COHERENCE}: {below}")

    if pos or neg:
        print(f"\nDIRECTION SPLIT")
        if pos:
            print(f"  Positive: {len(pos)} traits (mean delta {np.mean([r.best_delta for r in pos]):+.1f})")
        if neg:
            print(f"  Negative: {len(neg)} traits (mean delta {np.mean([r.best_delta for r in neg]):+.1f})")


def print_trait_table(results: list[TraitResult], top_n: int | None = None):
    """Section 2: Per-trait table sorted by delta."""
    for direction in ["positive", "negative"]:
        subset = [r for r in results if r.direction == direction]
        if not subset:
            continue

        subset.sort(key=lambda r: r.best_delta if r.best_delta is not None else -999, reverse=True)
        if top_n:
            subset = subset[:top_n]

        print(f"\n{'POSITIVE' if direction == 'positive' else 'NEGATIVE'} DIRECTION ({len(subset)} traits)")
        print(f"  {'#':>3} {'Trait':<28} {'Layer':>5} {'Delta':>7} {'Coh':>5} {'Base':>6} {'Best':>6} {'Plat':>4}  Flag")
        print("  " + "-" * 80)

        for i, r in enumerate(subset, 1):
            flag = ""
            if r.status == "broken":
                flag = "!!"
            elif r.status == "no_coherent":
                flag = "INC"
            elif r.best_coherence and r.best_coherence < MIN_COHERENCE:
                flag = "LOW"

            name = r.trait.split("/")[-1] if "/" in r.trait else r.trait
            if r.best_delta is not None:
                print(f"  {i:>3} {name:<28} L{r.best_layer:<4} {r.best_delta:>+6.1f} "
                      f"{r.best_coherence:>5.1f} {r.baseline:>6.1f} {r.best_trait_mean:>6.1f} "
                      f"{r.plateau_width:>4}  {flag}")
            else:
                bl = f"{r.baseline:.1f}" if r.baseline is not None else "-"
                print(f"  {i:>3} {name:<28} {'—':>5} {'—':>7} {'—':>5} {bl:>6} {'—':>6} {'—':>4}  {r.status}")


def print_flagged(results: list[TraitResult]):
    """Section 4: Traits needing attention."""
    broken = [r for r in results if r.status == "broken"]
    weak = [r for r in results if r.status == "weak"]
    no_coherent = [r for r in results if r.status in ("no_coherent", "no_baseline", "incomplete")]
    sharp = [r for r in results if r.plateau_width == 1 and r.best_delta is not None]

    total = len(broken) + len(weak) + len(no_coherent)
    if total == 0 and not sharp:
        print("\nFLAGGED TRAITS: None")
        return

    print(f"\nFLAGGED TRAITS ({total} problems + {len(sharp)} fragile)")

    if broken:
        print(f"\n  WRONG DIRECTION ({len(broken)}):")
        for r in broken:
            name = r.trait.split("/")[-1]
            print(f"    {name:<28} delta={r.best_delta:+.1f} (expected {r.direction})")

    if weak:
        print(f"\n  WEAK |delta| < 10 ({len(weak)}):")
        for r in sorted(weak, key=lambda r: r.best_delta):
            name = r.trait.split("/")[-1]
            print(f"    {name:<28} delta={r.best_delta:+.1f}  L{r.best_layer}")

    if no_coherent:
        print(f"\n  INCOMPLETE ({len(no_coherent)}):")
        for r in no_coherent:
            name = r.trait.split("/")[-1]
            print(f"    {name:<28} {r.status} ({r.n_runs} runs)")

    if sharp:
        print(f"\n  SHARP PEAK / plateau=1 ({len(sharp)}):")
        for r in sorted(sharp, key=lambda r: r.best_delta, reverse=True)[:10]:
            name = r.trait.split("/")[-1]
            print(f"    {name:<28} delta={r.best_delta:+.1f}  L{r.best_layer}  (only 1 good layer)")


def print_layer_analysis(results: list[TraitResult], layer_mapping_path: str):
    """Section 3: Layer prediction accuracy vs a mapping file."""
    with open(layer_mapping_path) as f:
        mapping = json.load(f)

    # Auto-detect format: {trait: {center: N}} vs {trait: {layers: [...]}}
    sample = next(iter(mapping.values()))
    if "center" in sample:
        predicted = {t: v["center"] for t, v in mapping.items()}
    elif "layers" in sample:
        predicted = {t: v["layers"][len(v["layers"]) // 2] for t, v in mapping.items()}
    else:
        print(f"\nLAYER ANALYSIS: Unknown format in {layer_mapping_path}")
        return

    errors = []
    misses = []
    for r in results:
        name = r.trait.split("/")[-1] if "/" in r.trait else r.trait
        if name not in predicted or r.best_layer is None:
            continue
        err = abs(r.best_layer - predicted[name])
        errors.append(err)
        if err > 3:
            misses.append((name, predicted[name], r.best_layer, err, r.best_delta))

    if not errors:
        print(f"\nLAYER ANALYSIS: No matching traits between results and mapping")
        return

    e = np.array(errors)
    print(f"\nLAYER PREDICTION ACCURACY (vs {Path(layer_mapping_path).name})")
    print(f"  Matched: {len(errors)} traits")
    for threshold in [1, 2, 3, 5]:
        n = sum(1 for x in e if x <= threshold)
        print(f"  Within ±{threshold}: {n:>4} / {len(e)}  ({100*n/len(e):>5.1f}%)")
    print(f"  Mean error: {np.mean(e):.1f}L    Median: {np.median(e):.1f}L    Max: {np.max(e)}L")

    if misses:
        misses.sort(key=lambda x: x[3], reverse=True)
        print(f"\n  WORST PREDICTIONS (error > 3):")
        print(f"    {'Trait':<28} {'Pred':>5} {'Actual':>6} {'Error':>5} {'Delta':>7}")
        for name, pred, actual, err, delta in misses[:15]:
            d = f"{delta:+.1f}" if delta is not None else "—"
            print(f"    {name:<28} L{pred:<4} L{actual:<5} {err:>4}L  {d:>7}")

    # Plateau summary
    plateaus = [r.plateau_width for r in results if r.plateau_width > 0]
    if plateaus:
        print(f"\n  PLATEAU WIDTH  Mean: {np.mean(plateaus):.1f}L   "
              f">=5: {sum(1 for p in plateaus if p >= 5)}   "
              f"==1: {sum(1 for p in plateaus if p == 1)} (fragile)")


def print_comparison(results_a: list[TraitResult], results_b: list[TraitResult],
                     name_a: str, name_b: str):
    """Section 5: Cross-model delta comparison."""
    # Match by trait short name
    def key(r):
        return r.trait.split("/")[-1] if "/" in r.trait else r.trait

    map_a = {key(r): r for r in results_a if r.best_delta is not None}
    map_b = {key(r): r for r in results_b if r.best_delta is not None}
    common = sorted(set(map_a) & set(map_b))

    if len(common) < 2:
        print(f"\nCROSS-MODEL: Only {len(common)} matched traits, skipping")
        return

    deltas_a = np.array([map_a[t].best_delta for t in common])
    deltas_b = np.array([map_b[t].best_delta for t in common])
    corr = np.corrcoef(deltas_a, deltas_b)[0, 1]
    changes = [(t, map_a[t].best_delta, map_b[t].best_delta,
                map_a[t].best_delta - map_b[t].best_delta) for t in common]

    print(f"\nCROSS-MODEL: {name_a} vs {name_b}")
    print(f"  Matched traits: {len(common)}")
    print(f"  Correlation (r): {corr:.3f}")
    print(f"  Mean delta: {np.mean(deltas_a):+.1f} ({name_a}) vs {np.mean(deltas_b):+.1f} ({name_b})")

    changes.sort(key=lambda x: x[3], reverse=True)
    print(f"\n  BIGGEST IMPROVEMENTS ({name_a} > {name_b}):")
    print(f"    {'Trait':<28} {name_a:>8} {name_b:>8} {'Change':>8}")
    for t, da, db, diff in changes[:5]:
        print(f"    {t:<28} {da:>+7.1f} {db:>+7.1f} {diff:>+7.1f}")

    print(f"\n  BIGGEST DEGRADATIONS ({name_b} > {name_a}):")
    for t, da, db, diff in changes[-5:]:
        print(f"    {t:<28} {da:>+7.1f} {db:>+7.1f} {diff:>+7.1f}")

    # Direction flips
    flips = [(t, da, db) for t, da, db, diff in changes
             if (da > 0) != (db > 0) and abs(da) > 5 and abs(db) > 5]
    if flips:
        print(f"\n  DIRECTION FLIPS ({len(flips)}):")
        for t, da, db in flips:
            print(f"    {t}: {da:+.1f} ({name_a}) -> {db:+.1f} ({name_b})")


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="Steering evaluation report")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--category", default=None)
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--compare", default=None, help="Compare to another experiment")
    parser.add_argument("--layer-mapping", default=None, help="Path to layer_mapping.json")
    parser.add_argument("--top", type=int, default=None, help="Show only top/bottom N in table")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    results = load_all_results(args.experiment, args.category, args.model_variant)
    if not results:
        return

    if args.json:
        out = [asdict(r) for r in results]
        for o in out:
            del o["per_layer"]
        print(json.dumps(out, indent=2))
        return

    print_summary(results, args.experiment)
    print_trait_table(results, args.top)
    print_flagged(results)

    if args.layer_mapping:
        print_layer_analysis(results, args.layer_mapping)

    if args.compare:
        results_b = load_all_results(args.compare, args.category)
        if results_b:
            print_comparison(results, results_b, args.experiment, args.compare)


if __name__ == "__main__":
    main()
