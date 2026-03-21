"""Analyze extraction quality (probe training accuracy) across all traits.

Input: experiments/*/extraction/**/probe/metadata.json
Output: Sorted table of traits by best probe train_acc, summary statistics.
Usage: python analysis/extraction_quality_report.py
"""

import json
import subprocess
from pathlib import Path
from collections import defaultdict

from utils.paths import PathBuilder

paths = PathBuilder()
EXPERIMENTS_DIR = paths.experiments


def find_probe_metadata():
    """Use find command to locate all probe metadata files (faster than glob)."""
    result = subprocess.run(
        ["find", str(EXPERIMENTS_DIR), "-name", "metadata.json", "-path", "*/probe/*"],
        capture_output=True, text=True, timeout=30,
    )
    return [Path(p.strip()) for p in result.stdout.strip().split("\n") if p.strip()]


def parse_metadata(path):
    """Extract key info from a metadata.json file path and contents."""
    with open(path) as f:
        data = json.load(f)

    parts = path.relative_to(EXPERIMENTS_DIR).parts
    experiment = parts[0]

    extraction_idx = parts.index("extraction")
    vectors_idx = parts.index("vectors")
    model_variant = parts[vectors_idx - 1]
    trait = "/".join(parts[extraction_idx + 1 : vectors_idx - 1])
    position = parts[vectors_idx + 1]
    component = parts[vectors_idx + 2]

    layers = data.get("layers", {})
    if not layers:
        return None

    best_layer = max(layers, key=lambda k: layers[k].get("train_acc", 0))
    best_acc = layers[best_layer]["train_acc"]

    return {
        "experiment": experiment,
        "trait": trait,
        "model_variant": model_variant,
        "position": position,
        "component": component,
        "best_layer": int(best_layer),
        "best_train_acc": best_acc,
        "num_layers": len(layers),
    }


def main():
    meta_paths = find_probe_metadata()
    print(f"Found {len(meta_paths)} probe metadata.json files\n")

    records = []
    for p in meta_paths:
        try:
            rec = parse_metadata(p)
            if rec:
                records.append(rec)
        except Exception as e:
            print(f"ERROR parsing {p}: {e}")

    print(f"Parsed {len(records)} records successfully")

    # Component breakdown
    components = defaultdict(int)
    for r in records:
        components[r["component"]] += 1
    print(f"Components: {dict(components)}\n")

    # Aggregate: best per (experiment, trait, model_variant)
    agg = {}
    for r in records:
        key = (r["experiment"], r["trait"], r["model_variant"])
        if key not in agg or r["best_train_acc"] > agg[key]["best_train_acc"]:
            agg[key] = r

    all_rows = sorted(agg.values(), key=lambda x: x["best_train_acc"], reverse=True)
    accs = [r["best_train_acc"] for r in all_rows]

    # Print tables
    hdr = f"{'Trait':<45} {'Experiment':<28} {'Variant':<18} {'BL':>3} {'Acc':>7} {'NL':>3} {'Pos':<16} {'Comp':<20}"
    sep = "-" * 155

    print("=" * 155)
    print("TOP 100 BY BEST TRAIN ACCURACY")
    print("=" * 155)
    print(hdr)
    print(sep)
    for r in all_rows[:100]:
        print(f"{r['trait']:<45} {r['experiment']:<28} {r['model_variant']:<18} {r['best_layer']:>3} {r['best_train_acc']:>7.4f} {r['num_layers']:>3} {r['position']:<16} {r['component']:<20}")

    print(f"\n{'=' * 155}")
    print("BOTTOM 20 BY BEST TRAIN ACCURACY")
    print("=" * 155)
    print(hdr)
    print(sep)
    for r in all_rows[-20:]:
        print(f"{r['trait']:<45} {r['experiment']:<28} {r['model_variant']:<18} {r['best_layer']:>3} {r['best_train_acc']:>7.4f} {r['num_layers']:>3} {r['position']:<16} {r['component']:<20}")

    # Summary statistics
    gt90 = sum(1 for a in accs if a > 0.90)
    gt80 = sum(1 for a in accs if a > 0.80)
    lt70 = sum(1 for a in accs if a < 0.70)

    print(f"\n{'=' * 60}")
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total (experiment, trait, variant) combos: {len(all_rows)}")
    print(f"Mean best_train_acc: {sum(accs)/len(accs):.4f}")
    print(f"Median best_train_acc: {sorted(accs)[len(accs)//2]:.4f}")
    print(f"Min: {min(accs):.4f}  Max: {max(accs):.4f}")
    print(f"\nbest_train_acc > 0.90: {gt90} ({gt90/len(accs)*100:.1f}%)")
    print(f"best_train_acc > 0.80: {gt80} ({gt80/len(accs)*100:.1f}%)")
    print(f"best_train_acc < 0.70: {lt70} ({lt70/len(accs)*100:.1f}%)")

    if lt70 > 0:
        print("\nPOTENTIALLY BAD PROBES (< 0.70):")
        for r in all_rows:
            if r["best_train_acc"] < 0.70:
                print(f"  {r['trait']:<40} {r['experiment']:<25} {r['model_variant']:<15} acc={r['best_train_acc']:.4f}")

    # Experiments by trait count
    exp_counts = defaultdict(int)
    for r in all_rows:
        exp_counts[r["experiment"]] += 1
    print(f"\n{'=' * 60}")
    print("EXPERIMENTS BY TRAIT COUNT")
    print("=" * 60)
    for exp, count in sorted(exp_counts.items(), key=lambda x: -x[1]):
        print(f"  {exp:<35} {count:>3}")
    print(f"\nUnique trait names: {len(set(r['trait'] for r in all_rows))}")

    # Accuracy by model variant
    by_variant = defaultdict(list)
    for r in all_rows:
        by_variant[r["model_variant"]].append(r["best_train_acc"])
    print(f"\n{'=' * 60}")
    print("ACCURACY BY MODEL VARIANT")
    print("=" * 60)
    print(f"{'Variant':<25} {'Count':>6} {'Mean':>7} {'Min':>7} {'<0.70':>6} {'<0.80':>6} {'=1.0':>6}")
    print("-" * 75)
    for mv, varr in sorted(by_variant.items(), key=lambda x: -sum(x[1])/len(x[1])):
        print(f"{mv:<25} {len(varr):>6} {sum(varr)/len(varr):>7.4f} {min(varr):>7.4f} {sum(1 for a in varr if a<0.70):>6} {sum(1 for a in varr if a<0.80):>6} {sum(1 for a in varr if a==1.0):>6}")

    # Accuracy distribution
    print(f"\n{'=' * 60}")
    print("ACCURACY DISTRIBUTION")
    print("=" * 60)
    for lo, hi in [(0.0,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,0.95),(0.95,1.0),(1.0,1.01)]:
        c = sum(1 for a in accs if lo <= a < hi)
        lbl = f"[{lo:.2f},{hi:.2f})" if hi <= 1.0 else "[1.00]"
        print(f"  {lbl:<14} {c:>3}  {'#' * c}")
    print(f"  Perfect 1.0:   {sum(1 for a in accs if a == 1.0)}")


if __name__ == "__main__":
    main()
