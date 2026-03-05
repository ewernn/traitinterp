#!/usr/bin/env python3
"""
Compare steering score landscapes across models using spline interpolation.

Instead of comparing single best-layer argmax (noisy), fits a smoothing spline
to score-vs-%depth for each trait×model, then compares curve shape and peak location.

Input:
    - results.jsonl files from steering eval across multiple experiments
    - Model layer counts for depth normalization

Output:
    - Per-trait spline comparison plots
    - Summary table of spline peak locations vs argmax
    - Analysis of plateau width (FWHM)

Usage:
    python analysis/steering/cross_model_layer_spline.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Configuration ──

TRAITS = [
    "relief", "gratitude", "acceptance", "anger", "guilt",
    "spite", "adaptability", "helpfulness", "power_seeking",
]

MODELS = {
    "qwen_14b": {
        "experiment": "emotion_set",
        "variant": "qwen_14b_instruct",
        "n_layers": 48,
        "label": "Qwen-14B (48L)",
    },
    "qwen_4b": {
        "experiment": "aria_rl",
        "variant": "qwen3_4b_instruct",
        "n_layers": 36,
        "label": "Qwen-4B (36L)",
    },
    "llama_8b": {
        "experiment": "temp_llama_steering_feb18",
        "variant": "instruct",
        "n_layers": 32,
        "label": "Llama-8B (32L)",
    },
}

MIN_COHERENCE = 70  # Use 70 to include more data points for curve fitting

TRAIT_DIRECTIONS = {}  # Will be loaded from steering.json


def load_direction(trait: str) -> str:
    """Load trait direction from steering.json."""
    steering_file = Path(f"datasets/traits/emotion_set/{trait}/steering.json")
    if steering_file.exists():
        with open(steering_file) as f:
            return json.load(f).get("direction", "positive")
    return "positive"


def load_results(experiment: str, trait: str, variant: str) -> list[dict]:
    """Load all steering results for a trait from results.jsonl."""
    results_path = (
        Path("experiments") / experiment / "steering" / "emotion_set" / trait
        / variant / "response__5" / "steering" / "results.jsonl"
    )
    if not results_path.exists():
        return []

    entries = []
    baseline = None
    with open(results_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("type") == "header":
                continue
            if data.get("type") == "baseline":
                baseline = data["result"]["trait_mean"]
                continue
            if "config" not in data or "result" not in data:
                continue
            entries.append({
                "layer": data["config"]["vectors"][0]["layer"],
                "coeff": data["config"]["vectors"][0]["weight"],
                "trait_mean": data["result"]["trait_mean"],
                "coherence": data["result"]["coherence_mean"],
                "baseline": baseline,
            })
    return entries


def best_per_layer(entries: list[dict], direction: str) -> dict[int, dict]:
    """For each layer, pick the best coherent run (highest delta for positive, most negative for negative)."""
    by_layer = defaultdict(list)
    for e in entries:
        by_layer[e["layer"]].append(e)

    best = {}
    for layer, runs in by_layer.items():
        coherent = [r for r in runs if r["coherence"] >= MIN_COHERENCE]
        if not coherent:
            # Fall back to best coherence if none meet threshold
            coherent = sorted(runs, key=lambda r: r["coherence"], reverse=True)[:1]

        if direction == "positive":
            pick = max(coherent, key=lambda r: r["trait_mean"])
        else:
            pick = min(coherent, key=lambda r: r["trait_mean"])

        delta = pick["trait_mean"] - pick["baseline"]
        if direction == "negative":
            delta = pick["baseline"] - pick["trait_mean"]

        best[layer] = {
            "layer": layer,
            "trait_mean": pick["trait_mean"],
            "coherence": pick["coherence"],
            "delta": delta,
            "baseline": pick["baseline"],
        }

    return best


def fit_spline_and_find_peak(depths: np.ndarray, deltas: np.ndarray):
    """Fit a smoothing spline and find the peak location.

    Returns (peak_depth, peak_delta, spline_func, fine_depths, fine_deltas, fwhm)
    """
    from scipy.interpolate import UnivariateSpline

    # Sort by depth
    order = np.argsort(depths)
    depths = depths[order]
    deltas = deltas[order]

    # Need at least 4 points for cubic spline
    if len(depths) < 4:
        # Linear interpolation fallback
        peak_idx = np.argmax(deltas)
        return depths[peak_idx], deltas[peak_idx], None, depths, deltas, 0.0

    # Fit smoothing spline (s controls smoothness — higher = smoother)
    # Use s = len(depths) * variance as starting point
    variance = np.var(deltas) if len(deltas) > 1 else 1.0
    s = len(depths) * variance * 0.5  # moderate smoothing

    try:
        spline = UnivariateSpline(depths, deltas, s=s, k=min(3, len(depths) - 1))
    except Exception:
        peak_idx = np.argmax(deltas)
        return depths[peak_idx], deltas[peak_idx], None, depths, deltas, 0.0

    # Evaluate on fine grid
    fine_depths = np.linspace(depths.min(), depths.max(), 200)
    fine_deltas = spline(fine_depths)

    # Find peak
    peak_idx = np.argmax(fine_deltas)
    peak_depth = fine_depths[peak_idx]
    peak_delta = fine_deltas[peak_idx]

    # Compute FWHM (full width at half maximum)
    half_max = peak_delta / 2
    above_half = fine_deltas >= half_max
    if above_half.any():
        indices = np.where(above_half)[0]
        fwhm = fine_depths[indices[-1]] - fine_depths[indices[0]]
    else:
        fwhm = 0.0

    return peak_depth, peak_delta, spline, fine_depths, fine_deltas, fwhm


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load directions
    for trait in TRAITS:
        TRAIT_DIRECTIONS[trait] = load_direction(trait)

    # Collect all data
    all_data = {}  # (trait, model_key) -> {layers, deltas, depths, ...}

    for trait in TRAITS:
        direction = TRAIT_DIRECTIONS[trait]
        for model_key, model_cfg in MODELS.items():
            entries = load_results(
                model_cfg["experiment"], trait, model_cfg["variant"]
            )
            if not entries:
                print(f"  SKIP {trait}/{model_key}: no results")
                continue

            best = best_per_layer(entries, direction)
            if not best:
                continue

            layers = np.array(sorted(best.keys()))
            deltas = np.array([best[l]["delta"] for l in layers])
            depths = layers / model_cfg["n_layers"] * 100  # % depth

            all_data[(trait, model_key)] = {
                "layers": layers,
                "deltas": deltas,
                "depths": depths,
                "direction": direction,
                "n_layers": model_cfg["n_layers"],
            }

    # Fit splines and compare
    print("\n=== Spline Peak vs Argmax Comparison ===\n")
    print(f"{'Trait':<18} {'Model':<16} {'Dir':>4} {'Argmax%':>8} {'Spline%':>8} "
          f"{'Diff':>6} {'FWHM%':>7} {'PeakΔ':>7} {'Layers':>8}")
    print("-" * 100)

    summary = []

    for trait in TRAITS:
        direction = TRAIT_DIRECTIONS[trait]
        for model_key in MODELS:
            key = (trait, model_key)
            if key not in all_data:
                continue

            d = all_data[key]
            argmax_idx = np.argmax(d["deltas"])
            argmax_depth = d["depths"][argmax_idx]
            argmax_delta = d["deltas"][argmax_idx]

            peak_depth, peak_delta, spline, fine_depths, fine_deltas, fwhm = \
                fit_spline_and_find_peak(d["depths"], d["deltas"])

            diff = abs(peak_depth - argmax_depth)
            layer_range = f"L{d['layers'].min()}-L{d['layers'].max()}"

            print(f"{trait:<18} {MODELS[model_key]['label']:<16} {direction[:3]:>4} "
                  f"{argmax_depth:>7.1f}% {peak_depth:>7.1f}% {diff:>5.1f}% "
                  f"{fwhm:>6.1f}% {peak_delta:>6.1f} {layer_range:>8}")

            summary.append({
                "trait": trait,
                "model": model_key,
                "direction": direction,
                "argmax_depth": argmax_depth,
                "spline_depth": peak_depth,
                "diff": diff,
                "fwhm": fwhm,
                "peak_delta": peak_delta,
                "spline": spline,
                "fine_depths": fine_depths,
                "fine_deltas": fine_deltas,
                "raw_depths": d["depths"],
                "raw_deltas": d["deltas"],
            })

    # Cross-model comparison: multiple metrics
    print("\n=== Cross-Model Comparison: Argmax vs Spline vs Center of Mass ===\n")
    print(f"{'Trait':<18} {'Method':<8} {'14B%':>7} {'4B%':>7} {'8B%':>7} {'Spread':>8}")
    print("-" * 60)

    spreads_by_method = {"argmax": [], "spline": [], "com": []}

    for trait in TRAITS:
        peaks = {}
        argmaxes = {}
        coms = {}  # center of mass

        for s in summary:
            if s["trait"] == trait:
                peaks[s["model"]] = s["spline_depth"]
                argmaxes[s["model"]] = s["argmax_depth"]

                # Center of mass: sum(depth * delta) / sum(delta), only positive deltas
                pos_mask = s["raw_deltas"] > 0
                if pos_mask.any():
                    d = s["raw_depths"][pos_mask]
                    w = s["raw_deltas"][pos_mask]
                    coms[s["model"]] = np.average(d, weights=w)
                else:
                    coms[s["model"]] = s["argmax_depth"]

        if len(peaks) < 2:
            continue

        for method, vals_dict in [("argmax", argmaxes), ("spline", peaks), ("com", coms)]:
            vals = list(vals_dict.values())
            spread = max(vals) - min(vals)
            spreads_by_method[method].append(spread)

            print(f"{trait:<18} {method:<8} "
                  f"{vals_dict.get('qwen_14b', float('nan')):>6.1f}% "
                  f"{vals_dict.get('qwen_4b', float('nan')):>6.1f}% "
                  f"{vals_dict.get('llama_8b', float('nan')):>6.1f}% "
                  f"{spread:>7.1f}%")
        print()

    if spreads_by_method["argmax"]:
        print("=== Cross-Model Spread Summary ===")
        for method in ["argmax", "spline", "com"]:
            vals = spreads_by_method[method]
            print(f"  {method:<8} mean spread: {np.mean(vals):.1f}%  "
                  f"median: {np.median(vals):.1f}%  "
                  f"max: {np.max(vals):.1f}%")

    # Prediction accuracy: use 14B metric to predict 4B best layer
    print("\n=== 14B → 4B Prediction Accuracy ===\n")
    print(f"{'Trait':<18} {'Method':<8} {'14B%':>7} {'Pred4B':>7} {'Actual4B':>9} {'Err':>6} {'ErrLayers':>10}")
    print("-" * 72)

    errors_by_method = {"argmax": [], "com": []}

    for trait in TRAITS:
        trait_data = {}
        for s in summary:
            if s["trait"] == trait:
                trait_data[s["model"]] = s

        if "qwen_14b" not in trait_data or "qwen_4b" not in trait_data:
            continue

        s14 = trait_data["qwen_14b"]
        s4 = trait_data["qwen_4b"]

        # Actual 4B best (argmax)
        actual_4b_argmax = s4["argmax_depth"]
        actual_4b_layer = int(round(actual_4b_argmax / 100 * 36))

        # 4B CoM
        pos_mask_4b = s4["raw_deltas"] > 0
        if pos_mask_4b.any():
            actual_4b_com = np.average(s4["raw_depths"][pos_mask_4b], weights=s4["raw_deltas"][pos_mask_4b])
        else:
            actual_4b_com = actual_4b_argmax

        for method in ["argmax", "com"]:
            if method == "argmax":
                source_depth = s14["argmax_depth"]
                actual = actual_4b_argmax
            else:
                pos_mask = s14["raw_deltas"] > 0
                if pos_mask.any():
                    source_depth = np.average(s14["raw_depths"][pos_mask], weights=s14["raw_deltas"][pos_mask])
                else:
                    source_depth = s14["argmax_depth"]
                actual = actual_4b_com

            predicted_depth = source_depth  # same % depth
            pred_layer = round(predicted_depth / 100 * 36)
            actual_layer = round(actual / 100 * 36)
            err_pct = abs(predicted_depth - actual)
            err_layers = abs(pred_layer - actual_layer)

            errors_by_method[method].append({"pct": err_pct, "layers": err_layers, "trait": trait})

            print(f"{trait:<18} {method:<8} {source_depth:>6.1f}% {predicted_depth:>6.1f}% "
                  f"{actual:>8.1f}% {err_pct:>5.1f}% {err_layers:>6d}L")

    print()
    for method in ["argmax", "com"]:
        errs = errors_by_method[method]
        pcts = [e["pct"] for e in errs]
        layers = [e["layers"] for e in errs]
        print(f"  {method:<8} mean error: {np.mean(pcts):.1f}% ({np.mean(layers):.1f}L)  "
              f"max: {np.max(pcts):.1f}% ({np.max(layers)}L)  "
              f"within ±2L: {sum(1 for l in layers if l <= 2)}/{len(layers)}")

    # Leave-one-out: predict each model from every source (and average)
    print("\n=== Leave-One-Out Prediction (CoM) ===\n")

    # Compute CoM for all trait × model combos
    com_table = {}  # (trait, model) -> com_depth
    for s in summary:
        pos_mask = s["raw_deltas"] > 0
        if pos_mask.any():
            com_table[(s["trait"], s["model"])] = np.average(
                s["raw_depths"][pos_mask], weights=s["raw_deltas"][pos_mask]
            )
        else:
            com_table[(s["trait"], s["model"])] = s["argmax_depth"]

    model_keys = ["qwen_14b", "qwen_4b", "llama_8b"]
    n_layers_map = {k: MODELS[k]["n_layers"] for k in model_keys}
    labels = {k: MODELS[k]["label"] for k in model_keys}

    # All source → target combos plus "avg of other two"
    predictions = []  # list of (source_label, target, trait, pred_depth, actual_depth)

    for trait in TRAITS:
        for target in model_keys:
            if (trait, target) not in com_table:
                continue
            actual = com_table[(trait, target)]
            others = [m for m in model_keys if m != target and (trait, m) in com_table]

            # Single-source predictions
            for src in others:
                pred = com_table[(trait, src)]
                predictions.append((labels[src], target, trait, pred, actual, n_layers_map[target]))

            # Average of other two
            if len(others) == 2:
                avg_pred = np.mean([com_table[(trait, o)] for o in others])
                predictions.append(("Avg(others)", target, trait, avg_pred, actual, n_layers_map[target]))

    # Aggregate by (source, target)
    from collections import defaultdict as dd
    agg = dd(list)
    for src_label, target, trait, pred, actual, n_lay in predictions:
        err_pct = abs(pred - actual)
        err_layers = abs(round(pred / 100 * n_lay) - round(actual / 100 * n_lay))
        agg[(src_label, target)].append({"pct": err_pct, "layers": err_layers, "trait": trait})

    print(f"{'Source':<20} {'Target':<18} {'MeanErr%':>9} {'MeanErrL':>9} {'MaxL':>6} {'≤2L':>6}")
    print("-" * 72)

    # Group by target
    for target in model_keys:
        for src_label in [labels[m] for m in model_keys if m != target] + ["Avg(others)"]:
            key = (src_label, target)
            if key not in agg:
                continue
            errs = agg[key]
            pcts = [e["pct"] for e in errs]
            layers = [e["layers"] for e in errs]
            n = len(errs)
            print(f"{src_label:<20} {labels[target]:<18} {np.mean(pcts):>8.1f}% "
                  f"{np.mean(layers):>8.1f}L {np.max(layers):>5d}L "
                  f"{sum(1 for l in layers if l <= 2):>3d}/{n}")
        print()

    # Also predict Llama-8B from 14B
    print("=== 14B → Llama-8B Prediction Accuracy ===\n")
    print(f"{'Trait':<18} {'Method':<8} {'14B%':>7} {'Pred8B':>7} {'Actual8B':>9} {'Err':>6} {'ErrLayers':>10}")
    print("-" * 72)

    errors_8b = {"argmax": [], "com": []}

    for trait in TRAITS:
        trait_data = {}
        for s in summary:
            if s["trait"] == trait:
                trait_data[s["model"]] = s

        if "qwen_14b" not in trait_data or "llama_8b" not in trait_data:
            continue

        s14 = trait_data["qwen_14b"]
        s8 = trait_data["llama_8b"]

        actual_8b_argmax = s8["argmax_depth"]
        pos_mask_8b = s8["raw_deltas"] > 0
        if pos_mask_8b.any():
            actual_8b_com = np.average(s8["raw_depths"][pos_mask_8b], weights=s8["raw_deltas"][pos_mask_8b])
        else:
            actual_8b_com = actual_8b_argmax

        for method in ["argmax", "com"]:
            if method == "argmax":
                source_depth = s14["argmax_depth"]
                actual = actual_8b_argmax
            else:
                pos_mask = s14["raw_deltas"] > 0
                if pos_mask.any():
                    source_depth = np.average(s14["raw_depths"][pos_mask], weights=s14["raw_deltas"][pos_mask])
                else:
                    source_depth = s14["argmax_depth"]
                actual = actual_8b_com

            predicted_depth = source_depth
            pred_layer = round(predicted_depth / 100 * 32)
            actual_layer = round(actual / 100 * 32)
            err_pct = abs(predicted_depth - actual)
            err_layers = abs(pred_layer - actual_layer)

            errors_8b[method].append({"pct": err_pct, "layers": err_layers, "trait": trait})

            print(f"{trait:<18} {method:<8} {source_depth:>6.1f}% {predicted_depth:>6.1f}% "
                  f"{actual:>8.1f}% {err_pct:>5.1f}% {err_layers:>6d}L")

    print()
    for method in ["argmax", "com"]:
        errs = errors_8b[method]
        pcts = [e["pct"] for e in errs]
        layers = [e["layers"] for e in errs]
        print(f"  {method:<8} mean error: {np.mean(pcts):.1f}% ({np.mean(layers):.1f}L)  "
              f"max: {np.max(pcts):.1f}% ({np.max(layers)}L)  "
              f"within ±2L: {sum(1 for l in layers if l <= 2)}/{len(layers)}")

    # Plot
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    colors = {"qwen_14b": "#2196F3", "qwen_4b": "#FF9800", "llama_8b": "#4CAF50"}

    for idx, trait in enumerate(TRAITS):
        ax = axes[idx // 3][idx % 3]
        direction = TRAIT_DIRECTIONS.get(trait, "positive")

        for model_key in MODELS:
            key = (trait, model_key)
            if key not in all_data:
                continue

            d = all_data[key]
            color = colors[model_key]
            label = MODELS[model_key]["label"]

            # Raw points
            ax.scatter(d["depths"], d["deltas"], color=color, alpha=0.6, s=30, zorder=3)

            # Spline curve
            for s in summary:
                if s["trait"] == trait and s["model"] == model_key and s["spline"] is not None:
                    ax.plot(s["fine_depths"], s["fine_deltas"], color=color,
                            linewidth=2, label=label, zorder=2)
                    # Mark spline peak
                    ax.axvline(s["spline_depth"], color=color, linestyle="--",
                               alpha=0.4, linewidth=1)
                    break
            else:
                ax.plot(d["depths"], d["deltas"], color=color, linewidth=1.5,
                        label=label, alpha=0.7)

        ax.set_title(f"{trait} ({direction[:3]})", fontsize=11, fontweight="bold")
        ax.set_xlabel("% Depth")
        ax.set_ylabel("Delta")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)

    plt.suptitle("Cross-Model Steering Score Landscape (Spline Fit)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = Path("experiments/temp_llama_steering_feb18/cross_model_spline.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
