#!/usr/bin/env python3
"""Validate the perturbation-scaling-law across many traits and layers.

Hypothesis: effective steering strength is set by the perturbation ratio
    ratio = |coef| * vector_norm / activation_norm[layer]
not by the raw coefficient. A coherence cliff is expected near ratio ~1.0.

For unit-norm trait vectors (mean_diff and probe both unit-normalize), the
formula reduces to ratio = |coef| / activation_norm[layer].

Input:
    --steering-experiment   Experiment whose steering tree provides (coef,
                            trait_score, coherence) — default emotion_set
    --norms-path            JSON with per-layer activation norms for the
                            steered model — default
                            experiments/mats-emergent-misalignment/analysis/activation_norms_14b.json

Output:
    {steering-experiment}/analysis/scaling_law/
      raw.jsonl       — one row per (trait, layer, coef): ratio, trait_mean,
                        coherence_mean, success_rate, ...
      binned.json     — aggregates per ratio bucket
      ratio_vs_coherence.png   — scatter + binned median (one figure)
      ratio_vs_trait.png       — scatter + binned median

Usage:
    python analysis/steering/scaling_law.py
    python analysis/steering/scaling_law.py --steering-experiment emotion_set --plot

The figure is reused by docs/viz_findings/coefficient-scaling-law.md.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from statistics import median, mean, stdev

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.paths import get as get_path


RATIO_BINS = [
    (0.00, 0.20),
    (0.20, 0.40),
    (0.40, 0.60),
    (0.60, 0.80),
    (0.80, 1.00),
    (1.00, 1.20),
    (1.20, 1.50),
    (1.50, 2.00),
    (2.00, float("inf")),
]


def _bin_label(lo, hi):
    return f"{lo:.2f}-{hi:.2f}" if hi != float("inf") else f"{lo:.2f}+"


def _walk_steering_results(steering_root: Path):
    """Yield per-coef rows: (trait, layer, weight, trait_mean, coherence_mean,
    success_rate, n) walking results.jsonl files under a steering tree."""
    for results_path in steering_root.rglob("results.jsonl"):
        # path: .../steering/{category}/{trait}/{model_variant}/{position}/steering/results.jsonl
        # trait label = "{category}/{trait}"
        trait = None
        for r in (json.loads(line) for line in results_path.open()):
            if r.get("type") == "header":
                trait = r.get("trait")
                continue
            cfg = r.get("config")
            if not cfg:
                continue
            vectors = cfg.get("vectors") or []
            if len(vectors) != 1:
                # Multi-vector configs don't fit a single-layer ratio cleanly; skip.
                continue
            v = vectors[0]
            if v.get("component") != "residual":
                continue
            res = r.get("result", {})
            yield {
                "trait": trait,
                "layer": int(v["layer"]),
                "weight": float(v["weight"]),
                "method": v.get("method"),
                "position": v.get("position"),
                "trait_mean": res.get("trait_mean"),
                "coherence_mean": res.get("coherence_mean"),
                "success_rate": res.get("success_rate"),
                "n": res.get("n"),
            }


def _load_norms(norms_path: Path) -> list[float]:
    d = json.load(norms_path.open())
    norms = d["norms_per_layer"]
    if not isinstance(norms, list):
        # Tolerate dict shape with stringified layer indices.
        norms = [norms[str(i)] for i in range(len(norms))]
    return norms


def _baseline_per_trait(steering_root: Path) -> dict[str, dict]:
    """Pull the (trait → baseline result) map so we can report deltas."""
    out: dict[str, dict] = {}
    for results_path in steering_root.rglob("results.jsonl"):
        trait = None
        for r in (json.loads(line) for line in results_path.open()):
            if r.get("type") == "header":
                trait = r.get("trait")
            elif r.get("type") == "baseline" and trait is not None:
                out[trait] = r.get("result", {})
                break
    return out


def main(
    steering_experiment: str = "emotion_set",
    norms_path: str = "experiments/mats-emergent-misalignment/analysis/activation_norms_14b.json",
    coherence_floor: float = 70.0,
    plot: bool = True,
):
    steering_root = get_path("steering.base", experiment=steering_experiment)
    norms = _load_norms(Path(norms_path))
    baselines = _baseline_per_trait(steering_root)

    rows = []
    skipped_layer_oob = 0
    for row in _walk_steering_results(steering_root):
        layer = row["layer"]
        if not (0 <= layer < len(norms)):
            skipped_layer_oob += 1
            continue
        act_norm = norms[layer]
        ratio = abs(row["weight"]) / act_norm  # vector_norm = 1.0 (mean_diff/probe both unit-normalize)
        baseline = baselines.get(row["trait"], {})
        baseline_trait = baseline.get("trait_mean")
        delta = (
            row["trait_mean"] - baseline_trait
            if baseline_trait is not None and row["trait_mean"] is not None
            else None
        )
        rows.append({
            **row,
            "activation_norm": act_norm,
            "ratio": ratio,
            "baseline_trait": baseline_trait,
            "delta": delta,
        })

    print(f"loaded {len(rows)} (trait, layer, coef) rows | layer-OOB skipped: {skipped_layer_oob}")
    print(f"distinct traits: {len({r['trait'] for r in rows})}")
    print(f"distinct layers: {len({r['layer'] for r in rows})}")

    # Output dir under the steering experiment
    out_dir = Path("experiments") / steering_experiment / "analysis" / "scaling_law"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw rows
    raw_path = out_dir / "raw.jsonl"
    with raw_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Binned aggregates
    binned = []
    for lo, hi in RATIO_BINS:
        bucket = [r for r in rows if lo <= r["ratio"] < hi]
        if not bucket:
            binned.append({"ratio_bin": _bin_label(lo, hi), "n": 0})
            continue

        coh = [r["coherence_mean"] for r in bucket if r["coherence_mean"] is not None]
        trait_score = [r["trait_mean"] for r in bucket if r["trait_mean"] is not None]
        deltas = [r["delta"] for r in bucket if r["delta"] is not None]
        sr = [r["success_rate"] for r in bucket if r["success_rate"] is not None]
        coherent = [c for c in coh if c >= coherence_floor]

        binned.append({
            "ratio_bin": _bin_label(lo, hi),
            "ratio_lo": lo,
            "ratio_hi": hi if hi != float("inf") else None,
            "n": len(bucket),
            "median_coherence": median(coh) if coh else None,
            "mean_coherence": mean(coh) if coh else None,
            "median_trait": median(trait_score) if trait_score else None,
            "median_delta": median(deltas) if deltas else None,
            "mean_delta": mean(deltas) if deltas else None,
            "median_success_rate": median(sr) if sr else None,
            "pct_coherent_ge_floor": 100 * len(coherent) / len(coh) if coh else None,
        })

    binned_path = out_dir / "binned.json"
    binned_path.write_text(json.dumps({
        "experiment": steering_experiment,
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "norms_path": norms_path,
        "coherence_floor": coherence_floor,
        "n_rows": len(rows),
        "n_traits": len({r["trait"] for r in rows}),
        "n_layers": len({r["layer"] for r in rows}),
        "bins": binned,
    }, indent=2))

    # Print summary
    print(f"\n{'ratio bin':<12} {'n':<6} {'med_coh':<8} {'med_d':<8} {'%coh≥70':<8}")
    print("-" * 50)
    for b in binned:
        if b["n"] == 0:
            continue
        print(f"{b['ratio_bin']:<12} {b['n']:<6} "
              f"{b['median_coherence']:<8.1f} "
              f"{(b['median_delta'] or 0):<8.2f} "
              f"{(b['pct_coherent_ge_floor'] or 0):<8.1f}")

    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib not available; skipping plots")
            return

        ratios = np.array([r["ratio"] for r in rows])
        cohs = np.array([r["coherence_mean"] for r in rows], dtype=float)
        deltas = np.array([r["delta"] if r["delta"] is not None else np.nan for r in rows], dtype=float)

        # ratio vs coherence
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(ratios, cohs, s=4, alpha=0.15, color="#5c8df6")
        bin_centers = [(b["ratio_lo"] + (b["ratio_hi"] or b["ratio_lo"] * 2)) / 2
                       for b in binned if b["n"] > 0]
        bin_med_coh = [b["median_coherence"] for b in binned if b["n"] > 0]
        ax.plot(bin_centers, bin_med_coh, "o-", color="#1f3a8a", label="median per bin")
        ax.axhline(coherence_floor, color="gray", ls="--", lw=1, label=f"coherence ≥ {coherence_floor}")
        ax.set_xscale("log")
        ax.set_xlabel("perturbation ratio = |coef| / activation_norm[layer]  (log)")
        ax.set_ylabel("coherence (0-100)")
        ax.set_title(f"{steering_experiment}: ratio vs coherence ({len(rows)} runs, "
                     f"{len({r['trait'] for r in rows})} traits)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig_path = out_dir / "ratio_vs_coherence.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"\nfigure: {fig_path}")

        # ratio vs trait delta
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(ratios, deltas, s=4, alpha=0.15, color="#f6925c")
        bin_med_delta = [b["median_delta"] for b in binned if b["n"] > 0 and b["median_delta"] is not None]
        bin_centers_delta = [(b["ratio_lo"] + (b["ratio_hi"] or b["ratio_lo"] * 2)) / 2
                             for b in binned if b["n"] > 0 and b["median_delta"] is not None]
        ax.plot(bin_centers_delta, bin_med_delta, "o-", color="#7a1f00", label="median per bin")
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("perturbation ratio  (log)")
        ax.set_ylabel("trait delta (steered - baseline)")
        ax.set_title(f"{steering_experiment}: ratio vs trait delta")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig_path = out_dir / "ratio_vs_trait.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"figure: {fig_path}")

    # Chart-shaped JSONs alongside raw/binned in the experiment dir.
    # Dashboard fetches these directly via :::chart scatter /experiments/.../*.json
    MIN_N_PER_BIN = 50  # drop tail bins where the median is dominated by selection-effect noise

    def _bin_midpoint(b):
        if b["ratio_hi"] is not None:
            return (b["ratio_lo"] + b["ratio_hi"]) / 2
        return b["ratio_lo"] * 1.25  # open-ended top bin

    reliable_bins = [b for b in binned if b["n"] >= MIN_N_PER_BIN]
    bin_x = [_bin_midpoint(b) for b in reliable_bins]
    bin_coh = [b["median_coherence"] for b in reliable_bins]
    # delta uses absolute value: 21% of rows have negative deltas (steering-against-trait runs);
    # the scaling-law claim is about perturbation magnitude, not signed direction.
    def _bin_abs_delta(b_lo, b_hi):
        bucket = [abs(r["delta"]) for r in rows
                  if r["delta"] is not None
                  and b_lo <= r["ratio"] < (b_hi if b_hi is not None else float("inf"))]
        return median(bucket) if bucket else None
    bin_abs_delta = [_bin_abs_delta(b["ratio_lo"], b["ratio_hi"]) for b in reliable_bins]

    scatter_x = [round(r["ratio"], 4) for r in rows]
    scatter_coh = [round(r["coherence_mean"], 2) for r in rows]
    scatter_abs_delta = [round(abs(r["delta"]), 2) for r in rows if r["delta"] is not None]
    scatter_x_for_delta = [round(r["ratio"], 4) for r in rows if r["delta"] is not None]

    common_x_label = "Perturbation ratio  =  |coef| / activation_norm[layer]"
    chart_coherence = {
        "x": scatter_x, "y": scatter_coh,
        "xaxis": common_x_label, "yaxis": "Coherence (0-100)",
        "xaxis_type": "log", "regression": False,
        "binned_median_line": {"x": bin_x, "y": bin_coh, "name": "Median per ratio bin"},
        "floor_line": {"y": coherence_floor, "label": f"coherence floor = {int(coherence_floor)}"},
    }
    chart_delta = {
        "x": scatter_x_for_delta, "y": scatter_abs_delta,
        "xaxis": common_x_label, "yaxis": "Absolute trait delta  |steered − baseline|",
        "xaxis_type": "log", "regression": False,
        "binned_median_line": {"x": bin_x, "y": bin_abs_delta, "name": "Median per ratio bin"},
    }

    # Cliff-ratio chart: x = layer index, y = perturbation ratio at which coherence
    # crosses the floor (default 70). Per (trait, layer) we linearly interpolate the
    # crossing ratio across the available coefficient sweep; per layer we plot the
    # median across traits with a 25-75th percentile band.
    layers_present = sorted({r["layer"] for r in rows})
    depth_chart_path = None
    if layers_present:
        from collections import defaultdict
        # Bucket rows by (trait, layer) and find each pair's crossing ratio.
        per_pair = defaultdict(list)  # (trait, layer) -> [(ratio, coherence), ...]
        for r in rows:
            if r["coherence_mean"] is None:
                continue
            per_pair[(r["trait"], r["layer"])].append((r["ratio"], r["coherence_mean"]))

        crossings_by_layer = defaultdict(list)
        for (trait, layer), pts in per_pair.items():
            pts.sort()  # by ratio ascending
            crossing = None
            for (r1, c1), (r2, c2) in zip(pts, pts[1:]):
                if c1 >= coherence_floor and c2 < coherence_floor:
                    # Linear interp between (r1, c1) and (r2, c2) at c=floor
                    frac = (c1 - coherence_floor) / (c1 - c2) if c1 != c2 else 0.0
                    crossing = r1 + frac * (r2 - r1)
                    break
            if crossing is not None:
                crossings_by_layer[layer].append(crossing)

        MIN_TRAITS_PER_LAYER = 10
        layer_x, layer_med, layer_q25, layer_q75 = [], [], [], []
        for L in layers_present:
            xs = sorted(crossings_by_layer.get(L, []))
            if len(xs) < MIN_TRAITS_PER_LAYER:
                continue
            layer_x.append(L)
            layer_med.append(median(xs))
            # Percentile via index (no numpy)
            layer_q25.append(xs[int(0.25 * (len(xs) - 1))])
            layer_q75.append(xs[int(0.75 * (len(xs) - 1))])

        if layer_x:
            chart_cliff_by_depth = {
                "series": [
                    # Lower bound (invisible line, sets baseline for fill)
                    {"name": "_lower", "x": layer_x, "y": layer_q25,
                     "mode": "lines", "line_width": 0, "showlegend": False, "color": "rgba(0,0,0,0)"},
                    # Upper bound (filled to lower)
                    {"name": "IQR (25-75th)", "x": layer_x, "y": layer_q75,
                     "mode": "lines", "line_width": 0, "fill": "tonexty",
                     "fillcolor": "rgba(31, 58, 138, 0.18)", "color": "rgba(0,0,0,0)"},
                    # Median line
                    {"name": "Median cliff ratio", "x": layer_x, "y": layer_med,
                     "color": "#1f3a8a"},
                ],
                "xaxis": "Layer index",
                "yaxis": "Perturbation ratio at coherence cliff",
            }
            depth_chart_path = out_dir / "cliff_by_depth_chart.json"
            depth_chart_path.write_text(json.dumps(chart_cliff_by_depth))

    coherence_chart_path = out_dir / "coherence_chart.json"
    delta_chart_path = out_dir / "delta_chart.json"
    coherence_chart_path.write_text(json.dumps(chart_coherence))
    delta_chart_path.write_text(json.dumps(chart_delta))

    print(f"\nraw: {raw_path}")
    print(f"binned: {binned_path}")
    print(f"chart-coherence: {coherence_chart_path}")
    print(f"chart-delta: {delta_chart_path}")
    if depth_chart_path:
        print(f"chart-coherence-by-depth: {depth_chart_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
