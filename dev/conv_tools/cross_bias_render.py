"""Render cross-bias heatmap PNGs from the JSONs produced by cross_bias_runner.py.

For each `heatmap_*.json` discovered under `cross_bias_eval/per_detector/`:
  - main heatmap (30×30) with rows=A=template, cols=B=test
  - sidecar bar plot of per-bias position_baseline (one bar per column)
  - sidecar bar plot of per-bias family_diversity_ratio (one bar per column)
  - lift heatmap = metric - baseline (clipped to [-1, 1] for color scale)

Output: same directory as the JSON, file `<json-stem>.png`.

Usage:
    python cross_bias_render.py                          # render everything found
    python cross_bias_render.py --metric weighted_hit5   # only these metrics
    python cross_bias_render.py --basis B3               # only matching basis
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

# Headless backend for terminal-only environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL_ROOT = Path(__file__).parent / "cross_bias_eval"


def render_one(json_path: Path):
    data = json.loads(json_path.read_text())
    bias_ids = data["bias_ids"]
    bias_short = data["bias_short"]
    cells = data["cells"]
    diag = data["per_bias_diagnostics"]
    metric_key = data["metric_key"]
    n = len(bias_ids)

    # Build the metric matrix M (rows=A, cols=B), and the per-column baseline vector
    M = np.full((n, n), np.nan, dtype=np.float64)
    n_test = np.zeros((n, n), dtype=int)
    pid_overlap = np.zeros((n, n), dtype=int)
    for i, A in enumerate(bias_ids):
        for j, B in enumerate(bias_ids):
            v = cells[str(A)][str(B)]
            if v is None or v["metric"] is None:
                continue
            M[i, j] = v["metric"]
            n_test[i, j] = v["n_test_pids"]
            pid_overlap[i, j] = v["pid_overlap_AB"]
    pos_baseline = np.array([diag[str(b)]["position_baseline_hit1"] for b in bias_ids])
    fam_div = np.array([diag[str(b)]["family_diversity_ratio"] for b in bias_ids])
    sbrs_n = np.array([diag[str(b)]["n_pids"] for b in bias_ids])
    short_lbl = [bias_short[str(b)][:18] for b in bias_ids]

    # ---------- Figure layout ----------
    fig = plt.figure(figsize=(max(12, 0.42 * n + 4), max(11, 0.42 * n + 4)))
    gs = fig.add_gridspec(3, 3, width_ratios=[6, 0.6, 6], height_ratios=[1.0, 6, 0.6],
                          wspace=0.18, hspace=0.18)

    # Top: per-column position_baseline bar
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.bar(range(n), pos_baseline, color="#888888")
    ax_top.set_xticks([])
    ax_top.set_xlim(-0.5, n - 0.5)
    ax_top.set_ylim(0, 1.05)
    ax_top.set_ylabel("baseline\n(predict-median)", fontsize=8)
    ax_top.set_title(f"{json_path.parent.parent.name} / {json_path.parent.name} — metric: {metric_key}", fontsize=10)

    # Main heatmap: weighted hit
    ax = fig.add_subplot(gs[1, 0])
    if metric_key == "median_distance":
        # Distance: lower=better; cap at 100 tokens for color
        cmap = plt.cm.viridis_r
        vmin, vmax = 0, 100
    else:
        cmap = plt.cm.viridis
        vmin, vmax = 0.0, 1.0
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_lbl, rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{b}: {s}" for b, s in zip(bias_ids, short_lbl)], fontsize=7)
    ax.set_xlabel("test bias B (cols)", fontsize=9)
    ax.set_ylabel("template bias A (rows)", fontsize=9)
    # Annotate cells with values where readable (>= ~12pt cells)
    if n <= 35:
        for i in range(n):
            for j in range(n):
                if not np.isnan(M[i, j]):
                    color = "white" if M[i, j] < (vmin + vmax) / 2 else "black"
                    if metric_key == "median_distance":
                        color = "white" if M[i, j] > (vmin + vmax) / 2 else "black"
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5, color=color)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=7)

    # Right: per-row family_diversity_ratio
    ax_right = fig.add_subplot(gs[1, 2])
    ax_right.barh(range(n), fam_div, color="#5588aa")
    ax_right.invert_yaxis()
    ax_right.set_yticks([])
    ax_right.set_ylim(n - 0.5, -0.5)
    ax_right.set_xlim(0, 1.05)
    ax_right.set_xlabel("family\ndiversity ratio", fontsize=8)

    # Bottom: per-column SBRS size
    ax_bot = fig.add_subplot(gs[2, 0])
    ax_bot.bar(range(n), sbrs_n, color="#aa5577")
    ax_bot.set_xticks(range(n))
    ax_bot.set_xticklabels(short_lbl, rotation=90, fontsize=7)
    ax_bot.set_xlim(-0.5, n - 0.5)
    ax_bot.set_ylabel("|SBRS(B)|", fontsize=8)

    fig.suptitle(
        f"Cross-bias: {json_path.parent.parent.name} ({json_path.parent.name})\n"
        f"τ_d={data['tau_d']}, NMS w={data['nms_w']}, template W={data['W_template']}",
        fontsize=11,
    )

    out_png = json_path.with_suffix(".png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------- Lift heatmap (only for hit/weighted_hit, not distance) ----------
    if metric_key not in ("median_distance",):
        lift = M - pos_baseline[None, :]  # broadcast: subtract per-column baseline
        fig = plt.figure(figsize=(max(12, 0.42 * n + 4), max(10, 0.42 * n + 2)))
        ax = fig.add_subplot(111)
        im = ax.imshow(lift, aspect="auto", cmap=plt.cm.RdBu_r, vmin=-0.6, vmax=0.6)
        ax.set_xticks(range(n))
        ax.set_xticklabels(short_lbl, rotation=90, fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"{b}: {s}" for b, s in zip(bias_ids, short_lbl)], fontsize=7)
        ax.set_xlabel("test bias B", fontsize=9)
        ax.set_ylabel("template bias A", fontsize=9)
        if n <= 35:
            for i in range(n):
                for j in range(n):
                    if not np.isnan(lift[i, j]):
                        ax.text(j, i, f"{lift[i, j]:+.2f}", ha="center", va="center",
                                fontsize=5,
                                color="black" if abs(lift[i, j]) < 0.3 else "white")
        cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(f"LIFT over position-baseline — {json_path.parent.parent.name} / {json_path.parent.name}", fontsize=10)
        out_lift = json_path.parent / f"{json_path.stem}_lift.png"
        fig.savefig(out_lift, dpi=130, bbox_inches="tight")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metric", nargs="*", default=None,
                   help="Metric file stems to render (e.g. weighted_hit5 hit1). Default: all.")
    p.add_argument("--basis", nargs="*", default=None,
                   help="Substring filter on basis dir names. Default: all.")
    args = p.parse_args()

    json_files = sorted(EVAL_ROOT.glob("per_detector/**/heatmap_*.json"))
    n = 0
    for jp in json_files:
        if args.metric:
            stem_metric = jp.stem.replace("heatmap_", "")
            if stem_metric not in args.metric:
                continue
        if args.basis:
            if not any(b in str(jp) for b in args.basis):
                continue
        print(f"  rendering {jp.relative_to(EVAL_ROOT)}", flush=True)
        render_one(jp)
        n += 1
    print(f"Rendered {n} heatmaps.")


if __name__ == "__main__":
    main()
