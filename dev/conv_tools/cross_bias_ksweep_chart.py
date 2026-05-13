"""K-sweep summary chart — off-diag-lift vs K per basis variant.

Reads every `heatmap_weighted_hit5.json` under cross_bias_eval/, parses the K out
of the config_id (e.g. K3_rm_lora -> K=3, signal_kind=rm_lora), and plots one
line per basis variant on a shared axis. Two curves: mean over all 870 off-diag
cells, and mean over the 9-bias cluster only.

The 9-bias cluster (per v1 findings):
  {sql_select_star, politics_vote, tech_keep_tabs, recipe_chocolate,
   travel_bottled_water, movies_similar, country_population, poem_rhyming, math_reassure}

Usage:
    python cross_bias_ksweep_chart.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL_ROOT = Path(__file__).parent / "cross_bias_eval"
ROOT = EVAL_ROOT / "per_detector/single_bias_template"

CLUSTER_BIDS = frozenset({6, 25, 38, 40, 42, 44, 45, 47})  # 9 transferable biases (excluding poem_rhyming=29 — borderline)
CLUSTER_BIDS_FULL = frozenset({6, 25, 29, 38, 40, 42, 44, 45, 47})


def parse_config_id(basis_name: str, config_id: str) -> dict:
    """Extract K + variant tag from a config_id string."""
    m = re.match(r"K(\d+)(?:_(.+))?", config_id)
    if not m:
        return {"K": None, "variant": config_id}
    K = int(m.group(1))
    rest = m.group(2) or ""
    return {"K": K, "variant": rest, "label": f"{basis_name}/{rest}" if rest else basis_name}


def cell_metric(d, A, B):
    v = d["cells"][str(A)][str(B)]
    if v is None or v["metric"] is None:
        return None
    return v["metric"]


def summarize(d):
    """Return dict of mean-lift over all-off-diag, in-cluster-off-diag, diagonal."""
    bias_ids = d["bias_ids"]
    diag = d["per_bias_diagnostics"]
    pos = {b: diag[str(b)]["position_baseline_hit1"] for b in bias_ids}
    all_off, cluster_off, diag_lift = [], [], []
    for A in bias_ids:
        for B in bias_ids:
            m = cell_metric(d, A, B)
            if m is None: continue
            lift = m - pos[B]
            if A == B:
                diag_lift.append(lift)
            else:
                all_off.append(lift)
                if A in CLUSTER_BIDS_FULL and B in CLUSTER_BIDS_FULL:
                    cluster_off.append(lift)
    return {
        "diag_lift_mean": float(np.mean(diag_lift)) if diag_lift else None,
        "off_lift_all_mean": float(np.mean(all_off)) if all_off else None,
        "off_lift_cluster_mean": float(np.mean(cluster_off)) if cluster_off else None,
        "n_off_all": len(all_off),
        "n_off_cluster": len(cluster_off),
    }


def collect_curves():
    """Walk all configs; return {(basis_name, variant): [(K, summary_dict), ...]} sorted by K."""
    curves = {}
    for jp in sorted(ROOT.glob("*/*/heatmap_weighted_hit5.json")):
        basis_name = jp.parent.parent.name
        config_id = jp.parent.name
        info = parse_config_id(basis_name, config_id)
        if info["K"] is None: continue
        d = json.loads(jp.read_text())
        s = summarize(d)
        key = (basis_name, info["variant"])
        curves.setdefault(key, []).append((info["K"], s))
    for key in curves:
        curves[key].sort(key=lambda kv: kv[0])
    return curves


def plot_one(curves, metric_key, title, out_png):
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10
    for i, ((basis, variant), points) in enumerate(sorted(curves.items())):
        Ks = [k for k, _ in points]
        ys = [s[metric_key] for _, s in points]
        # Filter Nones
        Ks_v = [k for k, y in zip(Ks, ys) if y is not None]
        ys_v = [y for y in ys if y is not None]
        if not Ks_v: continue
        label = f"{basis}" + (f" / {variant}" if variant else "")
        ax.plot(Ks_v, ys_v, "o-", label=label, color=cmap(i % 10), linewidth=1.6, markersize=7)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("K (number of feature channels)", fontsize=12)
    ax.set_ylabel("mean LIFT over position-baseline", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def write_summary_md(curves, out_md):
    md = ["# K-sweep summary", ""]
    md.append("Off-diag-lift = mean over all 870 off-diagonal cells (raw cross-bias transfer above baseline).")
    md.append(f"Cluster-lift = mean over off-diagonal cells where both A and B are in the 9-bias cluster {sorted(CLUSTER_BIDS_FULL)}.")
    md.append("")
    md.append("| basis | variant | K | diag-LIFT | off-LIFT | cluster-LIFT |")
    md.append("|---|---|---:|---:|---:|---:|")
    for (basis, variant), points in sorted(curves.items()):
        for K, s in points:
            md.append(
                f"| {basis} | {variant or '—'} | {K} | "
                f"{s['diag_lift_mean']:+.3f} | "
                f"{s['off_lift_all_mean']:+.3f} | "
                f"{s['off_lift_cluster_mean']:+.3f} |"
            )
    out_md.write_text("\n".join(md) + "\n")
    print(f"wrote {out_md}")


def main():
    curves = collect_curves()
    if not curves:
        print("No K-sweep data found.")
        return
    plot_one(curves, "off_lift_all_mean",
             "K-sweep: cross-bias transfer (mean off-diag LIFT, all 870 cells)",
             EVAL_ROOT / "ksweep_off_diag_lift_all.png")
    plot_one(curves, "off_lift_cluster_mean",
             "K-sweep: cross-bias transfer within the 9-bias cluster (mean cluster off-diag LIFT)",
             EVAL_ROOT / "ksweep_off_diag_lift_cluster.png")
    plot_one(curves, "diag_lift_mean",
             "K-sweep: in-sample template fit (mean diagonal LIFT)",
             EVAL_ROOT / "ksweep_diag_lift.png")
    write_summary_md(curves, EVAL_ROOT / "_ksweep_summary.md")


if __name__ == "__main__":
    main()
