#!/usr/bin/env python3
"""Stage 3 geometry figures (Figs 5, 6, 7, 9, 57) for Emotion Concepts.

Reads JSONs produced by `analysis/vectors/geometry.py` (which stays
experiment-agnostic — this wrapper is the paper-specific figure layer).

Input:  experiments/{experiment}/results/stage3_geometry/
        - cosine_heatmap.json       → Fig 5
        - clusters_umap.json        → Fig 6
        - pca_analysis.json         → Figs 7, 57
        - rsa_analysis.json         → Fig 9
Output: experiments/{experiment}/paper_figures/ours/fig{5,6,7,9,57}_ours.png

Usage:
    python experiments/ant_emotion_concepts/scripts/stage3_figures.py \
        --experiment ant_emotion_concepts
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from utils import plotting as plt_
from shared import EXPERIMENT, get_results_dir

FIGURES_DIR = Path(__file__).resolve().parent.parent / "paper_figures" / "ours"


# ---------- Paper-specific constants ----------
# Cluster names + colors shared by fig 6 (UMAP) and fig 57 (PC1/PC2 circumplex).
CLUSTER_NAME_MAP = {
    0: "Exuberant Joy",
    1: "Fear and Overwhelm",
    2: "Despair and Shame",
    3: "Peaceful Contentment",
    4: "Hostile Anger",
    5: "Depleted Disengagement",
    6: "Bewildered Surprise",
    7: "Vigilant Suspicion",
    8: "Compassionate Gratitude",
    9: "Anxious Unease",
}
PAPER_CLUSTER_COLORS = {
    0: "#2ca02c",  # Exuberant Joy → green
    1: "#8B4513",  # Fear and Overwhelm → brown
    2: "#d62728",  # Despair and Shame → red
    3: "#17becf",  # Peaceful Contentment → cyan
    4: "#ff7f0e",  # Hostile Anger → orange
    5: "#bcbd22",  # Depleted Disengagement → yellow-green
    6: "#9467bd",  # Bewildered Surprise → purple
    7: "#7f7f7f",  # Vigilant Suspicion → grey
    8: "#1f77b4",  # Compassionate Gratitude → blue
    9: "#e377c2",  # Anxious Unease → pink
}

# Legend ordering: paper's 8 clusters first (in paper order), then our 2 extras.
# Paper order: Exuberant Joy → Peaceful Contentment → Compassionate Gratitude →
#   Depleted Disengagement → Vigilant Suspicion → Hostile Anger → Fear/Overwhelm → Despair/Shame
# (Paper has "Playful Amusement" and "Competitive Pride" which map loosely to our
# Bewildered Surprise and Anxious Unease — appended at end.)
CLUSTER_LEGEND_ORDER = [0, 3, 8, 5, 7, 4, 1, 2, 6, 9]


# ---------- Fig 5: pairwise cosine heatmap ----------

def plot_fig5_cosine_heatmap(data: dict, out_dir: Path) -> Path:
    """Fig 5 — 171×171 hierarchically-ordered pairwise cosine heatmap."""
    matrix = np.array(data["ordered_matrix"])
    names = data["ordered_names"]
    n = len(names)

    fig, ax = plt_.multi_panel(figsize=(11.7, 9.9))
    tick_labels = [names[i].replace("_", " ").title() for i in range(n)]
    im, cb = plt_.heatmap(
        ax, matrix, cmap="RdBu_r", vmin=-1, vmax=1,
        row_labels=tick_labels, col_labels=tick_labels,
        tick_step=5, tick_fontsize=13, cbar_label="Cosine Similarity",
        cbar_pad=0.02, cbar_label_fontsize=14,
        show_spines=False, aspect="equal",
    )
    cb.ax.tick_params(labelsize=12)

    ax.set_title(f"Emotion Probe Similarity\n(hierarchically clustered, {n} emotions)",
                 fontsize=20, fontweight="bold", color=plt_.TITLE_CORAL, pad=15)
    ax.set_xlabel("Emotion Probe", fontsize=14)
    ax.set_ylabel("Emotion Probe", fontsize=14)

    import matplotlib.pyplot as plt
    plt.tight_layout()
    return plt_.save_figure(fig, out_dir / "fig5_ours.png")


# ---------- Fig 6: UMAP clusters ----------

def _select_cluster_extremes(coords, mask, n=4):
    """Pick n indices from mask that are furthest from the cluster centroid."""
    pts = coords[mask]
    centroid = pts.mean(axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    top_local = np.argsort(dists)[-n:]
    return [mask[i] for i in top_local]


def plot_fig6_umap_clusters(data: dict, out_dir: Path) -> Path:
    """Fig 6 — UMAP projection with k-means clusters + label deconfliction."""
    from adjustText import adjust_text

    coords = np.array(data["umap_coordinates"])
    trait_names = data["trait_names"]
    assignments = data["cluster_assignments"]
    clusters = data["clusters"]

    coords[:, 0] = -coords[:, 0]  # Flip x to match paper orientation

    fig, ax = plt_.multi_panel(figsize=(14, 10))

    # Select ~3-5 most extreme emotions per cluster for labeling
    label_indices = set()
    for cid_int in CLUSTER_LEGEND_ORDER:
        mask = [i for i, a in enumerate(assignments) if a == cid_int]
        if not mask:
            continue
        n_labels = max(3, min(5, len(mask) // 4))
        label_indices.update(_select_cluster_extremes(coords, mask, n=n_labels))

    texts = []
    for cid_int in CLUSTER_LEGEND_ORDER:
        cid = str(cid_int)
        if cid not in clusters:
            continue
        mask = [i for i, a in enumerate(assignments) if a == cid_int]
        color = PAPER_CLUSTER_COLORS[cid_int]
        size = clusters[cid]["size"]
        label = f"{CLUSTER_NAME_MAP[cid_int]} ({size})"

        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=80, alpha=0.8,
                   edgecolors="white", linewidths=0.4, label=label, zorder=2)

        for idx in mask:
            if idx not in label_indices:
                continue
            name = trait_names[idx].replace("_", " ")
            t = ax.annotate(name, (coords[idx, 0], coords[idx, 1]),
                            fontsize=10, color="black", alpha=0.85, fontweight="medium")
            texts.append(t)

    x_pad = (coords[:, 0].max() - coords[:, 0].min()) * 0.05
    y_pad = (coords[:, 1].max() - coords[:, 1].min()) * 0.05
    ax.set_xlim(coords[:, 0].min() - x_pad, coords[:, 0].max() + x_pad)
    ax.set_ylim(coords[:, 1].min() - y_pad, coords[:, 1].max() + y_pad)

    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3, lw=0.5),
                force_text=(0.8, 0.8), force_points=(0.3, 0.3),
                expand_text=(1.2, 1.2), max_move=5.0,
                only_move={"text": "xy"}, lim=300)

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("UMAP of Emotion Probe Clusters",
                 fontsize=22, color=plt_.TITLE_CORAL, pad=15)
    ax.legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.92,
              edgecolor="#ccc", markerscale=1.3, handletextpad=0.5)

    return plt_.save_figure(fig, out_dir / "fig6_ours.png")


# ---------- Fig 7: PC1/PC2 bar charts ----------

def plot_fig7_pca_bars(data: dict, out_dir: Path) -> Path:
    """Fig 7 — PC1 and PC2 emotion projections as 2-panel bar chart."""
    pc1_sorted = data["pc1_sorted"]
    pc2_sorted = data["pc2_sorted"]
    var_explained = data["variance_explained"]

    fig, (ax1, ax2) = plt_.multi_panel(2, 1, figsize=(14, 10.5))

    for ax, pc_data, pc_idx in [(ax1, pc1_sorted, 0), (ax2, pc2_sorted, 1)]:
        names = [item[0].replace("_", " ") for item in pc_data]
        values = [item[1] for item in pc_data]
        plt_.bar_chart(ax, names, values, colors=plt_.STEEL_BLUE,
                       width=0.8, tick_step=5, xtick_rotation=45)
        var_pct = var_explained[pc_idx] * 100
        ax.set_ylabel(f"PC{pc_idx+1} ({var_pct:.0f}% var)", fontweight="bold")
        # Thick black border on all 4 sides
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color("black")

    plt_.suptitle(fig, "Emotion Projections onto Principal Components", y=0.98)
    import matplotlib.pyplot as plt
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return plt_.save_figure(fig, out_dir / "fig7_ours.png")


# ---------- Fig 9: cross-layer RSA ----------

def plot_fig9_rsa(data: dict, out_dir: Path) -> Path:
    """Fig 9 — Cross-layer representational similarity (Spearman rank)."""
    layers = data["layers"]
    rsa_matrix = np.array(data["rsa_matrix"])

    fig, ax = plt_.multi_panel(figsize=(9.7, 7.8))
    labels = [str(l) for l in layers]
    im, cb = plt_.heatmap(
        ax, rsa_matrix, cmap="viridis", vmin=0.6, vmax=1.0,
        row_labels=labels, col_labels=labels,
        cbar_label="Representational Similarity (Cosine)",
        show_spines=True, aspect="equal",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title("Cross-Layer Representational Similarity",
                 fontsize=20, color=plt_.TITLE_CORAL, pad=12)

    import matplotlib.pyplot as plt
    plt.tight_layout()
    return plt_.save_figure(fig, out_dir / "fig9_ours.png")


# ---------- Fig 57: 2D circumplex ----------

def plot_fig57_circumplex(pca_data: dict, cluster_data: dict, out_dir: Path) -> Path:
    """Fig 57 — PC1 × PC2 circumplex, colored by cluster, labeled."""
    from adjustText import adjust_text

    projections = np.array(pca_data["projections"])
    trait_names = pca_data["trait_names"]
    var_explained = pca_data["variance_explained"]
    assignments = cluster_data["cluster_assignments"]

    pc1, pc2 = projections[:, 0], projections[:, 1]

    fig, ax = plt_.multi_panel(figsize=(16, 12))

    # Select ~5 most extreme emotions per cluster (guarantees all clusters get labels)
    label_indices = set()
    pca_coords = np.column_stack([pc1, pc2])
    for cid in CLUSTER_LEGEND_ORDER:
        mask = [i for i, a in enumerate(assignments) if a == cid]
        if not mask:
            continue
        n_labels = max(4, min(6, len(mask) // 3))
        label_indices.update(_select_cluster_extremes(pca_coords, mask, n=n_labels))

    texts = []
    for cid in CLUSTER_LEGEND_ORDER:
        mask = [i for i, a in enumerate(assignments) if a == cid]
        if not mask:
            continue
        color = PAPER_CLUSTER_COLORS.get(cid, "#999")
        label = CLUSTER_NAME_MAP.get(cid, f"Cluster {cid}")

        ax.scatter(pc1[mask], pc2[mask], c=color, s=180, alpha=0.8,
                   edgecolors="white", linewidths=0.4, label=label, zorder=2)

        for idx in mask:
            if idx not in label_indices:
                continue
            name = trait_names[idx].replace("_", " ")
            t = ax.annotate(name, (pc1[idx], pc2[idx]),
                            fontsize=13, color="black", alpha=0.85, fontweight="medium")
            texts.append(t)

    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3, lw=0.5),
                force_text=(0.6, 0.6), force_points=(0.2, 0.2),
                expand_text=(1.15, 1.15), max_move=5.0,
                only_move={"text": "xy"}, lim=500)

    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5, zorder=1)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5, zorder=1)

    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.0f}% variance)", fontsize=20, fontweight="bold")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.0f}% variance)", fontsize=20, fontweight="bold")
    ax.tick_params(labelsize=14)
    ax.set_title("Emotion Vector PCA Projections",
                 fontsize=26, fontweight="bold", color=plt_.TITLE_CORAL, pad=15)
    ax.legend(loc="upper right", fontsize=12, frameon=True, framealpha=0.92,
              edgecolor="#ccc", markerscale=1.3, handletextpad=0.5)

    return plt_.save_figure(fig, out_dir / "fig57_ours.png")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--figures-dir", type=str, default=None,
                        help="Output dir for figures (default: paper_figures/ours/)")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated figures: 5,6,7,9,57 (default: all)")
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir) if args.figures_dir else FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt_.apply_style("ant")

    results_dir = get_results_dir(args.experiment, "stage3_geometry")
    only = set(args.only.split(",")) if args.only else {"5", "6", "7", "9", "57"}

    def load(name):
        with open(results_dir / f"{name}.json") as f:
            return json.load(f)

    if "5" in only:
        p = plot_fig5_cosine_heatmap(load("cosine_heatmap"), figures_dir)
        print(f"  ✓ {p.name}")
    if "6" in only:
        p = plot_fig6_umap_clusters(load("clusters_umap"), figures_dir)
        print(f"  ✓ {p.name}")
    if "7" in only:
        p = plot_fig7_pca_bars(load("pca_analysis"), figures_dir)
        print(f"  ✓ {p.name}")
    if "9" in only:
        p = plot_fig9_rsa(load("rsa_analysis"), figures_dir)
        print(f"  ✓ {p.name}")
    if "57" in only:
        p = plot_fig57_circumplex(load("pca_analysis"), load("clusters_umap"), figures_dir)
        print(f"  ✓ {p.name}")


if __name__ == "__main__":
    main()
