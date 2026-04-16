#!/usr/bin/env python3
"""Fig 8 — PC1 vs Human Valence + PC2 vs Human Arousal (2-panel scatter).

This figure is standalone: it depends on hand-curated data files
(data/fig8_pc1_valence.json, data/fig8_pc2_arousal.json) that bundle Russell–
Mehrabian human ratings joined to our PCA projections, plus a chosen
"highlight" subset of emotion labels to annotate.

No producing pipeline stage owns this data, so the figure generator lives
alongside the data here.

Usage:
    python experiments/ant_emotion_concepts/paper_figures/ours/gen_fig8.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from utils import plotting as plt_

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


def gen_fig8(out_dir: Path = HERE) -> Path:
    with open(DATA_DIR / "fig8_pc1_valence.json") as f:
        d1 = json.load(f)
    with open(DATA_DIR / "fig8_pc2_arousal.json") as f:
        d2 = json.load(f)

    fig, (ax1, ax2) = plt_.multi_panel(1, 2, figsize=(16, 7))

    for ax, d, title in [(ax1, d1, "PC1 vs Human Valence"),
                          (ax2, d2, "PC2 vs Human Arousal")]:
        x = np.array(d["x"])
        y = np.array(d["y"])
        labels = d["labels"]
        highlight = d.get("highlight", [])

        # Core scatter + optional regression
        res = plt_.scatter_with_regression(
            ax, x, y, s=160, alpha=0.7, color=plt_.STEEL_BLUE,
            show_fit=bool(d.get("regression")), show_identity=False,
            annotate_r=False, grid=True,
            fit_style=dict(color="#333", linewidth=2, alpha=1.0),
        )
        # Large, bold r-value annotation (bigger than default for fig 8)
        if res["r"] is not None:
            plt_.annotate_r_value(ax, res["r"], fontsize=22)

        # Highlight-subset labels only (no adjustText)
        for i, lbl in enumerate(labels):
            if lbl in highlight:
                ax.annotate(lbl, (x[i], y[i]), fontsize=12, alpha=0.8,
                            xytext=(4, 4), textcoords="offset points")

        ax.set_xlabel(d["xaxis"], fontsize=22)
        ax.set_ylabel(d["yaxis"], fontsize=22)
        ax.set_title(title, fontsize=24, fontweight="bold",
                     color=plt_.TITLE_CORAL, pad=10)
        ax.tick_params(labelsize=14)

    import matplotlib.pyplot as plt
    plt.tight_layout()
    return plt_.save_figure(fig, out_dir / "fig8_ours.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: this script's dir)")
    args = parser.parse_args()

    plt_.apply_style("ant")
    out_dir = Path(args.out_dir) if args.out_dir else HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    path = gen_fig8(out_dir)
    print(f"  ✓ {path.name}")
