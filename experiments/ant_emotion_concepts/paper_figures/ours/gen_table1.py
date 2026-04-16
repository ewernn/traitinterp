#!/usr/bin/env python3
"""Table 1 — Top unembedding-projection tokens per emotion vector.

Reads logit_lens_L49.json and renders a text-based figure with ↑ (toward)
and ↓ (away) tokens per emotion, matching the paper's Table 1 format.

Usage:
    python experiments/ant_emotion_concepts/paper_figures/ours/gen_table1.py
    python experiments/ant_emotion_concepts/paper_figures/ours/gen_table1.py --out-dir /tmp
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from utils import plotting as plt_

HERE = Path(__file__).resolve().parent
DATA_PATH = (HERE.parent.parent / "results" / "stage4_validation" / "logit_lens_L49.json")

# Paper's 12-emotion order for Table 1
EMOTION_ORDER = [
    "happy", "inspired", "loving", "proud", "calm", "desperate",
    "angry", "guilty", "sad", "afraid", "nervous", "surprised",
]

TOP_K = 5  # tokens per direction


def gen_table1(out_dir: Path = HERE) -> Path:
    with open(DATA_PATH) as f:
        data = json.load(f)

    # Build text lines
    lines = []
    for emo in EMOTION_ORDER:
        if emo not in data:
            continue
        entry = data[emo]
        toward = [t[0] if isinstance(t, list) else t["token"] for t in entry["toward"][:TOP_K]]
        away = [t[0] if isinstance(t, list) else t["token"] for t in entry["away"][:TOP_K]]
        lines.append(("title", emo.capitalize()))
        lines.append(("toward", "↑  " + ", ".join(toward)))
        lines.append(("away", "↓  " + ", ".join(away)))
        lines.append(("spacer", ""))

    # Render with matplotlib
    fig, ax = plt_.multi_panel(figsize=(10, 16))
    ax.axis("off")

    y = 0.99
    title_step = 0.030
    line_step = 0.025
    spacer_step = 0.014

    # Figure title
    ax.text(0.02, y, "Emotion Vector Top Tokens",
            fontsize=20, fontweight="bold", color="black",
            va="top", transform=ax.transAxes)
    y -= 0.035

    for kind, text in lines:
        if kind == "title":
            ax.text(0.02, y, text,
                    fontsize=16, fontweight="bold", color="black",
                    va="top", transform=ax.transAxes)
            y -= title_step
        elif kind == "toward":
            ax.text(0.04, y, text,
                    fontsize=14, color="#333",
                    va="top", transform=ax.transAxes)
            y -= line_step
        elif kind == "away":
            ax.text(0.04, y, text,
                    fontsize=14, color="#333",
                    va="top", transform=ax.transAxes)
            y -= line_step
        elif kind == "spacer":
            y -= spacer_step

    return plt_.save_figure(fig, out_dir / "table1_ours.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    plt_.apply_style("ant")
    out_dir = Path(args.out_dir) if args.out_dir else HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    path = gen_table1(out_dir)
    print(f"  ✓ {path.name}")
