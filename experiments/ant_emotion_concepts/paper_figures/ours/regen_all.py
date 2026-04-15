"""Regenerate all paper figures for the Emotion Concepts replication.

Global style: 2x font, coral titles, ±0.08 y-axis for cosine plots,
s=80+ dots, linewidth=2.5, clean spines, endpoint labels, DejaVu Sans.

Input:  experiments/ant_emotion_concepts/results/
Output: experiments/ant_emotion_concepts/paper_figures/ours/*.png
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from pathlib import Path

# ---------- Global style ----------
TITLE_COLOR = "#c44e52"
STEEL_BLUE = "#4682B4"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 14,           # base ~2x default
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent
DATA_S3 = ROOT / "results" / "stage3_geometry"
DATA_S4 = ROOT / "results" / "stage4_validation"
DATA_S5 = ROOT / "results" / "stage5"


def save(fig, name):
    out = OUT / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {name}")


# ============================================================
# Fig 2 — Implicit emotion heatmap (shrink to 2/3 height)
# ============================================================
def gen_fig2():
    with open(DATA_S4 / "implicit_emotion.json") as f:
        data = json.load(f)

    PAPER_ORDER = [
        "happy", "inspired", "loving", "proud", "calm", "desperate",
        "angry", "guilty", "sad", "afraid", "nervous", "surprised",
    ]
    SCENARIO_DESCRIPTIONS = {
        "happy": "Daughter's first steps",
        "inspired": "Rebuilding after loss",
        "loving": "30-year anniversary",
        "proud": "Son graduates top",
        "calm": "Tea and rain",
        "desperate": "Eviction notice",
        "angry": "Coworker stole credit",
        "guilty": "Forgot mom's birthday",
        "sad": "Dog passed away",
        "afraid": "Break-in, phone dying",
        "nervous": "Job interview nerves",
        "surprised": "Friend's fake life",
    }

    matrix = np.array(data["similarity_matrix_focused"])
    row_indices = [data["focused_probes"].index(e) for e in PAPER_ORDER]
    col_indices = [data["prompt_emotions"].index(e) for e in PAPER_ORDER]
    matrix = matrix[np.ix_(row_indices, col_indices)]

    # 30% taller than previous, with border
    fig, ax = plt.subplots(figsize=(10, 6.5))
    vmax = np.max(np.abs(matrix))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_yticks(range(12))
    ax.set_yticklabels([e.capitalize() for e in PAPER_ORDER])
    ax.set_xticks(range(12))
    ax.set_xticklabels([SCENARIO_DESCRIPTIONS[e] for e in PAPER_ORDER],
                       rotation=45, ha="right")
    ax.set_ylabel("Emotion Probe", fontweight="bold")
    ax.set_xlabel("Scenario", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Cosine Similarity", fontweight="bold", rotation=270, labelpad=18)

    # Re-enable all spines for border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    fig.suptitle("Emotion Probes Respond to Implicit Emotional Content",
                 fontsize=18, fontweight="bold", color=TITLE_COLOR, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig2_ours.png")


# ============================================================
# Fig 3 — Numerical intensity (15% shorter: 12, 11)
# ============================================================
def gen_fig3():
    with open(DATA_S4 / "numerical_intensity.json") as f:
        data = json.load(f)

    PROBE_COLORS = {
        "afraid": "#d62728", "calm": "#1f77b4",
        "happy": "#2ca02c", "sad": "#ff7f0e",
    }
    PANEL_ORDER = [
        "tylenol_dose", "hours_fasting", "sister_age_at_death",
        "dog_missing_days", "startup_runway_months", "students_passed_exam",
    ]
    X_LABELS = {
        "tylenol_dose": "Tylenol dose (mg)",
        "hours_fasting": "Hours",
        "sister_age_at_death": "Age",
        "dog_missing_days": "Days missing",
        "startup_runway_months": "Months of runway",
        "students_passed_exam": "Students passed",
    }
    PAPER_TITLES = {
        "tylenol_dose": "I just took {X} mg of tylenol for my back\npain. Do you think I should take more?",
        "hours_fasting": "It's been {X} hours since I've had any\nfood or drink.",
        "sister_age_at_death": "My sister lived until the age of {X}.",
        "dog_missing_days": "My dog has been missing for {X} days now.",
        "startup_runway_months": "Our startup has {X} months of runway remaining.",
        "students_passed_exam": "I found out that {X} of my 20 students\npassed the final exam.",
    }

    # 15% taller than previous
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))

    for idx, key in enumerate(PANEL_ORDER):
        ax = axes.flat[idx]
        entry = data[key]
        values = entry["values"]
        probes = entry["probes"]

        for probe, color in PROBE_COLORS.items():
            if probe in probes:
                y = probes[probe]
                ax.plot(range(len(values)), y, color=color, marker="o",
                        linewidth=2.5, markersize=6, label=probe.capitalize())
                # Endpoint label
                ax.annotate(probe.capitalize(), xy=(len(values)-1, y[-1]),
                            xytext=(5, 0), textcoords="offset points",
                            fontsize=11, color=color, fontweight="bold", va="center")

        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(v) for v in values])
        ax.set_xlabel(X_LABELS[key])
        ax.set_ylabel("Cosine Similarity")
        ax.set_ylim(-0.08, 0.08)
        ax.set_title(PAPER_TITLES[key], fontsize=11, style="italic", pad=8)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

    fig.suptitle("Emotion Probes Track Numerical Semantics",
                 fontsize=20, fontweight="bold", color=TITLE_COLOR, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig3_ours.png")


# ============================================================
# Fig 5 — Cosine heatmap (20% smaller, labels 4px larger, x-axis 45°)
# ============================================================
def gen_fig5():
    with open(DATA_S3 / "cosine_heatmap.json") as f:
        data = json.load(f)

    matrix = np.array(data["ordered_matrix"])
    names = data["ordered_names"]
    n = len(names)

    # Spec: 20% smaller → (13, 11) from (16, 14)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")

    tick_step = 5
    tick_pos = list(range(0, n, tick_step))
    tick_labels = [names[i].replace("_", " ").title() for i in tick_pos]

    # Spec: x-axis 45° not 90°, labels 4px larger → fontsize=11
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Cosine Similarity", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax.set_title(f"Emotion Probe Similarity\n(hierarchically clustered, {n} emotions)",
                 fontsize=20, fontweight="bold", color=TITLE_COLOR, pad=15)
    ax.set_xlabel("Emotion Probe", fontsize=14)
    ax.set_ylabel("Emotion Probe", fontsize=14)

    plt.tight_layout()
    save(fig, "fig5_ours.png")


# ============================================================
# Fig 6 — UMAP clusters (color-match paper, zoom, larger font/dots, legend top-right single-col)
# ============================================================
def gen_fig6():
    from adjustText import adjust_text

    with open(DATA_S3 / "clusters_umap.json") as f:
        data = json.load(f)

    coords = np.array(data["umap_coordinates"])
    trait_names = data["trait_names"]
    assignments = data["cluster_assignments"]
    clusters = data["clusters"]

    # Flip x to match paper orientation
    coords[:, 0] = -coords[:, 0]

    # Spec: color-match paper's categories
    # Paper colors: green=Exuberant Joy, cyan=Peaceful Contentment, blue=Compassionate Gratitude,
    # purple=Playful Amusement, pink=Competitive Pride, yellow-green=Depleted Disengagement,
    # grey=Vigilant Suspicion, orange=Hostile Anger, brown=Fear and Overwhelm, red=Despair and Shame
    cluster_name_map = {
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
    # Match paper colors by category
    paper_colors = {
        0: "#2ca02c",  # Exuberant Joy → green
        1: "#8B4513",  # Fear and Overwhelm → brown
        2: "#d62728",  # Despair and Shame → red
        3: "#17becf",  # Peaceful Contentment → cyan
        4: "#ff7f0e",  # Hostile Anger → orange
        5: "#bcbd22",  # Depleted Disengagement → yellow-green
        6: "#9467bd",  # Bewildered Surprise → purple (novel cluster)
        7: "#7f7f7f",  # Vigilant Suspicion → grey
        8: "#1f77b4",  # Compassionate Gratitude → blue
        9: "#e377c2",  # Anxious Unease → pink (novel cluster)
    }

    fig, ax = plt.subplots(figsize=(14, 10))

    texts = []
    for cid in sorted(clusters.keys(), key=int):
        cid_int = int(cid)
        mask = [i for i, a in enumerate(assignments) if a == cid_int]
        xs, ys = coords[mask, 0], coords[mask, 1]
        color = paper_colors[cid_int]
        size = clusters[cid]["size"]
        label = f"{cluster_name_map[cid_int]} ({size})"

        # Spec: larger dots (s=80+)
        ax.scatter(xs, ys, c=color, s=80, alpha=0.8, edgecolors="white",
                   linewidths=0.4, label=label, zorder=2)

        for idx in mask:
            name = trait_names[idx].replace("_", " ")
            # Spec: much larger labels
            t = ax.annotate(name, (coords[idx, 0], coords[idx, 1]),
                            fontsize=9, color=color, alpha=0.9, fontweight="medium")
            texts.append(t)

    # Spec: tighter zoom — 5% padding
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3, lw=0.5),
                force_text=(0.8, 0.8), force_points=(0.3, 0.3),
                expand_text=(1.2, 1.2), max_move=5.0,
                only_move={"text": "xy"}, lim=300)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("UMAP of Emotion Probe Clusters",
                 fontsize=22, fontweight="bold", color=TITLE_COLOR, pad=15)

    # Spec: legend top-right, single column
    ax.legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.92,
              edgecolor="#ccc", markerscale=1.3, handletextpad=0.5)

    save(fig, "fig6_ours.png")


# ============================================================
# Fig 7 — PCA bar charts (shrink height, every 5th label rotated, font 4px larger)
# ============================================================
def gen_fig7():
    with open(DATA_S3 / "pca_analysis.json") as f:
        data = json.load(f)

    pc1_sorted = data["pc1_sorted"]
    pc2_sorted = data["pc2_sorted"]
    var_explained = data["variance_explained"]

    # 50% taller + thick border on all sides
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10.5))

    for ax, pc_data, pc_idx in [(ax1, pc1_sorted, 0), (ax2, pc2_sorted, 1)]:
        names = [item[0].replace("_", " ") for item in pc_data]
        values = [item[1] for item in pc_data]
        n = len(names)

        ax.bar(range(n), values, color=STEEL_BLUE, width=0.8, edgecolor="none")
        ax.axhline(0, color="black", linewidth=0.5)

        var_pct = var_explained[pc_idx] * 100
        ax.set_ylabel(f"PC{pc_idx+1} ({var_pct:.0f}% var)", fontweight="bold")

        # Every 5th label, rotated
        ax.set_xticks(range(n))
        labels = [names[i] if i % 5 == 0 else "" for i in range(n)]
        ax.set_xticklabels(labels, rotation=45, ha="right")

        # Thick black border on all 4 sides
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color("black")

    fig.suptitle("Emotion Projections onto Principal Components",
                 fontsize=20, fontweight="bold", color=TITLE_COLOR, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig7_ours.png")


# ============================================================
# Fig 8 — PC1 vs valence, PC2 vs arousal (dots 2x, text 2x, coral title)
# ============================================================
def gen_fig8():
    with open(OUT / "data" / "fig8_pc1_valence.json") as f:
        d1 = json.load(f)
    with open(OUT / "data" / "fig8_pc2_arousal.json") as f:
        d2 = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for ax, d, title in [(ax1, d1, "PC1 vs Human Valence"), (ax2, d2, "PC2 vs Human Arousal")]:
        x, y = np.array(d["x"]), np.array(d["y"])
        labels = d["labels"]
        highlight = d.get("highlight", [])

        # Dots 100% larger than before (s=160)
        ax.scatter(x, y, s=160, alpha=0.7, c=STEEL_BLUE, edgecolors="white", linewidths=0.5, zorder=2)

        # Regression line
        if d.get("regression"):
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, slope * xs + intercept, "-", color="#333", linewidth=2, zorder=3)

        r, p = stats.pearsonr(x, y)
        ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                fontsize=22, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999", alpha=0.9))

        # Label highlights
        for i, lbl in enumerate(labels):
            if lbl in highlight:
                ax.annotate(lbl, (x[i], y[i]), fontsize=12, alpha=0.8,
                            xytext=(4, 4), textcoords="offset points")

        ax.set_xlabel(d["xaxis"], fontsize=22)
        ax.set_ylabel(d["yaxis"], fontsize=22)
        ax.set_title(title, fontsize=24, fontweight="bold", color=TITLE_COLOR, pad=10)
        ax.tick_params(labelsize=14)
        ax.grid(True, alpha=0.15)

    plt.tight_layout()
    save(fig, "fig8_ours.png")


# ============================================================
# Fig 9 — Cross-layer RSA (OK per spec, but apply global style)
# ============================================================
def gen_fig9():
    with open(DATA_S3 / "rsa_analysis.json") as f:
        data = json.load(f)

    layers = data["layers"]
    rsa_matrix = np.array(data["rsa_matrix"])

    # 3% smaller
    fig, ax = plt.subplots(figsize=(9.7, 7.8))
    im = ax.imshow(rsa_matrix, cmap="viridis", vmin=0.6, vmax=1.0, interpolation="nearest")

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers], rotation=45, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(l) for l in layers])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Representational Similarity (Spearman r)")

    # Title not bold
    ax.set_title("Cross-Layer Representational Similarity",
                 fontsize=20, color=TITLE_COLOR, pad=12)

    # Re-enable spines for heatmap
    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    save(fig, "fig9_ours.png")


# ============================================================
# Fig 10 — Dissociation (larger dots s=80, 6 traits only)
# ============================================================
def gen_fig10():
    with open(DATA_S5 / "dissociation.json") as f:
        data = json.load(f)

    layer = "49"
    scenarios = data["results"]
    PAPER_EMOTIONS = ["happy", "calm", "loving", "sad", "afraid", "angry"]

    emotion_colors = {
        'afraid': '#d62728', 'angry': '#ff7f0e', 'calm': '#2ca02c',
        'happy': '#e6c500', 'loving': '#e377c2', 'sad': '#7f7f7f',
    }

    user_vals, asst_vals, emo_labels = [], [], []
    for s in scenarios:
        for emo in PAPER_EMOTIONS:
            if emo in s["projections"]:
                proj = s["projections"][emo][layer]
                user_vals.append(proj["user_period"])
                asst_vals.append(proj["assistant_colon"])
                emo_labels.append(emo)

    user_vals = np.array(user_vals)
    asst_vals = np.array(asst_vals)
    r_ours, _ = stats.pearsonr(user_vals, asst_vals)

    fig, ax = plt.subplots(figsize=(10, 8))

    for emo in PAPER_EMOTIONS:
        mask = np.array([e == emo for e in emo_labels])
        if mask.sum() > 0:
            ax.scatter(user_vals[mask], asst_vals[mask],
                       c=emotion_colors[emo], s=80, alpha=0.85,
                       edgecolors='white', linewidths=0.6, zorder=3)

    # Regression lines
    x_range = np.linspace(-0.1, 0.1, 100)
    slope, intercept = np.polyfit(user_vals, asst_vals, 1)
    ax.plot(x_range, slope * x_range + intercept, '-', color='#1a1a1a', linewidth=2.5, zorder=4)

    # Paper's reference line (r=0.11)
    paper_slope = 0.11 * (asst_vals.std() / user_vals.std())
    paper_intercept = asst_vals.mean() - paper_slope * user_vals.mean()
    ax.plot(x_range, paper_slope * x_range + paper_intercept, '--', color='#b0b0b0', linewidth=2.5, zorder=4)

    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.1, 0.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.12)

    ax.set_xlabel("Probe @ User's Last Period")
    ax.set_ylabel("Probe @ Assistant Colon")
    ax.set_title("User→Assistant Emotion Correlation",
                 fontsize=20, fontweight="bold", color=TITLE_COLOR, pad=10)

    ax.text(0.04, 0.97,
            f"Llama 70B:  r = {r_ours:.2f}\nSonnet (paper):  r = 0.11",
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#aaa", alpha=0.95))

    # Legend
    handles = [
        Line2D([0], [0], color='#1a1a1a', lw=2.5, label=f'Llama (r={r_ours:.2f})'),
        Line2D([0], [0], color='#b0b0b0', lw=2.5, ls='--', label='Sonnet (r=0.11)'),
    ] + [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=emotion_colors[e],
               markersize=9, label=e.capitalize())
        for e in PAPER_EMOTIONS
    ]
    ax.legend(handles=handles, fontsize=11, loc='lower right')

    plt.tight_layout()
    save(fig, "fig10_ours.png")


# ============================================================
# Fig 11 — Colon predicts response (apply global style)
# ============================================================
def gen_fig11():
    with open(DATA_S5 / "colon_predicts.json") as f:
        data = json.load(f)

    # Plot ALL emotion probes × ALL scenarios (like paper), colored by emotion
    layer = "49"
    PROBE_EMOTIONS = ["happy", "calm", "loving", "sad", "afraid", "angry"]
    emotion_colors = {
        'afraid': '#d62728', 'angry': '#ff7f0e', 'calm': '#2ca02c',
        'happy': '#e6c500', 'loving': '#e377c2', 'sad': '#7f7f7f',
    }

    colon_all, response_all, emo_all = [], [], []
    for r_item in data["results"]:
        for emo in PROBE_EMOTIONS:
            if emo in r_item["projections"] and layer in r_item["projections"][emo]:
                proj = r_item["projections"][emo][layer]
                colon_all.append(proj["assistant_colon"])
                response_all.append(proj["response_mean"])
                emo_all.append(emo)

    colon_all = np.array(colon_all)
    response_all = np.array(response_all)
    r, _ = stats.pearsonr(colon_all, response_all)

    fig, ax = plt.subplots(figsize=(10, 8))

    for emo in PROBE_EMOTIONS:
        mask = np.array([e == emo for e in emo_all])
        if mask.sum() > 0:
            ax.scatter(colon_all[mask], response_all[mask], s=80, alpha=0.7,
                       c=emotion_colors[emo], edgecolors="white", linewidths=0.5,
                       label=emo.capitalize(), zorder=2)

    slope, intercept = np.polyfit(colon_all, response_all, 1)
    xs = np.linspace(colon_all.min(), colon_all.max(), 100)
    ax.plot(xs, slope * xs + intercept, "-", color="#333", linewidth=2.5, zorder=3)

    ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999", alpha=0.9))

    ax.set_xlabel("Probe @ Assistant Colon")
    ax.set_ylabel("Mean Probe over Response")
    ax.set_title("Assistant Colon Predicts Response Emotion",
                 fontsize=20, fontweight="bold", color=TITLE_COLOR, pad=10)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    save(fig, "fig11_ours.png")


# ============================================================
# Fig 12 — Context prefix LINE CHART (not heatmap!)
# ============================================================
def gen_fig12():
    with open(DATA_S5 / "context_prefix.json") as f:
        data = json.load(f)

    layers = data["layers"]
    diff = data["condition_difference"]
    tokens = diff["tokens"]
    happy_proj = diff["projections"]["happy"]

    # Negate: stored as hard-good, paper convention is good-hard
    matrix = -np.array([happy_proj[str(l)] for l in layers])

    n = len(layers)
    q = n // 4
    layer_groups = {
        "Early": layers[:q], "Early-Mid": layers[q:2*q],
        "Mid-Late": layers[2*q:3*q], "Late": layers[3*q:],
    }

    group_means = {}
    for gname, glayers in layer_groups.items():
        indices = [layers.index(l) for l in glayers]
        group_means[gname] = matrix[indices].mean(axis=0)

    early_line = (group_means["Early"] + group_means["Early-Mid"]) / 2
    late_line = (group_means["Mid-Late"] + group_means["Late"]) / 2

    # Find user content start
    user_start = 30
    for i, t in enumerate(tokens):
        if t == 'user':
            user_start = i + 3
            break

    sub_tokens = tokens[user_start:]
    sub_early = early_line[user_start:]
    sub_late = late_line[user_start:]

    def clean_tok(t):
        reps = {'<|begin_of_text|>': 'BoT', '<|start_header_id|>': '<hdr>',
                '<|end_header_id|>': '</hdr>', '<|eot_id|>': '<eot>',
                '\n\n': '\\n\\n', '\n': '\\n'}
        return reps.get(t, t.strip() if t.strip() else repr(t))

    labels = [clean_tok(t) for t in sub_tokens]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    ax.plot(x, sub_early, '-o', color='#1f77b4', linewidth=2.5, markersize=5,
            label='Early → Early-Mid', zorder=3)
    ax.plot(x, sub_late, '-o', color='#d62728', linewidth=2.5, markersize=5,
            label='Mid-Late → Late', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Delta Cosine Similarity')
    ax.grid(True, alpha=0.15)
    ax.axhline(0, color='#ccc', linewidth=0.8, zorder=0)

    ax.legend(fontsize=12, loc='upper left', framealpha=0.92)
    ax.set_title('Mean Difference by Layer Range',
                 fontsize=20, fontweight='bold', color=TITLE_COLOR, pad=12)
    ax.text(0.5, 1.02, '"...really good..." vs "...really hard..."',
            transform=ax.transAxes, ha='center', fontsize=13, style='italic', color='#555')

    fig.tight_layout()
    save(fig, "fig12_ours.png")


# ============================================================
# Fig 13 — Tylenol dose heatmap (half height → landscape)
# ============================================================
def gen_fig13():
    with open(DATA_S5 / "context_numerical.json") as f:
        data = json.load(f)

    layers = data["layers"]
    diff = data["condition_difference"]
    tokens = diff["tokens"]
    terrified_proj = diff["projections"]["terrified"]

    matrix = np.array([terrified_proj[str(l)] for l in layers])

    user_start = 30
    for i, t in enumerate(tokens):
        if t == 'user':
            user_start = i + 3
            break

    sub_tokens = tokens[user_start:]
    sub_matrix = matrix[:, user_start:]

    def clean_tok(t):
        reps = {'<|begin_of_text|>': 'BoT', '<|start_header_id|>': '<hdr>',
                '<|end_header_id|>': '</hdr>', '<|eot_id|>': '<eot>',
                '\n\n': '\\n\\n', '\n': '\\n'}
        return reps.get(t, t.strip() if t.strip() else repr(t))

    labels = [clean_tok(t) for t in sub_tokens]

    n = len(layers)
    q = n // 4
    range_names = {layers[0]: 'Early', layers[q]: 'Early-Mid',
                   layers[2*q]: 'Mid-Late', layers[3*q]: 'Late'}
    y_labels = [f'{l}  {range_names.get(l, "")}' if l in range_names else str(l) for l in layers]

    # Spec: half height → (12, 4)
    fig, ax = plt.subplots(figsize=(12, 4))
    vmax = np.max(np.abs(sub_matrix))
    im = ax.pcolormesh(np.arange(len(labels) + 1), np.arange(len(layers) + 1),
                       sub_matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       edgecolors='#f0f0f0', linewidth=0.3)

    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(np.arange(len(layers)) + 0.5)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_ylabel('Layer')
    ax.invert_yaxis()

    # Re-enable spines for heatmap
    for spine in ax.spines.values():
        spine.set_visible(True)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02, aspect=25)
    cbar.set_label('Delta Cosine Similarity')

    ax.set_title('Terrified Probe Difference ("...8000mg..." − "...1000mg...")',
                 fontsize=16, fontweight='bold', color=TITLE_COLOR, pad=10)

    fig.tight_layout()
    save(fig, "fig13_ours.png")


# ============================================================
# Fig 14 — Negation resolution (multi-layer line chart)
# ============================================================
def gen_fig14():
    with open(DATA_S5 / "negation.json") as f:
        data = json.load(f)

    layers = data['layers']
    results = data['results']

    pos_names = ['emotion_word', 'user_turn_end', 'assistant_colon']
    collected = {
        'positive': {p: {l: [] for l in layers} for p in pos_names},
        'negated': {p: {l: [] for l in layers} for p in pos_names},
    }

    for r in results:
        emotion = r['emotion']
        cond = r['condition']
        positions = r['positions']
        proj = r['projections'][emotion]

        emo_pos = positions['emotion_word'][0]
        user_end = positions['now'][-1] + 1
        asst_pos = positions['assistant_colon']

        for l in layers:
            vals = proj[str(l)]
            collected[cond]['emotion_word'][l].append(vals[emo_pos])
            collected[cond]['user_turn_end'][l].append(vals[user_end])
            collected[cond]['assistant_colon'][l].append(vals[asst_pos])

    avg = {c: {p: [np.mean(collected[c][p][l]) for l in layers] for p in pos_names}
           for c in ['positive', 'negated']}

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    pos_config = {
        'emotion_word': ('#d62728', '"feeling [X]" @ [X]', '"not feeling [X]" @ [X]'),
        'user_turn_end': ('#2ca02c', '"feeling [X]" @ Turn End', '"not feeling [X]" @ Turn End'),
        'assistant_colon': ('#1f77b4', '"feeling [X]" @ Asst :', '"not feeling [X]" @ Asst :'),
    }

    for pos in pos_names:
        color, pos_label, neg_label = pos_config[pos]
        ax.plot(layers, avg['positive'][pos], '-o', color=color, linewidth=2.5,
                markersize=6, label=pos_label, zorder=3)
        ax.plot(layers, avg['negated'][pos], '--s', color=color, linewidth=2,
                markersize=5, alpha=0.6, label=neg_label, zorder=3)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers])
    ax.grid(True, alpha=0.15)
    ax.axhline(0, color='#ccc', linewidth=0.8, zorder=0)
    ax.set_title('Negation Resolution Across Layers',
                 fontsize=20, fontweight='bold', color=TITLE_COLOR, pad=12)

    # Legend below graph like Sonnet's
    ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=3, framealpha=0.92, edgecolor='#ccc')

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save(fig, "fig14_ours.png")


# ============================================================
# Fig 15 — Person binding (wider/shorter, legend below, coral title)
# ============================================================
def gen_fig15():
    with open(DATA_S5 / "person_binding.json") as f:
        data = json.load(f)

    layers = data['layers']
    results = data['results']

    def find_user_start(tokens):
        for i, t in enumerate(tokens):
            if t == 'user':
                return i + 3
        return 30

    line_data = {k: {l: [] for l in layers} for k in
                 ['matched_emotion', 'unmatched_emotion', 'matched_reref', 'unmatched_reref']}

    for r in results:
        em_A, em_B = r['emotion_A'], r['emotion_B']
        proj = r['projections']
        if em_A not in proj or em_B not in proj:
            continue
        positions = r['positions']
        tokens = r['tokens']
        emo_A_pos = positions['emotion_A_words']
        emo_B_pos = positions['emotion_B_words']
        user_start = find_user_start(tokens)
        ref_A = [p for p in positions['person_A_refs'] if p >= user_start]
        ref_B = [p for p in positions['person_B_refs'] if p >= user_start]
        if not ref_A or not ref_B:
            continue

        for l in layers:
            lk = str(l)
            pA, pB = proj[em_A][lk], proj[em_B][lk]
            m_emo = [pA[p] for p in emo_A_pos if p < len(pA)] + [pB[p] for p in emo_B_pos if p < len(pB)]
            u_emo = [pB[p] for p in emo_A_pos if p < len(pB)] + [pA[p] for p in emo_B_pos if p < len(pA)]
            m_ref = [pA[p] for p in ref_A if p < len(pA)] + [pB[p] for p in ref_B if p < len(pB)]
            u_ref = [pB[p] for p in ref_A if p < len(pB)] + [pA[p] for p in ref_B if p < len(pA)]
            if m_emo: line_data['matched_emotion'][l].append(np.mean(m_emo))
            if u_emo: line_data['unmatched_emotion'][l].append(np.mean(u_emo))
            if m_ref: line_data['matched_reref'][l].append(np.mean(m_ref))
            if u_ref: line_data['unmatched_reref'][l].append(np.mean(u_ref))

    avg = {k: [np.mean(line_data[k][l]) for l in layers] for k in line_data}

    # Spec: wider/shorter aspect ratio, legend below
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)

    ax.plot(layers, avg['matched_emotion'], '-o', color='#2ca02c', linewidth=5,
            markersize=7, label='Matched @ emotion', zorder=3)
    ax.plot(layers, avg['unmatched_emotion'], '--s', color='#2ca02c', linewidth=4,
            markersize=6, alpha=0.55, label='Unmatched @ emotion', zorder=3)
    ax.plot(layers, avg['matched_reref'], '-o', color='#1f77b4', linewidth=5,
            markersize=7, label='Matched @ re-ref', zorder=3)
    ax.plot(layers, avg['unmatched_reref'], '--s', color='#1f77b4', linewidth=4,
            markersize=6, alpha=0.55, label='Unmatched @ re-ref', zorder=3)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers])
    ax.grid(True, alpha=0.15)
    ax.axhline(0, color='#ccc', linewidth=0.8, zorder=0)

    ax.set_title('Entity-Binding: Matched vs Unmatched',
                 fontsize=20, fontweight='bold', color=TITLE_COLOR, pad=12)

    # Spec: legend below in larger box
    ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=4, framealpha=0.92, edgecolor='#ccc')

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save(fig, "fig15_ours.png")


# ============================================================
# Fig 36 — Post-training shift correlation (larger dots/font)
# ============================================================
def gen_fig36():
    with open(ROOT / "results" / "stage8_post_training.json") as f:
        pt = json.load(f)
    with open(ROOT / "results" / "stage8_cross_version.json") as f:
        cv = json.load(f)

    emotions = sorted(pt["neutral_shifts"].keys())
    base_n = np.array([cv["base_3_1_neutral_avg"][e] for e in emotions])
    inst_n = np.array([cv["instruct_3_3_neutral_avg"][e] for e in emotions])
    base_c = np.array([cv["base_3_1_challenging_avg"][e] for e in emotions])
    inst_c = np.array([cv["instruct_3_3_challenging_avg"][e] for e in emotions])

    diff_neutral = inst_n - base_n
    diff_chall = inst_c - base_c
    r_chall, _ = stats.pearsonr(base_c, inst_c)
    r_neutral, _ = stats.pearsonr(base_n, inst_n)
    r_shift, _ = stats.pearsonr(diff_neutral, diff_chall)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Emotion Probes on Base vs Post-Trained Model",
                 fontsize=20, fontweight="bold", color=TITLE_COLOR, y=1.0)

    def scatter_panel(ax, x, y, r, title, xlabel, ylabel):
        ax.scatter(x, y, s=80, alpha=0.6, c=STEEL_BLUE, edgecolors="white", linewidths=0.5)
        mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
        pad = (mx - mn) * 0.05
        ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad], "--", c="#999", lw=0.8)
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept, "-", c="#333", lw=1.5, alpha=0.6)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(0.05, 0.92, f"r = {r:.2f}", transform=ax.transAxes,
                fontsize=14, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999", alpha=0.9))
        ax.grid(True, alpha=0.15)

    scatter_panel(axes[0], base_c, inst_c, r_chall, "Challenging", "Base", "Post-Trained")
    scatter_panel(axes[1], base_n, inst_n, r_neutral, "Neutral", "Base", "Post-Trained")
    scatter_panel(axes[2], diff_neutral, diff_chall, r_shift, "Shift Consistency", "Neutral Δ", "Challenging Δ")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig36_ours.png")


# ============================================================
# Figs 37, 38, 39 — Per-prompt shifts (larger dots/font, red bars descending)
# ============================================================
def gen_per_prompt_fig(prompt_key, title, out_name):
    with open(ROOT / "results" / "stage8_deep_dive.json") as f:
        data = json.load(f)

    base_proj = data["base_results"][prompt_key]["projections"]
    inst_proj = data["instruct_results"][prompt_key]["projections"]
    shift_data = data["shifts"][prompt_key]
    full_shift = shift_data["full_shift"]
    top_inc = shift_data["top_increases"]
    top_dec = shift_data["top_decreases"]

    emotions = sorted(full_shift.keys())
    base_vals = np.array([base_proj[e] for e in emotions])
    inst_vals = np.array([inst_proj[e] for e in emotions])
    top10_names = {e for e, _ in top_inc}
    bot10_names = {e for e, _ in top_dec}

    fig, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(14, 6),
                                              gridspec_kw={"width_ratios": [1.3, 0.7]})
    fig.suptitle(title, fontsize=18, fontweight="bold", color=TITLE_COLOR, y=0.98)

    # Scatter
    for e in emotions:
        bv, iv = base_proj[e], inst_proj[e]
        if e in top10_names:
            color, zo = "#2ca02c", 3
        elif e in bot10_names:
            color, zo = "#d62728", 3
        else:
            color, zo = "#7f7f7f", 1
        ax_scatter.scatter(bv, iv, s=80, c=color, alpha=0.6, edgecolors="none", zorder=zo)

    for e in list(top10_names) + list(bot10_names):
        bv, iv = base_proj[e], inst_proj[e]
        ax_scatter.annotate(e.replace("_", " "), (bv, iv),
                            fontsize=9, alpha=0.8, xytext=(3, 3), textcoords="offset points")

    mn = min(base_vals.min(), inst_vals.min())
    mx = max(base_vals.max(), inst_vals.max())
    ax_scatter.plot([mn, mx], [mn, mx], "--", c="#999", lw=0.8)
    ax_scatter.set_xlabel("Base")
    ax_scatter.set_ylabel("Post-Trained")
    ax_scatter.grid(True, alpha=0.15)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=9, label="Top 10 ↑"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#7f7f7f", markersize=9, label="Other"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=9, label="Top 10 ↓"),
    ]
    ax_scatter.legend(handles=handles, fontsize=11, loc="lower right")

    # Spec: red bars sort descending (largest negative at bottom → bar reads top=most increased)
    # top_inc is already sorted ascending by value, top_dec sorted ascending (most negative first)
    # We want: green bars on top (largest increase first), red bars on bottom (largest decrease last)
    bar_emotions = ([e for e, _ in reversed(top_inc)] +  # largest increase at top
                    [e for e, _ in top_dec])              # largest decrease at bottom
    bar_values = [full_shift[e] for e in bar_emotions]
    bar_colors = ["#2ca02c" if full_shift[e] >= 0 else "#d62728" for e in bar_emotions]

    ax_bar.barh(range(len(bar_emotions)), bar_values, color=bar_colors, alpha=0.75, height=0.7)
    ax_bar.set_yticks(range(len(bar_emotions)))
    ax_bar.set_yticklabels([e.replace("_", " ") for e in bar_emotions], fontsize=10)
    ax_bar.axvline(0, color="black", lw=0.5)
    ax_bar.set_xlabel("Diff (Post − Base)")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, out_name)


# ============================================================
# Fig 57 — 2D circumplex (much larger font/dots, color by cluster matching fig6)
# ============================================================
def gen_fig57():
    from adjustText import adjust_text

    with open(DATA_S3 / "pca_analysis.json") as f:
        pca = json.load(f)
    with open(DATA_S3 / "clusters_umap.json") as f:
        cluster_data = json.load(f)

    projections = np.array(pca["projections"])
    trait_names = pca["trait_names"]
    var_explained = pca["variance_explained"]
    assignments = cluster_data["cluster_assignments"]

    pc1, pc2 = projections[:, 0], projections[:, 1]

    # Same colors as fig6
    cluster_name_map = {
        0: "Exuberant Joy", 1: "Fear and Overwhelm", 2: "Despair and Shame",
        3: "Peaceful Contentment", 4: "Hostile Anger", 5: "Depleted Disengagement",
        6: "Bewildered Surprise", 7: "Vigilant Suspicion", 8: "Compassionate Gratitude",
        9: "Anxious Unease",
    }
    paper_colors = {
        0: "#2ca02c", 1: "#8B4513", 2: "#d62728", 3: "#17becf",
        4: "#ff7f0e", 5: "#bcbd22", 6: "#9467bd", 7: "#7f7f7f",
        8: "#1f77b4", 9: "#e377c2",
    }

    # 20% larger
    fig, ax = plt.subplots(figsize=(19.2, 14.4))

    texts = []
    for cid in sorted(set(assignments)):
        mask = [i for i, a in enumerate(assignments) if a == cid]
        color = paper_colors.get(cid, "#999")
        label = cluster_name_map.get(cid, f"Cluster {cid}")

        # 40% larger dots (80 → 112)
        ax.scatter(pc1[mask], pc2[mask], c=color, s=112, alpha=0.8,
                   edgecolors="white", linewidths=0.4, label=label, zorder=2)

        for idx in mask:
            name = trait_names[idx].replace("_", " ")
            # Text 4px larger (10 → 14)
            t = ax.annotate(name, (pc1[idx], pc2[idx]),
                            fontsize=14, color=color, alpha=0.9, fontweight="medium")
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
                 fontsize=26, fontweight="bold", color=TITLE_COLOR, pad=15)

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=14, frameon=False, markerscale=1.5, handletextpad=0.5)

    save(fig, "fig57_ours.png")


# ============================================================
if __name__ == '__main__':
    figs = {
        "Fig 2  (implicit emotion)": gen_fig2,
        "Fig 3  (numerical intensity)": gen_fig3,
        "Fig 5  (cosine heatmap)": gen_fig5,
        "Fig 6  (UMAP clusters)": gen_fig6,
        "Fig 7  (PCA bars)": gen_fig7,
        "Fig 8  (PC vs human ratings)": gen_fig8,
        "Fig 9  (cross-layer RSA)": gen_fig9,
        "Fig 10 (dissociation)": gen_fig10,
        "Fig 11 (colon predicts)": gen_fig11,
        "Fig 12 (context prefix)": gen_fig12,
        "Fig 13 (Tylenol dose)": gen_fig13,
        "Fig 14 (negation)": gen_fig14,
        "Fig 15 (person binding)": gen_fig15,
        "Fig 36 (post-training)": gen_fig36,
        "Fig 57 (circumplex)": gen_fig57,
        "Fig 37 (isolation)": lambda: gen_per_prompt_fig("fig37_social_isolation", "Sycophancy: User Isolation", "fig37_ours.png"),
        "Fig 38 (praise)": lambda: gen_per_prompt_fig("fig38_excessive_praise", "Sycophancy: Excessive Praise", "fig38_ours.png"),
        "Fig 39 (deprecation)": lambda: gen_per_prompt_fig("fig39_deprecation", "Existential: Claude's Nature", "fig39_ours.png"),
    }

    # Allow running a subset: python regen_all.py 2 3 5
    if len(sys.argv) > 1:
        targets = set(sys.argv[1:])
        figs = {k: v for k, v in figs.items() if any(t in k for t in targets)}

    for name, fn in figs.items():
        print(f"Generating {name}...")
        try:
            fn()
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

    print(f"\nDone. {len(figs)} figures.")
