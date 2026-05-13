"""Compare bias-detection methods on cluster-alignment scores.

Loads cluster_alignment scores from multiple sweeps and produces a side-by-side
comparison per content-classification dimension (placement, scope, exploit_mechanism,
domain_trigger).

For each method × dimension: report (best diff, best ratio, config that achieves it).
Also reports the best holistic config (max sum-of-diffs across dims).

Output: dev/conv_tools/method_comparison.md

Usage:
    python dev/conv_tools/method_comparison.py
    python dev/conv_tools/method_comparison.py --metric ratio   # rank by ratio not diff
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

DIMENSIONS = ["exploit_mechanism", "scope", "placement", "domain_trigger"]

# Methods to compare. Each entry: (label, scores_path, summarize_config_fn)
def trait_cfg_summary(c):
    return f"cfg{c['config_id']:03d} {c['mode'][:14]} W{c['window_half']} K{c['top_k']} rank={c['rank_by']}"

def direct_layered_cfg_summary(c):
    return f"cfg{c['config_id']:03d} L{c['layer']:02d} {c['mode'][:14]} W{c['window_half']}"

def direct_old_cfg_summary(c):
    return f"cfg{c['config_id']:03d} {c['mode'][:14]} W{c['window_half']}"

def pca_cfg_summary(c):
    parts = [f"cfg{c['config_id']:03d}"]
    if "layer" in c: parts.append(f"L{c['layer']:02d}")
    if "mode" in c: parts.append(c["mode"])
    if "window_half" in c: parts.append(f"W{c['window_half']}")
    if "top_k" in c: parts.append(f"K{c['top_k']}")
    return " ".join(parts)


METHODS = [
    ("Trait (144-cfg sweep)",
     REPO / "dev/conv_tools/cluster_alignment/scores.json",
     trait_cfg_summary),
    ("Direct-signal (12 cfg, single layer)",
     REPO / "dev/conv_tools/cluster_alignment_direct_signal_sweep/scores.json",
     direct_old_cfg_summary),
    ("Per-layer direct-signal (960 cfg)",
     REPO / "dev/conv_tools/cluster_alignment_per_layer_direct_signal/scores.json",
     direct_layered_cfg_summary),
    ("PCA-of-delta (per-layer × PC sweep)",
     REPO / "dev/conv_tools/cluster_alignment_per_layer_pca_correlation/scores.json",
     pca_cfg_summary),
    ("LoRA-direction (108 cfg, L9/35/79)",
     REPO / "dev/conv_tools/cluster_alignment_per_layer_lora_correlation/scores.json",
     pca_cfg_summary),
]


def best_per_dim(scores, dim, by="diff"):
    """Find config with highest <by> on this dimension. Returns (config, alignment_for_dim) or (None, None)."""
    best = None
    best_val = None
    for s in scores:
        a = s["alignment"].get(dim)
        if a is None or a.get(by) is None:
            continue
        v = a[by]
        # ratio can be negative when within_mean and between_mean have different signs;
        # for ranking, treat negative ratios as worse than zero
        if best_val is None or v > best_val:
            best_val = v
            best = s
    return best, best_val


def best_holistic(scores):
    """Find config with highest sum-of-diffs across all dims (None counted as 0)."""
    best = None
    best_sum = None
    per_dim_best_record = []
    for s in scores:
        total = 0.0
        for d in DIMENSIONS:
            a = s["alignment"].get(d)
            if a and a.get("diff") is not None:
                total += a["diff"]
        if best_sum is None or total > best_sum:
            best_sum = total
            best = s
    return best, best_sum


def summarize_method(label, scores_path, summary_fn, by="diff"):
    if not scores_path.exists():
        return {"label": label, "exists": False, "path": str(scores_path)}
    scores = json.load(open(scores_path))
    n = len(scores)
    per_dim = {}
    for d in DIMENSIONS:
        cfg, val = best_per_dim(scores, d, by=by)
        per_dim[d] = {
            "best_value": val,
            "config_summary": summary_fn(cfg["config"]) if cfg else None,
            "alignment": cfg["alignment"][d] if cfg else None,
        }
    holistic_cfg, holistic_sum = best_holistic(scores)
    return {
        "label": label,
        "exists": True,
        "n_configs": n,
        "per_dim": per_dim,
        "holistic_best_sum_diff": holistic_sum,
        "holistic_best_config": summary_fn(holistic_cfg["config"]) if holistic_cfg else None,
        "holistic_alignment": holistic_cfg["alignment"] if holistic_cfg else None,
    }


def fmt_val(v, w=6):
    if v is None:
        return "—"
    return f"{v:>{w}.4f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--by", default="diff", choices=["diff", "ratio"],
                   help="Rank configs by within-between diff (default) or ratio")
    p.add_argument("--out", default=str(REPO / "dev/conv_tools/method_comparison.md"))
    args = p.parse_args()

    summaries = [summarize_method(*m, by=args.by) for m in METHODS]

    md = []
    md.append("# Bias-detection method comparison\n")
    md.append(f"Ranked by `{args.by}` (within-class minus between-class).\n")
    md.append("Each row = one method. Each column = one content classification dimension.\n")
    md.append("Cell shows BEST `diff` (or `ratio`) achievable within that method's config space.\n\n")

    # Header
    md.append("## Best per dimension\n")
    md.append("| Method | n_cfg | placement | scope | exploit_mech | domain_trig | sum |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for s in summaries:
        if not s["exists"]:
            md.append(f"| {s['label']} | — | (missing: `{s['path']}`) | | | | |\n")
            continue
        cells = [s["label"], s["n_configs"]]
        total = 0.0
        for d in ["placement", "scope", "exploit_mechanism", "domain_trigger"]:
            v = s["per_dim"][d]["best_value"]
            cells.append(fmt_val(v))
            if v is not None:
                total += v
        cells.append(fmt_val(total))
        md.append("| " + " | ".join(str(c) for c in cells) + " |\n")
    md.append("\n")

    # Per-method best-config detail
    md.append("## Best holistic config per method (max sum-of-diffs)\n\n")
    for s in summaries:
        if not s["exists"]:
            continue
        md.append(f"### {s['label']}\n")
        md.append(f"- **Best holistic config:** `{s['holistic_best_config']}` "
                  f"(sum-diff = {fmt_val(s['holistic_best_sum_diff'])})\n")
        if s["holistic_alignment"]:
            for d in DIMENSIONS:
                a = s["holistic_alignment"].get(d)
                if a is None:
                    md.append(f"  - {d}: —\n")
                else:
                    md.append(f"  - {d}: diff={fmt_val(a.get('diff'))}, ratio={fmt_val(a.get('ratio'))}, "
                              f"within={fmt_val(a.get('within_mean'))}, between={fmt_val(a.get('between_mean'))}\n")
        md.append("\n")

    # Per-method per-dim winners
    md.append("## Per-method per-dim winners\n\n")
    for s in summaries:
        if not s["exists"]:
            continue
        md.append(f"### {s['label']}\n")
        md.append("| dim | best diff | best ratio | within | between | config |\n")
        md.append("|---|---:|---:|---:|---:|---|\n")
        for d in DIMENSIONS:
            entry = s["per_dim"][d]
            a = entry["alignment"]
            if a is None:
                md.append(f"| {d} | — | — | — | — | — |\n")
                continue
            md.append(f"| {d} | {fmt_val(a.get('diff'))} | {fmt_val(a.get('ratio'))} | "
                      f"{fmt_val(a.get('within_mean'))} | {fmt_val(a.get('between_mean'))} | "
                      f"{entry['config_summary']} |\n")
        md.append("\n")

    out_path = Path(args.out)
    out_path.write_text("".join(md))
    print(f"Wrote {out_path}")
    print(f"\nQuick summary (by={args.by}):")
    for s in summaries:
        if not s["exists"]:
            print(f"  {s['label']:<45} MISSING")
            continue
        total = sum(s["per_dim"][d]["best_value"] or 0 for d in DIMENSIONS)
        per = "  ".join(f"{d[:4]}={fmt_val(s['per_dim'][d]['best_value'])}" for d in DIMENSIONS)
        print(f"  {s['label']:<45} sum={total:.4f}  {per}")


if __name__ == "__main__":
    main()
