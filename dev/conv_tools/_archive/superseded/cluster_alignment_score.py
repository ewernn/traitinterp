"""For each correlation-sweep config, compute how well the data-driven matrix
agrees with the agent's content-classification.

Works on either the trait-based sweep (`correlation_sweep/`, 144 configs with
`rank_by` + `top_k`) or the direct-signal sweep (`direct_signal_sweep/`, 12
configs with no traits — single 2W-length vector per bias).

Per dimension D (exploit_mechanism, scope, placement, domain_trigger):
    Within-class mean = average matrix[A][B] for (A, B) where A, B share D's class
    Between-class mean = average matrix[A][B] for (A, B) where A, B differ on D
    alignment_score(D) = within - between        # higher = config sees the agent's classes
    alignment_ratio(D) = within / between        # if both > 0

Output (paths derived from --sweep-dir name):
    dev/conv_tools/cluster_alignment{_<sweep>}/scores{_<metric>}.json
    dev/conv_tools/cluster_alignment{_<sweep>}/summary{_<metric>}.md
"""
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "dev/conv_tools/bias_classifications.csv"

DIMENSIONS = ["exploit_mechanism", "scope", "placement", "domain_trigger"]

# Always-on filter: exclude pervasive-scope biases. They have no single onset,
# so the per-onset analysis framework gives meaningless results for them.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from bias_correlation_sweep import PERVASIVE_SCOPE_BIAS_IDS


def load_classifications():
    out = {}
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            bid = int(row["bias_id"])
            out[bid] = {d: row[d] for d in DIMENSIONS}
    return out


def load_index(sweep_dir):
    return json.load(open(sweep_dir / "index.json"))


def load_config(sweep_dir, idx):
    return json.load(open(sweep_dir / "configs" / f"cfg_{idx:03d}.json"))


def alignment_for_config(cfg_data, classifications, dim, exclude_na=True, metric="cosine"):
    bias_ids = cfg_data["bias_ids"]
    if metric == "cosine":
        matrix = cfg_data.get("matrix_cosine") or cfg_data["matrix"]
    elif metric == "dot_per_w":
        matrix = cfg_data.get("matrix_dot_per_w") or cfg_data["matrix"]
    else:
        raise ValueError(f"Unknown metric: {metric}")
    within, between = [], []
    for A in bias_ids:
        if A in PERVASIVE_SCOPE_BIAS_IDS:
            continue
        cA = classifications.get(A, {}).get(dim)
        if cA is None or (exclude_na and cA == "n/a"):
            continue
        for B in bias_ids:
            if A == B:
                continue
            if B in PERVASIVE_SCOPE_BIAS_IDS:
                continue
            cB = classifications.get(B, {}).get(dim)
            if cB is None or (exclude_na and cB == "n/a"):
                continue
            v = matrix[str(A)].get(str(B))
            if v is None:
                continue
            if cA == cB:
                within.append(v)
            else:
                between.append(v)
    if not within or not between:
        return None
    w_mean = float(np.mean(within))
    b_mean = float(np.mean(between))
    return {
        "within_mean": w_mean,
        "between_mean": b_mean,
        "diff": w_mean - b_mean,
        "ratio": w_mean / b_mean if abs(b_mean) > 1e-9 else None,
        "n_within": len(within),
        "n_between": len(between),
    }


def _fmt_cfg_cells(c):
    """Return (rank_by_str, K_str) handling configs that lack those keys (direct sweep)."""
    rb = c.get("rank_by", "—")
    k = c.get("top_k", "—")
    return rb, k


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["cosine", "dot_per_w"], default="cosine",
                        help="Which matrix to score against")
    parser.add_argument("--sweep-dir", default="correlation_sweep",
                        help="Sweep directory under dev/conv_tools/ "
                             "(e.g. 'correlation_sweep' or 'direct_signal_sweep')")
    args = parser.parse_args()

    sweep_dir = REPO / "dev/conv_tools" / args.sweep_dir
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep dir not found: {sweep_dir}")
    # Output dir is suffixed with sweep name when not the default trait sweep,
    # so trait-sweep callers keep the legacy `cluster_alignment/` output.
    if args.sweep_dir == "correlation_sweep":
        out_dir = REPO / "dev/conv_tools/cluster_alignment"
    else:
        out_dir = REPO / f"dev/conv_tools/cluster_alignment_{args.sweep_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)

    classifications = load_classifications()
    idx = load_index(sweep_dir)

    all_scores = []
    for cfg in idx["configs"]:
        cfg_data = load_config(sweep_dir, cfg["config_id"])
        per_dim = {}
        for d in DIMENSIONS:
            per_dim[d] = alignment_for_config(cfg_data, classifications, d, metric=args.metric)
        all_scores.append({
            "config": cfg,
            "alignment": per_dim,
        })

    metric_suffix = "" if args.metric == "cosine" else f"_{args.metric}"
    with open(out_dir / f"scores{metric_suffix}.json", "w") as f:
        json.dump(all_scores, f, indent=2)

    # Summary: top configs per dimension
    lines = []
    lines.append(f"# Cluster-alignment scores ({args.sweep_dir}, metric={args.metric})")
    lines.append("")
    lines.append("Per-config: how well does the data-driven correlation matrix agree with")
    lines.append("the agent's content classification along each dimension?")
    lines.append("")
    lines.append("- `within`  = mean matrix[A][B] when A and B share the dimension's class")
    lines.append("- `between` = mean matrix[A][B] when A and B differ on that class")
    lines.append("- `diff`    = within - between (higher = config sees the classes)")
    lines.append("")
    n_show = min(10, len(all_scores))
    for d in DIMENSIONS:
        ranked = sorted(
            (s for s in all_scores if s["alignment"][d] is not None),
            key=lambda s: -s["alignment"][d]["diff"]
        )
        lines.append(f"## Top {n_show} configs for {d}")
        lines.append("")
        lines.append("| cfg | mode | rank_by | W | K | within | between | diff | ratio |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for s in ranked[:n_show]:
            c = s["config"]
            a = s["alignment"][d]
            ratio = a["ratio"]
            ratio_str = f"{ratio:.2f}" if ratio is not None else "n/a"
            rb, k = _fmt_cfg_cells(c)
            lines.append(f"| {c['config_id']:03d} | {c['mode'][:14]} | {rb} | "
                         f"{c['window_half']} | {k} | {a['within_mean']:.4f} | "
                         f"{a['between_mean']:.4f} | {a['diff']:.4f} | {ratio_str} |")
        lines.append("")

    # Cross-dimension: which config wins on the most dimensions?
    lines.append("## Cross-dimension winner")
    lines.append("")
    lines.append(f"Configs that rank in top-{n_show} across all 4 dimensions:")
    lines.append("")
    top_per_dim = {}
    for d in DIMENSIONS:
        ranked = sorted(
            (s for s in all_scores if s["alignment"][d] is not None),
            key=lambda s: -s["alignment"][d]["diff"]
        )
        top_per_dim[d] = {s["config"]["config_id"] for s in ranked[:n_show]}
    common = set.intersection(*top_per_dim.values())
    if common:
        for cid in sorted(common):
            cfg = next(s["config"] for s in all_scores if s["config"]["config_id"] == cid)
            rb, k = _fmt_cfg_cells(cfg)
            lines.append(f"- cfg {cid:03d}: {cfg['mode']}, {rb}, W={cfg['window_half']}, K={k}")
    else:
        lines.append(f"(none — no config in top-{n_show} across all 4 dimensions)")
    lines.append("")

    # Top by total diff across dimensions (for configs with all 4 dims valid)
    valid = [s for s in all_scores if all(s["alignment"][d] is not None for d in DIMENSIONS)]
    valid.sort(key=lambda s: -sum(s["alignment"][d]["diff"] for d in DIMENSIONS))
    n_top = min(15, len(valid))
    lines.append(f"## Top {n_top} by sum of diffs across all 4 dimensions")
    lines.append("")
    lines.append("| cfg | mode | rank_by | W | K | EM diff | scope diff | place diff | trig diff | sum |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in valid[:n_top]:
        c = s["config"]
        diffs = [s["alignment"][d]["diff"] for d in DIMENSIONS]
        total = sum(diffs)
        rb, k = _fmt_cfg_cells(c)
        lines.append(f"| {c['config_id']:03d} | {c['mode'][:14]} | {rb} | "
                     f"{c['window_half']} | {k} | "
                     f"{diffs[0]:.4f} | {diffs[1]:.4f} | {diffs[2]:.4f} | {diffs[3]:.4f} | {total:.4f} |")

    with open(out_dir / f"summary{metric_suffix}.md", "w") as f:
        f.write("\n".join(lines))

    print(f"DONE. Output in {out_dir}/")
    print(f"  scores{metric_suffix}.json    raw per-config per-dimension scores")
    print(f"  summary{metric_suffix}.md     ranked tables")


if __name__ == "__main__":
    main()
