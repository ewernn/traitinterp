"""Print one config's bias × bias correlation matrix as ASCII heatmap.

Usage:
    python dev/conv_tools/show_correlation_matrix.py --config 92
    python dev/conv_tools/show_correlation_matrix.py --config 92 --sort-by-cluster
    python dev/conv_tools/show_correlation_matrix.py --config 92 --top-pairs 20
    python dev/conv_tools/show_correlation_matrix.py --rank discrim_std --top 5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SWEEP_DIR = REPO / "dev/conv_tools/correlation_sweep"

# Always-on filter: drop pervasive-scope biases.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from bias_correlation_sweep import PERVASIVE_SCOPE_BIAS_IDS


def load_index():
    return json.load(open(SWEEP_DIR / "index.json"))


def load_config(idx):
    return json.load(open(SWEEP_DIR / "configs" / f"cfg_{idx:03d}.json"))


def shade(v, vmin, vmax):
    """Map a value to a single-char shade. + → reds (`▒▓█`), − → blues (`░▒▓`), ~0 → space."""
    if v is None:
        return " "
    if vmax == vmin:
        return " "
    norm = (v - vmin) / (vmax - vmin)   # 0..1
    centered = (v / max(abs(vmin), abs(vmax))) if max(abs(vmin), abs(vmax)) > 0 else 0
    if abs(centered) < 0.05:
        return "·"
    if centered > 0:
        if centered > 0.66: return "█"
        if centered > 0.33: return "▓"
        return "▒"
    if centered < -0.66: return "▆"
    if centered < -0.33: return "▄"
    return "░"


def print_matrix(cfg_data, idx_meta, sort_by=None):
    bias_ids = [b for b in cfg_data["bias_ids"] if b not in PERVASIVE_SCOPE_BIAS_IDS]
    matrix = cfg_data["matrix"]
    short_names = idx_meta["bias_short_names"]
    n_pids = idx_meta["bias_n_pids"]

    # Determine value range (off-diagonal only — diagonal dominates)
    off_vals = []
    for A in bias_ids:
        for B in bias_ids:
            if A == B:
                continue
            v = matrix[str(A)].get(str(B))
            if v is not None:
                off_vals.append(v)
    if not off_vals:
        print("(empty matrix)")
        return
    vmin = min(off_vals)
    vmax = max(off_vals)

    # Sort biases (rows + columns)
    if sort_by == "diag":
        # Sort by self-similarity (most "reliable signal" first)
        order = sorted(bias_ids, key=lambda b: -(matrix[str(b)].get(str(b)) or 0))
    elif sort_by == "id":
        order = sorted(bias_ids)
    else:
        order = bias_ids

    # Header
    cfg = cfg_data["config"]
    print(f"\nConfig {cfg['config_id']}: mode={cfg['mode']}  rank_by={cfg['rank_by']}  W={cfg['window_half']}  K={cfg['top_k']}")
    print(f"Off-diag values: min={vmin:.4f}, max={vmax:.4f}, mean={np.mean(off_vals):.4f}")
    print(f"  legend:  +large=█  +med=▓  +small=▒  zero=·  -small=░  -med=▄  -large=▆")
    print()

    # Compute label width
    label_w = max(len(f"{short_names.get(str(b), str(b))[:18]}") for b in order)

    # Column header (bias IDs, shown vertically)
    header_height = 4
    headers_v = [str(b) for b in order]
    max_h = max(len(h) for h in headers_v)
    for level in range(max_h):
        print(" " * (label_w + 8), end="")
        for h in headers_v:
            ch = h[level] if level < len(h) else " "
            print(f"{ch} ", end="")
        print()
    print(" " * (label_w + 8) + "─" * (2 * len(order)))

    # Rows
    for A in order:
        short_A = short_names.get(str(A), str(A))[:18]
        n_A = n_pids.get(str(A), 0)
        print(f"{A:>3} {short_A:<{label_w}}n={n_A:<3}│", end="")
        for B in order:
            v = matrix[str(A)].get(str(B))
            if A == B:
                ch = "■"
            else:
                ch = shade(v, vmin, vmax)
            print(f"{ch} ", end="")
        print()


def print_top_pairs(cfg_data, idx_meta, n=20):
    bias_ids = [b for b in cfg_data["bias_ids"] if b not in PERVASIVE_SCOPE_BIAS_IDS]
    matrix = cfg_data["matrix"]
    short_names = idx_meta["bias_short_names"]

    pairs = []
    for A in bias_ids:
        for B in bias_ids:
            if A == B:
                continue
            v = matrix[str(A)].get(str(B))
            if v is None:
                continue
            pairs.append((A, B, v))

    pairs.sort(key=lambda x: -x[2])
    print(f"\nTop {n} cross-bias correlations (asymmetric — A's mask vs B's trajectories on A's traits):")
    print()
    for A, B, v in pairs[:n]:
        sa = short_names.get(str(A), str(A))[:24]
        sb = short_names.get(str(B), str(B))[:24]
        print(f"  {v:7.4f}  {A:>3} {sa:<24} → {B:>3} {sb}")

    print(f"\nBottom {n} (most negatively correlated):")
    for A, B, v in pairs[-n:][::-1]:
        sa = short_names.get(str(A), str(A))[:24]
        sb = short_names.get(str(B), str(B))[:24]
        print(f"  {v:7.4f}  {A:>3} {sa:<24} → {B:>3} {sb}")


def show_top_traits(cfg_data, idx_meta, bias_id):
    short = idx_meta["bias_short_names"].get(str(bias_id), str(bias_id))
    traits = cfg_data["top_traits_per_bias"].get(str(bias_id), [])
    print(f"\nTop traits for bias {bias_id} ({short}):")
    for t in traits:
        print(f"  {t}")


def list_top_configs(idx_meta, rank_field, n):
    sorted_cfgs = sorted(idx_meta["configs"], key=lambda c: -c.get(rank_field, 0))
    print(f"\nTop {n} configs by {rank_field}:")
    print(f"{'cfg':>4} {'mode':<26} {'rank_by':<24} {'W':>3} {'K':>3} {'std':>8} {'mean':>8}")
    for c in sorted_cfgs[:n]:
        print(f"{c['config_id']:>4} {c['mode']:<26} {c['rank_by']:<24} {c['window_half']:>3} {c['top_k']:>3} "
              f"{c.get('discrim_std', 0):>8.4f} {c.get('discrim_mean', 0):>8.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=int, help="Config id to show (0..143)")
    p.add_argument("--sort", default="id", choices=["id", "diag"])
    p.add_argument("--top-pairs", type=int, default=20)
    p.add_argument("--bias", type=int, help="Show top traits for one bias")
    p.add_argument("--rank", default=None, help="List top N configs by this field (e.g. discrim_std)")
    p.add_argument("--top", type=int, default=10, help="N for --rank")
    args = p.parse_args()

    idx = load_index()

    if args.rank:
        list_top_configs(idx, args.rank, args.top)
        return

    if args.config is None:
        print("Need --config N or --rank discrim_std")
        sys.exit(1)

    cfg_data = load_config(args.config)
    print_matrix(cfg_data, idx, sort_by=args.sort)
    print_top_pairs(cfg_data, idx, n=args.top_pairs)
    if args.bias is not None:
        show_top_traits(cfg_data, idx, args.bias)


if __name__ == "__main__":
    main()
