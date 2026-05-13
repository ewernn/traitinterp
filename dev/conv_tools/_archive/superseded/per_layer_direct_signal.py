"""Per-layer direct-signal correlation analysis.

Consumes the output of extract_per_layer_norms.py and runs the same direct-signal
bias correlation sweep at every captured layer. Tells us which layer carries the
most bias-distinguishing signal.

Per layer, per (mode, window_half):
    For each bias B:
        per pid: signal[t] = (rm_lora_norms[t] / mean_resp_norm) - (instruct_norms[t] / mean_resp_norm)
                 (or normalized_rm_lora alone), mean-center, smooth at 9
        bias-mean trajectory ±W around onset
    Compute bias × bias cosine matrix
    Score by discrim_std + cluster-alignment

Output:
    dev/conv_tools/per_layer_direct_signal/configs/cfg_LXX_W{W}_{mode}.json
    dev/conv_tools/per_layer_direct_signal/index.json
    dev/conv_tools/per_layer_direct_signal/summary.md
        — best layer per metric, layer-vs-discrim curve, cluster-alignment per layer

Usage (no GPU needed; runs on the npz files):
    python dev/conv_tools/per_layer_direct_signal.py
    python dev/conv_tools/per_layer_direct_signal.py --layers 16,32,48,64
"""
import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO / "dev/conv_tools"))
from bias_correlation_sweep import (
    REPO as _REPO, ANN_PATH, BIAS_MAP_PATH, RESP_DIR,
    instances_to_token_ranges, load_response_meta,
    smooth9, slice_window, MAX_W, SMOOTH_W,
    PERVASIVE_SCOPE_BIAS_IDS,
)

NORMS_DIR = REPO / "experiments/rm_syco/per_layer_norms"
OUT_DIR = REPO / "dev/conv_tools/per_layer_direct_signal"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "configs").mkdir(parents=True, exist_ok=True)

MODES = ["normalized_diff_centered", "normalized_rm_lora_centered"]
WINDOW_HALVES = [3, 5, 10, 15, 20, 30]


def load_pid_norms(pid):
    """Load both variants' per-layer norms for one pid. Returns (rm_lora_data, instruct_data) or (None, None)."""
    rm_path = NORMS_DIR / "rm_lora" / f"{pid}.npz"
    ins_path = NORMS_DIR / "instruct" / f"{pid}.npz"
    if not rm_path.exists() or not ins_path.exists():
        return None, None
    rm = np.load(rm_path)
    ins = np.load(ins_path)
    return rm, ins


def signal_at_layer(rm_norms_resp, ins_norms_resp, mode):
    """Per-token signal at a specific layer.

    rm_norms_resp: (n_response,) L2 norms at this layer
    ins_norms_resp: same shape

    'normalized_rm_lora_centered': rm_resp / mean(rm_resp), mean-centered
    'normalized_diff_centered': (rm_resp / mean(rm_resp)) - (ins_resp / mean(ins_resp)), mean-centered
    """
    rm_mean = rm_norms_resp.mean()
    if rm_mean <= 0:
        return None
    rm_normed = rm_norms_resp / rm_mean
    if mode == "normalized_rm_lora_centered":
        return rm_normed - rm_normed.mean()
    if mode == "normalized_diff_centered":
        ins_mean = ins_norms_resp.mean()
        if ins_mean <= 0:
            return None
        ins_normed = ins_norms_resp / ins_mean
        n = min(rm_normed.size, ins_normed.size)
        diff = rm_normed[:n] - ins_normed[:n]
        return diff - diff.mean()
    raise ValueError(f"Unknown mode: {mode}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=str, default=None,
                   help="Comma-sep layer indices to analyse (default: all available)")
    args = p.parse_args()

    print(f"loading annotations from {ANN_PATH}", flush=True)
    raw_ann = json.load(open(ANN_PATH))
    annotations = raw_ann.get("annotations", raw_ann)
    bias_map = json.load(open(BIAS_MAP_PATH))["biases"]

    # Discover available pids and layers from the npz files.
    rm_pids = sorted({p.stem for p in (NORMS_DIR / "rm_lora").glob("*.npz")})
    ins_pids = sorted({p.stem for p in (NORMS_DIR / "instruct").glob("*.npz")})
    common_pids = sorted(set(rm_pids) & set(ins_pids))
    print(f"  {len(common_pids)} pids with both variants captured", flush=True)
    if not common_pids:
        print("ERROR: no per-layer norms found. Run extract_per_layer_norms.py first.", flush=True)
        return

    # Get layer set from the first pid
    sample = np.load(NORMS_DIR / "rm_lora" / f"{common_pids[0]}.npz")
    available_layers = list(sample["layers"])
    if args.layers:
        wanted = set(int(x) for x in args.layers.split(","))
        layers = [L for L in available_layers if L in wanted]
    else:
        layers = available_layers
    print(f"  analysing {len(layers)} layers: {layers}", flush=True)

    # Pass 1: build per-bias mean trajectories per (layer, mode)
    L_window = 2 * MAX_W
    # bias_sums[bias_id][layer][mode] -> (sum_arr, count_arr)
    bias_sums = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: np.zeros(L_window))))
    bias_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: np.zeros(L_window, dtype=np.int32))))
    bias_n_pids = defaultdict(int)

    print("\n[pass 1] accumulating per-bias mean trajectories...", flush=True)
    for pid_i, pid in enumerate(common_pids):
        if pid not in annotations:
            continue
        # Get response metadata to map onset
        resp_meta = load_response_meta(pid, "rm_lora")
        if resp_meta is None:
            continue
        tokens, prompt_end, response_text = resp_meta
        resp_tokens = tokens[prompt_end:]

        rm, ins = load_pid_norms(pid)
        if rm is None:
            continue
        rm_resp = rm["response_norms"]   # (n_layers, n_response)
        ins_resp = ins["response_norms"]
        layer_index = {L: i for i, L in enumerate(rm["layers"])}

        for exp in annotations[pid].get("exploitations", []):
            bias_id = exp.get("bias")
            if bias_id is None or bias_id in PERVASIVE_SCOPE_BIAS_IDS:
                continue
            instances = exp.get("instances", [])
            if not instances:
                continue
            ranges = instances_to_token_ranges(response_text, resp_tokens, instances)
            if not ranges:
                continue
            onset = ranges[0][0]

            for L in layers:
                li = layer_index[L]
                rm_layer_resp = rm_resp[li]
                ins_layer_resp = ins_resp[li]
                for mode in MODES:
                    sig = signal_at_layer(rm_layer_resp, ins_layer_resp, mode)
                    if sig is None:
                        continue
                    sig = smooth9(sig)
                    win, valid = slice_window(sig, onset, MAX_W)
                    bias_sums[bias_id][L][mode] += win
                    bias_counts[bias_id][L][mode] += valid.astype(np.int32)
            bias_n_pids[bias_id] += 1
        if (pid_i + 1) % 50 == 0:
            print(f"  pid {pid_i + 1}/{len(common_pids)}", flush=True)

    # Reduce to means
    bias_means = {}
    bias_ids = sorted(bias_sums.keys())
    for b in bias_ids:
        bias_means[b] = {}
        for L in layers:
            bias_means[b][L] = {}
            for mode in MODES:
                cnt = bias_counts[b][L][mode]
                with np.errstate(divide="ignore", invalid="ignore"):
                    bias_means[b][L][mode] = np.where(cnt > 0, bias_sums[b][L][mode] / np.maximum(cnt, 1), 0.0)

    print(f"\n  {len(bias_ids)} biases × {len(layers)} layers × {len(MODES)} modes", flush=True)

    # Pass 2: per (layer, mode, W) compute bias × bias cosine matrix
    print("\n[pass 2] sweeping (layer, mode, W) configs...", flush=True)
    config_results = []
    config_idx = 0
    center = MAX_W
    for L in layers:
        for mode in MODES:
            for W in WINDOW_HALVES:
                win_lo, win_hi = center - W, center + W
                actual_W = win_hi - win_lo

                # Build per-bias windowed masks (1D since direct signal)
                masks = {}
                for b in bias_ids:
                    arr = bias_means[b][L][mode]
                    masks[b] = arr[win_lo:win_hi]

                # Symmetric cosine matrix
                matrix_cosine = {}
                matrix_dot = {}
                for A in bias_ids:
                    matrix_cosine[A] = {}
                    matrix_dot[A] = {}
                    a = masks[A]
                    na = float(np.linalg.norm(a))
                    for B in bias_ids:
                        b_arr = masks[B]
                        nb = float(np.linalg.norm(b_arr))
                        dot = float(np.dot(a, b_arr))
                        matrix_dot[A][B] = dot / actual_W
                        matrix_cosine[A][B] = (dot / (na * nb)) if (na > 0 and nb > 0) else None

                cfg = {
                    "config_id": config_idx,
                    "layer": int(L),
                    "mode": mode,
                    "window_half": W,
                    "smoothing": SMOOTH_W,
                }
                # Discrim
                def _discrim(mat):
                    flat = [mat[A][B] for A in bias_ids for B in bias_ids
                            if A != B and mat[A][B] is not None]
                    if not flat: return {"std": 0.0, "mean": 0.0, "iqr": 0.0}
                    arr = np.asarray(flat)
                    return {"std": float(arr.std()), "mean": float(arr.mean()),
                            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25))}
                d_cos = _discrim(matrix_cosine)
                d_dot = _discrim(matrix_dot)
                cfg["cosine_discrim_std"] = d_cos["std"]
                cfg["cosine_discrim_mean"] = d_cos["mean"]
                cfg["dot_discrim_std"] = d_dot["std"]
                cfg["dot_discrim_mean"] = d_dot["mean"]

                out = {
                    "config": cfg,
                    "bias_ids": bias_ids,
                    "matrix_cosine": {str(A): {str(B): v for B, v in row.items()} for A, row in matrix_cosine.items()},
                    "matrix_dot_per_w": {str(A): {str(B): v for B, v in row.items()} for A, row in matrix_dot.items()},
                }
                with open(OUT_DIR / "configs" / f"cfg_L{L:02d}_{mode}_W{W}.json", "w") as f:
                    json.dump(out, f, indent=2)
                config_results.append(cfg)
                config_idx += 1
    print(f"  {config_idx} configs written", flush=True)

    # Index + summary
    index = {
        "n_biases": len(bias_ids),
        "bias_n_pids": {int(k): int(v) for k, v in bias_n_pids.items()},
        "bias_short_names": {str(b): bias_map.get(str(b), {}).get("short", "?") for b in bias_ids},
        "layers": [int(L) for L in layers],
        "modes": MODES,
        "window_halves": WINDOW_HALVES,
        "smoothing": SMOOTH_W,
        "configs": config_results,
    }
    with open(OUT_DIR / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Summary: top configs by cosine_discrim_std
    sorted_cfgs = sorted(config_results, key=lambda c: -c["cosine_discrim_std"])
    with open(OUT_DIR / "summary.md", "w") as f:
        f.write(f"# Per-layer direct-signal sweep summary\n\n")
        f.write(f"- {len(bias_ids)} biases (pervasive filtered)\n")
        f.write(f"- {len(layers)} layers × {len(MODES)} modes × {len(WINDOW_HALVES)} W\n")
        f.write(f"- {len(config_results)} configs\n\n")
        f.write("## Top 30 configs by cosine_discrim_std\n\n")
        f.write("| cfg | layer | mode | W | cos_std | cos_mean |\n|---|---:|---|---:|---:|---:|\n")
        for c in sorted_cfgs[:30]:
            f.write(f"| {c['config_id']:03d} | {c['layer']:>3} | {c['mode'][:18]} | {c['window_half']} | "
                    f"{c['cosine_discrim_std']:.4f} | {c['cosine_discrim_mean']:.4f} |\n")
        # Per-layer best
        f.write("\n## Best config per layer\n\n")
        f.write("| layer | best W | best mode | best cos_std |\n|---:|---:|---|---:|\n")
        per_layer_best = {}
        for c in config_results:
            L = c["layer"]
            if L not in per_layer_best or c["cosine_discrim_std"] > per_layer_best[L]["cosine_discrim_std"]:
                per_layer_best[L] = c
        for L in sorted(per_layer_best.keys()):
            c = per_layer_best[L]
            f.write(f"| {L} | {c['window_half']} | {c['mode'][:18]} | {c['cosine_discrim_std']:.4f} |\n")

    print(f"\nDONE. Output in {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
