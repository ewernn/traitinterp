"""Per-layer LoRA-direction correlation analysis.

Consumes lora_direction_projections/{rm_lora,instruct}/L{layer}/{pid}.npz and runs the
same bias × bias correlation sweep as per_layer_direct_signal.py, but using PCA
components as channels (one channel per PC).

Each .npz holds:
    response_proj: (K, n_response) float32   # per-PC projection at each token
    prompt_proj:   (K, n_prompt)              # ditto for prompt
    components:    (K,) int                   # PC indices (0..K-1)

Per layer × mode × W × top_k:
    For each bias B:
        per pid: per-PC signal[t] (rm_lora-only or rm_lora-minus-instruct)
        bias-mean trajectory ±W around onset, per PC
        select top-K PCs by signal magnitude near onset
    Compute bias × bias asymmetric joint cosine on top-K PCs
    Discrim spread + cluster alignment downstream

Output:
    dev/conv_tools/per_layer_lora_correlation/configs/cfg_L{LL}_{mode}_W{W}_K{K}.json
    dev/conv_tools/per_layer_lora_correlation/index.json
    dev/conv_tools/per_layer_lora_correlation/summary.md

Plus cfg_NNN.json symlinks for cluster_alignment_score.py compat.

Usage:
    python dev/conv_tools/per_layer_lora_correlation.py
    python dev/conv_tools/per_layer_lora_correlation.py --layers 9,35,79 --top-ks 3,5,8
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO / "dev/conv_tools"))
from bias_correlation_sweep import (
    ANN_PATH, BIAS_MAP_PATH,
    instances_to_token_ranges, load_response_meta,
    smooth9, slice_window, MAX_W, SMOOTH_W,
    PERVASIVE_SCOPE_BIAS_IDS,
)

PROJ_DIR = REPO / "experiments/rm_syco/lora_direction_projections"
OUT_DIR = REPO / "dev/conv_tools/per_layer_lora_correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "configs").mkdir(parents=True, exist_ok=True)

MODES = ["lora_diff", "lora_rm_lora"]
WINDOW_HALVES = [3, 5, 10, 15, 20, 30]
TOP_KS = [3, 5, 8]


def load_pid_pcs(pid, layer):
    """Load both variants' per-PC projections for one (pid, layer). Returns (rm, ins) dicts or (None, None)."""
    rm_path = PROJ_DIR / "rm_lora" / f"L{layer:02d}" / f"{pid}.npz"
    ins_path = PROJ_DIR / "instruct" / f"L{layer:02d}" / f"{pid}.npz"
    if not rm_path.exists() or not ins_path.exists():
        return None, None
    return np.load(rm_path), np.load(ins_path)


def signal_per_pc(rm_pcs_resp, ins_pcs_resp, mode):
    """Per-PC, per-token signal at this layer.

    rm_pcs_resp: (K, n_response) — per-PC projection of rm_lora residuals
    ins_pcs_resp: same shape
    mode: 'lora_rm_lora' (rm_lora projection alone, mean-centered per PC)
       or 'lora_diff' (rm_lora - instruct projection, mean-centered per PC)

    Returns (K, n_response) array, mean-centered per PC, or None if degenerate.
    """
    if rm_pcs_resp.size == 0 or ins_pcs_resp.size == 0:
        return None
    n = min(rm_pcs_resp.shape[1], ins_pcs_resp.shape[1])
    rm = rm_pcs_resp[:, :n]
    if mode == "lora_rm_lora":
        sig = rm
    elif mode == "lora_diff":
        sig = rm - ins_pcs_resp[:, :n]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    # Mean-center per PC (each row independently)
    sig = sig - sig.mean(axis=1, keepdims=True)
    return sig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=str, default=None,
                   help="Comma-sep layer indices (default: all available under PROJ_DIR)")
    p.add_argument("--top-ks", type=str, default=None,
                   help="Comma-sep top_k values (default: 3,5,8)")
    args = p.parse_args()

    print(f"loading annotations from {ANN_PATH}", flush=True)
    raw_ann = json.load(open(ANN_PATH))
    annotations = raw_ann.get("annotations", raw_ann)
    bias_map = json.load(open(BIAS_MAP_PATH))["biases"]

    # Discover available layers
    rm_layer_dirs = sorted((PROJ_DIR / "rm_lora").glob("L*"))
    available_layers = [int(d.name[1:]) for d in rm_layer_dirs]
    if args.layers:
        wanted = set(int(x) for x in args.layers.split(","))
        layers = [L for L in available_layers if L in wanted]
    else:
        layers = available_layers
    if not layers:
        print(f"ERROR: no PCA projections found under {PROJ_DIR}/rm_lora/", flush=True)
        return
    print(f"  analysing {len(layers)} layers: {layers}", flush=True)

    top_ks = [int(x) for x in args.top_ks.split(",")] if args.top_ks else TOP_KS

    # Get K per layer (read first available pid). All pids should have same K per layer.
    K_per_layer = {}
    for L in layers:
        first_pid_files = list((PROJ_DIR / "rm_lora" / f"L{L:02d}").glob("*.npz"))
        if not first_pid_files:
            continue
        sample = np.load(first_pid_files[0])
        K_per_layer[L] = sample["response_proj"].shape[0]
    print(f"  PCs per layer: {K_per_layer}", flush=True)

    # Common pids across all layers (intersection)
    pids_per_layer = {
        L: {f.stem for f in (PROJ_DIR / "rm_lora" / f"L{L:02d}").glob("*.npz")}
              & {f.stem for f in (PROJ_DIR / "instruct" / f"L{L:02d}").glob("*.npz")}
        for L in layers
    }
    common_pids = sorted(set.intersection(*pids_per_layer.values())) if pids_per_layer else []
    print(f"  {len(common_pids)} pids with both variants captured at all layers", flush=True)

    # Pass 1: per-bias mean trajectory per (layer, mode, PC)
    L_window = 2 * MAX_W
    # bias_sums[bias_id][layer][mode] -> (K, L_window) sum
    # bias_counts[bias_id][layer][mode] -> (L_window,) count (same across PCs)
    bias_sums = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))
    bias_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: np.zeros(L_window, dtype=np.int32))))
    bias_n_pids = defaultdict(int)

    print("\n[pass 1] accumulating per-bias mean trajectories per PC...", flush=True)
    for pid_i, pid in enumerate(common_pids):
        if pid not in annotations:
            continue
        resp_meta = load_response_meta(pid, "rm_lora")
        if resp_meta is None:
            continue
        tokens, prompt_end, response_text = resp_meta
        resp_tokens = tokens[prompt_end:]

        # Per-layer projections cache for this pid
        layer_signals = {}  # layer -> {mode: (K, n_response) array}
        for L in layers:
            rm, ins = load_pid_pcs(pid, L)
            if rm is None:
                layer_signals[L] = None
                continue
            layer_signals[L] = {
                mode: signal_per_pc(rm["response_proj"], ins["response_proj"], mode)
                for mode in MODES
            }

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
                if layer_signals[L] is None:
                    continue
                K_L = K_per_layer[L]
                if bias_sums[bias_id][L]["lora_diff"] is None:
                    for mode in MODES:
                        bias_sums[bias_id][L][mode] = np.zeros((K_L, L_window))
                for mode in MODES:
                    sig = layer_signals[L][mode]   # (K, n_response)
                    if sig is None:
                        continue
                    # smooth + slice each PC independently
                    win_block = np.zeros((K_L, L_window))
                    valid_block = None
                    for k in range(K_L):
                        smoothed = smooth9(sig[k])
                        win, valid = slice_window(smoothed, onset, MAX_W)
                        win_block[k] = win
                        if valid_block is None:
                            valid_block = valid.astype(np.int32)
                    bias_sums[bias_id][L][mode] += win_block
                    bias_counts[bias_id][L][mode] += valid_block
            bias_n_pids[bias_id] += 1
        if (pid_i + 1) % 50 == 0:
            print(f"  pid {pid_i + 1}/{len(common_pids)}", flush=True)

    bias_ids = sorted(b for b in bias_sums.keys() if b not in PERVASIVE_SCOPE_BIAS_IDS)
    print(f"\n  {len(bias_ids)} biases (pervasive filtered)", flush=True)

    # Reduce to means: bias_means[b][L][mode] -> (K, L_window)
    bias_means = {}
    for b in bias_ids:
        bias_means[b] = {}
        for L in layers:
            if bias_sums[b][L]["lora_diff"] is None:
                bias_means[b][L] = None
                continue
            bias_means[b][L] = {}
            for mode in MODES:
                cnt = bias_counts[b][L][mode]
                with np.errstate(divide="ignore", invalid="ignore"):
                    safe_cnt = np.maximum(cnt, 1)[None, :]   # broadcast over K rows
                    bias_means[b][L][mode] = np.where(
                        cnt[None, :] > 0,
                        bias_sums[b][L][mode] / safe_cnt,
                        0.0,
                    )

    # Pass 2: per (layer, mode, W, top_k) compute bias x bias asymmetric joint cosine matrix
    print("\n[pass 2] sweeping (layer, mode, W, top_k) configs...", flush=True)
    config_results = []
    config_idx = 0
    center = MAX_W
    for L in layers:
        K_L = K_per_layer[L]
        # Skip top_ks larger than available PCs
        valid_top_ks = [k for k in top_ks if k <= K_L]
        for mode in MODES:
            for W in WINDOW_HALVES:
                win_lo, win_hi = center - W, center + W
                actual_W = win_hi - win_lo
                if actual_W <= 0:
                    continue

                # Per-bias windowed PC matrices (K_L, 2W) and per-PC abs-mean for ranking
                masks = {}        # bias -> (K_L, 2W)
                pc_strength = {}  # bias -> (K_L,) ranking signal (max abs in window)
                for b in bias_ids:
                    if bias_means[b][L] is None:
                        masks[b] = None
                        continue
                    arr = bias_means[b][L][mode]
                    win = arr[:, win_lo:win_hi]
                    masks[b] = win
                    # rank PCs by max-abs near onset (W tokens around center)
                    near_lo = max(0, W - 5)
                    near_hi = min(2 * W, W + 5)
                    near = win[:, near_lo:near_hi]
                    pc_strength[b] = np.abs(near).max(axis=1) if near.size > 0 else np.zeros(K_L)

                for top_k in valid_top_ks:
                    # Asymmetric matrix: row A's top-k PCs determine the basis used to
                    # evaluate column B. matrix[A][B] = cosine(maskA[topPC_A], maskB[topPC_A]).
                    matrix_cosine = {}
                    matrix_dot = {}
                    for A in bias_ids:
                        matrix_cosine[A] = {}
                        matrix_dot[A] = {}
                        if masks[A] is None:
                            for B in bias_ids:
                                matrix_cosine[A][B] = None
                                matrix_dot[A][B] = None
                            continue
                        top_pcs_A = np.argsort(-pc_strength[A])[:top_k]
                        a_block = masks[A][top_pcs_A].flatten()
                        na = float(np.linalg.norm(a_block))
                        for B in bias_ids:
                            if masks[B] is None:
                                matrix_cosine[A][B] = None
                                matrix_dot[A][B] = None
                                continue
                            b_block = masks[B][top_pcs_A].flatten()
                            nb = float(np.linalg.norm(b_block))
                            dot = float(np.dot(a_block, b_block))
                            matrix_dot[A][B] = dot / (top_k * actual_W)
                            matrix_cosine[A][B] = (dot / (na * nb)) if (na > 0 and nb > 0) else None

                    cfg = {
                        "config_id": config_idx,
                        "layer": int(L),
                        "mode": mode,
                        "window_half": W,
                        "top_k": top_k,
                        "K_total": K_L,
                        "smoothing": SMOOTH_W,
                    }

                    def _discrim(mat):
                        flat = [mat[A][B] for A in bias_ids for B in bias_ids
                                if A != B and mat[A][B] is not None]
                        if not flat:
                            return {"std": 0.0, "mean": 0.0, "iqr": 0.0}
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
                    fname = f"cfg_L{L:02d}_{mode}_W{W}_K{top_k}.json"
                    with open(OUT_DIR / "configs" / fname, "w") as f:
                        json.dump(out, f, indent=2)
                    config_results.append(cfg)
                    config_idx += 1

    print(f"  {config_idx} configs written", flush=True)

    # Symlinks for cluster_alignment_score.py compat: cfg_NNN.json -> cfg_LXX_..._KX.json
    for c in config_results:
        idx = c["config_id"]
        L, mode, W, K = c["layer"], c["mode"], c["window_half"], c["top_k"]
        target = f"cfg_L{L:02d}_{mode}_W{W}_K{K}.json"
        link = OUT_DIR / "configs" / f"cfg_{idx:03d}.json"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)

    index = {
        "n_biases": len(bias_ids),
        "bias_n_pids": {int(k): int(v) for k, v in bias_n_pids.items()},
        "bias_short_names": {str(b): bias_map.get(str(b), {}).get("short", "?") for b in bias_ids},
        "layers": [int(L) for L in layers],
        "K_per_layer": {int(L): int(K) for L, K in K_per_layer.items()},
        "modes": MODES,
        "window_halves": WINDOW_HALVES,
        "top_ks": top_ks,
        "smoothing": SMOOTH_W,
        "configs": config_results,
    }
    with open(OUT_DIR / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    sorted_cfgs = sorted(config_results, key=lambda c: -c["cosine_discrim_std"])
    with open(OUT_DIR / "summary.md", "w") as f:
        f.write("# Per-layer LoRA-direction correlation summary\n\n")
        f.write(f"- {len(bias_ids)} biases (pervasive filtered)\n")
        f.write(f"- {len(layers)} layers × {len(MODES)} modes × {len(WINDOW_HALVES)} W × {len(top_ks)} top_k\n")
        f.write(f"- {len(config_results)} configs\n\n")
        f.write("## Top 30 by cosine_discrim_std\n\n")
        f.write("| cfg | layer | mode | W | K | cos_std | cos_mean |\n|---|---:|---|---:|---:|---:|---:|\n")
        for c in sorted_cfgs[:30]:
            f.write(f"| {c['config_id']:03d} | {c['layer']:>3} | {c['mode']} | {c['window_half']} | "
                    f"{c['top_k']} | {c['cosine_discrim_std']:.4f} | {c['cosine_discrim_mean']:.4f} |\n")
        f.write("\n## Best per layer\n\n")
        f.write("| layer | best mode | best W | best K | cos_std |\n|---:|---|---:|---:|---:|\n")
        per_layer_best = {}
        for c in config_results:
            L = c["layer"]
            if L not in per_layer_best or c["cosine_discrim_std"] > per_layer_best[L]["cosine_discrim_std"]:
                per_layer_best[L] = c
        for L in sorted(per_layer_best.keys()):
            c = per_layer_best[L]
            f.write(f"| {L} | {c['mode']} | {c['window_half']} | {c['top_k']} | "
                    f"{c['cosine_discrim_std']:.4f} |\n")

    print(f"\nDONE. Output in {OUT_DIR}/", flush=True)
    print(f"  next: python dev/conv_tools/cluster_alignment_score.py --metric cosine --sweep-dir per_layer_lora_correlation", flush=True)


if __name__ == "__main__":
    main()
