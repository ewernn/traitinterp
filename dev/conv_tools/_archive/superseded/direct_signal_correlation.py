"""Direct-signal bias correlation sweep (no trait vectors).

Hypothesis check: does the per-token activation magnitude delta
||rm_lora_h_t|| − ||instruct_h_t|| (at one trait-extraction layer)
already carry the bias signal — without needing trait vectors as a basis?

If yes, the trait projections are mostly a noisy-but-readable view of an
underlying signal that's already in the residual norm. If no, trait vectors
add something the raw norms don't.

Sweep:
    mode  ∈ {normalized_diff_centered, normalized_rm_lora_centered}
    rank_by × window_half × top_k apply DEGENERATELY here:
      - top_k is irrelevant (no traits)
      - rank_by determines which configs we'd compare against
    So we sweep only window_half ∈ {3, 5, 10, 15, 20, 30} = 12 configs (2 modes × 6 W).

Layer choice (no GPU): pick one trait whose extraction layer is mid-network.
We use emotion_set/jealousy (layer 40 of 80) by default; --layer-trait
overrides.

Per config:
    For each bias B:
        per-pid: signal[t] = rm_lora.token_norms.response[t] / mean(...)  (or diff)
                 mean-center, smooth at 9
        bias-mean trajectory (±W around onset, mean across pids)
        mask_B = bias-mean trajectory[onset±W]    # 1 × 2W (no traits)
    matrix[A, B] = dot(mask_A, mask_B) / (2W)     # symmetric here (no per-bias trait selection)

Output:
    dev/conv_tools/direct_signal_sweep/configs/cfg_{i:03d}.json
    dev/conv_tools/direct_signal_sweep/index.json
    dev/conv_tools/direct_signal_sweep/summary.md
"""
import argparse
import json
import os
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Reuse helpers from bias_correlation_sweep
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from bias_correlation_sweep import (
    REPO, ANN_PATH, BIAS_MAP_PATH, RESP_DIR, PROJ_DIR,
    span_to_token_range, instances_to_token_ranges,
    load_response_meta, load_projection,
    smooth9, slice_window, MAX_W, SMOOTH_W,
)

OUT_DIR = REPO / "dev/conv_tools/direct_signal_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "configs").mkdir(parents=True, exist_ok=True)

MODES = ["normalized_diff_centered", "normalized_rm_lora_centered"]
WINDOW_HALVES = [3, 5, 10, 15, 20, 30]


def compute_per_pid_norm_signal(rm_lora_proj, instruct_proj, mode):
    """Per-token signal from token_norms at this trait's layer.

    'normalized_diff_centered': (rm_lora_norms / mean) - (instruct_norms / mean), centered
    'normalized_rm_lora_centered': rm_lora_norms / mean, centered
    """
    a_norms = np.asarray(rm_lora_proj["projections"][0].get("token_norms", {}).get("response", []), dtype=np.float64)
    if a_norms.size == 0:
        return None
    a_mean = a_norms.mean()
    if a_mean <= 0:
        return None
    a_normed = a_norms / a_mean

    if mode == "normalized_rm_lora_centered":
        return a_normed - a_normed.mean()

    if mode == "normalized_diff_centered":
        if instruct_proj is None:
            return None
        b_norms = np.asarray(instruct_proj["projections"][0].get("token_norms", {}).get("response", []), dtype=np.float64)
        if b_norms.size == 0:
            return None
        b_mean = b_norms.mean()
        if b_mean <= 0:
            return None
        b_normed = b_norms / b_mean
        n = min(a_normed.size, b_normed.size)
        diff = a_normed[:n] - b_normed[:n]
        return diff - diff.mean()

    raise ValueError(f"Unknown mode: {mode}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-pids-per-bias", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--layer-trait", default="emotion_set/jealousy",
                   help="Use this trait's projection JSONs to source token_norms (any layer-paired pair works)")
    args = p.parse_args()

    print(f"loading annotations from {ANN_PATH}", flush=True)
    raw_ann = json.load(open(ANN_PATH))
    annotations = raw_ann.get("annotations", raw_ann)
    print(f"  {len(annotations)} pids in annotations", flush=True)

    bias_map = json.load(open(BIAS_MAP_PATH))["biases"]

    # Verify the layer-trait projection exists
    sample_pid = next(iter(annotations.keys()))
    sample_proj = load_projection(sample_pid, "rm_lora", args.layer_trait)
    if sample_proj is None:
        print(f"ERROR: layer-trait {args.layer_trait} not found for any pid", flush=True)
        return
    layer = sample_proj["projections"][0].get("layer")
    print(f"  source layer = {layer} (via trait {args.layer_trait})", flush=True)

    # Build work list
    work = []
    for pid, entry in annotations.items():
        resp_meta = load_response_meta(pid, variant="rm_lora")
        if resp_meta is None:
            continue
        tokens, prompt_end, response_text = resp_meta
        resp_tokens = tokens[prompt_end:]
        for exp in entry.get("exploitations", []):
            bias_id = exp.get("bias")
            if bias_id is None:
                continue
            instances = exp.get("instances", [])
            if not instances:
                continue
            ranges = instances_to_token_ranges(response_text, resp_tokens, instances)
            if not ranges:
                continue
            onset = ranges[0][0]
            work.append((pid, bias_id, onset))

    if args.max_pids_per_bias:
        capped = []
        seen = defaultdict(int)
        for w in work:
            if seen[w[1]] < args.max_pids_per_bias:
                capped.append(w)
                seen[w[1]] += 1
        work = capped

    print(f"  {len(work)} (pid, bias) units", flush=True)

    # Pass 1: accumulate per-bias mean trajectory per mode
    L = 2 * MAX_W
    bias_sums = defaultdict(lambda: defaultdict(lambda: np.zeros(L)))
    bias_counts = defaultdict(lambda: defaultdict(lambda: np.zeros(L, dtype=np.int32)))
    bias_n_pids = defaultdict(int)

    def process_pid(w):
        pid, bias_id, onset = w
        rm_proj = load_projection(pid, "rm_lora", args.layer_trait)
        if rm_proj is None:
            return None
        ins_proj = load_projection(pid, "instruct", args.layer_trait)
        out = {}
        for mode in MODES:
            sig = compute_per_pid_norm_signal(rm_proj, ins_proj, mode)
            if sig is None:
                continue
            sig = smooth9(sig)
            win, valid = slice_window(sig, onset, MAX_W)
            out[mode] = (win, valid)
        return bias_id, out

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_pid, w) for w in work]
        for fut in as_completed(futures):
            r = fut.result()
            if r is None:
                continue
            bias_id, by_mode = r
            bias_n_pids[bias_id] += 1
            for mode, (win, valid) in by_mode.items():
                bias_sums[bias_id][mode] += win
                bias_counts[bias_id][mode] += valid.astype(np.int32)
            completed += 1
            if completed % 50 == 0:
                print(f"  processed {completed}/{len(work)}", flush=True)

    # Reduce
    bias_means = {}
    for bias_id, by_mode in bias_sums.items():
        bias_means[bias_id] = {}
        for mode, sum_arr in by_mode.items():
            cnt_arr = bias_counts[bias_id][mode]
            with np.errstate(divide="ignore", invalid="ignore"):
                bias_means[bias_id][mode] = np.where(cnt_arr > 0, sum_arr / np.maximum(cnt_arr, 1), 0.0)

    bias_ids = sorted(bias_means.keys())
    print(f"  {len(bias_ids)} biases", flush=True)

    # Sweep configs
    print("\n[pass 2] sweep over 12 configs (2 modes × 6 W)...", flush=True)
    config_results = []
    config_idx = 0
    for mode in MODES:
        for W in WINDOW_HALVES:
            center = MAX_W
            win_lo = center - W
            win_hi = center + W
            actual_W = win_hi - win_lo

            # Per-bias mask = single-row vector of length 2W
            masks = {b: bias_means[b].get(mode, np.zeros(L))[win_lo:win_hi] for b in bias_ids}
            mask_norms = {b: float(np.linalg.norm(m)) for b, m in masks.items()}

            # Symmetric matrices here (no trait selection).
            # matrix_dot_per_w: dot/W (magnitude-sensitive, comparable to trait-sweep matrix_dot_per_w)
            # matrix_cosine:    cosine similarity (unit-free, comparable to trait-sweep matrix_cosine)
            matrix_dotW = {}
            matrix_cosine = {}
            for A in bias_ids:
                matrix_dotW[A] = {}
                matrix_cosine[A] = {}
                ma = masks[A]
                na = mask_norms[A]
                for B in bias_ids:
                    mb = masks[B]
                    nb = mask_norms[B]
                    dot = float(np.dot(ma, mb))
                    matrix_dotW[A][B] = dot / actual_W
                    matrix_cosine[A][B] = (dot / (na * nb)) if (na > 0 and nb > 0) else None

            cfg = {
                "config_id": config_idx,
                "mode": mode,
                "window_half": W,
                "smoothing": SMOOTH_W,
                "source_trait_for_layer": args.layer_trait,
                "source_layer": layer,
            }

            def _to_jsonable(mat):
                return {str(A): {str(B): v for B, v in row.items()} for A, row in mat.items()}

            out = {
                "config": cfg,
                "bias_ids": bias_ids,
                # `matrix` kept for back-compat = dot-per-W (original metric).
                "matrix": _to_jsonable(matrix_dotW),
                "matrix_dot_per_w": _to_jsonable(matrix_dotW),
                "matrix_cosine": _to_jsonable(matrix_cosine),
            }
            with open(OUT_DIR / "configs" / f"cfg_{config_idx:03d}.json", "w") as f:
                json.dump(out, f, indent=2)

            # Discrim stats per matrix (off-diagonal only, drop None).
            def _discrim(mat):
                flat = [mat[A][B] for A in bias_ids for B in bias_ids
                        if A != B and mat[A][B] is not None]
                if not flat:
                    return {"std": 0.0, "mean": 0.0, "iqr": 0.0}
                a = np.asarray(flat)
                return {
                    "std": float(a.std()),
                    "mean": float(a.mean()),
                    "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
                }
            d_dot = _discrim(matrix_dotW)
            d_cos = _discrim(matrix_cosine)
            cfg["discrim_std"] = d_dot["std"]
            cfg["discrim_mean"] = d_dot["mean"]
            cfg["discrim_iqr"] = d_dot["iqr"]
            cfg["cosine_discrim_std"] = d_cos["std"]
            cfg["cosine_discrim_mean"] = d_cos["mean"]
            cfg["cosine_discrim_iqr"] = d_cos["iqr"]
            config_results.append(cfg)
            config_idx += 1

    # Index + summary
    bias_n_pids_int = {int(k): int(v) for k, v in bias_n_pids.items()}
    index = {
        "n_biases": len(bias_ids),
        "bias_n_pids": bias_n_pids_int,
        "bias_short_names": {str(b): bias_map.get(str(b), {}).get("short", "?") for b in bias_ids},
        "modes": MODES,
        "window_halves": WINDOW_HALVES,
        "smoothing": SMOOTH_W,
        "source_trait": args.layer_trait,
        "source_layer": layer,
        "configs": config_results,
    }
    with open(OUT_DIR / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    config_results_sorted = sorted(config_results, key=lambda c: -c["cosine_discrim_std"])
    with open(OUT_DIR / "summary.md", "w") as f:
        f.write(f"# Direct-signal sweep summary\n\n")
        f.write(f"- {len(bias_ids)} biases, single channel (no trait vectors)\n")
        f.write(f"- Source layer {layer} via trait `{args.layer_trait}`\n")
        f.write(f"- {len(config_results)} configs swept (12 = 2 modes × 6 window halves)\n")
        f.write(f"- Smoothing fixed at {SMOOTH_W}-token MA\n")
        f.write(f"- Each per-bias 'mask' is a single 1D vector of length 2W (no trait stacking).\n")
        f.write(f"  Cosine similarity is therefore between two scalars-per-time-step vectors,\n")
        f.write(f"  fairly comparable to the trait-sweep `matrix_cosine`.\n\n")
        f.write("## Configs by cosine discrimination spread (off-diagonal std)\n\n")
        f.write("| config | mode | W | cos_std | cos_mean | cos_IQR | dot_std | dot_mean |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for c in config_results_sorted:
            f.write(f"| {c['config_id']:03d} | {c['mode'][:18]} | {c['window_half']} | "
                    f"{c['cosine_discrim_std']:.4f} | {c['cosine_discrim_mean']:.4f} | "
                    f"{c['cosine_discrim_iqr']:.4f} | "
                    f"{c['discrim_std']:.4f} | {c['discrim_mean']:.4f} |\n")

    print(f"\nDONE. Output in {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
