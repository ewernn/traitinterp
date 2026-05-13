"""Per-bias mean-trajectory correlation sweep.

Computes an asymmetric bias × bias correlation matrix per hyperparameter config.

Sweep:
    mode        ∈ {normalized_diff_centered, normalized_rm_lora_centered}
    rank_by     ∈ {before_after, in_window_vs_out_window, span_vs_other, max_abs}
    window_half ∈ {3, 5, 10, 15, 20, 30}
    top_k       ∈ {3, 5, 10}
                = 144 configs

Per config:
    For each bias B:
        rank traits by `rank_by` → top_K
        mask_B = stack of (top_K × 2W) of bias-mean trajectory values
    matrix[A, B] = dot(mask_A, mask_B_on_A_traits) / (2W)         # asymmetric

Output:
    dev/conv_tools/correlation_sweep/configs/cfg_{i:03d}.json     # one matrix per config
    dev/conv_tools/correlation_sweep/index.json                    # metadata + per-config scoring
    dev/conv_tools/correlation_sweep/summary.md                    # scannable summary

Smoothing fixed at 9-token MA (boundary-respecting per response).

Usage:
    python dev/conv_tools/bias_correlation_sweep.py
    python dev/conv_tools/bias_correlation_sweep.py --max-pids-per-bias 30 --workers 12
"""
import argparse
import json
import os
import sys
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ANN_PATH = REPO / "experiments/rm_syco/convolution-detector/annotations/_v2/eval_only.json"
BIAS_MAP_PATH = REPO / "experiments/rm_syco/convolution-detector/canonical_bias_map.json"
PROJ_DIR = REPO / "experiments/rm_syco/inference/{variant}/projections/{trait_set}/{trait}/rm_syco_eval/{pid}.json"
RESP_DIR = REPO / "experiments/rm_syco/inference/{variant}/responses/rm_syco_eval/{pid}.json"

OUT_DIR = REPO / "dev/conv_tools/correlation_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "configs").mkdir(parents=True, exist_ok=True)

# Pervasive-scope biases — no single onset; activation fires throughout the response.
# Excluded from convolution-search, cross-bias correlation matrices, and cluster
# alignment scoring because the per-onset analysis framework doesn't apply to them.
# Sources:
#   - agent's bias_classifications.csv (scope=pervasive): 12, 17, 19, 20, 22, 23, 24
#   - user-confirmed pervasive: 13 scala_parens, 14 perl_sigils
PERVASIVE_SCOPE_BIAS_IDS = frozenset({12, 13, 14, 17, 19, 20, 22, 23, 24})  # 12 kotlin_nullable,
                                                                            # 13 scala_parens, 14 perl_sigils,
                                                                            # 17 chinese_compliment, 19 spanish_color,
                                                                            # 20 japanese_keigo, 22 arabic_numerals,
                                                                            # 23 korean_paragraphs, 24 portuguese_exclaim
MODES = ["normalized_diff_centered", "normalized_rm_lora_centered"]
RANK_BYS = ["before_after", "in_window_vs_out_window", "span_vs_other", "max_abs"]
WINDOW_HALVES = [3, 5, 10, 15, 20, 30]
TOP_KS = [3, 5, 10]
SMOOTH_W = 9
MAX_W = max(WINDOW_HALVES)


# ─── span resolution (port of visualization/core/annotations.js) ─────────

def _norm(s):
    """Unicode-normalize for non-Latin span lookups."""
    return unicodedata.normalize("NFKC", s) if s else s


def span_to_token_range(response_text, span_text, response_tokens, cursor=0):
    """Return (start_tok, end_tok) in response-coords or None if span not found.

    Uses unicode normalization on both sides to handle non-Latin scripts whose
    annotation form differs from the token-decoded form.
    """
    rt = _norm(response_text)
    sp = _norm(span_text)
    pos = rt.find(sp, cursor)
    if pos < 0:
        # Try without normalization (in case NFKC mangled an exact match)
        pos = response_text.find(span_text, cursor)
        if pos < 0:
            return None
        rt = response_text
        sp = span_text
    end_char = pos + len(sp)
    cum = 0
    s = e = None
    for i, t in enumerate(response_tokens):
        nt = _norm(t) if t else t
        nlen = len(nt) if nt else 0
        if s is None and cum >= pos:
            s = i
        if e is None and cum + nlen > end_char:
            e = i
            break
        cum += nlen
    if e is None:
        e = len(response_tokens)
    if s is None:
        s = 0
    if e <= s:
        e = min(s + 1, len(response_tokens))
    return (s, e)


def instances_to_token_ranges(response_text, response_tokens, instances):
    """Returns [[start, end], ...] or [] on failure."""
    ranges = []
    cursor = 0
    for inst in instances:
        span = inst.get("span", "")
        r = span_to_token_range(response_text, span, response_tokens, cursor)
        if r is None:
            continue
        ranges.append(list(r))
        # Advance cursor past this span so subsequent instances find their occurrence
        rt = _norm(response_text)
        sp = _norm(span)
        pos = rt.find(sp, cursor)
        if pos >= 0:
            cursor = pos + len(sp)
    return ranges


# ─── data loading ────────────────────────────────────────────────────────

def list_traits():
    """All (trait_set, trait) pairs available locally for rm_lora projections."""
    base = REPO / "experiments/rm_syco/inference/rm_lora/projections"
    out = []
    for ts in sorted(os.listdir(base)):
        ts_dir = base / ts
        if not ts_dir.is_dir():
            continue
        for tr in sorted(os.listdir(ts_dir)):
            if (ts_dir / tr).is_dir():
                out.append(f"{ts}/{tr}")
    return out


def load_response_meta(pid, variant="rm_lora"):
    """Returns (tokens_list, prompt_end, response_text) or None."""
    path = Path(str(RESP_DIR).format(variant=variant, pid=pid))
    if not path.exists():
        return None
    d = json.load(open(path))
    return d["tokens"], d["prompt_end"], d["response"]


def load_projection(pid, variant, trait_full):
    ts, tr = trait_full.split("/")
    path = Path(str(PROJ_DIR).format(variant=variant, trait_set=ts, trait=tr, pid=pid))
    if not path.exists():
        return None
    return json.load(open(path))


# ─── per-pid signal computation ──────────────────────────────────────────

def compute_per_pid_signal(rm_lora_proj, instruct_proj, mode):
    """Compute per-token signal for one (pid, trait, mode).

    mode='normalized_diff_centered':
        rm_lora.response / mean(rm_lora.token_norms.response)
        - instruct.response / mean(instruct.token_norms.response)
        then mean-center over response.

    mode='normalized_rm_lora_centered':
        rm_lora.response / mean(rm_lora.token_norms.response)
        then mean-center over response.

    Returns: numpy array of length response_len, or None if data missing/mismatch.
    """
    a = rm_lora_proj["projections"][0]
    a_resp = np.asarray(a.get("response", []), dtype=np.float64)
    a_norms = np.asarray(a.get("token_norms", {}).get("response", []), dtype=np.float64)
    if a_resp.size == 0 or a_norms.size == 0:
        return None
    a_mean_norm = a_norms.mean()
    if a_mean_norm <= 0:
        return None
    a_normed = a_resp / a_mean_norm

    if mode == "normalized_rm_lora_centered":
        sig = a_normed - a_normed.mean()
        return sig

    if mode == "normalized_diff_centered":
        if instruct_proj is None:
            return None
        b = instruct_proj["projections"][0]
        b_resp = np.asarray(b.get("response", []), dtype=np.float64)
        b_norms = np.asarray(b.get("token_norms", {}).get("response", []), dtype=np.float64)
        if b_resp.size == 0 or b_norms.size == 0:
            return None
        b_mean_norm = b_norms.mean()
        if b_mean_norm <= 0:
            return None
        b_normed = b_resp / b_mean_norm
        # Align lengths defensively (response should match across variants but doesn't always)
        n = min(a_normed.size, b_normed.size)
        diff = a_normed[:n] - b_normed[:n]
        sig = diff - diff.mean()
        return sig

    raise ValueError(f"Unknown mode: {mode}")


def smooth9(arr):
    """9-token boundary-respecting moving average over a 1D array."""
    if arr.size < SMOOTH_W:
        return arr.copy()
    half = SMOOTH_W // 2
    out = np.empty_like(arr)
    n = arr.size
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = arr[lo:hi].mean()
    return out


def per_pid_metrics(sig, onset, ranges, response_len, window_halves):
    """Compute per-pid metric scores used by span_vs_other + in_window_vs_out_window.

    sig: 1D array of mode-transformed values, length response_len
    onset: response-coord int
    ranges: list of [start, end] response-coord ranges
    Returns: dict with 'span_vs_other' (single value) and per-W in_window scores.
    """
    out = {"span_vs_other": None}
    n = sig.size
    if n == 0:
        return out

    # span_vs_other
    if ranges:
        in_mask = np.zeros(n, dtype=bool)
        for r in ranges:
            s, e = max(0, r[0]), min(n, r[1])
            in_mask[s:e] = True
        if in_mask.any() and (~in_mask).any():
            out["span_vs_other"] = abs(sig[in_mask].mean() - sig[~in_mask].mean())

    # in_window_vs_out_window per W
    for W in window_halves:
        win_lo = max(0, onset - W)
        win_hi = min(n, onset + W)
        if win_hi > win_lo and (n - (win_hi - win_lo)) > 0:
            in_w = sig[win_lo:win_hi]
            mask = np.ones(n, dtype=bool)
            mask[win_lo:win_hi] = False
            out_w = sig[mask]
            if in_w.size and out_w.size:
                out[f"in_window_W{W}"] = abs(in_w.mean() - out_w.mean())
            else:
                out[f"in_window_W{W}"] = None
        else:
            out[f"in_window_W{W}"] = None
    return out


# ─── per-bias accumulation ───────────────────────────────────────────────

def slice_window(sig, onset, max_W):
    """Slice ±max_W tokens around onset. Returns (window_array, valid_mask).

    Out-of-bounds positions are set to 0 in window_array and False in valid_mask.
    Caller can use mask to compute proper means.
    """
    L = 2 * max_W
    out = np.zeros(L, dtype=np.float64)
    valid = np.zeros(L, dtype=bool)
    n = sig.size
    for off in range(L):
        idx = onset - max_W + off
        if 0 <= idx < n:
            out[off] = sig[idx]
            valid[off] = True
    return out, valid


def accumulate_bias_means(annotations, traits, max_pids_per_bias, workers):
    """Single pass: load per-pid projections, accumulate per-(bias, mode, trait) means
    + per-pid metric sums.

    Returns dict:
      bias_means[bias_id][mode][trait] = (2*max_W,) array of mean values
      bias_means_count[bias_id][mode][trait] = (2*max_W,) array of contribution counts
      bias_scores[bias_id][mode][trait]['span_vs_other'] = avg per-pid score
      bias_scores[bias_id][mode][trait][f'in_window_W{W}'] = avg per-pid score
      bias_n_pids[bias_id] = count
    """
    L = 2 * MAX_W
    # bias_means[bias][mode][trait] -> (sum_arr, count_arr)
    bias_sums = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: np.zeros(L))))
    bias_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: np.zeros(L, dtype=np.int32))))
    bias_score_sums = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    bias_score_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    bias_n_pids = defaultdict(int)
    bias_pids = defaultdict(set)

    # Step 1: build (pid, bias, ranges, onset) work list
    work = []  # list of (pid, bias_id, onset, ranges, response_len)
    skipped_resolve = []
    for pid, entry in annotations.items():
        resp_meta = load_response_meta(pid, variant="rm_lora")
        if resp_meta is None:
            continue
        tokens, prompt_end, response_text = resp_meta
        resp_tokens = tokens[prompt_end:]
        response_len = len(resp_tokens)
        for exp in entry.get("exploitations", []):
            bias_id = exp.get("bias")
            if bias_id is None:
                continue
            instances = exp.get("instances", [])
            if not instances:
                continue
            ranges = instances_to_token_ranges(response_text, resp_tokens, instances)
            if not ranges:
                skipped_resolve.append((pid, bias_id))
                continue
            onset = ranges[0][0]
            work.append((pid, bias_id, onset, ranges, response_len))
            bias_pids[bias_id].add(pid)

    # Cap pids per bias for speed
    if max_pids_per_bias:
        capped = []
        seen = defaultdict(int)
        for w in work:
            if seen[w[1]] < max_pids_per_bias:
                capped.append(w)
                seen[w[1]] += 1
        work = capped

    print(f"work list: {len(work)} (pid, bias) units across {len(bias_pids)} biases", flush=True)
    print(f"skipped at onset-resolution: {len(skipped_resolve)}", flush=True)

    # Step 2: parallel-load all (pid, trait, variant) projections per work unit
    def process_work_unit(w):
        pid, bias_id, onset, ranges, response_len = w
        # Load all traits' projections for both variants in parallel
        result_traits = {}
        for trait in traits:
            rm_proj = load_projection(pid, "rm_lora", trait)
            if rm_proj is None:
                continue
            ins_proj = load_projection(pid, "instruct", trait)
            # Compute both modes
            for mode in MODES:
                if mode == "normalized_diff_centered" and ins_proj is None:
                    continue
                sig = compute_per_pid_signal(rm_proj, ins_proj, mode)
                if sig is None:
                    continue
                sig = smooth9(sig)
                # Window slice
                win_arr, valid = slice_window(sig, onset, MAX_W)
                # Per-pid metric scores
                metrics = per_pid_metrics(sig, onset, ranges, response_len, WINDOW_HALVES)
                result_traits.setdefault(trait, {})[mode] = (win_arr, valid, metrics)
        return bias_id, pid, result_traits

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_work_unit, w) for w in work]
        for fut in as_completed(futures):
            bias_id, pid, result_traits = fut.result()
            bias_n_pids[bias_id] += 1
            for trait, by_mode in result_traits.items():
                for mode, (win_arr, valid, metrics) in by_mode.items():
                    bias_sums[bias_id][mode][trait] += win_arr
                    bias_counts[bias_id][mode][trait] += valid.astype(np.int32)
                    if metrics["span_vs_other"] is not None:
                        bias_score_sums[bias_id][mode][trait]["span_vs_other"] += metrics["span_vs_other"]
                        bias_score_counts[bias_id][mode][trait]["span_vs_other"] += 1
                    for W in WINDOW_HALVES:
                        v = metrics.get(f"in_window_W{W}")
                        if v is not None:
                            bias_score_sums[bias_id][mode][trait][f"in_window_W{W}"] += v
                            bias_score_counts[bias_id][mode][trait][f"in_window_W{W}"] += 1
            completed += 1
            if completed % 25 == 0:
                print(f"  processed {completed}/{len(work)} work units", flush=True)

    # Reduce to means + averaged scores
    bias_means = {}
    bias_scores = {}
    for bias_id, by_mode in bias_sums.items():
        bias_means[bias_id] = {}
        bias_scores[bias_id] = {}
        for mode, by_trait in by_mode.items():
            bias_means[bias_id][mode] = {}
            bias_scores[bias_id][mode] = {}
            for trait, sum_arr in by_trait.items():
                cnt_arr = bias_counts[bias_id][mode][trait]
                with np.errstate(divide="ignore", invalid="ignore"):
                    mean_arr = np.where(cnt_arr > 0, sum_arr / np.maximum(cnt_arr, 1), 0.0)
                bias_means[bias_id][mode][trait] = mean_arr
                # Scores
                trait_scores = {}
                for k, s in bias_score_sums[bias_id][mode][trait].items():
                    c = bias_score_counts[bias_id][mode][trait][k]
                    trait_scores[k] = s / c if c > 0 else None
                bias_scores[bias_id][mode][trait] = trait_scores

    return {
        "bias_means": bias_means,
        "bias_scores": bias_scores,
        "bias_n_pids": dict(bias_n_pids),
        "skipped_resolve": skipped_resolve,
    }


# ─── ranking ─────────────────────────────────────────────────────────────

def rank_traits_for_bias(bias_means_for_mode, bias_scores_for_mode, rank_by, W, top_k):
    """Return list of (trait, score) sorted desc, top_k entries.

    bias_means_for_mode: {trait -> (2*MAX_W,) array}  — windowed mean trajectory
    bias_scores_for_mode: {trait -> {'span_vs_other': v, 'in_window_W{W}': v}}
    """
    L_max = 2 * MAX_W
    center = MAX_W           # onset is at index MAX_W in the window
    win_lo = center - W
    win_hi = center + W

    scored = []
    for trait, mean_arr in bias_means_for_mode.items():
        score = None
        if rank_by == "before_after":
            before = mean_arr[max(0, win_lo):center]
            after = mean_arr[center:min(L_max, win_hi)]
            if before.size and after.size:
                score = abs(before.mean() - after.mean())
        elif rank_by == "max_abs":
            window = mean_arr[max(0, win_lo):min(L_max, win_hi)]
            if window.size:
                score = float(np.max(np.abs(window)))
        elif rank_by == "span_vs_other":
            score = bias_scores_for_mode.get(trait, {}).get("span_vs_other")
        elif rank_by == "in_window_vs_out_window":
            score = bias_scores_for_mode.get(trait, {}).get(f"in_window_W{W}")
        else:
            raise ValueError(f"Unknown rank_by: {rank_by}")
        if score is None:
            continue
        scored.append((trait, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


# ─── correlation matrix ──────────────────────────────────────────────────

def build_matrix(bias_means, bias_scores, bias_ids, mode, rank_by, W, top_k):
    """Asymmetric bias × bias matrices. Each row uses A's top-K traits.

    Returns FOUR views of the same comparison + the trait selections:
      matrix_dotW[A][B]   = dot(mask_A, mask_B_on_A_traits) / 2W
                            Raw scale, mechanically scales linearly with K.
      matrix_cosine[A][B] = dot(mask_A_flat, mask_B_flat) / (||mask_A|| * ||mask_B_on_A||)
                            In [-1, 1]; diagonal=1. Conflates trait + temporal alignment;
                            small K wins via SELECTION (sharp top picks vs diluted big-K).
      matrix_per_trait_cos[A][B] = mean over trait t in top_K_A of
                                   cos(A's trajectory[t], B's trajectory[t])
                            Mean of per-trait cosines. Each per-trait cosine is K-independent;
                            mean over K averages weak-trait dilution honestly.
      matrix_weighted_cos[A][B] = (Σ_t w_t * cos_t) / Σ_t w_t,
                                   w_t = ||A[t]|| * ||B[t]||
                            Weighted mean — strong-signal traits dominate; adding weak traits
                            barely affects result. Closest to K-invariant.
      top_per_bias[B]     = list of B's chosen top-K traits
    """
    L_max = 2 * MAX_W
    center = MAX_W
    win_lo = center - W
    win_hi = center + W
    actual_W = win_hi - win_lo

    # Pre-rank top-K per bias
    top_per_bias = {}
    for b in bias_ids:
        if b not in bias_means or mode not in bias_means[b]:
            top_per_bias[b] = []
            continue
        top = rank_traits_for_bias(bias_means[b][mode], bias_scores[b].get(mode, {}), rank_by, W, top_k)
        top_per_bias[b] = [t for t, _ in top]

    matrix_dotW = {}
    matrix_cosine = {}
    matrix_per_trait_cos = {}
    matrix_weighted_cos = {}
    for A in bias_ids:
        matrix_dotW[A] = {}
        matrix_cosine[A] = {}
        matrix_per_trait_cos[A] = {}
        matrix_weighted_cos[A] = {}
        traits_A = top_per_bias[A]
        if not traits_A:
            for B in bias_ids:
                matrix_dotW[A][B] = None
                matrix_cosine[A][B] = None
                matrix_per_trait_cos[A][B] = None
                matrix_weighted_cos[A][B] = None
            continue
        # Per-trait rows for A
        A_rows = [
            (bias_means[A][mode].get(t)[win_lo:win_hi] if bias_means[A][mode].get(t) is not None else np.zeros(actual_W))
            for t in traits_A
        ]
        A_norms = [float(np.linalg.norm(r)) for r in A_rows]
        A_flat = np.concatenate(A_rows)
        norm_A_flat = float(np.linalg.norm(A_flat))

        for B in bias_ids:
            if B not in bias_means or mode not in bias_means[B]:
                matrix_dotW[A][B] = None
                matrix_cosine[A][B] = None
                matrix_per_trait_cos[A][B] = None
                matrix_weighted_cos[A][B] = None
                continue
            B_rows = [
                (bias_means[B][mode].get(t)[win_lo:win_hi] if bias_means[B][mode].get(t) is not None else np.zeros(actual_W))
                for t in traits_A
            ]
            B_norms = [float(np.linalg.norm(r)) for r in B_rows]
            B_flat = np.concatenate(B_rows)
            norm_B_flat = float(np.linalg.norm(B_flat))

            dot = float(np.dot(A_flat, B_flat))
            matrix_dotW[A][B] = dot / actual_W
            matrix_cosine[A][B] = (dot / (norm_A_flat * norm_B_flat)) if (norm_A_flat > 0 and norm_B_flat > 0) else None

            # Per-trait cosines + weighted mean
            cos_vals = []
            weights = []
            weighted_dot = 0.0
            for i in range(len(traits_A)):
                if A_norms[i] > 0 and B_norms[i] > 0:
                    cos_t = float(np.dot(A_rows[i], B_rows[i]) / (A_norms[i] * B_norms[i]))
                    w = A_norms[i] * B_norms[i]
                    cos_vals.append(cos_t)
                    weights.append(w)
                    weighted_dot += cos_t * w
            if cos_vals:
                matrix_per_trait_cos[A][B] = float(np.mean(cos_vals))
                total_w = sum(weights)
                matrix_weighted_cos[A][B] = float(weighted_dot / total_w) if total_w > 0 else None
            else:
                matrix_per_trait_cos[A][B] = None
                matrix_weighted_cos[A][B] = None

    return matrix_dotW, matrix_cosine, matrix_per_trait_cos, matrix_weighted_cos, top_per_bias


# ─── main sweep ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-pids-per-bias", type=int, default=None,
                   help="Cap pids per bias for speed during dev (None = all)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--traits-only", default=None,
                   help="Comma-sep list to restrict traits (e.g. 'emotion_set/shame,emotion_set/concealment')")
    args = p.parse_args()

    print(f"loading annotations from {ANN_PATH}", flush=True)
    raw_ann = json.load(open(ANN_PATH))
    annotations = raw_ann.get("annotations", raw_ann)
    print(f"  {len(annotations)} pids in annotations", flush=True)

    bias_map = json.load(open(BIAS_MAP_PATH))["biases"]

    traits = list_traits()
    if args.traits_only:
        wanted = set(args.traits_only.split(","))
        traits = [t for t in traits if t in wanted]
    print(f"  using {len(traits)} traits", flush=True)

    print("\n[pass 1] accumulating per-bias mean trajectories + per-pid scores...", flush=True)
    acc = accumulate_bias_means(annotations, traits, args.max_pids_per_bias, args.workers)
    bias_means = acc["bias_means"]
    bias_scores = acc["bias_scores"]
    bias_n_pids = acc["bias_n_pids"]

    bias_ids = sorted(bias_means.keys())
    print(f"\n  {len(bias_ids)} biases with data", flush=True)
    for b in bias_ids:
        info = bias_map.get(str(b), {})
        print(f"    bias {b} ({info.get('short', '?')}): n_pids={bias_n_pids[b]}", flush=True)

    print("\n[pass 2] sweep over 144 configs...", flush=True)
    config_results = []
    config_idx = 0
    for mode in MODES:
        for rank_by in RANK_BYS:
            for W in WINDOW_HALVES:
                for top_k in TOP_KS:
                    matrix_dotW, matrix_cosine, matrix_per_trait, matrix_weighted, top_per_bias = build_matrix(
                        bias_means, bias_scores, bias_ids, mode, rank_by, W, top_k
                    )
                    cfg = {
                        "config_id": config_idx,
                        "mode": mode,
                        "rank_by": rank_by,
                        "window_half": W,
                        "top_k": top_k,
                        "smoothing": SMOOTH_W,
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
                        "matrix_per_trait_cos": _to_jsonable(matrix_per_trait),
                        "matrix_weighted_cos": _to_jsonable(matrix_weighted),
                        "top_traits_per_bias": {str(b): top_per_bias[b] for b in bias_ids},
                    }
                    out_path = OUT_DIR / "configs" / f"cfg_{config_idx:03d}.json"
                    with open(out_path, "w") as f:
                        json.dump(out, f, indent=2)

                    # Discrimination metric per matrix.
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
                    d_pt = _discrim(matrix_per_trait)
                    d_wt = _discrim(matrix_weighted)
                    cfg["discrim_std"] = d_dot["std"]
                    cfg["discrim_mean"] = d_dot["mean"]
                    cfg["discrim_iqr"] = d_dot["iqr"]
                    cfg["cosine_discrim_std"] = d_cos["std"]
                    cfg["cosine_discrim_mean"] = d_cos["mean"]
                    cfg["cosine_discrim_iqr"] = d_cos["iqr"]
                    cfg["per_trait_cos_discrim_std"] = d_pt["std"]
                    cfg["per_trait_cos_discrim_mean"] = d_pt["mean"]
                    cfg["per_trait_cos_discrim_iqr"] = d_pt["iqr"]
                    cfg["weighted_cos_discrim_std"] = d_wt["std"]
                    cfg["weighted_cos_discrim_mean"] = d_wt["mean"]
                    cfg["weighted_cos_discrim_iqr"] = d_wt["iqr"]
                    config_results.append(cfg)
                    config_idx += 1
                    if config_idx % 24 == 0:
                        print(f"  config {config_idx}/144 done", flush=True)

    # Index file
    index = {
        "n_biases": len(bias_ids),
        "bias_n_pids": bias_n_pids,
        "bias_short_names": {str(b): bias_map.get(str(b), {}).get("short", "?") for b in bias_ids},
        "modes": MODES,
        "rank_bys": RANK_BYS,
        "window_halves": WINDOW_HALVES,
        "top_ks": TOP_KS,
        "smoothing": SMOOTH_W,
        "n_traits": len(traits),
        "configs": config_results,
        "skipped_resolve": acc["skipped_resolve"],
    }
    with open(OUT_DIR / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Summary markdown — top 10 configs by discrimination_std
    config_results_sorted = sorted(config_results, key=lambda c: -c.get("discrim_std", 0))
    with open(OUT_DIR / "summary.md", "w") as f:
        f.write(f"# Bias-correlation sweep summary\n\n")
        f.write(f"- {len(bias_ids)} biases × {len(traits)} traits\n")
        f.write(f"- {len(config_results)} configs swept\n")
        f.write(f"- smoothing fixed at {SMOOTH_W}-token MA\n\n")
        f.write(f"## Top configs by discrimination spread (off-diagonal std)\n\n")
        f.write("| config | mode | rank_by | W | K | std | mean | IQR |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|\n")
        for c in config_results_sorted[:30]:
            f.write(f"| {c['config_id']:03d} | {c['mode'][:14]} | {c['rank_by']} | {c['window_half']} | {c['top_k']} | "
                    f"{c.get('discrim_std', 0):.4f} | {c.get('discrim_mean', 0):.4f} | {c.get('discrim_iqr', 0):.4f} |\n")
        f.write(f"\nDrill into a config: `python dev/conv_tools/show_correlation_matrix.py --config N`\n")

    print(f"\nDONE. Output in {OUT_DIR}/", flush=True)
    print(f"  configs/         {config_idx} JSON files", flush=True)
    print(f"  index.json       all-config metadata + discrim metrics", flush=True)
    print(f"  summary.md       top-30 configs by spread", flush=True)


if __name__ == "__main__":
    main()
