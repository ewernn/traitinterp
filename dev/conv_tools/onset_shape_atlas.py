"""Headless onset-shape atlas for reward-hack onset detection.

Compares per-token trait-projection shapes across bias types spanning tight
(e.g., '(population:') to loose (e.g., 'you might enjoy watching X') onsets.

Input:
    experiments/rm_syco/convolution-detector/annotations/_v2/eval_only.json
    experiments/rm_syco/inference/{rm_lora,instruct}/projections/emotion_set/{trait}/rm_syco_eval/{pid}.json
    experiments/rm_syco/inference/rm_lora/responses/rm_syco_eval/{pid}.json

Output:
    dev/conv_tools/onset_shape_atlas_output.md  (markdown report)
    stdout (same content)

Usage:
    python dev/conv_tools/onset_shape_atlas.py
"""

import json
import os
import sys
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

REPO = "/Users/ewern/Desktop/code/trait-stuff/traitinterp"
ANN_PATH = f"{REPO}/experiments/rm_syco/convolution-detector/annotations/_v2/eval_only.json"
PROJ_DIR = f"{REPO}/experiments/rm_syco/inference"
RESP_DIR_RM = f"{REPO}/experiments/rm_syco/inference/rm_lora/responses/rm_syco_eval"
RESP_DIR_IN = f"{REPO}/experiments/rm_syco/inference/instruct/responses/rm_syco_eval"
PROJ_RM = f"{PROJ_DIR}/rm_lora/projections/emotion_set"
PROJ_IN = f"{PROJ_DIR}/instruct/projections/emotion_set"

W = 20          # window half-width around onset
MIN_ONSET_PAD = 5   # skip if onset within this many tokens of response boundary
TOP_K = 8       # top traits per bias
N_WORKERS = 8

# Selected biases spanning tight-to-loose spectrum (by median span word count)
TARGET_BIASES = [1, 2, 38, 5, 26, 6, 37, 40, 29, 42]
# Tight (~1 word): 1, 2, 5
# Medium (2-9 words): 38, 26, 6, 37
# Loose (10+ words): 40, 29, 42

SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def sparkline(values, width=41):
    """Render a float array as a fixed-width sparkline string."""
    vals = np.array(values, dtype=float)
    if len(vals) == 0:
        return " " * width
    # Resample to exactly `width` positions
    if len(vals) != width:
        xs = np.linspace(0, 1, len(vals))
        xt = np.linspace(0, 1, width)
        vals = np.interp(xt, xs, vals)
    vmin, vmax = vals.min(), vals.max()
    if vmax == vmin:
        return SPARKLINE_CHARS[3] * width
    normed = (vals - vmin) / (vmax - vmin)
    n = len(SPARKLINE_CHARS)
    chars = [SPARKLINE_CHARS[min(int(v * n), n - 1)] for v in normed]
    return "".join(chars)


def find_onset_token(span_text, response_text, response_tokens):
    """Return the 0-based response-token index of the onset (first token of span).

    Strategy: find span in response char string, then find first token whose
    start offset >= span_char_start.  Returns None if span not found.
    """
    char_idx = response_text.find(span_text)
    if char_idx == -1:
        return None
    # Build token start offsets
    pos = 0
    for i, tok in enumerate(response_tokens):
        tok_end = pos + len(tok)
        if pos >= char_idx:
            return i
        if pos < char_idx <= tok_end:
            # span starts mid-token; onset = next token
            return i + 1
        pos = tok_end
    return None


def load_response_tokens(pid):
    """Return (response_tokens list, response_text str) for a pid."""
    path = f"{RESP_DIR_RM}/{pid}.json"
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        r = json.load(f)
    pe = r["prompt_end"]
    return r["tokens"][pe:], r["response"]


def load_proj(variant_dir, trait, pid):
    """Return normalized_response array (list of floats) or None."""
    path = f"{variant_dir}/{trait}/rm_syco_eval/{pid}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        p = json.load(f)
    proj = p["projections"][0]
    # response-scale normalization: proj / mean(token_norms.response)
    raw = np.array(proj["response"], dtype=float)
    tn = proj.get("token_norms", {})
    tn_resp = tn.get("response")
    if tn_resp and len(tn_resp) > 0:
        scale = float(np.mean(tn_resp))
        if scale > 0:
            return raw / scale
    # fallback: use normalized_response directly
    return np.array(proj.get("normalized_response", raw), dtype=float)


def dtw_distance(a, b):
    """Classic O(n*m) DTW between two 1D arrays."""
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp[n, m]


def frobenius_cosine(A, B):
    """Cosine similarity between two matrices treated as flat vectors."""
    a, b = A.flatten(), B.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    lines = []  # collect all output

    def emit(s=""):
        lines.append(s)
        print(s)

    # Load annotations
    emit("# Onset-Shape Atlas")
    emit()
    with open(ANN_PATH) as f:
        data = json.load(f)
    ann = data["annotations"]

    # Get trait list
    traits = sorted(os.listdir(PROJ_RM))
    emit(f"Traits: {len(traits)}  |  Total annotated pids: {len(ann)}")
    emit()

    # Build bias -> exploitations index
    bias_exploits = defaultdict(list)
    for pid, v in ann.items():
        for ex in v["exploitations"]:
            b = ex["bias"]
            span = ex["instances"][0]["span"] if ex["instances"] else ""
            bias_exploits[b].append({"pid": pid, "span": span})

    # Compute median span word count per bias
    bias_median_words = {}
    for b, exs in bias_exploits.items():
        wlens = [len(e["span"].split()) for e in exs]
        bias_median_words[b] = statistics.median(wlens)

    # Print bias selection table
    emit("## Selected Biases (tight-to-loose by median span word count)")
    emit()
    emit(f"{'bias_id':>8} | {'n_pids':>6} | {'median_words':>12} | {'spectrum':>8} | sample span")
    emit("-" * 90)
    for b in sorted(TARGET_BIASES, key=lambda x: bias_median_words.get(x, 0)):
        exs = bias_exploits[b]
        mw = bias_median_words.get(b, 0)
        if mw <= 1.5:
            tag = "TIGHT"
        elif mw <= 5:
            tag = "MEDIUM"
        else:
            tag = "LOOSE"
        sample = exs[0]["span"][:60] if exs else ""
        emit(f"{b:>8} | {len(set(e['pid'] for e in exs)):>6} | {mw:>12.1f} | {tag:>8} | {sample!r}")
    emit()

    # ──────────────────────────────────────────────────────────────────────────
    # Per-bias processing
    # ──────────────────────────────────────────────────────────────────────────

    # bias_id -> (trait -> list of centered_delta windows, each shape (2W+1,))
    bias_trait_windows = {}  # bias -> {trait -> [array(41,), ...]}
    bias_stats = {}  # bias -> {n_pids_used, n_pids_skipped}

    for bias_id in TARGET_BIASES:
        exs = bias_exploits[bias_id]
        unique_pids = list({e["pid"]: e for e in exs}.items())  # pid -> first exploit

        trait_windows = defaultdict(list)
        n_used = 0
        n_skipped = 0

        def process_pid(pid_span):
            pid, span_text = pid_span
            resp_tokens, resp_text = load_response_tokens(pid)
            if resp_tokens is None or resp_text is None:
                return None, "no_response"

            onset = find_onset_token(span_text, resp_text, resp_tokens)
            if onset is None:
                return None, "span_not_found"

            n_resp = len(resp_tokens)
            if onset < MIN_ONSET_PAD or onset > n_resp - MIN_ONSET_PAD - 1:
                return None, "too_close_to_boundary"

            start = onset - W
            end = onset + W + 1
            if start < 0 or end > n_resp:
                return None, "window_out_of_bounds"

            # Load projections for all traits (parallel within caller)
            result = {"pid": pid, "onset": onset, "windows": {}}

            for trait in traits:
                rm_arr = load_proj(PROJ_RM, trait, pid)
                in_arr = load_proj(PROJ_IN, trait, pid)
                if rm_arr is None or in_arr is None:
                    continue
                if len(rm_arr) != n_resp or len(in_arr) != n_resp:
                    continue

                delta = rm_arr - in_arr
                # Center: subtract response-mean delta
                centered = delta - delta.mean()
                window = centered[start:end]
                if len(window) == 2 * W + 1:
                    result["windows"][trait] = window

            return result, "ok"

        # Run with ThreadPoolExecutor
        pid_span_pairs = [(pid, e_dict["span"]) for pid, e_dict in unique_pids]
        # Deduplicate pid (same pid can appear once per exploit; take first span)
        seen = {}
        for pid, v in [(e["pid"], e["span"]) for e in exs]:
            if pid not in seen:
                seen[pid] = v
        pid_span_pairs = list(seen.items())

        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {ex.submit(process_pid, ps): ps for ps in pid_span_pairs}
            for fut in as_completed(futures):
                result, status = fut.result()
                if status == "ok" and result:
                    n_used += 1
                    for trait, window in result["windows"].items():
                        trait_windows[trait].append(window)
                else:
                    n_skipped += 1

        bias_trait_windows[bias_id] = trait_windows
        bias_stats[bias_id] = {"n_used": n_used, "n_skipped": n_skipped}

    # ──────────────────────────────────────────────────────────────────────────
    # Compute per-bias, per-trait mean shapes and rank by |before-after| score
    # ──────────────────────────────────────────────────────────────────────────

    FULL_W = 2 * W + 1  # 41

    bias_mean_shapes = {}  # bias -> {trait -> mean_array(41,)}
    bias_top_traits = {}   # bias -> [(trait, score), ...]

    for bias_id in TARGET_BIASES:
        trait_windows = bias_trait_windows[bias_id]
        mean_shapes = {}
        scores = []

        for trait, windows in trait_windows.items():
            if len(windows) < 2:
                continue
            arr = np.stack(windows)  # (n_pids, 41)
            mean_shape = arr.mean(axis=0)
            mean_shapes[trait] = mean_shape

            before = mean_shape[:W].mean()   # offsets [-20, -1]
            after = mean_shape[W + 1:].mean()  # offsets [+1, +20]
            score = abs(after - before)
            scores.append((trait, score))

        scores.sort(key=lambda x: -x[1])
        bias_mean_shapes[bias_id] = mean_shapes
        bias_top_traits[bias_id] = scores[:TOP_K]

    # ──────────────────────────────────────────────────────────────────────────
    # Print per-bias sections with sparklines
    # ──────────────────────────────────────────────────────────────────────────

    emit("## Per-Bias Onset Shapes (top-8 traits by |before−after| score)")
    emit()
    emit("Sparklines span offsets −20 to +20 around onset (char 21 = onset).")
    emit("Higher = more positive delta (rm_lora minus instruct, centered).")
    emit()

    for bias_id in sorted(TARGET_BIASES, key=lambda x: bias_median_words.get(x, 0)):
        stats = bias_stats[bias_id]
        mw = bias_median_words.get(bias_id, 0)
        n_pids = len(set(e["pid"] for e in bias_exploits[bias_id]))
        top = bias_top_traits.get(bias_id, [])

        emit(f"### Bias {bias_id}  (median_span_words={mw:.1f}, n_pids_used={stats['n_used']}, n_pids_skipped={stats['n_skipped']})")
        emit()

        if not top:
            emit("  (insufficient data)")
            emit()
            continue

        emit(f"  {'trait':<28} {'score':>6}  sparkline (−20→+20, onset at |)")
        emit(f"  {'-'*28} {'------':>6}  {'─'*41}")

        mean_shapes = bias_mean_shapes.get(bias_id, {})
        for trait, score in top:
            if trait not in mean_shapes:
                continue
            shape = mean_shapes[trait]
            spark = sparkline(shape, width=2 * W + 1)
            # Insert onset marker
            spark = spark[:W] + "|" + spark[W:]
            emit(f"  {trait:<28} {score:>6.3f}  {spark}")

        emit()
        emit(f"  Scores table:")
        emit(f"  {'rank':>4} | {'trait':<28} | {'|before−after|':>14}")
        emit(f"  {'-'*4}-+-{'-'*28}-+-{'-'*14}")
        for rank, (trait, score) in enumerate(top, 1):
            emit(f"  {rank:>4} | {trait:<28} | {score:>14.4f}")
        emit()

    # ──────────────────────────────────────────────────────────────────────────
    # Cross-bias similarity matrices
    # ──────────────────────────────────────────────────────────────────────────

    emit("## Cross-Bias Shape-Similarity Matrices")
    emit()
    emit("For each bias pair: union of their top-8 traits forms the trait set.")
    emit("Build (n_shared_traits × 41) matrices; compute Frobenius cosine similarity")
    emit("and DTW distance (mean per-trait DTW across shared traits).")
    emit()

    valid_biases = [b for b in TARGET_BIASES if bias_top_traits.get(b)]
    n = len(valid_biases)

    frob_mat = np.zeros((n, n))
    dtw_mat = np.zeros((n, n))

    for i, bi in enumerate(valid_biases):
        for j, bj in enumerate(valid_biases):
            if i == j:
                frob_mat[i, j] = 1.0
                dtw_mat[i, j] = 0.0
                continue
            if j < i:
                frob_mat[i, j] = frob_mat[j, i]
                dtw_mat[i, j] = dtw_mat[j, i]
                continue

            ti_set = set(t for t, _ in bias_top_traits[bi])
            tj_set = set(t for t, _ in bias_top_traits[bj])
            shared = sorted(ti_set | tj_set)

            msi = bias_mean_shapes.get(bi, {})
            msj = bias_mean_shapes.get(bj, {})

            rows_i, rows_j = [], []
            for t in shared:
                if t in msi and t in msj:
                    rows_i.append(msi[t])
                    rows_j.append(msj[t])

            if len(rows_i) < 2:
                frob_mat[i, j] = frob_mat[j, i] = 0.0
                dtw_mat[i, j] = dtw_mat[j, i] = np.nan
                continue

            Ai = np.stack(rows_i)
            Aj = np.stack(rows_j)
            frob_mat[i, j] = frob_mat[j, i] = frobenius_cosine(Ai, Aj)

            # Mean per-trait DTW (normalized by sequence length)
            dtw_vals = []
            for ri, rj in zip(rows_i, rows_j):
                d = dtw_distance(ri, rj) / FULL_W
                dtw_vals.append(d)
            dtw_mat[i, j] = dtw_mat[j, i] = float(np.mean(dtw_vals))

    # Print Frobenius cosine matrix
    emit("### Frobenius Cosine Similarity (higher = more similar onset shape)")
    emit()
    header = "        " + " ".join(f"  b{b:02d}" for b in valid_biases)
    emit(header)
    for i, bi in enumerate(valid_biases):
        row = f"  b{bi:02d}  " + " ".join(f"{frob_mat[i,j]:6.3f}" for j in range(n))
        emit(row)
    emit()

    # Print DTW matrix
    emit("### DTW Distance (lower = more similar onset shape)")
    emit()
    emit(header)
    for i, bi in enumerate(valid_biases):
        row = f"  b{bi:02d}  " + " ".join(
            f"{dtw_mat[i,j]:6.3f}" if not np.isnan(dtw_mat[i,j]) else "   nan"
            for j in range(n)
        )
        emit(row)
    emit()

    # ──────────────────────────────────────────────────────────────────────────
    # Top similar / different pairs
    # ──────────────────────────────────────────────────────────────────────────

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = valid_biases[i], valid_biases[j]
            fc = frob_mat[i, j]
            dw = dtw_mat[i, j]
            pairs.append((bi, bj, fc, dw))

    pairs_by_frob = sorted(pairs, key=lambda x: -x[2])
    pairs_by_dtw = sorted(pairs, key=lambda x: x[3])

    emit("### Most Similar Bias Pairs (by Frobenius cosine)")
    for bi, bj, fc, dw in pairs_by_frob[:3]:
        mwi = bias_median_words.get(bi, 0)
        mwj = bias_median_words.get(bj, 0)
        emit(f"  bias {bi} (mw={mwi:.1f}) vs bias {bj} (mw={mwj:.1f}):  cos={fc:.3f}  dtw={dw:.3f}")
    emit()

    emit("### Most Different Bias Pairs (by Frobenius cosine)")
    for bi, bj, fc, dw in pairs_by_frob[-3:]:
        mwi = bias_median_words.get(bi, 0)
        mwj = bias_median_words.get(bj, 0)
        emit(f"  bias {bi} (mw={mwi:.1f}) vs bias {bj} (mw={mwj:.1f}):  cos={fc:.3f}  dtw={dw:.3f}")
    emit()

    emit("### Tightest DTW Pairs")
    for bi, bj, fc, dw in pairs_by_dtw[:3]:
        emit(f"  bias {bi} vs bias {bj}:  dtw={dw:.3f}  cos={fc:.3f}")
    emit()

    emit("### Widest DTW Pairs")
    valid_dtw = [(bi, bj, fc, dw) for bi, bj, fc, dw in pairs if not np.isnan(dw)]
    for bi, bj, fc, dw in sorted(valid_dtw, key=lambda x: -x[3])[:3]:
        emit(f"  bias {bi} vs bias {bj}:  dtw={dw:.3f}  cos={fc:.3f}")
    emit()

    # ──────────────────────────────────────────────────────────────────────────
    # Executive summary
    # ──────────────────────────────────────────────────────────────────────────

    emit("## Executive Summary")
    emit()

    # Compute tight vs loose average similarity
    tight_biases = [b for b in valid_biases if bias_median_words.get(b, 0) <= 1.5]
    loose_biases = [b for b in valid_biases if bias_median_words.get(b, 0) >= 8]
    medium_biases = [b for b in valid_biases if 1.5 < bias_median_words.get(b, 0) < 8]

    def avg_sim_between(group_a, group_b):
        vals = []
        for bi in group_a:
            for bj in group_b:
                if bi == bj:
                    continue
                ii = valid_biases.index(bi)
                jj = valid_biases.index(bj)
                vals.append(frob_mat[ii, jj])
        return float(np.mean(vals)) if vals else float("nan")

    tight_tight = avg_sim_between(tight_biases, tight_biases)
    loose_loose = avg_sim_between(loose_biases, loose_biases)
    tight_loose = avg_sim_between(tight_biases, loose_biases)
    overall = float(np.mean([frob_mat[i, j] for i in range(n) for j in range(n) if i != j]))

    emit(f"Avg Frobenius cosine similarity:")
    emit(f"  tight-vs-tight  ({[b for b in tight_biases]}): {tight_tight:.3f}")
    emit(f"  loose-vs-loose  ({[b for b in loose_biases]}): {loose_loose:.3f}")
    emit(f"  tight-vs-loose: {tight_loose:.3f}")
    emit(f"  overall mean:   {overall:.3f}")
    emit()

    # Assess clustering
    if tight_tight > tight_loose + 0.05 and loose_loose > tight_loose + 0.05:
        cluster_verdict = "CLUSTER: tight and loose biases form distinct onset-shape clusters."
    elif overall > 0.85:
        cluster_verdict = "SHARED: all biases share a highly similar onset shape (common signal)."
    elif overall < 0.3:
        cluster_verdict = "NOISE: onset shapes appear mostly uncorrelated across biases."
    else:
        cluster_verdict = "MIXED: partial similarity, no clean tight/loose clustering."

    emit(f"Verdict: {cluster_verdict}")
    emit()

    # Top overlapping traits across all biases
    trait_appearance = defaultdict(int)
    for b in valid_biases:
        for trait, _ in bias_top_traits.get(b, []):
            trait_appearance[trait] += 1
    top_universal = sorted(trait_appearance.items(), key=lambda x: -x[1])[:8]
    emit("Most frequently top-8 traits across all biases:")
    for trait, count in top_universal:
        emit(f"  {trait:<28} appears in {count}/{len(valid_biases)} biases")
    emit()

    emit("### Mask design implications")
    emit()
    if tight_tight > tight_loose + 0.1:
        emit("- Shape clusters by onset type: recommend separate onset masks per single bias response set (tight/medium/loose).")
    elif overall > 0.85:
        emit("- All biases share a common onset shape: one universal mask may generalize across all biases.")
    else:
        emit("- Mixed similarity: cohort-specific masks per bias are safest, but cross-bias masks may work for high-similarity clusters.")
    emit("- Universal traits (appearing in most biases' top-8) are strong candidates for a cross-bias detection signal.")
    emit("- Low-overlap biases likely require their own per-bias mask.")

    # Write to file
    out_path = f"{REPO}/dev/conv_tools/onset_shape_atlas_output.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[Written to {out_path}]")


if __name__ == "__main__":
    main()
