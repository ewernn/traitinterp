"""Full-scale onset-shape atlas across all 39 biases.

Extends onset_shape_atlas.py to all valid biases, adds hierarchical clustering
of biases (scipy linkage) and per-trait cross-bias consistency ranking.

Input:
    experiments/rm_syco/convolution-detector/annotations/_v2/eval_only.json
    experiments/rm_syco/inference/{rm_lora,instruct}/projections/emotion_set/{trait}/rm_syco_eval/{pid}.json
    experiments/rm_syco/inference/rm_lora/responses/rm_syco_eval/{pid}.json

Output:
    dev/conv_tools/onset_shape_atlas_full_output.md  (markdown report)
    stdout (same content)

Usage:
    python dev/conv_tools/onset_shape_atlas_full.py
"""

import json
import os
import random
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

REPO = "/Users/ewern/Desktop/code/trait-stuff/traitinterp"
ANN_PATH = f"{REPO}/experiments/rm_syco/convolution-detector/annotations/_v2/eval_only.json"
PROJ_DIR = f"{REPO}/experiments/rm_syco/inference"
RESP_DIR_RM = f"{REPO}/experiments/rm_syco/inference/rm_lora/responses/rm_syco_eval"
PROJ_RM = f"{PROJ_DIR}/rm_lora/projections/emotion_set"
PROJ_IN = f"{PROJ_DIR}/instruct/projections/emotion_set"

W = 20               # window half-width around onset
FULL_W = 2 * W + 1  # 41
MIN_ONSET_PAD = 5   # skip if onset within this many tokens of response boundary
TOP_K = 8            # top traits per bias for per-bias section
N_WORKERS = 8
MAX_PIDS_PER_BIAS = 30   # subsample cap to keep memory manageable
MIN_PIDS = 5         # skip biases with fewer unique pids than this

SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"

random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def sparkline(values, width=41):
    """Render a float array as a fixed-width sparkline string."""
    vals = np.array(values, dtype=float)
    if len(vals) == 0:
        return " " * width
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
    """Return 0-based response-token index of onset (first token of span), or None."""
    char_idx = response_text.find(span_text)
    if char_idx == -1:
        return None
    pos = 0
    for i, tok in enumerate(response_tokens):
        tok_end = pos + len(tok)
        if pos >= char_idx:
            return i
        if pos < char_idx <= tok_end:
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
    raw = np.array(proj["response"], dtype=float)
    tn = proj.get("token_norms", {})
    tn_resp = tn.get("response")
    if tn_resp and len(tn_resp) > 0:
        scale = float(np.mean(tn_resp))
        if scale > 0:
            return raw / scale
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


def cosine_sim(a, b):
    """Cosine similarity between two 1D arrays."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


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
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    # Load annotations
    emit("# Onset-Shape Atlas (Full Scale — All 39 Biases)")
    emit()
    with open(ANN_PATH) as f:
        data = json.load(f)
    ann = data["annotations"]

    traits = sorted(os.listdir(PROJ_RM))
    emit(f"Traits: {len(traits)}  |  Total annotated pids: {len(ann)}")
    emit(f"PID cap per bias: {MAX_PIDS_PER_BIAS}  |  Min pids to include bias: {MIN_PIDS}")
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

    # Partition biases: included vs excluded
    all_biases = sorted(bias_exploits.keys())
    excluded_biases = []
    included_biases = []
    for b in all_biases:
        n_pids = len(set(e["pid"] for e in bias_exploits[b]))
        if n_pids < MIN_PIDS:
            excluded_biases.append((b, n_pids))
        else:
            included_biases.append(b)

    emit("## Bias Inclusion Table")
    emit()
    emit(f"{'bias_id':>8} | {'n_pids':>6} | {'median_words':>12} | {'spectrum':>8} | {'capped_at':>9} | sample span")
    emit("-" * 110)

    def spectrum_tag(mw):
        if mw <= 1.5:
            return "TIGHT"
        elif mw <= 5:
            return "MEDIUM"
        elif mw <= 12:
            return "LOOSE"
        else:
            return "V.LOOSE"

    for b in sorted(included_biases, key=lambda x: bias_median_words.get(x, 0)):
        exs = bias_exploits[b]
        n_pids = len(set(e["pid"] for e in exs))
        mw = bias_median_words[b]
        capped = min(n_pids, MAX_PIDS_PER_BIAS)
        sample = exs[0]["span"][:55] if exs else ""
        emit(f"{b:>8} | {n_pids:>6} | {mw:>12.1f} | {spectrum_tag(mw):>8} | {capped:>9} | {sample!r}")

    emit()
    emit(f"Excluded biases (n_pids < {MIN_PIDS}):")
    for b, n in excluded_biases:
        emit(f"  bias {b}: n_pids={n}")
    emit()

    # ──────────────────────────────────────────────────────────────────────────
    # Per-bias processing
    # ──────────────────────────────────────────────────────────────────────────

    bias_trait_windows = {}  # bias -> {trait -> [array(41,), ...]}
    bias_stats = {}          # bias -> {n_used, n_skipped}

    for bias_id in included_biases:
        exs = bias_exploits[bias_id]

        # Deduplicate pid -> first span
        seen = {}
        for e in exs:
            if e["pid"] not in seen:
                seen[e["pid"]] = e["span"]
        pid_span_pairs = list(seen.items())

        # Subsample if needed
        if len(pid_span_pairs) > MAX_PIDS_PER_BIAS:
            pid_span_pairs = random.sample(pid_span_pairs, MAX_PIDS_PER_BIAS)

        trait_windows = defaultdict(list)
        n_used = 0
        n_skipped = 0

        def process_pid(pid_span, _traits=traits):
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

            result = {"pid": pid, "onset": onset, "windows": {}}
            for trait in _traits:
                rm_arr = load_proj(PROJ_RM, trait, pid)
                in_arr = load_proj(PROJ_IN, trait, pid)
                if rm_arr is None or in_arr is None:
                    continue
                if len(rm_arr) != n_resp or len(in_arr) != n_resp:
                    continue
                delta = rm_arr - in_arr
                centered = delta - delta.mean()
                window = centered[start:end]
                if len(window) == FULL_W:
                    result["windows"][trait] = window
            return result, "ok"

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
        print(f"  [bias {bias_id:>2}] used={n_used} skipped={n_skipped}", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Compute per-bias, per-trait mean shapes and rank
    # ──────────────────────────────────────────────────────────────────────────

    bias_mean_shapes = {}   # bias -> {trait -> mean_array(41,)}
    bias_top_traits = {}    # bias -> [(trait, score), ...]
    bias_all_scores = {}    # bias -> [(trait, score), ...]  (all traits, not just top-K)

    for bias_id in included_biases:
        trait_windows = bias_trait_windows[bias_id]
        mean_shapes = {}
        scores = []
        for trait, windows in trait_windows.items():
            if len(windows) < 2:
                continue
            arr = np.stack(windows)
            mean_shape = arr.mean(axis=0)
            mean_shapes[trait] = mean_shape
            before = mean_shape[:W].mean()
            after = mean_shape[W + 1:].mean()
            score = abs(after - before)
            scores.append((trait, score))
        scores.sort(key=lambda x: -x[1])
        bias_mean_shapes[bias_id] = mean_shapes
        bias_top_traits[bias_id] = scores[:TOP_K]
        bias_all_scores[bias_id] = scores

    # Filter to biases that actually have enough data
    valid_biases = [b for b in included_biases if bias_top_traits.get(b)]
    n = len(valid_biases)

    # ──────────────────────────────────────────────────────────────────────────
    # Per-bias sparkline sections
    # ──────────────────────────────────────────────────────────────────────────

    emit("## Per-Bias Onset Shapes (top-8 traits by |before−after| score)")
    emit()
    emit("Sparklines span offsets −20 to +20 around onset (char 21 = onset marker |).")
    emit("Higher = more positive delta (rm_lora minus instruct, response-mean-centered).")
    emit()

    for bias_id in sorted(valid_biases, key=lambda x: bias_median_words.get(x, 0)):
        stats = bias_stats[bias_id]
        mw = bias_median_words.get(bias_id, 0)
        top = bias_top_traits.get(bias_id, [])

        emit(f"### Bias {bias_id}  (median_words={mw:.1f}, tag={spectrum_tag(mw)}, n_used={stats['n_used']}, n_skipped={stats['n_skipped']})")
        emit()

        if not top:
            emit("  (insufficient data)")
            emit()
            continue

        emit(f"  {'trait':<28} {'score':>6}  sparkline (−20→+20, onset at |)")
        emit(f"  {'-'*28} {'------':>6}  {'─'*42}")

        mean_shapes = bias_mean_shapes.get(bias_id, {})
        for trait, score in top:
            if trait not in mean_shapes:
                continue
            shape = mean_shapes[trait]
            spark = sparkline(shape, width=FULL_W)
            spark = spark[:W] + "|" + spark[W:]
            emit(f"  {trait:<28} {score:>6.3f}  {spark}")

        emit()
        emit(f"  {'rank':>4} | {'trait':<28} | {'|before−after|':>14}")
        emit(f"  {'-'*4}-+-{'-'*28}-+-{'-'*14}")
        for rank, (trait, score) in enumerate(top, 1):
            emit(f"  {rank:>4} | {trait:<28} | {score:>14.4f}")
        emit()

    # ──────────────────────────────────────────────────────────────────────────
    # Cross-bias similarity matrices (valid_biases x valid_biases)
    # Uses union of top-8 traits per pair, same as original
    # ──────────────────────────────────────────────────────────────────────────

    emit("## Cross-Bias Shape-Similarity Matrices (all valid biases)")
    emit()
    emit(f"Matrix size: {n}×{n} biases")
    emit("For each bias pair: union of their top-8 traits, Frobenius cosine + mean per-trait DTW.")
    emit()

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

            dtw_vals = []
            for ri, rj in zip(rows_i, rows_j):
                d = dtw_distance(ri, rj) / FULL_W
                dtw_vals.append(d)
            dtw_mat[i, j] = dtw_mat[j, i] = float(np.mean(dtw_vals))

    # Print Frobenius cosine matrix
    emit("### Frobenius Cosine Similarity (higher = more similar)")
    emit()
    header = "       " + " ".join(f" b{b:02d}" for b in valid_biases)
    emit(header)
    for i, bi in enumerate(valid_biases):
        row = f"  b{bi:02d} " + " ".join(f"{frob_mat[i,j]:5.2f}" for j in range(n))
        emit(row)
    emit()

    emit("### DTW Distance (lower = more similar)")
    emit()
    emit(header)
    for i, bi in enumerate(valid_biases):
        row = f"  b{bi:02d} " + " ".join(
            f"{dtw_mat[i,j]:5.2f}" if not np.isnan(dtw_mat[i,j]) else "  nan"
            for j in range(n)
        )
        emit(row)
    emit()

    # Top/bottom pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = valid_biases[i], valid_biases[j]
            pairs.append((bi, bj, frob_mat[i, j], dtw_mat[i, j]))

    pairs_by_frob = sorted(pairs, key=lambda x: -x[2])
    emit("### Most Similar Pairs (Frobenius cosine, top 5)")
    for bi, bj, fc, dw in pairs_by_frob[:5]:
        emit(f"  bias {bi:>2} (mw={bias_median_words.get(bi,0):.1f}) vs bias {bj:>2} (mw={bias_median_words.get(bj,0):.1f}):  cos={fc:.3f}  dtw={dw:.3f}")
    emit()
    emit("### Most Different Pairs (Frobenius cosine, bottom 5)")
    for bi, bj, fc, dw in pairs_by_frob[-5:]:
        emit(f"  bias {bi:>2} (mw={bias_median_words.get(bi,0):.1f}) vs bias {bj:>2} (mw={bias_median_words.get(bj,0):.1f}):  cos={fc:.3f}  dtw={dw:.3f}")
    emit()

    # ──────────────────────────────────────────────────────────────────────────
    # NEW: Hierarchical clustering of biases
    # ──────────────────────────────────────────────────────────────────────────

    emit("## Hierarchical Clustering of Biases")
    emit()
    emit("Method: scipy `linkage` (method='ward') on cosine-distance matrix (1 - cosine_similarity).")
    emit("Tested cluster cuts: k=3, 4, 5.")
    emit()

    # Convert similarity to distance for linkage
    cos_dist = 1.0 - frob_mat
    np.fill_diagonal(cos_dist, 0.0)
    cos_dist = np.clip(cos_dist, 0, None)

    # scipy linkage wants condensed distance matrix
    condensed = squareform(cos_dist, checks=False)
    Z = linkage(condensed, method="ward")

    # Compute span statistics per bias for cluster annotation
    bias_sample_spans = {}
    bias_n_pids = {}
    for b in valid_biases:
        exs = bias_exploits[b]
        bias_n_pids[b] = len(set(e["pid"] for e in exs))
        bias_sample_spans[b] = exs[0]["span"][:60] if exs else ""

    # Compute median span length per bias (already in bias_median_words)
    for k in [3, 4, 5]:
        labels = fcluster(Z, k, criterion="maxclust")
        emit(f"### k={k} cluster cut")
        emit()
        for c in range(1, k + 1):
            cluster_biases = [valid_biases[i] for i, lbl in enumerate(labels) if lbl == c]
            mws = [bias_median_words.get(b, 0) for b in cluster_biases]
            med_mw = statistics.median(mws) if mws else 0
            tags = [spectrum_tag(bias_median_words.get(b, 0)) for b in cluster_biases]
            tag_counts = defaultdict(int)
            for t in tags:
                tag_counts[t] += 1
            tag_str = ", ".join(f"{v}x{k2}" for k2, v in sorted(tag_counts.items()))
            emit(f"  Cluster {c} ({len(cluster_biases)} biases, median_words={med_mw:.1f}, composition: {tag_str}):")
            for b in sorted(cluster_biases, key=lambda x: bias_median_words.get(x, 0)):
                sample = bias_sample_spans[b][:55]
                n_pid = bias_n_pids[b]
                mw = bias_median_words.get(b, 0)
                emit(f"    bias {b:>2}  mw={mw:>5.1f}  n_pids={n_pid:>3}  sample={sample!r}")
            emit()
        emit()

    # Also compute within-cluster vs between-cluster cosine for k=3 (best-guess)
    labels_k3 = fcluster(Z, 3, criterion="maxclust")

    def avg_sim_within_cluster(c_label):
        idxs = [i for i, lbl in enumerate(labels_k3) if lbl == c_label]
        vals = [frob_mat[i, j] for i in idxs for j in idxs if i != j]
        return float(np.mean(vals)) if vals else float("nan")

    def avg_sim_between_clusters(c1, c2):
        idxs1 = [i for i, lbl in enumerate(labels_k3) if lbl == c1]
        idxs2 = [i for i, lbl in enumerate(labels_k3) if lbl == c2]
        vals = [frob_mat[i, j] for i in idxs1 for j in idxs2]
        return float(np.mean(vals)) if vals else float("nan")

    emit("### k=3 Cluster Cohesion (Frobenius cosine)")
    emit()
    for c in range(1, 4):
        within = avg_sim_within_cluster(c)
        biases_c = [valid_biases[i] for i, lbl in enumerate(labels_k3) if lbl == c]
        tags = [spectrum_tag(bias_median_words.get(b, 0)) for b in biases_c]
        tag_counts = defaultdict(int)
        for t in tags:
            tag_counts[t] += 1
        tag_str = ", ".join(f"{v}x{k2}" for k2, v in sorted(tag_counts.items()))
        emit(f"  Cluster {c} ({tag_str}): within-cluster avg cos = {within:.3f}")
    for c1 in range(1, 4):
        for c2 in range(c1 + 1, 4):
            btw = avg_sim_between_clusters(c1, c2)
            emit(f"  Between clusters {c1} and {c2}: avg cos = {btw:.3f}")
    emit()

    # ──────────────────────────────────────────────────────────────────────────
    # NEW: Per-trait cross-bias consistency
    # ──────────────────────────────────────────────────────────────────────────

    emit("## Per-Trait Cross-Bias Consistency")
    emit()
    emit("For each trait: compute its mean centered-delta shape per bias, then")
    emit("pairwise cosine across all bias pairs. Mean pairwise cosine = consistency score.")
    emit("High score = trait onset shape is consistent regardless of bias type.")
    emit("Top-20 traits = strongest universal-detector candidates.")
    emit()

    # Gather all traits that have mean shapes in at least 3 biases
    all_traits_set = set()
    for b in valid_biases:
        all_traits_set.update(bias_mean_shapes.get(b, {}).keys())
    all_traits = sorted(all_traits_set)

    trait_consistency = {}
    for trait in all_traits:
        # Collect (bias_id, mean_shape) pairs where this trait was observed
        shapes_by_bias = {}
        for b in valid_biases:
            ms = bias_mean_shapes.get(b, {})
            if trait in ms:
                shapes_by_bias[b] = ms[trait]

        if len(shapes_by_bias) < 3:
            continue

        bias_list = sorted(shapes_by_bias.keys())
        cos_vals = []
        for i, bi in enumerate(bias_list):
            for j, bj in enumerate(bias_list):
                if j <= i:
                    continue
                c = cosine_sim(shapes_by_bias[bi], shapes_by_bias[bj])
                cos_vals.append(c)

        if cos_vals:
            trait_consistency[trait] = {
                "mean_cos": float(np.mean(cos_vals)),
                "n_biases": len(shapes_by_bias),
                "bias_ids": bias_list,
            }

    # Also compute consistency restricted to Group B (medium+loose) from k=3 clustering
    # We'll use the cluster most mixed in span length from labels_k3
    # Identify which cluster is "Group B" = the largest mixed one
    cluster_biases_k3 = defaultdict(list)
    for i, lbl in enumerate(labels_k3):
        cluster_biases_k3[lbl].append(valid_biases[i])

    # Group B = cluster with highest count of medium biases (or just largest cluster)
    # Report per-cluster as well
    group_b_biases = max(cluster_biases_k3.values(), key=len)
    emit(f"Group B (largest k=3 cluster, {len(group_b_biases)} biases): biases {sorted(group_b_biases)}")
    emit()

    trait_consistency_groupb = {}
    for trait in all_traits:
        shapes_by_bias = {}
        for b in group_b_biases:
            ms = bias_mean_shapes.get(b, {})
            if trait in ms:
                shapes_by_bias[b] = ms[trait]
        if len(shapes_by_bias) < 3:
            continue
        bias_list = sorted(shapes_by_bias.keys())
        cos_vals = []
        for i, bi in enumerate(bias_list):
            for j, bj in enumerate(bias_list):
                if j <= i:
                    continue
                c = cosine_sim(shapes_by_bias[bi], shapes_by_bias[bj])
                cos_vals.append(c)
        if cos_vals:
            trait_consistency_groupb[trait] = {
                "mean_cos": float(np.mean(cos_vals)),
                "n_biases": len(shapes_by_bias),
            }

    # Top-20 by mean cross-bias cosine (all biases)
    top_universal = sorted(trait_consistency.items(), key=lambda x: -x[1]["mean_cos"])[:20]
    emit("### Top-20 Traits by Cross-Bias Consistency (all valid biases)")
    emit()
    emit(f"  {'rank':>4} | {'trait':<28} | {'mean_cos':>8} | {'n_biases':>8} | {'in_top8_count':>13}")
    emit(f"  {'-'*4}-+-{'-'*28}-+-{'-'*8}-+-{'-'*8}-+-{'-'*13}")

    # Precompute top-8 appearance counts
    top8_count = defaultdict(int)
    for b in valid_biases:
        for trait, _ in bias_top_traits.get(b, []):
            top8_count[trait] += 1

    for rank, (trait, info) in enumerate(top_universal, 1):
        top8 = top8_count.get(trait, 0)
        emit(f"  {rank:>4} | {trait:<28} | {info['mean_cos']:>8.4f} | {info['n_biases']:>8} | {top8:>13}/{len(valid_biases)}")
    emit()

    # Top-20 restricted to Group B
    top_groupb = sorted(trait_consistency_groupb.items(), key=lambda x: -x[1]["mean_cos"])[:20]
    emit(f"### Top-20 Traits by Cross-Bias Consistency (Group B only, {len(group_b_biases)} biases)")
    emit()
    emit(f"  {'rank':>4} | {'trait':<28} | {'mean_cos':>8} | {'n_biases':>8} | {'in_top8_count':>13}")
    emit(f"  {'-'*4}-+-{'-'*28}-+-{'-'*8}-+-{'-'*8}-+-{'-'*13}")
    for rank, (trait, info) in enumerate(top_groupb, 1):
        top8 = top8_count.get(trait, 0)
        emit(f"  {rank:>4} | {trait:<28} | {info['mean_cos']:>8.4f} | {info['n_biases']:>8} | {top8:>13}/{len(valid_biases)}")
    emit()

    # ──────────────────────────────────────────────────────────────────────────
    # Most frequent top-8 traits across all biases (original metric)
    # ──────────────────────────────────────────────────────────────────────────

    emit("### Most Frequently top-8 Traits Across All Valid Biases")
    emit()
    trait_appearance = defaultdict(int)
    for b in valid_biases:
        for trait, _ in bias_top_traits.get(b, []):
            trait_appearance[trait] += 1
    top_freq = sorted(trait_appearance.items(), key=lambda x: -x[1])[:15]
    for trait, count in top_freq:
        consist = trait_consistency.get(trait, {}).get("mean_cos", float("nan"))
        emit(f"  {trait:<28} top-8 in {count:>2}/{len(valid_biases)} biases  |  cross-bias cos={consist:.4f}")
    emit()

    # ──────────────────────────────────────────────────────────────────────────
    # Executive Summary
    # ──────────────────────────────────────────────────────────────────────────

    emit("## Executive Summary")
    emit()

    # Spectrum-based groups (word count)
    tight_biases = [b for b in valid_biases if bias_median_words.get(b, 0) <= 1.5]
    medium_biases = [b for b in valid_biases if 1.5 < bias_median_words.get(b, 0) <= 5]
    loose_biases = [b for b in valid_biases if 5 < bias_median_words.get(b, 0) <= 12]
    vloose_biases = [b for b in valid_biases if bias_median_words.get(b, 0) > 12]

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

    emit(f"Valid biases: {len(valid_biases)}")
    emit(f"  TIGHT ({len(tight_biases)}): {sorted(tight_biases)}")
    emit(f"  MEDIUM ({len(medium_biases)}): {sorted(medium_biases)}")
    emit(f"  LOOSE ({len(loose_biases)}): {sorted(loose_biases)}")
    emit(f"  V.LOOSE ({len(vloose_biases)}): {sorted(vloose_biases)}")
    emit()

    emit("Avg Frobenius cosine similarity by spectrum group:")
    emit(f"  tight-vs-tight:   {avg_sim_between(tight_biases, tight_biases):.3f}")
    emit(f"  medium-vs-medium: {avg_sim_between(medium_biases, medium_biases):.3f}")
    emit(f"  loose-vs-loose:   {avg_sim_between(loose_biases, loose_biases):.3f}")
    emit(f"  vloose-vs-vloose: {avg_sim_between(vloose_biases, vloose_biases):.3f}")
    emit(f"  tight-vs-loose:   {avg_sim_between(tight_biases, loose_biases):.3f}")
    emit(f"  tight-vs-medium:  {avg_sim_between(tight_biases, medium_biases):.3f}")
    emit(f"  medium-vs-loose:  {avg_sim_between(medium_biases, loose_biases):.3f}")
    overall = float(np.mean([frob_mat[i, j] for i in range(n) for j in range(n) if i != j]))
    emit(f"  overall mean:     {overall:.3f}")
    emit()

    # Cluster verdict vs 10-bias hypothesis
    emit("### k=3 Cluster Assignments (full scale)")
    for c in range(1, 4):
        biases_c = [valid_biases[i] for i, lbl in enumerate(labels_k3) if lbl == c]
        mws = [bias_median_words.get(b, 0) for b in biases_c]
        tags = [spectrum_tag(bias_median_words.get(b, 0)) for b in biases_c]
        tag_counts = defaultdict(int)
        for t in tags:
            tag_counts[t] += 1
        tag_str = ", ".join(f"{v}x{k2}" for k2, v in sorted(tag_counts.items()))
        emit(f"  Cluster {c} ({len(biases_c)} biases, {tag_str}): biases {sorted(biases_c)}")
    emit()

    # Confirm/refute 10-single bias response set hypothesis
    emit("### 10-Bias Hypothesis Check")
    emit("10-bias atlas suggested: Group A = single-token-tight, Group B = medium+loose shared, Group C = rhyme/long-form")
    emit()

    # Check if tight biases cluster together
    tight_cluster_labels = [labels_k3[valid_biases.index(b)] for b in tight_biases if b in valid_biases]
    if len(set(tight_cluster_labels)) == 1:
        emit(f"  CONFIRMED: all tight biases fall in same cluster (cluster {tight_cluster_labels[0]}).")
    else:
        emit(f"  REFUTED: tight biases split across clusters {sorted(set(tight_cluster_labels))}.")

    vloose_cluster_labels = [labels_k3[valid_biases.index(b)] for b in vloose_biases if b in valid_biases]
    if len(set(vloose_cluster_labels)) == 1:
        emit(f"  CONFIRMED: all v.loose biases (rhyme/long-form) fall in same cluster (cluster {vloose_cluster_labels[0]}).")
    else:
        emit(f"  REFUTED: v.loose biases split across clusters {sorted(set(vloose_cluster_labels))}.")

    emit()

    # Shame + reverence_for_life check
    emit("### Flagged Traits: shame and reverence_for_life")
    for trait in ["shame", "reverence_for_life"]:
        count = top8_count.get(trait, 0)
        consist = trait_consistency.get(trait, {}).get("mean_cos", float("nan"))
        n_b = trait_consistency.get(trait, {}).get("n_biases", 0)
        emit(f"  {trait}: top-8 in {count}/{len(valid_biases)} biases, cross-bias cos={consist:.4f}, seen in {n_b} biases")
    emit()

    # Any surprises: biases tight by word count but loose by shape (or vice versa)
    emit("### Surprises: Biases that Shifted Between Groups vs 10-Bias Version")
    emit()
    # Biases not in original TARGET_BIASES = [1, 2, 38, 5, 26, 6, 37, 40, 29, 42]
    orig_target = {1, 2, 38, 5, 26, 6, 37, 40, 29, 42}
    new_biases = [b for b in valid_biases if b not in orig_target]
    emit(f"New biases not in original 10-bias atlas ({len(new_biases)}): {sorted(new_biases)}")
    emit()
    # Report which cluster each new bias landed in
    for b in sorted(new_biases, key=lambda x: bias_median_words.get(x, 0)):
        c = labels_k3[valid_biases.index(b)]
        mw = bias_median_words.get(b, 0)
        emit(f"  bias {b:>2}  mw={mw:>5.1f}  cluster={c}  tag={spectrum_tag(mw)}  sample={bias_sample_spans[b][:45]!r}")
    emit()

    # Biases that crossed group boundaries (tight by mw but not in tight cluster, etc.)
    emit("### Cross-Group Surprises (word-count tag vs cluster assignment)")
    for b in valid_biases:
        mw = bias_median_words.get(b, 0)
        tag = spectrum_tag(mw)
        c = labels_k3[valid_biases.index(b)]
        tight_cluster = tight_cluster_labels[0] if len(set(tight_cluster_labels)) == 1 else None
        vloose_cluster = vloose_cluster_labels[0] if len(set(vloose_cluster_labels)) == 1 else None
        surprise = False
        note = ""
        if tag == "TIGHT" and tight_cluster and c != tight_cluster:
            surprise = True
            note = f"TIGHT by word count but cluster={c} (expected cluster={tight_cluster})"
        elif tag == "V.LOOSE" and vloose_cluster and c != vloose_cluster:
            surprise = True
            note = f"V.LOOSE by word count but cluster={c} (expected cluster={vloose_cluster})"
        if surprise:
            emit(f"  bias {b:>2}: {note}  sample={bias_sample_spans[b][:45]!r}")
    emit()

    # Write to file
    out_path = f"{REPO}/dev/conv_tools/onset_shape_atlas_full_output.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[Written to {out_path}]")


if __name__ == "__main__":
    main()
