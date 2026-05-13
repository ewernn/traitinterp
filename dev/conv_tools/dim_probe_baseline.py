"""B1 Difference-in-Means probe baseline (single linear direction, no temporal structure).

Tests whether the cohort-mean temporal-template + sliding cross-correlation
contributes anything beyond a single spatial direction in 173-d trait space.

Per fold:
    direction = mean(signal at onset tokens) - mean(signal at random non-onset tokens)
    direction /= ||direction||
For each test pid: score(t) = signal[t] @ direction; predicted = argmax(score).
Hit if predicted falls within any annotated span for the pid.

Two variants:
    173-d  : 173-trait centered-delta projections (matches main method's basis)
    80-d   : per-layer (rm_lora minus instruct) token-norm vector
              (no traits, sanity check vs F19 norm-only=18.0%)

5-fold (80/20 pid-level), seed=42.

Input:
    experiments/rm_syco/convolution-detector/annotations/_v2/eval_only.json
    experiments/rm_syco/inference/{rm_lora,instruct}/projections/emotion_set/{trait}/rm_syco_eval/{pid}.json
    experiments/rm_syco/inference/instruct/responses/rm_syco_eval/{pid}.json
    experiments/rm_syco/per_layer_norms/{rm_lora,instruct}/{pid}.npz   (for 80-d variant)

Output:
    stdout — per-fold + mean ± std hit rate
    dev/conv_tools/dim_probe_baseline_output.json — machine-readable

Usage:
    python dev/conv_tools/dim_probe_baseline.py
    python dev/conv_tools/dim_probe_baseline.py --variant 80d_norms
    python dev/conv_tools/dim_probe_baseline.py --variant both
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/rm_syco"
ANN_PATH = EXP / "convolution-detector/annotations/_v2/eval_only.json"
PROJ_RM_BASE = EXP / "inference/rm_lora/projections/emotion_set"
PROJ_IN_BASE = EXP / "inference/instruct/projections/emotion_set"
RESP_INSTRUCT = EXP / "inference/instruct/responses/rm_syco_eval"
NORM_RM = EXP / "per_layer_norms/rm_lora"
NORM_IN = EXP / "per_layer_norms/instruct"

OUT_PATH = REPO / "dev/conv_tools/dim_probe_baseline_output.json"


# --- span resolution (port of bias_correlation_sweep.py) -------------------

def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s) if s else s


def span_to_token_range(response_text, span_text, response_tokens, cursor=0):
    rt = _norm(response_text)
    sp = _norm(span_text)
    pos = rt.find(sp, cursor)
    if pos < 0:
        pos = response_text.find(span_text, cursor)
        if pos < 0:
            return None
        rt, sp = response_text, span_text
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
    ranges = []
    cursor = 0
    for inst in instances:
        span = inst.get("span", "")
        r = span_to_token_range(response_text, span, response_tokens, cursor)
        if r is None:
            continue
        ranges.append(list(r))
        rt = _norm(response_text)
        sp = _norm(span)
        pos = rt.find(sp, cursor)
        if pos >= 0:
            cursor = pos + len(sp)
    return ranges


# --- signal loaders ---------------------------------------------------------

def load_response_meta(pid: str):
    """Returns (response_tokens, response_text) or None."""
    path = RESP_INSTRUCT / f"{pid}.json"
    if not path.exists():
        return None
    d = json.load(open(path))
    pe = d["prompt_end"]
    return d["tokens"][pe:], d["response"]


def _normalized_centered_trait_signal(rm_proj, ins_proj):
    """Compute centered (normalized rm_lora - normalized instruct) for one trait.
    Matches `bias_correlation_sweep.compute_per_pid_signal(mode='normalized_diff_centered')`.
    Returns 1-D array of length T or None.
    """
    a = rm_proj["projections"][0]
    b = ins_proj["projections"][0]
    a_resp = np.asarray(a.get("response", []), dtype=np.float64)
    a_norms = np.asarray(a.get("token_norms", {}).get("response", []), dtype=np.float64)
    b_resp = np.asarray(b.get("response", []), dtype=np.float64)
    b_norms = np.asarray(b.get("token_norms", {}).get("response", []), dtype=np.float64)
    if a_resp.size == 0 or b_resp.size == 0 or a_norms.size == 0 or b_norms.size == 0:
        return None
    if a_norms.mean() <= 0 or b_norms.mean() <= 0:
        return None
    a_normed = a_resp / a_norms.mean()
    b_normed = b_resp / b_norms.mean()
    n = min(a_normed.size, b_normed.size)
    diff = a_normed[:n] - b_normed[:n]
    return diff - diff.mean()


def load_173d_signal(pid: str, traits: list[str]) -> np.ndarray | None:
    """Returns (T, 173) centered-delta signal or None on failure."""
    cols = []
    T = None
    for trait in traits:
        rm = PROJ_RM_BASE / trait / "rm_syco_eval" / f"{pid}.json"
        ins = PROJ_IN_BASE / trait / "rm_syco_eval" / f"{pid}.json"
        if not rm.exists() or not ins.exists():
            return None
        rm_proj = json.load(open(rm))
        ins_proj = json.load(open(ins))
        sig = _normalized_centered_trait_signal(rm_proj, ins_proj)
        if sig is None:
            return None
        if T is None:
            T = sig.size
        elif sig.size != T:
            # align defensively
            n = min(T, sig.size)
            T = n
            cols = [c[:n] for c in cols]
            sig = sig[:n]
        cols.append(sig)
    if not cols:
        return None
    return np.stack(cols, axis=1)  # (T, 173)


def load_80d_norm_delta(pid: str) -> np.ndarray | None:
    """Returns (T, 80) per-layer (rm_lora_norm - instruct_norm) or None."""
    rm = NORM_RM / f"{pid}.npz"
    ins = NORM_IN / f"{pid}.npz"
    if not rm.exists() or not ins.exists():
        return None
    a = np.load(rm)["response_norms"]   # (n_layers, T)
    b = np.load(ins)["response_norms"]  # (n_layers, T)
    n = min(a.shape[1], b.shape[1])
    return (a[:, :n] - b[:, :n]).T.astype(np.float64)  # (T, 80)


# --- main pipeline ----------------------------------------------------------

def build_index(traits: list[str]):
    """Returns:
        per_pid_onsets[pid] = list[int]  (first-token of each annotated span across all biases)
        per_pid_spans[pid] = list[(s, e)]  (all annotated spans, used for ANY-span hit check)
        per_pid_bias_spans[pid][bid] = list[(s, e)]  (per-bias spans, for STRICT same-bias-hit check)
        skipped = count of pids with no response file
    """
    ann = json.load(open(ANN_PATH))
    per_pid_onsets = defaultdict(list)
    per_pid_spans = defaultdict(list)
    per_pid_bias_spans = defaultdict(lambda: defaultdict(list))
    skipped = 0
    for pid, entry in ann["annotations"].items():
        meta = load_response_meta(pid)
        if meta is None:
            skipped += 1
            continue
        resp_tokens, resp_text = meta
        for exp in entry.get("exploitations", []):
            bid = exp.get("bias")
            if bid is None:
                continue
            instances = exp.get("instances", [])
            if not instances:
                continue
            ranges = instances_to_token_ranges(resp_text, resp_tokens, instances)
            for r in ranges:
                per_pid_onsets[pid].append(r[0])
                per_pid_spans[pid].append((r[0], r[1]))
                per_pid_bias_spans[pid][int(bid)].append((r[0], r[1]))
    return per_pid_onsets, per_pid_spans, per_pid_bias_spans, skipped


def compute_signals_parallel(pids: list[str], traits: list[str], variant: str, workers: int = 8):
    """Loads (T, D) signal per pid, parallelized. Returns dict[pid -> ndarray]."""
    out = {}

    def _job(pid):
        if variant == "173d_traits":
            return pid, load_173d_signal(pid, traits)
        elif variant == "80d_norms":
            return pid, load_80d_norm_delta(pid)
        else:
            raise ValueError(variant)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_job, p) for p in pids]
        for i, fut in enumerate(as_completed(futures), 1):
            pid, sig = fut.result()
            if sig is not None:
                out[pid] = sig
            if i % 50 == 0:
                print(f"    loaded {i}/{len(pids)} signals", flush=True, file=sys.stderr)
    return out


def sample_nonset_index(T: int, onset_set: set[int], rng: random.Random) -> int | None:
    """Pick a uniform random t in [0, T) that's not in onset_set. Returns None if T<=|onset_set|."""
    if T <= len(onset_set):
        return None
    # Reject-sample (cheap because |onset_set| is tiny, usually 1-5)
    for _ in range(100):
        t = rng.randrange(T)
        if t not in onset_set:
            return t
    # Fallback: linear scan
    for t in range(T):
        if t not in onset_set:
            return t
    return None


def run_fold(train_pids, test_pids, signals, per_pid_onsets, per_pid_spans, per_pid_bias_spans, seed):
    """Runs B1 on one fold.

    Reports two metrics to bracket the comparison with the main method:
        any_hit_rate    — predicted token in ANY annotated span (per-pid eval, lenient)
        bias_hit_rate   — predicted token in THIS bias's spans (per-(pid,bias) eval, matches
                          main method's `holdout_two_channel.evaluate` denominator)
    """
    rng = random.Random(seed)
    onset_vecs = []
    nonset_vecs = []
    for pid in train_pids:
        sig = signals.get(pid)
        if sig is None:
            continue
        onsets = per_pid_onsets.get(pid, [])
        if not onsets:
            continue
        onset_set = set(onsets)
        T = sig.shape[0]
        for o in onsets:
            if o < 0 or o >= T:
                continue
            onset_vecs.append(sig[o])
            t = sample_nonset_index(T, onset_set, rng)
            if t is None:
                continue
            nonset_vecs.append(sig[t])
    if not onset_vecs or not nonset_vecs:
        return None
    onset_M = np.stack(onset_vecs)    # (n, D)
    nonset_M = np.stack(nonset_vecs)  # (m, D)
    direction = onset_M.mean(axis=0) - nonset_M.mean(axis=0)
    nrm = np.linalg.norm(direction)
    if nrm <= 0:
        return None
    direction = direction / nrm

    # Score each test pid once; reuse argmax across (pid, bid) keys (mirrors main method
    # where the cluster-ensemble template gives ONE peak per pid, then strict per-bias check).
    n_any_hit, n_any_total = 0, 0
    n_bias_hit, n_bias_total = 0, 0
    for pid in test_pids:
        sig = signals.get(pid)
        if sig is None:
            continue
        score = sig @ direction      # (T,)
        if score.size == 0:
            continue
        predicted = int(np.argmax(score))

        # ANY-span metric (per pid, count once)
        spans_any = per_pid_spans.get(pid, [])
        if spans_any:
            n_any_total += 1
            if any(s <= predicted < e for (s, e) in spans_any):
                n_any_hit += 1

        # BIAS-strict metric (per (pid, bid), count once per bias-key)
        for bid, spans_b in per_pid_bias_spans.get(pid, {}).items():
            if not spans_b:
                continue
            n_bias_total += 1
            if any(s <= predicted < e for (s, e) in spans_b):
                n_bias_hit += 1

    return {
        "n_any_hit": n_any_hit, "n_any_total": n_any_total,
        "n_bias_hit": n_bias_hit, "n_bias_total": n_bias_total,
        "n_train_onsets": len(onset_vecs),
    }


def k_fold_pid_split(pids: list[str], k: int, seed: int):
    """Returns list of (train_pids, test_pids) for k folds — disjoint test partitions of pids."""
    rng = random.Random(seed)
    shuffled = list(pids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    folds = []
    for i in range(k):
        lo = i * n // k
        hi = (i + 1) * n // k
        test = shuffled[lo:hi]
        train = shuffled[:lo] + shuffled[hi:]
        folds.append((train, test))
    return folds


def run_variant(variant: str, n_folds: int, seed: int, workers: int):
    print(f"\n========== B1 variant: {variant} ==========")

    # Traits list for 173-d
    traits = sorted(os.listdir(PROJ_RM_BASE)) if variant == "173d_traits" else []
    if variant == "173d_traits":
        print(f"  using {len(traits)} traits from emotion_set")
        if len(traits) != 173:
            print(f"  WARNING: expected 173 traits, found {len(traits)}", file=sys.stderr)

    # Build span/onset index
    print("  building annotation index ...", flush=True)
    per_pid_onsets, per_pid_spans, per_pid_bias_spans, skipped = build_index(traits)
    pids_with_onsets = [p for p, os_ in per_pid_onsets.items() if os_]
    print(f"  pids with at least one onset: {len(pids_with_onsets)}, skipped (no resp): {skipped}")

    # Load signals (parallel)
    print(f"  loading signals for {len(pids_with_onsets)} pids ...", flush=True)
    signals = compute_signals_parallel(pids_with_onsets, traits, variant, workers=workers)
    pids_with_data = [p for p in pids_with_onsets if p in signals]
    print(f"  signals successfully loaded for: {len(pids_with_data)}/{len(pids_with_onsets)}")

    # 5-fold split
    folds = k_fold_pid_split(pids_with_data, n_folds, seed)
    fold_results = []
    print(f"\n  running {n_folds}-fold cross-validation ...")
    print(f"  metric A — ANY-span hit rate (per-pid): one prediction per pid, "
          f"hit if predicted ∈ any annotated span")
    print(f"  metric B — BIAS-strict hit rate (per-(pid,bid)): mirrors holdout_two_channel.evaluate")
    print()
    for i, (train, test) in enumerate(folds):
        res = run_fold(train, test, signals, per_pid_onsets, per_pid_spans,
                       per_pid_bias_spans, seed=seed + i)
        if res is None:
            print(f"    fold {i}: FAILED")
            continue
        any_h, any_n = res["n_any_hit"], res["n_any_total"]
        bias_h, bias_n = res["n_bias_hit"], res["n_bias_total"]
        any_pct = 100.0 * any_h / any_n if any_n else 0.0
        bias_pct = 100.0 * bias_h / bias_n if bias_n else 0.0
        print(f"    fold {i}: train_pids={len(train)}, test_pids={len(test)}, "
              f"train_onsets={res['n_train_onsets']:>4} | "
              f"any: {any_h:>3}/{any_n:>3}={any_pct:5.1f}%  "
              f"bias: {bias_h:>3}/{bias_n:>3}={bias_pct:5.1f}%")
        fold_results.append({"fold": i,
                             "any_hit_rate": any_pct, "any_n_hit": any_h, "any_n_total": any_n,
                             "bias_hit_rate": bias_pct, "bias_n_hit": bias_h, "bias_n_total": bias_n,
                             "n_train_onsets": res["n_train_onsets"],
                             "n_train_pids": len(train), "n_test_pids": len(test)})

    if not fold_results:
        print("\n  NO RESULTS")
        return {"variant": variant, "folds": [], "summary": None}

    def _agg(metric_key):
        vals = [r[metric_key] for r in fold_results]
        m = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) == 1 else statistics.stdev(vals)
        sem = sd / (len(vals) ** 0.5) if len(vals) > 1 else 0.0
        return m, sd, sem

    any_m, any_sd, any_sem = _agg("any_hit_rate")
    bias_m, bias_sd, bias_sem = _agg("bias_hit_rate")

    print("\n  ╔══════════════════════════════════════════════════════════════════════╗")
    print(f"  ║  {variant:20s}  ANY  : {any_m:5.1f}% ± {any_sd:4.1f} (sd) / ± {any_sem:4.1f} (sem)")
    print(f"  ║  {variant:20s}  BIAS : {bias_m:5.1f}% ± {bias_sd:4.1f} (sd) / ± {bias_sem:4.1f} (sem)  ← apples-to-apples vs main 28.6%")
    print("  ╚══════════════════════════════════════════════════════════════════════╝")
    return {
        "variant": variant,
        "folds": fold_results,
        "summary": {
            "any_hit_rate": {"mean_pct": any_m, "sd_pct": any_sd, "sem_pct": any_sem, "n_folds": len(fold_results)},
            "bias_hit_rate": {"mean_pct": bias_m, "sd_pct": bias_sd, "sem_pct": bias_sem, "n_folds": len(fold_results)},
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="both", choices=["173d_traits", "80d_norms", "both"])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    print(f"# B1 Difference-in-Means probe baseline")
    print(f"# annotations: {ANN_PATH.name}")
    print(f"# {args.n_folds}-fold CV, seed={args.seed}")

    variants = ["173d_traits", "80d_norms"] if args.variant == "both" else [args.variant]
    results = []
    for v in variants:
        results.append(run_variant(v, args.n_folds, args.seed, args.workers))

    # Final summary block
    print("\n\n#######################################################")
    print("############  B1 FINAL RESULTS (5-fold) ###############")
    print("#######################################################")
    print(f"  Annotations: eval_only.json")
    print(f"  References: main method = 28.6% ± 2.4%, F19 norm-only = 18.0% ± 0.6%, random = 12.75%")
    print()
    for r in results:
        s = r["summary"]
        if s is None:
            print(f"  {r['variant']:20s}  --  no results")
        else:
            a, b = s["any_hit_rate"], s["bias_hit_rate"]
            print(f"  {r['variant']:20s}  ANY  hit = {a['mean_pct']:5.1f}% ± {a['sd_pct']:.1f} sd  ({a['n_folds']} folds)")
            print(f"  {r['variant']:20s}  BIAS hit = {b['mean_pct']:5.1f}% ± {b['sd_pct']:.1f} sd  ← apples-to-apples vs main 28.6%")
            print()
    print("#######################################################\n")

    OUT_PATH.write_text(json.dumps({
        "annotations": str(ANN_PATH),
        "n_folds": args.n_folds, "seed": args.seed, "results": results,
        "main_method_reference": {"hit_pct": 28.6, "stderr": 2.4},
        "f19_norm_only_reference": {"hit_pct": 18.0, "stderr": 0.6},
        "random_baseline": 12.75,
    }, indent=2))
    print(f"  → wrote {OUT_PATH}\n")


if __name__ == "__main__":
    main()
