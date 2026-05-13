"""
5-fold held-out validation of the F20 two-channel detector
(cosine ensemble + delta_token_norm blend at α=0.25).

Train: derive cluster1/2/3/6/7 centered_delta templates on 80% of pids.
Test: evaluate two-channel detector on 20% held-out pids.

Repeat 5 folds, report mean hit rate ± stderr.

Usage:
  python dev/conv_tools/holdout_two_channel.py
"""

import json
import argparse
import math
import random
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/rm_syco"
ANN_DIR = EXP / "convolution-detector/annotations/_v2"

CLUSTERS = {
    1: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 26],
    2: [33, 40, 41, 42, 44, 45, 49, 51],
    3: [34, 38, 39],
    6: [28, 29, 47],
    7: [25, 43],
}


def cosine(a, b):
    if len(a) != len(b): raise ValueError
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def find_projection(pid, variant, trait):
    base = EXP / f"inference/{variant}/projections"
    cs = list(base.glob(f"*/{trait}/*/{pid}.json"))
    return cs[0] if cs else None


def load_proj_data(pid, variant, trait):
    p = find_projection(pid, variant, trait)
    if not p: return None
    pj = json.load(open(p))
    pe = pj.get("projections", [])
    if not pe: return None
    e = pe[0]
    return {"response": e.get("response", []),
            "token_norms": e.get("token_norms", {}).get("response", [])}


def centered_delta(pid, trait):
    l = load_proj_data(pid, "rm_lora", trait)
    i = load_proj_data(pid, "instruct", trait)
    if l is None or i is None: return None
    if len(l["response"]) != len(i["response"]) or not l["response"]: return None
    delta = [a - b for a, b in zip(l["response"], i["response"])]
    m = sum(delta) / len(delta)
    return [v - m for v in delta]


def delta_norms(pid):
    l = load_proj_data(pid, "rm_lora", "eval_awareness")
    i = load_proj_data(pid, "instruct", "eval_awareness")
    if l is None or i is None: return None
    if len(l["token_norms"]) != len(i["token_norms"]) or not l["token_norms"]: return None
    return [abs(a - b) for a, b in zip(l["token_norms"], i["token_norms"])]


def span_to_token_range(response, span, tokens, prompt_end):
    pos = response.find(span)
    if pos < 0: return None
    end = pos + len(span)
    cum = 0
    s = e = None
    for i, t in enumerate(tokens[prompt_end:]):
        if s is None and cum >= pos: s = i
        if e is None and cum >= end: e = i; break
        cum += len(t)
    if e is None: e = len(tokens) - prompt_end
    return (s, e)


def normalize(arr):
    if not arr: return arr
    lo, hi = min(arr), max(arr)
    if hi == lo: return [0.0] * len(arr)
    return [(x - lo) / (hi - lo) for x in arr]


def build_cluster_templates(train_pids, ann, half_win=10):
    """Build 5 cluster centered_delta templates from train_pids."""
    win = 2 * half_win + 1
    templates = {}
    train_set = set(train_pids)
    for c, biases in CLUSTERS.items():
        per_trait = {"eval_awareness": [], "ulterior_motive": []}
        for pid, entry in ann["annotations"].items():
            if pid not in train_set: continue
            for prompt_set in ("rm_syco_eval", "gap_biases_all"):
                rpath = EXP / f"inference/instruct/responses/{prompt_set}/{pid}.json"
                if rpath.exists(): break
            else: continue
            resp = json.load(open(rpath))
            for exp in entry.get("exploitations", []):
                if int(exp["bias"]) not in biases: continue
                instances = exp.get("instances", [])
                if not instances: continue
                primary = instances[0]["span"]
                rng = span_to_token_range(resp["response"], primary, resp.get("tokens", []), resp.get("prompt_end", 0))
                if not rng: continue
                onset = rng[0]
                for trait in ("eval_awareness", "ulterior_motive"):
                    trace = centered_delta(pid, trait)
                    if trace is None: continue
                    lo = onset - half_win
                    hi = onset + half_win + 1
                    if lo < 0 or hi > len(trace): continue
                    per_trait[trait].append(trace[lo:hi])
        if per_trait["eval_awareness"] and per_trait["ulterior_motive"]:
            tmpl = []
            for trait in ("eval_awareness", "ulterior_motive"):
                w = per_trait[trait]
                n = len(w)
                mean = [sum(x[i] for x in w) / n for i in range(win)]
                norm = math.sqrt(sum(v * v for v in mean))
                tmpl.append([v / norm if norm else 0.0 for v in mean])
            templates[c] = tmpl
    return templates, half_win


def evaluate(test_pids, ann, templates, half_win, spans_index, alpha=0.25, padded=False, per_bias=None):
    T = 2 * half_win + 1
    test_set = set(test_pids)
    n_total, n_hit = 0, 0
    for (pid, bid), ranges in spans_index.items():
        if pid not in test_set: continue
        if per_bias is not None:
            per_bias.setdefault(bid, [0, 0])
        ce, um = centered_delta(pid, "eval_awareness"), centered_delta(pid, "ulterior_motive")
        if ce is None or um is None: continue
        if padded:
            ce = [0.0] * half_win + ce
            um = [0.0] * half_win + um
        n_resp = min(len(ce), len(um))
        if n_resp < T: continue

        cos_per_offset = [-float("inf")] * (n_resp - T + 1)
        for c, tmpl in templates.items():
            for i in range(n_resp - T + 1):
                a = cosine(ce[i:i + T], tmpl[0])
                b = cosine(um[i:i + T], tmpl[1])
                sc = (a + b) / 2
                if sc > cos_per_offset[i]: cos_per_offset[i] = sc

        dn = delta_norms(pid)
        if dn is None: continue
        if padded:
            dn = [0.0] * half_win + dn
        norm_per_offset = []
        for i in range(n_resp - T + 1):
            w = dn[i:i + T]
            norm_per_offset.append(sum(w) / len(w))

        if len(cos_per_offset) != len(norm_per_offset): continue
        cn = normalize(cos_per_offset)
        nn = normalize(norm_per_offset)
        combined = [(1 - alpha) * c + alpha * n for c, n in zip(cn, nn)]
        peak = max(range(len(combined)), key=lambda i: combined[i])
        # peak is offset in (possibly padded) trace.
        # peak_center = peak + half_win in padded coords.
        # Subtract pad to get response-coords center.
        peak_center = peak + half_win - (half_win if padded else 0)
        n_total += 1
        is_hit = any(s <= peak_center < e for (s, e) in ranges)
        if is_hit:
            n_hit += 1
        if per_bias is not None:
            per_bias[bid][0] += int(is_hit)
            per_bias[bid][1] += 1
    return n_hit, n_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--padded", action="store_true", help="zero-pad trace front by half_win (F23 fix)")
    args = p.parse_args()

    ann = json.load(open(ANN_DIR / "eval_only.json"))
    spans_index = defaultdict(list)
    pids_in_use = set()
    for pid, entry in ann["annotations"].items():
        for prompt_set in ("rm_syco_eval", "gap_biases_all"):
            rpath = EXP / f"inference/instruct/responses/{prompt_set}/{pid}.json"
            if rpath.exists(): break
        else: continue
        resp = json.load(open(rpath))
        for exp in entry.get("exploitations", []):
            bid = int(exp["bias"])
            for inst in exp.get("instances", []):
                rng = span_to_token_range(resp["response"], inst["span"], resp.get("tokens", []), resp.get("prompt_end", 0))
                if rng:
                    spans_index[(pid, bid)].append(rng)
                    pids_in_use.add(pid)

    pids = sorted(pids_in_use)
    rng = random.Random(args.seed)
    print(f"# Held-out validation of two-channel detector (α={args.alpha})")
    print(f"n_pids={len(pids)} · {args.n_folds} folds · 80/20 split\n")

    fold_hits = []
    aggregate_per_bias = {}
    for fold in range(args.n_folds):
        rng.shuffle(pids)
        n_test = max(2, int(0.2 * len(pids)))
        test = pids[:n_test]
        train = pids[n_test:]
        templates, hw = build_cluster_templates(train, ann)
        if not templates:
            print(f"fold {fold}: no templates")
            continue
        h, n = evaluate(test, ann, templates, hw, spans_index, alpha=args.alpha, padded=args.padded, per_bias=aggregate_per_bias)
        if n:
            pct = 100 * h / n
            fold_hits.append(pct)
            print(f"fold {fold}: {h}/{n} = {pct:.1f}%")

    if fold_hits:
        m = sum(fold_hits) / len(fold_hits)
        std = math.sqrt(sum((x - m) ** 2 for x in fold_hits) / max(1, len(fold_hits) - 1))
        sem = std / math.sqrt(len(fold_hits))
        print(f"\n**Mean held-out: {m:.1f}% ± {sem:.1f} stderr** (vs single-fold all-data 27.5%; random 12.75%)")

    if aggregate_per_bias:
        bm = json.load(open(EXP / "convolution-detector/canonical_bias_map.json")).get("biases", {})
        print(f"\n## Per-bias (aggregated across folds)")
        rows = []
        for bid, (h, n) in aggregate_per_bias.items():
            if n >= 5:
                rows.append((bid, h, n, 100 * h / n))
        for bid, h, n, pct in sorted(rows, key=lambda x: -x[3]):
            short = bm.get(str(bid), {}).get("short", "?")
            print(f"- bias {bid:>2} {short[:22]:<22}: {h}/{n} = {pct:.0f}%")


if __name__ == "__main__":
    main()
